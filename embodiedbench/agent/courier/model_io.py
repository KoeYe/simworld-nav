"""Talking to a model, without the benchmark learning that model's habits.

Three times in one evaluation session a model was recorded as unable to follow
the reply contract when the truth was a setting on this side of the wire:

* ``max_new_tokens=48`` cut Qwen3-VL-4B's reasoning off before its fenced call
  and reported a held-out format score of 0.125 for a model that scores 1.0;
* ``--limit-mm-per-prompt 6`` against a turn that offers nine pictures returned
  HTTP 400, whose text was fed to the parser as though the model had said it,
  and killed 26 of 40 episodes as "format errors";
* Qwen3.5-9B rehearses candidate calls inside ``<think>`` in fenced blocks, so
  the parser found two action blocks and refused 66 of 80 turns from a model
  whose answers were all well formed.

Each was found by hand, after the number had already been believed once. The
pattern is the same every time: **a harness that only works for the model it was
debugged against measures its own configuration.** So the adaptation lives here,
once, rather than in the reader's head.

What this does, following mini-swe-agent's separation of a *model call* from an
*environment step*:

* **Unparseable output is requeried, not charged.** mini-swe-agent re-prompts on
  a format error and keeps the bad output in the conversation; the environment
  never sees it. Here likewise: the turn is retried with the error fed back, up
  to ``max_requeries``, and only a turn that never parses reaches the world. The
  count is reported rather than hidden, so a model that needs three attempts
  every turn still looks worse than one that needs none.
* **Reasoning is separated from the answer**, whichever way the model marks it:
  a ``reasoning_content`` field, a ``reasoning`` field, or inline ``<think>``.
* **Truncation is a configuration fault, not a model output.** ``finish_reason
  == "length"`` means the budget was too small; the reply is not parsed, the
  budget is raised, and the call is retried.
* **Transport failures are never mistaken for replies.** They raise, carrying
  the server's own message.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

# Every convention seen in the wild for "this part was me thinking". Matched
# non-greedily and only when closed: an unterminated block means the model was
# cut off mid-thought, and the calls inside it are the ones it was arguing
# itself out of.
THINK_BLOCKS = (
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<|thinking|>", "<|/thinking|>"),
    ("<reasoning>", "</reasoning>"),
)


@dataclass
class CallStats:
    """What it cost to get an action out of the model, in the model's own terms."""

    calls: int = 0
    requeries: int = 0
    truncations: int = 0
    transport_retries: int = 0
    unparseable: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "model_calls": self.calls,
            "requeries": self.requeries,
            "truncations": self.truncations,
            "transport_retries": self.transport_retries,
            "unparseable_turns": self.unparseable,
        }


def split_reasoning(message: dict[str, Any]) -> tuple[str, str]:
    """``(answer, reasoning)`` from one chat-completion message.

    The serving stack may hand the thought back in its own field, in which case
    ``content`` is already only the answer. Otherwise the thought is inline and
    has to be cut off the front. Both happen, on the same model, depending on
    whether a reasoning parser is configured -- so neither can be assumed.
    """
    content = message.get("content") or ""
    field_reasoning = message.get("reasoning_content") or message.get("reasoning") or ""

    for open_tag, close_tag in THINK_BLOCKS:
        end = content.rfind(close_tag)
        if end != -1:
            return content[end + len(close_tag):].strip(), content[:end]
        if open_tag in content:
            # Opened and never closed: there is no answer in here at all.
            return "", content
    return content.strip(), field_reasoning


def looks_truncated(choice: dict[str, Any]) -> bool:
    return choice.get("finish_reason") == "length"


class ModelClient:
    """An OpenAI-compatible endpoint, made to behave the same for every model.

    ``parse`` is the benchmark's own reply parser. It is passed in rather than
    imported so this module stays a transport concern and the reply contract
    stays where it is defined.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        max_tokens: int = 2048,
        max_requeries: int = 3,
        max_transport_retries: int = 2,
        truncation_growth: float = 2.0,
        max_tokens_ceiling: int = 8192,
        timeout: float = 600.0,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.max_tokens = max_tokens
        self.max_requeries = max_requeries
        self.max_transport_retries = max_transport_retries
        self.truncation_growth = truncation_growth
        self.max_tokens_ceiling = max_tokens_ceiling
        self.timeout = timeout
        self.stats = CallStats()
        self.last_reasoning = ""

    # ── transport ────────────────────────────────────────────────────────────

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.endpoint, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            # The server's own words. "HTTP Error 400: Bad Request" on its own
            # sent a previous investigation looking at the model for a day.
            detail = error.read().decode(errors="replace")[:500]
            raise RuntimeError(f"HTTP {error.code}: {detail}") from None

    def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(self.max_transport_retries + 1):
            try:
                return self._post(payload)
            except Exception as error:  # noqa: BLE001
                last = error
                # A 4xx is a request this harness built wrongly; retrying it
                # unchanged just spends the clock. Only transient faults are
                # worth another go.
                if isinstance(error, RuntimeError) and str(error).startswith("HTTP 4"):
                    raise
                if attempt < self.max_transport_retries:
                    self.stats.transport_retries += 1
                    time.sleep(2 ** attempt)
        raise last  # type: ignore[misc]

    # ── one turn ─────────────────────────────────────────────────────────────

    def act(
        self,
        messages: list[dict[str, Any]],
        parse: Callable[[str], Any],
        *,
        on_requery: Callable[[str, str], None] | None = None,
    ) -> tuple[str, Any | None, list[str]]:
        """Get one *parseable* action, or report that none was forthcoming.

        Returns ``(reply, parsed, rejected)``: the reply that finally parsed,
        what it parsed to, and every reply that did not. ``parsed`` is None when
        every attempt failed, and the caller decides what a turn like that costs
        -- this layer does not step the world.
        """
        budget = self.max_tokens
        conversation = list(messages)
        rejected: list[str] = []

        for attempt in range(self.max_requeries + 1):
            body = self._post_with_retries({
                "model": self.model, "messages": conversation,
                "max_tokens": budget, "temperature": 0.0,
            })
            self.stats.calls += 1
            choice = body["choices"][0]
            reply, reasoning = split_reasoning(choice.get("message") or {})
            self.last_reasoning = reasoning

            if looks_truncated(choice) and budget < self.max_tokens_ceiling:
                # Not a malformed answer -- an unfinished one. Parsing it would
                # record a configuration fault as a model failure.
                self.stats.truncations += 1
                budget = min(int(budget * self.truncation_growth), self.max_tokens_ceiling)
                continue

            try:
                return reply, parse(reply), rejected
            except Exception as error:  # noqa: BLE001 - parser raises its own type
                rejected.append(reply)
                if attempt >= self.max_requeries:
                    break
                self.stats.requeries += 1
                if on_requery is not None:
                    on_requery(reply, str(error))
                conversation = conversation + [
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content":
                        f"That reply could not be read as an action: {error}\n"
                        "Reply again, in the required shape, with exactly one call."},
                ]

        self.stats.unparseable += 1
        return (rejected[-1] if rejected else ""), None, rejected


def probe(endpoint: str, model: str, *, timeout: float = 300.0) -> dict[str, Any]:
    """Ask the model one trivial question and note how it answers.

    Cheap, and it turns three separate hand-debugged surprises into a line of
    setup: whether it thinks out loud, and how many tokens a bare answer costs
    it -- which is what the generation budget has to clear.
    """
    client = ModelClient(endpoint, model, max_tokens=2048, max_requeries=0,
                         timeout=timeout)
    body = client._post_with_retries({
        "model": model, "max_tokens": 2048, "temperature": 0.0,
        "messages": [{"role": "user", "content":
                      "Reply with exactly this and nothing else:\n```\nok()\n```"}],
    })
    choice = body["choices"][0]
    answer, reasoning = split_reasoning(choice.get("message") or {})
    usage = body.get("usage") or {}
    return {
        "reasons_out_loud": bool(reasoning),
        "reasoning_chars": len(reasoning),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "answer": answer[:120],
    }
