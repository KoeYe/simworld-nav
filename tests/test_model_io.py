"""The model-facing side of the harness, checked without a model.

Everything pinned here is a mistake this benchmark actually made and then
believed: a truncated reply scored as a format failure, a transport error fed
to the parser as though the model had said it, a reasoning model refused for
thinking out loud, and a format error charged as a world step. They are cheap
to test and expensive to rediscover.
"""

from __future__ import annotations

import pytest

from embodiedbench.agent.courier.model_io import (
    ModelClient,
    looks_truncated,
    split_reasoning,
)


class Boom(Exception):
    pass


def parse_walk(reply: str):
    """Stand-in for the reply contract: accepts exactly one fenced call."""
    if reply.count("```") != 2 or "walk_to" not in reply:
        raise Boom("no single fenced call")
    return reply.split("```")[1].strip()


def client(replies, **kwargs):
    """A ModelClient whose transport returns canned chat-completion bodies."""
    c = ModelClient("http://unused", "stub", **kwargs)
    calls = []

    def fake_post(payload):
        calls.append(payload)
        return replies[min(len(calls) - 1, len(replies) - 1)]

    c._post_with_retries = fake_post  # type: ignore[assignment]
    c.sent = calls  # type: ignore[attr-defined]
    return c


def body(content, finish="stop", **message):
    return {"choices": [{"finish_reason": finish,
                         "message": {"content": content, **message}}]}


class TestReasoningIsNotTheAnswer:
    def test_inline_think_block_is_cut_off(self):
        answer, reasoning = split_reasoning(
            {"content": "<think>\nmaybe ```walk_to(9)```\n</think>\nTHOUGHT: go\n```\nwalk_to(2)\n```"})
        assert "walk_to(9)" not in answer
        assert "walk_to(2)" in answer
        assert "maybe" in reasoning

    def test_a_separate_reasoning_field_is_respected(self):
        """vLLM with a reasoning parser puts the thought in its own field and
        leaves content clean. The same model does the opposite without one, so
        neither shape can be assumed."""
        answer, reasoning = split_reasoning(
            {"content": "```\nwalk_to(2)\n```", "reasoning_content": "thinking"})
        assert answer == "```\nwalk_to(2)\n```"
        assert reasoning == "thinking"

    def test_an_unclosed_thought_yields_no_answer(self):
        """Cut off mid-thought. The calls inside are the ones it was arguing
        itself out of, and running one is a guess."""
        answer, _ = split_reasoning({"content": "<think>\nmaybe ```walk_to(9)```"})
        assert answer == ""

    @pytest.mark.parametrize("open_tag,close_tag", [
        ("<think>", "</think>"), ("<thinking>", "</thinking>"),
        ("<reasoning>", "</reasoning>"),
    ])
    def test_the_common_conventions_are_all_handled(self, open_tag, close_tag):
        answer, _ = split_reasoning(
            {"content": f"{open_tag}x{close_tag}\n```\nwalk_to(1)\n```"})
        assert answer == "```\nwalk_to(1)\n```"


class TestATruncatedReplyIsNotAModelFailure:
    def test_length_finish_raises_the_budget_and_retries(self):
        c = client([body("THOUGHT: I am thinking about", finish="length"),
                    body("THOUGHT: go\n```\nwalk_to(2)\n```")],
                   max_tokens=100)
        reply, parsed, rejected = c.act([{"role": "user", "content": "x"}], parse_walk)
        assert parsed == "walk_to(2)"
        assert c.stats.truncations == 1
        assert rejected == [], "a truncated reply must not count as a bad reply"
        assert c.sent[1]["max_tokens"] > c.sent[0]["max_tokens"]

    def test_looks_truncated_reads_finish_reason(self):
        assert looks_truncated({"finish_reason": "length"})
        assert not looks_truncated({"finish_reason": "stop"})


class TestAFormatErrorIsRequeriedNotCharged:
    """mini-swe-agent re-prompts on an unparseable reply and never lets the
    world see it. Charging it as a turn measures the model's luck at
    formatting rather than its competence at the task."""

    def test_a_bad_reply_is_retried_with_the_error_fed_back(self):
        c = client([body("I will walk north."),
                    body("THOUGHT: ok\n```\nwalk_to(1)\n```")])
        reply, parsed, rejected = c.act([{"role": "user", "content": "x"}], parse_walk)
        assert parsed == "walk_to(1)"
        assert c.stats.requeries == 1
        assert rejected == ["I will walk north."]
        # the failed attempt and the complaint are both in the retry
        assert c.sent[1]["messages"][-2]["role"] == "assistant"
        assert "could not be read" in c.sent[1]["messages"][-1]["content"]

    def test_giving_up_is_reported_rather_than_guessed(self):
        c = client([body("nope")], max_requeries=2)
        reply, parsed, rejected = c.act([{"role": "user", "content": "x"}], parse_walk)
        assert parsed is None, "the harness must not invent an action"
        assert len(rejected) == 3
        assert c.stats.unparseable == 1

    def test_requeries_are_counted_not_hidden(self):
        """A model needing three attempts a turn should still look worse than
        one needing none, so the count is part of the result."""
        c = client([body("bad"), body("bad"), body("THOUGHT: x\n```\nwalk_to(3)\n```")])
        c.act([{"role": "user", "content": "x"}], parse_walk)
        assert c.stats.as_dict()["requeries"] == 2
        assert c.stats.as_dict()["model_calls"] == 3


class TestTransportFailuresAreNeverReplies:
    def test_an_http_error_carries_the_servers_own_message(self):
        c = ModelClient("http://unused", "stub")

        def explode(payload):
            raise RuntimeError('HTTP 400: {"message":"At most 6 image(s)"}')

        c._post_with_retries = explode  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match="At most 6 image"):
            c.act([{"role": "user", "content": "x"}], parse_walk)

    def test_a_4xx_is_not_retried(self):
        """It is a request this harness built wrongly; sending it again
        unchanged only spends the clock."""
        c = ModelClient("http://unused", "stub", max_transport_retries=3)
        tries = []

        def explode(payload):
            tries.append(1)
            raise RuntimeError("HTTP 400: bad")

        c._post = explode  # type: ignore[assignment]
        with pytest.raises(RuntimeError):
            c._post_with_retries({})
        assert len(tries) == 1


class TestTheBudgetNeverWalksIntoTheContextLimit:
    """Raising max_tokens after a truncation is right until it is not.

    Escalating to 10000 output tokens against a 10240-token model leaves 240
    for the prompt, and every turn 400s. Qwen3.5-9B episodes ended after a mean
    of 2.6 turns that way, with zero format errors -- the harness had broken its
    own requests while adapting. The server states both numbers in the refusal,
    so the fix is to read them rather than carry a per-model ceiling.
    """

    def _clamping_client(self):
        c = ModelClient("http://unused", "stub", max_tokens=8000,
                        max_tokens_ceiling=10000)
        seen = []

        def fake(payload):
            # A realistic server: it refuses exactly when the request cannot
            # fit, and says so with both numbers.
            seen.append(payload["max_tokens"])
            if payload["max_tokens"] + 8000 > 10240:
                raise RuntimeError(
                    'HTTP 400: {"message":"This model\'s maximum context length '
                    'is 10240 tokens. However, you requested '
                    f'{payload["max_tokens"]} output tokens and your prompt '
                    'contains 8000 tokens"}')
            return body("THOUGHT: ok\n```\nwalk_to(1)\n```")

        c._post_with_retries = fake  # type: ignore[assignment]
        c.seen = seen  # type: ignore[attr-defined]
        return c

    def test_it_backs_off_to_the_room_the_server_reports(self):
        c = self._clamping_client()
        _, parsed, _ = c.act([{"role": "user", "content": "x"}], parse_walk)
        assert parsed == "walk_to(1)"
        assert c.seen[1] == 10240 - 8000 - 64
        assert c.stats.budget_clamps == 1

    def test_the_clamp_sticks_for_later_turns(self):
        c = self._clamping_client()
        c.act([{"role": "user", "content": "x"}], parse_walk)
        assert c.max_tokens_ceiling <= 10240 - 8000 - 64

    def test_an_unrelated_400_still_raises(self):
        """Only a context-limit refusal carries a usable number. Swallowing
        every 400 would hide the six-image limit that started all this."""
        c = ModelClient("http://unused", "stub")

        def fake(payload):
            raise RuntimeError('HTTP 400: {"message":"At most 6 image(s)"}')

        c._post_with_retries = fake  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match="At most 6 image"):
            c.act([{"role": "user", "content": "x"}], parse_walk)


class TestAnOverlongPromptShedsHistoryRatherThanDying:
    """When the budget cannot shrink any further, the prompt is what is too big.

    A reasoning model behind a 10k context fills it after a handful of turns of
    history, and every turn then 400s. The alternative to shedding history here
    is a per-model history setting, which is one more number to guess wrong for
    each new model.
    """

    def test_the_oldest_exchange_is_dropped_and_the_call_retried(self):
        c = ModelClient("http://unused", "stub", max_tokens=512)
        sizes = []

        def fake(payload):
            n = len(payload["messages"])
            sizes.append(n)
            if n > 3:
                raise RuntimeError(
                    'HTTP 400: {"message":"This model\'s maximum context length '
                    'is 10240 tokens. However, you requested 512 output tokens '
                    'and your prompt contains 10200 tokens"}')
            return body("THOUGHT: ok\n```\nwalk_to(1)\n```")

        c._post_with_retries = fake  # type: ignore[assignment]
        convo = [{"role": "system", "content": "s"}]
        convo += [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}] * 3
        _, parsed, _ = c.act(convo, parse_walk)
        assert parsed == "walk_to(1)"
        assert c.stats.history_drops >= 1
        assert sizes[-1] < sizes[0]

    def test_it_gives_up_rather_than_looping_when_nothing_is_left_to_drop(self):
        c = ModelClient("http://unused", "stub")

        def fake(payload):
            raise RuntimeError(
                'HTTP 400: {"message":"This model\'s maximum context length is '
                '10240 tokens. However, you requested 100 output tokens and '
                'your prompt contains 10200 tokens"}')

        c._post_with_retries = fake  # type: ignore[assignment]
        with pytest.raises(RuntimeError):
            c.act([{"role": "system", "content": "s"},
                   {"role": "user", "content": "u"}], parse_walk)


class TestBothWordingsOfTheContextRefusalAreUnderstood:
    """vLLM phrases the same refusal two ways depending on where it fires.

    Handling only the first shape let the second through as fatal and ended
    Qwen3.5-9B episodes after a mean of 6.6 turns -- the harness reading the
    server's limits, but only when the server used the words it expected.
    """

    @pytest.mark.parametrize("message", [
        "This model's maximum context length is 10240 tokens. However, you "
        "requested 4000 output tokens and your prompt contains 10200 tokens",
        "Input length (10384) exceeds model's maximum context length (10240).",
    ])
    def test_an_overlong_prompt_sheds_history_either_way(self, message):
        c = ModelClient("http://unused", "stub", max_tokens=512)
        seen = []

        def fake(payload):
            seen.append(len(payload["messages"]))
            if len(payload["messages"]) > 3:
                raise RuntimeError('HTTP 400: {"message":"%s"}' % message)
            return body("THOUGHT: ok\n```\nwalk_to(1)\n```")

        c._post_with_retries = fake  # type: ignore[assignment]
        convo = [{"role": "system", "content": "s"}]
        convo += [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}] * 3
        _, parsed, _ = c.act(convo, parse_walk)
        assert parsed == "walk_to(1)", f"not handled: {message[:40]}"
        assert c.stats.history_drops >= 1
