"""One HTTP client per UE render instance.

Deliberately thin. The client's whole job is to speak nav-render/v0 to one
``base_url`` and to translate the two ways that can fail into a taxonomy the
pool can act on:

* the *service* answered with an error -- a non-200 carrying
  ``{"error": {"code", "message"}}`` -- which becomes the exception class for
  that code, so a caller can tell "you sent garbage" (its own bug, do not
  retry) from "the engine is down" (the instance's problem, fail over);
* the *transport* failed -- connection refused, timeout -- which becomes
  ``ServiceUnreachable`` after exactly one reconnect attempt.

One reconnect and no more, on purpose: retries hide a dying instance from the
pool, and the pool's quarantine is the mechanism that is supposed to see it.
A client that retried five times would turn "instance ue-2 is dead" into
"renders are mysteriously slow", which is the harder bug to find.

stdlib urllib only. The trainer imports this in every rollout worker, and a
requests/httpx dependency for two endpoints is a supply chain for a GET.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .protocol import (
    PROTOCOL,
    Healthz,
    ProtocolViolation,
    RenderBatch,
    RenderResponse,
    RenderResult,
    WireError,
)

# Long enough for a cold instance to settle a big batch, short enough that a
# hung engine is a failure rather than a stall. Overridable per client.
DEFAULT_RENDER_TIMEOUT_S = 120.0
DEFAULT_HEALTH_TIMEOUT_S = 5.0


class RenderServiceError(RuntimeError):
    """Any failure talking to a render service. ``code`` says which."""

    code = "unknown"


class BadRequestError(RenderServiceError):
    """The service rejected the request as malformed. This is the caller's
    bug; failing over to another instance would send the same garbage."""

    code = "bad_request"


class EngineDownError(RenderServiceError):
    """The service is up but its UE instance is not."""

    code = "engine_down"


class MapMismatchError(RenderServiceError):
    """The instance is serving a different map than the request assumes.

    Not retryable anywhere: a pool whose endpoints file mixes maps is
    misconfigured, and rendering Paris poses against another city would
    produce frames that look plausible and mean nothing.
    """

    code = "map_mismatch"


class RenderFailedError(RenderServiceError):
    """The whole batch failed inside the engine. (A *single* bad item is not
    this -- it comes back as a ``failed`` result in a 200 response.)"""

    code = "render_failed"


class BusyError(RenderServiceError):
    """The service refused the batch under load. Another instance may not."""

    code = "busy"


class ServiceUnreachable(RenderServiceError):
    """No HTTP conversation happened at all, even after one reconnect."""

    code = "unreachable"


_BY_CODE: dict[str, type[RenderServiceError]] = {
    cls.code: cls
    for cls in (BadRequestError, EngineDownError, MapMismatchError,
                RenderFailedError, BusyError)
}


def error_for(code: str, message: str) -> RenderServiceError:
    """The exception for a wire error code. Unknown codes stay errors --
    a service speaking codes this client does not know is a version skew,
    not a success."""
    cls = _BY_CODE.get(code, RenderServiceError)
    out = cls(f"{code}: {message}")
    if cls is RenderServiceError:
        out.code = code  # keep the wire's own word for the report
    return out


class UERenderClient:
    """nav-render/v0 over HTTP against one instance's ``base_url``."""

    def __init__(
        self,
        base_url: str,
        *,
        instance_id: str = "",
        render_timeout_s: float = DEFAULT_RENDER_TIMEOUT_S,
        health_timeout_s: float = DEFAULT_HEALTH_TIMEOUT_S,
    ):
        self.base_url = base_url.rstrip("/")
        self.instance_id = instance_id or self.base_url
        self.render_timeout_s = float(render_timeout_s)
        self.health_timeout_s = float(health_timeout_s)

    # ── the two endpoints ────────────────────────────────────────────────────

    def healthz(self) -> Healthz:
        data = self._request("GET", "/healthz", timeout_s=self.health_timeout_s)
        return Healthz.from_dict(data)

    def render(self, batch: RenderBatch) -> tuple[RenderResult, ...]:
        data = self._request("POST", "/render", body=batch.to_dict(),
                             timeout_s=self.render_timeout_s)
        return RenderResponse.from_dict(data).results

    # ── plumbing ─────────────────────────────────────────────────────────────

    def _request(self, method: str, path: str, *, body: dict[str, Any] | None = None,
                 timeout_s: float) -> dict[str, Any]:
        payload = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path, data=payload, method=method,
            headers={"Content-Type": "application/json"} if payload else {},
        )
        last: Exception | None = None
        # Two passes: the original attempt and one reconnect. A service
        # restarting between batches produces exactly one refused connection,
        # and that one is not worth a failover; a second is.
        for _ in range(2):
            try:
                with urllib.request.urlopen(request, timeout=timeout_s) as response:
                    return self._parse(response.read())
            except urllib.error.HTTPError as error:
                # The service answered; this is a protocol error, not a
                # transport one, and a reconnect would just be told again.
                raise self._wire_error(error) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
                last = error
        raise ServiceUnreachable(
            f"{self.instance_id}: {self.base_url}{path} unreachable after one "
            f"reconnect attempt ({last})")

    @staticmethod
    def _parse(raw: bytes) -> dict[str, Any]:
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as error:
            raise ProtocolViolation(f"response is not JSON: {error}") from None
        if not isinstance(data, dict):
            raise ProtocolViolation(f"response is {type(data).__name__}, not an object")
        return data

    def _wire_error(self, error: urllib.error.HTTPError) -> RenderServiceError:
        try:
            wire = WireError.from_dict(self._parse(error.read()))
        except (ProtocolViolation, OSError):
            return RenderServiceError(
                f"{self.instance_id}: HTTP {error.code} with a body that is "
                f"not a {PROTOCOL} error")
        return error_for(wire.code, wire.message)
