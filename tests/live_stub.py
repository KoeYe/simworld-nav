"""A real HTTP nav-render/v0 service for the live-backend tests.

Not a mock of the client -- an actual ``ThreadingHTTPServer`` on a real port,
because the thing the tests defend is the *wire*: a client bug that only a
socket can show (timeouts, HTTP error bodies, JSON framing) passes straight
through a mocked transport. The frames are labelled PNGs drawn with PIL, so a
test can open what came back and a human debugging one can see which request
produced it.

The service is deliberately deterministic: the same key always draws the same
pixels, hence the same PNG bytes -- the property the album cache and
FrameAliases assume of a real service process, reproduced here so the tests
can assert on it.

Failure injection is explicit state, flipped by the test that needs it:
``healthz_ok`` (health probes fail), ``fail_keys`` (per-item failures inside
a 200 response), ``reject_code`` (the whole request dies with a wire error;
``busy`` goes out as HTTP 503, the way the real service sends it),
``busy_batches`` (the next N render batches answer 503 busy, then service
resumes -- the transient the busy taxonomy exists for), ``delay_s`` (every
render sleeps first, so a test can hold a request in flight on purpose).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PROTOCOL = "nav-render/v0"


def draw_frame(key: str, width: int, height: int) -> bytes:
    """A labelled PNG whose pixels are a pure function of (key, size)."""
    from PIL import Image, ImageDraw

    digest = hashlib.sha256(key.encode()).digest()
    background = (digest[0], digest[1], digest[2])
    image = Image.new("RGB", (width, height), background)
    ImageDraw.Draw(image).text((8, 8), key, fill=(255, 255, 255))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


class FakeRenderService:
    """One in-process UE stand-in speaking nav-render/v0 over HTTP."""

    def __init__(self, out_dir: Path, *, instance_id: str = "fake-0",
                 map_name: str = "citycore-paris"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.instance_id = instance_id
        self.map_name = map_name
        # Test-visible state.
        self.healthz_ok = True
        self.fail_keys: set[str] = set()
        self.reject_code: str | None = None
        self.busy_batches = 0
        self.busy_hits = 0
        self.delay_s = 0.0
        self.render_counts: Counter[str] = Counter()
        self.batches: list[dict[str, Any]] = []      # raw request bodies
        self.health_probes = 0
        self._lock = threading.Lock()

        service = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # keep pytest output clean
                pass

            def do_GET(self) -> None:
                if self.path != "/healthz":
                    self._send(404, {"error": {"code": "bad_request",
                                               "message": f"no route {self.path}"}})
                    return
                with service._lock:
                    service.health_probes += 1
                    ok = service.healthz_ok
                if not ok:
                    self._send(500, {"error": {"code": "engine_down",
                                               "message": "engine not attached"}})
                    return
                self._send(200, {
                    "status": "ok", "protocol": PROTOCOL,
                    "instance_id": service.instance_id,
                    "map_name": service.map_name,
                    "engine_connected": True, "episodes_active": 0,
                    "uptime_s": 1.0,
                })

            def do_POST(self) -> None:
                if self.path != "/render":
                    self._send(404, {"error": {"code": "bad_request",
                                               "message": f"no route {self.path}"}})
                    return
                body = json.loads(self.rfile.read(
                    int(self.headers.get("Content-Length", "0"))))
                if body.get("protocol") != PROTOCOL:
                    self._send(400, {"error": {"code": "bad_request",
                                               "message": "wrong protocol"}})
                    return
                with service._lock:
                    reject = service.reject_code
                    if service.busy_batches > 0:
                        service.busy_batches -= 1
                        service.busy_hits += 1
                        reject = reject or "busy"
                    delay = service.delay_s
                    service.batches.append(body)
                if reject:
                    # busy is 503 on the real service; other injected codes
                    # keep the generic 500 the earlier tests were written for.
                    self._send(503 if reject == "busy" else 500,
                               {"error": {"code": reject,
                                          "message": "injected by test"}})
                    return
                if delay:
                    time.sleep(delay)
                self._send(200, {"results": [self._result(body, item)
                                             for item in body.get("requests", [])]})

            def _result(self, body: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
                key = item["key"]
                with service._lock:
                    service.render_counts[key] += 1
                    failed = key in service.fail_keys
                if failed:
                    return {"key": key, "status": "failed",
                            "error": "injected per-item failure"}
                camera = item.get("camera") or body["camera"]
                png = draw_frame(key, int(camera["width"]), int(camera["height"]))
                digest = hashlib.sha256(png).hexdigest()
                out = {"key": key, "status": "ok", "path": None, "png_base64": None,
                       "sha256": digest,
                       "width": int(camera["width"]), "height": int(camera["height"])}
                if body.get("return_mode") == "base64":
                    out["png_base64"] = base64.b64encode(png).decode("ascii")
                else:
                    target = service.out_dir / f"{digest}.png"
                    if not target.exists():
                        target.write_bytes(png)
                    out["path"] = str(target)
                return out

            def _send(self, code: int, payload: dict[str, Any]) -> None:
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> "FakeRenderService":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def write_endpoints(path: Path, services: list[Any]) -> Path:
    """An endpoints.json for these services (or (id, base_url) pairs)."""
    instances = []
    for entry in services:
        if isinstance(entry, FakeRenderService):
            instances.append({"id": entry.instance_id, "base_url": entry.base_url,
                              "map_name": entry.map_name, "gpu_uuid": ""})
        else:
            member_id, base_url = entry
            instances.append({"id": member_id, "base_url": base_url,
                              "map_name": "citycore-paris", "gpu_uuid": ""})
    path.write_text(json.dumps({"version": 0, "instances": instances}, indent=2))
    return path
