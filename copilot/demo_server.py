"""Thin local web demo for Interview Copilot."""

from __future__ import annotations

import asyncio
import json
import threading
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote


STATIC_DIR = Path(__file__).resolve().parent / "web_demo"


@dataclass(slots=True)
class DemoRuntime:
    """Small synchronous wrapper around the async agent loop."""

    agent_loop: object
    model: str
    session_id: str = "web:demo"
    channel: str = "cli"
    chat_id: str = "web-demo"
    _lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def handle_message(self, message: str) -> str:
        with self._lock:
            return asyncio.run(
                self.agent_loop.process_direct(
                    message,
                    session_key=self.session_id,
                    channel=self.channel,
                    chat_id=self.chat_id,
                )
            )

    def metadata(self) -> dict[str, str]:
        return {
            "model": self.model,
            "session_id": self.session_id,
        }


class DemoHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], runtime: DemoRuntime):
        super().__init__(server_address, DemoRequestHandler)
        self.runtime = runtime


class DemoRequestHandler(BaseHTTPRequestHandler):
    server: DemoHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/index.html"}:
            self._serve_static("index.html", "text/html; charset=utf-8")
            return
        if self.path == "/styles.css":
            self._serve_static("styles.css", "text/css; charset=utf-8")
            return
        if self.path == "/app.js":
            self._serve_static("app.js", "application/javascript; charset=utf-8")
            return
        if self.path == "/api/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    **self.server.runtime.metadata(),
                },
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/chat":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return

        content_length = int(self.headers.get("Content-Length", "0") or "0")
        raw_body = self.rfile.read(content_length).decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON payload"})
            return

        message = str(payload.get("message", "")).strip()
        if not message:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Message is required"})
            return

        try:
            response = self.server.runtime.handle_message(message)
        except Exception as exc:  # pragma: no cover - demo path
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return

        self._send_json(
            HTTPStatus.OK,
            {
                "message": message,
                "response": response,
                **self.server.runtime.metadata(),
            },
        )

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return

    def _serve_static(self, name: str, content_type: str) -> None:
        path = STATIC_DIR / unquote(name)
        if not path.exists():
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Asset not found"})
            return
        content = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_demo_server(
    *,
    runtime: DemoRuntime,
    host: str = "127.0.0.1",
    port: int = 18820,
    open_browser: bool = False,
) -> None:
    server = DemoHTTPServer((host, port), runtime)
    url = f"http://{host}:{port}"
    if open_browser:
        webbrowser.open(url)
    print(f"Interview Copilot demo running at {url}")
    print(f"Model: {runtime.model}")
    print("Use Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


__all__ = [
    "DemoRuntime",
    "DemoHTTPServer",
    "DemoRequestHandler",
    "STATIC_DIR",
    "run_demo_server",
]
