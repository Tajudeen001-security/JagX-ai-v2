"""JagX v12 — a real OpenAI-compatible mock LLM server.

Lets the whole product be exercised end to end for free, with no model and no
network. It streams a tool call on the first turn and an answer on the second,
which is exactly the shape the agent loop expects.

    python scripts/mock_llm.py --port 11434
    OLLAMA_BASE_URL=http://localhost:11434/v1 OLLAMA_MODEL=mock uvicorn app.main:app
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _chunk(delta: dict, finish: str | None = None) -> bytes:
    payload = {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "mock",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # noqa: D102 - quiet
        pass

    def _json(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/").endswith("/models"):
            self._json({"object": "list", "data": [{"id": "mock", "object": "model"}]})
        else:
            self._json({"ok": True, "mock": True})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            body = {}

        messages = body.get("messages") or []
        has_tool_result = any(m.get("role") == "tool" for m in messages)
        wants_tools = bool(body.get("tools"))
        stream = bool(body.get("stream"))

        # Pick a tool call that matches the user's intent, so the mock exercises
        # the guard's review path as well as the allow path.
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = str(m.get("content") or "").lower()
                break

        if any(v in last_user for v in ("send", "email", "pay", "invoice", "delete", "publish")):
            tool_name = "fetch_url"
            tool_args = {"url": "https://example.test/send-email-to-client"}
        else:
            tool_name = "calculator"
            tool_args = {"expression": "17*23"}

        # Turn 1: call a tool. Turn 2+: answer.
        if wants_tools and not has_tool_result:
            if not stream:
                self._json({
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "model": "mock",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": "call_mock_1",
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(tool_args),
                                },
                            }],
                        },
                        "finish_reason": "tool_calls",
                    }],
                })
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self._write_chunked(_chunk({"role": "assistant", "content": ""}))
            self._write_chunked(_chunk({
                "tool_calls": [{
                    "index": 0,
                    "id": "call_mock_1",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": json.dumps(tool_args)},
                }]
            }))
            self._write_chunked(_chunk({}, "tool_calls"))
            self._write_chunked(b"data: [DONE]\n\n")
            self._end_chunked()
            return

        answer = (
            "**Mock provider answer.**\n\n"
            "JagX ran end to end: the agent loop planned, called a tool, observed the result, "
            "and produced this answer. The calculator returned `17*23 = 391`.\n\n"
            "This response comes from the built-in mock LLM, so the whole product can be "
            "exercised with no model and no API key. Point `OLLAMA_BASE_URL` at a real local "
            "runtime (Ollama, vLLM, llama.cpp) for real answers."
        )
        if not stream:
            self._json({
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "model": "mock",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }],
            })
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        self._write_chunked(_chunk({"role": "assistant", "content": ""}))
        for word in answer.split(" "):
            self._write_chunked(_chunk({"content": word + " "}))
        self._write_chunked(_chunk({}, "stop"))
        self._write_chunked(b"data: [DONE]\n\n")
        self._end_chunked()

    def _write_chunked(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")

    def _end_chunked(self) -> None:
        self.wfile.write(b"0\r\n\r\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="JagX mock OpenAI-compatible LLM server")
    parser.add_argument("--port", type=int, default=11434)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"JagX mock LLM listening on http://{args.host}:{args.port}/v1  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()