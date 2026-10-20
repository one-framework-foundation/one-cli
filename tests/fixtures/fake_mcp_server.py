#!/usr/bin/env python3
"""A real MCP server over stdio, small enough to reason about.

The client is tested against a subprocess speaking the actual protocol rather
than a mocked transport, because the parts that break are framing and ordering -
exactly what a mock cannot get wrong.

Behaviour is chosen by argv[1]:
  ok        normal server, three tools
  noisy     writes a non-JSON log line before every reply
  dies      exits during the handshake
  hang      accepts the handshake, then never answers a call
"""

import json
import sys
import time

MODE = sys.argv[1] if len(sys.argv) > 1 else "ok"

TOOLS = [
    {"name": "echo", "description": "Echo the text back.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}},
                     "required": ["text"]},
     "annotations": {"readOnlyHint": True}},
    {"name": "touch", "description": "A tool that changes something.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "explode", "description": "Always fails.",
     "inputSchema": {"type": "object", "properties": {}}},
]


def send(payload):
    if MODE == "noisy":
        sys.stdout.write("[fake-server] a log line that is not JSON\n")
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        method, mid = msg.get("method"), msg.get("id")
        if method == "initialize":
            if MODE == "dies":
                sys.exit(1)
            # A server answers with the revision it speaks; echo the client's so
            # the fixture stays protocol-agnostic.
            requested = (msg.get("params") or {}).get("protocolVersion")
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": requested or "", "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1"}}})
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            if MODE == "hang":
                time.sleep(30)
                continue
            if MODE == "notifications":
                for _ in range(20):
                    send({"jsonrpc": "2.0", "method": "notifications/progress"})
                    time.sleep(.03)
            params = msg.get("params", {})
            name = params.get("name")
            args = params.get("arguments", {})
            if name == "echo":
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": args.get("text", "")}]}})
            elif name == "touch":
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "image", "data": "..."}]}})
            elif name == "explode":
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "isError": True,
                    "content": [{"type": "text", "text": "it exploded"}]}})
            else:
                send({"jsonrpc": "2.0", "id": mid,
                      "error": {"code": -32602, "message": f"no tool {name}"}})
        else:
            send({"jsonrpc": "2.0", "id": mid,
                  "error": {"code": -32601, "message": "unknown method"}})


if __name__ == "__main__":
    main()
