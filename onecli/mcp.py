"""MCP client - the servers your config names, reachable from one-cli.

one-cli's built-in tool surface is what this package ships. Anything else
reachable over MCP - a guard, a scheduler, a gateway - is reachable through
`mcp__server__tool` permission clauses once a server is configured.

Shape of the thing, and why:

**stdio only.** An MCP server is a subprocess speaking newline-delimited JSON-RPC
2.0 on its stdin/stdout. HTTP and SSE transports exist in the spec; every server
we run locally is stdio, and adding a transport we cannot test against would be
guessing.

**Config is Claude Code's shape** - `{"mcpServers": {"name": {"command", "args",
"env"}}}` - so a config file written for that harness runs here unchanged. That
is the whole point of a parity feature: not a second dialect.

**We do not read the user's global `~/.mcp.json`.** Claude Code does, and it is
the sharpest edge in that design: a sandboxed run silently inherits every server
the operator has configured, which can include cron editing and agent
spawning. one-cli takes explicit `--mcp-config` files. Project `.mcp.json`
additionally requires per-run `--trust-project-config`; `--strict-mcp-config`
excludes it even in a trusted project. Servers receive a minimal environment;
credentials must be explicitly supplied in each server's `env` configuration.

**A server that misbehaves must not take the session with it.** Every failure -
a command that does not exist, a handshake that times out, a call that raises -
comes back as tool text the model can read and route around. The one thing that
is not tolerated is a hang: reads are deadlined, and a server that misses its
deadline is dropped rather than waited on.

**Mutability comes from the protocol, not from a guess.** MCP tools may carry a
`readOnlyHint` annotation; a tool that does not claim to be read-only is treated
as mutating, so `--read-only` withholds it. Unknown means careful, the same rule
the local registry uses.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import json
import os
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, field

from onecli import __version__

# The MCP revision this client opens with; the server answers with the
# revision it speaks.
PROTOCOL_VERSION = "2025-06-18"

CLIENT_INFO = {"name": "one-cli", "version": __version__}

INIT_TIMEOUT = 20.0
LIST_TIMEOUT = 20.0
CALL_TIMEOUT = 120.0


@dataclass
class ServerSpec:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = ""

    @property
    def argv(self) -> list[str]:
        return [self.command, *self.args]


class MCPError(RuntimeError):
    pass


def parse_config(blob: str | dict, source: str = "") -> list[ServerSpec]:
    """Read one config - a JSON string or an already-parsed dict.

    Accepts the `{"mcpServers": {...}}` wrapper and a bare mapping, because both
    spellings are in the wild and refusing one buys nothing.
    """
    data = json.loads(blob) if isinstance(blob, str) else blob
    if not isinstance(data, dict):
        raise MCPError(f"{source or 'mcp config'}: expected a JSON object")
    servers = data.get("mcpServers", data)
    if not isinstance(servers, dict):
        raise MCPError(f"{source or 'mcp config'}: mcpServers must be an object")

    specs = []
    for name, entry in servers.items():
        if not isinstance(entry, dict):
            raise MCPError(f"{source or 'mcp config'}: server {name!r} is not an object")
        transport = entry.get("type", "stdio")
        if transport != "stdio":
            # Named, not silently skipped: a server that will never answer is
            # something the operator has to know about.
            raise MCPError(
                f"{source or 'mcp config'}: server {name!r} uses transport "
                f"{transport!r}; one-cli speaks stdio only")
        command = entry.get("command")
        if not command:
            raise MCPError(f"{source or 'mcp config'}: server {name!r} has no command")
        args = entry.get("args") or []
        if isinstance(args, str):
            args = shlex.split(args)
        specs.append(ServerSpec(
            name=name,
            command=command,
            args=[str(a) for a in args],
            env={str(k): str(v) for k, v in (entry.get("env") or {}).items()},
            cwd=entry.get("cwd", "") or "",
        ))
    return specs


def load_configs(sources: list[str], workspace: str = "",
                 strict: bool = False, trust_project: bool = False) -> list[ServerSpec]:
    """Explicit configs, plus project config only with explicit per-run trust.

    A source is a path or an inline JSON string, the same as `--mcp-config`.
    """
    specs: list[ServerSpec] = []
    for source in sources:
        if os.path.isfile(source):
            with open(source) as fh:
                specs.extend(parse_config(fh.read(), source))
        elif source.lstrip().startswith("{"):
            specs.extend(parse_config(source, "--mcp-config"))
        else:
            raise MCPError(f"--mcp-config: no such file, and not JSON: {source}")
    if trust_project and not strict and workspace:
        project = os.path.join(workspace, ".mcp.json")
        if os.path.isfile(project):
            with open(project) as fh:
                specs.extend(parse_config(fh.read(), project))
    seen, unique = set(), []
    for spec in specs:
        if spec.name in seen:      # first wins: explicit before discovered
            continue
        seen.add(spec.name)
        unique.append(spec)
    return unique


class StdioServer:
    """One MCP server subprocess and the JSON-RPC conversation with it."""

    def __init__(self, spec: ServerSpec):
        self.spec = spec
        self.proc: subprocess.Popen | None = None
        self.tools: list[dict] = []
        self.error = ""
        self._id = 0
        self._lock = threading.Lock()

    # --- transport --------------------------------------------------------

    def start(self) -> bool:
        # Do not pass provider keys, agent credentials, or injection variables
        # (PYTHONPATH, LD_PRELOAD, etc.) to every server. Credentials needed by
        # a server must be explicitly supplied in that server's env config.
        env = {key: os.environ[key] for key in
               ("PATH", "HOME", "USERPROFILE", "SYSTEMROOT", "TEMP", "TMPDIR", "LANG")
               if key in os.environ}
        env.update(self.spec.env)
        try:
            self.proc = subprocess.Popen(
                self.spec.argv,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True, bufsize=1, env=env,
                cwd=self.spec.cwd or None,
            )
        except OSError as exc:
            self.error = f"could not start: {exc}"
            return False
        try:
            self._request("initialize", {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": CLIENT_INFO,
            }, timeout=INIT_TIMEOUT)
            self._notify("notifications/initialized")
            result = self._request("tools/list", {}, timeout=LIST_TIMEOUT)
        except MCPError as exc:
            self.error = str(exc)
            self.stop()
            return False
        if (not isinstance(result, dict) or not isinstance(result.get("tools", []), list)
                or any(not isinstance(t, dict) for t in result.get("tools", []))):
            self.error = "malformed tools/list reply"
            self.stop()
            return False
        self.tools = [t for t in result.get("tools", [])
                      if isinstance(t, dict) and isinstance(t.get("name"), str)]
        return True

    def stop(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.terminate()
            proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()
            proc.wait(timeout=5)

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def _send(self, payload: dict) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise MCPError("server is not running")
        try:
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise MCPError(f"server closed its input: {exc}") from exc

    def _notify(self, method: str, params: dict | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def _read_line(self, timeout: float) -> str:
        """One line, or a timeout. A hung server must not hang the agent.

        `readline` on a pipe cannot be deadlined, so the read runs on a thread
        and the deadline is enforced by joining it. A server that blows the
        deadline is left behind with its process torn down, because a thread
        still blocked on a dead pipe is a leak, not a fallback.
        """
        box: list[str] = []
        if self.proc is None or self.proc.stdout is None:
            raise MCPError("server is not running")
        stdout = self.proc.stdout

        def _read():
            try:
                box.append(stdout.readline())
            except (OSError, ValueError):
                box.append("")

        thread = threading.Thread(target=_read, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            raise MCPError(f"no response within {timeout:.0f}s")
        return box[0] if box else ""

    def _request(self, method: str, params: dict, timeout: float) -> dict:
        with self._lock:
            if self.proc is None or self.proc.stdout is None:
                raise MCPError("server is not running")
            request_id = self._next_id()
            self._send({"jsonrpc": "2.0", "id": request_id,
                        "method": method, "params": params})
            deadline = time.monotonic() + timeout
            valid_error = False
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise MCPError(f"no response within {timeout:.0f}s")
                    line = self._read_line(remaining)
                    if not line:
                        raise MCPError("server exited without answering")
                    try:
                        message = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # stdout logs do not extend the deadline
                    if not isinstance(message, dict):
                        raise MCPError("invalid RPC envelope: expected object")
                    if message.get("id") != request_id:
                        continue
                    if "error" in message:
                        err = message["error"]
                        if not isinstance(err, dict):
                            raise MCPError("invalid RPC error: expected object")
                        valid_error = True
                        raise MCPError(f"{err.get('code', '?')}: {err.get('message', err)}")
                    result = message.get("result", {})
                    if not isinstance(result, dict):
                        raise MCPError("invalid RPC result: expected object")
                    return result
            except MCPError:
                if not valid_error:
                    self.stop()
                raise

    # --- the one call the agent makes ------------------------------------

    def call(self, tool: str, arguments: dict) -> str:
        result = self._request("tools/call",
                               {"name": tool, "arguments": arguments},
                               timeout=CALL_TIMEOUT)
        return _render_content(result)


def _render_content(result: dict) -> str:
    """MCP content blocks -> the string the model reads.

    Non-text blocks are named rather than dropped: a model told "[image]" can
    ask for something else, where silence reads as an empty answer. Length is
    not bounded here - `tools.run` spills an oversized result to a file, which
    beats truncating one inside the client.
    """
    parts = []
    for block in result.get("content", []) or []:
        kind = block.get("type")
        if kind == "text":
            parts.append(block.get("text", ""))
        elif kind == "resource":
            resource = block.get("resource", {})
            parts.append(resource.get("text")
                         or f"[resource {resource.get('uri', 'unknown')}]")
        else:
            parts.append(f"[{kind or 'unknown'} content, not text]")
    text = "\n".join(p for p in parts if p).strip()
    if not text:
        text = "(the server returned no content)"
    if result.get("isError"):
        text = f"error from server: {text}"
    return text


def tool_name(server: str, tool: str) -> str:
    return f"mcp__{server}__{tool}"


def normalize_schema(node):
    """Give every schema node an explicit `type`, recursively.

    An MCP server's `inputSchema` is another process's output, and pydantic -
    which most of your configured endpoints use - renders `Optional[str]` as
    ``{"anyOf": [{"type": "string"}, {"type": "null"}]}`` with no `type` of its
    own. That is valid JSON Schema and the Anthropic wire accepts it, but a
    moonshot-flavoured endpoint refuses the whole request:

        tools.function.parameters is not a valid moonshot flavored json schema,
        details: <At path '$defs.SurfaceElement.properties.paradigm':
        type is not defined>

    One untyped property in one server's schema therefore fails every request
    to such an endpoint, including calls to unrelated tools. Collapsing an ``anyOf`` of ``[T, null]`` to
    ``type: T`` loses nothing that matters: whether the argument may be omitted
    is carried by `required`, not by a null branch.

    Done here rather than in the openai wire because the schema is wrong for a
    strict reader whichever backend is selected, and a repair that only runs on
    the lane where it was noticed is a repair the next lane re-discovers.
    """
    if isinstance(node, list):
        return [normalize_schema(x) for x in node]
    if not isinstance(node, dict):
        return node

    out = {k: normalize_schema(v) for k, v in node.items()}

    if "type" not in out:
        for key in ("anyOf", "oneOf"):
            branches = out.get(key)
            if not isinstance(branches, list):
                continue
            typed = [b for b in branches
                     if isinstance(b, dict) and b.get("type")
                     and b.get("type") != "null"]
            if len(typed) == 1 and len(branches) - len(typed) == sum(
                    1 for b in branches
                    if isinstance(b, dict) and b.get("type") == "null"):
                # A single real type beside null branches: hoist it and drop
                # the union. Anything richer (two real types) is left alone -
                # guessing which one the server meant would be worse than the
                # 400.
                merged = dict(typed[0])
                merged.pop("title", None)
                out.pop(key)
                for k, v in merged.items():
                    out.setdefault(k, v)
                break
    return out


class MCPClient:
    """Every configured server, and the tools they add to the registry."""

    def __init__(self, specs: list[ServerSpec]):
        self.servers = [StdioServer(s) for s in specs]
        self.failures: list[tuple[str, str]] = []

    def start(self) -> list[StdioServer]:
        if not self.servers:
            return []
        live = []
        # Servers own independent pipes and RPC locks. Overlap their bounded
        # handshakes, but publish results in config order so the tool surface
        # and failure reporting do not depend on process scheduling.
        try:
            with ThreadPoolExecutor(max_workers=min(4, len(self.servers))) as pool:
                outcomes = list(pool.map(lambda server: server.start(), self.servers))
        except BaseException:
            self.stop()
            raise
        for server, started in zip(self.servers, outcomes):
            if started:
                live.append(server)
            else:
                self.failures.append((server.spec.name, server.error))
        self.servers = live
        return live

    def stop(self) -> None:
        for server in self.servers:
            server.stop()

    def register(self, registry: dict, tool_cls) -> list[str]:
        """Add every discovered tool to a registry. Returns the names added."""
        added = []
        for server in self.servers:
            for spec in server.tools:
                name = tool_name(server.spec.name, spec["name"])
                registry[name] = tool_cls(
                    name=name,
                    description=(spec.get("description")
                                 or f"{spec['name']} on MCP server {server.spec.name}"),
                    parameters=normalize_schema(
                        spec.get("inputSchema")
                        or {"type": "object", "properties": {}}),
                    fn=_make_fn(server, spec["name"]),
                    # Only a real boolean True claims read-only: the
                    # string "false" is truthy and would unlock --read-only.
                    mutating=(spec.get("annotations") or {}).get("readOnlyHint") is not True,
                    takes_workspace=False,
                )
                added.append(name)
        return added


def _make_fn(server: StdioServer, tool: str):
    def call(**arguments) -> str:
        try:
            return server.call(tool, arguments)
        except MCPError as exc:
            # Tool failures are data for the model, never an exception out of
            # the loop - the same contract every local tool holds.
            return f"error: MCP call {tool} failed: {exc}"
    return call
