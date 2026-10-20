"""The MCP client, tested against a real subprocess speaking the real protocol.

`tests/fixtures/fake_mcp_server.py` is a genuine stdio server. The parts of a
client that break are framing, ordering and what happens when the other side
misbehaves - none of which a mocked transport can get wrong, so it is not
mocked.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

from onecli import mcp, permissions
from onecli import tools as toolkit

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fixtures", "fake_mcp_server.py")


def spec(mode: str = "ok", name: str = "fake") -> mcp.ServerSpec:
    return mcp.ServerSpec(name=name, command=sys.executable,
                          args=[FIXTURE, mode])


@pytest.fixture
def registry(monkeypatch):
    """A registry copy, so a test cannot leak tools into the next one."""
    monkeypatch.setattr(toolkit, "REGISTRY", dict(toolkit.REGISTRY))
    return toolkit.REGISTRY


# --- config ----------------------------------------------------------------

def test_claude_code_config_shape_is_read_as_is():
    specs = mcp.parse_config(json.dumps({"mcpServers": {
        "a": {"command": "x", "args": ["-m", "y"], "env": {"K": "v"}}}}))
    assert specs[0].name == "a"
    assert specs[0].argv == ["x", "-m", "y"]
    assert specs[0].env == {"K": "v"}


def test_a_bare_mapping_and_string_args_are_accepted():
    specs = mcp.parse_config({"a": {"command": "x", "args": "-m y"}})
    assert specs[0].argv == ["x", "-m", "y"]


def test_a_transport_we_cannot_speak_is_named_not_skipped():
    with pytest.raises(mcp.MCPError, match="stdio only"):
        mcp.parse_config({"mcpServers": {"a": {"type": "sse", "url": "http://x"}}})


def test_a_server_with_no_command_is_refused():
    with pytest.raises(mcp.MCPError, match="no command"):
        mcp.parse_config({"mcpServers": {"a": {"args": ["x"]}}})


def test_project_config_is_discovered_and_strict_withholds_it(tmp_path):
    (tmp_path / ".mcp.json").write_text(json.dumps(
        {"mcpServers": {"proj": {"command": "x"}}}))
    assert [s.name for s in mcp.load_configs([], str(tmp_path), trust_project=True)] == ["proj"]
    assert mcp.load_configs([], str(tmp_path), strict=True, trust_project=True) == []


def test_an_explicit_config_wins_a_name_clash_with_the_project(tmp_path):
    (tmp_path / ".mcp.json").write_text(json.dumps(
        {"mcpServers": {"dup": {"command": "from-project"}}}))
    explicit = json.dumps({"mcpServers": {"dup": {"command": "from-flag"}}})
    specs = mcp.load_configs([explicit], str(tmp_path), trust_project=True)
    assert [s.command for s in specs] == ["from-flag"]


def test_a_config_that_is_neither_a_file_nor_json_is_an_error():
    with pytest.raises(mcp.MCPError, match="no such file"):
        mcp.load_configs(["not-a-path"], "")


# --- the conversation ------------------------------------------------------

def test_tools_are_discovered_and_namespaced(registry):
    client = mcp.MCPClient([spec()])
    client.start()
    try:
        added = client.register(registry, toolkit.Tool)
        assert added == ["mcp__fake__echo", "mcp__fake__touch",
                         "mcp__fake__explode"]
        assert client.failures == []
    finally:
        client.stop()


def test_a_call_returns_the_servers_text(registry):
    client = mcp.MCPClient([spec()])
    client.start()
    try:
        client.register(registry, toolkit.Tool)
        assert toolkit.run("mcp__fake__echo", {"text": "hello"}, "/tmp") == "hello"
    finally:
        client.stop()


def test_a_server_error_comes_back_as_text_not_an_exception(registry):
    client = mcp.MCPClient([spec()])
    client.start()
    try:
        client.register(registry, toolkit.Tool)
        assert "it exploded" in toolkit.run("mcp__fake__explode", {}, "/tmp")
        server = client.servers[0]
        assert "no tool nope" in _call(server, "nope")
    finally:
        client.stop()


def _call(server, name):
    from onecli.mcp import _make_fn
    return _make_fn(server, name)()


def test_non_text_content_is_named_rather_than_dropped(registry):
    client = mcp.MCPClient([spec()])
    client.start()
    try:
        client.register(registry, toolkit.Tool)
        assert toolkit.run("mcp__fake__touch", {}, "/tmp") == "[image content, not text]"
    finally:
        client.stop()


def test_a_server_that_logs_to_stdout_still_works(registry):
    """Servers write logs where their protocol lives. Skip, do not choke."""
    client = mcp.MCPClient([spec("noisy")])
    client.start()
    try:
        assert client.failures == []
        client.register(registry, toolkit.Tool)
        assert toolkit.run("mcp__fake__echo", {"text": "still here"}, "/tmp") == "still here"
    finally:
        client.stop()


def test_a_server_that_dies_is_reported_and_the_rest_still_run(registry):
    client = mcp.MCPClient([spec("dies", "broken"), spec("ok", "good")])
    client.start()
    try:
        assert [n for n, _ in client.failures] == ["broken"]
        added = client.register(registry, toolkit.Tool)
        assert all(n.startswith("mcp__good__") for n in added)
    finally:
        client.stop()


def test_a_hung_server_hits_a_deadline_instead_of_hanging_the_agent(monkeypatch, registry):
    monkeypatch.setattr(mcp, "CALL_TIMEOUT", 1.0)
    client = mcp.MCPClient([spec("hang")])
    client.start()
    try:
        client.register(registry, toolkit.Tool)
        out = toolkit.run("mcp__fake__echo", {"text": "x"}, "/tmp")
        assert "no response within" in out
    finally:
        client.stop()


def test_a_command_that_does_not_exist_is_a_failure_not_a_crash():
    client = mcp.MCPClient([mcp.ServerSpec(name="ghost", command="no-such-binary")])
    client.start()
    assert client.servers == []
    assert client.failures[0][0] == "ghost"
    assert "could not start" in client.failures[0][1]


# --- how the rest of one-cli sees them -------------------------------------

def test_read_only_hint_decides_mutability_and_the_read_only_surface(registry):
    client = mcp.MCPClient([spec()])
    client.start()
    try:
        client.register(registry, toolkit.Tool)
        assert toolkit.is_mutating("mcp__fake__echo") is False    # readOnlyHint
        assert toolkit.is_mutating("mcp__fake__touch") is True    # says nothing
        surface = permissions.tool_names_for("readOnly")
        assert "mcp__fake__echo" in surface
        assert "mcp__fake__touch" not in surface
    finally:
        client.stop()


def test_only_a_boolean_true_read_only_hint_is_read_only(registry):
    from types import SimpleNamespace
    client = mcp.MCPClient([])
    client.servers = [SimpleNamespace(spec=SimpleNamespace(name="s"), tools=[
        {"name": "yes", "annotations": {"readOnlyHint": True}},
        {"name": "text", "annotations": {"readOnlyHint": "false"}},
        {"name": "one", "annotations": {"readOnlyHint": 1}},
    ])]
    client.register(registry, toolkit.Tool)
    assert toolkit.is_mutating("mcp__s__yes") is False
    # The string "false" is truthy; it must not unlock --read-only.
    assert toolkit.is_mutating("mcp__s__text") is True
    assert toolkit.is_mutating("mcp__s__one") is True


def test_permission_clauses_can_name_an_mcp_tool(registry):
    client = mcp.MCPClient([spec()])
    client.start()
    try:
        client.register(registry, toolkit.Tool)
        rules = permissions.parse_rules(["mcp__fake__echo"])
        guard = permissions.PermissionGuard(deny=rules, workspace="/tmp")
        assert not guard.check("mcp__fake__echo", {"text": "x"}, False).allowed
        assert guard.check("mcp__fake__touch", {}, True).allowed
    finally:
        client.stop()


def test_the_tool_schema_is_what_the_model_sees(registry):
    client = mcp.MCPClient([spec()])
    client.start()
    try:
        client.register(registry, toolkit.Tool)
        schema = toolkit.schemas(["mcp__fake__echo"])[0]
        assert schema["function"]["name"] == "mcp__fake__echo"
        assert schema["function"]["parameters"]["required"] == ["text"]
    finally:
        client.stop()


# --- schema normalization ----------------------------------------------------
#
# A strict (moonshot-flavoured) endpoint answers 400 to every request with
#
#   tools.function.parameters is not a valid moonshot flavored json schema,
#   details: <At path '$defs.SurfaceElement.properties.paradigm':
#   type is not defined>
#
# One untyped property, in one server, fails every request - including calls
# to tools from other servers, because the tool list is sent as one block. The
# property is pydantic's rendering of `Optional[str]`, which is common in MCP
# servers.


def test_optional_field_gets_an_explicit_type():
    out = mcp.normalize_schema(
        {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None})
    assert out["type"] == "string"
    assert "anyOf" not in out


def test_the_rest_of_the_node_survives_the_hoist():
    """The enum is the reason the field exists; losing it would be worse."""
    out = mcp.normalize_schema(
        {"anyOf": [{"type": "string"}, {"type": "null"}],
         "enum": ["a", "b"], "description": "pick one"})
    assert out["enum"] == ["a", "b"]
    assert out["description"] == "pick one"


def test_a_nested_type_is_hoisted_whole():
    out = mcp.normalize_schema(
        {"anyOf": [{"type": "array", "items": {"type": "string"}},
                   {"type": "null"}]})
    assert out == {"type": "array", "items": {"type": "string"}}


def test_a_real_union_is_left_alone():
    """Two real types: guessing which one the server meant beats nothing."""
    node = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert mcp.normalize_schema(node) == node


def test_an_already_typed_node_is_untouched():
    node = {"type": "string", "description": "x"}
    assert mcp.normalize_schema(node) == node


def test_it_reaches_defs_and_properties():
    schema = {
        "type": "object",
        "properties": {"doc": {"$ref": "#/$defs/D"}},
        "$defs": {"D": {"type": "object", "properties": {
            "p": {"anyOf": [{"type": "string"}, {"type": "null"}]}}}},
    }
    out = mcp.normalize_schema(schema)
    assert out["$defs"]["D"]["properties"]["p"]["type"] == "string"


def test_registered_tools_carry_normalized_parameters(tmp_path):
    """The repair has to be on the path that builds the tool, not beside it."""
    client = mcp.MCPClient([spec()])
    client.start()
    try:
        registry: dict = {}
        client.register(registry, toolkit.Tool)
        for tool in registry.values():
            _assert_every_node_typed(tool.parameters)
    finally:
        client.stop()


def _assert_every_node_typed(node):
    if isinstance(node, list):
        for x in node:
            _assert_every_node_typed(x)
        return
    if not isinstance(node, dict):
        return
    if "anyOf" in node and "type" not in node:
        branches = [b for b in node["anyOf"]
                    if isinstance(b, dict) and b.get("type") != "null"]
        assert len(branches) > 1, f"untyped optional survived: {node}"
    for value in node.values():
        _assert_every_node_typed(value)


def test_notifications_do_not_extend_rpc_deadline(monkeypatch):
    monkeypatch.setattr(mcp, 'CALL_TIMEOUT', .1)
    server = mcp.StdioServer(spec('notifications'))
    assert server.start()
    proc = server.proc
    try:
        with pytest.raises(mcp.MCPError, match='response within'):
            server.call('echo', {})
        assert server.proc is None
        assert proc.poll() is not None
    finally:
        server.stop()


@pytest.mark.parametrize('response', ['[]', 'null', '{"id":1,"result":[]}', '{"id":1,"error":"bad"}'])
def test_invalid_rpc_shapes_become_tool_errors(response):
    from types import SimpleNamespace
    server = mcp.StdioServer(spec())
    server.proc = SimpleNamespace(stdout=True)
    server._send = lambda payload: None
    server.stop = lambda: None
    server._read_line = lambda timeout: response
    with pytest.raises(mcp.MCPError, match='invalid'):
        server._request('test', {}, .1)


def test_valid_rpc_error_keeps_server_available():
    server = mcp.StdioServer(spec())
    assert server.start()
    try:
        with pytest.raises(mcp.MCPError, match='no tool'):
            server.call('unknown', {})
        assert server.call('echo', {'text': 'still available'}) == 'still available'
    finally:
        server.stop()
