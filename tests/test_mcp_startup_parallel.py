"""Independent handshakes overlap; published tools retain config order."""
import threading

from onecli import mcp


def test_initialization_overlaps_without_reordering_failures(monkeypatch):
    rendezvous = threading.Barrier(3)
    client = mcp.MCPClient([
        mcp.ServerSpec(name=name, command="unused")
        for name in ("first", "broken", "last")
    ])

    def start(server):
        # A sequential initializer breaks this barrier and cannot pass.
        rendezvous.wait(timeout=2)
        server.error = "unavailable" if server.spec.name == "broken" else ""
        return not server.error

    monkeypatch.setattr(mcp.StdioServer, "start", start)
    assert [s.spec.name for s in client.start()] == ["first", "last"]
    assert client.failures == [("broken", "unavailable")]


def test_no_servers_needs_no_executor():
    client = mcp.MCPClient([])
    assert client.start() == []
