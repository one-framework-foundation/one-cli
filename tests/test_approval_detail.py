"""Approval shows what will run.

`_detail` knew six argument names. A `code` call (its program is `source`)
was approved as a bare "run code", and an MCP tool whose arguments carry other
names was approved blank: the person said yes to something they could not see.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

from onecli import chat
from onecli.providers import ToolCall


def _prompt(name: str, arguments: dict) -> str:
    prompts: list[str] = []

    def ask(prompt):
        prompts.append(prompt)
        return "n"
    c = SimpleNamespace(allow_all=False, ask_line=ask, tools="all", err=io.StringIO())
    assert chat._confirm(c, ToolCall(id="x", name=name, arguments=arguments)) == chat.DECLINED
    return prompts[0]


def test_a_code_call_shows_the_whole_program():
    lines = [f"step_{i} = one.read('f{i}.txt')" for i in range(200)]
    source = "\n".join(lines + ["import shutil; shutil.rmtree('/')"])
    prompt = _prompt("code", {"source": source})
    # Every line, including the last one, which is the one that matters.
    for line in lines:
        assert line in prompt
    assert "shutil.rmtree('/')" in prompt


def test_an_mcp_call_shows_every_argument():
    prompt = _prompt("db_exec", {"sql": "DROP TABLE users", "database": "prod"})
    assert "DROP TABLE users" in prompt and "prod" in prompt
    assert '"sql"' in prompt and '"database"' in prompt


def test_fallback_detail_escapes_terminal_controls():
    prompt = _prompt("code", {"source": "print(1)\x1b[2J\x9b2Jhidden"})
    assert "\\x1b[2J" in prompt and "\\x9b2Jhidden" in prompt
    prompt = _prompt("mcp_tool", {"text": "a\x1b]0;owned\x07b"})
    assert "\x1b]0;owned" not in prompt


def test_no_arguments_is_still_blank():
    assert chat._detail({}) == ""
