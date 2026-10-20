"""`-m <name>` that is offline or fails never lands somewhere else silently
Every mode of `on_unavailable_model`, and
`fallback_allow_paid`, against real endpoints that are up or down.
"""

from __future__ import annotations

import pytest

from onecli import cli, config, fallback, presence, unavailable
from onecli._vendor import dropins
from test_presence_gate import Rig


@pytest.fixture
def lanes(monkeypatch, tmp_path):
    """`down` is offline, `alt` is a free lane that is up, `dash` is paid."""
    dropins._PRESENCE_CACHE.clear()
    monkeypatch.setattr(presence, "SELECT_MAX_AGE", 0.0)
    for var in (unavailable.MODE_ENV, unavailable.ALLOW_PAID_ENV,
                fallback.FALLBACK_ENV, config.PROVIDER_DIR_ENV):
        monkeypatch.delenv(var, raising=False)
    down, alt = Rig(), Rig()
    alt.up()
    (tmp_path / "down.env").write_text(
        f"ONE_PROVIDER_MODEL=m\nOPENAI_BASE_URL={down.base}\n")
    (tmp_path / "alt.env").write_text(
        f"ONE_PROVIDER_MODEL=m\nOPENAI_BASE_URL={alt.base}\n")
    (tmp_path / "dash.env").write_text(
        "ONE_PROVIDER_MODEL=qwen\nOPENAI_BASE_URL=https://dashscope.example.com/v1\n"
        "OPENAI_API_KEY=placeholder\n")
    monkeypatch.setattr(config, "PROVIDER_DIR", tmp_path)
    monkeypatch.setattr(config, "llamacpp_backends", lambda *a, **k: [])
    yield {"down": down, "alt": alt}
    alt.down()
    dropins.await_presence_priming(5)
    dropins._PRESENCE_CACHE.clear()


class Args:
    model = "down"
    print_mode = False
    json = False
    on_unavailable_model = None
    fallback_allow_paid = False


def _resolve(args):
    primary = config.resolve(name="down")
    return unavailable.resolve(args, primary, presence.refusal(primary))


def test_error_mode_exits_with_the_reason_and_free_first_suggestions(lanes, capsys):
    assert cli.main(["-m", "down", "--on-unavailable-model", "error", "hi"]) == 2
    err = capsys.readouterr().err
    assert "down is offline" in err
    assert err.index("alt (m)") < err.index("dash (qwen)  [paid]")


def test_ask_without_a_terminal_behaves_as_error(lanes, capsys):
    # pytest's stdin is not a terminal, and ask is the default.
    assert cli.main(["-m", "down", "hi"]) == 2
    assert "online alternatives" in capsys.readouterr().err


def test_ask_on_a_terminal_lets_the_person_pick(lanes, monkeypatch):
    monkeypatch.setattr(unavailable, "_interactive", lambda args: True)
    monkeypatch.setattr("builtins.input", lambda: "1")
    assert _resolve(Args()).name == "alt"


def test_ask_on_a_terminal_can_abort(lanes, monkeypatch):
    monkeypatch.setattr(unavailable, "_interactive", lambda args: True)
    monkeypatch.setattr("builtins.input", lambda: "")
    assert _resolve(Args()) is None


def test_fallback_switches_to_the_first_free_lane_and_says_so(lanes, capsys):
    args = Args()
    args.on_unavailable_model = "fallback"
    assert _resolve(args).name == "alt"
    assert "running on alt" in capsys.readouterr().err


def test_fallback_never_reaches_a_paid_lane_unless_allowed(lanes, monkeypatch):
    lanes["alt"].down()                   # only the paid lane is left
    args = Args()
    args.on_unavailable_model = "fallback"
    assert _resolve(args) is None
    monkeypatch.setenv(unavailable.ALLOW_PAID_ENV, "1")
    assert _resolve(args).name == "dash"


def test_the_env_sets_the_host_default_and_the_flag_wins(monkeypatch):
    monkeypatch.setenv(unavailable.MODE_ENV, "fallback")
    args = Args()
    assert unavailable.mode(args) == "fallback"
    args.on_unavailable_model = "error"
    assert unavailable.mode(args) == "error"
    monkeypatch.setenv(unavailable.MODE_ENV, "typo")
    args.on_unavailable_model = None
    assert unavailable.mode(args) == "ask"


def test_a_named_model_gets_no_implicit_chain(monkeypatch):
    monkeypatch.delenv(unavailable.MODE_ENV, raising=False)
    args = Args()
    assert unavailable.chain_spec(args, None) == []          # auto/declared off
    assert unavailable.chain_spec(args, ["x"]) == ["x"]      # --fallback-model wins
    args.on_unavailable_model = "fallback"
    assert unavailable.chain_spec(args, None) is None
    args.model = None
    args.on_unavailable_model = None
    assert unavailable.chain_spec(args, None) is None        # unnamed: unchanged


def test_an_implicit_chain_drops_paid_rungs_when_not_allowed(lanes):
    primary = config.resolve(name="down")
    chain = fallback.build(primary, None, build_provider=lambda b, t: object(),
                           allow_paid=False)
    assert "dash" not in chain.spares
    assert any("dash" in w and "paid" in w for w in chain.warnings)
    named = fallback.build(primary, ["dash"], build_provider=lambda b, t: object(),
                           allow_paid=False)
    assert named.spares == ["dash"]


@pytest.mark.parametrize("error,why", [
    ("HTTP 401 Invalid API Key", "auth"),
    ("HTTP 404 model not found", "not found"),
    ("connection refused", "offline"),
    ("HTTP 500", "failed"),
])
def test_a_failure_names_its_reason(error, why):
    assert unavailable.reason_of(error) == why
