"""Tests for the one-cli TUI themes and ASCII GlyphJourney integration."""

from onecli import cli, config, tui


def test_tui_themes_registered():
    for theme in ("mono", "pill", "glyph", "phosphor", "zen"):
        assert theme in tui.THEMES
    assert "avionics" in tui.THEMES
    assert "darkflow" in tui.THEMES
    assert "aviary" in tui.THEMES
    assert "bento" in tui.THEMES
    assert "cyberdeck" in tui.THEMES


def test_tui_render_all_banners():
    for theme in tui.THEMES:
        banner = tui.render_banner(
            theme,
            version="0.4.0",
            backend_name="opencode-go-deepseek-v4.1-flash",
            model_name="deepseek-v4.1-flash",
            workspace="/home/user",
            sid="test_session_123",
        )
        assert len(banner) > 0
        assert "deepseek-v4.1-flash" in banner


def test_tui_banners_use_local_host_without_fake_telemetry(monkeypatch):
    import socket

    monkeypatch.setattr(socket, "gethostname", lambda: "outsider.example.org")
    for theme in tui.THEMES:
        banner = tui.render_banner(theme, "0.4.0", "fixture", "model", "/tmp", "sid")
        assert "outsider" in banner.lower(), theme
        assert "0.9s" not in banner and "0.88s" not in banner, theme
        assert "SYS:OK" not in banner, theme


def test_cli_theme_flag_selects_theme(monkeypatch):
    args = cli.build_parser().parse_args(["--theme", "cyberdeck"])
    backend = config.Backend(name="test", model="deepseek-v4.1-flash", base_url="http://127.0.0.1:1")
    banner = cli._banner(args, backend, None, "default", "sid999")
    assert "TERMINAL DECK" in banner

    args_pill = cli.build_parser().parse_args(["--theme", "pill"])
    banner_pill = cli._banner(args_pill, backend, None, "default", "sid999")
    assert "Chat mode active" in banner_pill
    assert "deepseek-v4.1-flash" in banner_pill
