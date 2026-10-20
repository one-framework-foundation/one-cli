"""How the chain is built when nobody names it, and what it refuses to include."""

from __future__ import annotations


from onecli import config, fallback


def be(name, url, profile="default", model="m", fallback_names=None):
    return config.Backend(name=name, model=model, base_url=url, profile=profile,
                          fallback=list(fallback_names or []))


def _build(b, timeout=1.0):
    return f"provider:{b.name}"


# --- what the flag and the env say --------------------------------------------

def test_a_spec_reads_commas_repeats_and_the_off_words():
    assert fallback.parse_spec(["a,b", "c"]) == ["a", "b", "c"]
    assert fallback.parse_spec([" a , b "]) == ["a", "b"]
    assert fallback.parse_spec(["off"]) == []
    assert fallback.parse_spec(["none"]) == []
    assert fallback.parse_spec(["auto"]) is None
    assert fallback.parse_spec(None) is None


def test_off_anywhere_in_the_spec_wins():
    """`--fallback-model a --no-fallback` must not answer somewhere else."""
    assert fallback.parse_spec(["a", "off"]) == []


def test_the_env_pins_the_behaviour_without_argv(monkeypatch):
    monkeypatch.delenv("ONE_DETACHED_OWNER", raising=False)
    monkeypatch.setenv(fallback.FALLBACK_ENV, "off")
    assert fallback.spec_from_env() == []
    monkeypatch.setenv(fallback.FALLBACK_ENV, "spare,third")
    assert fallback.spec_from_env() == ["spare", "third"]


# --- the auto rank ------------------------------------------------------------

def test_auto_drops_anything_on_the_primarys_own_endpoint():
    """One endpoint is one outage: a sibling model there is not a fallback."""
    primary = be("local-a", "http://192.0.2.13:8137/v1")
    found = [primary,
             be("local-a-b", "http://192.0.2.13:8137/v1"),
             be("kimi", "https://api.moonshot.ai/v1")]
    assert [b.name for b in fallback.auto_backends(primary, found)] == ["kimi"]


def test_auto_prefers_the_same_profile_then_keeps_the_operators_order():
    primary = be("a", "http://a/v1", profile="coding")
    found = [primary,
             be("z-other", "http://z/v1", profile="default"),
             be("m-coding", "http://m/v1", profile="coding")]
    assert [b.name for b in fallback.auto_backends(primary, found)] == [
        "m-coding", "z-other"]


def test_auto_puts_one_local_ollama_last_and_only_one():
    primary = be("kimi", "https://api.moonshot.ai/v1")
    found = [primary,
             be("ollama/gemma4:27b", "http://127.0.0.1:11434/v1"),
             be("ollama/qwen3.6:9b", "http://127.0.0.1:11434/v1"),
             be("dashscope", "https://dashscope.example/v1")]
    ranked = [b.name for b in fallback.auto_backends(primary, found)]
    assert ranked == ["dashscope", "ollama/gemma4:27b"]


def test_the_local_floor_is_never_an_embedding_model():
    """An embedder on a chat endpoint errors every turn: not a floor."""
    primary = be("kimi", "https://api.moonshot.ai/v1")
    found = [primary,
             be("ollama/bge-m3:latest", "http://127.0.0.1:11434/v1"),
             be("ollama/nomic-embed-text:latest", "http://127.0.0.1:11434/v1"),
             be("ollama/qwen3.5:9b", "http://127.0.0.1:11434/v1")]
    assert [b.name for b in fallback.auto_backends(primary, found)] == [
        "ollama/qwen3.5:9b"]


def test_auto_is_capped_so_a_broken_request_does_not_tour_every_lane():
    primary = be("a", "http://a/v1")
    found = [primary] + [be(f"b{i}", f"http://b{i}/v1") for i in range(10)]
    assert len(fallback.auto_backends(primary, found)) == fallback.MAX_AUTO_RUNGS


def test_the_local_floor_keeps_its_rung_when_the_cap_is_full():
    """Every paid lane busy at once is exactly when the free one matters."""
    primary = be("a", "http://a/v1")
    found = ([primary] + [be(f"b{i}", f"http://b{i}/v1") for i in range(10)]
             + [be("ollama/x", "http://127.0.0.1:11434/v1")])
    ranked = [b.name for b in fallback.auto_backends(primary, found)]
    assert len(ranked) == fallback.MAX_AUTO_RUNGS
    assert ranked[-1] == "ollama/x"


# --- what a drop-in declares for itself ---------------------------------------

def test_a_drop_in_declares_its_own_successors(tmp_path, monkeypatch):
    (tmp_path / "primary.env").write_text(
        "ONE_PROVIDER_MODEL=big\n"
        "OPENAI_BASE_URL=http://primary/v1\n"
        "ONE_PROVIDER_FALLBACK=spare, third\n")
    (tmp_path / "spare.env").write_text(
        "ONE_PROVIDER_MODEL=small\nOPENAI_BASE_URL=http://spare/v1\n")
    monkeypatch.setattr(config, "PROVIDER_DIR", tmp_path)
    monkeypatch.delenv(config.PROVIDER_DIR_ENV, raising=False)

    found = {b.name: b for b in config.discover(include_local=False)}
    assert found["primary"].fallback == ["spare", "third"]
    assert found["spare"].fallback == []


def test_a_declared_chain_never_reaches_the_request_body(tmp_path, monkeypatch):
    """`extra` is merged into the wire body; a chain is ours, not the wire's."""
    (tmp_path / "primary.env").write_text(
        "ONE_PROVIDER_MODEL=big\n"
        "OPENAI_BASE_URL=http://primary/v1\n"
        "ONE_PROVIDER_FALLBACK=spare\n"
        "ONE_PROVIDER_MODALITIES=vision\n")
    monkeypatch.setattr(config, "PROVIDER_DIR", tmp_path)
    monkeypatch.delenv(config.PROVIDER_DIR_ENV, raising=False)

    from onecli import providers
    primary = config.discover(include_local=False)[0]
    assert "fallback" not in primary.extra
    wire = providers.build(primary)
    assert wire.extra_body == {}          # modalities is ours too


# --- the built chain ----------------------------------------------------------

def test_off_is_a_chain_of_one():
    chain = fallback.build(be("a", "http://a/v1"), [], build_provider=_build)
    assert chain.rungs[0].name == "a" and chain.spares == []
    assert chain.has_next() is False and chain.source == "off"


def test_an_explicit_chain_wins_over_what_the_drop_in_declared(monkeypatch):
    primary = be("a", "http://a/v1", fallback_names=["declared"])
    monkeypatch.setattr(config, "resolve",
                        lambda name=None, **kw: be(name, f"http://{name}/v1"))
    chain = fallback.build(primary, ["chosen"], build_provider=_build)
    assert chain.spares == ["chosen"] and chain.source == "explicit"


def test_a_declared_chain_is_used_when_the_operator_names_nothing(monkeypatch):
    primary = be("a", "http://a/v1", fallback_names=["declared"])
    monkeypatch.setattr(config, "resolve",
                        lambda name=None, **kw: be(name, f"http://{name}/v1"))
    chain = fallback.build(primary, None, build_provider=_build)
    assert chain.spares == ["declared"] and chain.source == "declared"


def test_a_name_that_resolves_to_nothing_warns_and_shortens_the_chain(monkeypatch):
    def resolve(name=None, **kw):
        if name == "ghost":
            raise RuntimeError("no backend matching 'ghost'")
        return be(name, f"http://{name}/v1")

    monkeypatch.setattr(config, "resolve", resolve)
    chain = fallback.build(be("a", "http://a/v1"), ["ghost", "real"],
                           build_provider=_build)
    assert chain.spares == ["real"]
    assert chain.warnings and "ghost" in chain.warnings[0]


def test_the_primary_is_never_also_a_rung(monkeypatch):
    monkeypatch.setattr(config, "resolve",
                        lambda name=None, **kw: be(name, f"http://{name}/v1"))
    chain = fallback.build(be("a", "http://a/v1"), ["a", "b", "b"],
                           build_provider=_build)
    assert chain.spares == ["b"]


def test_auto_is_withheld_from_an_explicit_endpoint(monkeypatch):
    """--base-url says where this run goes; auto must not answer elsewhere."""
    called = []
    monkeypatch.setattr(fallback, "auto_backends",
                        lambda *a, **k: called.append(1) or [])
    chain = fallback.build(be("explicit", "http://given/v1"), None,
                           build_provider=_build, allow_auto=False)
    assert chain.spares == [] and chain.source == "off" and called == []


def test_advance_is_one_way_and_reports_each_hop():
    chain = fallback.Chain(rungs=[fallback.Rung("a", None),
                                  fallback.Rung("b", None)])
    assert chain.current.name == "a"
    note = chain.advance("HTTP 429")
    assert note == {"from": "a", "to": "b", "reason": "HTTP 429"}
    assert chain.current.name == "b"
    assert chain.advance("HTTP 429") is None      # bottom of the ladder
    assert chain.history == [note]



def test_standalone_keeps_environment_selection(monkeypatch):
    monkeypatch.delenv('ONE_DETACHED_OWNER',raising=False)
    monkeypatch.setenv('ONE_CLI_FALLBACK','spare')
    assert fallback.spec_from_env()==['spare']


def test_host_owner_does_not_discover_or_build_spare(monkeypatch):
    monkeypatch.setenv('ONE_DETACHED_OWNER','caller')
    monkeypatch.setenv('ONE_CLI_FALLBACK','auto')
    def forbidden(*a,**kw):raise AssertionError('host must not discover fallback backends')
    monkeypatch.setattr(fallback,'declared_for',forbidden)
    primary=config.Backend(name='kimi',model='kimi-for-coding',base_url='https://example.invalid',fallback=['premium'])
    built=[]
    chain=fallback.build(primary,fallback.spec_from_env(),build_provider=lambda b,t:built.append(b.name),timeout=1)
    assert built==['kimi'] and chain.source=='off' and not chain.has_next()
