"""Removed release surface stays removed without disabling registration."""
import dataclasses
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("module,name,field", [
    ("onecli.checkpoints", "TurnSlot", "seq"),
    ("onecli.serving.recipes", "Recipe", "needs_draft"),
])
def test_unused_fields_removed(module, name, field):
    import importlib
    cls = getattr(importlib.import_module(module), name)
    assert field not in {f.name for f in dataclasses.fields(cls)}


