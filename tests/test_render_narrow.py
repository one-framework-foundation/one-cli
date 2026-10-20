"""On a narrow terminal the post-turn rule and stat line each take one row."""
import os
from unittest import mock

from onecli import render


def _cols(n):
    return mock.patch("onecli.render.shutil.get_terminal_size",
                      return_value=os.terminal_size((n, 12)))


def test_rule_fits_a_narrow_terminal():
    for cols in (30, 40):
        with _cols(cols):
            assert len(render.rule()) <= cols - 1


def test_footer_is_two_unwrapped_rows_on_a_narrow_terminal():
    bits = ["sample-run · 3 steps · 2 tools · 2,460 input processed"
            " · 916 current context · 288 out"]
    for cols in (30, 40):
        with _cols(cols):
            rows = render.footer(bits).split("\n")
            assert len(rows) == 2
            assert all(len(r) <= cols - 1 for r in rows), rows
