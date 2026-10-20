from onecli.transcript import layout
from onecli.terminal import cells


def test_prose_wraps_at_words_and_user_continuations_indent():
    rows = layout(['> hello wonderful world'], 18)
    assert [''.join(t for t, _ in r).rstrip() for r in rows] == ['> hello wonderful', '  world']
    assert all(role == 'user' for r in rows for _, role in r)


def test_markdown_styles_and_literal_code_are_distinct():
    rows = layout(['## Result', 'Some **strong** and `code`.', '```', '**literal**', '```'], 60)
    assert rows[0] == [('Result', 'heading')]
    assert ('strong', 'strong') in rows[1]
    assert ('code', 'code') in rows[1]
    assert rows[3] == [('**literal**', 'code')]


def test_tool_output_alignment_unicode_and_error_hierarchy():
    rows = layout(['│ NAME    SIZE │', 'error: failed', '界' * 30], 16)
    assert rows[0] == [('│ NAME    SIZE │', 'output')]
    assert rows[1][0][1] == 'error'
    assert all(sum(cells(t) for t, _ in r) <= 16 for r in rows)


def test_mobile_boxes_resize_without_orphaned_borders_or_lost_output():
    rows = layout(['╭─ bash ' + '─' * 60 + '╮', '│ ' + '0123456789' * 5 + ' │', '╰' + '─' * 68 + '╯'], 24)
    text = [''.join(t for t, _ in row) for row in rows]
    assert all(cells(row) == 24 for row in text)
    assert text[0].endswith('╮') and text[-1].endswith('╯')
    assert ''.join(row[1:-1].strip() for row in text[1:-1]) == '0123456789' * 5


def test_long_fence_labels_stay_inside_terminal():
    rows = layout(['```' + 'language' * 12, 'literal', '```'], 20)
    assert all(sum(cells(t) for t, _ in row) <= 20 for row in rows)


def test_layout_cache_updates_for_streaming_and_resize():
    from onecli.terminal import Surface
    ui = Surface()
    ui.write('first response')
    first = ui.transcript_rows(30)
    assert ui.transcript_rows(30) is first
    ui.write(' continues\nnext')
    assert ui.transcript_rows(30) is not first
    assert len(ui.transcript_rows(10)) > len(ui.transcript_rows(30))
