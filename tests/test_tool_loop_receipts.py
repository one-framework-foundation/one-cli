from onecli.tools.shell import bash
from onecli import terminal
from onecli.toolview import result

def test_tail_preserves_failure():
    assert bash('.', "printf 'model failed to load\\n' >&2; (exit 2) | tail -40").startswith('exit 2\n')

def test_successful_pipeline():
    assert bash('.', 'printf hello | cat') == 'exit 0\nhello'

def test_compact_command_expands_losslessly():
    ui = terminal.Surface()
    command = 'python3 look.py --ask "' + 'describe this image ' * 50 + '"'
    ui.tool_start('bash', {'command': command}, True)
    rows = ui.transcript_rows(48)
    assert len(rows) <= 3
    ui.key('\x0f')
    text = ''.join(s for row in ui.transcript_rows(48) for s, _ in row)
    assert command in text

def test_error_flood_bounded_and_final_failure_visible():
    output = 'exit 2\n' + '\n'.join('error detail '+str(i) for i in range(100)) + '\nmodel failed to load'
    rows = result(output)
    assert len(rows) <= 8
    assert 'model failed to load' in '\n'.join(rows)
    assert '\n'.join(line[2:] for line in result(output, True)) == output


def test_long_output_narrow_screen_expands():
    ui = terminal.Surface()
    output = 'exit 2\nerror: ' + 'detail ' * 100
    ui.tool_result(output)
    assert len(ui.transcript_rows(48)) <= 4
    ui.key('\x0f')
    assert len(ui.transcript_rows(48)) > 10


def test_command_rerenders_after_resize():
    ui = terminal.Surface()
    ui.tool_start('bash', {'command': 'echo ' + 'x' * 100})
    assert 'Ctrl-O' not in str(ui.transcript_rows(120))
    assert 'Ctrl-O' not in str(ui.transcript_rows(40))
    assert '…' in str(ui.transcript_rows(40))
    ui.key('\x0f')
    assert 'x' * 30 in str(ui.transcript_rows(40))


def test_intentional_pipeline_recovery():
    assert bash('.', '(exit 2) | cat || printf recovered') == 'exit 0\nrecovered'
