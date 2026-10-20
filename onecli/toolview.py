"""Human tool summaries, with complete output available in the transcript."""
import json
import re


FAILURE = re.compile(r"\b(?:errors?|failed|failure|denied|blocked|traceback)\b", re.I)
PREVIEW_ROWS = 4

# Names whose terminal label reads better than the tool name itself.
_LABELS = {'bash': 'Run', 'read': 'Read', 'write': 'Write', 'edit': 'Edit',
           'grep': 'Search', 'glob': 'Find files', 'ls': 'List',
           'web_search': 'Search web', 'web_fetch': 'Fetch'}


def _bounded(lines):
    """Keep the head and final failure evidence within a fixed row budget."""
    if len(lines) <= PREVIEW_ROWS:
        return lines
    failures = [i for i, line in enumerate(lines) if FAILURE.search(line)][-2:]
    indices = set(failures)
    for i in range(len(lines)):
        if len(indices) >= PREVIEW_ROWS:
            break
        indices.add(i)
    return [lines[i] for i in sorted(indices)]


def _tool_label(name):
    """A tool's own label and default action, where it declares them."""
    from onecli import tools as toolkit  # lazy: the viewer imports no tool at startup
    tool = toolkit.REGISTRY.get(name)
    if tool is None:
        return "", ""
    action = ""
    for key in ("operation", "action"):
        values = (tool.parameters.get("properties", {}).get(key) or {}).get("enum")
        if values:
            action = str(values[0])
            break
    return tool.label, action


def describe(name, args):
    action = args.get('operation') or args.get('action')
    if not action:
        action = _tool_label(name)[1]
    if name in _LABELS:
        name = _LABELS[name]
    else:
        name = _tool_label(name)[0] or name
    detail = args.get('path') or args.get('command') or args.get('pattern') or args.get('topic_key') or ''
    return ' · '.join(str(v).replace('\n', ' ') for v in (name, action, detail) if v)


def result(output, expanded=False):
    try:
        value = json.loads(output)
    except (ValueError, TypeError):
        value = None
    if expanded:
        body = json.dumps(value, indent=2, ensure_ascii=False) if isinstance(value, (dict, list)) else output
        return ['│ ' + line for line in (body.splitlines() or ['(no output)'])]
    if isinstance(value, dict):
        if (isinstance(value.get('items'), list) and 'state_kind' in value
                and 'total' in value and 'error' not in value):
            count = len(value['items'])
            summary = 'No topics waiting' if value['total'] == 0 else f"{count} of {value['total']} topics shown"
            return ['│ ' + summary, '│ Ctrl-O: full tool output']
        rows = []
        for key, item in value.items():
            if isinstance(item, list):
                rows.append(f'{key}: {len(item)} items')
            elif isinstance(item, dict):
                rows.append(f'{key}: {len(item)} fields')
            else:
                rows.extend(f'{key}: {line}' for line in (str(item).splitlines() or ['']))
        return ['│ ' + line for line in _bounded(rows or ['empty result'])] + ['│ Ctrl-O: full tool output']
    if isinstance(value, list):
        return [f'│ {len(value)} items returned', '│ Ctrl-O: full tool output']
    lines = output.splitlines() or ['(no output)']
    # Error-bearing lines stay visible even when the normal output is folded.
    shown = _bounded(lines)
    return ['│ ' + line for line in shown] + (['│ Ctrl-O: full tool output'] if len(lines) > 4 else [])


def command_rows(command, width, expanded=False):
    """One visible row; Ctrl-O reveals the retained command."""
    if expanded:
        return [command]
    return [clip_row(command, max(1, width - 1))]


def clip_row(row, width):
    """One row at most `width` cells wide; a clipped path keeps its filename.

    The directory is the part a person can do without. An error line that
    ends mid-directory hides the one thing the error is about: the file.
    """
    from onecli.terminal import cells, wrap
    parts = wrap(row, width)
    if len(parts) == 1:
        return row
    name = row[row.rfind('/'):] if '/' in row else ''
    if name and cells(name) <= width // 2:
        return wrap(row, width - cells(name) - 1)[0] + '…' + name
    return parts[0] + '…'


def result_rows(output, width, expanded=False):
    rows = result(output, expanded)
    if expanded:
        return rows
    # One summary, with failure evidence taking priority over routine output.
    rows = [row for row in rows if 'Ctrl-O:' not in row]
    rows = [row for row in rows if row.strip() != '│ exit 0']
    failures = [row for row in rows if FAILURE.search(row) or re.search(r"\bexit [1-9]\d*\b", row)]
    if failures:
        shown = failures[-2:]
    else:
        shown = rows[:1] or ['│ Done']
        if len(output.splitlines()) > 2:
            shown = [f'{shown[0]} … ({len(output.splitlines())} lines)']
    return [clip_row(row, max(1, width - 1)) for row in shown]


def plain_result(output, width=80):
    """Compact scrollback output without controls requiring a live surface."""
    from onecli import render
    rows = [clip_row(row, max(8, width - 1)) for row in result(render.untrusted(output))]
    rows = [row.replace('Ctrl-O: full tool output', '… tool output shortened') for row in rows]
    return '\n'.join(render.dim(render.glyph('│') + row[1:]) for row in rows)
