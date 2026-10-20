"""Semantic transcript layout; editor wrapping remains character-exact."""
import re
from onecli.terminal import cells

INLINE = re.compile(r'(`[^`]+`|\*\*[^*]+\*\*)')


def layout(lines, width):
    """Return rows of (text, role) spans, preserving literal tool/code output."""
    width = max(2, width)
    rows = []
    code = False
    for line in lines:
        stripped = line.strip()
        literal = code or stripped.startswith(('│', '╭', '╰', '┌', '└'))
        role = 'body'
        if stripped.startswith('```'):
            code = not code
            line = line.replace('```', '── ', 1)
            literal, role = True, 'meta'
        if literal:
            role = role if role == 'meta' else 'code' if code else 'output'
            if stripped.startswith(('╭', '╰', '┌', '└')):
                role = 'meta'
            elif 'lines folded' in line:
                role = 'meta'
        elif line.startswith('> '):
            role = 'user'
        elif re.match(r'^\s*[◆◇]\s|^\s*\d+\s+(bash|read|write)\b', line):
            role = 'tool'
        elif re.match(r'^(error|failed|terminal stopped)\b', stripped, re.I):
            role = 'error'
        elif re.match(r'^(theme:|exit \d|.* · .*tokens)', stripped):
            role = 'meta'
        elif re.match(r'^#{1,6} ', line):
            line = re.sub(r'^#{1,6} ', '', line)
            role = 'heading'
        if not code and cells(line) > width and stripped.startswith(('╭', '╰', '┌', '└', '│')):
            left, right = stripped[0], stripped[-1]
            if left == '│':
                content = stripped[1:-1].rstrip() if right == '│' else stripped[1:].rstrip()
                from onecli.terminal import wrap
                for chunk in wrap(content, max(1, width - 2)):
                    rows.append([('│' + chunk + ' ' * max(0, width - 2 - cells(chunk)) + '│', role)])
            else:
                from onecli.terminal import wrap
                middle = stripped[1:-1].rstrip('─')
                middle = wrap(middle, max(1, width - 3))[0]
                rows.append([(left + middle + '─' * max(0, width - 2 - cells(middle)) + right, role)])
            continue
        spans = []
        for part in ([line] if literal or role == 'user' else INLINE.split(line)):
            if not literal and role != 'user' and part.startswith('**') and part.endswith('**') and len(part) > 4:
                spans.append((part[2:-2], 'strong'))
            elif not literal and role != 'user' and part.startswith('`') and part.endswith('`') and len(part) > 2:
                spans.append((part[1:-1], 'code'))
            elif part:
                spans.append((part, role))
        # Word tokens retain their styling; only overlong tokens split at cells.
        row, size = [], 0
        indent = '  ' if line.startswith('> ') or re.match(r'^\s*(?:[-*+] |\d+[.)] )', line) else ''
        for text, style in spans:
            tokens = list(text.expandtabs(4)) if literal else re.findall(r'\s+|\S+', text.expandtabs(4))
            for token in tokens:
                if not literal and not token.isspace() and size and size + cells(token) > width:
                    rows.append(row)
                    row, size = ([(indent, role)] if indent else []), len(indent)
                if not literal and token.isspace() and size == len(indent) and rows:
                    continue
                for ch in token:
                    n = cells(ch)
                    if size + n > width:
                        rows.append(row)
                        row, size = [], 0
                    if row and row[-1][1] == style:
                        row[-1] = (row[-1][0] + ch, style)
                    else:
                        row.append((ch, style))
                    size += n
        rows.append(row or [('', role)])
    return rows
