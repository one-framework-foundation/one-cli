# Slash menu behavior

How the `/` command menu responds to each key.

Sources:
- **Claude Code**: observed through a terminal, since it is closed source.
- **Codex**: the open-source TUI's slash-popup key handling.

| Key | Claude Code (observed) | Codex (source) | one-cli |
|---|---|---|---|
| Down | moves highlight; wraps last to first | `move_down_wrap`, then `ensure_visible` scrolls the window | moves highlight, wraps, window scrolls |
| Up | moves highlight; wraps first to last | `move_up_wrap`, then `ensure_visible` | same as Down, upward |
| Ctrl-P / Ctrl-N | not checked | aliases for Up/Down | not added: Ctrl-N is one-cli's newline key |
| Tab | completes the highlighted command into the input, closes menu | completes the selected command (a few commands dispatch at once) | completes the highlighted command plus a space |
| Enter | runs the highlighted command (`/hel` + Enter opened help) | runs the selected command, clears the draft | runs the highlighted command |
| Esc | closes the menu, keeps the draft | closes the popup, keeps the draft, stays closed for that token | closes the menu, keeps the draft; reopens on the next edit |
| PageUp / PageDown | no visible highlight change seen (inconclusive) | not handled by the slash popup | unchanged: scrolls the transcript |
| Typing | filters the list | filters and resets the selection to the top | filters, highlight back to the top |
| Backspace past `/` | not checked separately | popup closes once the draft is no longer a slash token | unchanged |
| Visible rows | depends on terminal height | `MAX_POPUP_ROWS = 8` | 8-row window that follows the highlight, `i/N` counter |

Up and Down only move through history when no menu is open.
