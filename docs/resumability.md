# Session resumability

The interactive CLI supports:

- `/resume ? words`: case-insensitive search over saved names, IDs and opening prompts.
- `/resume @workspace`: sessions recorded in the selected workspace.
- `/rename name`: name the current session.
- `/fork name`: save an independent transcript and latest todo snapshot; prints the new ID to resume. The current session remains active.
- `--sessions --session-search words` and `--sessions-here`, with `--json` support.
- Curses drafts saved after input events and restored when reopening the same session ID with `--resume`. Submission clears the saved draft. Permission answers are excluded. Draft files use atomic replacement and owner-only permissions.

`/resume <id>` adopts that session's history into the current session: it writes the imported history under the current ID. Startup `--resume <id>` continues the specified ID. Neither sends a prompt merely to list, name or fork sessions.

Transcripts without workspace metadata remain visible in global search, but cannot be reliably assigned to a workspace. Search covers names and opening prompts, not full transcript text. Draft recovery covers unsent editor text, not submitted queue entries or partial provider output. Concurrent editors of the same session use last-writer-wins draft storage. Legacy `one chat` has its separate command interface.
