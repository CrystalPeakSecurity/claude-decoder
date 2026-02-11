# claude-decoder

Recover files and conversations from Claude Code session logs.

Claude Code stores every tool invocation in JSONL session logs under `~/.claude/projects/`. claude-decoder parses these logs to reconstruct source files and render conversation transcripts.

## Install

```
uv tool install .
```

Or run directly:

```
uv run claude-decoder /path/to/project
```

## Usage

Point it at any project directory that has Claude Code history:

```
claude-decoder /path/to/project
```

This opens an interactive TUI where you can:

- **Restore files** — reconstruct source files with a diff preview. Write back in-place, to a separate directory, or as a git patch.
- **Browse conversations** — preview sessions and dump to HTML or plain text.
- **Search conversations** — full-text search across all sessions.
- **Resume sessions** — jump back into a session with `claude --resume`.

For scripting, the `restore` and `chat` subcommands expose the same functionality non-interactively. Run `claude-decoder restore --help` or `claude-decoder chat --help` for options.

## Library

```python
from claude_decoder.models import parse_entries, extract_operations, Write, Edit, BashCommand

for entry in parse_entries("session.jsonl"):
    print(entry.type, entry.timestamp)

for op in extract_operations("session.jsonl"):
    match op:
        case Write(file_path=p):       print(f"  write {p}")
        case Edit(file_path=p):        print(f"  edit {p}")
        case BashCommand(command=c):   print(f"  $ {c}")
```

## How it works

Claude Code's session logs are JSONL files where each line is an entry containing content blocks. Tool invocations span two entries: an assistant entry with a `tool_use` block, and a user entry with the matching `tool_result`.

File reconstruction works by:

1. Joining `tool_use`/`tool_result` pairs into unified operations (Read, Write, Edit, etc.)
2. Finding the latest **snapshot** for each file — the most recent Write or full Read
3. Applying subsequent Edit operations on top of that snapshot

Files that were written or fully read can be recovered exactly. Files that were only partially read or only edited (with no baseline snapshot) cannot be reconstructed — the tool reports these as "unrecoverable" and warns about any edits that failed to apply.
