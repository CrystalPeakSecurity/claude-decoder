#!/usr/bin/env python3
"""Render a Claude Code session JSONL as a readable conversation transcript.

Supports plain text (.txt) and HTML (.html) output formats.
"""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from .models import (
    Entry, TextBlock, ThinkingBlock, ImageBlock, ToolUseBlock, ToolResultBlock,
    parse_entries,
)


def _local_tz():
    """Get the current local timezone (computed fresh each call)."""
    return datetime.now().astimezone().tzinfo


# =============================================================================
# INTERMEDIATE REPRESENTATION
# =============================================================================

@dataclass(frozen=True)
class ToolArg:
    """A single named argument to a tool invocation."""
    name: str
    value: str


@dataclass(frozen=True)
class Block:
    """A single renderable block within a conversation turn."""
    kind: str  # "text", "thinking", "tool_use", "tool_result", "tool_result_error", "image", "system", "snapshot", "summary", "task"
    content: str  # Primary text content
    tool_name: str = ""  # For tool_use blocks
    tool_args: tuple[ToolArg, ...] = ()  # Structured args for tool_use
    tool_use_id: str = ""  # For tool_use and tool_result blocks
    snapshot_files: tuple[dict, ...] = ()  # For snapshot blocks


@dataclass(frozen=True)
class Turn:
    """A conversation turn: one or more blocks from the same speaker at the same time."""
    timestamp: datetime | None
    speaker: str  # "USER", "CLAUDE", "TOOL", "SYSTEM", "SNAPSHOT", "SUMMARY", "TASK"
    blocks: tuple[Block, ...] = ()
    session_id: str = ""
    uuid: str = ""
    model: str = ""
    stop_reason: str = ""
    cwd: str = ""


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def _truncate(s: str, max_len: int) -> str:
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def fmt_tool_input(name: str, inp: dict, truncate: bool = False) -> list[ToolArg]:
    """Format tool input as structured args."""
    def _t(s: str, max_len: int) -> str:
        return _truncate(s, max_len) if truncate else s

    args: list[ToolArg] = []

    if name == "Bash":
        args.append(ToolArg("command", inp.get("command", "")))
        if inp.get("timeout"):
            args.append(ToolArg("timeout", str(inp["timeout"])))
    elif name == "Read":
        args.append(ToolArg("file_path", inp.get("file_path", "")))
        if inp.get("offset"):
            args.append(ToolArg("offset", str(inp["offset"])))
        if inp.get("limit"):
            args.append(ToolArg("limit", str(inp["limit"])))
    elif name == "Write":
        args.append(ToolArg("file_path", inp.get("file_path", "")))
        content = inp.get("content", "")
        if truncate:
            args.append(ToolArg("content", f"({len(content)} chars) {_truncate(content, 200)}"))
        else:
            args.append(ToolArg("content", content))
    elif name == "Edit":
        args.append(ToolArg("file_path", inp.get("file_path", "")))
        args.append(ToolArg("old_string", repr(_t(inp.get("old_string", ""), 100))))
        args.append(ToolArg("new_string", repr(_t(inp.get("new_string", ""), 100))))
        if inp.get("replace_all"):
            args.append(ToolArg("replace_all", "true"))
    elif name == "Glob":
        args.append(ToolArg("pattern", inp.get("pattern", "")))
        if inp.get("path"):
            args.append(ToolArg("path", inp["path"]))
    elif name == "Grep":
        args.append(ToolArg("pattern", inp.get("pattern", "")))
        if inp.get("path"):
            args.append(ToolArg("path", inp["path"]))
        if inp.get("glob"):
            args.append(ToolArg("glob", inp["glob"]))
    elif name == "Task":
        args.append(ToolArg("description", inp.get("description", "")))
        prompt = inp.get("prompt", "")
        args.append(ToolArg("prompt", _t(prompt, 300)))
    else:
        for k, v in inp.items():
            args.append(ToolArg(k, _t(str(v), 200)))

    return args


def fmt_tool_result(content: str, truncate: bool = False) -> str:
    """Format tool result content."""
    if truncate and len(content) > 500:
        return content[:500] + f"... ({len(content)} chars total)"
    return content


# =============================================================================
# PARSING: JSONL -> TURNS
# =============================================================================

def parse_entry(entry: Entry, truncate: bool = False) -> Turn | None:
    """Parse an Entry into a Turn. Returns None if nothing renderable."""
    ts = entry.timestamp

    # Handle non-message entry types
    if entry.type == "file-history-snapshot" and entry.snapshot is not None:
        files = []
        for filename, info in entry.snapshot.items():
            files.append({
                "name": filename,
                "version": info.get("version", ""),
                "backupTime": info.get("backupTime", ""),
                "backupFileName": info.get("backupFileName"),
            })
        block = Block(kind="snapshot", content="", snapshot_files=tuple(files))
        return Turn(timestamp=ts, speaker="SNAPSHOT", blocks=(block,), session_id=entry.session_id, uuid=entry.uuid)

    if entry.type == "summary" and entry.summary_text:
        block = Block(kind="summary", content=entry.summary_text)
        return Turn(timestamp=ts, speaker="SUMMARY", blocks=(block,), session_id=entry.session_id, uuid=entry.uuid)

    if entry.type == "queue-operation":
        op = entry.operation or ""
        content = entry.queue_content or ""
        block = Block(kind="task", content=f"{op}\n{content}" if content else op)
        return Turn(timestamp=ts, speaker="TASK", blocks=(block,), session_id=entry.session_id, uuid=entry.uuid)

    if entry.type == "system":
        parts = []
        if entry.subtype:
            parts.append(entry.subtype)
        if entry.duration_ms is not None:
            parts.append(f"{entry.duration_ms}ms")
        if parts:
            block = Block(kind="system", content=": ".join(parts))
            return Turn(timestamp=ts, speaker="SYSTEM", blocks=(block,), session_id=entry.session_id, uuid=entry.uuid)
        return None

    if not entry.message:
        return None

    role = entry.message.role

    has_text = any(
        isinstance(b, TextBlock) and b.text.strip()
        for b in entry.message.content
    )
    has_tool_result = any(
        isinstance(b, ToolResultBlock)
        for b in entry.message.content
    )

    if role == "user" and has_tool_result and not has_text:
        speaker = "TOOL"
    elif role == "user":
        speaker = "USER"
    else:
        speaker = "CLAUDE"

    blocks: list[Block] = []

    for block in entry.message.content:
        if isinstance(block, TextBlock):
            t = block.text.strip()
            if t:
                blocks.append(Block(kind="text", content=t))

        elif isinstance(block, ThinkingBlock):
            t = block.thinking.strip()
            if t:
                blocks.append(Block(kind="thinking", content=t))

        elif isinstance(block, ImageBlock):
            media = block.source.get("media_type", "image")
            blocks.append(Block(kind="image", content=media))

        elif isinstance(block, ToolUseBlock):
            tool_args = fmt_tool_input(block.name, block.input, truncate=truncate)
            blocks.append(Block(
                kind="tool_use",
                content="",
                tool_name=block.name,
                tool_args=tuple(tool_args),
                tool_use_id=block.id,
            ))

        elif isinstance(block, ToolResultBlock):
            kind = "tool_result_error" if block.is_error else "tool_result"
            formatted = fmt_tool_result(block.content, truncate=truncate)
            blocks.append(Block(kind=kind, content=formatted, tool_use_id=block.tool_use_id))

    if not blocks:
        return None

    # Populate metadata on Turn for user/assistant entries
    model = ""
    stop_reason = ""
    cwd = entry.cwd or ""
    if entry.message.model:
        model = entry.message.model
    if entry.message.stop_reason:
        stop_reason = entry.message.stop_reason

    return Turn(
        timestamp=ts, speaker=speaker, blocks=tuple(blocks),
        session_id=entry.session_id, uuid=entry.uuid,
        model=model, stop_reason=stop_reason, cwd=cwd,
    )


def parse_session_turns(jsonl_path: str, truncate: bool = False) -> list[Turn]:
    """Parse a JSONL session file into a list of Turns."""
    session_id_fallback = Path(jsonl_path).stem
    turns: list[Turn] = []

    for entry in parse_entries(jsonl_path):
        if not entry.session_id:
            entry = replace(entry, session_id=session_id_fallback)
        turn = parse_entry(entry, truncate=truncate)
        if turn:
            turns.append(turn)

    return turns


# =============================================================================
# TEXT RENDERER
# =============================================================================

def fmt_date(dt: datetime) -> str:
    """Format datetime as 'Feb 4, 2026 4:16 PM' (no seconds, no timezone)."""
    local = dt.astimezone(_local_tz())
    month = local.strftime("%b")
    day = local.day
    year = local.year
    hour = local.hour % 12 or 12
    minute = local.strftime("%M")
    ampm = local.strftime("%p")
    return f"{month} {day}, {year} {hour}:{minute} {ampm}"


def _fmt_timestamp_full(dt: datetime | None) -> str:
    """Format datetime as 'Feb 4, 2026 4:16:30 PM EST'."""
    if dt is None:
        return ""
    local = dt.astimezone(_local_tz())
    second = local.strftime("%S")
    tz = local.strftime("%Z")
    base = fmt_date(dt)
    # Insert seconds after minute: "... 4:16 PM" -> "... 4:16:30 PM EST"
    return base.replace(f" {local.strftime('%p')}", f":{second} {local.strftime('%p')} {tz}")


_TEXT_WIDTH = 80
_TEXT_RULE = "_" * _TEXT_WIDTH


def render_text(turns: list[Turn]) -> str:
    """Render turns as plain text."""
    # Build tool_use_id -> tool_name map
    tool_use_to_name: dict[str, str] = {}
    for turn in turns:
        for b in turn.blocks:
            if b.tool_use_id and b.kind == "tool_use":
                tool_use_to_name[b.tool_use_id] = b.tool_name

    lines: list[str] = []

    for turn in turns:
        # Header line 1: SPEAKER · timestamp              ids
        ts = _fmt_timestamp_full(turn.timestamp)
        left = f"{turn.speaker} · {ts}" if ts else turn.speaker
        ids = ""
        if turn.session_id:
            ids = turn.session_id[:8]
            if turn.uuid:
                ids += f" · {turn.uuid[:8]}"
        lines.append(_TEXT_RULE)
        if ids:
            pad = max(_TEXT_WIDTH - len(left) - len(ids), 1)
            lines.append(f"{left}{' ' * pad}{ids}")
        else:
            lines.append(left)

        # Subheader: model/stop_reason or task operation (left), cwd (right)
        sub_left = ""
        meta_parts = []
        if turn.model:
            meta_parts.append(turn.model)
        if turn.stop_reason:
            meta_parts.append(turn.stop_reason)
        if meta_parts:
            sub_left = " · ".join(meta_parts)
        if turn.speaker == "TASK" and turn.blocks:
            task_lines = turn.blocks[0].content.split("\n", 1)
            sub_left = task_lines[0]

        if sub_left and turn.cwd:
            pad = max(_TEXT_WIDTH - len(sub_left) - len(turn.cwd), 1)
            lines.append(f"{sub_left}{' ' * pad}{turn.cwd}")
        elif sub_left:
            lines.append(sub_left)
        elif turn.cwd:
            lines.append(f"{turn.cwd:>{_TEXT_WIDTH}}")

        lines.append("")

        for block in turn.blocks:
            if block.kind == "text":
                lines.append(block.content)
                lines.append("")

            elif block.kind == "thinking":
                lines.append(f"[thinking] {block.content}")
                lines.append("")

            elif block.kind == "image":
                lines.append(f"[image: {block.content}]")
                lines.append("")

            elif block.kind == "tool_use":
                tool_id = f" · {block.tool_use_id}" if block.tool_use_id else ""
                lines.append(f"{block.tool_name}{tool_id}")
                lines.append("")
                for arg in block.tool_args:
                    lines.append(f"{arg.name}:")
                    for val_line in arg.value.splitlines():
                        lines.append(f" | {val_line}")
                    lines.append("")

            elif block.kind in ("tool_result", "tool_result_error"):
                tool_name = tool_use_to_name.get(block.tool_use_id, "")
                tool_id = f" · {block.tool_use_id}" if block.tool_use_id else ""
                if not block.content.strip():
                    if tool_name or tool_id:
                        lines.append(f"{tool_name}{tool_id}")
                    lines.append("(empty response)")
                    lines.append("")
                else:
                    if tool_name or tool_id:
                        lines.append(f"{tool_name}{tool_id}")
                        lines.append("")
                    if block.kind == "tool_result_error":
                        lines.append(f"[error] {block.content}")
                    else:
                        lines.append(block.content)
                    lines.append("")

            elif block.kind == "system":
                lines.append(block.content)
                lines.append("")

            elif block.kind == "snapshot":
                for sf in block.snapshot_files:
                    parts = [sf["name"]]
                    if sf.get("version"):
                        parts.append(f"v{sf['version']}")
                    if sf.get("backupTime"):
                        parts.append(str(sf["backupTime"]))
                    if sf.get("backupFileName"):
                        parts.append(str(sf["backupFileName"]))
                    lines.append(" · ".join(parts))
                lines.append("")

            elif block.kind == "summary":
                lines.append(block.content)
                lines.append("")

            elif block.kind == "task":
                # operation is already in subheader, just show content
                task_lines = block.content.split("\n", 1)
                if len(task_lines) > 1 and task_lines[1].strip():
                    lines.append(task_lines[1])
                    lines.append("")

    return "\n".join(lines)


# =============================================================================
# HTML RENDERER
# =============================================================================

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  --bg: #1a1a2e;
  --surface: #16213e;
  --surface-alt: #0f3460;
  --text: #e0e0e0;
  --text-dim: #8899aa;
  --user: #ffffff;
  --claude: #de7356;
  --tool: #56b6c2;
  --system: #5a5a7a;
  --snapshot: #b668cd;
  --summary: #d4a656;
  --task: #56b87a;
  --thinking: #666;
  --border: #2a2a4a;
  --code-bg: #0d1117;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  padding: 0;
}}
.container {{
  max-width: 900px;
  margin: 0 auto;
  padding: 20px;
}}
h1 {{
  font-size: 1.2em;
  color: var(--text-dim);
  padding: 16px 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 20px;
  font-weight: 400;
}}
.turn {{
  margin-bottom: 12px;
  padding: 12px 16px;
  border-radius: 6px;
  border-left: 3px solid transparent;
}}
.turn-user {{
  background: rgba(255, 255, 255, 0.06);
  border-left-color: var(--user);
}}
.turn-claude {{
  background: rgba(222, 115, 86, 0.06);
  border-left-color: var(--claude);
}}
.turn-tool {{
  background: rgba(86, 182, 194, 0.06);
  border-left-color: var(--tool);
}}
.turn-system {{
  background: rgba(90, 90, 122, 0.04);
  border-left-color: var(--system);
}}
.turn-snapshot {{
  background: rgba(182, 104, 205, 0.04);
  border-left-color: var(--snapshot);
}}
.turn-summary {{
  background: rgba(212, 166, 86, 0.04);
  border-left-color: var(--summary);
}}
.turn-task {{
  background: rgba(86, 184, 122, 0.04);
  border-left-color: var(--task);
}}
.turn-header {{
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 6px;
}}
.turn-header-left {{
  display: flex;
  flex-direction: column;
}}
.turn-header-row {{
  display: flex;
  align-items: baseline;
  gap: 10px;
}}
.turn-meta {{
  font-size: 0.65em;
  color: #556;
  font-family: "SF Mono", "Fira Code", monospace;
  text-align: right;
  display: flex;
  flex-direction: column;
  align-items: flex-end;
}}
.speaker {{
  font-weight: 700;
  font-size: 0.8em;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}}
.speaker-model {{
  font-size: 0.6em;
  color: #556;
  font-family: "SF Mono", "Fira Code", monospace;
  margin-bottom: 4px;
}}
.speaker-user {{ color: var(--user); }}
.speaker-claude {{ color: var(--claude); }}
.speaker-tool {{ color: var(--tool); }}
.speaker-system {{ color: var(--system); }}
.speaker-snapshot {{ color: var(--snapshot); }}
.speaker-summary {{ color: var(--summary); }}
.speaker-task {{ color: var(--task); }}
.timestamp {{
  font-size: 0.75em;
  color: var(--text-dim);
  font-family: "SF Mono", "Fira Code", monospace;
}}
.block {{ margin-bottom: 8px; }}
.block:last-child {{ margin-bottom: 0; }}
.block-text {{
  white-space: pre-wrap;
  word-wrap: break-word;
}}
.block-thinking {{
  border-radius: 4px;
  padding: 4px 0;
  font-size: 0.8em;
  color: #777;
  white-space: pre-wrap;
  word-wrap: break-word;
}}
.block-tool-use {{
  font-family: "SF Mono", "Fira Code", monospace;
  font-size: 0.85em;
}}
.tool-header {{
  display: flex;
  align-items: baseline;
  gap: 8px;
  margin-bottom: 6px;
}}
.tool-name {{
  color: #79c0ff;
  font-weight: 600;
}}
.tool-use-id {{
  color: #556;
  font-weight: 400;
  font-size: 0.8em;
}}
.tool-link {{
  color: #556;
  text-decoration: none;
}}
.tool-link:hover {{
  text-decoration: underline;
}}
.tool-arg {{
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: 4px;
  margin-bottom: 4px;
  overflow: hidden;
}}
.tool-arg:last-child {{
  margin-bottom: 0;
}}
.tool-arg-name {{
  background: rgba(255, 255, 255, 0.05);
  color: #999;
  font-size: 0.75em;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  padding: 3px 10px;
  border-bottom: 1px solid var(--border);
}}
.tool-arg-value {{
  padding: 6px 10px;
  white-space: pre-wrap;
  word-wrap: break-word;
  color: #c9d1d9;
}}
.block-result {{
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 8px 12px;
  font-family: "SF Mono", "Fira Code", monospace;
  font-size: 0.82em;
  white-space: pre-wrap;
  word-wrap: break-word;
  color: #8b949e;
  max-height: 300px;
  overflow-y: auto;
}}
.block-result-error {{
  border-color: rgba(233, 69, 96, 0.4);
  color: #f85149;
}}
.block-result-error::before {{
  content: "error";
  display: block;
  font-size: 0.7em;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--user);
  margin-bottom: 4px;
}}
.block-image {{
  color: var(--text-dim);
  font-style: italic;
}}
.block-system {{
  font-family: "SF Mono", "Fira Code", monospace;
  font-size: 0.82em;
  color: var(--text-dim);
}}
.block-snapshot {{
  font-family: "SF Mono", "Fira Code", monospace;
  font-size: 0.82em;
  color: var(--text-dim);
}}
.snapshot-file {{
  display: flex;
  gap: 16px;
  padding: 2px 0;
}}
.snapshot-file-name {{
  color: #c9d1d9;
}}
.snapshot-file-meta {{
  color: #666;
  font-size: 0.9em;
}}
.block-summary {{
  color: var(--text-dim);
  font-style: italic;
}}
.block-task {{
  font-family: "SF Mono", "Fira Code", monospace;
  font-size: 0.82em;
}}
.task-operation {{
  color: #79c0ff;
  font-weight: 600;
  margin-bottom: 4px;
}}
.task-content {{
  color: var(--text-dim);
  white-space: pre-wrap;
  word-wrap: break-word;
}}
</style>
</head>
<body>
<div class="container">
<h1>{title}</h1>
{body}
</div>
</body>
</html>
"""


def render_html(turns: list[Turn], title: str = "Conversation") -> str:
    """Render turns as an HTML document."""
    # Build tool_use_id -> uuid/name maps for cross-linking
    tool_use_to_uuid: dict[str, str] = {}  # tool_use_id -> uuid of invoking turn
    tool_use_to_name: dict[str, str] = {}  # tool_use_id -> tool name
    tool_result_to_uuid: dict[str, str] = {}  # tool_use_id -> uuid of result turn
    for turn in turns:
        for b in turn.blocks:
            if b.tool_use_id and b.kind == "tool_use":
                tool_use_to_uuid[b.tool_use_id] = turn.uuid
                tool_use_to_name[b.tool_use_id] = b.tool_name
            elif b.tool_use_id and b.kind in ("tool_result", "tool_result_error"):
                tool_result_to_uuid[b.tool_use_id] = turn.uuid

    parts: list[str] = []

    for turn in turns:
        speaker_lower = turn.speaker.lower()
        turn_id = f' id="{escape(turn.uuid)}"' if turn.uuid else ""
        parts.append(f'<div class="turn turn-{speaker_lower}"{turn_id}>')
        parts.append(f'<div class="turn-header">')
        parts.append(f'<div class="turn-header-left">')
        parts.append(f'<div class="turn-header-row">')
        parts.append(f'<span class="speaker speaker-{speaker_lower}">{escape(turn.speaker)}</span>')
        ts_str = _fmt_timestamp_full(turn.timestamp)
        if ts_str:
            parts.append(f'<span class="timestamp">{escape(ts_str)}</span>')
        parts.append('</div>')
        if turn.model:
            parts.append(f'<div class="speaker-model">{escape(turn.model)}</div>')
        parts.append('</div>')
        # Right-aligned metadata (stop_reason, session_id, cwd)
        meta_items = []
        if turn.stop_reason:
            meta_items.append(f'<span>{escape(turn.stop_reason)}</span>')
        if turn.session_id:
            sid = escape(turn.session_id[:8])
            if turn.uuid:
                sid += f" &middot; {escape(turn.uuid[:8])}"
            meta_items.append(f'<span>{sid}</span>')
        if turn.cwd:
            meta_items.append(f'<span>{escape(turn.cwd)}</span>')
        if meta_items:
            parts.append(f'<div class="turn-meta">{"".join(meta_items)}</div>')
        parts.append('</div>')

        for block in turn.blocks:
            if block.kind == "text":
                parts.append(f'<div class="block block-text">{escape(block.content)}</div>')

            elif block.kind == "thinking":
                parts.append(f'<div class="block block-thinking">{escape(block.content)}</div>')

            elif block.kind == "image":
                parts.append(f'<div class="block block-image">[Image: {escape(block.content)}]</div>')

            elif block.kind == "tool_use":
                parts.append('<div class="block block-tool-use">')
                parts.append(f'<div class="tool-header"><span class="tool-name">{escape(block.tool_name)}</span>')
                if block.tool_use_id:
                    parts.append(f'<span class="tool-use-id">{escape(block.tool_use_id)}</span>')
                    if block.tool_use_id in tool_result_to_uuid:
                        target = escape(tool_result_to_uuid[block.tool_use_id])
                        parts.append(f'<span class="tool-use-id">&middot; <a class="tool-link" href="#{target}">response &#x2193;</a></span>')
                parts.append('</div>')
                for arg in block.tool_args:
                    parts.append('<div class="tool-arg">')
                    parts.append(f'<div class="tool-arg-name">{escape(arg.name)}</div>')
                    parts.append(f'<div class="tool-arg-value">{escape(arg.value)}</div>')
                    parts.append('</div>')
                parts.append('</div>')

            elif block.kind in ("tool_result", "tool_result_error"):
                parts.append('<div class="block block-tool-use">')
                tool_name = tool_use_to_name.get(block.tool_use_id, "")
                parts.append('<div class="tool-header">')
                if tool_name:
                    parts.append(f'<span class="tool-name">{escape(tool_name)}</span>')
                if block.tool_use_id:
                    parts.append(f'<span class="tool-use-id">{escape(block.tool_use_id)}</span>')
                    if block.tool_use_id in tool_use_to_uuid:
                        target = escape(tool_use_to_uuid[block.tool_use_id])
                        parts.append(f'<span class="tool-use-id">&middot; <a class="tool-link" href="#{target}">invocation &#x2191;</a></span>')
                parts.append('</div>')
                if not block.content.strip():
                    parts.append('<div class="block block-image"><em>empty response</em></div>')
                elif block.kind == "tool_result_error":
                    parts.append(f'<div class="block block-result block-result-error">{escape(block.content)}</div>')
                else:
                    parts.append(f'<div class="block block-result">{escape(block.content)}</div>')
                parts.append('</div>')

            elif block.kind == "system":
                parts.append(f'<div class="block block-system">{escape(block.content)}</div>')

            elif block.kind == "snapshot":
                parts.append('<div class="block block-snapshot">')
                for sf in block.snapshot_files:
                    name = escape(sf.get("name", ""))
                    version = sf.get("version", "")
                    backup_time = sf.get("backupTime", "")
                    backup_file = sf.get("backupFileName")
                    meta = f"v{version}" if version else ""
                    if backup_time:
                        meta += f" &middot; {escape(str(backup_time))}"
                    if backup_file:
                        meta += f" &middot; {escape(str(backup_file))}"
                    parts.append(f'<div class="snapshot-file"><span class="snapshot-file-name">{name}</span><span class="snapshot-file-meta">{meta}</span></div>')
                parts.append('</div>')

            elif block.kind == "summary":
                parts.append(f'<div class="block block-summary">{escape(block.content)}</div>')

            elif block.kind == "task":
                # content is "operation\ncontent" or just "operation"
                task_lines = block.content.split("\n", 1)
                op = escape(task_lines[0])
                content = escape(task_lines[1]) if len(task_lines) > 1 else ""
                parts.append('<div class="block block-task">')
                parts.append(f'<div class="task-operation">{op}</div>')
                if content:
                    parts.append(f'<div class="task-content">{content}</div>')
                parts.append('</div>')

        parts.append('</div>')

    body = "\n".join(parts)
    return HTML_TEMPLATE.format(title=escape(title), body=body)


# =============================================================================
# PUBLIC API
# =============================================================================

def render_session(jsonl_path: str, output_path: str | None = None, truncate: bool = False) -> str:
    """Render a full session JSONL as a readable conversation.

    Output format is detected from the file extension (.html for HTML, anything else for text).
    If no output_path, returns plain text.
    """
    turns = parse_session_turns(jsonl_path, truncate=truncate)

    if output_path and output_path.endswith(".html"):
        title = Path(jsonl_path).stem
        text = render_html(turns, title=title)
    else:
        text = render_text(turns)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(text)

    return text


def render_sessions(jsonl_paths: list[str], output_path: str, title: str = "Conversation", truncate: bool = False) -> str:
    """Render multiple session JSONLs interleaved by timestamp.

    All turns are merged and sorted chronologically into a single output file.
    """
    all_turns: list[Turn] = []
    for jsonl_path in jsonl_paths:
        all_turns.extend(parse_session_turns(jsonl_path, truncate=truncate))

    _epoch = datetime.min.replace(tzinfo=timezone.utc)
    all_turns.sort(key=lambda t: t.timestamp or _epoch)

    if output_path.endswith(".html"):
        text = render_html(all_turns, title=title)
    else:
        text = render_text(all_turns)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(text)

    return text
