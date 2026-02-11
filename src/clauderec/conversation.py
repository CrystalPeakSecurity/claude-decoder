#!/usr/bin/env python3
"""Render a Claude Code session JSONL as a readable conversation transcript.

Supports plain text (.txt) and HTML (.html) output formats.
"""

import json
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

from datetime import datetime

from .models import (
    Entry,
    TextBlock,
    ThinkingBlock,
    ImageBlock,
    ToolUseBlock,
    ToolResultBlock,
)


# =============================================================================
# INTERMEDIATE REPRESENTATION
# =============================================================================

@dataclass
class ToolArg:
    """A single named argument to a tool invocation."""
    name: str
    value: str


@dataclass
class Block:
    """A single renderable block within a conversation turn."""
    kind: str  # "text", "thinking", "tool_use", "tool_result", "tool_result_error", "image"
    content: str  # Primary text content
    tool_name: str = ""  # For tool_use blocks
    tool_args: list[ToolArg] = field(default_factory=list)  # Structured args for tool_use


@dataclass
class Turn:
    """A conversation turn: one or more blocks from the same speaker at the same time."""
    timestamp: datetime
    speaker: str  # "USER", "CLAUDE", "TOOL"
    blocks: list[Block] = field(default_factory=list)
    session_id: str = ""


# =============================================================================
# FORMATTING HELPERS
# =============================================================================

def _truncate(s: str, max_len: int) -> str:
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def fmt_tool_input(name: str, inp: dict) -> list[ToolArg]:
    """Format tool input as structured args."""
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
        args.append(ToolArg("content", f"({len(content)} chars) {_truncate(content, 200)}"))
    elif name == "Edit":
        args.append(ToolArg("file_path", inp.get("file_path", "")))
        args.append(ToolArg("old_string", repr(_truncate(inp.get("old_string", ""), 100))))
        args.append(ToolArg("new_string", repr(_truncate(inp.get("new_string", ""), 100))))
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
        args.append(ToolArg("prompt", _truncate(prompt, 300)))
    else:
        for k, v in inp.items():
            args.append(ToolArg(k, _truncate(str(v), 200)))

    return args


def fmt_tool_result(content: str | list, max_len: int = 500) -> str:
    """Format tool result content."""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        text = "\n".join(parts)
    elif isinstance(content, str):
        text = content
    else:
        text = str(content)
    if len(text) > max_len:
        text = text[:max_len] + f"... ({len(text)} chars total)"
    return text


# =============================================================================
# PARSING: JSONL -> TURNS
# =============================================================================

def parse_entry(entry: Entry) -> Turn | None:
    """Parse an Entry into a Turn. Returns None if nothing renderable."""
    if not entry.message:
        return None

    role = entry.message.role
    ts = entry.timestamp

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
            tool_args = fmt_tool_input(block.name, block.input)
            blocks.append(Block(
                kind="tool_use",
                content="",
                tool_name=block.name,
                tool_args=tool_args,
            ))

        elif isinstance(block, ToolResultBlock):
            kind = "tool_result_error" if block.is_error else "tool_result"
            formatted = fmt_tool_result(block.content)
            blocks.append(Block(kind=kind, content=formatted))

    if not blocks:
        return None

    return Turn(timestamp=ts, speaker=speaker, blocks=blocks, session_id=entry.session_id)


def parse_session(jsonl_path: str) -> list[Turn]:
    """Parse a JSONL session file into a list of Turns."""
    turns: list[Turn] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if data.get("type") == "progress":
                continue

            entry = Entry.from_dict(data)
            turn = parse_entry(entry)
            if turn:
                turns.append(turn)

    return turns


# =============================================================================
# TEXT RENDERER
# =============================================================================

_LOCAL_TZ = datetime.now().astimezone().tzinfo


def _fmt_timestamp_full(dt: datetime) -> str:
    """Format datetime as 'Feb 4, 2026 4:16:30 PM EST'."""
    local = dt.astimezone(_LOCAL_TZ)
    return local.strftime("%-b %-d, %Y %-I:%M:%S %p %Z")


def render_text(turns: list[Turn]) -> str:
    """Render turns as plain text."""
    lines: list[str] = []

    for turn in turns:
        for block in turn.blocks:
            lines.append(f"=== [{_fmt_timestamp_full(turn.timestamp)}] {turn.speaker} ===")

            if block.kind == "text":
                lines.append(block.content)
            elif block.kind == "thinking":
                lines.append(f"[THINKING] {block.content}")
            elif block.kind == "image":
                lines.append(f"[IMAGE: {block.content}]")
            elif block.kind == "tool_use":
                lines.append(f"[TOOL: {block.tool_name}]")
                for arg in block.tool_args:
                    lines.append(f"  {arg.name}: {arg.value}")
            elif block.kind == "tool_result":
                lines.append(f"[RESULT] {block.content}")
            elif block.kind == "tool_result_error":
                lines.append(f"[RESULT ERROR] {block.content}")

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
  --tool: #5a5a7a;
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
  background: rgba(90, 90, 122, 0.06);
  border-left-color: var(--tool);
}}
.turn-header {{
  display: flex;
  align-items: baseline;
  gap: 10px;
  margin-bottom: 6px;
}}
.session-id {{
  margin-left: auto;
  font-size: 0.65em;
  color: #555;
  font-family: "SF Mono", "Fira Code", monospace;
}}
.speaker {{
  font-weight: 700;
  font-size: 0.8em;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}}
.speaker-user {{ color: var(--user); }}
.speaker-claude {{ color: var(--claude); }}
.speaker-tool {{ color: var(--tool); }}
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
.tool-name {{
  color: #79c0ff;
  font-weight: 600;
  margin-bottom: 6px;
  display: block;
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
    parts: list[str] = []

    for turn in turns:
        speaker_lower = turn.speaker.lower()
        parts.append(f'<div class="turn turn-{speaker_lower}">')
        parts.append(f'<div class="turn-header">')
        parts.append(f'<span class="speaker speaker-{speaker_lower}">{escape(turn.speaker)}</span>')
        parts.append(f'<span class="timestamp">{escape(_fmt_timestamp_full(turn.timestamp))}</span>')
        if turn.session_id:
            short_id = turn.session_id[:8]
            parts.append(f'<span class="session-id">{escape(short_id)}</span>')
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
                parts.append(f'<span class="tool-name">{escape(block.tool_name)}</span>')
                for arg in block.tool_args:
                    parts.append('<div class="tool-arg">')
                    parts.append(f'<div class="tool-arg-name">{escape(arg.name)}</div>')
                    parts.append(f'<div class="tool-arg-value">{escape(arg.value)}</div>')
                    parts.append('</div>')
                parts.append('</div>')

            elif block.kind == "tool_result":
                parts.append(f'<div class="block block-result">{escape(block.content)}</div>')

            elif block.kind == "tool_result_error":
                parts.append(f'<div class="block block-result block-result-error">{escape(block.content)}</div>')

        parts.append('</div>')

    body = "\n".join(parts)
    return HTML_TEMPLATE.format(title=escape(title), body=body)


# =============================================================================
# PUBLIC API
# =============================================================================

def render_session(jsonl_path: str, output_path: str | None = None) -> str:
    """Render a full session JSONL as a readable conversation.

    Output format is detected from the file extension (.html for HTML, anything else for text).
    If no output_path, returns plain text.
    """
    turns = parse_session(jsonl_path)

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


def render_sessions(jsonl_paths: list[str], output_path: str, title: str = "Conversation") -> str:
    """Render multiple session JSONLs interleaved by timestamp.

    All turns are merged and sorted chronologically into a single output file.
    """
    all_turns: list[Turn] = []
    for jsonl_path in jsonl_paths:
        all_turns.extend(parse_session(jsonl_path))

    all_turns.sort(key=lambda t: t.timestamp)

    if output_path.endswith(".html"):
        text = render_html(all_turns, title=title)
    else:
        text = render_text(all_turns)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(text)

    return text


