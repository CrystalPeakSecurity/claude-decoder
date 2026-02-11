"""
Claude Code JSONL Log Data Models
=================================

Data models for parsing Claude Code session logs (~/.claude/projects/<project>/<session>.jsonl)
to extract file operations for recovery purposes.

JSONL STRUCTURE OVERVIEW
------------------------

Each line in the JSONL is an Entry with a `type` field:

    {"type": "user", "message": {...}, "uuid": "...", "timestamp": "...", ...}
    {"type": "assistant", "message": {...}, "uuid": "...", "timestamp": "...", ...}
    {"type": "system", "subtype": "init", ...}
    {"type": "result", "subtype": "success", ...}
    {"type": "progress", ...}  # SKIP THESE - redundant sub-agent snapshots

The `message` field contains the actual conversation content with `role` and `content`.
The `content` is a list of content blocks (text, tool_use, tool_result, thinking, image).

TOOL USE/RESULT RELATIONSHIP
----------------------------

Tool invocations span two entries:

1. Assistant entry with tool_use block:
   {"type": "assistant", "message": {"role": "assistant", "content": [
       {"type": "tool_use", "id": "toolu_abc123", "name": "Read", "input": {"file_path": "/foo.py"}}
   ]}}

2. User entry with tool_result block:
   {"type": "user", "message": {"role": "user", "content": [
       {"type": "tool_result", "tool_use_id": "toolu_abc123", "content": "file contents here"}
   ]}}

The relationship is 1:1. Join them via tool_use.id == tool_result.tool_use_id.

SUB-AGENTS
----------

Messages from sub-agents (Task tool) have a `parent_tool_use_id` field on the Entry.
These are separate Claude instances with their own tool invocations.
For file recovery, treat them the same as main agent operations.

USAGE
-----

1. Parse: Load JSONL, deserialize each line into Entry
2. Filter: Skip type="progress" entries (bloated, redundant)
3. Extract: Pull tool_use blocks from assistant messages, tool_result from user messages  
4. Join: Match tool_use.id to tool_result.tool_use_id
5. Convert: Transform joined pairs into unified FileOperation objects

Example:

    from models import Entry, Read, Write, Edit
    import json
    
    def parse_session(path: str) -> list[Read | Write | Edit | ...]:
        entries = []
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                if data.get("type") == "progress":
                    continue  # Skip bloated progress entries
                entries.append(Entry.from_dict(data))
        
        # Build tool_use_id -> result content map
        results: dict[str, str] = {}
        for entry in entries:
            if entry.type == "user" and entry.message:
                for block in entry.message.content:
                    if isinstance(block, ToolResultBlock):
                        results[block.tool_use_id] = block.content
        
        # Extract file operations
        operations = []
        for entry in entries:
            if entry.type == "assistant" and entry.message:
                for block in entry.message.content:
                    if isinstance(block, ToolUseBlock):
                        result_content = results.get(block.id, "")
                        op = make_file_operation(block, result_content, entry.timestamp)
                        if op:
                            operations.append(op)
        
        return operations
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# =============================================================================
# CONTENT BLOCKS (inside message.content list)
# =============================================================================

@dataclass
class TextBlock:
    """Plain text content from Claude or user."""
    type: str  # "text"
    text: str
    
    @classmethod
    def from_dict(cls, d: dict) -> "TextBlock":
        return cls(type=d["type"], text=d.get("text", ""))


@dataclass
class ThinkingBlock:
    """Claude's internal reasoning (extended thinking mode)."""
    type: str  # "thinking"
    thinking: str
    
    @classmethod
    def from_dict(cls, d: dict) -> "ThinkingBlock":
        return cls(type=d["type"], thinking=d.get("thinking", ""))


@dataclass
class ImageBlock:
    """Image content (screenshots, pasted images)."""
    type: str  # "image"
    source: dict  # {"type": "base64", "media_type": "image/png", "data": "..."}
    
    @classmethod
    def from_dict(cls, d: dict) -> "ImageBlock":
        return cls(type=d["type"], source=d.get("source", {}))


@dataclass
class ToolUseBlock:
    """Tool invocation by Claude."""
    type: str  # "tool_use"
    id: str  # e.g. "toolu_01ABC..."
    name: str  # e.g. "Read", "Write", "Edit", "Bash"
    input: dict  # Tool-specific input parameters
    
    @classmethod
    def from_dict(cls, d: dict) -> "ToolUseBlock":
        return cls(
            type=d["type"],
            id=d["id"],
            name=d.get("name", ""),
            input=d.get("input", {}),
        )


@dataclass
class ToolResultBlock:
    """Result of a tool execution."""
    type: str  # "tool_result"
    tool_use_id: str  # Links back to ToolUseBlock.id
    content: str | list  # Result content (often string, sometimes list of blocks)
    is_error: bool = False
    
    @classmethod
    def from_dict(cls, d: dict) -> "ToolResultBlock":
        content = d.get("content", "")
        # Normalize list content to string for simplicity
        if isinstance(content, list):
            # Extract text from content blocks
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
            content = "\n".join(parts) if parts else str(d.get("content", ""))
        return cls(
            type=d["type"],
            tool_use_id=d.get("tool_use_id", ""),
            content=content,
            is_error=d.get("is_error", False),
        )


ContentBlock = TextBlock | ThinkingBlock | ImageBlock | ToolUseBlock | ToolResultBlock


def parse_content_block(d: dict) -> ContentBlock:
    """Parse a content block dict into the appropriate type."""
    block_type = d.get("type", "")
    if block_type == "text":
        return TextBlock.from_dict(d)
    elif block_type == "thinking":
        return ThinkingBlock.from_dict(d)
    elif block_type == "image":
        return ImageBlock.from_dict(d)
    elif block_type == "tool_use":
        return ToolUseBlock.from_dict(d)
    elif block_type == "tool_result":
        return ToolResultBlock.from_dict(d)
    else:
        # Unknown block type - return as text with raw content
        return TextBlock(type=block_type, text=str(d))


# =============================================================================
# MESSAGE (the message field of an Entry)
# =============================================================================

@dataclass
class TokenUsage:
    """Token consumption for a message."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    
    @classmethod
    def from_dict(cls, d: dict | None) -> "TokenUsage":
        if not d:
            return cls()
        return cls(
            input_tokens=d.get("input_tokens", 0),
            output_tokens=d.get("output_tokens", 0),
            cache_creation_input_tokens=d.get("cache_creation_input_tokens", 0),
            cache_read_input_tokens=d.get("cache_read_input_tokens", 0),
        )


@dataclass
class Message:
    """A conversation message (user or assistant)."""
    role: str  # "user" or "assistant"
    content: list[ContentBlock]
    model: str | None = None
    stop_reason: str | None = None
    usage: TokenUsage | None = None
    
    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        raw_content = d.get("content", [])
        if isinstance(raw_content, str):
            # Sometimes content is just a string
            content = [TextBlock(type="text", text=raw_content)]
        else:
            content = [parse_content_block(block) for block in raw_content]
        
        usage = None
        if "usage" in d:
            usage = TokenUsage.from_dict(d["usage"])
        
        return cls(
            role=d.get("role", ""),
            content=content,
            model=d.get("model"),
            stop_reason=d.get("stop_reason"),
            usage=usage,
        )


# =============================================================================
# ENTRY (a single line in the JSONL)
# =============================================================================

@dataclass
class Entry:
    """
    A single entry (line) in the Claude Code JSONL log.
    
    Key fields:
    - type: "user", "assistant", "system", "result", or "progress"
    - message: The conversation message (for user/assistant types)
    - uuid: Unique identifier for this entry
    - parent_uuid: Links to parent entry (for conversation threading)
    - parent_tool_use_id: Set when this is from a sub-agent (Task tool)
    - session_id: The session this entry belongs to
    - timestamp: ISO-8601 timestamp
    """
    type: str
    uuid: str
    timestamp: datetime
    session_id: str = ""
    parent_uuid: str | None = None
    parent_tool_use_id: str | None = None  # Present for sub-agent messages
    message: Message | None = None
    cwd: str | None = None
    version: str | None = None
    git_branch: str | None = None
    subtype: str | None = None  # For system/result entries
    
    @classmethod
    def from_dict(cls, d: dict) -> "Entry":
        message = None
        if "message" in d and d["message"]:
            message = Message.from_dict(d["message"])
        
        timestamp_str = d.get("timestamp", "")
        try:
            # Handle ISO format with or without timezone
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            timestamp = datetime.now()
        
        return cls(
            type=d.get("type", ""),
            uuid=d.get("uuid", ""),
            timestamp=timestamp,
            session_id=d.get("session_id", d.get("sessionId", "")),
            parent_uuid=d.get("parent_uuid", d.get("parentUuid")),
            parent_tool_use_id=d.get("parent_tool_use_id", d.get("parentToolUseId")),
            message=message,
            cwd=d.get("cwd"),
            version=d.get("version"),
            git_branch=d.get("git_branch", d.get("gitBranch")),
            subtype=d.get("subtype"),
        )


# =============================================================================
# UNIFIED FILE OPERATIONS (joined tool_use + tool_result)
# =============================================================================

@dataclass
class Read:
    """
    A file read operation.

    Created from: tool_use(name="Read") + tool_result
    - file_path: from tool_use.input.file_path
    - content: from tool_result.content
    - offset: line number to start reading from (1-indexed)
    - limit: number of lines to read
    """
    file_path: str
    content: str
    timestamp: datetime
    tool_use_id: str = ""

    # Metadata for partial reads
    offset: int | None = None  # Line offset (if partial read)
    limit: int | None = None   # Line limit (if partial read)
    is_error: bool = False     # True if the read failed


@dataclass
class Write:
    """
    A file write operation (full file replacement).
    
    Created from: tool_use(name="Write")
    - file_path: from tool_use.input.file_path
    - content: from tool_use.input.content
    
    Note: Content comes from input, not result (result just confirms success).
    """
    file_path: str
    content: str
    timestamp: datetime
    tool_use_id: str = ""


@dataclass
class Edit:
    """
    A single string replacement edit.

    Created from: tool_use(name="Edit")
    - file_path: from tool_use.input.file_path
    - old_string: from tool_use.input.old_string
    - new_string: from tool_use.input.new_string
    - replace_all: if True, replace all occurrences (default False)
    """
    file_path: str
    old_string: str
    new_string: str
    timestamp: datetime
    tool_use_id: str = ""
    replace_all: bool = False


@dataclass
class MultiEdit:
    """
    Multiple edits to a single file in one operation.
    
    Created from: tool_use(name="MultiEdit")
    - file_path: from tool_use.input.file_path
    - edits: list of (old_string, new_string) tuples from tool_use.input.edits
    """
    file_path: str
    edits: list[tuple[str, str]]  # [(old_str, new_str), ...]
    timestamp: datetime
    tool_use_id: str = ""


@dataclass
class BashCommand:
    """
    A bash command execution.
    
    May contain file content if command was cat/head/tail, or file
    modifications if command was rm/mv/cp/etc.
    
    Created from: tool_use(name="Bash") + tool_result
    - command: from tool_use.input.command
    - output: from tool_result.content
    """
    command: str
    output: str
    timestamp: datetime
    tool_use_id: str = ""
    timeout: int | None = None


@dataclass
class NotebookEdit:
    """
    Jupyter notebook cell edit.
    
    Created from: tool_use(name="NotebookEdit")
    """
    notebook_path: str
    cell_number: int
    new_source: str
    timestamp: datetime
    tool_use_id: str = ""


FileOperation = Read | Write | Edit | MultiEdit | BashCommand | NotebookEdit


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def make_file_operation(
    tool_use: ToolUseBlock,
    result_content: str,
    timestamp: datetime,
    is_error: bool = False,
) -> FileOperation | None:
    """
    Create a FileOperation from a tool_use block and its result.

    Args:
        tool_use: The ToolUseBlock from an assistant message
        result_content: The content from the matching tool_result
        timestamp: The timestamp of the entry
        is_error: Whether the tool execution failed

    Returns:
        A FileOperation subclass instance, or None if not a file operation
    """
    name = tool_use.name
    inp = tool_use.input

    if name == "Read":
        return Read(
            file_path=inp.get("file_path", ""),
            content=result_content,
            timestamp=timestamp,
            tool_use_id=tool_use.id,
            offset=inp.get("offset"),
            limit=inp.get("limit"),
            is_error=is_error,
        )
    
    elif name == "Write":
        return Write(
            file_path=inp.get("file_path", ""),
            content=inp.get("content", ""),
            timestamp=timestamp,
            tool_use_id=tool_use.id,
        )
    
    elif name == "Edit":
        return Edit(
            file_path=inp.get("file_path", ""),
            old_string=inp.get("old_string", ""),
            new_string=inp.get("new_string", ""),
            timestamp=timestamp,
            tool_use_id=tool_use.id,
            replace_all=inp.get("replace_all", False),
        )
    
    elif name == "MultiEdit":
        edits = []
        for edit in inp.get("edits", []):
            old = edit.get("old_str", edit.get("old_string", ""))
            new = edit.get("new_str", edit.get("new_string", ""))
            edits.append((old, new))
        return MultiEdit(
            file_path=inp.get("file_path", ""),
            edits=edits,
            timestamp=timestamp,
            tool_use_id=tool_use.id,
        )
    
    elif name == "Bash":
        return BashCommand(
            command=inp.get("command", ""),
            output=result_content,
            timestamp=timestamp,
            tool_use_id=tool_use.id,
            timeout=inp.get("timeout"),
        )
    
    elif name == "NotebookEdit":
        return NotebookEdit(
            notebook_path=inp.get("notebook_path", inp.get("file_path", "")),
            cell_number=inp.get("cell_number", 0),
            new_source=inp.get("new_source", inp.get("source", "")),
            timestamp=timestamp,
            tool_use_id=tool_use.id,
        )
    
    # Not a file operation we care about
    return None


# =============================================================================
# CONVENIENCE PARSER
# =============================================================================

def parse_session(jsonl_path: str) -> list[FileOperation]:
    """
    Parse a Claude Code session JSONL file and extract all file operations.
    
    Args:
        jsonl_path: Path to the session .jsonl file
        
    Returns:
        List of FileOperation instances in chronological order
    """
    import json
    
    entries: list[Entry] = []
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # Skip progress entries - they're bloated sub-agent snapshots
            if data.get("type") == "progress":
                continue
            
            entries.append(Entry.from_dict(data))
    
    # Build tool_use_id -> (result_content, is_error) mapping
    results: dict[str, tuple[str, bool]] = {}
    for entry in entries:
        if entry.type == "user" and entry.message:
            for block in entry.message.content:
                if isinstance(block, ToolResultBlock):
                    content = block.content if isinstance(block.content, str) else str(block.content)
                    results[block.tool_use_id] = (content, block.is_error)

    # Extract file operations from assistant messages
    operations: list[FileOperation] = []
    for entry in entries:
        if entry.type == "assistant" and entry.message:
            for block in entry.message.content:
                if isinstance(block, ToolUseBlock):
                    result_content, is_error = results.get(block.id, ("", False))
                    op = make_file_operation(block, result_content, entry.timestamp, is_error)
                    if op is not None:
                        operations.append(op)
    
    # Sort by timestamp
    operations.sort(key=lambda op: op.timestamp)
    
    return operations


def get_file_history(operations: list[FileOperation], file_path: str) -> list[FileOperation]:
    """
    Filter operations for a specific file path.
    
    Args:
        operations: List of all file operations
        file_path: The file path to filter for
        
    Returns:
        Operations affecting the specified file, in chronological order
    """
    result = []
    for op in operations:
        if isinstance(op, (Read, Write, Edit, NotebookEdit)):
            if op.file_path == file_path:
                result.append(op)
        elif isinstance(op, MultiEdit):
            if op.file_path == file_path:
                result.append(op)
        elif isinstance(op, BashCommand):
            # Check if command references the file
            if file_path in op.command:
                result.append(op)
    return result


def reconstruct_file(operations: list[FileOperation], file_path: str) -> str | None:
    """
    Reconstruct a file's content by replaying operations.
    
    Finds the last Write or Read (as baseline), then applies subsequent Edits.
    
    Args:
        operations: List of file operations (should be pre-filtered and sorted)
        file_path: The file to reconstruct
        
    Returns:
        Reconstructed file content, or None if no baseline found
    """
    # Get operations for this file
    file_ops = get_file_history(operations, file_path)
    if not file_ops:
        return None
    
    # Find baseline (last Write, or last Read if no Write)
    content: str | None = None
    baseline_idx = -1
    
    for i, op in enumerate(file_ops):
        if isinstance(op, Write):
            content = op.content
            baseline_idx = i
        elif isinstance(op, Read) and content is None:
            content = op.content
            baseline_idx = i
    
    if content is None:
        return None
    
    # Apply edits after baseline
    for op in file_ops[baseline_idx + 1:]:
        if isinstance(op, Edit):
            if op.old_string in content:
                content = content.replace(op.old_string, op.new_string, 1)
        elif isinstance(op, MultiEdit):
            for old_str, new_str in op.edits:
                if old_str in content:
                    content = content.replace(old_str, new_str, 1)
        elif isinstance(op, Write):
            # A later Write replaces everything
            content = op.content
    
    return content
