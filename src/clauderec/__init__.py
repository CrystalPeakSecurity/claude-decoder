"""clauderec - Extract and recover file operations from Claude Code session logs.

Library usage:
    from clauderec import parse_session, Entry, Read, Write, Edit
    from clauderec import extract_project, reconstruct_file, render_session
"""

from .models import (
    # Content blocks
    TextBlock,
    ThinkingBlock,
    ImageBlock,
    ToolUseBlock,
    ToolResultBlock,
    ContentBlock,
    # Messages and entries
    TokenUsage,
    Message,
    Entry,
    # File operations
    Read,
    Write,
    Edit,
    MultiEdit,
    BashCommand,
    NotebookEdit,
    FileOperation,
    # Core functions
    parse_content_block,
    make_file_operation,
    parse_session,
    get_file_history,
)

from .extract import extract_project, ExtractionStats
from .reconstruct import (
    reconstruct_file,
    FileReconstruction,
    plan_project_reconstruction,
    write_reconstruction,
    ProjectReconstruction,
    strip_line_prefixes,
    strip_system_reminders,
)
from .conversation import render_session

__all__ = [
    # Content blocks
    "TextBlock", "ThinkingBlock", "ImageBlock", "ToolUseBlock", "ToolResultBlock",
    "ContentBlock",
    # Messages and entries
    "TokenUsage", "Message", "Entry",
    # File operations
    "Read", "Write", "Edit", "MultiEdit", "BashCommand", "NotebookEdit", "FileOperation",
    # Core functions
    "parse_content_block", "make_file_operation", "parse_session", "get_file_history",
    # Extract/reconstruct
    "extract_project", "ExtractionStats",
    "reconstruct_file", "FileReconstruction",
    "plan_project_reconstruction", "write_reconstruction", "ProjectReconstruction",
    "strip_line_prefixes", "strip_system_reminders",
    # Conversation
    "render_session",
]
