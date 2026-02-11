#!/usr/bin/env python3
"""Extract file operations from Claude session logs."""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .models import (
    Read, Write, Edit, MultiEdit, BashCommand, NotebookEdit,
    FileOperation, parse_session,
)


@dataclass
class ExtractionStats:
    """Track extraction statistics."""
    writes: int = 0
    edits: int = 0
    full_reads: int = 0
    partial_reads: int = 0
    multiedits: int = 0
    notebook_edits: int = 0
    errors: int = 0
    files: set = field(default_factory=set)

    @property
    def total(self) -> int:
        return self.writes + self.edits + self.full_reads + self.partial_reads + self.multiedits + self.notebook_edits

    def print_summary(self) -> None:
        """Print extraction summary."""
        print("\n" + "=" * 50)
        print("EXTRACTION SUMMARY")
        print("=" * 50)
        print(f"Total operations: {self.total}")
        print(f"  Writes:         {self.writes}")
        print(f"  Edits:          {self.edits}")
        print(f"  Full reads:     {self.full_reads}")
        print(f"  Partial reads:  {self.partial_reads}")
        if self.multiedits:
            print(f"  MultiEdits:     {self.multiedits}")
        if self.notebook_edits:
            print(f"  NotebookEdits:  {self.notebook_edits}")
        print(f"Unique files:     {len(self.files)}")
        if self.errors:
            print(f"Errors:           {self.errors}")
        print("=" * 50)


def format_timestamp(dt) -> str:
    """Convert datetime to compact format: YYYYMMDD-HHMMSSmmm"""
    return dt.strftime('%Y%m%d-%H%M%S') + f"{dt.microsecond // 1000:03d}"


def relative_path(file_path: str, project_root: str) -> str:
    """Get path relative to project root."""
    if file_path.startswith(project_root):
        return file_path[len(project_root):]
    return file_path


def save_operation(
    op: FileOperation,
    output_dir: Path,
    project_root: str,
    stats: ExtractionStats,
) -> None:
    """Save a file operation to output directory."""
    # Get file path from operation
    if isinstance(op, (Read, Write, Edit)):
        file_path = op.file_path
    elif isinstance(op, NotebookEdit):
        file_path = op.notebook_path
    elif isinstance(op, MultiEdit):
        file_path = op.file_path
    elif isinstance(op, BashCommand):
        # Skip bash commands - they don't have a single file path
        return
    else:
        return

    # Filter to project files only
    if not file_path.startswith(project_root):
        return

    rel = relative_path(file_path, project_root)
    ts = format_timestamp(op.timestamp)
    stats.files.add(rel)

    # Determine operation type and content
    if isinstance(op, Write):
        op_name = "write"
        content = op.content
        suffix = ""
        stats.writes += 1
    elif isinstance(op, Read):
        if op.is_error:
            stats.errors += 1
            return  # Skip failed reads - no usable content
        op_name = "read"
        content = op.content
        if op.offset is None and op.limit is None:
            suffix = "_full"
            stats.full_reads += 1
        else:
            offset_str = str(op.offset) if op.offset is not None else "0"
            limit_str = str(op.limit) if op.limit is not None else "all"
            suffix = f"_{offset_str}_{limit_str}"
            stats.partial_reads += 1
    elif isinstance(op, Edit):
        op_name = "edit"
        edit_data = {
            'old_string': op.old_string,
            'new_string': op.new_string,
        }
        if op.replace_all:
            edit_data['replace_all'] = True
        content = json.dumps(edit_data, indent=2)
        suffix = ""
        stats.edits += 1
    elif isinstance(op, MultiEdit):
        op_name = "multiedit"
        content = json.dumps({
            'edits': [{'old_string': old, 'new_string': new} for old, new in op.edits]
        }, indent=2)
        suffix = ""
        stats.multiedits += 1
    elif isinstance(op, NotebookEdit):
        op_name = "notebook"
        content = json.dumps({
            'cell_number': op.cell_number,
            'new_source': op.new_source,
        }, indent=2)
        suffix = ""
        stats.notebook_edits += 1
    else:
        return

    output_path = output_dir / f"{rel}.{ts}.{op_name}{suffix}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(content)

    print(f"  {op_name}{suffix}: {rel}")


def extract_project(
    claude_project_dir: Path,
    output_dir: Path,
    project_root: str,
) -> ExtractionStats:
    """Extract all file operations from a Claude project's session logs.

    Args:
        claude_project_dir: Path to ~/.claude/projects/<project>/
        output_dir: Where to save extracted operations
        project_root: Filter to files under this path (with trailing /)

    Returns:
        ExtractionStats with counts and file list
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = ExtractionStats()

    # Get all jsonl files sorted by modification time
    jsonl_files = sorted(claude_project_dir.glob('*.jsonl'), key=lambda p: p.stat().st_mtime)
    print(f"Found {len(jsonl_files)} session files")

    for jsonl_path in jsonl_files:
        print(f"\nProcessing: {jsonl_path.name}")
        operations = parse_session(str(jsonl_path))
        for op in operations:
            save_operation(op, output_dir, project_root, stats)

    stats.print_summary()
    return stats


