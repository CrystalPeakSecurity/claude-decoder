#!/usr/bin/env python3
"""Reconstruct source files from extracted operations."""

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .models import (
    Read, Write, Edit, MultiEdit,
    FileOperation, parse_session,
)


# Line prefix pattern: spaces + line number + → (unicode arrow)
LINE_PREFIX = re.compile(r'^\s*\d+→', re.MULTILINE)

# System reminder tags injected by Claude Code
SYSTEM_REMINDER = re.compile(r'\n*<system-reminder>.*?</system-reminder>\s*$', re.DOTALL)


def strip_line_prefixes(content: str) -> str:
    """Remove line number prefixes from Read output.

    Format: '     1→content'
    """
    return LINE_PREFIX.sub('', content)


def strip_system_reminders(content: str) -> str:
    """Remove <system-reminder> tags that leak into read content."""
    return SYSTEM_REMINDER.sub('', content)


@dataclass
class FileReconstruction:
    """Result of attempting to reconstruct a file."""
    path: str
    content: str | None
    success: bool
    baseline_type: str | None = None  # 'write' or 'read'
    baseline_timestamp: str | None = None
    edits_applied: int = 0
    error: str | None = None


def reconstruct_file(
    path: str,
    operations: list[FileOperation],
) -> FileReconstruction:
    """Reconstruct a single file from its operations.

    Strategy:
    1. Find latest snapshot (write or full read)
    2. Apply edits after that snapshot chronologically
    """
    # Filter to this file's operations
    file_ops = []
    for op in operations:
        if isinstance(op, (Read, Write, Edit)):
            if op.file_path == path:
                file_ops.append(op)
        elif isinstance(op, MultiEdit):
            if op.file_path == path:
                file_ops.append(op)

    if not file_ops:
        return FileReconstruction(
            path=path,
            content=None,
            success=False,
            error="No operations found",
        )

    # Sort by timestamp
    file_ops.sort(key=lambda op: op.timestamp)

    # Find latest snapshot (write or full read)
    snapshot = None
    snapshot_idx = -1
    snapshot_type = None

    for i, op in enumerate(file_ops):
        if isinstance(op, Write):
            snapshot = op.content
            snapshot_idx = i
            snapshot_type = 'write'
        elif isinstance(op, Read) and op.offset is None and op.limit is None and not op.is_error:
            snapshot = strip_system_reminders(strip_line_prefixes(op.content))
            snapshot_idx = i
            snapshot_type = 'read'

    if snapshot is None:
        return FileReconstruction(
            path=path,
            content=None,
            success=False,
            error="No snapshot (write or full read) found",
        )

    # Apply edits after snapshot
    content = snapshot
    edits_applied = 0

    for op in file_ops[snapshot_idx + 1:]:
        if isinstance(op, Edit):
            if op.old_string in content:
                if op.replace_all:
                    content = content.replace(op.old_string, op.new_string)
                else:
                    content = content.replace(op.old_string, op.new_string, 1)
                edits_applied += 1
            else:
                # Edit doesn't match - might be from a different branch of edits
                # or content changed in unexpected way
                pass
        elif isinstance(op, MultiEdit):
            for old_str, new_str in op.edits:
                if old_str in content:
                    content = content.replace(old_str, new_str, 1)
                    edits_applied += 1
        elif isinstance(op, Write):
            # A later write replaces everything
            content = op.content
            edits_applied = 0  # Reset since write is new baseline

    return FileReconstruction(
        path=path,
        content=content,
        success=True,
        baseline_type=snapshot_type,
        baseline_timestamp=file_ops[snapshot_idx].timestamp.isoformat(),
        edits_applied=edits_applied,
    )


@dataclass
class ProjectReconstruction:
    """Result of reconstructing all files in a project."""
    results: list[tuple[str, FileReconstruction]]  # (rel_path, result)
    total_operations: int
    session_count: int

    @property
    def succeeded(self) -> list[tuple[str, FileReconstruction]]:
        return [(p, r) for p, r in self.results if r.success]

    @property
    def failed(self) -> list[tuple[str, FileReconstruction]]:
        return [(p, r) for p, r in self.results if not r.success]


def plan_project_reconstruction(
    claude_project_dir: Path,
    project_root: str,
) -> ProjectReconstruction:
    """Plan reconstruction of all project files (dry-run).

    Parses all sessions and reconstructs files in memory without writing.

    Args:
        claude_project_dir: Path to ~/.claude/projects/<project>/
        project_root: Filter to files under this path (with trailing /)

    Returns:
        ProjectReconstruction with all results ready to inspect or write.
    """
    jsonl_files = sorted(claude_project_dir.glob('*.jsonl'), key=lambda p: p.stat().st_mtime)

    all_operations: list[FileOperation] = []
    for jsonl_path in jsonl_files:
        operations = parse_session(str(jsonl_path))
        all_operations.extend(operations)

    # Collect unique file paths
    files: set[str] = set()
    for op in all_operations:
        if isinstance(op, (Read, Write, Edit)):
            if op.file_path.startswith(project_root):
                files.add(op.file_path)
        elif isinstance(op, MultiEdit):
            if op.file_path.startswith(project_root):
                files.add(op.file_path)

    # Reconstruct each file in memory
    results: list[tuple[str, FileReconstruction]] = []
    for file_path in sorted(files):
        result = reconstruct_file(file_path, all_operations)
        rel_path = file_path[len(project_root):]
        results.append((rel_path, result))

    return ProjectReconstruction(
        results=results,
        total_operations=len(all_operations),
        session_count=len(jsonl_files),
    )


def write_reconstruction(
    reconstruction: ProjectReconstruction,
    output_dir: Path,
) -> None:
    """Write reconstructed files to disk.

    Args:
        reconstruction: Result from plan_project_reconstruction()
        output_dir: Where to write reconstructed files
    """
    for rel_path, result in reconstruction.succeeded:
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result.content)
