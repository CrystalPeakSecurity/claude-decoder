#!/usr/bin/env python3
"""Reconstruct source files from extracted operations."""

import difflib
import re
from dataclasses import dataclass, replace
from pathlib import Path

from .models import (
    Read, Write, Edit,
    FileOperation, file_path_of, list_session_files, extract_operations,
)


# Line prefix pattern: spaces + line number + → (unicode arrow)
LINE_PREFIX = re.compile(r'^\s*\d+→', re.MULTILINE)

# System reminder tags injected by Claude Code
SYSTEM_REMINDER = re.compile(r'\n*<system-reminder>.*?</system-reminder>\s*$', re.DOTALL)


def strip_line_prefixes(content: str) -> str:
    """Remove line number prefixes from Read output.

    Format: '     1→content'

    Only strips if ALL non-empty lines match the prefix pattern,
    to avoid corrupting files that legitimately contain similar text.
    """
    lines = content.split('\n')
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return content
    if all(LINE_PREFIX.match(line) for line in non_empty):
        return LINE_PREFIX.sub('', content)
    return content


def strip_system_reminders(content: str) -> str:
    """Remove <system-reminder> tags that leak into read content."""
    return SYSTEM_REMINDER.sub('', content)


def _ensure_trailing_slash(path: str) -> str:
    """Ensure a path string ends with /."""
    return path if path.endswith("/") else path + "/"


@dataclass(frozen=True)
class FileReconstruction:
    """Result of attempting to reconstruct a file."""
    path: str
    content: str | None
    success: bool
    baseline_type: str | None = None  # 'write' or 'read'
    baseline_timestamp: str | None = None
    edits_applied: int = 0
    edits_failed: int = 0
    error: str | None = None
    warnings: tuple[str, ...] = ()


def reconstruct_file(
    path: str,
    file_ops: list[FileOperation],
) -> FileReconstruction:
    """Reconstruct a single file from its operations.

    Strategy:
    1. Find latest snapshot (write or full read) — whichever is most recent wins
    2. Apply edits after that snapshot chronologically

    Args:
        path: Absolute file path
        file_ops: Operations for this file only (pre-filtered by caller)
    """
    if not file_ops:
        return FileReconstruction(
            path=path,
            content=None,
            success=False,
            error="No operations found",
        )

    # Find latest snapshot — most recent Write or full Read wins
    # Skip failed writes
    snapshot = None
    snapshot_idx = -1
    snapshot_type = None

    for i, op in enumerate(file_ops):
        if isinstance(op, Write) and not op.is_error:
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
    edits_failed = 0
    warnings: list[str] = []

    for op in file_ops[snapshot_idx + 1:]:
        if isinstance(op, Edit):
            if op.old_string in content:
                if not op.replace_all and content.count(op.old_string) > 1:
                    warnings.append(
                        f"Ambiguous edit at {op.timestamp.isoformat()}: "
                        f"old_string appears {content.count(op.old_string)} times"
                    )
                if op.replace_all:
                    content = content.replace(op.old_string, op.new_string)
                else:
                    content = content.replace(op.old_string, op.new_string, 1)
                edits_applied += 1
            else:
                edits_failed += 1
                warnings.append(
                    f"Edit failed at {op.timestamp.isoformat()}: "
                    f"old_string not found ({op.old_string[:60]!r})"
                )

    return FileReconstruction(
        path=path,
        content=content,
        success=True,
        baseline_type=snapshot_type,
        baseline_timestamp=file_ops[snapshot_idx].timestamp.isoformat(),
        edits_applied=edits_applied,
        edits_failed=edits_failed,
        warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class ProjectReconstruction:
    """Result of reconstructing all files in a project."""
    results: tuple[tuple[str, FileReconstruction], ...]  # (rel_path, result)
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
        project_root: Filter to files under this path

    Returns:
        ProjectReconstruction with all results ready to inspect or write.
    """
    project_root = _ensure_trailing_slash(project_root)
    jsonl_files = list_session_files(claude_project_dir)

    all_operations: list[FileOperation] = []
    for jsonl_path in jsonl_files:
        operations = extract_operations(str(jsonl_path))
        all_operations.extend(operations)

    # Group operations by file path
    ops_by_file: dict[str, list[FileOperation]] = {}
    for op in all_operations:
        op_path = file_path_of(op)
        if op_path and op_path.startswith(project_root):
            ops_by_file.setdefault(op_path, []).append(op)

    # Reconstruct each file in memory
    results: list[tuple[str, FileReconstruction]] = []
    for file_path in sorted(ops_by_file):
        file_ops = ops_by_file[file_path]
        file_ops.sort(key=lambda op: op.timestamp)
        result = reconstruct_file(file_path, file_ops)
        rel_path = file_path[len(project_root):]
        results.append((rel_path, result))

    return ProjectReconstruction(
        results=tuple(results),
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


# =============================================================================
# RESTORE PLAN
# =============================================================================

@dataclass(frozen=True)
class FileDiff:
    """Diff info for a single recovered file."""
    rel_path: str
    result: FileReconstruction
    added: int = 0
    removed: int = 0
    is_new: bool = False


@dataclass(frozen=True)
class RestorePlan:
    """Summary of what a restore would do."""
    reconstruction: ProjectReconstruction
    project_path: str
    output_dir: Path
    changed: tuple[FileDiff, ...] = ()
    matched: tuple[str, ...] = ()
    new_files: tuple[FileDiff, ...] = ()
    failed: tuple[FileDiff, ...] = ()

    @property
    def restorable(self) -> list[FileDiff]:
        return list(self.changed) + list(self.new_files)

    @property
    def restorable_paths(self) -> set[str]:
        return {f.rel_path for f in self.restorable}


def plan_restore(project_path: str, claude_dir: Path, output_dir: Path | None = None) -> RestorePlan:
    """Scan sessions and compute diff against files on disk.

    Always compares reconstructed files against the source project_path,
    regardless of output_dir.
    """
    project_root = _ensure_trailing_slash(project_path)
    reconstruction = plan_project_reconstruction(claude_dir, project_root)
    out = output_dir or Path(project_path)
    project_dir = Path(project_path)

    changed: list[FileDiff] = []
    matched: list[str] = []
    new_files: list[FileDiff] = []

    for rel_path, result in reconstruction.succeeded:
        disk_path = project_dir / rel_path
        if not disk_path.exists():
            new_files.append(FileDiff(rel_path, result, is_new=True))
            continue

        disk_content = disk_path.read_text(errors="replace")
        if disk_content == result.content:
            matched.append(rel_path)
            continue

        disk_lines = disk_content.splitlines(keepends=True)
        restored_lines = result.content.splitlines(keepends=True)
        added = removed = 0
        for line in difflib.unified_diff(disk_lines, restored_lines, n=0):
            if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
                continue
            if line.startswith("+"):
                added += 1
            elif line.startswith("-"):
                removed += 1
        changed.append(FileDiff(rel_path, result, added=added, removed=removed))

    failed = tuple(
        FileDiff(rel_path, result)
        for rel_path, result in reconstruction.failed
    )

    return RestorePlan(
        reconstruction=reconstruction,
        project_path=project_path,
        output_dir=out,
        changed=tuple(changed),
        matched=tuple(matched),
        new_files=tuple(new_files),
        failed=failed,
    )


def execute_restore(plan: RestorePlan, output_dir: Path) -> int:
    """Write the restorable files from a plan. Returns count written."""
    restorable_paths = plan.restorable_paths
    filtered = replace(
        plan.reconstruction,
        results=tuple(
            (p, r) for p, r in plan.reconstruction.results
            if p in restorable_paths
        ),
    )
    write_reconstruction(filtered, output_dir)
    return len(restorable_paths)


def write_patch(reconstruction: ProjectReconstruction, project_path: str, patch_path: str | None = None) -> None:
    """Write reconstruction as a unified diff patch file."""
    project_dir = Path(project_path)
    patches: list[str] = []

    for rel_path, result in reconstruction.succeeded:
        original = project_dir / rel_path
        restored_lines = result.content.splitlines(keepends=True)

        if original.exists():
            original_lines = original.read_text().splitlines(keepends=True)
        else:
            original_lines = []

        diff = difflib.unified_diff(
            original_lines,
            restored_lines,
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
        )
        diff_text = "".join(diff)
        if diff_text:
            patches.append(diff_text)

    if not patches:
        print("No differences found — files already match.")
        return

    out = Path(patch_path) if patch_path else Path("claude-decoder-restore.patch")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(patches))
    print(f"\nPatch written to {out.resolve()}")
    print(f"Apply with: cd {project_path} && git apply {out.resolve()}")
