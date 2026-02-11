#!/usr/bin/env python3
"""CLI entry point for claude-decoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .extract import extract_project
from .models import list_session_files
from .reconstruct import (
    RestorePlan,
    plan_restore, execute_restore, write_patch,
)


class ProjectNotFoundError(Exception):
    """Raised when no Claude project directory can be found for a given path."""
    pass


def _read_cwd_from_project(project_dir: Path) -> str | None:
    """Read the real project path from the first JSONL entry's cwd field."""
    for jsonl in list_session_files(project_dir):
        with open(jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cwd = data.get("cwd", "")
                if cwd:
                    return cwd
    return None


def find_claude_project_dir(project_path: str) -> Path:
    """Auto-detect the Claude project directory for a given project path.

    Mangles the absolute path by replacing / with - to match
    ~/.claude/projects/<mangled>/ naming convention.

    Falls back to matching by cwd stored in JSONL entries, since Claude Code's
    mangling may differ from a simple / -> - replacement (e.g. underscores).

    Raises:
        ProjectNotFoundError: if no matching project directory is found.
    """
    resolved = Path(project_path).resolve()
    mangled = str(resolved).replace("/", "-")
    claude_dir = Path.home() / ".claude" / "projects" / mangled
    if claude_dir.exists():
        return claude_dir

    # Fallback: search project dirs by matching cwd from JSONL entries
    projects_dir = Path.home() / ".claude" / "projects"
    available: list[str] = []
    resolved_str = str(resolved)
    if projects_dir.exists():
        for p in sorted(projects_dir.iterdir()):
            if not p.is_dir():
                continue
            real_path = _read_cwd_from_project(p)
            if real_path:
                if real_path == resolved_str:
                    return p
                available.append(f"  {real_path}")
            else:
                available.append(f"  {p.name}")

    msg = f"No Claude data found for {resolved}\nExpected: {claude_dir}\n"
    if available:
        msg += "\nAvailable projects:\n" + "\n".join(available)
    raise ProjectNotFoundError(msg)


def _print_plan(plan: RestorePlan) -> None:
    """Print restore plan summary to stdout (for non-interactive modes)."""
    print(f"  {plan.reconstruction.session_count} sessions, {plan.reconstruction.total_operations} operations")
    total_recoverable = len(plan.changed) + len(plan.new_files) + len(plan.matched)
    print(f"  {total_recoverable} files recoverable, {len(plan.failed)} unrecoverable")
    print()

    if plan.changed or plan.new_files:
        print(f"Will write to: {plan.output_dir.resolve()}/\n")
        for f in plan.changed:
            print(f"  {f.rel_path:<50} +{f.added} -{f.removed}")
        for f in plan.new_files:
            print(f"  {f.rel_path:<50} new file")

    if plan.matched:
        print(f"\n  {len(plan.matched)} files already match")

    if plan.failed:
        print(f"\n  {len(plan.failed)} unrecoverable:")
        for f in plan.failed:
            print(f"    {f.rel_path:<48} ({f.result.error})")

    # Surface reconstruction warnings
    for f in list(plan.changed) + list(plan.new_files):
        if f.result.warnings:
            for w in f.result.warnings:
                print(f"  warning: {f.rel_path}: {w}")


def cmd_restore(args: argparse.Namespace) -> str | None:
    """Handle the restore subcommand. Returns a status message or None."""
    project_path = str(Path(args.project_path).resolve())
    claude_dir = find_claude_project_dir(project_path)

    if args.dump_operations:
        if args.output or args.patch or args.in_place:
            print("Error: --dump-operations cannot be combined with -o, --in-place, or --patch.", file=sys.stderr)
            sys.exit(1)
        extract_project(claude_dir, Path(args.dump_operations), project_path.rstrip("/") + "/")
        return None

    n_output_flags = sum(bool(x) for x in [args.output, args.patch, args.in_place])
    if n_output_flags > 1:
        print("Error: -o, --in-place, and --patch are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output) if args.output else None

    print(f"Scanning {claude_dir.name}...", end="", flush=True)
    plan = plan_restore(project_path, claude_dir, output_dir)
    print(f" {plan.reconstruction.session_count} sessions, {plan.reconstruction.total_operations} operations")

    if not plan.restorable:
        _print_plan(plan)
        print("Nothing to restore.")
        return None

    # Non-interactive modes
    if args.output:
        _print_plan(plan)
        count = execute_restore(plan, plan.output_dir)
        msg = f"Restored {count} files to {plan.output_dir.resolve()}/"
        print(f"\n{msg}")
        return msg

    if args.patch:
        _print_plan(plan)
        write_patch(plan.reconstruction, project_path, patch_path=args.patch)
        return None

    if args.in_place:
        _print_plan(plan)
        if not args.yes:
            confirm = input(f"Restore {len(plan.restorable)} files to {project_path}/? [Y/n] ").strip().lower()
            if confirm and confirm != "y":
                print("Aborted.")
                return None
        count = execute_restore(plan, Path(project_path))
        msg = f"Restored {count} files to {project_path}/"
        print(msg)
        return msg

    # Interactive mode — show plan in TUI
    from .tui import run_restore_interactive
    return run_restore_interactive(plan)


def cmd_chat(args: argparse.Namespace) -> None:
    """Handle the chat subcommand."""
    from .conversation import render_session as render_conversation

    project_path = str(Path(args.project_path).resolve())
    claude_dir = find_claude_project_dir(project_path)

    # Collect session files
    jsonl_files = list_session_files(claude_dir)
    if not jsonl_files:
        print(f"No session files found in {claude_dir}", file=sys.stderr)
        sys.exit(1)

    truncate = args.truncate

    if args.dump_html or args.dump_text:
        from .conversation import render_sessions

        fmt = "html" if args.dump_html else "txt"
        out_path = Path(args.dump_html or args.dump_text)

        if args.session:
            # Single session by ID
            matches = [f for f in jsonl_files if f.stem == args.session]
            if not matches:
                print(f"Session not found: {args.session}", file=sys.stderr)
                sys.exit(1)
            target = out_path if out_path.suffix else out_path / f"{args.session}.{fmt}"
            render_conversation(str(matches[0]), str(target), truncate=truncate)
            print(f"Written to {target}")
        elif out_path.suffix:
            # Output looks like a file — pick sessions interactively if multiple
            if len(jsonl_files) == 1:
                render_conversation(str(jsonl_files[0]), str(out_path), truncate=truncate)
                print(f"Written to {out_path}")
            else:
                from .tui import scan_sessions, MultiSessionPicker
                sessions = scan_sessions(jsonl_files)
                selected_indices = MultiSessionPicker(sessions).run(inline=True)
                if not selected_indices:
                    print("No sessions selected.")
                    return
                selected_paths = [str(sessions[i].path) for i in selected_indices]
                render_sessions(selected_paths, str(out_path), title=Path(project_path).name, truncate=truncate)
                print(f"Written {len(selected_paths)} sessions to {out_path}")
        else:
            # Output is a directory — dump all sessions individually
            out_path.mkdir(parents=True, exist_ok=True)
            for jsonl_path in jsonl_files:
                target = out_path / f"{jsonl_path.stem}.{fmt}"
                render_conversation(str(jsonl_path), str(target), truncate=truncate)
                print(f"  {target.name}")
            print(f"\nDumped {len(jsonl_files)} sessions to {out_path.resolve()}/")
        return

    # Interactive mode
    from .tui import run_chat_picker
    run_chat_picker(jsonl_files, project_path=project_path)


USAGE = """\
usage: claude-decoder [PROJECT_PATH]
       claude-decoder restore PROJECT_PATH [-i] [-o DIR] [--patch FILE] [--dump-operations DIR]
       claude-decoder chat PROJECT_PATH [--dump-html PATH] [--dump-text PATH] [--session ID] [--truncate]

Recover files and conversations from Claude Code session logs.

Commands:
  claude-decoder PATH                  Interactive mode (TUI)
  claude-decoder restore PATH          Restore project files (with dry-run)
  claude-decoder chat PATH             Browse and dump conversations

Run 'claude-decoder <command> --help' for command-specific options.
"""


def _parse_restore(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="claude-decoder restore", description="Restore project files")
    parser.add_argument("project_path", help="Path to the project to recover")
    parser.add_argument("-o", "--output", default=None, help="Output directory")
    parser.add_argument("-i", "--in-place", action="store_true", help="Restore files back to the project directory")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt (with --in-place)")
    parser.add_argument("--patch", metavar="FILE", help="Write a unified diff patch instead of restoring files")
    parser.add_argument(
        "--dump-operations", metavar="DIR",
        help="Instead of restoring, dump raw timestamped operation files to DIR",
    )
    return parser.parse_args(argv)


def _parse_chat(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="claude-decoder chat", description="Browse and dump conversations")
    parser.add_argument("project_path", help="Path to the project")
    parser.add_argument("--dump-html", metavar="PATH", help="Dump session(s) as HTML")
    parser.add_argument("--dump-text", metavar="PATH", help="Dump session(s) as text")
    parser.add_argument("--session", metavar="ID", help="Target a specific session by ID")
    parser.add_argument("--truncate", action="store_true", help="Truncate long tool inputs/outputs")
    return parser.parse_args(argv)


def main() -> None:
    argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return

    command = argv[0]

    try:
        if command == "restore":
            args = _parse_restore(argv[1:])
            cmd_restore(args)
        elif command == "chat":
            args = _parse_chat(argv[1:])
            cmd_chat(args)
        else:
            # Bare: claude-decoder PROJECT_PATH → interactive TUI
            if "--help" in argv or "-h" in argv:
                print(USAGE)
                return
            project_path = str(Path(command).resolve())
            from .tui import run_interactive
            claude_dir = find_claude_project_dir(project_path)
            run_interactive(claude_dir, project_path)
    except ProjectNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
