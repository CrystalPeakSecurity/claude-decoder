#!/usr/bin/env python3
"""CLI entry point for clauderec."""

import argparse
import sys
from pathlib import Path

from .extract import extract_project
from .reconstruct import plan_project_reconstruction, write_reconstruction


def _read_cwd_from_project(project_dir: Path) -> str | None:
    """Read the real project path from the first JSONL entry's cwd field."""
    import json
    for jsonl in project_dir.glob("*.jsonl"):
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
    """
    resolved = Path(project_path).resolve()
    mangled = str(resolved).replace("/", "-")
    claude_dir = Path.home() / ".claude" / "projects" / mangled
    if not claude_dir.exists():
        projects_dir = Path.home() / ".claude" / "projects"
        available: list[str] = []
        if projects_dir.exists():
            for p in sorted(projects_dir.iterdir()):
                if not p.is_dir():
                    continue
                real_path = _read_cwd_from_project(p)
                if real_path:
                    available.append(f"  {real_path}")
                else:
                    available.append(f"  {p.name}")
        msg = f"No Claude data found for {resolved}\nExpected: {claude_dir}\n"
        if available:
            msg += "\nAvailable projects:\n" + "\n".join(available)
        print(msg, file=sys.stderr)
        sys.exit(1)
    return claude_dir


def _write_patch(reconstruction, project_path: str) -> None:
    """Write reconstruction as a unified diff patch file."""
    import difflib

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

    patch_path = Path("clauderec-restore.patch")
    patch_path.write_text("\n".join(patches))
    print(f"\nPatch written to {patch_path.resolve()}")
    print(f"Apply with: cd {project_path} && git apply {patch_path.resolve()}")


def cmd_restore(args: argparse.Namespace) -> None:
    """Handle the restore subcommand."""
    project_path = str(Path(args.project_path).resolve())
    project_root = project_path.rstrip("/") + "/"
    claude_dir = find_claude_project_dir(project_path)

    if args.dump_operations:
        if args.output:
            print("Error: --dump-operations and -o cannot be used together.", file=sys.stderr)
            sys.exit(1)
        extract_project(claude_dir, Path(args.dump_operations), project_root)
        return

    # Plan (dry-run)
    print(f"Scanning {claude_dir.name}...")
    reconstruction = plan_project_reconstruction(claude_dir, project_root)

    print(f"  {reconstruction.session_count} sessions, {reconstruction.total_operations} operations")
    print(f"  {len(reconstruction.succeeded)} files recoverable, {len(reconstruction.failed)} unrecoverable")
    print()

    if reconstruction.succeeded:
        output_dir = Path(args.output) if args.output else Path(project_path)
        print(f"Will write to: {output_dir.resolve()}/\n")
        for rel_path, result in reconstruction.succeeded:
            status = f"[{result.baseline_type}]"
            if result.edits_applied:
                status += f" +{result.edits_applied} edits"
            print(f"  {rel_path:<50} {status}")

    if reconstruction.failed:
        print(f"\n  {len(reconstruction.failed)} unrecoverable:")
        for rel_path, result in reconstruction.failed:
            print(f"    {rel_path:<48} ({result.error})")

    if not reconstruction.succeeded:
        print("Nothing to restore.")
        return

    # If -o was given, skip the interactive prompt
    if args.output:
        output_dir = Path(args.output)
        write_reconstruction(reconstruction, output_dir)
        print(f"\nRestored {len(reconstruction.succeeded)} files to {output_dir.resolve()}/")
        return

    from .tui import RestoreConfirm, DirectoryPrompt

    choice = RestoreConfirm(project_path).run(inline=True)

    if choice == "overwrite":
        write_reconstruction(reconstruction, Path(project_path))
        print(f"Restored {len(reconstruction.succeeded)} files to {project_path}/")
    elif choice == "different":
        dir_path = DirectoryPrompt().run(inline=True)
        if not dir_path:
            print("Aborted.")
            return
        output_dir = Path(dir_path)
        write_reconstruction(reconstruction, output_dir)
        print(f"Restored {len(reconstruction.succeeded)} files to {output_dir.resolve()}/")
    elif choice == "patch":
        _write_patch(reconstruction, project_path)
    else:
        print("Aborted.")


def cmd_chat(args: argparse.Namespace) -> None:
    """Handle the chat subcommand."""
    from .conversation import render_session as render_conversation

    project_path = str(Path(args.project_path).resolve())
    claude_dir = find_claude_project_dir(project_path)

    # Collect session files
    jsonl_files = sorted(claude_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not jsonl_files:
        print(f"No session files found in {claude_dir}", file=sys.stderr)
        sys.exit(1)

    if args.dump_html or args.dump_text:
        from .conversation import render_sessions

        fmt = "html" if args.dump_html else "text"
        out_path = Path(args.dump_html or args.dump_text)

        if args.session:
            # Single session by ID
            matches = [f for f in jsonl_files if f.stem == args.session]
            if not matches:
                print(f"Session not found: {args.session}", file=sys.stderr)
                sys.exit(1)
            target = out_path if out_path.suffix else out_path / f"{args.session}.{fmt}"
            render_conversation(str(matches[0]), str(target))
            print(f"Written to {target}")
        elif out_path.suffix:
            # Output looks like a file — pick sessions interactively if multiple
            if len(jsonl_files) == 1:
                render_conversation(str(jsonl_files[0]), str(out_path))
                print(f"Written to {out_path}")
            else:
                from .tui import scan_sessions, MultiSessionPicker
                sessions = scan_sessions(jsonl_files)
                selected_indices = MultiSessionPicker(sessions).run(inline=True)
                if not selected_indices:
                    print("No sessions selected.")
                    return
                selected_paths = [str(sessions[i].path) for i in selected_indices]
                render_sessions(selected_paths, str(out_path), title=Path(project_path).name)
                print(f"Written {len(selected_paths)} sessions to {out_path}")
        else:
            # Output is a directory — dump all sessions individually
            out_path.mkdir(parents=True, exist_ok=True)
            for jsonl_path in jsonl_files:
                target = out_path / f"{jsonl_path.stem}.{fmt}"
                render_conversation(str(jsonl_path), str(target))
                print(f"  {target.name}")
            print(f"\nDumped {len(jsonl_files)} sessions to {out_path.resolve()}/")
        return

    # Interactive mode
    from .tui import run_chat_picker
    run_chat_picker(claude_dir, jsonl_files)


USAGE = """\
usage: clauderec [PROJECT_PATH]
       clauderec restore PROJECT_PATH [-o DIR] [--dump-operations DIR]
       clauderec chat PROJECT_PATH [--dump-html PATH] [--dump-text PATH] [--session ID]

Recover files and conversations from Claude Code session logs.

Commands:
  clauderec PATH                  Interactive mode (TUI)
  clauderec restore PATH          Restore project files (with dry-run)
  clauderec chat PATH             Browse and dump conversations

Run 'clauderec <command> --help' for command-specific options.
"""


def _parse_restore(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="clauderec restore", description="Restore project files")
    parser.add_argument("project_path", help="Path to the project to recover")
    parser.add_argument("-o", "--output", default=None, help="Output directory (default: project path)")
    parser.add_argument(
        "--dump-operations", metavar="DIR",
        help="Instead of restoring, dump raw timestamped operation files to DIR",
    )
    return parser.parse_args(argv)


def _parse_chat(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="clauderec chat", description="Browse and dump conversations")
    parser.add_argument("project_path", help="Path to the project")
    parser.add_argument("--dump-html", metavar="PATH", help="Dump session(s) as HTML")
    parser.add_argument("--dump-text", metavar="PATH", help="Dump session(s) as text")
    parser.add_argument("--session", metavar="ID", help="Target a specific session by ID")
    return parser.parse_args(argv)


def main() -> None:
    argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return

    command = argv[0]

    if command == "restore":
        args = _parse_restore(argv[1:])
        cmd_restore(args)
    elif command == "chat":
        args = _parse_chat(argv[1:])
        cmd_chat(args)
    else:
        # Bare: clauderec PROJECT_PATH → interactive TUI
        project_path = str(Path(command).resolve())
        from .tui import run_interactive
        claude_dir = find_claude_project_dir(project_path)
        run_interactive(claude_dir, project_path)
