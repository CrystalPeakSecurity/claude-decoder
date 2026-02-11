#!/usr/bin/env python3
"""Textual TUI for clauderec interactive mode."""

from __future__ import annotations

import json
import os
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Input, OptionList, SelectionList, Static
from textual.widgets.option_list import Option
from textual.widgets.selection_list import Selection


# =============================================================================
# SESSION METADATA
# =============================================================================

@dataclass
class SessionInfo:
    """Lightweight metadata about a session, extracted without full parsing."""
    session_id: str
    path: Path
    first_message: str
    message_count: int
    timestamp: datetime
    size_bytes: int


def _local_tz() -> datetime:
    return datetime.now().astimezone().tzinfo


def _fmt_date(dt: datetime) -> str:
    local = dt.astimezone(_local_tz())
    return local.strftime("%-b %-d, %Y %-I:%M %p")


def scan_session(jsonl_path: Path) -> SessionInfo:
    """Scan a session JSONL for metadata without full parsing."""
    first_message = ""
    message_count = 0
    first_timestamp = None

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = data.get("type", "")
            if entry_type not in ("user", "assistant"):
                continue

            msg = data.get("message", {})
            if not msg:
                continue

            role = msg.get("role", "")
            content = msg.get("content", [])

            if first_timestamp is None:
                ts_str = data.get("timestamp", "")
                try:
                    first_timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    first_timestamp = datetime.now()

            # Count user and assistant text messages
            if isinstance(content, list):
                has_text = any(
                    isinstance(b, dict) and b.get("type") == "text" and b.get("text", "").strip()
                    for b in content
                )
                has_tool = any(
                    isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result")
                    for b in content
                )
                if has_text or has_tool:
                    message_count += 1

            # Grab first user text message
            if role == "user" and not first_message and isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "").strip()
                        if text:
                            # Clean up: take first line, truncate
                            first_line = text.split("\n")[0]
                            first_message = first_line[:80] + ("..." if len(first_line) > 80 else "")
                            break

    return SessionInfo(
        session_id=jsonl_path.stem,
        path=jsonl_path,
        first_message=first_message or "(empty session)",
        message_count=message_count,
        timestamp=first_timestamp or datetime.now(),
        size_bytes=jsonl_path.stat().st_size,
    )


def scan_sessions(jsonl_files: list[Path]) -> list[SessionInfo]:
    """Scan all sessions for metadata, sorted newest first."""
    sessions = [scan_session(p) for p in jsonl_files]
    sessions.sort(key=lambda s: s.timestamp, reverse=True)
    return sessions


# =============================================================================
# TUI: MAIN MENU
# =============================================================================

class MainMenu(App[str]):
    """Top-level menu: Restore files or Browse conversations."""

    CSS = """
    Screen {
        height: auto;
    }
    #title {
        color: $text-muted;
        margin-bottom: 1;
    }
    OptionList {
        height: auto;
        max-height: 6;
    }
    """

    BINDINGS = [Binding("q", "quit", "Quit")]

    def __init__(self, session_count: int) -> None:
        super().__init__()
        self.session_count = session_count

    def compose(self) -> ComposeResult:
        yield Static(f"clauderec — {self.session_count} sessions found", id="title")
        options = OptionList(
            Option("Restore files", id="restore"),
            Option("Browse conversations", id="chat"),
        )
        yield options

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.exit(event.option.id)


# =============================================================================
# TUI: SESSION PICKER
# =============================================================================

class SessionPicker(App[str | None]):
    """Pick a session from the list."""

    CSS = """
    Screen {
        height: auto;
    }
    #title {
        color: $text-muted;
        margin-bottom: 1;
    }
    OptionList {
        height: auto;
        max-height: 20;
    }
    """

    BINDINGS = [Binding("q", "quit", "Quit")]

    def __init__(self, sessions: list[SessionInfo]) -> None:
        super().__init__()
        self.sessions = sessions

    def compose(self) -> ComposeResult:
        yield Static("Select a session:", id="title")
        options = []
        for i, s in enumerate(self.sessions):
            date_str = _fmt_date(s.timestamp)
            label = f"{date_str:<24} {s.message_count:>4} msgs  {s.first_message}"
            options.append(Option(label, id=str(i)))
        yield OptionList(*options)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.exit(event.option.id)


# =============================================================================
# TUI: MULTI-SELECT SESSION PICKER
# =============================================================================

class MultiSessionPicker(App[list[int] | None]):
    """Pick one or more sessions from the list."""

    CSS = """
    Screen {
        height: auto;
    }
    #title {
        color: $text-muted;
        margin-bottom: 1;
    }
    #hint {
        color: $text-muted;
        margin-top: 1;
    }
    SelectionList {
        height: auto;
        max-height: 20;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("enter", "confirm", "Confirm"),
    ]

    def __init__(self, sessions: list[SessionInfo]) -> None:
        super().__init__()
        self.sessions = sessions

    def compose(self) -> ComposeResult:
        yield Static("Select sessions (space to toggle, enter to confirm):", id="title")
        selections = []
        for i, s in enumerate(self.sessions):
            date_str = _fmt_date(s.timestamp)
            label = f"{date_str:<24} {s.message_count:>4} msgs  {s.first_message}"
            selections.append(Selection(label, i, True))  # all selected by default
        yield SelectionList(*selections)

    def action_confirm(self) -> None:
        sel_list = self.query_one(SelectionList)
        selected = list(sel_list.selected)
        self.exit(selected if selected else None)


# =============================================================================
# TUI: SESSION ACTION
# =============================================================================

class SessionAction(App[str]):
    """Pick what to do with a selected session."""

    CSS = """
    Screen {
        height: auto;
    }
    #title {
        color: $text-muted;
        margin-bottom: 1;
    }
    OptionList {
        height: auto;
        max-height: 8;
    }
    """

    BINDINGS = [Binding("q", "quit", "Quit")]

    def __init__(self, session: SessionInfo) -> None:
        super().__init__()
        self.session = session

    def compose(self) -> ComposeResult:
        date_str = _fmt_date(self.session.timestamp)
        yield Static(f"Session: {date_str} ({self.session.message_count} messages)", id="title")
        yield OptionList(
            Option("Dump to HTML", id="html"),
            Option("Dump to text", id="text"),
            Option("Resume in Claude Code", id="resume"),
        )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.exit(event.option.id)


# =============================================================================
# TUI: RESTORE CONFIRMATION
# =============================================================================

class RestoreConfirm(App[str | None]):
    """Choose how to write restored files."""

    CSS = """
    Screen {
        height: auto;
    }
    #title {
        color: $text-muted;
        margin-bottom: 1;
    }
    OptionList {
        height: auto;
        max-height: 8;
    }
    """

    BINDINGS = [Binding("q", "quit", "Quit")]

    def __init__(self, project_path: str) -> None:
        super().__init__()
        self.project_path = project_path

    def compose(self) -> ComposeResult:
        yield Static("How should the files be written?", id="title")
        yield OptionList(
            Option("Cancel", id="cancel"),
            Option(f"Overwrite files in {self.project_path}", id="overwrite"),
            Option("Write to a different directory", id="different"),
            Option("Write as a git patch", id="patch"),
        )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.exit(event.option.id)


class DirectoryPrompt(App[str | None]):
    """Prompt for a directory path."""

    CSS = """
    Screen {
        height: auto;
    }
    #title {
        color: $text-muted;
        margin-bottom: 1;
    }
    Input {
        margin-top: 1;
    }
    """

    BINDINGS = [Binding("escape", "quit", "Cancel")]

    def compose(self) -> ComposeResult:
        yield Static("Enter output directory:", id="title")
        yield Input(placeholder="path/to/output")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        self.exit(value if value else None)


# =============================================================================
# ENTRY POINTS (called from cli.py)
# =============================================================================

def run_interactive(claude_dir: Path, project_path: str) -> None:
    """Run the full interactive TUI flow."""
    jsonl_files = sorted(claude_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not jsonl_files:
        print(f"No session files found in {claude_dir}")
        return

    choice = MainMenu(len(jsonl_files)).run(inline=True)

    if choice == "restore":
        # Delegate to the restore flow
        from .cli import cmd_restore
        import argparse
        args = argparse.Namespace(
            project_path=project_path,
            output=None,
            dump_operations=None,
        )
        cmd_restore(args)
    elif choice == "chat":
        run_chat_picker(claude_dir, jsonl_files)


def run_chat_picker(claude_dir: Path, jsonl_files: list[Path]) -> None:
    """Run the interactive chat session picker."""
    from .conversation import render_session as render_conversation

    print("Scanning sessions...")
    sessions = scan_sessions(jsonl_files)

    idx_str = SessionPicker(sessions).run(inline=True)
    if idx_str is None:
        return

    session = sessions[int(idx_str)]

    action = SessionAction(session).run(inline=True)
    if action is None:
        return

    if action == "html":
        out_path = f"{session.session_id}.html"
        render_conversation(str(session.path), out_path)
        print(f"Written to {out_path}")
    elif action == "text":
        out_path = f"{session.session_id}.txt"
        render_conversation(str(session.path), out_path)
        print(f"Written to {out_path}")
    elif action == "resume":
        cmd = ["claude", "--resume", session.session_id]
        print(f"Running: {' '.join(cmd)}")
        os.execvp("claude", cmd)
