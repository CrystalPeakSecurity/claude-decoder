#!/usr/bin/env python3
"""Textual TUI for claude-decoder interactive mode."""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.theme import Theme
from textual.widgets import Input, OptionList, SelectionList, Static
from textual.widgets.option_list import Option
from textual.widgets.selection_list import Selection


# =============================================================================
# SHARED
# =============================================================================

_THEME = Theme(
    name="claude-decoder",
    primary="rgb(90, 187, 92)",
    accent="rgb(90, 187, 92)",
    background="rgb(33, 33, 33)",
    surface="rgb(33, 33, 33)",
    panel="rgb(33, 33, 33)",
    dark=True,
)

_RUN = dict(inline=True, mouse=True)

_BASE_CSS = """
Screen {
    height: auto;
    padding: 0 1;
    border-top: tall $primary;
    border-bottom: tall $primary;
}
Static {
    background: $background;
}
#title {
    color: $text-muted;
    margin-bottom: 1;
}
 OptionList {
    height: auto;
    border: none;
    background: $background;
    padding: 0;
}
OptionList > .option-list--option {
    background: $background;
}
OptionList > .option-list--option-hover {
    background: $background;
}
OptionList > .option-list--option-highlighted {
    background: $background;
    text-style: bold;
}
"""


from rich.text import Text
from rich.panel import Panel
from rich.console import Group

_DOT = Text("● ", style="rgb(90,187,92)")
_NO_DOT = Text("  ")


class MarkerOptionList(OptionList):
    """OptionList that shows a colored dot next to the highlighted option."""

    def __init__(self, *args, **kwargs) -> None:
        self._labels: list[str] = []
        super().__init__(*args, **kwargs)

    def on_mount(self) -> None:
        self._labels = [
            self.get_option_at_index(i).prompt.plain
            if isinstance(self.get_option_at_index(i).prompt, Text)
            else str(self.get_option_at_index(i).prompt)
            for i in range(self.option_count)
        ]
        self._update_markers()

    def watch_highlighted(self, value: int | None) -> None:
        super().watch_highlighted(value)
        if self._labels:
            self._update_markers()

    def _update_markers(self) -> None:
        for i in range(self.option_count):
            label = self._labels[i] if i < len(self._labels) else ""
            if i == self.highlighted:
                prompt = Text.assemble(_DOT, label)
            else:
                prompt = Text.assemble(_NO_DOT, label)
            self.replace_option_prompt_at_index(i, prompt)


class _BaseApp(App):
    """Base app with claude-decoder theme and shared settings."""

    INLINE_PADDING = 1
    CSS = _BASE_CSS
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False, system=True),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(ansi_color=True, **kwargs)
        self.register_theme(_THEME)
        self.theme = "claude-decoder"


class EmacsInput(Input):
    """Input with emacs-style alt keybindings."""

    BINDINGS = [
        Binding("alt+d", "delete_right_word", "Delete word right", show=False),
        Binding("alt+backspace", "delete_left_word", "Delete word left", show=False),
        Binding("alt+f", "cursor_right_word", "Move right word", show=False),
        Binding("alt+b", "cursor_left_word", "Move left word", show=False),
    ]


# =============================================================================
# SESSION METADATA
# =============================================================================

from .conversation import fmt_date as _fmt_date
from .models import Entry, ToolUseBlock, ToolResultBlock, TextBlock, list_session_files


@dataclass(frozen=True)
class EntryPreview:
    """Preview of a single entry for display."""
    role: str  # "user", "claude", "tool", "system"
    text: str  # text content or empty for tools
    tool_name: str = ""  # tool name if role == "tool"
    tool_input: dict | None = None  # tool arguments if role == "tool"


@dataclass(frozen=True)
class SessionInfo:
    """Lightweight metadata about a session, extracted without full parsing."""
    session_id: str
    path: Path
    first_message: str
    entry_count: int
    timestamp: datetime
    size_bytes: int
    previews: tuple[EntryPreview, ...] | None = None  # first 2 + last 2

    def __hash__(self) -> int:
        return hash(self.session_id)


def _entry_to_preview(entry: Entry) -> EntryPreview | None:
    """Extract a lightweight preview from a parsed Entry."""
    if entry.type == "file-history-snapshot" and entry.snapshot:
        filenames = ", ".join(entry.snapshot.keys())
        return EntryPreview(role="system", text=f"[snapshot] {filenames}")

    if entry.type == "summary" and entry.summary_text:
        return EntryPreview(role="system", text=f"[summary] {entry.summary_text}")

    if entry.type == "queue-operation" and entry.queue_content:
        return EntryPreview(role="system", text=str(entry.queue_content))

    if entry.type == "system" and entry.subtype:
        text = entry.subtype
        if entry.duration_ms is not None:
            text += f": {entry.duration_ms}ms"
        return EntryPreview(role="system", text=f"[system] {text}")

    if not entry.message:
        return None

    role = entry.message.role

    tool_uses = [b for b in entry.message.content if isinstance(b, ToolUseBlock)]
    if tool_uses:
        if len(tool_uses) == 1:
            return EntryPreview(
                role="tool",
                text="",
                tool_name=tool_uses[0].name,
                tool_input=tool_uses[0].input,
            )
        names = ", ".join(b.name for b in tool_uses)
        return EntryPreview(role="tool", text="", tool_name=f"[{names}]")

    for block in entry.message.content:
        if isinstance(block, ToolResultBlock):
            text = block.content.strip()
            if text:
                return EntryPreview(role="tool", text=text)

    for block in entry.message.content:
        if isinstance(block, TextBlock):
            text = block.text.strip()
            if text:
                display_role = "claude" if role == "assistant" else role
                return EntryPreview(role=display_role, text=text)

    return None


def _read_head_tail_entries(path: Path, n: int = 2) -> tuple[list[EntryPreview], list[EntryPreview], datetime | None, str, int]:
    """Read first n and last n parseable entries from a JSONL file in a single pass.

    Returns (head_entries, tail_entries, timestamp, first_user_text, entry_count).
    """
    head: list[EntryPreview] = []
    tail: deque[EntryPreview] = deque(maxlen=n)
    timestamp: datetime | None = None
    first_text = ""
    entry_count = 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            entry_count += 1

            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                continue

            if data.get("type") == "progress":
                continue

            entry = Entry.from_dict(data)

            if timestamp is None:
                timestamp = entry.timestamp

            preview = _entry_to_preview(entry)
            if preview:
                if len(head) < n:
                    head.append(preview)
                    if not first_text and preview.role == "user" and preview.text:
                        first_line = preview.text.split("\n")[0]
                        first_text = first_line[:80] + ("..." if len(first_line) > 80 else "")
                else:
                    tail.append(preview)

    return head, list(tail), timestamp, first_text, entry_count


def scan_session(jsonl_path: Path) -> SessionInfo:
    """Scan a session JSONL for metadata in a single pass."""
    head_entries, tail_entries, timestamp, first_text, entry_count = _read_head_tail_entries(jsonl_path)
    previews = head_entries + tail_entries

    return SessionInfo(
        session_id=jsonl_path.stem,
        path=jsonl_path,
        first_message=first_text or "(empty session)",
        entry_count=entry_count,
        timestamp=timestamp or datetime.now().astimezone(),
        size_bytes=jsonl_path.stat().st_size,
        previews=tuple(previews) if previews else None,
    )


def scan_sessions(jsonl_files: list[Path]) -> list[SessionInfo]:
    """Scan all sessions for metadata, sorted newest first."""
    sessions = [scan_session(p) for p in jsonl_files]
    sessions.sort(key=lambda s: s.timestamp, reverse=True)
    return sessions


# =============================================================================
# SEARCH
# =============================================================================

def _search_session_file(path: Path, query_lower: str) -> list[EntryPreview]:
    """Search a single JSONL file for entries matching a query."""
    matches: list[EntryPreview] = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or query_lower not in stripped.lower():
                continue

            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                continue

            if data.get("type") == "progress":
                continue

            preview = _entry_to_preview(Entry.from_dict(data))
            if preview is None:
                continue

            searchable = preview.text
            if preview.tool_name:
                searchable += " " + preview.tool_name
            if preview.tool_input:
                searchable += " " + " ".join(str(v) for v in preview.tool_input.values())

            if query_lower in searchable.lower():
                matches.append(preview)

    return matches


def search_sessions(
    sessions: list[SessionInfo], query: str,
) -> list[tuple[SessionInfo, list[EntryPreview]]]:
    """Search all sessions for entries matching query. Returns matches sorted newest first."""
    query_lower = query.lower()
    results: list[tuple[SessionInfo, list[EntryPreview]]] = []

    for session in sessions:
        matches = _search_session_file(session.path, query_lower)
        if matches:
            results.append((session, matches))

    results.sort(key=lambda r: r[0].timestamp, reverse=True)
    return results


# =============================================================================
# TUI: MAIN MENU
# =============================================================================

class MainMenu(_BaseApp):
    """Top-level menu: Restore files or Browse conversations."""

    CSS = _BASE_CSS + """
    OptionList {
        max-height: 6;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "quit", "Quit"),
    ]

    def __init__(self, session_count: int) -> None:
        super().__init__()
        self.session_count = session_count

    def compose(self) -> ComposeResult:
        yield Static(f"claude-decoder — {self.session_count} sessions found", id="title")
        yield MarkerOptionList(
            Option("Restore files", id="restore"),
            Option("Browse conversations", id="chat"),
            Option("Search conversations", id="search"),
        )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.exit(event.option.id)


# =============================================================================
# TUI: SESSION PICKER
# =============================================================================

class SessionPicker(_BaseApp):
    """Pick a session from the list."""

    CSS = _BASE_CSS + """
    OptionList {
        max-height: 20;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "quit", "Back"),
        Binding("/", "search", "Search"),
    ]

    def __init__(self, sessions: list[SessionInfo]) -> None:
        super().__init__()
        self.sessions = sessions

    def compose(self) -> ComposeResult:
        yield Static("Select a session (/ to search):", id="title")
        options = []
        for i, s in enumerate(self.sessions):
            date_str = _fmt_date(s.timestamp)
            sid = s.session_id[:8]
            label = f"{sid}  {date_str:<24} {s.entry_count:>4} entries  {s.first_message}"
            options.append(Option(label, id=str(i)))
        yield MarkerOptionList(*options)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.exit(event.option.id)

    def action_search(self) -> None:
        self.exit("__search__")


# =============================================================================
# TUI: MULTI-SELECT SESSION PICKER
# =============================================================================

class MultiSessionPicker(_BaseApp):
    """Pick one or more sessions from the list."""

    CSS = _BASE_CSS + """
    SelectionList {
        height: auto;
        max-height: 20;
        border: none;
        background: $background;
        padding: 0;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "quit", "Back"),
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
            sid = s.session_id[:8]
            label = f"{sid}  {date_str:<24} {s.entry_count:>4} entries  {s.first_message}"
            selections.append(Selection(label, i, True))  # all selected by default
        yield SelectionList(*selections)

    def action_confirm(self) -> None:
        sel_list = self.query_one(SelectionList)
        selected = list(sel_list.selected)
        self.exit(selected if selected else None)


# =============================================================================
# TUI: SESSION ACTION
# =============================================================================

def _mid_truncate(text: str, max_len: int = 512, max_lines: int = 0) -> str:
    """Truncate text in the middle, showing trimmed count on its own line."""
    if max_lines > 0:
        lines = text.split("\n")
        if len(lines) > max_lines:
            head_n = max_lines // 2
            tail_n = max_lines - head_n
            trimmed = len(lines) - max_lines
            head = "\n".join(lines[:head_n])
            tail = "\n".join(lines[-tail_n:])
            return f"{head}\n...\n<{trimmed} lines trimmed>\n...\n{tail}"
    if len(text) <= max_len:
        return text
    half = max_len // 2
    trimmed = len(text) - max_len
    return f"{text[:half]}...\n<{trimmed} characters trimmed>\n...{text[-half:]}"


def _role_title(role: str) -> str:
    if role == "claude":
        return f"[#DE7356]*[/#DE7356] [bold white]{role}[/bold white]"
    return f"[bold white]{role}[/bold white]"


def _border_style(role: str) -> str:
    if role.startswith("tool"):
        return "#2A5A6B"
    if role == "claude":
        return "#6F3A2B"
    return "grey50"


def _render_preview(entry: EntryPreview) -> Panel:
    """Render an EntryPreview as a Rich Panel."""
    if entry.role == "tool" and entry.tool_input:
        parts = []
        for key, value in entry.tool_input.items():
            val_str = _mid_truncate(str(value), 128, max_lines=5)
            indented = val_str.replace("\n", "\n [dim]│[/dim] ")
            parts.append(f"[bold]{key}[/bold]\n [dim]│[/dim] {indented}")
        body = Text.from_markup("\n\n".join(parts))
        title_label = f"tool: {entry.tool_name}" if entry.tool_name else "tool"
    else:
        body = _mid_truncate(entry.text, max_lines=5)
        title_label = entry.role

    return Panel(
        body,
        title=_role_title(title_label),
        title_align="left",
        border_style=_border_style(entry.role),
        padding=(0, 3),
    )


def _render_previews(previews: tuple[EntryPreview, ...], entry_count: int) -> Group:
    """Render first 2 + last 2 entry previews with a separator."""
    head = previews[:2]
    tail = previews[2:]
    between = entry_count - len(previews)

    parts = []
    for entry in head:
        parts.append(_render_preview(entry))

    if between > 0:
        parts.append(Text(""))
        parts.append(Text(f"< {between} other entries >", style="dim", justify="center"))
        parts.append(Text(""))

    for entry in tail:
        parts.append(_render_preview(entry))

    return Group(*parts)


class SessionAction(_BaseApp):
    """Pick what to do with a selected session."""

    CSS = _BASE_CSS + """
    #preview {
        margin-bottom: 1;
    }
    OptionList {
        max-height: 8;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "quit", "Back"),
    ]

    def __init__(self, session: SessionInfo) -> None:
        super().__init__()
        self.session = session

    def compose(self) -> ComposeResult:
        date_str = _fmt_date(self.session.timestamp)
        yield Static(f"Session: {date_str} ({self.session.entry_count} entries)", id="title")

        if self.session.previews:
            yield Static(
                _render_previews(self.session.previews, self.session.entry_count),
                id="preview",
            )

        yield MarkerOptionList(
            Option("Dump to HTML", id="html"),
            Option("Dump to text", id="text"),
            Option("Dump to HTML (truncated)", id="html_truncated"),
            Option("Dump to text (truncated)", id="text_truncated"),
            Option("Resume in Claude Code", id="resume"),
        )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.exit(event.option.id)


# =============================================================================
# TUI: RESTORE PREVIEW
# =============================================================================

class RestorePreview(_BaseApp):
    """Show restore plan and choose action."""

    CSS = _BASE_CSS + """
    #summary {
        color: $text-muted;
        margin-bottom: 1;
    }
    #files {
        margin-bottom: 1;
    }
    OptionList {
        max-height: 8;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "quit", "Back"),
    ]

    def __init__(self, plan) -> None:
        super().__init__()
        self.plan = plan

    def compose(self) -> ComposeResult:
        plan = self.plan
        yield Static(
            f"{plan.reconstruction.session_count} sessions, {plan.reconstruction.total_operations} operations",
            id="summary",
        )

        lines: list[str] = []
        for f in plan.changed:
            lines.append(f"  {f.rel_path:<50} +{f.added} -{f.removed}")
        for f in plan.new_files:
            lines.append(f"  {f.rel_path:<50} new file")
        if plan.matched:
            lines.append(f"\n  {len(plan.matched)} files already match")
        if plan.failed:
            lines.append(f"\n  {len(plan.failed)} unrecoverable:")
            for f in plan.failed:
                lines.append(f"    {f.rel_path:<48} ({f.result.error})")

        for f in list(plan.changed) + list(plan.new_files):
            if f.result.warnings:
                for w in f.result.warnings:
                    lines.append(f"  warning: {f.rel_path}: {w}")

        if lines:
            yield Static("\n".join(lines), id="files")

        yield MarkerOptionList(
            Option("Cancel", id="cancel"),
            Option(f"Overwrite files in {plan.project_path}", id="overwrite"),
            Option("Write to a different directory", id="different"),
            Option("Write as a git patch", id="patch"),
        )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.exit(event.option.id)


class InputPrompt(_BaseApp):
    """Prompt for a single text input (directory path, search query, etc.)."""

    CSS = _BASE_CSS + """
    Input {
        padding: 0 1;
    }
    """

    BINDINGS = [Binding("escape", "quit", "Cancel")]

    def __init__(self, title: str = "Enter value:", default: str = "", placeholder: str = "") -> None:
        super().__init__()
        self._title = title
        self._default = default
        self._placeholder = placeholder

    def compose(self) -> ComposeResult:
        yield Static(self._title, id="title")
        yield EmacsInput(value=self._default, placeholder=self._placeholder)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        self.exit(value if value else None)


class SearchResults(_BaseApp):
    """Display search results and pick a session."""

    CSS = _BASE_CSS + """
    OptionList {
        max-height: 20;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "quit", "Back"),
    ]

    def __init__(self, query: str, results: list[tuple[SessionInfo, list[EntryPreview]]]) -> None:
        super().__init__()
        self.query = query
        self.results = results

    def compose(self) -> ComposeResult:
        total = sum(len(entries) for _, entries in self.results)
        yield Static(
            f'"{self.query}" — {total} matches in {len(self.results)} sessions',
            id="title",
        )

        if not self.results:
            return

        options = []
        for i, (session, matches) in enumerate(self.results):
            date_str = _fmt_date(session.timestamp)
            first = matches[0]
            if first.tool_name:
                snippet = f"[{first.tool_name}]"
            else:
                first_line = first.text.split("\n")[0]
                snippet = first_line[:60] + ("..." if len(first_line) > 60 else "")
            sid = session.session_id[:8]
            label = f"{sid}  {date_str:<24} {len(matches):>3} hits  {snippet}"
            options.append(Option(label, id=str(i)))

        yield MarkerOptionList(*options)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.exit(event.option.id)


# =============================================================================
# ENTRY POINTS (called from cli.py)
# =============================================================================

def run_restore_interactive(plan) -> str | None:
    """Show restore preview in TUI, handle action, return status message."""
    from .reconstruct import execute_restore, write_patch

    choice = RestorePreview(plan).run(**_RUN)
    cwd = str(Path.cwd().resolve())

    if choice == "overwrite":
        count = execute_restore(plan, Path(plan.project_path))
        return f"Restored {count} files to {plan.project_path}/"
    elif choice == "different":
        dir_path = InputPrompt(
            title="Enter output directory:",
            default=cwd,
        ).run(**_RUN)
        if not dir_path:
            return None
        out = Path(dir_path)
        count = execute_restore(plan, out)
        return f"Restored {count} files to {out.resolve()}/"
    elif choice == "patch":
        patch_path = InputPrompt(
            title="Enter patch file path:",
            default=str(Path(cwd) / "claude-decoder-restore.patch"),
        ).run(**_RUN)
        if not patch_path:
            return None
        write_patch(plan.reconstruction, plan.project_path, patch_path=patch_path)
        return f"Patch written to {Path(patch_path).resolve()}"
    else:
        return None


def run_interactive(claude_dir: Path, project_path: str) -> None:
    """Run the full interactive TUI flow."""
    jsonl_files = list_session_files(claude_dir)
    if not jsonl_files:
        print(f"No session files found in {claude_dir}")
        return

    sessions: list[SessionInfo] | None = None

    def _get_sessions() -> list[SessionInfo]:
        nonlocal sessions
        if sessions is None:
            print("Scanning sessions...", end="", flush=True)
            sessions = scan_sessions(jsonl_files)
            print(f" {len(sessions)} found")
        return sessions

    while True:
        choice = MainMenu(len(jsonl_files)).run(**_RUN)

        if choice == "restore":
            from .reconstruct import plan_restore
            print(f"Scanning {claude_dir.name}...", end="", flush=True)
            plan = plan_restore(project_path, claude_dir)
            print(f" {plan.reconstruction.session_count} sessions, {plan.reconstruction.total_operations} operations")
            if not plan.restorable:
                print("Nothing to restore.")
            else:
                result = run_restore_interactive(plan)
                if result:
                    print(result)
            # loop back to main menu
        elif choice == "chat":
            run_chat_picker(jsonl_files, sessions=_get_sessions(), project_path=project_path)
            # After chat picker returns (via escape), loop back to main menu
        elif choice == "search":
            _run_search(jsonl_files, sessions=_get_sessions(), project_path=project_path)
        else:
            return


def _browse_search_results(query: str, results: list[tuple[SessionInfo, list[EntryPreview]]], project_path: str | None = None) -> None:
    """Let the user pick from search results and act on sessions."""
    while True:
        result_idx = SearchResults(query, results).run(**_RUN)
        if result_idx is None:
            return

        session, matches = results[int(result_idx)]
        match_previews = tuple(matches[:2] + matches[-2:]) if len(matches) > 4 else tuple(matches)
        display_session = replace(session, previews=match_previews)
        _handle_session_action(display_session, project_path)


def _run_search(jsonl_files: list[Path], *, sessions: list[SessionInfo] | None = None, project_path: str | None = None) -> None:
    """Run the search flow: prompt → search → results → action."""
    query = InputPrompt(title="Search conversations:", placeholder="search query").run(**_RUN)
    if not query:
        return

    if sessions is None:
        print(f'Scanning sessions...', end="", flush=True)
        sessions = scan_sessions(jsonl_files)
    print(f'Searching for "{query}"...', end="", flush=True)
    results = search_sessions(sessions, query)
    total = sum(len(m) for _, m in results)
    print(f" {total} matches in {len(results)} sessions")

    if not results:
        return

    _browse_search_results(query, results, project_path)


def _handle_session_action(session: SessionInfo, project_path: str | None = None) -> None:
    """Show session action menu and handle the chosen action."""
    from .conversation import render_session as render_conversation

    action = SessionAction(session).run(**_RUN)
    if action is None:
        return

    if action in ("html", "text", "html_truncated", "text_truncated"):
        truncate = action.endswith("_truncated")
        fmt = action.replace("_truncated", "")
        cwd = str(Path.cwd().resolve())
        ext = "html" if fmt == "html" else "txt"
        default = str(Path(cwd) / f"{session.session_id}.{ext}")
        out_path = InputPrompt(
            title=f"Save {fmt.upper()} file to:",
            default=default,
        ).run(**_RUN)
        if not out_path:
            return
        render_conversation(str(session.path), out_path, truncate=truncate)
        print(f"Written to {out_path}")
    elif action == "resume":
        resume_dir = Path(project_path) if project_path else None
        if resume_dir and not resume_dir.exists():
            confirm = input(f"{resume_dir} does not exist. Create it? [Y/n] ").strip().lower()
            if confirm and confirm != "y":
                print("Aborted.")
                return
            resume_dir.mkdir(parents=True, exist_ok=True)

        cmd = ["claude", "--resume", session.session_id]
        if resume_dir:
            print(f"Running: cd {resume_dir} && {' '.join(cmd)}")
            os.chdir(resume_dir)
        else:
            print(f"Running: {' '.join(cmd)}")
        os.execvp("claude", cmd)


def run_chat_picker(jsonl_files: list[Path], *, sessions: list[SessionInfo] | None = None, project_path: str | None = None) -> None:
    """Run the interactive chat session picker."""
    if sessions is None:
        print("Scanning sessions...", end="", flush=True)
        sessions = scan_sessions(jsonl_files)
        print(f" {len(sessions)} found")

    while True:
        idx_str = SessionPicker(sessions).run(**_RUN)
        if idx_str is None:
            return

        if idx_str == "__search__":
            query = InputPrompt(title="Search conversations:", placeholder="search query").run(**_RUN)
            if not query:
                continue

            print(f'Searching for "{query}"...', end="", flush=True)
            results = search_sessions(sessions, query)
            total = sum(len(m) for _, m in results)
            print(f" {total} matches in {len(results)} sessions")

            if results:
                _browse_search_results(query, results, project_path)
            continue

        session = sessions[int(idx_str)]
        _handle_session_action(session, project_path)
