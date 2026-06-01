"""TDD: Linear decisions should credit the decisions dimension in closeout scoring.

RED phase: Write failing test proving Linear comments/decisions don't score.
See test_close_thread_command.py for full context on _init_test_kanban.

The workflow: Discord thread → /close-thread → build_close_thread_packet.
When no Kanban task is attached, decisions=0 and the closeout scores YELLOW(17/20)
because Linear decisions are never read as a source.

The fix: add Linear as a decision source alongside Kanban.
"""
from __future__ import annotations

from pathlib import Path

from hermes_cli import kanban_db as kb
from hermes_cli.close_thread import build_close_thread_packet


FIXED_NOW = __import__("datetime", fromlist=["datetime"]).datetime(2026, 5, 24, 12, 0, 0,
    tzinfo=__import__("datetime", fromlist=["datetime"]).timezone.utc)


def _init_test_kanban(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    home = tmp_path / "hermes-home"
    office_root = tmp_path / "foundation-discord-office"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("FOUNDATION_DISCORD_OFFICE_ROOT", str(office_root))
    db_path = home / "kanban.db"
    kb.init_db(db_path)
    return home, db_path


def test_linear_comments_with_decision_keywords_score_as_decisions(tmp_path, monkeypatch):
    """Linear comments containing decision keywords should contribute 2 points
    to the decisions dimension, even when no Kanban task is attached.

    Regression test for: Discord thread 1509921015948513386 scored YELLOW because
    decisions=0 despite Linear having comments like "Decision: ship the close-thread fix."
    """
    _init_test_kanban(tmp_path, monkeypatch)

    # Simulate: thread has no Kanban task, but Linear issue has a decision comment
    packet, response = build_close_thread_packet(
        "/close-thread --mode close",
        source={
            "requested_by": "HD",
            "guild_id": "1",
            "channel_id": "123",
            "thread_id": "555",
            "thread_name": "Flywheel options trading",
            "message_id": "999",
        },
        session={"profile": "gateway", "command_surface": "discord_slash", "request_id": "999"},
        closeout_context={
            # No Kanban task — decisions must come from Linear
            "linear": {
                "comments": [
                    {
                        "author": "HD",
                        "body": "Decision: ship the close-thread fix as-is.",
                        "createdAt": "2026-05-29T14:08:00Z",
                    },
                    {
                        "author": "reviewer",
                        "body": "Approved — looks good to merge.",
                        "createdAt": "2026-05-29T15:00:00Z",
                    },
                ],
            },
            "thread_transcript": {
                "message_count": 2,
                "messages_captured": [
                    {"id": "1", "author": "HD", "content": "Summary: close-thread is working now."},
                ],
                "truncated": False,
            },
            "extracted": {},  # no decisions extracted from transcript
            "implementation_refs": {
                "changed_files": ["plugins/platforms/discord/adapter.py", "hermes_cli/close_thread.py"],
                "tests_run": [{"command": "pytest tests/hermes_cli/test_close_thread_linear_decisions.py -q", "result": "passed"}],
            },
            "task_scan": {"status": "no_followup_required"},
            "blocker_scan": {"status": "no_open_blockers"},
        },
        now=FIXED_NOW,
    )

    # The decisions dimension must credit Linear decision keywords
    assert packet["score"]["dimensions"]["decisions"] == 2, (
        f"Expected decisions=2 from Linear comment 'Decision: ship...', "
        f"got {packet['score']['dimensions']['decisions']}. "
        f"Full score: {packet['score']}"
    )

    # Overall score must be at least GREEN (18+) when Linear provides decisions
    assert packet["score"]["total"] >= 18, (
        f"Expected GREEN (18+), got {packet['score']['total']}/20. "
        f"Dims: {packet['score']['dimensions']}"
    )
    assert packet["status"] == "GREEN", (
        f"Expected GREEN, got {packet['status']} ({packet['score']['total']}/20)"
    )

    # Decisions list in packet must include the Linear decision text
    assert any(
        "ship the close-thread fix" in d for d in packet["summary"]["decisions"]
    ), f"No Linear decision text in packet decisions: {packet['summary']['decisions']}"


def test_linear_issue_state_transition_scores_as_decision(tmp_path, monkeypatch):
    """A Linear issue status transition (e.g. Done → Cancelled) should count
    as a decision when the thread uses Linear as its tracking layer.

    No Kanban task, no comments with keywords — but a status change happened.
    """
    _init_test_kanban(tmp_path, monkeypatch)

    packet, _response = build_close_thread_packet(
        "/close-thread --mode close",
        source={
            "requested_by": "HD",
            "thread_id": "555",
            "thread_name": "Flywheel options trading",
        },
        session={"profile": "gateway", "command_surface": "discord_slash"},
        closeout_context={
            "linear": {
                "issue_status": "Done",
                "issue_previous_status": "In Progress",
                "comments": [],  # no keyword comments, but status changed
            },
            "thread_transcript": {"message_count": 1, "messages_captured": [], "truncated": False},
            "extracted": {},
            "task_scan": {"status": "no_followup_required"},
            "blocker_scan": {"status": "no_open_blockers"},
        },
        now=FIXED_NOW,
    )

    # Status transition counts as a decision
    assert packet["score"]["dimensions"]["decisions"] >= 1, (
        f"Expected at least 1 decision from Linear status transition "
        f"(In Progress → Done), got {packet['score']['dimensions']['decisions']}"
    )


def test_no_decisions_when_linear_commslack_and_no_kanban(tmp_path, monkeypatch):
    """When Linear is present but has no relevant comments AND no Kanban task,
    decisions should still be 0 — we don't invent decisions from nothing.
    """
    _init_test_kanban(tmp_path, monkeypatch)

    packet, _response = build_close_thread_packet(
        "/close-thread --mode close",
        source={
            "requested_by": "HD",
            "thread_id": "555",
            "thread_name": "Idle thread",
        },
        session={"profile": "gateway", "command_surface": "discord_slash"},
        closeout_context={
            "linear": {
                "comments": [
                    {"author": "HD", "body": "Random note, nothing decisive.", "createdAt": "2026-05-29T14:00:00Z"},
                ],
            },
            "thread_transcript": {"message_count": 1, "messages_captured": [], "truncated": False},
            "extracted": {},
            "task_scan": {"status": "no_followup_required"},
            "blocker_scan": {"status": "no_open_blockers"},
        },
        now=FIXED_NOW,
    )

    # No keyword matches in Linear comments and no Kanban task = decisions still 0
    assert packet["score"]["dimensions"]["decisions"] == 0
