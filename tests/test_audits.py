"""Run the repository audits as tests, so they gate rather than wait to be run.

Both scripts exist because of defects that had already reached the paper by the
time anyone noticed: functions written and never wired, and a documented CLI
surface that did not match the real one. An audit nobody remembers to run would
have caught neither.

They are invoked as subprocesses rather than imported, so what is tested is the
entry point a person actually types.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        encoding="utf-8",
        errors="replace",
    )


def test_no_public_code_is_left_unwired():
    """Six pieces of working code were found unreachable in one session, two of
    them feeding numbers the paper quotes. None failed loudly -- the report
    printed a dash, a zero, or a fallback."""
    result = _run("audit-reachability.py")
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_documented_cli_matches_the_real_one():
    """§18 has drifted both ways: commands documented but never written, and
    commands written but never documented."""
    result = _run("audit-cli-docs.py")
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_committed_report_names_its_producer():
    """PRD NFR-6: every reported table regenerates from a command. Before
    scripts/regenerate-reports.py, which command made which report was
    recorded nowhere, and §VI quoted a section no committed report contained.
    A new report has to be added to that table -- as a producer, or as legacy
    with the reason -- or this fails."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("regen", ROOT / "scripts" / "regenerate-reports.py")
    regen = importlib.util.module_from_spec(spec)
    sys.modules["regen"] = regen  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(regen)

    declared = {r.name for r in regen.REPORTS}
    committed = {p.name for p in (ROOT / "evals").glob("report-*.txt")}
    assert committed - declared == set(), f"no producer declared for {sorted(committed - declared)}"
    assert declared - committed == set(), f"declared but not committed: {sorted(declared - committed)}"
    assert all(r.note for r in regen.REPORTS if r.argv is None), "a legacy report must say why"
