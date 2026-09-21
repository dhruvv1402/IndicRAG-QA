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
