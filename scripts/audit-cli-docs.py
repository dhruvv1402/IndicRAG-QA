"""Check that the documented CLI surface matches the one that exists.

`docs/ARCHITECTURE.md` §18 has drifted in both directions. It once listed four
commands that were never written -- `index build` among them, so the embeddings
every retrieval number depends on could not be rebuilt by anything committed --
and then, once those were added, it went stale the other way and omitted five
that did exist.

Neither direction fails loudly. A documented command that does not exist is
found by the person who tries it; an undocumented one is simply never used.

    python scripts/audit-cli-docs.py

Exits non-zero on a mismatch.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "ARCHITECTURE.md"
SECTION = "## 18. CLI surface"

sys.path.insert(0, str(ROOT / "src"))


def documented() -> set[str]:
    """Command paths named in the §18 fenced block, e.g. {"eval qa", "ask"}."""
    text = DOC.read_text(encoding="utf-8")
    start = text.index(SECTION)
    end = text.index("\n## ", start + 1)
    block = text[start:end]

    found: set[str] = set()
    for line in block.splitlines():
        match = re.match(r"^indicrag\s+([a-z-]+)(?:\s+([a-z-]+))?", line.strip())
        if not match:
            continue
        group, sub = match.group(1), match.group(2)
        # A second word is a subcommand only when it is not an argument or flag;
        # `indicrag ask "<query>"` has none.
        found.add(f"{group} {sub}" if sub else group)
    return found


def implemented() -> set[str]:
    from indicrag.cli import app

    found = {c.name for c in app.registered_commands if c.name}
    for group in app.registered_groups:
        instance = group.typer_instance
        if instance is None:
            continue
        for command in instance.registered_commands:
            if command.name:
                found.add(f"{group.name} {command.name}")
    return found


def main() -> int:
    doc, code = documented(), implemented()

    missing_from_docs = sorted(code - doc)
    missing_from_code = sorted(doc - code)

    print(f"{len(code)} commands implemented, {len(doc)} documented in §18")
    print()

    if not missing_from_docs and not missing_from_code:
        print("The documented CLI surface matches the implemented one.")
        return 0

    if missing_from_code:
        print("Documented but NOT implemented -- someone will try these and fail:")
        for name in missing_from_code:
            print(f"  indicrag {name}")
        print()

    if missing_from_docs:
        print("Implemented but NOT documented -- nobody will know these exist:")
        for name in missing_from_docs:
            print(f"  indicrag {name}")
        print()

    print(f"Update {DOC.relative_to(ROOT)} §18, or the CLI, so the two agree.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
