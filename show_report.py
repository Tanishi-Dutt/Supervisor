"""Render one agent's incident report from the current session (Dev A).

    python3 show_report.py 6        # by agent number
    python3 show_report.py liar     # or by name
"""
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


def find_report(key: str, root: Path = Path(".")) -> Path | None:
    session_file = root / ".session"
    session = session_file.read_text().strip() if session_file.exists() else ""
    key = key.lower()
    matches = sorted(p for p in (root / "reports").glob(f"{session}-*")
                     if p.stem.split("-", 2)[1] == key or key in p.stem.split("-", 2)[2])
    return matches[0] if matches else None


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = find_report(sys.argv[1])
    if path is None:
        sys.exit(f"no report for {sys.argv[1]!r} in this session (looked in reports/)")
    text = path.read_text()
    if path.suffix == ".md":
        body = Markdown(text)
    elif path.suffix == ".json":
        body = Syntax(json.dumps(json.loads(text), indent=2), "json", word_wrap=True)
    else:
        body = text
    console.print(Panel(body, title=f"Black box — {path.name}", border_style="red"))


if __name__ == "__main__":
    main()
