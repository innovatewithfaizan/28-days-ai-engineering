"""Rebuild the progress table in README.md from the week checklists.

Counts `- [ ]` / `- [x]` lines under each week's "## Deliverables" heading
and rewrites the block between the progress markers in the root README.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
START, END = "<!-- progress:start -->", "<!-- progress:end -->"
BOX = re.compile(r"^\s*- \[( |x|X)\]")


def week_stats(readme: Path) -> tuple[str, int, int]:
    text = readme.read_text(encoding="utf-8")
    title = text.splitlines()[0].lstrip("# ").split(": ", 1)[-1]
    section = text.split("## Deliverables", 1)[-1].split("\n## ", 1)[0]
    boxes = [m.group(1) for line in section.splitlines() if (m := BOX.match(line))]
    return title, sum(b.lower() == "x" for b in boxes), len(boxes)


def bar(done: int, total: int, width: int = 10) -> str:
    filled = round(width * done / total) if total else 0
    return "█" * filled + "░" * (width - filled)


def main() -> None:
    rows, all_done, all_total = [], 0, 0
    for i, folder in enumerate(sorted(ROOT.glob("week-*")), start=1):
        title, done, total = week_stats(folder / "README.md")
        all_done, all_total = all_done + done, all_total + total
        mark = " ✅" if total and done == total else ""
        rows.append(f"| {i} | [{title}]({folder.name}) | {bar(done, total)} {done}/{total}{mark} |")

    table = "\n".join([
        START,
        "| Week | Theme | Done |",
        "|---|---|---|",
        *rows,
        f"| | **Total** | **{all_done}/{all_total}** |",
        END,
    ])
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), table, text, flags=re.S)
    if new != text:
        readme.write_text(new, encoding="utf-8")
        print(f"Progress updated: {all_done}/{all_total}")
    else:
        print("Progress unchanged")


if __name__ == "__main__":
    main()
