"""research/FINAL_REPORT.md -> research/FINAL_REPORT.html: one self-contained file, identical content.

Inline CSS only (no scripts, no external requests), light/dark via prefers-color-scheme, tables scroll sideways inside
their own box on a phone.

    .venv/bin/python scripts/research/final_report_html.py [--src research/FINAL_REPORT.md] [--out research/FINAL_REPORT.html]
"""
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[2]

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1f;--muted:#5f6368;--line:#dcd9d2;--head:#f0ede6;--zebra:#f6f4ef;--accent:#1f5fa8;
--code:#f1efe9;--good:#1b7a3d;--bad:#b3261e;--eq:#8a6d00}
@media (prefers-color-scheme:dark){:root{--bg:#141517;--fg:#e8e6e3;--muted:#a3a7ad;--line:#34373c;--head:#1f2226;
--zebra:#1a1c1f;--accent:#7fb2ff;--code:#202327;--good:#6fcf8f;--bad:#ff8a80;--eq:#e6c65c}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
"Helvetica Neue",Arial,sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.65rem;line-height:1.25;margin:.2em 0 .6em}
h2{font-size:1.3rem;margin:2.2em 0 .6em;padding-top:.6em;border-top:1px solid var(--line)}
h3{font-size:1.08rem;margin:1.6em 0 .5em}
p,li{max-width:75ch}
a{color:var(--accent)}
code{font:.88em/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--code);padding:.08em .3em;
border-radius:4px;overflow-wrap:anywhere}
pre{background:var(--code);padding:12px 14px;border-radius:8px;overflow-x:auto;font-size:.85rem;line-height:1.45}
pre code{background:none;padding:0;overflow-wrap:normal}
.tw{overflow-x:auto;margin:1em 0;border:1px solid var(--line);border-radius:8px;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:.86rem;line-height:1.4}
th,td{padding:7px 9px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}
th{background:var(--head);font-weight:600;position:sticky;top:0}
tbody tr:nth-child(even){background:var(--zebra)}
tbody tr:last-child td{border-bottom:none}
td strong{font-weight:650}
.v-better{color:var(--good);font-weight:650}.v-worse{color:var(--bad);font-weight:650}.v-equal{color:var(--eq);font-weight:650}
nav.toc{border:1px solid var(--line);border-radius:8px;padding:10px 16px;margin:1em 0 2em;font-size:.92rem}
nav.toc ul{margin:.3em 0;padding-left:1.1em}
@media (max-width:600px){body{font-size:15px}table{font-size:.8rem}th,td{padding:6px 7px}h1{font-size:1.35rem}}
@media print{.tw{overflow:visible}th{position:static}}
"""


def build(src: Path) -> str:
    text = src.read_text()
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"], extension_configs={"toc": {"toc_depth": "2"}})
    body = md.convert(text).replace("\\|", "|")  # escaped pipes inside table code spans
    body = re.sub(r"<table>", '<div class="tw"><table>', body)
    body = body.replace("</table>", "</table></div>")
    # colour the verdict words in table cells only
    for word, cls in (("better", "v-better"), ("worse", "v-worse"), ("equal", "v-equal")):
        body = re.sub(rf"(<td>)(\s*<strong>)?({word})\b", lambda m, c=cls: f'{m.group(1)}{m.group(2) or ""}<span class="{c}">{m.group(3)}</span>', body, flags=re.I)
    title = html.escape(re.search(r"^# (.+)$", text, re.M).group(1))
    toc = f'<nav class="toc"><strong>Contents</strong>{md.toc}</nav>'
    body = body.replace("</h1>", "</h1>" + toc, 1)
    return (f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n<meta name="color-scheme" content="light dark">\n'
            f"<title>Final Report</title>\n<meta name=\"description\" content=\"{title}\">\n<style>{CSS}</style>\n</head>\n"
            f"<body>\n<main>\n{body}\n</main>\n</body>\n</html>\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=str(ROOT / "research/FINAL_REPORT.md"))
    ap.add_argument("--out", default=str(ROOT / "research/FINAL_REPORT.html"))
    a = ap.parse_args()
    Path(a.out).write_text(build(Path(a.src)))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
