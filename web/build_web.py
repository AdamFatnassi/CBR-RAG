"""Assemble web/index.html from web/template.html + web/engine.js + web/data.json.

    python web/build_web.py
"""
from pathlib import Path

WEB = Path(__file__).resolve().parent
html = (WEB / "template.html").read_text(encoding="utf-8")
engine = (WEB / "engine.js").read_text(encoding="utf-8")
data = (WEB / "data.json").read_text(encoding="utf-8").replace("<", "\\u003c")
assert "/*__ENGINE__*/" in html and "__DATA__" in html
(WEB / "index.html").write_text(html.replace("/*__ENGINE__*/", engine).replace("__DATA__", data), encoding="utf-8")
print(f"web/index.html {(WEB / 'index.html').stat().st_size / 1e6:.2f} MB")
