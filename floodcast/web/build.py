"""Generation du tableau de bord HTML autonome (aucune dependance externe)."""
from __future__ import annotations

import json
import os

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "template.html")


def build(result: dict, out_path: str) -> str:
    with open(TEMPLATE, encoding="utf-8") as fh:
        html = fh.read()
    title = f"{result['meta'].get('cours_eau') or ''} — {result['meta']['station']}".strip(" —")
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False, default=str)
    html = html.replace("__TITLE__", title).replace("__DATA__", payload)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out_path
