#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reconstruit docs/index.html a partir du gabarit et des dernieres donnees.

Utile quand seule la mise en page change : prevoir.py refait sinon six minutes
d'appels a Open-Meteo, qui facture un appel par point de grille, pour aboutir
aux memes chiffres.
"""
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(BASE, "docs")

with open(os.path.join(DOCS, "tableau.json"), encoding="utf-8") as fh:
    donnees = json.load(fh)
with open(os.path.join(DOCS, "gabarit_tableau.html"), encoding="utf-8") as fh:
    page = fh.read()
page = page.replace("__DATA__", json.dumps(donnees, ensure_ascii=False, separators=(",", ":")))
with open(os.path.join(DOCS, "index.html"), "w", encoding="utf-8") as fh:
    fh.write(page)
print("docs/index.html reconstruit —", donnees["meta"]["genere_le"])
