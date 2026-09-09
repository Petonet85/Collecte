#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Archive les previsions emises, puis les confronte a ce qui s'est passe.

Tout ce qu'on sait aujourd'hui de la justesse de la chaine vient de REJEUX :
on lui a redonne des crues passees en connaissant deja la pluie tombee. C'est
la seule chose qu'on puisse faire avant d'avoir vecu une crue avec l'outil,
mais cela surestime forcement la performance reelle — le rejeu ne contient
aucune erreur de prevision meteorologique.

Ce module archive donc chaque prevision au moment ou elle est emise, sans
retouche possible, puis la compare a l'observation quand celle-ci arrive. Au
bout d'une saison on saura dire « a 24 h d'echeance, la cote annoncee tombe
a ±X cm pres huit fois sur dix », mesure et non extrapolee.

Un fichier par mois, une ligne par passage, en JSON par lignes : le format se
relit sans rien installer, se concatene sans se corrompre, et un passage
interrompu ne peut pas abimer les precedents.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
DOSSIER = os.path.join(BASE, "donnees", "previsions")
QUANTILES = ("5", "10", "25", "50", "75", "90", "95")


def archiver(prevision: dict, observe_h: float | None = None) -> str:
    """Ajoute la prevision du moment a l'archive mensuelle. Idempotent."""
    t0 = str(prevision["date_prevision"])
    ligne = {
        "date_prevision": t0,
        "genere_le": prevision.get("genere_le")
        or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "time": prevision["time"],
        # Au millimetre : c'est deja au-dela de ce que la chaine sait dire, et
        # l'archive grossit de quatre passages par jour pendant des annees.
        "h_saint_laurent": {q: [round(float(v), 3) for v in prevision["h_saint_laurent"][q]]
                            for q in QUANTILES if q in prevision.get("h_saint_laurent", {})},
        "z_rochereau": {q: [round(float(v), 3) for v in prevision["z_rochereau"][q]]
                        for q in QUANTILES if q in prevision.get("z_rochereau", {})},
        "observe_h_t0": observe_h,
        "propagation": prevision.get("propagation"),
    }
    os.makedirs(DOSSIER, exist_ok=True)
    chemin = os.path.join(DOSSIER, f"{t0[:7]}.jsonl")

    deja = set()
    if os.path.exists(chemin):
        with open(chemin, encoding="utf-8") as fh:
            for l in fh:
                try:
                    deja.add(json.loads(l)["date_prevision"])
                except (ValueError, KeyError):
                    continue
    if t0 in deja:
        return chemin
    # Ajout en fin de fichier : une interruption ne peut pas abimer l'existant,
    # contrairement a une reecriture complete.
    with open(chemin, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(ligne, ensure_ascii=False) + "\n")
    return chemin


def lire(depuis: str | None = None) -> list[dict]:
    if not os.path.isdir(DOSSIER):
        return []
    out = []
    for nom in sorted(os.listdir(DOSSIER)):
        if not nom.endswith(".jsonl"):
            continue
        if depuis and nom[:7] < depuis[:7]:
            continue
        with open(os.path.join(DOSSIER, nom), encoding="utf-8") as fh:
            for l in fh:
                try:
                    out.append(json.loads(l))
                except ValueError:
                    continue
    return out


def _observations(depuis, jusqu_a) -> pd.Series:
    """Hauteur observee a Saint-Laurent, au pas horaire."""
    import hydroportail as hp
    s = hp.serie("M703243010", "H", pd.Timestamp(depuis).strftime("%d/%m/%Y"),
                 pd.Timestamp(jusqu_a).strftime("%d/%m/%Y"))
    return s.resample("1h").mean().dropna() if len(s) else pd.Series(dtype=float)


def verifier(depuis: str | None = None) -> pd.DataFrame:
    """Erreur de chaque prevision archivee, echeance par echeance."""
    runs = lire(depuis)
    if not runs:
        return pd.DataFrame()
    t_min = min(pd.Timestamp(r["date_prevision"].replace(" ", "T").rstrip("Z")) for r in runs)
    t_max = max(pd.Timestamp(r["time"][-1].replace(" ", "T").rstrip("Z")) for r in runs)
    obs = _observations(t_min, t_max + pd.Timedelta(days=1))
    if not len(obs):
        return pd.DataFrame()
    lignes = []
    for r in runs:
        t0 = pd.Timestamp(r["date_prevision"].replace(" ", "T").rstrip("Z"))
        idx = pd.to_datetime([t.replace(" ", "T").rstrip("Z") for t in r["time"]])
        med = np.asarray(r["h_saint_laurent"].get("50", []), dtype=float)
        b10 = np.asarray(r["h_saint_laurent"].get("10", []), dtype=float)
        b90 = np.asarray(r["h_saint_laurent"].get("90", []), dtype=float)
        if not len(med):
            continue
        reel = obs.reindex(idx)
        for k, t in enumerate(idx):
            if k >= len(med) or not np.isfinite(reel.iloc[k]):
                continue
            lignes.append({
                "date_prevision": str(t0), "echeance_h": int((t - t0).total_seconds() // 3600),
                "prevu_m": float(med[k]), "observe_m": float(reel.iloc[k]),
                "erreur_cm": 100 * (float(med[k]) - float(reel.iloc[k])),
                "dans_80": bool(len(b10) > k and len(b90) > k
                                and b10[k] <= reel.iloc[k] <= b90[k]),
            })
    return pd.DataFrame(lignes)


def bilan(df: pd.DataFrame) -> pd.DataFrame:
    """Score par tranche d'echeance : biais, dispersion, fiabilite du faisceau."""
    if df.empty:
        return df
    tr = pd.cut(df["echeance_h"], [0, 6, 12, 24, 48, 72],
                labels=["1-6 h", "7-12 h", "13-24 h", "25-48 h", "49-72 h"])
    g = df.groupby(tr, observed=True)
    return pd.DataFrame({
        "n": g.size(),
        "biais_cm": g["erreur_cm"].mean().round(1),
        "abs_median_cm": g["erreur_cm"].apply(lambda x: x.abs().median()).round(1),
        "rmse_cm": g["erreur_cm"].apply(lambda x: float(np.sqrt((x ** 2).mean()))).round(1),
        # Le faisceau a 80 % doit contenir l'observation 80 fois sur 100 : plus
        # bas il ment par optimisme, plus haut il est inutilement large.
        "dans_faisceau_80_pct": (100 * g["dans_80"].mean()).round(0),
    })


if __name__ == "__main__":
    import sys
    depuis = sys.argv[1] if len(sys.argv) > 1 else None
    runs = lire(depuis)
    print(f"{len(runs)} previsions archivees")
    if not runs:
        raise SystemExit("rien a verifier : l'archive se remplit a chaque passage")
    d = verifier(depuis)
    if d.empty:
        raise SystemExit("aucune echeance encore observee")
    print(f"{len(d)} couples prevu/observe\n")
    print(bilan(d).to_string())
    d.to_csv(os.path.join(BASE, "data", "verification_previsions.csv"), index=False)
