"""Analyse frequentielle : periodes de retour a partir des maxima annuels."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

RETURN_PERIODS = (2, 5, 10, 20, 50, 100)


def gumbel_quantiles(q_daily: pd.Series, periods=RETURN_PERIODS,
                     peak_factor: float = 1.0) -> dict:
    """Ajuste une loi de Gumbel (moments) sur les maxima annuels du debit journalier.

    `peak_factor` corrige le passage du debit moyen journalier au debit de pointe
    instantane ; il est estime sur les crues de reference quand elles sont connues.
    """
    s = q_daily.dropna()
    if len(s) < 365 * 5:
        return {}
    # Annee hydrologique : 1er septembre, pour ne pas couper une crue d'hiver en deux.
    year = s.index.year + (s.index.month >= 9).astype(int)
    maxima = s.groupby(year).max()
    counts = s.groupby(year).size()
    maxima = maxima[counts > 300]
    if len(maxima) < 8:
        return {}
    x = maxima.to_numpy(dtype=float)
    sigma = x.std(ddof=1) * np.sqrt(6) / np.pi
    mu = x.mean() - 0.5772 * sigma
    out = {}
    for T in periods:
        y = -np.log(-np.log(1 - 1.0 / T))
        out[int(T)] = round(float((mu + sigma * y) * peak_factor), 1)
    out["n_annees"] = int(len(maxima))
    out["max_observe_journalier"] = round(float(x.max()), 1)
    return out


MOIS_FR = {"janvier": 1, "fevrier": 2, "février": 2, "mars": 3, "avril": 4, "mai": 5,
           "juin": 6, "juillet": 7, "aout": 8, "août": 8, "septembre": 9,
           "octobre": 10, "novembre": 11, "decembre": 12, "décembre": 12}


def _parse_date_fr(label: str) -> pd.Timestamp | None:
    """"Crue du 17 octobre 2024" -> Timestamp. Les libelles Vigicrues sont en clair."""
    m = re.search(r"(\d{1,2})\s+([A-Za-zéûôàè]+)\s+(\d{4})", str(label))
    if not m:
        return None
    mois = MOIS_FR.get(m.group(2).lower())
    if not mois:
        return None
    try:
        return pd.Timestamp(int(m.group(3)), mois, int(m.group(1)))
    except ValueError:
        return None


def estimate_peak_factor(q_daily: pd.Series, crues: pd.DataFrame) -> float:
    """Rapport debit de pointe / debit journalier maximum, d'apres les crues connues."""
    if crues is None or crues.empty or "q" not in crues.columns:
        return 1.0
    ratios = []
    for _, row in crues.iterrows():
        date = _parse_date_fr(row.get("libelle", ""))
        if date is None:
            continue
        window = q_daily.loc[str((date - pd.Timedelta(days=1)).date()):
                             str((date + pd.Timedelta(days=2)).date())]
        if len(window) and window.max() > 0:
            ratios.append(float(row["q"]) / float(window.max()))
    ratios = [r for r in ratios if 0.9 < r < 3.0]
    return float(np.clip(np.median(ratios), 1.0, 2.5)) if ratios else 1.0
