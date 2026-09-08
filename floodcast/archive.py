"""Archive locale des observations temps reel.

Hub'Eau ne conserve que 30 jours d'observations au pas fin : impossible d'y
trouver une crue la plupart du temps. En accumulant cette fenetre a chaque
execution, on se constitue un historique horaire qui permet, au fil des mois,
de caler la propagation amont et l'hydrogramme unitaire sur de vrais evenements.
C'est le composant qui fait progresser l'outil tout seul.
"""
from __future__ import annotations

import os

import pandas as pd

ARCHIVE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "archive")


def _path(key: str) -> str:
    safe = key.replace("/", "_").replace(" ", "_")
    return os.path.join(ARCHIVE_DIR, f"{safe}.csv")


def update(key: str, series: pd.Series) -> pd.Series:
    """Fusionne une serie fraiche avec l'archive et renvoie l'union complete."""
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    merged = series.dropna()
    path = _path(key)
    if os.path.exists(path):
        old = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        merged = pd.concat([old, merged])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.to_frame("value").to_csv(path)
    return merged


def load(key: str) -> pd.Series:
    path = _path(key)
    if not os.path.exists(path):
        return pd.Series(dtype=float)
    ser = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    return ser.dropna().sort_index()


def summary() -> pd.DataFrame:
    """Etat de l'archive : profondeur disponible par serie."""
    if not os.path.isdir(ARCHIVE_DIR):
        return pd.DataFrame()
    rows = []
    for fn in sorted(os.listdir(ARCHIVE_DIR)):
        if not fn.endswith(".csv"):
            continue
        ser = load(fn[:-4])
        if ser.empty:
            continue
        rows.append({"serie": fn[:-4], "debut": ser.index[0], "fin": ser.index[-1],
                     "n_pas": len(ser), "jours": round((ser.index[-1] - ser.index[0]).days, 1)})
    return pd.DataFrame(rows)
