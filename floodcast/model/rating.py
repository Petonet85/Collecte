"""Courbe de tarage hauteur <-> debit, ajustee sur les couples (H, Q) temps reel.

Vigicrues raisonne en hauteur (c'est la grandeur mesuree et comprise du public),
le modele pluie-debit en debit. La courbe de tarage fait le lien, et elle est
reestimee sur les donnees recentes plutot que figee.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RatingCurve:
    a: float
    b: float
    h0: float
    r2: float
    h_min: float
    h_max: float
    n: int

    def to_q(self, h):
        h = np.asarray(h, dtype=float)
        return self.a * np.maximum(h - self.h0, 1e-6) ** self.b

    def to_h(self, q):
        q = np.maximum(np.asarray(q, dtype=float), 1e-9)
        return self.h0 + (q / self.a) ** (1.0 / self.b)

    def extrapolation_ratio(self, h) -> float:
        """Part d'extrapolation au-dela des hauteurs observees (indicateur de confiance)."""
        span = max(self.h_max - self.h_min, 1e-3)
        return float(max(0.0, (np.max(h) - self.h_max) / span))

    def to_dict(self) -> dict:
        return {"a": round(self.a, 4), "b": round(self.b, 3), "h0": round(self.h0, 3),
                "r2": round(self.r2, 4), "n": self.n,
                "plage_h": [round(self.h_min, 2), round(self.h_max, 2)]}


def fit(h: pd.Series, q: pd.Series) -> RatingCurve | None:
    """Ajuste Q = a (H - h0)^b : h0 par balayage, (a, b) par moindres carres en log."""
    df = pd.concat([h.rename("h"), q.rename("q")], axis=1).dropna()
    df = df[(df["q"] > 0)]
    if len(df) < 30:
        return None
    hv, qv = df["h"].to_numpy(), df["q"].to_numpy()
    best = None
    for h0 in np.linspace(hv.min() - 3.0 * (np.ptp(hv) + 0.05), hv.min() - 1e-3, 120):
        x = np.log(hv - h0)
        y = np.log(qv)
        if not np.all(np.isfinite(x)):
            continue
        b, la = np.polyfit(x, y, 1)
        pred = la + b * x
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf
        if 0.3 < b < 6.0 and (best is None or r2 > best[0]):
            best = (r2, float(np.exp(la)), float(b), float(h0))
    if best is None:
        return None
    r2, a, b, h0 = best
    return RatingCurve(a, b, h0, r2, float(hv.min()), float(hv.max()), len(df))
