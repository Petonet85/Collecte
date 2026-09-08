"""Assimilation : correction des erreurs recentes du modele et incertitude.

Un modele pluie-debit derive lentement (etat des reservoirs, biais de pluie).
Les dernieres heures observees disent dans quel sens il se trompe : on prolonge
cette erreur en la faisant decroitre selon son autocorrelation estimee.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ErrorModel:
    rho: float            # autocorrelation a 1 h des residus (log)
    e0: float             # dernier residu observe (log)
    sigma: float          # ecart-type des residus (log)
    n: int

    def correction(self, horizons: np.ndarray) -> np.ndarray:
        """Facteur multiplicatif a appliquer au debit simule, par echeance."""
        return np.exp(self.e0 * self.rho ** np.asarray(horizons, dtype=float))

    def spread(self, horizons: np.ndarray) -> np.ndarray:
        """Ecart-type (log) de l'erreur residuelle, croissant avec l'echeance."""
        h = np.asarray(horizons, dtype=float)
        var = self.sigma ** 2 * (1 - self.rho ** (2 * h))
        return np.sqrt(np.maximum(var, (0.15 * self.sigma) ** 2))

    def to_dict(self) -> dict:
        return {"rho_1h": round(self.rho, 4),
                "biais_actuel_pct": round(100 * (float(np.exp(self.e0)) - 1), 1),
                "sigma_log": round(self.sigma, 4), "n_pas": self.n}


def fit(q_obs: np.ndarray, q_sim: np.ndarray, min_q: float = 1e-3,
        sigma_floor: float = 0.0) -> ErrorModel:
    """Estime un AR(1) sur les residus logarithmiques des dernieres heures.

    `sigma_floor` est un plancher d'incertitude issu du rejeu long terme. Sans lui,
    une fenetre d'etiage calme produit une dispersion quasi nulle et un faisceau de
    prevision faussement etroit : l'erreur du modele en crue n'a rien a voir avec
    son erreur en basses eaux.
    """
    o = np.asarray(q_obs, dtype=float)
    s = np.asarray(q_sim, dtype=float)
    m = np.isfinite(o) & np.isfinite(s) & (o > min_q) & (s > min_q)
    if m.sum() < 12:
        return ErrorModel(rho=0.9, e0=0.0, sigma=max(0.35, sigma_floor), n=int(m.sum()))
    e = np.log(o[m]) - np.log(s[m])
    e_c = e - e.mean()
    denom = float((e_c[:-1] ** 2).sum())
    rho = float((e_c[:-1] * e_c[1:]).sum() / denom) if denom > 0 else 0.9
    rho = float(np.clip(rho, 0.0, 0.995))
    sigma = float(max(e.std(), sigma_floor, 1e-3))
    return ErrorModel(rho=rho, e0=float(e[-1]), sigma=sigma, n=int(m.sum()))
