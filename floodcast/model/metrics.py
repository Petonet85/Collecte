"""Criteres de performance hydrologiques et probabilistes."""
from __future__ import annotations

import numpy as np


def _clean(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    m = np.isfinite(obs) & np.isfinite(sim)
    return obs[m], sim[m]


def nse(obs, sim) -> float:
    obs, sim = _clean(obs, sim)
    if len(obs) < 3:
        return -np.inf
    denom = ((obs - obs.mean()) ** 2).sum()
    return float(1 - ((sim - obs) ** 2).sum() / denom) if denom > 0 else -np.inf


def kge(obs, sim) -> float:
    """Kling-Gupta (2009) : correlation, biais de variance, biais de moyenne."""
    obs, sim = _clean(obs, sim)
    if len(obs) < 3 or obs.std() == 0 or sim.std() == 0:
        return -np.inf
    r = float(np.corrcoef(obs, sim)[0, 1])
    alpha = float(sim.std() / obs.std())
    beta = float(sim.mean() / obs.mean()) if obs.mean() != 0 else np.inf
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def kge_sqrt(obs, sim) -> float:
    """KGE sur racine des debits : compromis crues / etiages, standard en calage."""
    obs, sim = _clean(obs, sim)
    return kge(np.sqrt(np.maximum(obs, 0)), np.sqrt(np.maximum(sim, 0)))


def crps_ensemble(obs: float, members: np.ndarray) -> float:
    """CRPS d'une prevision d'ensemble (formule par decomposition, O(n log n))."""
    x = np.sort(np.asarray(members, dtype=float))
    n = len(x)
    if n == 0 or not np.isfinite(obs):
        return np.nan
    mae = np.abs(x - obs).mean()
    spread = (2.0 / (n * n)) * np.sum((2 * np.arange(1, n + 1) - n - 1) * x)
    return float(mae - 0.5 * spread)


def pit(obs: float, members: np.ndarray) -> float:
    """Rang de l'observation dans l'ensemble : diagnostic de fiabilite."""
    x = np.asarray(members, dtype=float)
    return float((x <= obs).mean()) if len(x) else np.nan


def contingency(obs, sim, threshold: float) -> dict:
    """Detection d'un depassement de seuil : POD, FAR, CSI."""
    obs, sim = _clean(obs, sim)
    o, s = obs >= threshold, sim >= threshold
    hits = int((o & s).sum())
    misses = int((o & ~s).sum())
    fa = int((~o & s).sum())
    return {
        "hits": hits, "misses": misses, "fausses_alertes": fa,
        "POD": round(hits / (hits + misses), 3) if hits + misses else None,
        "FAR": round(fa / (hits + fa), 3) if hits + fa else None,
        "CSI": round(hits / (hits + misses + fa), 3) if hits + misses + fa else None,
    }
