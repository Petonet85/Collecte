"""Calage des parametres GR par DDS (Tolson & Shoemaker, 2007).

DDS explore d'autant moins de dimensions a la fois que le budget avance : c'est
un algorithme sans derivee, robuste sur les modeles hydrologiques, et qui ne
demande aucune dependance externe.
"""
from __future__ import annotations

import numpy as np

from . import gr
from .metrics import kge, kge_sqrt


def mixed_objective(obs, sim) -> float:
    """Compromis dynamique complete / pointes : moitie KGE(sqrt Q), moitie KGE(Q)."""
    a, b = kge_sqrt(obs, sim), kge(obs, sim)
    if not np.isfinite(a) or not np.isfinite(b):
        return -np.inf
    return 0.5 * a + 0.5 * b

# Bornes de recherche en espace transforme (log/asinh), usage standard pour GR.
BOUNDS = np.array([
    [np.log(20.0), np.log(4000.0)],    # X1
    [-8.0, 8.0],                        # X2 (asinh)
    [np.log(5.0), np.log(1500.0)],      # X3
    [np.log(0.6), np.log(60.0)],        # X4 (en pas de temps)
    [np.log(0.70), np.log(1.90)],       # CP (correction de pluie)
])


def _to_params(u: np.ndarray) -> gr.GRParams:
    return gr.GRParams(float(np.exp(u[0])), float(np.sinh(u[1])),
                       float(np.exp(u[2])), float(np.exp(u[3])), float(np.exp(u[4])))


def _from_params(p: gr.GRParams) -> np.ndarray:
    return np.array([np.log(p.x1), np.arcsinh(p.x2), np.log(p.x3), np.log(p.x4),
                     np.log(p.cp)])


def calibrate(P, E, Qobs, area_km2: float, hours_per_step: float,
              warmup: int = 365, iters: int = 1200, seed: int = 0,
              x0: gr.GRParams | None = None, objective=None):
    """Cale (X1..X4) sur une serie observee. Renvoie (params, score, historique)."""
    objective = objective or mixed_objective
    P = np.asarray(P, dtype=float)
    E = np.asarray(E, dtype=float)
    Qobs = np.asarray(Qobs, dtype=float)
    warmup = min(warmup, max(len(P) // 4, 1))

    def score(u: np.ndarray) -> float:
        p = _to_params(u)
        q_mm = gr.run(P, E, p)
        q = gr.mm_to_m3s(q_mm, area_km2, hours_per_step)
        return objective(Qobs[warmup:], q[warmup:])

    rng = np.random.default_rng(seed)
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    best_u = _from_params(x0) if x0 is not None else lo + 0.5 * (hi - lo)
    best_u = np.clip(best_u, lo, hi)
    best = score(best_u)
    history = [best]

    r = 0.2
    for i in range(1, iters + 1):
        prob = 1.0 - np.log(i) / np.log(iters + 1)
        n_dim = len(lo)
        mask = rng.random(n_dim) < max(prob, 1.0 / n_dim)
        if not mask.any():
            mask[rng.integers(n_dim)] = True
        u = best_u.copy()
        u[mask] += r * (hi - lo)[mask] * rng.normal(size=mask.sum())
        # Reflexion aux bornes (DDS standard)
        u = np.where(u < lo, lo + (lo - u), u)
        u = np.where(u > hi, hi - (u - hi), u)
        u = np.clip(u, lo, hi)
        s = score(u)
        if s > best:
            best, best_u = s, u
        history.append(best)

    return _to_params(best_u), float(best), history


def daily_to_hourly(p_day: gr.GRParams) -> gr.GRParams:
    """Transpose des parametres journaliers au pas horaire.

    X1 et X3 sont des capacites (invariantes), X2 est un flux par pas de temps
    et X4 un temps exprime en nombre de pas. Le facteur 1/ (24**0.25) sur X4 est
    la correction empirique usuelle entre GR4J et GR4H (Ficchi et al., 2016).
    """
    return gr.GRParams(
        x1=p_day.x1,
        x2=p_day.x2 / 24.0,
        x3=p_day.x3,
        x4=float(np.clip(p_day.x4 * 24.0 / (24.0 ** 0.25), 1.0, 200.0)),
        cp=p_day.cp,
    )


def refine_hourly(P, E, Qobs, area_km2: float, p_start: gr.GRParams,
                  iters: int = 300, seed: int = 1):
    """Reajuste X3/X4 (dynamique de crue) au pas horaire, X1/X2 restant du calage long."""
    P, E, Qobs = np.asarray(P, float), np.asarray(E, float), np.asarray(Qobs, float)
    lo = np.array([np.log(max(p_start.x3 * 0.3, 5)), np.log(max(p_start.x4 * 0.35, 1.0))])
    hi = np.array([np.log(p_start.x3 * 3.0), np.log(min(p_start.x4 * 2.5, 200))])
    rng = np.random.default_rng(seed)

    def score(u):
        p = gr.GRParams(p_start.x1, p_start.x2, float(np.exp(u[0])),
                        float(np.exp(u[1])), p_start.cp)
        q = gr.mm_to_m3s(gr.run(P, E, p), area_km2, 1.0)
        n = max(len(Qobs) // 6, 24)
        return kge_sqrt(Qobs[n:], q[n:])

    best_u = np.array([np.log(p_start.x3), np.log(p_start.x4)])
    best = score(best_u)
    for i in range(1, iters + 1):
        prob = 1.0 - np.log(i) / np.log(iters + 1)
        mask = rng.random(2) < max(prob, 0.5)
        if not mask.any():
            mask[rng.integers(2)] = True
        u = best_u.copy()
        u[mask] += 0.2 * (hi - lo)[mask] * rng.normal(size=mask.sum())
        u = np.clip(u, lo, hi)
        s = score(u)
        if s > best:
            best, best_u = s, u
    return gr.GRParams(p_start.x1, p_start.x2, float(np.exp(best_u[0])),
                       float(np.exp(best_u[1])), p_start.cp), float(best)


def refit_forcing(P, E, Qobs, p_base: gr.GRParams, area_km2: float,
                  hours_per_step: float = 24.0, warmup: int = 120,
                  iters: int = 250, seed: int = 3):
    """Reajuste le modele au forcage operationnel (runs Meteo-France archives).

    Le calage structurel se fait sur 20 ans de reanalyse ERA5, seule serie assez
    longue. Mais l'operationnel tourne avec AROME/ARPEGE, dont les cumuls en
    episode convectif sont tres differents : utiliser tels quels des parametres
    cales sur ERA5 revient a exploiter le modele hors de son domaine de calage.
    On ne rejoue donc que deux degres de liberte — la correction de pluie CP et
    la capacite de production X1 — ce qui suffit a resorber le decalage de forcage
    sans risquer de surajuster sur une periode courte.
    """
    P, E, Qobs = np.asarray(P, float), np.asarray(E, float), np.asarray(Qobs, float)
    warmup = min(warmup, max(len(P) // 4, 1))
    lo = np.array([np.log(0.45), np.log(p_base.x1 * 0.6)])
    hi = np.array([np.log(2.20), np.log(p_base.x1 * 1.6)])
    rng = np.random.default_rng(seed)

    def build(u):
        return gr.GRParams(float(np.exp(u[1])), p_base.x2, p_base.x3, p_base.x4,
                           float(np.exp(u[0])))

    def score(u):
        q = gr.mm_to_m3s(gr.run(P, E, build(u)), area_km2, hours_per_step)
        return mixed_objective(Qobs[warmup:], q[warmup:])

    best_u = np.clip(np.array([np.log(p_base.cp), np.log(p_base.x1)]), lo, hi)
    best = score(best_u)
    for i in range(1, iters + 1):
        prob = 1.0 - np.log(i) / np.log(iters + 1)
        mask = rng.random(2) < max(prob, 0.5)
        if not mask.any():
            mask[rng.integers(2)] = True
        u = best_u.copy()
        u[mask] += 0.25 * (hi - lo)[mask] * rng.normal(size=mask.sum())
        u = np.clip(u, lo, hi)
        s = score(u)
        if s > best:
            best, best_u = s, u
    return build(best_u), float(best)
