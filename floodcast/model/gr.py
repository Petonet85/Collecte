"""Modele pluie-debit conceptuel de type GR (GR4J journalier / GR4H horaire).

Structure classique a 4 parametres (Perrin et al., 2003) :
  X1 (mm)   capacite du reservoir de production
  X2 (mm/pas) coefficient d'echange souterrain
  X3 (mm)   capacite du reservoir de routage
  X4 (pas)  temps de base de l'hydrogramme unitaire

Le meme code sert aux deux pas de temps : seule l'unite de X2 et de X4 change.
Le pas horaire est celui qui compte pour la crue, le pas journalier sert au calage
long terme (on dispose de 20+ ans de debits journaliers, contre 30 jours d'horaire).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GRParams:
    x1: float = 350.0
    x2: float = 0.0
    x3: float = 90.0
    x4: float = 1.7
    cp: float = 1.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x1, self.x2, self.x3, self.x4, self.cp], dtype=float)

    @staticmethod
    def from_array(a) -> "GRParams":
        a = list(a) + [1.0]
        return GRParams(float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4]))

    def to_dict(self) -> dict:
        return {"X1": round(self.x1, 1), "X2": round(self.x2, 3),
                "X3": round(self.x3, 1), "X4": round(self.x4, 2),
                "CP": round(self.cp, 3)}


@dataclass
class GRState:
    s: float          # reservoir de production (mm)
    r: float          # reservoir de routage (mm)
    uh1: np.ndarray   # retards en cours dans l'hydrogramme unitaire 1
    uh2: np.ndarray

    def copy(self) -> "GRState":
        return GRState(self.s, self.r, self.uh1.copy(), self.uh2.copy())


def _uh_ordinates(x4: float) -> tuple[np.ndarray, np.ndarray]:
    """Ordonnees des deux hydrogrammes unitaires (UH1 sur X4, UH2 sur 2*X4)."""
    x4 = max(x4, 0.5)
    n1 = int(np.ceil(x4))
    n2 = int(np.ceil(2 * x4))

    def sh1(t):
        t = np.asarray(t, dtype=float)
        return np.where(t <= 0, 0.0, np.where(t < x4, (t / x4) ** 2.5, 1.0))

    def sh2(t):
        t = np.asarray(t, dtype=float)
        u = np.clip(t / x4, 0.0, 2.0)
        out = np.where(t <= 0, 0.0, 0.5 * u ** 2.5)
        mid = (t > x4) & (t < 2 * x4)
        out = np.where(mid, 1 - 0.5 * np.maximum(2 - u, 0.0) ** 2.5, out)
        return np.where(t >= 2 * x4, 1.0, out)

    t1 = np.arange(1, n1 + 1)
    t2 = np.arange(1, n2 + 1)
    return np.diff(sh1(np.r_[0, t1])), np.diff(sh2(np.r_[0, t2]))


def initial_state(p: GRParams) -> GRState:
    o1, o2 = _uh_ordinates(p.x4)
    return GRState(0.5 * p.x1, 0.4 * p.x3, np.zeros(len(o1)), np.zeros(len(o2)))


def run(P: np.ndarray, E: np.ndarray, p: GRParams,
        state: GRState | None = None, return_state: bool = False,
        return_trace: bool = False):
    """Deroule le modele. P, E en mm/pas ; renvoie Q en mm/pas.

    `return_trace` renvoie en plus l'etat des reservoirs a chaque pas : c'est ce
    qui permet de relancer une prevision depuis n'importe quelle date passee, donc
    de rejouer honnetement la chaine sur des crues deja survenues.
    """
    P = np.nan_to_num(np.asarray(P, dtype=float), nan=0.0) * max(p.cp, 0.05)
    E = np.nan_to_num(np.asarray(E, dtype=float), nan=0.0)
    o1, o2 = _uh_ordinates(p.x4)
    st = state.copy() if state is not None else initial_state(p)
    if len(st.uh1) != len(o1):
        st.uh1 = np.resize(st.uh1, len(o1)) * 0
    if len(st.uh2) != len(o2):
        st.uh2 = np.resize(st.uh2, len(o2)) * 0

    x1, x2, x3 = max(p.x1, 10.0), p.x2, max(p.x3, 5.0)
    s, r = st.s, st.r
    uh1, uh2 = st.uh1, st.uh2
    q = np.empty(len(P))
    trace: list[GRState] = []

    for i in range(len(P)):
        pi, ei = P[i], E[i]
        # --- Interception / bilan net
        if pi >= ei:
            pn, en = pi - ei, 0.0
            ratio = np.tanh(min(pn / x1, 13.0))
            sr = s / x1
            ps = x1 * (1 - sr * sr) * ratio / (1 + sr * ratio)
            es = 0.0
        else:
            pn, en = 0.0, ei - pi
            ratio = np.tanh(min(en / x1, 13.0))
            sr = s / x1
            es = s * (2 - sr) * ratio / (1 + (1 - sr) * ratio)
            ps = 0.0
        # La formulation en tangente hyperbolique garantit deja S <= X1 ; le clip
        # n'est qu'une securite numerique et doit donc borner a X1 exactement,
        # sinon il detruit de l'eau a chaque pas ou le reservoir est plein.
        s = min(max(s - es + ps, 0.0), x1)

        # --- Percolation
        perc = s * (1 - (1 + (s / (2.25 * x1)) ** 4) ** -0.25)
        s -= perc
        pr = perc + (pn - ps)

        # --- Hydrogrammes unitaires
        uh1 = np.roll(uh1, -1)
        uh1[-1] = 0.0
        uh1 += 0.9 * pr * o1
        uh2 = np.roll(uh2, -1)
        uh2[-1] = 0.0
        uh2 += 0.1 * pr * o2

        # --- Routage + echange souterrain
        rr = r / x3
        exch = x2 * rr ** 3.5
        r = max(r + uh1[0] + exch, 0.0)
        qr = r * (1 - (1 + (r / x3) ** 4) ** -0.25)
        r -= qr
        qd = max(uh2[0] + exch, 0.0)
        q[i] = qr + qd
        if return_trace:
            trace.append(GRState(s, r, uh1.copy(), uh2.copy()))

    if return_trace:
        return (q, trace, GRState(s, r, uh1, uh2)) if return_state else (q, trace)
    if return_state:
        return q, GRState(s, r, uh1, uh2)
    return q


def mm_to_m3s(q_mm: np.ndarray, area_km2: float, hours_per_step: float) -> np.ndarray:
    """mm/pas -> m3/s pour un bassin de surface donnee."""
    return np.asarray(q_mm) * area_km2 * 1000.0 / (3600.0 * hours_per_step)


def m3s_to_mm(q_m3s, area_km2: float, hours_per_step: float):
    return np.asarray(q_m3s) * 3600.0 * hours_per_step / (area_km2 * 1000.0)
