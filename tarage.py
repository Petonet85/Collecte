#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Courbe hauteur-debit empirique, pour afficher l'amont en hauteur d'echelle.

Le modele produit des debits ; l'oeil, lui, lit des hauteurs — c'est ce que
donne la station, c'est ce qu'on voit sur le terrain, et c'est la seule
grandeur comparable d'une station a l'autre sans conversion mentale.

La conversion ne peut pas passer par une loi de puissance unique. A
Saint-Mesmin, passer de 1 a 10 m3/s fait monter l'eau de 0,37 a 1,29 m, mais
passer de 30 a 200 m3/s ne la fait monter que de 2,32 a 3,02 m : le lit
deborde et la section change de nature. Un seul exposant ajuste sur cette
plage donne 14 cm d'erreur en etiage a l'Ouin, ce qui est enorme pour une
riviere dont la hauteur d'etiage vaut 10 cm.

On construit donc la courbe directement sur les couples (H, Q) publies par la
station elle-meme : les maxima mensuels sur trente ans, qui couvrent toute la
gamme des crues, et les mesures temps reel des trente derniers jours, qui
epinglent l'etiage du moment. Mediane par tranche de debit, monotonie imposee,
interpolation lineaire en log-debit. Au-dela du plus fort couple observe, on
prolonge par la pente locale, faute de mieux — et la page le signale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_PAR_TRANCHE = 3


class CourbeHauteur:
    """H(Q) monotone, tabulee. `hors_gamme` borne le domaine mesure."""

    def __init__(self, q: np.ndarray, h: np.ndarray, source: str):
        self.q, self.h, self.source = q, h, source

    @property
    def domaine(self) -> tuple[float, float]:
        return float(self.q[0]), float(self.q[-1])

    def to_h(self, q):
        q = np.asarray(q, dtype=float)
        lq = np.log10(np.clip(q, 1e-4, None))
        h = np.interp(lq, np.log10(self.q), self.h)
        # Au-dessus du domaine mesure : prolongement par la pente du haut de courbe.
        haut = q > self.q[-1]
        if haut.any() and len(self.q) >= 2:
            pente = (self.h[-1] - self.h[-2]) / (np.log10(self.q[-1]) - np.log10(self.q[-2]))
            h = np.where(haut, self.h[-1] + pente * (lq - np.log10(self.q[-1])), h)
        return h


def _couples(site: str, code: str, jours_tr: int = 30) -> pd.DataFrame:
    """Tous les couples (H, Q) simultanes que publie la station."""
    from floodcast import sevre
    from floodcast.sources import hubeau as hb

    lots = []
    mens = pd.DataFrame({"H": sevre._elabore(site, "HIXM"),
                         "Q": sevre._elabore(site, "QIXM")}).dropna()
    lots.append(mens)
    try:
        tr = pd.DataFrame({"H": hb.hourly(hb.observations_tr(code, "H", jours_tr)),
                           "Q": hb.hourly(hb.observations_tr(code, "Q", jours_tr))}).dropna()
        if len(tr):
            lots.append(tr)
    except Exception:
        pass
    df = pd.concat(lots)
    return df[(df["Q"] > 0) & np.isfinite(df["H"])]


def construire(site: str, code: str, n_tranches: int = 22) -> CourbeHauteur:
    df = _couples(site, code)
    if len(df) < 20:
        raise RuntimeError(f"trop peu de couples (H, Q) a {code}")
    lq = np.log10(df["Q"].to_numpy())
    bords = np.linspace(lq.min(), lq.max(), n_tranches + 1)
    idx = np.clip(np.digitize(lq, bords) - 1, 0, n_tranches - 1)
    qs, hs = [], []
    for k in range(n_tranches):
        m = idx == k
        if m.sum() < MIN_PAR_TRANCHE:
            continue
        qs.append(float(np.median(df["Q"].to_numpy()[m])))
        hs.append(float(np.median(df["H"].to_numpy()[m])))
    q = np.array(qs)
    # La hauteur croit avec le debit : la mediane par tranche peut violer cette
    # evidence physique la ou les couples sont rares, on la retablit.
    h = np.maximum.accumulate(np.array(hs))
    ordre = np.argsort(q)
    return CourbeHauteur(q[ordre], h[ordre], f"{len(df)} couples (H, Q) publiés par la station")


def diagnostic(courbe: CourbeHauteur, site: str, code: str) -> dict:
    """Ecart de la courbe aux mesures temps reel, qu'elle n'a pas vues seules."""
    from floodcast.sources import hubeau as hb

    tr = pd.DataFrame({"H": hb.hourly(hb.observations_tr(code, "H", 30)),
                       "Q": hb.hourly(hb.observations_tr(code, "Q", 30))}).dropna()
    tr = tr[tr["Q"] > 0]
    if not len(tr):
        return {}
    e = courbe.to_h(tr["Q"].to_numpy()) - tr["H"].to_numpy()
    return {"n_temps_reel": int(len(tr)), "biais_cm": round(float(e.mean()) * 100, 1),
            "ecart_type_cm": round(float(e.std()) * 100, 1),
            "domaine_q": [round(x, 3) for x in courbe.domaine]}


if __name__ == "__main__":
    for site, code in (("M7022410", "M702241010"), ("M7044010", "M704401010")):
        c = construire(site, code)
        print(code, c.source, "| noeuds", len(c.q), "| Q", c.domaine)
        print("   ", diagnostic(c, site, code))
        for q in (0.05, 0.1, 1, 10, 100, 300):
            print(f"      Q={q:>6} -> H={c.to_h(q):.3f} m")
