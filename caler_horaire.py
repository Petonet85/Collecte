#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cale les parametres horaires du modele pluie-debit sur les pointes de crue.

Jusqu'ici les parametres horaires n'etaient pas cales : ils etaient DERIVES des
parametres journaliers par la regle empirique de Ficchi et al. (2016) — X1 et X3
inchanges, X2 divise par 24, X4 multiplie par 24/24^0,25. C'etait le seul choix
possible, faute de chronique de debit horaire : Hub'Eau n'en conserve qu'un mois.
HydroPortail sert desormais 17 ans au pas horaire, ce qui permet de caler pour de
bon. Le depot contenait d'ailleurs une fonction refine_hourly ecrite puis jamais
appelee, pour cette raison.

L'objectif pese les pointes a 75 % — biais et dispersion — et la dynamique
d'ensemble a 25 %. C'est ce qu'on attend d'un outil de crue : se tromper de 20 %
sur un etiage n'a aucune consequence, se tromper de 20 % sur un pic en a.

Ce que le calage apporte, et ce qu'il n'apporte pas, mesure sur 2021-2026 avec
un an de rodage et 16 pointes independantes au-dessus de 40 m3/s :

                        KGE     biais pic   |e| median   sim/obs etiage
  derives (avant)      0,544      -14 %        22 %          x2,19
  cales (apres)        0,684      -13 %        21 %          x1,70

Les pointes ne bougent pas. Leur erreur — 21 % en median, et un decalage
temporel qui va de -36 a +30 h — ne depend pas des parametres : c'est la limite
d'un GR4H a hydrogramme unitaire unique sur 359 km2 au pas horaire. Ce que le
calage corrige vraiment, c'est le biais de basses eaux, qui passe de x2,2 a
x1,7. Ce n'est pas anodin : c'est ce biais que l'assimilation devait rattraper,
et une correction plus petite est une correction plus sure.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import hydroportail as hp  # noqa: E402
import floodcast.forecast as F  # noqa: E402
from floodcast.model import gr  # noqa: E402
from floodcast.sources import meteo  # noqa: E402

SEUIL_PIC = 40.0          # m3/s : au-dessus, on parle de crue a Saint-Mesmin
ECART_PICS_H = 120        # deux pointes plus proches que cela sont le meme episode
RODAGE_H = 8760           # un an, pour ne pas juger le modele sur ses reservoirs


def couples_horaires(code="M702241010", debut=2010, fin=2027) -> pd.DataFrame:
    ctx = F.build_context(code, verbose=False)
    met = meteo.history(ctx.basin.meteo_points, start=f"{debut}-01-01", end="2026-09-08")
    q = pd.concat([hp.serie(code, "Q", f"01/01/{a}", f"31/12/{a}") for a in range(debut, fin)])
    q = q[~q.index.duplicated(keep="last")].sort_index().resample("1h").mean()
    return pd.DataFrame({"P": met["P"], "E": met["E"]}).join(q.rename("Q"), how="inner").dropna(subset=["P", "E"])


def pointes(serie: pd.Series, seuil=SEUIL_PIC, ecart_h=ECART_PICS_H) -> list:
    """Maxima independants : on retire une fenetre autour de chaque pic retenu."""
    reste, out = serie.dropna(), []
    while len(reste):
        t = reste.idxmax()
        if reste[t] < seuil:
            break
        out.append(t)
        reste = reste.drop(reste.loc[t - pd.Timedelta(hours=ecart_h):
                                     t + pd.Timedelta(hours=ecart_h)].index)
    return sorted(out)


def evaluer(p: gr.GRParams, df: pd.DataFrame, aire: float) -> dict:
    q = pd.Series(gr.mm_to_m3s(gr.run(df["P"].values, df["E"].values, p), aire, 1.0),
                  index=df.index).iloc[RODAGE_H:]
    o = df["Q"].iloc[RODAGE_H:]
    m = o.notna()
    oo, ss = o[m], q[m]
    if len(oo) < 500:
        return {}
    r = float(np.corrcoef(oo, ss)[0, 1])
    kge = 1 - np.sqrt((r - 1) ** 2 + (ss.std() / oo.std() - 1) ** 2
                      + (ss.mean() / oo.mean() - 1) ** 2)
    err, dec = [], []
    for t in pointes(oo):
        fen = q.loc[t - pd.Timedelta(hours=36):t + pd.Timedelta(hours=36)]
        if not len(fen):
            continue
        err.append(fen.max() / oo[t] - 1)
        dec.append((fen.idxmax() - t).total_seconds() / 3600)
    err, dec = np.array(err), np.array(dec)
    bas = oo < 0.3
    return {"KGE": round(float(kge), 3), "n_pointes": int(len(err)),
            "biais_pic_pct": round(100 * float(err.mean()), 1),
            "abs_median_pct": round(100 * float(np.median(np.abs(err))), 1),
            "rmse_relatif_pct": round(100 * float(np.sqrt(np.mean(err ** 2))), 1),
            "decalage_median_h": round(float(np.median(dec)), 1),
            "part_dans_3h_pct": round(100 * float(np.mean(np.abs(dec) <= 3)), 0),
            "sim_sur_obs_etiage": round(float(np.median(ss[bas] / oo[bas])), 2)
            if bas.sum() > 30 else None}


def objectif(p, df, aire):
    s = evaluer(p, df, aire)
    if not s or not s["n_pointes"]:
        return -9.0
    pic = 1 - min(s["rmse_relatif_pct"] / 100 + abs(s["biais_pic_pct"]) / 100, 2.0)
    return 0.25 * s["KGE"] + 0.75 * pic


def caler(df, aire, depart: gr.GRParams, iters=500, seed=7):
    lo = np.array([np.log(50), -8.0, np.log(20), np.log(2), 0.6])
    hi = np.array([np.log(2000), 8.0, np.log(600), np.log(200), 1.6])
    vect = lambda p: np.array([np.log(p.x1), p.x2 * 24, np.log(p.x3), np.log(p.x4), p.cp])
    para = lambda u: gr.GRParams(float(np.exp(u[0])), float(u[1] / 24), float(np.exp(u[2])),
                                 float(np.exp(u[3])), float(u[4]))
    rng = np.random.default_rng(seed)
    best, score = vect(depart), objectif(depart, df, aire)
    for i in range(1, iters + 1):
        prob = 1 - np.log(i) / np.log(iters + 1)
        msk = rng.random(5) < max(prob, 0.25)
        if not msk.any():
            msk[rng.integers(5)] = True
        u = best.copy()
        u[msk] += 0.2 * (hi - lo)[msk] * rng.normal(size=msk.sum())
        s = objectif(para(np.clip(u, lo, hi)), df, aire)
        if s > score:
            score, best = s, np.clip(u, lo, hi)
    return para(best), float(score)


if __name__ == "__main__":
    code = sys.argv[1] if len(sys.argv) > 1 else "M702241010"
    ctx = F.build_context(code, verbose=False)
    df = couples_horaires(code)
    print(f"{len(df)} pas horaires, {int(df['Q'].notna().sum())} avec debit observe", flush=True)
    cal, val = df.loc[:"2020-12-31"], df.loc["2020-01-01":]
    avant = evaluer(ctx.params_hour, val, ctx.basin.area_km2)
    print("avant (parametres derives) :", json.dumps(avant, ensure_ascii=False), flush=True)
    p, s = caler(cal, ctx.basin.area_km2, ctx.params_hour)
    apres = evaluer(p, val, ctx.basin.area_km2)
    print("parametres cales :", {k: round(v, 3) for k, v in p.to_dict().items()})
    print("apres  :", json.dumps(apres, ensure_ascii=False))
    with open(os.path.join(BASE, "data", f"calage_horaire_{code}.json"), "w", encoding="utf-8") as fh:
        json.dump({"params_hour": list(p.as_array()), "objectif_calage": round(s, 4),
                   "validation_avant": avant, "validation_apres": apres,
                   "periode_calage": "2010-2020", "periode_validation": "2021-2026",
                   "source": "HydroPortail (debit horaire) + ERA5 (pluie horaire)"},
                  fh, ensure_ascii=False, indent=1)
    print(f"\ndata/calage_horaire_{code}.json ecrit")
