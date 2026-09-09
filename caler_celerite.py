#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cale sur les crues passees le retard entre le debit amont et la cote aval.

Pourquoi ce n'est pas une celerite. On serait tente de mesurer le temps que met
l'onde a descendre les 30 km qui separent Saint-Mesmin de Saint-Laurent, et d'en
tirer une vitesse. La mesure le dement : le pic de l'Ouin precede celui de
Saint-Laurent de 14 a 24 h alors que l'Ouin n'est qu'a 18 km, ce qui donnerait
une onde a 0,2 m/s — physiquement impossible. C'est que l'Ouin est un petit
bassin nerveux de 61 km² : il culmine tot apres l'averse, quand Saint-Laurent,
dix fois plus grand, met bien plus longtemps a rassembler son eau. Le decalage
qu'on observe entre deux stations melange donc le trajet dans la riviere et
l'ecart de temps de reponse entre les bassins, et rien ne permet de les separer
avec les seules stations disponibles.

Ce qu'on cale est donc le retard *effectif* entre l'entree et la sortie du
modele : d'un cote le debit amont somme, qui est exactement ce que la chaine
calcule, de l'autre la hauteur a Saint-Laurent, qui est ce qu'elle predit. Peu
importe alors ce qui, du trajet ou de la reponse des bassins, fait ce retard :
c'est celui-la qu'il faut appliquer, et lui seul est mesurable.

Il depend de l'ampleur de la crue, et c'est bien le sens physique attendu : plus
la riviere est haute, plus l'onde va vite et plus les versants ont deja rendu
leur eau. On ajuste retard = a / H^n, forme que donne Manning, ou H est la cote
au pic a Saint-Laurent.
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
from floodcast import sevre  # noqa: E402

CIBLE = "M703243010"
# Saint-Mesmin seule : l'Ouin rejoint la Sevre 2,2 km sous la station de
# Saint-Laurent et n'a donc aucun effet sur sa hauteur.
AMONT = ("M702241010",)
PAS_MIN = 10
LISSAGE_H = 3.0
RETARD_MAX_H = 36.0
ECART_TOLERE_H = 5.0
R_MINI = 0.6


def _cadrer(s: pd.Series, t_pic: pd.Timestamp, avant_j=3.5, apres_j=3.5) -> pd.Series:
    if not len(s):
        return s
    fen = s.loc[t_pic - pd.Timedelta(days=avant_j):t_pic + pd.Timedelta(days=apres_j)]
    return fen.resample(f"{PAS_MIN}min").mean().interpolate(limit=36)


def _montee(s: pd.Series) -> np.ndarray:
    """Derivee lissee, normalisee : seule la forme de la montee est comparee."""
    n = int(LISSAGE_H * 60 / PAS_MIN)
    d = s.diff().rolling(n, center=True, min_periods=n // 2).mean().fillna(0.0)
    d = np.clip(d.to_numpy(dtype=float), 0, None)
    return d / d.max() if d.max() > 0 else d


def retard_correlation(amont: pd.Series, aval: pd.Series):
    a, b = _montee(amont), _montee(aval)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    pas_h = PAS_MIN / 60.0
    meilleur = (-2.0, None)
    for k in range(int(RETARD_MAX_H / pas_h) + 1):
        x, y = a[:n - k], b[k:]
        if len(x) < 100 or x.std() < 1e-12 or y.std() < 1e-12:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if r > meilleur[0]:
            meilleur = (r, k * pas_h)
    return meilleur[1], meilleur[0]


def crues(depuis="2010-06-01", n_grosses=20, n_moyennes=12):
    """Crues distinctes, reperees par l'instant du pic et non par le mois.

    Les fenetres debordent d'un mois sur l'autre : sans ce filtre, une meme crue
    de janvier compte deux fois et pese double dans l'ajustement.
    """
    h = sevre._elabore("M7032430", "HIXM")
    h = h[h.index >= depuis]
    grosses = h.sort_values(ascending=False).head(n_grosses)
    reste = h.drop(grosses.index)
    reste = reste[(reste >= 0.95) & (reste <= 1.55)].sort_values(ascending=False)
    mois = sorted(set(grosses.index) | set(reste.iloc[::3][:n_moyennes].index))
    vues = {}
    for m in mois:
        s = hp.evenement(CIBLE, "H", m)
        if len(s) < 50:
            continue
        vues.setdefault(s.idxmax().round("6h"), m)
    return sorted(vues.values(), key=lambda m: str(m))


def analyser(mois: pd.Timestamp) -> dict | None:
    aval_brut = hp.evenement(CIBLE, "H", mois)
    if len(aval_brut) < 50:
        return None
    t_pic = aval_brut.idxmax()
    aval = _cadrer(aval_brut, t_pic)

    # Le debit amont somme : exactement l'entree de la chaine de prevision.
    q = None
    for code in AMONT:
        s = _cadrer(hp.evenement(code, "Q", mois), t_pic)
        if not len(s):
            return None
        q = s if q is None else q.add(s, fill_value=np.nan)
    commun = q.dropna().index.intersection(aval.dropna().index)
    if len(commun) < 300:
        return None
    q, av = q.reindex(commun), aval.reindex(commun)

    tau_c, r = retard_correlation(q, av)
    tau_p = (av.idxmax() - q.idxmax()).total_seconds() / 3600.0
    if tau_c is None or r < R_MINI:
        return None
    if not (0 < tau_p < RETARD_MAX_H) or abs(tau_c - tau_p) > ECART_TOLERE_H:
        return None
    return {"crue": str(t_pic.date()), "h_pic_m": round(float(av.max()), 3),
            "q_pic_amont_m3s": round(float(q.max()), 1),
            "retard_correlation_h": round(tau_c, 2), "retard_pics_h": round(tau_p, 2),
            "retard_h": round((tau_c + tau_p) / 2, 2), "r": round(r, 3)}


# Le troncon Saint-Laurent -> Tiffauges est le seul dont on puisse tirer une
# vraie celerite : meme riviere, meme regime, et Rochereau se trouve entre les
# deux, a 13,2 des 25,1 km. Le bassin gagne 238 km² dans l'intervalle, donc la
# mesure n'est pas non plus une celerite pure — mais elle en est bien plus
# proche que celle d'un affluent nerveux compare a son exutoire.
AVAL = "M711241020"
DIST_TIFFAUGES_KM = 25.1
DIST_ROCHEREAU_KM = 13.2


def analyser_bief(mois: pd.Timestamp) -> dict | None:
    amont = hp.evenement(CIBLE, "H", mois)
    aval = hp.evenement(AVAL, "H", mois)
    if len(amont) < 50 or len(aval) < 50:
        return None
    t_pic = amont.idxmax()
    a, b = _cadrer(amont, t_pic), _cadrer(aval, t_pic)
    commun = a.dropna().index.intersection(b.dropna().index)
    if len(commun) < 300:
        return None
    a, b = a.reindex(commun), b.reindex(commun)
    tau_c, r = retard_correlation(a, b)
    tau_p = (b.idxmax() - a.idxmax()).total_seconds() / 3600.0
    if tau_c is None or not np.isfinite(tau_c) or r < R_MINI:
        return None
    # Tolerance resserree : le bief est court, les deux mesures doivent coller.
    if not (0 < tau_p < RETARD_MAX_H) or abs(tau_c - tau_p) > 3.0:
        return None
    tau = (tau_c + tau_p) / 2
    return {"crue": str(t_pic.date()), "h_pic_m": round(float(a.max()), 3),
            "retard_h": round(tau, 2), "celerite_ms": round(DIST_TIFFAUGES_KM * 1000 / tau / 3600, 3),
            "r": round(r, 3)}


def ajuster(df: pd.DataFrame) -> dict | None:
    d = df[(df["retard_h"] > 0) & (df["h_pic_m"] > 0)]
    if len(d) < 6:
        return None
    x = np.log(d["h_pic_m"].to_numpy())
    y = np.log(d["retard_h"].to_numpy())
    n, c = np.polyfit(x, y, 1)
    pred = np.exp(c + n * x)
    resid = pred - d["retard_h"].to_numpy()
    r = float(np.corrcoef(x, y)[0, 1])
    return {"a": round(float(np.exp(c)), 3), "n": round(float(-n), 3),
            "R2_log": round(r ** 2, 3), "n_crues": int(len(d)),
            "biais_h": round(float(resid.mean()), 2),
            "ecart_type_h": round(float(resid.std()), 2),
            "h_calage": [round(float(d["h_pic_m"].min()), 2),
                         round(float(d["h_pic_m"].max()), 2)]}


if __name__ == "__main__":
    mois = crues()

    print("=== amont -> Saint-Laurent : retard effectif ===", flush=True)
    amont = pd.DataFrame([x for x in (analyser(m) for m in mois) if x]).sort_values("h_pic_m")
    with pd.option_context("display.width", 200):
        print(amont.to_string(index=False))
    fit = ajuster(amont)
    print("\nretard = a / H^n  ->", fit)
    plat = float(amont["retard_h"].std())
    print(f"a retard constant, l'ecart-type residuel serait {plat:.2f} h : "
          f"la dependance a la cote en retire {100*(1-fit['ecart_type_h']/plat):.0f} %")

    print("\n=== Saint-Laurent -> Tiffauges : celerite du bief ===", flush=True)
    bief = pd.DataFrame([x for x in (analyser_bief(m) for m in mois) if x]).sort_values("h_pic_m")
    with pd.option_context("display.width", 200):
        print(bief.to_string(index=False))
    c_moy = float(bief["celerite_ms"].mean())
    pente, _ = np.polyfit(np.log(bief["h_pic_m"]), np.log(bief["retard_h"]), 1)
    print(f"\ncelerite {c_moy:.2f} ± {bief['celerite_ms'].std():.2f} m/s sur {len(bief)} crues ; "
          f"pente en cote {pente:+.3f} — indetectable, on garde une constante")
    tau_roch = DIST_ROCHEREAU_KM * 1000 / c_moy / 3600

    calage = {
        "amont_vers_saint_laurent": {
            "forme": "retard_h = a / H_pic^n", "a": fit["a"], "n": fit["n"],
            "n_crues": fit["n_crues"], "R2_log": fit["R2_log"],
            "ecart_type_h": fit["ecart_type_h"], "h_calage": fit["h_calage"],
        },
        "saint_laurent_vers_rochereau": {
            "forme": "distance / celerite", "distance_km": DIST_ROCHEREAU_KM,
            "celerite_ms": round(c_moy, 3),
            "ecart_type_celerite_ms": round(float(bief["celerite_ms"].std()), 3),
            "retard_h": round(tau_roch, 2), "n_crues": int(len(bief)),
            "mesure_sur": f"Saint-Laurent -> Tiffauges, {DIST_TIFFAUGES_KM} km, "
                          f"retard {bief['retard_h'].mean():.2f} h",
        },
        "source": "HydroPortail, chroniques instantanees, crues depuis 2010",
    }
    os.makedirs(os.path.join(BASE, "data"), exist_ok=True)
    amont.to_csv(os.path.join(BASE, "data", "calage_retard.csv"), index=False)
    bief.to_csv(os.path.join(BASE, "data", "calage_bief.csv"), index=False)
    with open(os.path.join(BASE, "data", "calage_propagation.json"), "w", encoding="utf-8") as fh:
        json.dump(calage, fh, ensure_ascii=False, indent=1)
    print("\n" + json.dumps(calage, ensure_ascii=False, indent=1))
