#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cale la relation debit amont -> hauteur a Saint-Laurent sur les chroniques.

Saint-Laurent ne publie aucun debit. Toute la chaine repose donc sur une
relation entre le debit somme des deux stations amont et la hauteur a l'echelle
de Saint-Laurent. Cette relation etait ajustee sur les maxima mensuels : le plus
fort debit amont du mois apparie a la plus forte hauteur du mois. Deux defauts.

D'abord ces deux maxima ne sont pas forcement le meme evenement — un mois peut
porter une pointe amont breve et, dix jours plus tard, une crue aval plus haute
venue d'ailleurs dans le bassin. Ensuite cela ne donnait que soixante-neuf
couples, tous au pic, et rien en dessous de 0,80 m : sous ce seuil la relation
s'aplatissait et rendait une hauteur constante, incapable de reproduire meme
l'etiage du jour.

On l'ajuste maintenant sur des couples reellement simultanes. HydroPortail sert
les chroniques instantanees ; le retard cale par caler_celerite.py permet de
ramener le debit amont a l'heure ou il se manifeste a l'aval. Vingt-sept crues
depuis 2010 et quatorze etiages d'ete donnent 170 000 couples, de 0,03 a
186 m3/s, de 0,30 a 2,64 m.

La relation n'est pas une loi de puissance et on n'en impose pas une : mediane
de hauteur par tranche de debit, monotonie retablie, interpolation en
log-debit. Au-dela du plus fort couple observe on prolonge par la pente locale,
faute de mieux, et c'est la que le calage de Rochereau vient reprendre la main
en ancrant sur la crue de 1983.

Verification : hysteresis negligeable une fois le retard applique (l'ecart
montee/descente a hauteur egale va de -16 a +7 %, sans signe constant), et
aucune derive de la station (+0,46 cm/an en hautes eaux, p = 0,56, en ponderant
par crue et non par point — ponderer par point faisait croire a +26 cm depuis
2021, les crues recentes etant simplement plus longues).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import caler_celerite as cc  # noqa: E402
import hydroportail as hp  # noqa: E402
from floodcast import sevre  # noqa: E402

CIBLE = "M703243010"
# Saint-Mesmin seule, pour la meme raison que dans caler_celerite.py : l'Ouin
# est en aval de la station cible.
AMONT = ("M702241010",)
ANNEES_ETIAGE = range(2012, 2026)
H_MINI = 0.30          # en dessous, c'est le capteur qui decroche, pas la riviere
N_TRANCHES = 34
MINI_PAR_TRANCHE = 25


def _appariee(H: pd.Series, Q: pd.Series, retard_h: float) -> pd.DataFrame:
    """Apparie hauteur aval et debit amont ramene a l'heure ou il se manifeste.

    On interpole sur une echelle de secondes relatives plutot que sur les index :
    un decalage non entier change la resolution des datetime64 de pandas, et
    l'appariement echoue alors silencieusement, sans lever la moindre erreur.
    """
    ref = H.index[0]
    xh = (H.index - ref).total_seconds().to_numpy() - retard_h * 3600.0
    xq = (Q.index - ref).total_seconds().to_numpy()
    q = np.interp(xh, xq, Q.to_numpy(), left=np.nan, right=np.nan)
    d = pd.DataFrame({"H": H.to_numpy(), "Q": q}, index=H.index).dropna()
    return d[(d["Q"] > 0.01) & (d["H"] > H_MINI)]


def couples_crues() -> pd.DataFrame:
    calage = sevre.calage_propagation()
    lots = []
    for mois in cc.crues():
        h = hp.evenement(CIBLE, "H", mois)
        if len(h) < 50:
            continue
        t_pic = h.idxmax()
        H = cc._cadrer(h, t_pic, avant_j=8, apres_j=10).dropna()
        if not len(H):
            continue
        Q, complet = None, True
        for code in AMONT:
            s = cc._cadrer(hp.evenement(code, "Q", mois), t_pic, avant_j=8, apres_j=10)
            if not len(s):
                complet = False
                break
            Q = s if Q is None else Q.add(s, fill_value=np.nan)
        if not complet:
            continue
        Q = Q.dropna()
        if len(Q) < 100:
            continue
        d = _appariee(H, Q, sevre.retard_amont(float(H.max()), calage))
        if len(d) < 200:
            continue
        d["episode"] = str(t_pic.date())
        lots.append(d)
    return pd.concat(lots) if lots else pd.DataFrame()


def couples_etiages() -> pd.DataFrame:
    """Basses eaux : c'est ce qui manquait le plus a l'ancienne relation."""
    lots = []
    for an in ANNEES_ETIAGE:
        deb, fin = f"15/07/{an}", f"30/09/{an}"
        H = hp.serie(CIBLE, "H", deb, fin)
        if len(H) < 50:
            continue
        H = H.resample("10min").mean().interpolate(limit=36).dropna()
        Q, complet = None, True
        for code in AMONT:
            s = hp.serie(code, "Q", deb, fin)
            if len(s) < 50:
                complet = False
                break
            s = s.resample("10min").mean().interpolate(limit=36)
            Q = s if Q is None else Q.add(s, fill_value=np.nan)
        if not complet:
            continue
        Q = Q.dropna()
        if len(Q) < 100:
            continue
        # En etiage tout est plat : le retard n'a aucun effet mesurable.
        d = _appariee(H, Q, 0.0)
        if len(d) < 200:
            continue
        d["episode"] = f"etiage-{an}"
        lots.append(d)
    return pd.concat(lots) if lots else pd.DataFrame()


def construire(d: pd.DataFrame, n=N_TRANCHES, mini=MINI_PAR_TRANCHE):
    """Table H(Q) monotone : mediane par tranche de log-debit."""
    lq = np.log10(d["Q"].to_numpy())
    bords = np.linspace(lq.min(), lq.max(), n + 1)
    idx = np.clip(np.digitize(lq, bords) - 1, 0, n - 1)
    q, h = [], []
    for k in range(n):
        m = idx == k
        if m.sum() < mini:
            continue
        q.append(float(np.median(d["Q"].to_numpy()[m])))
        h.append(float(np.median(d["H"].to_numpy()[m])))
    q = np.array(q)
    h = np.maximum.accumulate(np.array(h))   # la hauteur croit avec le debit
    o = np.argsort(q)
    return q[o], h[o]


def valider(P: pd.DataFrame) -> dict:
    """Validation croisee : on retire un episode, on cale sur les autres.

    C'est la seule validation honnete ici. Decouper par annee ferait croire a
    une derive de la station qui n'existe pas, et valider sur les points ayant
    servi au calage ne dirait rien, un episode apportant des milliers de points
    tres correles entre eux.
    """
    ancienne, _, _ = sevre.relation_transfert()
    lignes = []
    for ep in sorted(P["episode"].unique()):
        app, test = P[P["episode"] != ep], P[P["episode"] == ep]
        tab = construire(app)
        e_t = sevre.CourbeTransfert(tab[0], tab[1]).to_h(test["Q"].to_numpy()) - test["H"].to_numpy()
        e_a = ancienne.to_h(test["Q"].to_numpy()) - test["H"].to_numpy()
        i = int(np.argmax(test["H"].to_numpy()))
        lignes.append({"episode": ep, "h_pic": round(float(test["H"].max()), 2),
                       "table_median_cm": round(float(np.median(e_t)) * 100, 1),
                       "ancienne_median_cm": round(float(np.median(e_a)) * 100, 1),
                       "table_pic_cm": round(float(e_t[i]) * 100, 1),
                       "ancienne_pic_cm": round(float(e_a[i]) * 100, 1)})
    d = pd.DataFrame(lignes)
    crue = d[~d["episode"].str.startswith("etiage")]
    return {"detail": d, "resume": {
        "n_episodes": int(len(d)),
        "table_biais_cm": round(float(d["table_median_cm"].mean()), 1),
        "table_abs_median_cm": round(float(d["table_median_cm"].abs().median()), 1),
        "ancienne_biais_cm": round(float(d["ancienne_median_cm"].mean()), 1),
        "ancienne_abs_median_cm": round(float(d["ancienne_median_cm"].abs().median()), 1),
        "table_pic_biais_cm": round(float(crue["table_pic_cm"].mean()), 1),
        "table_pic_ecart_type_cm": round(float(crue["table_pic_cm"].std()), 1),
        "ancienne_pic_biais_cm": round(float(crue["ancienne_pic_cm"].mean()), 1),
        "ancienne_pic_ecart_type_cm": round(float(crue["ancienne_pic_cm"].std()), 1),
    }}


if __name__ == "__main__":
    P = pd.concat([couples_crues(), couples_etiages()])
    print(f"{len(P)} couples simultanes sur {P['episode'].nunique()} episodes | "
          f"H {P['H'].min():.2f}–{P['H'].max():.2f} m | Q {P['Q'].min():.3f}–{P['Q'].max():.1f} m³/s",
          flush=True)
    q, h = construire(P)
    print(f"table : {len(q)} noeuds, Q {q[0]:.3f}–{q[-1]:.1f} m³/s, H {h[0]:.3f}–{h[-1]:.3f} m")

    v = valider(P)
    print("\nvalidation croisee, un episode laisse de cote a chaque fois :")
    print(v["detail"].to_string(index=False))
    print("\n" + json.dumps(v["resume"], ensure_ascii=False, indent=1))

    calage = {
        "forme": "table monotone H(Q), interpolation en log-debit",
        "q": [round(float(x), 4) for x in q],
        "h": [round(float(x), 4) for x in h],
        "n_couples": int(len(P)), "n_episodes": int(P["episode"].nunique()),
        "domaine_q": [round(float(q[0]), 3), round(float(q[-1]), 1)],
        "domaine_h": [round(float(h[0]), 3), round(float(h[-1]), 3)],
        "validation": v["resume"],
        "source": "HydroPortail, chroniques instantanees ; debit amont recale du "
                  "retard mesure (voir caler_celerite.py)",
    }
    with open(os.path.join(BASE, "data", "calage_transfert.json"), "w", encoding="utf-8") as fh:
        json.dump(calage, fh, ensure_ascii=False, indent=1)
    v["detail"].to_csv(os.path.join(BASE, "data", "calage_transfert.csv"), index=False)
    print("\ndata/calage_transfert.json ecrit")
