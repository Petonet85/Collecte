#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construit et tient a jour docs/historique.json : la chronique journaliere longue.

La page embarquait 400 jours dans son propre code. Passer a seize ans aurait
triple son poids pour une fonction qu'on n'ouvre pas a chaque visite. L'historique
vit donc dans un fichier separe, que la page va chercher seulement quand on
demande une fenetre longue ou une annee passee — un aller-retour, puis le cache
du navigateur.

Le fichier se met a jour par la fin : on ne redemande a HydroPortail que les
jours manquants. Reconstruire seize ans a chaque passage serait absurde, et
surtout impoli envers un service public.

Chaque journee est resumee par son minimum et son maximum. A cette echelle un
pic de crue dure quelques heures : une moyenne journaliere l'effacerait, alors
que c'est exactement ce qu'on vient regarder.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
FICHIER = os.path.join(BASE, "docs", "historique.json")
DEBUT = "2010-01-01"
SERIES = (("M703243010", "H"), ("M702241010", "Q"), ("M704401010", "Q"))
MARGE_JOURS = 10          # on redemande toujours les derniers jours : ils bougent


def _journalier(code: str, grandeur: str, deb: pd.Timestamp, fin: pd.Timestamp):
    """Min et max du jour, annee par annee pour ne pas etrangler le service."""
    import hydroportail as hp
    lots = []
    for an in range(deb.year, fin.year + 1):
        d1 = max(deb, pd.Timestamp(f"{an}-01-01")).strftime("%d/%m/%Y")
        d2 = min(fin, pd.Timestamp(f"{an}-12-31")).strftime("%d/%m/%Y")
        try:
            s = hp.serie(code, grandeur, d1, d2)
        except Exception:  # noqa: BLE001
            continue
        if len(s):
            lots.append(s)
    if not lots:
        return pd.DataFrame()
    s = pd.concat(lots)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.resample("1D").agg(["min", "max"]).dropna()


def _pluie(deb: pd.Timestamp, fin: pd.Timestamp) -> pd.Series:
    from floodcast import basin as basin_mod
    from floodcast.sources import meteo
    pts = basin_mod.delineate("M703243010").meteo_points
    try:
        return meteo.history_daily(pts, start=deb.strftime("%Y-%m-%d"),
                                   end=fin.strftime("%Y-%m-%d"))["P"]
    except Exception:  # noqa: BLE001
        return pd.Series(dtype=float)


def _fusion(ancien: dict, neuf: pd.DataFrame, cols=("min", "max")) -> dict:
    """Recolle l'existant et les jours neufs, les neufs faisant foi."""
    idx = pd.date_range(ancien["debut"], periods=len(ancien[cols[0]]), freq="D") \
        if ancien else pd.DatetimeIndex([])
    vieux = pd.DataFrame({c: ancien[c] for c in cols}, index=idx) if ancien else pd.DataFrame()
    tout = pd.concat([vieux[~vieux.index.isin(neuf.index)], neuf]) if len(vieux) else neuf
    tout = tout.sort_index()
    plein = tout.reindex(pd.date_range(tout.index[0], tout.index[-1], freq="D"))
    return {"debut": plein.index[0].strftime("%Y-%m-%d"),
            **{c: [None if not np.isfinite(v) else round(float(v), 3)
                   for v in plein[c]] for c in cols}}


def construire(fin: pd.Timestamp | None = None, verbose=True) -> dict:
    fin = pd.Timestamp(fin or pd.Timestamp.now("UTC")).tz_localize(None).normalize() \
        if fin is None else pd.Timestamp(fin).normalize()
    vieux = {}
    if os.path.exists(FICHIER):
        try:
            with open(FICHIER, encoding="utf-8") as fh:
                vieux = json.load(fh)
        except ValueError:
            vieux = {}

    def depuis(bloc):
        if not bloc:
            return pd.Timestamp(DEBUT)
        fin_bloc = pd.Timestamp(bloc["debut"]) + pd.Timedelta(days=len(bloc["max"]) - 1)
        return max(pd.Timestamp(DEBUT), fin_bloc - pd.Timedelta(days=MARGE_JOURS))

    out = {"stations": {}}
    for code, grandeur in SERIES:
        anc = (vieux.get("stations") or {}).get(code)
        d0 = depuis(anc)
        neuf = _journalier(code, grandeur, d0, fin)
        if neuf.empty and not anc:
            continue
        out["stations"][code] = _fusion(anc, neuf) if not neuf.empty else anc
        if verbose:
            n = len(out["stations"][code]["max"])
            print(f"  {code} {grandeur} : {n} jours depuis {out['stations'][code]['debut']}"
                  f" (+{len(neuf)} rafraichis)", flush=True)

    sl = out["stations"].get("M703243010")
    if sl:
        from floodcast import sevre
        courbe, _, _ = sevre.relation_transfert()
        # La cote chez vous se deduit de la hauteur amont : on la recalcule
        # entierement, pour qu'un recalage du bief se propage a tout l'historique.
        conv = lambda v: [None if x is None else round(float(
            sevre.niveau_rochereau(np.array([x]), courbe)[0]), 3) for x in v]
        out["z_rochereau"] = {"debut": sl["debut"], "min": conv(sl["min"]), "max": conv(sl["max"])}

    anc_p = vieux.get("pluie")
    d0 = depuis({"debut": anc_p["debut"], "max": anc_p["mm"]} if anc_p else None)
    pl = _pluie(d0, fin)
    if len(pl):
        neuf = pd.DataFrame({"mm": pl.to_numpy()}, index=pd.DatetimeIndex(pl.index))
        out["pluie"] = _fusion({"debut": anc_p["debut"], "mm": anc_p["mm"]} if anc_p else None,
                               neuf, cols=("mm",))
    elif anc_p:
        out["pluie"] = anc_p

    out["genere_le"] = pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    out["source"] = "HydroPortail (hauteurs et debits) + ERA5 (pluie de bassin)"
    os.makedirs(os.path.dirname(FICHIER), exist_ok=True)
    tmp = FICHIER + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, FICHIER)
    if verbose:
        print(f"  docs/historique.json : {os.path.getsize(FICHIER)/1024:.0f} ko")
    return out


if __name__ == "__main__":
    construire()
