#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chroniques instantanees de HydroPortail, pour les crues passees.

Hub'Eau ne conserve le temps reel qu'un mois et Vigicrues cinquante jours :
ni l'un ni l'autre ne permet de voir une crue passee au pas fin. HydroPortail
(hydro.eaufrance.fr, meme banque, service du SCHAPI) sert les memes mesures
sur n'importe quelle periode, avec leur pas d'acquisition d'origine — variable,
la station enregistrant sur seuil de variation, donc dense justement pendant
les crues. C'est la seule source qui permette de caler un temps de parcours.

L'endpoint est celui qu'utilise la page « Series de mesures » du portail. Les
reponses sont mises en cache sur disque sans expiration : une mesure de 2014 ne
changera plus, et il n'y a aucune raison de redemander deux fois la meme chose
a un service public.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import pandas as pd
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "data", "cache_hydroportail")
URL = "https://hydro.eaufrance.fr/stationhydro/ajax/{code}/series"
ENTETES = {"User-Agent": "floodcast/1.0 (calage hydrologique, usage personnel)",
           "X-Requested-With": "XMLHttpRequest",
           "Accept": "application/json, text/javascript, */*; q=0.01"}
PAUSE_S = 1.5
_dernier_appel = [0.0]

# Les valeurs arrivent en unites entieres : millimetres pour une hauteur,
# litres par seconde pour un debit.
DIVISEUR = {"H": 1000.0, "Q": 1000.0}


def _attendre():
    ecart = time.time() - _dernier_appel[0]
    if ecart < PAUSE_S:
        time.sleep(PAUSE_S - ecart)
    _dernier_appel[0] = time.time()


def serie(code_station: str, metric: str, debut: str, fin: str,
          statut: str = "most_valid", essais: int = 3) -> pd.Series:
    """Chronique instantanee entre deux dates (format jj/mm/aaaa)."""
    cle = hashlib.sha1(f"{code_station}|{metric}|{debut}|{fin}|{statut}".encode()).hexdigest()
    os.makedirs(CACHE, exist_ok=True)
    chemin = os.path.join(CACHE, f"{cle}.json")
    if os.path.exists(chemin):
        with open(chemin, encoding="utf-8") as fh:
            brut = json.load(fh)
    else:
        params = {
            "hydro_series[startAt]": debut, "hydro_series[endAt]": fin,
            "hydro_series[variableType]": "simple_and_interpolated_and_hourly_variable",
            "hydro_series[simpleAndInterpolatedAndHourlyVariable]": metric,
            "hydro_series[statusData]": statut,
        }
        brut = None
        for k in range(essais):
            _attendre()
            try:
                r = requests.get(URL.format(code=code_station), params=params,
                                 headers=ENTETES, timeout=180)
                if r.status_code == 200:
                    brut = (r.json().get("series") or {}).get("data") or []
                    break
                # 500 signifie le plus souvent « pas cette grandeur ici » :
                # inutile d'insister, mais on distingue du reseau qui flanche.
                if r.status_code == 500 and k == essais - 1:
                    brut = []
            except requests.RequestException:
                pass
            time.sleep(2.0 * (k + 1))
        if brut is None:
            brut = []
        with open(chemin, "w", encoding="utf-8") as fh:
            json.dump(brut, fh)
    if not brut:
        return pd.Series(dtype=float)
    s = pd.Series([p["v"] for p in brut],
                  index=pd.to_datetime([p["t"] for p in brut], utc=True).tz_localize(None),
                  dtype=float).sort_index() / DIVISEUR.get(metric, 1.0)
    return s[~s.index.duplicated(keep="last")]


def evenement(code_station: str, metric: str, mois: pd.Timestamp, marge_j: int = 6) -> pd.Series:
    """Chronique autour d'un mois de crue, avec de la marge de part et d'autre."""
    deb = (mois - pd.Timedelta(days=marge_j)).strftime("%d/%m/%Y")
    fin = (mois + pd.offsets.MonthEnd(0) + pd.Timedelta(days=marge_j)).strftime("%d/%m/%Y")
    return serie(code_station, metric, deb, fin)
