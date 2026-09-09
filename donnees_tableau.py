#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rassemble tout ce qu'affiche le tableau de bord : hydrogramme, seuils, pluie.

La pluie passee combine deux sources de nature differente, et la page les
distingue : la lame d'eau radar effectivement mesuree sur le bassin (archivee
par ce depot toutes les cinq minutes) et, la ou l'archive ne remonte pas encore,
l'analyse du modele Meteo-France. Les confondre laisserait croire a une mesure
la ou il n'y a qu'un modele.
"""
from __future__ import annotations

import glob
import json
import os
import time

import numpy as np
import pandas as pd
import requests

from ancrage import raccorder_quantiles

BASE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(BASE, "docs")
BASSIN_RADAR = "M703243010"
HEURES_PASSEES = 48

# Open-Meteo facture un appel par point : on reste sous les 600 par minute.
TAILLE_LOT = 200
PAUSE_LOT = 24.0


def pluie_radar(fin: pd.Timestamp, heures: int = HEURES_PASSEES) -> pd.Series:
    """Lame d'eau radar horaire archivee sur le bassin (mm/h)."""
    fichiers = sorted(glob.glob(os.path.join(BASE, "donnees", "radar", BASSIN_RADAR, "*.csv")))
    if not fichiers:
        return pd.Series(dtype=float)
    lot = pd.concat([pd.read_csv(f, parse_dates=["instant_utc"]) for f in fichiers])
    idx = pd.DatetimeIndex(lot["instant_utc"]).tz_convert(None)
    ser = pd.Series(lot["lame_mm"].to_numpy(dtype=float), index=idx).sort_index()
    ser = ser[~ser.index.duplicated(keep="last")]
    # Chaque valeur est un cumul sur cinq minutes : la somme horaire est un mm/h.
    horaire = ser.resample("1h").sum(min_count=1)
    return horaire.loc[fin - pd.Timedelta(hours=heures):fin]


def pluie_modele(points, heures: int = HEURES_PASSEES, jours_prevus: int = 4):
    """Analyse Meteo-France passee et prevision deterministe, horaires."""
    from floodcast.sources import meteo

    df = meteo.recent(points, past_days=max(int(np.ceil(heures / 24)) + 1, 3),
                      forecast_days=jours_prevus)
    return df["P"].dropna()


def pluie_prevue(points, jours: int = 4) -> pd.DataFrame:
    """Quantiles de pluie horaire prevue sur le bassin, grand ensemble multi-modeles."""
    from floodcast.sources import meteo

    ens = meteo.ensemble(points, forecast_days=jours)
    if ens.empty:
        return pd.DataFrame()
    tab = ens.to_numpy()
    return pd.DataFrame({q: np.percentile(tab, q, axis=1) for q in (10, 50, 90)},
                        index=ens.index)


def _cumul_point(fin: pd.Timestamp, heures: int = HEURES_PASSEES):
    """Cumul radar au point suivi, a comparer directement a un pluviometre."""
    fichiers = sorted(glob.glob(os.path.join(BASE, "donnees", "radar", BASSIN_RADAR, "*.csv")))
    if not fichiers:
        return None
    lot = pd.concat([pd.read_csv(f, parse_dates=["instant_utc"]) for f in fichiers])
    if "pt_rochereau" not in lot.columns:
        return None
    idx = pd.DatetimeIndex(lot["instant_utc"]).tz_convert(None)
    ser = pd.Series(pd.to_numeric(lot["pt_rochereau"], errors="coerce").to_numpy(), index=idx)
    ser = ser[~ser.index.duplicated(keep="last")].sort_index()
    ser = ser.loc[fin - pd.Timedelta(hours=heures):fin].dropna()
    return round(float(ser.sum()), 2) if len(ser) else None


_SITE_DE = {"M702241010": "M7022410", "M704401010": "M7044010"}


def _stations(prevision, h_saint_laurent, q_amont):
    """Les trois stations qui portent la prevision, toutes en hauteur d'echelle.

    Elles sont affichees en hauteur et non en debit. C'est la grandeur que la
    station mesure reellement — le debit en est deja une interpretation —, c'est
    celle qu'on lit sur le terrain, et c'est la seule qui se compare d'une
    station a l'autre d'un coup d'oeil : trois hauteurs empilees se lisent
    ensemble, un debit de 0,1 m3/s a l'Ouin et de 200 m3/s a Saint-Mesmin ne se
    comparent pas. Saint-Laurent, du reste, ne publie que de la hauteur.

    Le modele, lui, travaille en debit. La conversion passe par la courbe
    hauteur-debit de chaque station, construite sur ses propres couples publies
    (voir tarage.py). Convertir chaque quantile separement est licite : la
    relation est monotone, elle preserve donc l'ordre des scenarios.
    """
    from floodcast.sources import hubeau as hb

    import tarage

    def dernier(valeurs):
        valides = [v for v in valeurs if v is not None and np.isfinite(v)]
        return valides[-1] if valides else None

    obs_sl = [None if not np.isfinite(x) else round(float(x), 3)
              for x in h_saint_laurent.to_numpy()]
    out = [{
        "code": "M703243010", "nom": "Sèvre Nantaise à Saint-Laurent-sur-Sèvre",
        "grandeur": "hauteur", "unite": "m", "decimales": 2, "surface_km2": 576,
        "observe": {"time": [d.isoformat() for d in h_saint_laurent.index], "v": obs_sl},
        "prevu": {"time": prevision["time"],
                  # 18 h et non 22 : c'est le temps de demi-vie du residu de
                  # transfert reellement mesure (rho = 0,73 a six heures).
                  "q": raccorder_quantiles(prevision["h_saint_laurent"], dernier(obs_sl),
                                           horizon_decroissance_h=18.0, plafond=0.5)},
    }]
    for code, bloc in (prevision.get("stations") or {}).items():
        obs = bloc["observe"]
        idx = pd.DatetimeIndex(pd.to_datetime(obs["time"]))
        mesure = hb.hourly(hb.observations_tr(code, "H", 25)).dropna()
        h_obs = mesure.reindex(idx).interpolate(limit=3)
        obs_h = [None if not np.isfinite(x) else round(float(x), 3) for x in h_obs.to_numpy()]

        courbe = tarage.construire(_SITE_DE[code], code)
        # Quatre decimales, pas trois : sur une bande zoomee au centimetre,
        # l'arrondi au millimetre se voit comme un escalier qui n'existe pas.
        h_prev = {k: [round(float(x), 4) for x in courbe.to_h(np.asarray(v, dtype=float))]
                  for k, v in bloc["Q"].items()}
        out.append({
            "code": code, "nom": bloc["nom"], "grandeur": "hauteur", "unite": "m",
            "decimales": 2, "surface_km2": bloc["surface_km2"],
            "observe": {"time": obs["time"], "v": obs_h, "debit": obs["Q"]},
            "prevu": {"time": bloc["time"],
                      "q": raccorder_quantiles(h_prev, dernier(obs_h),
                                               horizon_decroissance_h=20.0, plafond=0.4),
                      "debit": bloc["Q"].get("50")},
            "tarage": tarage.diagnostic(courbe, _SITE_DE[code], code),
        })
    return out


def assembler(prevision: dict, horizon_h: int = 72) -> dict:
    from floodcast import sevre
    from floodcast.sources import hubeau as hb

    courbe, _, _ = sevre.relation_transfert()
    calage = json.load(open(os.path.join(DOCS, "calage.json"), encoding="utf-8"))
    seuils = json.load(open(os.path.join(DOCS, "seuils.json"), encoding="utf-8"))
    mnt = json.load(open(os.path.join(DOCS, "mnt_fin.json"), encoding="utf-8"))
    points = [(mnt["centre"][0], mnt["centre"][1])]

    # --- observe : la cote a Rochereau se lit sur la hauteur MESUREE a
    # Saint-Laurent, decalee du temps de parcours du bief — ce qui passe chez
    # vous maintenant est passe devant l'echelle 2,3 h plus tot. C'est le chemin
    # le plus court : une mesure, un retard cale, une relation calee.
    #
    # Il passait auparavant par le debit amont, parce que la relation
    # hauteur-debit de Saint-Laurent, ajustee sur les seuls maxima mensuels
    # au-dessus de 0,80 m, s'aplatissait en etiage et figeait la cote au fond du
    # lit. La table calee sur 170 000 couples instantanes descend a 0,49 m : le
    # detour n'a plus lieu d'etre, et il avait le defaut d'ignorer le retard.
    t0 = pd.Timestamp(prevision["date_prevision"].replace(" ", "T").rstrip("Z"))
    h_obs = hb.hourly(hb.observations_tr("M703243010", "H", 20)).dropna()
    h_obs = h_obs.loc[t0 - pd.Timedelta(days=12):]
    tau_bief = sevre.retard_rochereau()
    h_bief = h_obs.copy()
    h_bief.index = h_bief.index + pd.Timedelta(hours=tau_bief)
    # Les premieres heures n'ont pas d'antecedent : on les comble par la mesure
    # la plus proche plutot que de laisser un trou en tete de courbe.
    h_bief = h_bief.reindex(h_obs.index, method="nearest",
                            tolerance=pd.Timedelta(minutes=90)).ffill().bfill()
    z_obs = (sevre.niveau_rochereau(h_bief.to_numpy(dtype=float), courbe)
             if len(h_bief) else np.array([]))

    # Le debit amont mesure reste affiche : c'est lui qui porte la prevision.
    q_amont = None
    for site in ("M7022410", "M7044010"):
        q = hb.hourly(hb.observations_tr(site, "Q", 20)).dropna()
        q_amont = q if q_amont is None else q_amont.add(q, fill_value=0.0)
    q_amont = (q_amont.reindex(h_obs.index).interpolate(limit=3)
               if q_amont is not None else pd.Series(index=h_obs.index, dtype=float))

    radar = pluie_radar(t0)
    modele = pluie_modele(points)
    prevue = pluie_prevue(points)
    passe = modele.loc[t0 - pd.Timedelta(hours=HEURES_PASSEES):t0]

    return {
        "meta": {
            "lieu": "Rochereau, Mortagne-sur-Sèvre",
            "riviere": "Sèvre Nantaise",
            "station": "La Sèvre Nantaise à Saint-Laurent-sur-Sèvre",
            "date_prevision": prevision["date_prevision"],
            "genere_le": prevision["genere_le"],
            "horizon_h": horizon_h,
        },
        "seuils": calage["seuils_propriete"],
        "propagation": prevision.get("propagation"),
        "transfert": prevision.get("transfert"),
        "scenarios": [s for s in calage["scenarios"] if not s.get("ancre")],
        "reperes": [s for s in calage["scenarios"] if s.get("ancre")],
        "observe": {
            "time": [d.isoformat() for d in q_amont.index],
            "q_amont": [None if not np.isfinite(v) else round(float(v), 3)
                        for v in q_amont.to_numpy()],
            "h_echelle": [None if not np.isfinite(v) else round(float(v), 3)
                          for v in h_obs.to_numpy()],
            "z_rochereau": [round(float(v), 3) for v in z_obs],
        },
        "prevision": {
            "time": prevision["time"],
            "z_rochereau": raccorder_quantiles(
                prevision["z_rochereau"],
                float(z_obs[-1]) if len(z_obs) else None,
                horizon_decroissance_h=22.0, plafond=0.5),
            "h_echelle": prevision["h_saint_laurent"],
        },
        "pluie": {
            "radar": {"time": [d.isoformat() for d in radar.index],
                      "mm": [None if not np.isfinite(v) else round(float(v), 3)
                             for v in radar.to_numpy()]},
            "analyse": {"time": [d.isoformat() for d in passe.index],
                        "mm": [round(float(v), 2) for v in passe.to_numpy()]},
            "prevue": {"time": [d.isoformat() for d in prevue.index],
                       "p10": [round(float(v), 2) for v in prevue[10]],
                       "p50": [round(float(v), 2) for v in prevue[50]],
                       "p90": [round(float(v), 2) for v in prevue[90]]} if len(prevue) else None,
            "cumuls": {
                "radar_mesure_mm": round(float(radar.sum()), 1) if len(radar) else None,
                "radar_heures": int(radar.notna().sum()) if len(radar) else 0,
                "analyse_48h_mm": round(float(passe.sum()), 1),
                "point_rochereau_mm": _cumul_point(t0),
                "prevu_median_mm": round(float(prevue[50].sum()), 1) if len(prevue) else None,
                "prevu_p90_mm": round(float(prevue[90].sum()), 1) if len(prevue) else None,
            },
        },
        "stations": _stations(prevision, h_obs, q_amont),
        "periodes_retour": {"maison": seuils["T_maison"], "atelier": seuils["T_atelier"]},
    }
