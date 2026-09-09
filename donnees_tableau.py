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


def _stations(prevision, h_saint_laurent, q_amont):
    """Les deux stations qui portent la prevision, chacune dans son unite.

    Saint-Laurent est limnimetrique : on la suit en hauteur a l'echelle, la
    seule grandeur qu'elle publie. Saint-Mesmin jauge le debit et couvre a
    elle seule 62 % du bassin amont : c'est la que la prevision se confronte
    a une mesure de meme nature.
    """
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
                  "q": raccorder_quantiles(prevision["h_saint_laurent"], dernier(obs_sl),
                                           horizon_decroissance_h=22.0, plafond=0.5)},
    }]
    for code, bloc in (prevision.get("stations") or {}).items():
        obs = bloc["observe"]
        out.append({
            "code": code, "nom": bloc["nom"], "grandeur": "débit", "unite": "m³/s",
            "decimales": 2, "surface_km2": bloc["surface_km2"],
            "observe": {"time": obs["time"], "v": obs["Q"]},
            "prevu": {"time": bloc["time"],
                      "q": raccorder_quantiles(bloc["Q"], dernier(obs["Q"]),
                                               horizon_decroissance_h=20.0)},
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

    # --- observe : on part du DEBIT amont, mesure en temps reel, et non de la
    # hauteur a Saint-Laurent. La relation hauteur-debit de cette station n'est
    # calee qu'au-dessus de 0,80 m ; en etiage elle renvoie un debit nul et la
    # cote se fige au fond du lit, ce qui donne une courbe plate et fausse.
    t0 = pd.Timestamp(prevision["date_prevision"].replace(" ", "T").rstrip("Z"))
    q_amont = None
    for site in ("M7022410", "M7044010"):
        q = hb.hourly(hb.observations_tr(site, "Q", 20)).dropna()
        q_amont = q if q_amont is None else q_amont.add(q, fill_value=0.0)
    q_amont = q_amont.loc[t0 - pd.Timedelta(days=12):] if q_amont is not None else pd.Series(dtype=float)
    z_obs = (sevre.ROCHEREAU["z_fond"]
             + sevre.ROCHEREAU["a"] * np.maximum(q_amont.to_numpy(dtype=float), 0.0)
             ** sevre.ROCHEREAU["b"]) if len(q_amont) else np.array([])
    h_obs = hb.hourly(hb.observations_tr("M703243010", "H", 20)).dropna()
    h_obs = h_obs.reindex(q_amont.index).interpolate(limit=3) if len(q_amont) else h_obs

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
        "scenarios": [s for s in calage["scenarios"] if not s.get("ancre")],
        "reperes": [s for s in calage["scenarios"] if s.get("ancre")],
        "observe": {
            "time": [d.isoformat() for d in q_amont.index],
            "q_amont": [round(float(v), 3) for v in q_amont.to_numpy()],
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
