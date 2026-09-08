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


def _prevision_precedente():
    """Champs de prevision de la derniere execution, encore valables.

    Un refus temporaire du fournisseur ne doit pas vider le panneau : mieux
    vaut un champ d'il y a six heures, date comme tel, que rien du tout.
    """
    chemin = os.path.join(DOCS, "tableau.json")
    if not os.path.exists(chemin):
        return []
    try:
        with open(chemin, encoding="utf-8") as fh:
            ancien = json.load(fh)["animation"]["images"]
    except Exception:
        return []
    maintenant = pd.Timestamp.now("UTC").tz_localize(None)
    return [i for i in ancien
            if i.get("type") == "prevu" and pd.Timestamp(i["t"]) > maintenant]


def _fond_carte():
    """Reperes ponctuels de l'animation.

    Le fond de carte lui-meme vient des tuiles IGN, chargees par la page : les
    dessiner a partir de vecteurs embarques coutait 60 ko pour un resultat
    moins lisible qu'un Plan IGN, et sans photo aerienne.
    """
    return {"cible": {"n": "Rochereau", "lon": -0.99276, "lat": 47.000408}}


def animation_radar(fin: pd.Timestamp, heures: int = HEURES_PASSEES) -> dict:
    """Vignettes radar des dernieres heures, pour l'animation sur le bassin.

    Chaque pas de temps ou il a plu porte une grille 20 x 20 quantifiee ; les
    pas secs n'en portent pas et sont restitues comme des images vides.
    """
    bassin = json.load(open(os.path.join(BASE, "bassins.json"), encoding="utf-8"))[BASSIN_RADAR]
    em = np.asarray(bassin["emprise"], dtype=float)
    cadre = {"lon0": float(em[:, 0].min()), "lon1": float(em[:, 0].max()),
             "lat0": float(em[:, 1].min()), "lat1": float(em[:, 1].max()),
             "facteur": 24.0, "n": 24}
    contour = [[round(float(x), 4), round(float(y), 4)] for x, y in em[::3]]
    prevues_vide = prevision_grille(cadre) or _prevision_precedente()
    vide = {"images": prevues_vide, "emprise": cadre, "contour": contour,
            "fond": _fond_carte(), "facteur": 24.0, "n_radar": 0}

    fichiers = sorted(glob.glob(os.path.join(BASE, "donnees", "radar", BASSIN_RADAR, "*.csv")))
    if not fichiers:
        return vide
    lot = pd.concat([pd.read_csv(f, parse_dates=["instant_utc"]) for f in fichiers])
    if "grille" not in lot.columns:
        return vide
    lot = lot.set_index(pd.DatetimeIndex(lot["instant_utc"]).tz_convert(None)).sort_index()
    lot = lot.loc[fin - pd.Timedelta(hours=heures):fin]
    lot = lot[~lot.index.duplicated(keep="last")]

    images = []
    for instant, ligne in lot.iterrows():
        g = ligne.get("grille")
        images.append({
            "t": instant.isoformat(),
            "moy": None if pd.isna(ligne["lame_mm"]) else round(float(ligne["lame_mm"]), 4),
            "max": None if pd.isna(ligne["lame_max_mm"]) else round(float(ligne["lame_max_mm"]), 3),
            "pt": None if "pt_rochereau" not in lot.columns or pd.isna(ligne.get("pt_rochereau"))
                  else round(float(ligne["pt_rochereau"]), 3),
            "g": "" if (g is None or (isinstance(g, float) and pd.isna(g))) else str(g),
            "type": "radar",
        })
    prevues = prevision_grille(cadre)
    if not prevues:
        prevues = _prevision_precedente()
    images += prevues
    return {"images": images, "emprise": cadre, "contour": contour,
            "fond": _fond_carte(),
            "facteur": 24.0, "n_radar": sum(1 for i in images if i.get("type") == "radar")}


def prevision_grille(cadre, heures: int = 96) -> list:
    """Champs de pluie prevue sur le bassin, a la maille de la vignette.

    Preleve la prevision AROME/ARPEGE sur la meme grille que la vignette radar
    — 24 x 24 points, soit 1,2 km — et l'encode a l'identique, pour que
    l'animation enchaine le passe mesure et l'avenir prevu dans la meme unite :
    l'intensite en millimetres par heure. Le prelevement se fait par lots :
    l'URL sature au-dela de quelques centaines de coordonnees, et le quota par
    minute d'Open-Meteo se declenche vite sur un champ de mille points.
    """
    import radar as radar_mf

    n = cadre["n"]
    lons = np.linspace(cadre["lon0"], cadre["lon1"], n)
    lats = np.linspace(cadre["lat1"], cadre["lat0"], n)   # du nord au sud, comme l'image
    LO, LA = np.meshgrid(lons, lats)
    # Open-Meteo compte UN APPEL PAR POINT : un champ de mille points depasse
    # d'un coup la limite de six cents appels par minute. On preleve donc par
    # lots espaces, sous le plafond, plutot que de se faire refuser en bloc.
    lat_p, lon_p = LA.ravel(), LO.ravel()
    lot = []
    for d in range(0, len(lat_p), TAILLE_LOT):
        if d:
            time.sleep(PAUSE_LOT)
        tranche = slice(d, d + TAILLE_LOT)
        rep = None
        for essai in range(3):
            try:
                rep = requests.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={"latitude": ",".join(f"{v:.4f}" for v in lat_p[tranche]),
                            "longitude": ",".join(f"{v:.4f}" for v in lon_p[tranche]),
                            "hourly": "precipitation", "models": "meteofrance_seamless",
                            "forecast_days": max(1, min(int(np.ceil(heures / 24)), 4)),
                            "timezone": "UTC"},
                    headers={"User-Agent": "collecte-sevre-nantaise/2.0"}, timeout=120)
                if rep.status_code == 429:
                    # Quota par minute : un champ de mille points pese lourd.
                    time.sleep(62)
                    continue
                rep.raise_for_status()
                break
            except Exception:
                time.sleep(8 * (essai + 1))
                rep = None
        if rep is None or rep.status_code != 200:
            return []
        part = rep.json()
        lot += part if isinstance(part, list) else [part]
    if not lot:
        return []
    temps = pd.to_datetime(lot[0]["hourly"]["time"])
    champ = np.array([x["hourly"]["precipitation"] for x in lot], dtype=float)
    champ = np.nan_to_num(champ, nan=0.0).reshape(n, n, len(temps))

    images = []
    maintenant = pd.Timestamp.now("UTC").tz_localize(None)
    for k, t in enumerate(temps):
        if t <= maintenant:
            continue
        grille = champ[:, :, k]
        images.append({
            "t": t.isoformat(), "type": "prevu",
            "moy": round(float(grille.mean()), 3),
            "max": round(float(grille.max()), 2),
            "g": radar_mf.encoder_vignette(grille) if grille.max() > 0.005 else "",
        })
    return images


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
            "z_rochereau": prevision["z_rochereau"],
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
        "animation": animation_radar(t0),
        "periodes_retour": {"maison": seuils["T_maison"], "atelier": seuils["T_atelier"]},
    }
