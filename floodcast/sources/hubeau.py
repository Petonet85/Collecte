"""Connecteurs Hub'Eau : hydrometrie (temps reel + historique) et piezometrie."""
from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd

from ..http import get_json, get_paginated

HYDRO = "https://hubeau.eaufrance.fr/api/v2/hydrometrie"
NAPPES = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes"

# --------------------------------------------------------------------------- #
# Referentiel
# --------------------------------------------------------------------------- #


def site(code_site: str) -> dict:
    data = get_json(f"{HYDRO}/referentiel/sites", {"code_site": code_site}, ttl=86400)["data"]
    if not data:
        raise ValueError(f"site hydrometrique inconnu : {code_site}")
    return data[0]


def station(code_station: str) -> dict:
    data = get_json(
        f"{HYDRO}/referentiel/stations", {"code_station": code_station}, ttl=86400
    )["data"]
    if not data:
        raise ValueError(f"station hydrometrique inconnue : {code_station}")
    return data[0]


def resolve(code: str) -> tuple[dict, dict]:
    """Accepte un code station (10 car.) ou site (8 car.) et renvoie (station, site)."""
    code = code.strip().upper()
    if len(code) >= 10:
        st = station(code)
        return st, site(st["code_site"])
    si = site(code)
    stations = get_json(
        f"{HYDRO}/referentiel/stations", {"code_site": code, "en_service": "true"}, ttl=86400
    )["data"]
    if not stations:
        stations = get_json(
            f"{HYDRO}/referentiel/stations", {"code_site": code}, ttl=86400
        )["data"]
    if not stations:
        raise ValueError(f"aucune station rattachee au site {code}")
    return stations[0], si


def stations_in_bbox(lon: float, lat: float, radius_km: float) -> pd.DataFrame:
    """Stations hydrometriques dans une boite englobante centree sur (lon, lat)."""
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(math.cos(math.radians(lat)), 0.2))
    bbox = f"{lon - dlon:.4f},{lat - dlat:.4f},{lon + dlon:.4f},{lat + dlat:.4f}"
    rows = get_paginated(
        f"{HYDRO}/referentiel/stations",
        {"bbox": bbox, "en_service": "true", "size": 1000, "format": "json"},
        ttl=86400,
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.dropna(subset=["longitude_station", "latitude_station"])


def sites_info(codes: list[str]) -> pd.DataFrame:
    """Infos site (dont surface_bv) pour une liste de codes site."""
    out = []
    for i in range(0, len(codes), 40):
        chunk = ",".join(codes[i : i + 40])
        out += get_paginated(
            f"{HYDRO}/referentiel/sites", {"code_site": chunk, "size": 1000}, ttl=86400
        )
    return pd.DataFrame(out) if out else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Observations temps reel (fenetre glissante ~30 jours)
# --------------------------------------------------------------------------- #


def observations_tr(code_entite: str, grandeur: str = "H", days: int = 30) -> pd.Series:
    """Serie temps reel. H en mm -> m ; Q en L/s -> m3/s. Index UTC tz-naive."""
    start = (dt.datetime.now(dt.UTC) - dt.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = get_paginated(
        f"{HYDRO}/observations_tr",
        {
            "code_entite": code_entite,
            "grandeur_hydro": grandeur,
            "date_debut_obs": start,
            "sort": "asc",
            "size": 5000,
            "fields": "date_obs,resultat_obs",
        },
        ttl=600,
        max_pages=80,
    )
    if not rows:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], name="date"))
    df = pd.DataFrame(rows)
    idx = pd.to_datetime(df["date_obs"], utc=True, format="mixed").dt.tz_localize(None)
    ser = pd.Series(df["resultat_obs"].astype(float).values / 1000.0, index=idx, name=grandeur)
    return ser[~ser.index.duplicated(keep="last")].sort_index()


def hourly(ser: pd.Series) -> pd.Series:
    """Reechantillonnage horaire (moyenne) d'une serie brute a pas variable."""
    if ser.empty:
        return ser
    return ser.resample("1h").mean()


# --------------------------------------------------------------------------- #
# Historique elabore (debit moyen journalier, tout l'historique de la station)
# --------------------------------------------------------------------------- #


def debits_journaliers(code_entite: str, start: str = "1990-01-01") -> pd.Series:
    """QmnJ : debit moyen journalier en m3/s (Hub'Eau renvoie des L/s)."""
    rows = get_paginated(
        f"{HYDRO}/obs_elab",
        {
            "code_entite": code_entite,
            "grandeur_hydro_elab": "QmnJ",
            "date_debut_obs_elab": start,
            "sort": "asc",
            "size": 5000,
            "fields": "date_obs_elab,resultat_obs_elab",
        },
        ttl=86400,
        max_pages=40,
    )
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    idx = pd.to_datetime(df["date_obs_elab"], format="mixed").dt.tz_localize(None)
    ser = pd.Series(df["resultat_obs_elab"].astype(float).values / 1000.0, index=idx, name="Q")
    ser = ser[~ser.index.duplicated(keep="last")].sort_index()
    return ser.where(ser >= 0)


# --------------------------------------------------------------------------- #
# Piezometrie (niveaux de nappe)
# --------------------------------------------------------------------------- #


def piezos_in_bbox(lon: float, lat: float, radius_km: float) -> pd.DataFrame:
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(math.cos(math.radians(lat)), 0.2))
    rows = get_paginated(
        f"{NAPPES}/stations",
        {
            "bbox": f"{lon - dlon:.4f},{lat - dlat:.4f},{lon + dlon:.4f},{lat + dlat:.4f}",
            "size": 1000,
        },
        ttl=86400,
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in ("x", "y", "longitude", "latitude"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def piezo_chronique(code_bss: str, days: int = 400) -> pd.Series:
    """Niveau de nappe (m NGF). Fusionne l'historique valide et le temps reel."""
    start = (dt.datetime.now(dt.UTC) - dt.timedelta(days=days)).strftime("%Y-%m-%d")
    hist = get_paginated(
        f"{NAPPES}/chroniques",
        {"code_bss": code_bss, "date_debut_mesure": start, "size": 5000,
         "fields": "date_mesure,niveau_nappe_eau"},
        ttl=86400,
        max_pages=20,
    )
    tr = get_paginated(
        f"{NAPPES}/chroniques_tr",
        {"code_bss": code_bss, "size": 5000, "fields": "date_mesure,niveau_eau_ngf"},
        ttl=3600,
        max_pages=20,
    )
    frames = []
    if hist:
        d = pd.DataFrame(hist).rename(columns={"niveau_nappe_eau": "h"})
        frames.append(d[["date_mesure", "h"]])
    if tr:
        d = pd.DataFrame(tr).rename(columns={"niveau_eau_ngf": "h"})
        frames.append(d[["date_mesure", "h"]])
    if not frames:
        return pd.Series(dtype=float)
    df = pd.concat(frames, ignore_index=True).dropna()
    idx = pd.to_datetime(df["date_mesure"], format="mixed", utc=True).dt.tz_localize(None)
    ser = pd.Series(pd.to_numeric(df["h"], errors="coerce").values, index=idx, name="nappe")
    ser = ser.dropna()
    ser = ser[~ser.index.duplicated(keep="last")].sort_index()
    return ser.resample("1D").mean().interpolate(limit=15)


def haversine_km(lon1: float, lat1: float, lon2, lat2):
    r = 6371.0
    p1, p2 = math.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2, dtype=float) - lon1)
    a = np.sin(dp / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def catalogue(en_service: bool = True) -> pd.DataFrame:
    """Catalogue national des stations hydrometriques (mis en cache 24 h)."""
    rows = get_paginated(
        f"{HYDRO}/referentiel/stations",
        {"en_service": "true" if en_service else "false", "size": 1000, "format": "json",
         "fields": "code_station,libelle_station,code_site,libelle_cours_eau,"
                   "libelle_commune,code_departement,libelle_departement,"
                   "longitude_station,latitude_station"},
        ttl=86400,
        max_pages=20,
    )
    return pd.DataFrame(rows)


def rechercher(texte: str = "", departement: str | None = None,
               lon: float | None = None, lat: float | None = None,
               rayon_km: float = 30.0, limite: int = 25) -> pd.DataFrame:
    """Recherche une station par texte libre, departement et/ou proximite."""
    df = catalogue()
    if df.empty:
        return df
    if departement:
        df = df[df["code_departement"].astype(str) == str(departement).zfill(2)]
    if texte:
        key = _normalise(texte)
        cols = ["libelle_station", "libelle_cours_eau", "libelle_commune"]
        blob = df[cols].fillna("").agg(" ".join, axis=1).map(_normalise)
        df = df[blob.str.contains(key, regex=False)]
    if lon is not None and lat is not None:
        df = df.dropna(subset=["longitude_station", "latitude_station"]).copy()
        df["dist_km"] = haversine_km(lon, lat, df["longitude_station"], df["latitude_station"])
        df = df[df["dist_km"] <= rayon_km].sort_values("dist_km")
    return df.head(limite).reset_index(drop=True)


def _normalise(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFD", str(s))
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
