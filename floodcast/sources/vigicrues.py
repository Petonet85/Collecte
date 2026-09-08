"""API Vigicrues : reference officielle, utilisee comme source ET comme temoin.

Trois apports :
  * `CruesHistoriques` fournit des couples (hauteur, debit) de crue, qui ancrent
    la courbe de tarage la ou les 30 jours de temps reel ne vont jamais ;
  * `StationsBassin` donne la composition officielle du bassin, plus fiable
    qu'un filtre geographique ;
  * les previsions et la couleur de vigilance servent de point de comparaison.
"""
from __future__ import annotations

import pandas as pd

from ..http import get_json

BASE = "https://www.vigicrues.gouv.fr/services"


def station(code_station: str) -> dict:
    try:
        return get_json(f"{BASE}/station.json/index.php",
                        {"CdStationHydro": code_station}, ttl=7 * 86400)
    except Exception:  # noqa: BLE001 - station hors perimetre Vigicrues
        return {}


def crues_historiques(code_station: str) -> pd.DataFrame:
    """Couples (hauteur m, debit m3/s) des crues de reference."""
    meta = station(code_station).get("VigilanceCrues") or {}
    rows = meta.get("CruesHistoriques") or []
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.rename(columns={"LbUsuel": "libelle", "ValHauteur": "h", "ValDebit": "q"})
    keep = [c for c in ("libelle", "h", "q") if c in df.columns]
    df = df[keep].dropna()
    return df.sort_values("h").reset_index(drop=True) if "h" in df else df


def stations_bassin(code_station: str) -> pd.DataFrame:
    meta = station(code_station).get("VigilanceCrues") or {}
    rows = meta.get("StationsBassin") or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(
        columns={"CdStationHydro": "code_station", "LbStationHydro": "libelle",
                 "LbCoursEau": "cours_eau"})
    return df[["code_station", "libelle", "cours_eau"]]


def observations(code_station: str, grandeur: str = "H") -> pd.Series:
    """Serie observee publiee par Vigicrues (H en m, Q en m3/s)."""
    try:
        payload = get_json(f"{BASE}/observations.json/index.php",
                           {"CdStationHydro": code_station, "GrdSerie": grandeur,
                            "FormatDate": "iso"}, ttl=900)
    except Exception:  # noqa: BLE001
        return pd.Series(dtype=float)
    obs = ((payload.get("Serie") or {}).get("ObssHydro")) or []
    if not obs:
        return pd.Series(dtype=float)
    df = pd.DataFrame(obs)
    idx = pd.to_datetime(df["DtObsHydro"], utc=True, format="mixed").dt.tz_localize(None)
    ser = pd.Series(pd.to_numeric(df["ResObsHydro"], errors="coerce").values, index=idx)
    return ser.dropna().sort_index()


def previsions(code_station: str) -> dict:
    """Prevision officielle (souvent vide hors episode) : {'grandeur','series':DataFrame}."""
    try:
        payload = get_json(f"{BASE}/previsions.json/index.php",
                           {"CdStationHydro": code_station}, ttl=900)
    except Exception:  # noqa: BLE001
        return {"grandeur": None, "date_production": None, "series": pd.DataFrame()}
    sim = payload.get("Simul") or {}
    prevs = sim.get("Prevs") or []
    rows = []
    for p in prevs:
        if isinstance(p, dict) and "DtPrev" in p:
            rows.append(p)
        elif isinstance(p, list):
            rows.extend([x for x in p if isinstance(x, dict) and "DtPrev" in x])
    df = pd.DataFrame(rows)
    if not df.empty:
        df["DtPrev"] = pd.to_datetime(df["DtPrev"], utc=True, format="mixed").dt.tz_localize(None)
    return {"grandeur": sim.get("GrdSimul"), "date_production": sim.get("DtProdSimul"),
            "series": df}


def vigilance(code_station: str) -> dict:
    """Couleur de vigilance du troncon auquel la station appartient."""
    meta = station(code_station).get("VigilanceCrues") or {}
    pere = meta.get("PereBoitEntVigiCru") or {}
    cd = pere.get("CdEntVigiCru")
    if not cd:
        return {}
    try:
        bul = get_json(f"{BASE}/bulletin.json/index.php", {"CdEntVigiCru": cd}, ttl=1800)
    except Exception:  # noqa: BLE001
        return {"CdEntVigiCru": cd}
    return {"CdEntVigiCru": cd, "bulletin": bul}
