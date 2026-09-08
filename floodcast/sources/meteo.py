"""Forcages meteorologiques (Open-Meteo) moyennes sur le bassin versant.

Trois usages :
  * `history`   : reanalyse ERA5-Land pour le calage long terme (pluie, ETP) ;
  * `recent`    : modele deterministe Meteo-France (AROME 1.5 km + ARPEGE) avec
                  les jours passes, pour l'initialisation des reservoirs ;
  * `ensemble`  : grand ensemble multi-modeles -> incertitude de la prevision.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..http import get_json

FORECAST = "https://api.open-meteo.com/v1/forecast"
ENSEMBLE = "https://ensemble-api.open-meteo.com/v1/ensemble"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

# 0-48 h a 2.2 km (orages), 0-120 h a 7 km, 0-15 j a 25 km : ils se completent.
ENSEMBLE_MODELS = ("icon_d2", "icon_eu", "ecmwf_ifs025")

HOURLY_VARS = "precipitation,et0_fao_evapotranspiration,temperature_2m,snow_depth"


def _as_list(payload) -> list[dict]:
    return payload if isinstance(payload, list) else [payload]


def _mean_frame(payloads: list[dict], columns: list[str]) -> pd.DataFrame:
    """Moyenne arithmetique des points d'echantillonnage -> pluie de bassin."""
    stacks: dict[str, list[np.ndarray]] = {c: [] for c in columns}
    index = None
    for p in payloads:
        h = p.get("hourly") or {}
        idx = pd.to_datetime(h["time"])
        if index is None:
            index = idx
        for c in columns:
            v = np.asarray(h.get(c, [np.nan] * len(idx)), dtype=float)
            s = pd.Series(v, index=idx).reindex(index)
            stacks[c].append(s.to_numpy())
    data = {}
    with np.errstate(invalid="ignore"):
        for c, v in stacks.items():
            if not v:
                continue
            stack = np.vstack(v)
            col = np.full(stack.shape[1], np.nan)
            ok = ~np.all(np.isnan(stack), axis=0)
            col[ok] = np.nanmean(stack[:, ok], axis=0)
            data[c] = col
    return pd.DataFrame(data, index=index)


def _daily_frame(payloads: list[dict], columns: list[str]) -> pd.DataFrame:
    """Moyenne des points pour un bloc `daily` (meme logique que `_mean_frame`)."""
    stacks: dict[str, list[np.ndarray]] = {c: [] for c in columns}
    index = None
    for p in payloads:
        d = p.get("daily") or {}
        if "time" not in d:
            continue
        idx = pd.to_datetime(d["time"])
        index = idx if index is None else index
        for c in columns:
            v = np.asarray(d.get(c, [np.nan] * len(idx)), dtype=float)
            stacks[c].append(pd.Series(v, index=idx).reindex(index).to_numpy())
    if index is None:
        return pd.DataFrame()
    data = {}
    with np.errstate(invalid="ignore"):
        for c, v in stacks.items():
            if not v:
                continue
            stack = np.vstack(v)
            col = np.full(stack.shape[1], np.nan)
            ok = ~np.all(np.isnan(stack), axis=0)
            col[ok] = np.nanmean(stack[:, ok], axis=0)
            data[c] = col
    return pd.DataFrame(data, index=index)


def _coords(points: list[tuple[float, float]]) -> dict:
    return {
        "latitude": ",".join(f"{lat:.4f}" for _, lat in points),
        "longitude": ",".join(f"{lon:.4f}" for lon, _ in points),
    }


def history(points: list[tuple[float, float]], start: str, end: str) -> pd.DataFrame:
    """Reanalyse horaire moyennee sur le bassin (mm, mm, degC, m).

    `best_match` combine ERA5 (pluie, ETP) et ERA5-Land (temperature, neige) :
    ERA5-Land seul ne sert pas la pluie via cette API.
    """
    cols = HOURLY_VARS.split(",")
    payload = get_json(
        ARCHIVE,
        {**_coords(points), "start_date": start, "end_date": end,
         "hourly": HOURLY_VARS, "models": "best_match", "timezone": "UTC"},
        ttl=7 * 86400,
        timeout=180,
    )
    df = _mean_frame(_as_list(payload), cols)
    return df.rename(columns={"precipitation": "P", "et0_fao_evapotranspiration": "E"})


def recent(points: list[tuple[float, float]], past_days: int = 60,
           forecast_days: int = 4) -> pd.DataFrame:
    """Pluie observee/analysee recente + prevision deterministe Meteo-France."""
    cols = HOURLY_VARS.split(",")
    payload = get_json(
        FORECAST,
        {**_coords(points), "hourly": HOURLY_VARS, "models": "meteofrance_seamless",
         "past_days": min(past_days, 92), "forecast_days": min(forecast_days, 16),
         "timezone": "UTC"},
        ttl=1800,
        timeout=120,
    )
    df = _mean_frame(_as_list(payload), cols)
    return df.rename(columns={"precipitation": "P", "et0_fao_evapotranspiration": "E"})


def ensemble(points: list[tuple[float, float]], forecast_days: int = 5,
             models: tuple[str, ...] = ENSEMBLE_MODELS) -> pd.DataFrame:
    """Grand ensemble de pluie de bassin : colonnes = membres, index = heures UTC.

    Les membres plus courts (ICON-D2 : 48 h) sont prolonges par la moyenne du
    modele le plus long, ce qui evite de sous-estimer les cumuls a longue echeance.
    """
    pts = points[:6] if len(points) > 6 else points
    frames: list[pd.DataFrame] = []
    for model in models:
        try:
            payload = get_json(
                ENSEMBLE,
                {**_coords(pts), "hourly": "precipitation", "models": model,
                 "forecast_days": min(forecast_days, 16), "timezone": "UTC"},
                ttl=1800,
                timeout=120,
            )
        except Exception:  # noqa: BLE001 - un modele indisponible ne doit pas tout bloquer
            continue
        payloads = _as_list(payload)
        members = [k for k in payloads[0]["hourly"] if k.startswith("precipitation")]
        df = _mean_frame(payloads, members)
        df.columns = [f"{model}:{m.split('member')[-1] or '00'}" for m in members]
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    index = frames[int(np.argmax([len(f) for f in frames]))].index
    ref = pd.concat([f.reindex(index) for f in frames], axis=1).mean(axis=1)
    out = []
    for f in frames:
        f = f.reindex(index)
        out.append(f.apply(lambda col: col.fillna(ref)))
    grand = pd.concat(out, axis=1)
    return grand.clip(lower=0.0)


def sample_points(lon: float, lat: float, area_km2: float, n: int = 9,
                  bias_upstream: tuple[float, float] | None = None) -> list[tuple[float, float]]:
    """Grille d'echantillonnage couvrant un disque de surface equivalente au BV.

    `bias_upstream` (dlon, dlat normalises) decale le nuage de points vers l'amont
    lorsqu'on connait la direction du bassin ; sinon le disque est centre sur la station.
    """
    radius_km = float(np.sqrt(max(area_km2, 1.0) / np.pi))
    radius_km = float(np.clip(radius_km, 3.0, 90.0))
    cx, cy = lon, lat
    if bias_upstream is not None:
        cx += bias_upstream[0] * radius_km / (111.0 * np.cos(np.radians(lat)))
        cy += bias_upstream[1] * radius_km / 111.0
    pts = [(round(cx, 4), round(cy, 4))]
    rings = [(0.55, 4), (1.0, max(n - 5, 4))]
    for frac, k in rings:
        for i in range(k):
            ang = 2 * np.pi * i / k + (0.4 if frac > 0.8 else 0.0)
            dx = frac * radius_km * np.cos(ang) / (111.0 * np.cos(np.radians(lat)))
            dy = frac * radius_km * np.sin(ang) / 111.0
            pts.append((round(cx + dx, 4), round(cy + dy, 4)))
    return pts[:n]


def history_daily(points: list[tuple[float, float]], start: str, end: str) -> pd.DataFrame:
    """Reanalyse journaliere (pluie, ETP, temperature) pour le calage long terme."""
    varis = "precipitation_sum,et0_fao_evapotranspiration,temperature_2m_mean"
    payload = get_json(
        ARCHIVE,
        {**_coords(points), "start_date": start, "end_date": end,
         "daily": varis, "models": "best_match", "timezone": "UTC"},
        ttl=7 * 86400,
        timeout=300,
    )
    cols = varis.split(",")
    stacks = {c: [] for c in cols}
    index = None
    for p in _as_list(payload):
        d = p.get("daily") or {}
        idx = pd.to_datetime(d["time"])
        index = idx if index is None else index
        for c in cols:
            stacks[c].append(pd.Series(np.asarray(d.get(c, []), dtype=float),
                                       index=idx).reindex(index).to_numpy())
    data = {}
    with np.errstate(invalid="ignore"):
        for c, v in stacks.items():
            stack = np.vstack(v)
            col = np.full(stack.shape[1], np.nan)
            ok = ~np.all(np.isnan(stack), axis=0)
            col[ok] = np.nanmean(stack[:, ok], axis=0)
            data[c] = col
    df = pd.DataFrame(data, index=index)
    return df.rename(columns={"precipitation_sum": "P", "et0_fao_evapotranspiration": "E",
                              "temperature_2m_mean": "T"})


HISTORICAL_FORECAST = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def past_forecast_daily(points: list[tuple[float, float]], start: str, end: str,
                        lead_days: int, model: str = "meteofrance_seamless",
                        budget_s: float = 180.0) -> pd.Series:
    """Pluie de bassin telle qu'elle etait *prevue* `lead_days` jours a l'avance.

    Open-Meteo archive les anciens runs : `precipitation_previous_dayN` restitue,
    pour chaque heure, la valeur issue du run emis N jours plus tot. C'est la seule
    facon d'evaluer honnetement une chaine de prevision sur des crues passees —
    sinon on se juge avec une pluie que l'on n'aurait jamais connue a temps.
    """
    var = f"precipitation_previous_day{int(lead_days)}"
    pts = points[:3] if len(points) > 3 else points
    frames = []
    t_fin = time.monotonic() + budget_s
    for y0, y1 in _year_chunks(max(start, OPERATIONAL_START), end):
        if time.monotonic() > t_fin:
            break
        try:
            payload = get_json(
                HISTORICAL_FORECAST,
                {**_coords(pts), "start_date": y0, "end_date": y1,
                 "hourly": var, "models": model, "timezone": "UTC"},
                ttl=30 * 86400,
                retries=2,
                timeout=110,
            )
        except Exception:  # noqa: BLE001
            continue
        part = _mean_frame(_as_list(payload), [var])
        if not part.empty and var in part.columns:
            frames.append(part[var])
    if not frames:
        return pd.Series(dtype=float)
    ser = pd.concat(frames).sort_index()
    ser = ser[~ser.index.duplicated(keep="last")]
    return ser.resample("1D").sum(min_count=1).rename(f"P_lead{lead_days}")


OPERATIONAL_START = "2023-01-01"   # profondeur de l'archive des runs Meteo-France


def _year_chunks(start: str, end: str) -> list[tuple[str, str]]:
    """Decoupe [start, end] en tranches annuelles."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    cur = s
    while cur < e:
        nxt = min(pd.Timestamp(year=cur.year + 1, month=1, day=1) - pd.Timedelta(days=1), e)
        out.append((cur.date().isoformat(), nxt.date().isoformat()))
        cur = nxt + pd.Timedelta(days=1)
    return out


def operational_history_daily(points: list[tuple[float, float]], start: str, end: str,
                              model: str = "meteofrance_seamless",
                              budget_s: float = 240.0) -> pd.DataFrame:
    """Pluie journaliere issue des runs Meteo-France archives (AROME/ARPEGE).

    C'est le forcage reellement utilise en operationnel. Il differe nettement
    d'ERA5 sur les episodes convectifs en relief : caler le modele sur l'un puis
    le faire tourner avec l'autre introduit un biais systematique.
    """
    # Variables deja agregees a la journee et 3 points seulement : l'archive des
    # runs est facturee au volume, une requete horaire multi-points epuise le quota.
    varis = "precipitation_sum,et0_fao_evapotranspiration_sum"
    pts = points[:3] if len(points) > 3 else points
    # Requetes annuelles : le service archive est lent et coupe les gros intervalles.
    frames = []
    t_fin = time.monotonic() + budget_s
    for y0, y1 in _year_chunks(max(start, OPERATIONAL_START), end):
        if time.monotonic() > t_fin:
            break   # budget epuise : on se contente des annees deja obtenues
        try:
            payload = get_json(
                HISTORICAL_FORECAST,
                {**_coords(pts), "start_date": y0, "end_date": y1,
                 "daily": varis, "models": model, "timezone": "UTC"},
                ttl=30 * 86400,
                retries=2,
                timeout=110,
            )
        except Exception:  # noqa: BLE001
            continue
        part = _daily_frame(_as_list(payload), varis.split(","))
        if not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    out = df.rename(columns={"precipitation_sum": "P",
                             "et0_fao_evapotranspiration_sum": "E"})
    return out.dropna(how="all")
