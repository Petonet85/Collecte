"""Caracterisation du bassin versant : emprise, relief, temps de reponse, voisins.

L'emprise est approchee sans calcul de directions d'ecoulement : on part de la
surface de bassin publiee par Hub'Eau (donnee de reference), puis on selectionne
sur une grille RGE ALTI les mailles topographiquement plausibles (situees au-dessus
de l'exutoire) les plus proches, jusqu'a atteindre cette surface. C'est suffisant
pour une pluie de bassin, et cela cale la direction amont sur le vrai relief.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .sources import hubeau as hb
from .sources import ign


@dataclass
class Basin:
    code_station: str
    code_site: str
    name: str
    river: str
    lon: float
    lat: float
    area_km2: float
    altitude_m: float
    cells: pd.DataFrame = field(default_factory=pd.DataFrame)
    meteo_points: list[tuple[float, float]] = field(default_factory=list)
    relief: dict = field(default_factory=dict)
    upstream: pd.DataFrame = field(default_factory=pd.DataFrame)
    piezos: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def tc_hours(self) -> float:
        return float(self.relief.get("tc_hours", 6.0))


def _grid(lon: float, lat: float, radius_km: float, n: int = 34):
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(math.cos(math.radians(lat)), 0.2))
    xs = np.linspace(lon - dlon, lon + dlon, n)
    ys = np.linspace(lat - dlat, lat + dlat, n)
    gx, gy = np.meshgrid(xs, ys)
    return gx.ravel(), gy.ravel()


def delineate(code: str, grid_n: int = 34, n_meteo_points: int = 10) -> Basin:
    """Construit l'objet Basin a partir d'un code station/site Hub'Eau."""
    st, si = hb.resolve(code)
    lon = float(st["longitude_station"])
    lat = float(st["latitude_station"])
    area = float(si.get("surface_bv") or 0.0)
    if area <= 0:
        area = 200.0  # defaut prudent si la surface n'est pas renseignee

    radius = float(np.clip(math.sqrt(area / math.pi) * 2.0, 6.0, 130.0))
    gx, gy = _grid(lon, lat, radius, grid_n)
    z = ign.elevations(gx, gy)
    # Altitude de l'exutoire au point exact de la station : la maille la plus proche
    # peut etre a flanc de versant et surestimer le fond de vallee de 100 m.
    z_out = float(ign.elevations([lon], [lat])[0])
    if not np.isfinite(z_out):
        z_out = float(np.nanmedian(z[np.argsort((gx - lon) ** 2 + (gy - lat) ** 2)[:4]]))
    if not np.isfinite(z_out):
        z_out = float(st.get("altitude_ref_alti_station") or 0.0)

    cell_km2 = (2 * radius / (grid_n - 1)) ** 2
    dist = hb.haversine_km(lon, lat, gx, gy)

    # Cout : proximite a l'exutoire, avec une penalite forte pour les mailles
    # situees en contrebas (elles ne peuvent pas drainer vers la station).
    drop = z - z_out
    cost = dist + np.where(drop < -5, 1000.0, 0.0) + np.maximum(-drop, 0) * 0.05
    cost = np.where(np.isfinite(z), cost, 1e6)

    order = np.argsort(cost)
    keep = order[: max(int(round(area / cell_km2)), 6)]
    keep = keep[cost[keep] < 1e5]
    cells = pd.DataFrame({"lon": gx[keep], "lat": gy[keep], "z": z[keep],
                          "dist_km": dist[keep]}).dropna()

    zc = cells["z"].to_numpy()
    length_km = float(cells["dist_km"].max()) if len(cells) else radius
    z_mean = float(np.nanmean(zc)) if len(zc) else z_out
    z_max = float(np.nanpercentile(zc, 97)) if len(zc) else z_out + 100
    relief = {
        "z_outlet": round(z_out, 1),
        "z_mean": round(z_mean, 1),
        "z_max": round(z_max, 1),
        "denivele_m": round(z_mean - z_out, 1),
        "longueur_km": round(length_km, 1),
        "pente_moy_pct": round(100 * (z_mean - z_out) / max(length_km * 1000, 1), 2),
        "cell_km2": round(cell_km2, 2),
        "tc_hours": round(_giandotti(area, length_km, z_mean, z_out), 1),
    }

    basin = Basin(
        code_station=st["code_station"], code_site=st["code_site"],
        name=st["libelle_station"], river=st.get("libelle_cours_eau") or "",
        lon=lon, lat=lat, area_km2=area, altitude_m=z_out,
        cells=cells, relief=relief,
    )
    basin.meteo_points = _representative_points(cells, lon, lat, n_meteo_points)
    return basin


def _giandotti(area_km2: float, length_km: float, z_mean: float, z_out: float) -> float:
    """Temps de concentration de Giandotti (h), borne pour rester physique."""
    dz = max(z_mean - z_out, 5.0)
    tc = (4 * math.sqrt(max(area_km2, 1.0)) + 1.5 * max(length_km, 0.5)) / (0.8 * math.sqrt(dz))
    return float(np.clip(tc, 1.0, 96.0))


def _representative_points(cells: pd.DataFrame, lon: float, lat: float,
                           n: int) -> list[tuple[float, float]]:
    """Sous-echantillonne l'emprise en n points (k-means leger, depart regulier)."""
    if cells.empty:
        return [(lon, lat)]
    pts = cells[["lon", "lat"]].to_numpy()
    if len(pts) <= n:
        return [(round(x, 4), round(y, 4)) for x, y in pts]
    rng = np.random.default_rng(0)
    centers = pts[rng.choice(len(pts), n, replace=False)]
    for _ in range(25):
        d = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        lab = d.argmin(axis=1)
        new = np.array([pts[lab == k].mean(axis=0) if np.any(lab == k) else centers[k]
                        for k in range(n)])
        if np.allclose(new, centers, atol=1e-5):
            break
        centers = new
    return [(round(float(x), 4), round(float(y), 4)) for x, y in centers]


# --------------------------------------------------------------------------- #
# Voisinage : stations amont candidates et piezometres
# --------------------------------------------------------------------------- #


def find_upstream_candidates(basin: Basin, max_n: int = 12) -> pd.DataFrame:
    """Stations hydrometriques potentiellement en amont (filtre geo + surface).

    Le filtre est volontairement large : c'est la correlation croisee des
    chroniques (model/propagation.py) qui tranchera sur l'appartenance reelle.
    """
    radius = float(np.clip(math.sqrt(basin.area_km2 / math.pi) * 2.2, 15.0, 140.0))
    df = hb.stations_in_bbox(basin.lon, basin.lat, radius)
    if df.empty:
        return pd.DataFrame()
    df = df[df["code_station"] != basin.code_station].copy()
    df["dist_km"] = hb.haversine_km(basin.lon, basin.lat,
                                    df["longitude_station"], df["latitude_station"])
    info = hb.sites_info(sorted(df["code_site"].dropna().unique().tolist()))
    if not info.empty and "surface_bv" in info.columns:
        df = df.merge(info[["code_site", "surface_bv"]].drop_duplicates("code_site"),
                      on="code_site", how="left")
    else:
        df["surface_bv"] = np.nan

    z = ign.elevations(df["longitude_station"].to_numpy(), df["latitude_station"].to_numpy())
    df["z"] = z
    # Un affluent/amont a un bassin plus petit et se situe plus haut que l'exutoire.
    df = df[(df["surface_bv"].isna()) | (df["surface_bv"] < 0.95 * basin.area_km2)]
    df = df[(df["z"].isna()) | (df["z"] >= basin.altitude_m - 8)]
    df["score"] = df["dist_km"] / max(radius, 1) - 0.3 * (df["surface_bv"].fillna(0) / max(basin.area_km2, 1))
    return df.sort_values("score").head(max_n).reset_index(drop=True)


def find_piezos(basin: Basin, max_n: int = 6) -> pd.DataFrame:
    """Piezometres les plus proches disposant de mesures recentes."""
    radius = float(np.clip(math.sqrt(basin.area_km2 / math.pi) * 1.8, 15.0, 90.0))
    df = hb.piezos_in_bbox(basin.lon, basin.lat, radius)
    if df.empty:
        return pd.DataFrame()
    lon_col = "x" if "x" in df.columns else "longitude"
    lat_col = "y" if "y" in df.columns else "latitude"
    df = df.dropna(subset=[lon_col, lat_col]).copy()
    df["dist_km"] = hb.haversine_km(basin.lon, basin.lat, df[lon_col], df[lat_col])
    if "date_fin_mesure" in df.columns:
        fin = pd.to_datetime(df["date_fin_mesure"], errors="coerce", format="mixed")
        recent = fin > (pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=120))
        df = df[recent.fillna(False)]
    df = df.rename(columns={lon_col: "lon", lat_col: "lat"})
    return df.sort_values("dist_km").head(max_n).reset_index(drop=True)
