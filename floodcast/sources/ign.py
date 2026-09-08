"""Geoplateforme IGN : altimetrie RGE ALTI (1 m, issu du LiDAR HD) et BD TOPO."""
from __future__ import annotations

import numpy as np

from ..http import get_json

ALTI = "https://data.geopf.fr/altimetrie/1.0/calcul/alti/rest/elevation.json"
WFS = "https://data.geopf.fr/wfs/ows"
CHUNK = 180  # limite pratique de l'API GET (longueur d'URL)


def elevations(lons, lats, resource: str = "ign_rge_alti_wld") -> np.ndarray:
    """Altitudes RGE ALTI (m) pour des points quelconques, par paquets."""
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    out = np.full(lons.shape, np.nan)
    for i in range(0, len(lons), CHUNK):
        sl = slice(i, i + CHUNK)
        try:
            payload = get_json(
                ALTI,
                {
                    "lon": "|".join(f"{v:.5f}" for v in lons[sl]),
                    "lat": "|".join(f"{v:.5f}" for v in lats[sl]),
                    "resource": resource,
                    "zonly": "true",
                },
                ttl=30 * 86400,
                timeout=120,
            )
        except Exception:  # noqa: BLE001 - hors metropole ou service indisponible
            continue
        vals = np.asarray(payload.get("elevations", []), dtype=float)
        if len(vals) == len(lons[sl]):
            out[sl] = vals
    out[out < -1000] = np.nan  # -99999 = hors emprise
    return out


def troncons_hydro(lon: float, lat: float, radius_km: float, max_features: int = 400) -> list[dict]:
    """Troncons du reseau hydrographique BD TOPO autour d'un point (GeoJSON)."""
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(np.cos(np.radians(lat)), 0.2))
    try:
        payload = get_json(
            WFS,
            {
                "SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
                "TYPENAMES": "BDTOPO_V3:troncon_hydrographique",
                "SRSNAME": "EPSG:4326", "OUTPUTFORMAT": "application/json",
                "COUNT": max_features,
                # Ce service attend lon,lat malgre SRSNAME=EPSG:4326, dont l'ordre
                # d'axes normalise est lat,lon : l'inverser rend zero entite,
                # sans erreur ni avertissement.
                "BBOX": f"{lon - dlon:.5f},{lat - dlat:.5f},{lon + dlon:.5f},{lat + dlat:.5f},EPSG:4326",
            },
            ttl=30 * 86400,
            timeout=120,
        )
    except Exception:  # noqa: BLE001
        return []
    return payload.get("features", []) or []
