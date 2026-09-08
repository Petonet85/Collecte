"""Client HTTP mutualise : cache disque, retry, pagination Hub'Eau."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from urllib.parse import urlsplit
from typing import Any

import requests

CACHE_DIR = os.environ.get(
    "FLOODCAST_CACHE",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache"),
)
USER_AGENT = "floodcast/0.1 (prevision de crue; contact: local)"

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

# Les services publics imposent des quotas. On s'auto-limite par hote plutot que
# d'attendre le 429 : une chaine de prevision qui se fait bannir ne previent rien.
MIN_INTERVAL = {
    "historical-forecast-api.open-meteo.com": 2.5,
    "archive-api.open-meteo.com": 1.0,
    "ensemble-api.open-meteo.com": 0.6,
    "api.open-meteo.com": 0.4,
    "data.geopf.fr": 0.2,
}
_last_call: dict[str, float] = {}
_lock = threading.Lock()


def _throttle(url: str) -> None:
    host = urlsplit(url).netloc
    wait = MIN_INTERVAL.get(host, 0.0)
    if not wait:
        return
    with _lock:
        elapsed = time.time() - _last_call.get(host, 0.0)
        if elapsed < wait:
            time.sleep(wait - elapsed)
        _last_call[host] = time.time()


def _cache_path(url: str, params: dict | None) -> str:
    key = hashlib.sha256((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()[:32]
    return os.path.join(CACHE_DIR, key + ".json")


def get_json(
    url: str,
    params: dict | None = None,
    *,
    ttl: float = 900.0,
    retries: int = 3,
    timeout: float = 60.0,
) -> Any:
    """GET JSON avec cache disque (ttl en secondes, 0 = pas de cache)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(url, params)
    if ttl > 0 and os.path.exists(path) and time.time() - os.path.getmtime(path) < ttl:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            _throttle(url)
            resp = _session.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                # Quota atteint : on respecte Retry-After, sinon backoff exponentiel.
                delay = float(resp.headers.get("Retry-After") or 0) or 6.0 * (3 ** attempt)
                time.sleep(min(delay, 90.0))
                raise requests.HTTPError("HTTP 429 (quota)")
            if resp.status_code in (502, 503, 504):
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 - on retente puis on retombe sur le cache
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
            continue
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        return data

    # Degradation gracieuse : on sert un cache perime plutot que de planter une prevision.
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    raise RuntimeError(f"echec GET {url} params={params}: {last_err}")


def get_paginated(url: str, params: dict, *, max_pages: int = 60, ttl: float = 900.0) -> list[dict]:
    """Suit la pagination `next` des APIs Hub'Eau (curseur ou page)."""
    out: list[dict] = []
    payload = get_json(url, params, ttl=ttl)
    for _ in range(max_pages):
        out.extend(payload.get("data") or [])
        nxt = payload.get("next")
        if not nxt or not payload.get("data"):
            break
        payload = get_json(nxt, None, ttl=ttl)
    return out
