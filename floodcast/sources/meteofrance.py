"""Donnees climatologiques Meteo-France (API DPClim) : la pluie *mesuree*.

Le modele est cale sur ERA5 faute de mieux : c'est une reanalyse a 25 km, qui
lisse severement les episodes convectifs. Sur la Loire amont, le 17 octobre 2024,
elle voyait 25 mm quand AROME en annoncait 85. Les postes au sol donnent la
seule pluie reellement observee, sur plusieurs decennies : de quoi recaler le
modele sans attendre des mois d'archivage radar.

L'API fonctionne par commande asynchrone :
    1. `commander(...)`   -> numero de commande
    2. `telecharger(...)` -> le CSV, des qu'il est pret (on patiente)

La cle se lit dans METEOFRANCE_API_KEY, ou a defaut dans ~/.config/floodcast/env.
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import threading
import time

import numpy as np
import pandas as pd
import requests

BASE = "https://public-api.meteofrance.fr/public/DPClim/v1"
FREQUENCES = ("infrahoraire-6m", "horaire", "quotidienne", "decadaire", "mensuelle")

_session = requests.Session()
_session.headers.update({"Accept": "*/*", "User-Agent": "floodcast/0.2"})

# DPClim plafonne a 50 requetes par minute (le palier figure dans le jeton).
# Une serie de 30 ans en demande une centaine : sans regulation on depasse en
# quelques secondes, et l'API repond alors 429 pendant plusieurs minutes.
INTERVALLE_MIN = 60.0 / 45          # marge sous les 50/min annonces
_dernier_appel = 0.0
_verrou = threading.Lock()

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "..", "data", "cache_dpclim")


def _reguler():
    global _dernier_appel
    with _verrou:
        ecart = time.time() - _dernier_appel
        if ecart < INTERVALLE_MIN:
            time.sleep(INTERVALLE_MIN - ecart)
        _dernier_appel = time.time()


def _attente_429(message: str) -> float:
    """L'API indique dans son refus l'instant a partir duquel elle repondra."""
    trouve = re.search(r"after ([0-9]{4})-([^\s-]+)-?([0-9]{1,2})?[^0-9]*([0-9]{2}):([0-9]{2}):([0-9]{2})",
                       message)
    if trouve:
        try:
            heure, minute, seconde = (int(trouve.group(i)) for i in (4, 5, 6))
            maintenant = time.gmtime()
            cible = (heure - maintenant.tm_hour) * 3600 + (minute - maintenant.tm_min) * 60 \
                + (seconde - maintenant.tm_sec)
            if 0 < cible < 900:
                return cible + 3
        except (TypeError, ValueError):
            pass
    return 65.0


def charger_cle() -> str | None:
    cle = os.environ.get("METEOFRANCE_API_KEY", "").strip()
    if cle:
        return cle
    chemin = os.path.expanduser("~/.config/floodcast/env")
    if os.path.exists(chemin):
        with open(chemin, encoding="utf-8") as fh:
            for ligne in fh:
                if ligne.startswith("METEOFRANCE_API_KEY="):
                    return ligne.split("=", 1)[1].strip()
    return None


class AccesRefuse(RuntimeError):
    """Cle absente, invalide, ou non abonnee a DPClim."""


class ErreurRequete(RuntimeError):
    """Refus de l'API portant un code et un message exploitables."""

    def __init__(self, statut: int, message: str):
        super().__init__(f"HTTP {statut} — {message}")
        self.statut = statut
        self.message = message


def _get(chemin: str, params: dict | None = None, cle: str | None = None,
         essais: int = 4, timeout: float = 90):
    cle = cle or charger_cle()
    if not cle:
        raise AccesRefuse("METEOFRANCE_API_KEY absente")
    entetes = {"apikey": cle}
    derniere = None
    for i in range(essais):
        _reguler()
        r = _session.get(f"{BASE}{chemin}", params=params, headers=entetes, timeout=timeout)
        if r.status_code == 429:
            # Le quota est temporel : le refus porte l'heure de reouverture.
            time.sleep(_attente_429(r.text))
            derniere = f"HTTP 429 — {r.text[:120]}"
            continue
        if r.status_code in (200, 201, 202, 204):
            return r
        if r.status_code == 403:
            raise AccesRefuse(
                "DPClim refuse la cle (403). L'abonnement est inscrit dans le jeton : "
                "apres s'etre abonne sur le portail, il faut REGENERER le jeton.")
        if r.status_code == 401:
            raise AccesRefuse("cle invalide (401)")
        # Un 4xx est un refus argumente : reessayer a l'identique ne changera
        # rien et coute une minute par appel. Seul le 429 merite d'attendre.
        if 400 <= r.status_code < 500 and r.status_code != 429:
            raise ErreurRequete(r.status_code, r.text[:200].strip())
        derniere = f"HTTP {r.status_code} — {r.text[:150]}"
        time.sleep(5 * (i + 1))
    raise RuntimeError(derniere or "echec inconnu")


# --------------------------------------------------------------------------- #
# Referentiel
# --------------------------------------------------------------------------- #


def stations(departement: str, frequence: str = "horaire",
             parametre: str = "precipitation") -> pd.DataFrame:
    """Postes d'un departement publiant a la frequence demandee."""
    if frequence not in FREQUENCES:
        raise ValueError(f"frequence inconnue : {frequence}")
    params = {"id-departement": str(departement).lstrip("0")}
    if parametre:
        params["parametre"] = parametre
    donnees = _get(f"/liste-stations/{frequence}", params).json()
    df = pd.DataFrame(donnees)
    for colonne in ("lat", "lon", "alt", "latitude", "longitude", "altitude"):
        if colonne in df.columns:
            df[colonne] = pd.to_numeric(df[colonne], errors="coerce")
    return df


def information_station(id_station: str) -> dict:
    return _get("/information-station", {"id-station": id_station}).json()


# --------------------------------------------------------------------------- #
# Commande et recuperation
# --------------------------------------------------------------------------- #


def _iso(date) -> str:
    """L'API exige un ISO 8601 en UTC, secondes comprises : 2024-10-17T00:00:00Z."""
    ts = pd.Timestamp(date)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def commander(id_station: str, debut, fin, frequence: str = "horaire") -> str:
    reponse = _get(f"/commande-station/{frequence}",
                   {"id-station": id_station,
                    "date-deb-periode": _iso(debut), "date-fin-periode": _iso(fin)})
    try:
        charge = reponse.json()
    except ValueError:
        return reponse.text.strip()
    # La reponse encapsule le numero sous "elaboreProduitAvecDemandeResponse".
    while isinstance(charge, dict):
        for cle in ("return", "elaboreProduitAvecDemandeResponse", "id-cmde", "idCmde"):
            if cle in charge:
                charge = charge[cle]
                break
        else:
            charge = next(iter(charge.values())) if len(charge) == 1 else str(charge)
    return str(charge).strip()


def _cache_lire(cle_cache: str) -> str | None:
    chemin = os.path.join(CACHE, cle_cache + ".csv")
    if os.path.exists(chemin) and os.path.getsize(chemin) > 0:
        with open(chemin, encoding="utf-8") as fh:
            return fh.read()
    return None


def _cache_ecrire(cle_cache: str, texte: str) -> None:
    os.makedirs(CACHE, exist_ok=True)
    with open(os.path.join(CACHE, cle_cache + ".csv"), "w", encoding="utf-8") as fh:
        fh.write(texte)


def _cle_cache(id_station: str, debut, fin, frequence: str) -> str:
    brut = f"{id_station}|{frequence}|{pd.Timestamp(debut)}|{pd.Timestamp(fin)}"
    return hashlib.sha256(brut.encode()).hexdigest()[:24]


def telecharger(id_cmde: str, patience: float = 180.0, pas: float = 6.0) -> str:
    """Recupere le CSV de la commande. Le 204 signifie 'pas encore pret'."""
    fin = time.time() + patience
    while True:
        reponse = _get("/commande/fichier", {"id-cmde": id_cmde}, essais=2)
        if reponse.status_code in (200, 201) and reponse.content:
            return reponse.text
        if time.time() > fin:
            raise TimeoutError(f"commande {id_cmde} toujours pas prete apres {patience:.0f} s")
        time.sleep(pas)


def _lire_csv(texte: str) -> pd.DataFrame:
    """Le CSV DPClim est separe par ';' et utilise la virgule decimale."""
    df = pd.read_csv(io.StringIO(texte), sep=";", decimal=",", dtype={"POSTE": str})
    df.columns = [c.strip().upper() for c in df.columns]
    colonne_date = next((c for c in df.columns if c in ("DATE", "AAAAMMJJHH", "AAAAMMJJ")), None)
    if colonne_date is None:
        return df
    brut = df[colonne_date].astype(str).str.strip()
    longueur = int(brut.str.len().mode().iloc[0]) if len(brut) else 0
    format_date = {8: "%Y%m%d", 10: "%Y%m%d%H", 12: "%Y%m%d%H%M"}.get(longueur)
    instant = pd.to_datetime(brut, format=format_date, errors="coerce") \
        if format_date else pd.to_datetime(brut, errors="coerce", format="mixed")
    df = df.set_axis(instant, axis=0)
    df.index.name = "instant"
    return df[df.index.notna()].sort_index()


def _decouper(id_station: str, debut, fin, frequence: str,
              tranche: pd.Timedelta, journal=None) -> list[tuple]:
    """Passe toutes les commandes couvrant [debut, fin], en decoupant si besoin.

    L'API borne la duree d'une commande, differemment selon la frequence (un an
    en quotidien, davantage en horaire). Plutot que de coder ces limites en dur,
    on laisse le refus les enseigner : un 400 evoquant la periode fait couper la
    tranche en deux. Le code s'adapte donc si Meteo-France change ses regles.
    """
    a_traiter = []
    curseur = pd.Timestamp(debut)
    fin = pd.Timestamp(fin)
    while curseur < fin:
        borne = min(curseur + tranche, fin)
        a_traiter.append((curseur, borne))
        curseur = borne

    # On sonde la limite de duree une seule fois, en tentant une commande : un
    # refus 400 evoquant la periode fait couper la tranche en deux.
    valides = []
    while a_traiter:
        deb, bor = a_traiter.pop(0)
        # Deja telecharge lors d'une execution precedente : inutile de recommander.
        if _cache_lire(_cle_cache(id_station, deb, bor, frequence)) is not None:
            valides.append((deb, bor, None))
            continue
        try:
            numero = commander(id_station, deb, bor, frequence)
        except ErreurRequete as e:
            duree = bor - deb
            trop_long = e.statut == 400 and ("riode" in e.message or "eriod" in e.message)
            if trop_long and duree > pd.Timedelta(days=2):
                milieu = deb + duree / 2
                a_traiter[:0] = [(deb, milieu), (milieu, bor)]
                continue
            if journal is not None:
                journal.append(f"{deb.date()}->{bor.date()} : {e.message[:80]}")
            continue
        except RuntimeError as e:
            if journal is not None:
                journal.append(f"{deb.date()}->{bor.date()} : {e}")
            continue
        valides.append((deb, bor, numero))
    return valides


def serie(id_station: str, debut, fin, frequence: str = "horaire",
          colonne: str = "RR1", tranche: str = "365D",
          qualites_admises=(1, 2), journal=None) -> pd.Series:
    """Une variable d'un poste sur une longue periode.

    Toutes les commandes sont passees d'abord, les fichiers recuperes ensuite :
    Meteo-France les prepare pendant ce temps, ce qui evite d'attendre chaque
    fichier separement.

    `colonne` : RR = cumul quotidien, RR1 = cumul horaire.
    """
    def extraire(texte):
        df = _lire_csv(texte)
        if colonne not in df.columns:
            return None
        valeurs = pd.to_numeric(df[colonne], errors="coerce")
        # Chaque parametre est double d'un indicateur qualite (Q + son nom).
        drapeau = f"Q{colonne}"
        if qualites_admises and drapeau in df.columns:
            q = pd.to_numeric(df[drapeau], errors="coerce")
            valeurs = valeurs.where(q.isin(qualites_admises) | q.isna())
        return valeurs

    # Une tranche perdue laisse un trou de plusieurs annees dans la serie, et un
    # trou silencieux est pire qu'une erreur : on rejoue les tranches echouees,
    # commande comprise, et on signale celles qui resistent.
    restantes = _decouper(id_station, debut, fin, frequence,
                          pd.Timedelta(tranche), journal)
    obtenus = {}
    for essai in range(3):
        if not restantes:
            break
        if essai:
            restantes = [(d, b, commander(id_station, d, b, frequence))
                         for d, b, _ in restantes]
        echouees = []
        for deb, bor, numero in restantes:
            cle_cache = _cle_cache(id_station, deb, bor, frequence)
            texte = _cache_lire(cle_cache)
            try:
                if texte is None:
                    texte = telecharger(numero, patience=240)
                    _cache_ecrire(cle_cache, texte)
                valeurs = extraire(texte)
            except (TimeoutError, RuntimeError):
                echouees.append((deb, bor, numero))
                continue
            if valeurs is None:
                continue
            obtenus[(deb, bor)] = valeurs
        restantes = echouees
    if restantes and journal is not None:
        for deb, bor, _ in restantes:
            journal.append(f"ABANDON {deb.date()}->{bor.date()} apres 3 tentatives")

    morceaux = list(obtenus.values())
    if not morceaux:
        return pd.Series(dtype=float, name=id_station)
    out = pd.concat(morceaux)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.rename(id_station)


# --------------------------------------------------------------------------- #
# Pluie de bassin
# --------------------------------------------------------------------------- #


def postes_du_bassin(departements, emprise, rayon_km: float = 25.0,
                     frequence: str = "horaire") -> pd.DataFrame:
    """Postes situes dans ou pres de l'emprise du bassin, du plus proche au plus loin."""
    from .hubeau import haversine_km

    lot = []
    for dep in departements:
        try:
            df = stations(dep, frequence)
        except AccesRefuse:
            raise
        except Exception:  # noqa: BLE001 - departement sans poste a cette frequence
            continue
        if not df.empty:
            df = df.copy()
            df["departement"] = dep
            lot.append(df)
    if not lot:
        return pd.DataFrame()
    postes = pd.concat(lot, ignore_index=True)
    col_lat = next((c for c in ("lat", "latitude") if c in postes.columns), None)
    col_lon = next((c for c in ("lon", "longitude") if c in postes.columns), None)
    if not col_lat or not col_lon:
        return postes
    centre_lon = float(np.mean([p[0] for p in emprise]))
    centre_lat = float(np.mean([p[1] for p in emprise]))
    postes["dist_km"] = haversine_km(centre_lon, centre_lat,
                                     postes[col_lon], postes[col_lat])
    return postes[postes["dist_km"] <= rayon_km].sort_values("dist_km").reset_index(drop=True)


def pluie_bassin(ids_postes, debut, fin, frequence: str = "horaire",
                 colonne: str = "RR1", poids=None) -> pd.Series:
    """Moyenne des postes disponibles a chaque pas de temps.

    Un poste absent une annee ne doit pas creuser un trou dans la serie : la
    moyenne se fait sur les postes presents, ce qui degrade la representativite
    plutot que la continuite.
    """
    series = {}
    for identifiant in ids_postes:
        s = serie(identifiant, debut, fin, frequence, colonne)
        if len(s):
            series[identifiant] = s
    if not series:
        return pd.Series(dtype=float)
    tableau = pd.DataFrame(series)
    if poids is not None:
        w = pd.Series(poids).reindex(tableau.columns).fillna(0.0)
        valides = tableau.notna()
        return (tableau.fillna(0.0) * w).sum(axis=1) / (valides * w).sum(axis=1).replace(0, np.nan)
    return tableau.mean(axis=1)
