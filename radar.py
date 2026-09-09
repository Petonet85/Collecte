#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lame d'eau radar Meteo-France, moyennee sur un bassin versant.

L'API DPRadar ne sert que le dernier pas de 5 minutes : rien n'est conserve.
Une donnee non captee est perdue definitivement. D'ou ce module, concu pour
etre appele aussi souvent que possible et n'archiver qu'un nombre par pas de
temps : la lame d'eau moyenne sur le bassin, soit une cinquantaine d'octets
au lieu des 2 Mo de la grille nationale.

    python radar.py                     # capture le pas de temps courant
    python radar.py --boucle 5          # capture en continu, toutes les 5 min
    python radar.py --bassin M702241010

La cle se lit dans METEOFRANCE_API_KEY, ou a defaut dans ~/.config/floodcast/env.
"""

import argparse
import base64
import csv
import gzip
import io
import json
import os
import re
import sys
import tarfile
import time
from datetime import datetime, timezone

import numpy as np
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
API = "https://public-api.meteofrance.fr/public/DPRadar/v1"
PAQUET = "https://public-api.meteofrance.fr/public/DPPaquetRadar"
ZONE = "METROPOLE"
PRODUIT = "LAME_D_EAU"
MAILLE = 500          # 500 m -> HDF5 ODIM lisible avec h5py ; 1000 m -> BUFR (eccodes)

# Dans le paquet, chaque produit porte un code OMM. IPRN20 est la lame d'eau
# (ACRR) metropole au pas de 500 m : la meme grille que DPRadar, en trois
# exemplaires correspondant aux trois derniers pas de cinq minutes.
CODE_LAME_METROPOLE = "IPRN20"

ENTETES = {"Accept": "*/*", "User-Agent": "collecte-sevre-nantaise/2.0"}

# Points suivis individuellement, en plus de la moyenne de bassin. Une averse
# convective couvre quelques kilometres carres : moyennee sur 576 km2 elle
# disparait, alors qu'elle tombe bel et bien sur la propriete.
POINTS = {"rochereau": (-0.99276, 47.000408)}

# Vignette du bassin conservee pour l'animation : 24 x 24 mailles, soit
# environ 1,2 km. La finesse n'est pas limitee par le radar, qui donne du
# 500 m, mais par le champ de prevision qui prolonge l'animation : Open-Meteo
# facture UN APPEL PAR POINT, et 24 x 24 tient dans le quota avec de la marge
# pour le reste de la chaine. Les deux grilles doivent coincider. L'intensite
# est stockee en racine carree — un octet couvre alors 0 a 113 mm/h avec une
# resolution fine dans les faibles valeurs, la ou la lecture est la plus utile.
# Environ 400 octets par pas de temps, et rien du tout quand il ne pleut pas.
VIGNETTE = 24
VIGNETTE_FACTEUR = 24.0        # q = racine(mm/h) * facteur

# La passerelle Meteo-France coupe son point d'entree ("303001 SUSPENDED") des
# qu'on l'interroge trop vite. On reessaie largement espace plutot que d'insister.
ATTENTES = (20, 45, 90, 120)


# --------------------------------------------------------------------------- #
# Cle d'API
# --------------------------------------------------------------------------- #

def charger_cle():
    cle = os.environ.get("METEOFRANCE_API_KEY", "").strip()
    if cle:
        return cle
    chemin = os.path.expanduser("~/.config/floodcast/env")
    if os.path.exists(chemin):
        with open(chemin) as f:
            for ligne in f:
                if ligne.startswith("METEOFRANCE_API_KEY="):
                    return ligne.split("=", 1)[1].strip()
    return None


# --------------------------------------------------------------------------- #
# Projection stereographique polaire de la mosaique ODIM
# --------------------------------------------------------------------------- #

RAYON = 6378137.0                     # demi-grand axe WGS84
APLAT = 1 / 298.257223563
EXC = np.sqrt(2 * APLAT - APLAT * APLAT)
LAT_TS = np.radians(45.0)             # latitude de reference de la mosaique


def _projeter(lon, lat):
    """Coordonnees stereographiques polaires nord (lon_0 = 0, lat_ts = 45).

    Formules EPSG 9810. On ne se sert pas des faux Est/Nord du fichier : la
    grille est calee sur son coin superieur gauche, ce qui rend le calcul
    independant des conventions de l'emetteur.
    """
    lon = np.radians(np.asarray(lon, dtype=float))
    lat = np.radians(np.asarray(lat, dtype=float))
    t = np.tan(np.pi / 4 - lat / 2) / ((1 - EXC * np.sin(lat)) / (1 + EXC * np.sin(lat))) ** (EXC / 2)
    t_ref = np.tan(np.pi / 4 - LAT_TS / 2) / (
        (1 - EXC * np.sin(LAT_TS)) / (1 + EXC * np.sin(LAT_TS))) ** (EXC / 2)
    m_ref = np.cos(LAT_TS) / np.sqrt(1 - EXC ** 2 * np.sin(LAT_TS) ** 2)
    rho = RAYON * m_ref * t / t_ref
    return rho * np.sin(lon), -rho * np.cos(lon)


def masque_bassin(where, emprise, surface_km2):
    """Indices des pixels radar couvrant le bassin.

    L'emprise est une liste de points ; on retient les pixels situes a moins
    d'un rayon du point le plus proche, ce rayon etant ajuste pour que la
    surface masquee colle a la surface officielle du bassin. Le masque se
    calibre donc tout seul, sans dependre de la finesse de l'emprise fournie.
    """
    ulx, uly = _projeter(float(where["UL_lon"]), float(where["UL_lat"]))
    sx, sy = float(where["xscale"]), float(where["yscale"])
    bx, by = _projeter([p[0] for p in emprise], [p[1] for p in emprise])

    marge = 6
    c0 = max(int((bx.min() - ulx) / sx) - marge, 0)
    c1 = min(int((bx.max() - ulx) / sx) + marge + 1, int(where["xsize"]))
    l0 = max(int((uly - by.max()) / sy) - marge, 0)
    l1 = min(int((uly - by.min()) / sy) + marge + 1, int(where["ysize"]))

    cols, lignes = np.meshgrid(np.arange(c0, c1), np.arange(l0, l1))
    px = ulx + (cols + 0.5) * sx
    py = uly - (lignes + 0.5) * sy
    d2 = np.min((px[..., None] - bx) ** 2 + (py[..., None] - by) ** 2, axis=-1)

    aire_pixel = sx * sy / 1e6                      # km2 par pixel
    cible = surface_km2 / aire_pixel                # nombre de pixels vise
    ordre = np.sort(d2.ravel())
    seuil = ordre[min(int(cible), len(ordre) - 1)]
    masque = d2 <= seuil
    return {"l0": l0, "l1": l1, "c0": c0, "c1": c1, "masque": masque,
            "surface_masque_km2": round(float(masque.sum() * aire_pixel), 1)}


# --------------------------------------------------------------------------- #
# Acces a l'API
# --------------------------------------------------------------------------- #

def _get(url, cle, params=None, essais=4, timeout=120):
    entetes = dict(ENTETES, apikey=cle)
    derniere = None
    for i in range(essais):
        try:
            r = requests.get(url, headers=entetes, params=params, timeout=timeout)
        except Exception as e:
            derniere = f"{type(e).__name__}"
            time.sleep(ATTENTES[min(i, len(ATTENTES) - 1)])
            continue
        if r.status_code == 200:
            return r
        derniere = f"HTTP {r.status_code} {r.text[:120]}"
        # 403 = abonnement absent, 401 = cle invalide : reessayer ne sert a rien.
        if r.status_code in (401, 403):
            break
        time.sleep(ATTENTES[min(i, len(ATTENTES) - 1)])
    raise RuntimeError(derniere or "echec inconnu")


def paquet_grilles(cle):
    """Les trois dernieres grilles de lame d'eau metropole, via l'API paquet.

    Un seul appel rend le dernier quart d'heure. C'est ce qui rend la collecte
    viable sur GitHub Actions : le planificateur n'y honore pas les cadences
    inferieures au quart d'heure, mesure a l'appui (33 % de couverture en mode
    simple). Une fenetre de quinze minutes recouvre l'intervalle reellement
    obtenu, et absorbe donc les retards au lieu de les subir.
    """
    reponse = _get(f"{PAQUET}/mosaique/paquet", cle, timeout=180)
    archive = tarfile.open(fileobj=io.BytesIO(reponse.content), mode="r:gz")
    grilles = []
    for membre in archive.getmembers():
        if CODE_LAME_METROPOLE not in membre.name or not membre.name.endswith(".h5"):
            continue
        horodatage = re.search(r"(\d{14})", membre.name)
        contenu = archive.extractfile(membre).read()
        grilles.append((horodatage.group(1) if horodatage else None, contenu))
    return sorted(grilles, key=lambda g: g[0] or "")


def derniere_grille(cle, maille=MAILLE):
    """(instant de validite, contenu binaire) de la mosaique la plus recente."""
    meta = _get(f"{API}/mosaiques/{ZONE}/observations/{PRODUIT}", cle).json()
    instant = None
    for lien in meta.get("links", []):
        if f"maille={maille}" in lien.get("href", ""):
            instant = lien.get("validity_time")
            break
    r = _get(f"{API}/mosaiques/{ZONE}/observations/{PRODUIT}/produit", cle,
             params={"maille": maille})
    contenu = r.content
    if contenu[:2] == b"\x1f\x8b":
        contenu = gzip.decompress(contenu)
    return instant, contenu


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def encoder_vignette(champ_mm_h):
    """Quantifie un champ d'intensite (mm/h) en un octet par maille."""
    q = np.clip(np.round(np.sqrt(np.maximum(champ_mm_h, 0)) * VIGNETTE_FACTEUR), 0, 255)
    return base64.b64encode(q.astype(np.uint8).tobytes()).decode()


def _vignette(bloc, masque, duree_min, n=VIGNETTE):
    """Reduit la fenetre du bassin a une petite grille d'intensite, en mm/h.

    On passe en mm/h plutot qu'en cumul sur le pas de temps : c'est la seule
    unite qui permette de comparer une image radar de cinq minutes a une
    prevision horaire dans la meme animation.
    """
    h, w = bloc.shape
    lignes = np.array_split(np.arange(h), min(n, h))
    colonnes = np.array_split(np.arange(w), min(n, w))
    out = np.zeros((len(lignes), len(colonnes)))
    for i, li in enumerate(lignes):
        for j, co in enumerate(colonnes):
            sous = bloc[np.ix_(li, co)]
            m = masque[np.ix_(li, co)]
            out[i, j] = float(np.nanmean(sous[m])) if m.any() else 0.0
    return encoder_vignette(out * (60.0 / max(duree_min, 1))), out.shape


def lame_bassin(contenu, emprise, surface_km2, cache_masque=None, points=None):
    """Lame d'eau moyenne (mm) sur le bassin, pour le pas de temps du fichier."""
    import h5py

    with h5py.File(io.BytesIO(contenu), "r") as f:
        where = dict(f["where"].attrs)
        if cache_masque is None or cache_masque.get("_signature") != (
                float(where["UL_lon"]), float(where["xscale"]), int(where["xsize"])):
            cache_masque = masque_bassin(where, emprise, surface_km2)
            cache_masque["_signature"] = (float(where["UL_lon"]), float(where["xscale"]),
                                          int(where["xsize"]))
        m = cache_masque
        jeu = f["dataset1/data1"]
        att = jeu["what"].attrs
        bloc = jeu["data"][m["l0"]:m["l1"], m["c0"]:m["c1"]].astype(float)[m["masque"]]

        gain, offset = float(att["gain"]), float(att["offset"])
        nodata, undetect = float(att["nodata"]), float(att["undetect"])
        valides = bloc != nodata
        # "undetect" = le radar a regarde et n'a rien vu : c'est un vrai zero,
        # a ne surtout pas confondre avec "nodata" (pas de mesure du tout).
        pluie = np.where(bloc == undetect, 0.0, bloc * gain + offset)

        qualite = None
        if "quality1" in jeu:
            qa = jeu["quality1"]["what"].attrs
            qbloc = jeu["quality1"]["data"][m["l0"]:m["l1"], m["c0"]:m["c1"]].astype(float)[m["masque"]]
            qualite = float(np.mean(qbloc[valides] * float(qa["gain"]) + float(qa["offset"]))) \
                if valides.any() else None

        # --- valeur ponctuelle aux points suivis
        valeurs_points = {}
        if points:
            ulx, uly = _projeter(float(where["UL_lon"]), float(where["UL_lat"]))
            sx, sy = float(where["xscale"]), float(where["yscale"])
            for nom, (lon, lat) in points.items():
                px, py = _projeter(lon, lat)
                i = int((px - ulx) / sx)
                j = int((uly - py) / sy)
                if 0 <= i < int(where["xsize"]) and 0 <= j < int(where["ysize"]):
                    brut = float(jeu["data"][j, i])
                    valeurs_points[nom] = (0.0 if brut == undetect
                                           else None if brut == nodata
                                           else round(brut * gain + offset, 3))

        quoi_tmp = f["dataset1/what"].attrs
        _dec = lambda v: v.decode() if isinstance(v, bytes) else str(v)
        _deb = f"{_dec(quoi_tmp['startdate'])}{_dec(quoi_tmp['starttime'])}"
        _fin = f"{_dec(quoi_tmp['enddate'])}{_dec(quoi_tmp['endtime'])}"
        _duree = (datetime.strptime(_fin, "%Y%m%d%H%M%S")
                  - datetime.strptime(_deb, "%Y%m%d%H%M%S")).total_seconds() / 60

        # --- vignette pour l'animation, seulement s'il pleut quelque part
        grille = None
        if ARCHIVER_VIGNETTE and valides.any() and pluie[valides].max() > 0:
            plein = jeu["data"][m["l0"]:m["l1"], m["c0"]:m["c1"]].astype(float)
            plein = np.where(plein == undetect, 0.0,
                             np.where(plein == nodata, np.nan, plein * gain + offset))
            grille, forme = _vignette(plein, m["masque"], _duree)

        quoi = f["dataset1/what"].attrs
        dec = lambda v: v.decode() if isinstance(v, bytes) else str(v)
        debut = f"{dec(quoi['startdate'])}{dec(quoi['starttime'])}"
        fin = f"{dec(quoi['enddate'])}{dec(quoi['endtime'])}"
        instant = datetime.strptime(fin, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        duree = (datetime.strptime(fin, "%Y%m%d%H%M%S")
                 - datetime.strptime(debut, "%Y%m%d%H%M%S")).total_seconds() / 60

    if not valides.any():
        return None, cache_masque
    mesure = {
        "instant_utc": instant.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duree_min": int(round(duree)),
        "lame_mm": round(float(pluie[valides].mean()), 4),
        "lame_max_mm": round(float(pluie[valides].max()), 3),
        "pixels": int(valides.sum()),
        "couverture": round(float(valides.mean()), 3),
        "qualite": None if qualite is None else round(qualite, 3),
        "grille": grille or "",
    }
    for nom, v in valeurs_points.items():
        mesure[f"pt_{nom}"] = v
    return mesure, cache_masque


# --------------------------------------------------------------------------- #
# Archivage
# --------------------------------------------------------------------------- #

COLONNES = ["instant_utc", "duree_min", "lame_mm", "lame_max_mm", "pixels",
            "couverture", "qualite"] + [f"pt_{n}" for n in POINTS] + ["grille"]


def ecrire(mesure, code_bassin, dossier):
    """Ajoute une mesure au CSV mensuel du bassin. Idempotent sur l'instant."""
    chemin = os.path.join(dossier, "radar", code_bassin,
                          f"{mesure['instant_utc'][:7]}.csv")
    os.makedirs(os.path.dirname(chemin), exist_ok=True)

    existant = {}
    if os.path.exists(chemin):
        with open(chemin, newline="") as f:
            for ligne in csv.DictReader(f):
                existant[ligne["instant_utc"]] = ligne
    if mesure["instant_utc"] in existant:
        return False

    existant[mesure["instant_utc"]] = {c: mesure.get(c, "") for c in COLONNES}
    with open(chemin, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLONNES)
        w.writeheader()
        for instant in sorted(existant):
            w.writerow(existant[instant])
    return True


def collecter(bassins, dossier, cle, caches=None, mode="auto"):
    """Capture les pas de temps disponibles pour chaque bassin.

    `mode` : "paquet" (le dernier quart d'heure), "simple" (le dernier pas de
    cinq minutes), ou "auto" qui tente le paquet et retombe sur le simple si
    l'abonnement manque.
    """
    caches = caches if caches is not None else {}
    grilles = []
    if mode in ("auto", "paquet"):
        try:
            grilles = paquet_grilles(cle)
        except Exception as e:  # noqa: BLE001 - abonnement absent ou service indisponible
            if mode == "paquet":
                raise
            print(f"    (paquet indisponible : {str(e)[:70]} — repli sur le pas simple)")
    if not grilles:
        instant, contenu = derniere_grille(cle)
        grilles = [(instant, contenu)]

    resultats = []
    for _, contenu in grilles:
        for code, bassin in bassins.items():
            mesure, caches[code] = lame_bassin(contenu, bassin["emprise"],
                                               bassin["surface_km2"], caches.get(code),
                                               points=POINTS if code == "M703243010" else None)
            if mesure is None:
                resultats.append((code, None, False))
                continue
            resultats.append((code, mesure, ecrire(mesure, code, dossier)))
    return len(grilles), resultats, caches


def main(argv=None):
    p = argparse.ArgumentParser(description="Lame d'eau radar moyennee par bassin")
    p.add_argument("--depot", default=os.path.join(BASE, "donnees"),
                   help="dossier d'archivage")
    p.add_argument("--bassins", default=os.path.join(BASE, "bassins.json"))
    p.add_argument("--bassin", help="ne traiter qu'un bassin (code station)")
    p.add_argument("--boucle", type=int, metavar="MINUTES",
                   help="capture en continu a cet intervalle (5 = pas natif du radar)")
    p.add_argument("--duree", type=int, metavar="MINUTES",
                   help="avec --boucle : duree totale avant de rendre la main")
    p.add_argument("--mode", choices=("auto", "paquet", "simple"), default="auto",
                   help="paquet = dernier quart d'heure ; simple = dernier pas de 5 min")
    args = p.parse_args(argv)

    cle = charger_cle()
    if not cle:
        print("  METEOFRANCE_API_KEY absente : collecte radar ignoree.")
        return 0

    with open(args.bassins, encoding="utf-8") as f:
        bassins = json.load(f)
    if args.bassin:
        bassins = {args.bassin: bassins[args.bassin]}

    fin = time.time() + args.duree * 60 if args.duree else None
    caches = {}
    total = 0
    while True:
        horodatage = datetime.now(timezone.utc).strftime("%H:%M")
        try:
            n_grilles, resultats, caches = collecter(bassins, args.depot, cle,
                                                     caches, args.mode)
            for code, mesure, nouveau in resultats:
                if mesure is None:
                    print(f"  [{horodatage}] {code} : aucun pixel valide")
                    continue
                total += int(nouveau)
                etat = "archive" if nouveau else "deja connu"
                print(f"  [{horodatage}] {code} {mesure['instant_utc'][11:16]} : "
                      f"{mesure['lame_mm']:.3f} mm/{mesure['duree_min']}min "
                      f"(max {mesure['lame_max_mm']:.2f}, q {mesure['qualite']}) — {etat}")
        except Exception as e:
            print(f"  [{horodatage}] ECHEC radar : {e}")
            if not args.boucle:
                return 1
        if not args.boucle:
            break
        if fin and time.time() >= fin:
            break
        time.sleep(args.boucle * 60)

    print(f"  radar : {total} pas de temps ajoutes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
