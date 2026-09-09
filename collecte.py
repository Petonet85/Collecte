#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Archivage des observations hydrologiques de la Sevre Nantaise.

Trois sources, toutes a fenetre glissante : ce qui n'est pas capte est perdu.
  * Hub'Eau hydrometrie : hauteurs et debits, un mois d'historique ;
  * Hub'Eau piezometrie : niveaux de nappe, etat de saturation du bocage ;
  * Meteo-France DPRadar : lame d'eau, cinq minutes d'historique seulement
    (voir radar.py, appele par --radar).

Tout est idempotent : relancer le script ne cree aucun doublon.

    python collecte.py --depot donnees          # collecte complete
    python collecte.py --depot donnees --radar  # avec la lame d'eau radar
    python collecte.py --rapport                # etat de l'archive, sans reseau
    python collecte.py --verifier               # controle les codes configures

A lancer une fois par jour. Sur macOS, le plus simple est cron :
    crontab -e
    puis la ligne (adaptez les chemins) :
    30 6 * * * cd /Users/benoit/sevre && /usr/bin/python3 collecte.py >> collecte.log 2>&1

Une collecte quotidienne suffit largement : la fenetre Hub'Eau couvre un mois,
donc meme plusieurs jours d'interruption ne perdent rien.
"""

import argparse
import csv
import os
import sqlite3
import sys
from datetime import datetime, timezone

import requests

import time

BASE = os.path.dirname(os.path.abspath(__file__))
FICHIER = os.path.join(BASE, "historique.db")

# Codes a 10 caracteres = station de mesure, a 8 = site (qui peut porter
# plusieurs stations). Hub'Eau accepte les deux dans code_entite.
# Verifiez toute nouvelle entree avec : python collecte.py --verifier
# Le debit se demande sur le code SITE (8 caracteres), la hauteur sur le code
# STATION (10). Saint-Laurent ne publie aucun debit, nulle part : c'est une
# station limnimetrique seule, d'ou la prevision par propagation depuis l'amont.
STATIONS = {
    "M703243010": {"nom": "Sevre Nantaise a Saint-Laurent-sur-Sevre",
                   "grandeurs": ("H",), "bv_km2": 576},
    "M702241010": {"nom": "Sevre Nantaise a Saint-Mesmin [La Branle]",
                   "grandeurs": ("H", "Q"), "bv_km2": 359},
    "M704401010": {"nom": "Ouin a Mauleon [La Voie Moulins]",
                   "grandeurs": ("H", "Q"), "bv_km2": 61},
    "M711241020": {"nom": "Sevre Nantaise a Tiffauges",
                   "grandeurs": ("H", "Q"), "bv_km2": 814},
}

# Le niveau de nappe dit dans quel etat de saturation se trouve le bassin, donc
# quelle part de la pluie a venir ruissellera plutot que de s'infiltrer.
#
# Seuls les piezometres du reseau TEMPS REEL sont archives ici, et il n'y en a
# qu'un dans tout le secteur. Les autres (La Rucette, a 11 km) ne publient que
# dans l'API "chroniques", qui conserve elle-meme tout l'historique : rien ne
# presse de les recopier, on les interrogera au moment voulu.
PIEZOS = {
    "05092X0009/P": "Vallet (44) — reseau temps reel",
}

# Nom de colonne et facteur de conversion depuis l'unite brute Hub'Eau.
GRANDEURS = {
    "H": {"colonne": "hauteur_cm", "diviseur": 10.0},      # mm -> cm
    "Q": {"colonne": "debit_m3s", "diviseur": 1000.0},     # L/s -> m3/s
}

# API v2 : la v1 a ete arretee, tous ses points d'entree renvoient 403.
API = "https://hubeau.eaufrance.fr/api/v2/hydrometrie/observations_tr"

# Un agent utilisateur explicite : le defaut de requests est frequemment
# refuse (403) par les pare-feux applicatifs des API publiques.
ENTETES = {
    "User-Agent": "suivi-sevre-nantaise/1.0 (usage personnel)",
    "Accept": "application/json",
}

# Hub'Eau plafonne la taille de page. 1000 passe partout ; 5000 est refuse
# sur certains points d'entree.
TAILLE_PAGE = 1000


ATTENTES = (5, 15, 40)


def _reecrire(chemin, entete, lignes):
    """Reecrit un CSV d'archive sans risquer de le perdre.

    Ouvrir la cible en "w" la vide avant de savoir si l'ecriture aboutira : une
    exception au milieu de la boucle laisse une archive reduite a son en-tete.
    On passe donc par un fichier temporaire, remplace d'un seul geste.
    """
    tmp = chemin + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(entete)
        w.writerows(lignes)
    os.replace(tmp, chemin)


def _get(url, params, timeout=60):
    """GET avec reprises espacees : les API publiques coupent sous charge."""
    derniere = None
    for i, attente in enumerate((0,) + ATTENTES):
        if attente:
            time.sleep(attente)
        try:
            r = requests.get(url, params=params, headers=ENTETES, timeout=timeout)
        except Exception as e:
            derniere = f"{type(e).__name__}"
            continue
        if r.status_code == 404:
            return None
        if r.ok:
            return r.json()
        derniere = f"HTTP {r.status_code} — {r.text[:150]}"
    raise RuntimeError(derniere or "echec inconnu")


def init():
    with sqlite3.connect(FICHIER) as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS mesures (
                source    TEXT NOT NULL,      -- station, code BSS ou bassin radar
                grandeur  TEXT NOT NULL,      -- H, Q, nappe, lame
                instant   TEXT NOT NULL,
                valeur    REAL NOT NULL,
                PRIMARY KEY (source, grandeur, instant)
            )
        """)
        cx.execute("CREATE INDEX IF NOT EXISTS idx_instant "
                   "ON mesures(source, grandeur, instant)")


def recuperer(code, grandeur="H", taille=20000):
    """Observations disponibles pour une station, dans la grandeur demandee.

    La v2 pagine par curseur : chaque reponse porte l'URL de la page suivante
    dans son champ "next". On la suit telle quelle plutot que de reconstruire
    les parametres.
    """
    code = code.replace(" ", "")
    # Le debit n'est publie que sur le site, la hauteur sur la station.
    entite = code[:8] if grandeur == "Q" else code
    lot = []
    url = API
    params = {
        "code_entite": entite,
        "grandeur_hydro": grandeur,
        "size": min(taille, TAILLE_PAGE),
        "sort": "asc",
    }

    while url:
        contenu = _get(url, params)
        if contenu is None:
            return []
        donnees = contenu.get("data", [])
        lot.extend(donnees)

        if not donnees or len(lot) >= taille:
            break
        url = contenu.get("next")
        params = None      # l'URL "next" porte deja tous les parametres

    # Interroge par site, Hub'Eau renvoie chaque mesure deux fois : une ligne
    # portant le code station, une ligne agregee ou code_station vaut null.
    # On garde la station configuree, sinon n'importe quelle ligne identifiee.
    if any(o.get("code_station") == code for o in lot):
        lot = [o for o in lot if o.get("code_station") == code]
    elif any(o.get("code_station") for o in lot):
        lot = [o for o in lot if o.get("code_station")]

    diviseur = GRANDEURS[grandeur]["diviseur"]
    return [
        (code, o["date_obs"], o["resultat_obs"] / diviseur)
        for o in lot
        if o.get("resultat_obs") is not None and o.get("date_obs")
    ]


REFERENTIEL = "https://hubeau.eaufrance.fr/api/v2/hydrometrie/referentiel"


def verifier():
    """Identifie chaque code configure : nom, commune, position, surface.

    Evite de collecter pendant des semaines sur un code errone, et confirme
    au passage la surface officielle du bassin versant.
    """
    for code, config in STATIONS.items():
        code = code.replace(" ", "")
        print(f"\n  {code}  (configure comme : {config['nom']}, "
              f"grandeurs {'+'.join(config['grandeurs'])})")
        trouve = False
        joignable = False

        for genre, cle in (("station", "code_station"), ("site", "code_site")):
            try:
                r = requests.get(f"{REFERENTIEL}/{genre}s", params={cle: code, "size": 5},
                                 headers=ENTETES, timeout=30)
            except Exception as e:
                print(f"    reseau injoignable ({type(e).__name__})")
                continue
            if not r.ok:
                print(f"    HTTP {r.status_code} sur le referentiel des {genre}s")
                continue
            joignable = True
            donnees = r.json().get("data", [])

            for d in donnees:
                trouve = True
                nom = d.get(f"libelle_{genre}") or "?"
                print(f"    trouve comme {genre.upper()} : {nom}")
                if d.get("libelle_commune"):
                    print(f"      commune    : {d['libelle_commune']}")
                if d.get("en_service") is not None:
                    print(f"      en service : {'oui' if d['en_service'] else 'NON'}")
                surface = d.get("surface_bv")
                if surface:
                    print(f"      bassin     : {surface} km2")
                lat = d.get(f"latitude_{genre}")
                lon = d.get(f"longitude_{genre}")
                if lat and lon:
                    print(f"      position   : {lat:.5f}, {lon:.5f}")
                if genre == "site" and d.get("code_site"):
                    stations_du_site(d["code_site"])

        if trouve:
            continue
        if joignable:
            print("    INTROUVABLE dans le referentiel — code errone.")
        else:
            # Sans reponse du serveur, on ne peut rien conclure sur le code.
            print("    Verification impossible : Hub'Eau n'a pas repondu.")


def stations_du_site(code_site):
    """Liste les stations rattachees a un site, pour lever l'ambiguite."""
    try:
        r = requests.get(f"{REFERENTIEL}/stations", params={"code_site": code_site, "size": 20},
                         headers=ENTETES, timeout=30)
        if not r.ok:
            return
        donnees = r.json().get("data", [])
    except Exception:
        return
    if len(donnees) > 1:
        print(f"      ce site porte {len(donnees)} stations :")
        for d in donnees:
            etat = "" if d.get("en_service") else "  (hors service)"
            print(f"        {d.get('code_station')}  {d.get('libelle_station', '?')}{etat}")
        print("      preferez le code de la station en service.")


def enregistrer(lignes, grandeur="H"):
    """Insere sans ecraser : une mesure deja archivee fait foi."""
    if not lignes:
        return 0
    with sqlite3.connect(FICHIER) as cx:
        avant = cx.execute("SELECT COUNT(*) FROM mesures").fetchone()[0]
        cx.executemany(
            "INSERT OR IGNORE INTO mesures (source, grandeur, instant, valeur) "
            "VALUES (?,?,?,?)",
            [(src, grandeur, instant, val) for src, instant, val in lignes],
        )
        apres = cx.execute("SELECT COUNT(*) FROM mesures").fetchone()[0]
    return apres - avant


def ecrire_csv(lignes, dossier, grandeur="H", decimales=1):
    """Archive en CSV mensuels : donnees/<station>/<grandeur>/<AAAA-MM>.csv

    Format texte plutot que SQLite : git versionne alors des ajouts de lignes
    au lieu de reecrire un binaire entier a chaque collecte.
    """
    colonne = GRANDEURS[grandeur]["colonne"]
    par_fichier = {}
    for station, instant, valeur in lignes:
        cle = (station, instant[:7])           # AAAA-MM
        par_fichier.setdefault(cle, {})[instant] = valeur

    ajouts = 0
    for (station, mois), mesures in sorted(par_fichier.items()):
        chemin = os.path.join(dossier, station, grandeur, f"{mois}.csv")
        os.makedirs(os.path.dirname(chemin), exist_ok=True)

        existant = {}
        if os.path.exists(chemin):
            with open(chemin, newline="") as f:
                for ligne in csv.DictReader(f):
                    existant[ligne["instant_utc"]] = float(ligne[colonne])

        nouvelles = {k: v for k, v in mesures.items() if k not in existant}
        if not nouvelles:
            continue
        existant.update(nouvelles)
        ajouts += len(nouvelles)

        _reecrire(chemin, ["instant_utc", colonne],
                  [[i, round(existant[i], decimales)] for i in sorted(existant)])

    return ajouts


NAPPES = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes/chroniques_tr"


def recuperer_piezo(code_bss, taille=5000):
    """Niveaux de nappe temps reel (m NGF) pour un piezometre."""
    lot = []
    url, params = NAPPES, {"code_bss": code_bss, "size": min(taille, TAILLE_PAGE),
                           "sort": "asc"}
    while url:
        contenu = _get(url, params, timeout=90)
        if contenu is None:
            return []
        donnees = contenu.get("data", [])
        lot.extend(donnees)
        if not donnees or len(lot) >= taille:
            break
        url, params = contenu.get("next"), None
    sorties = []
    for o in lot:
        niveau = o.get("niveau_eau_ngf")
        instant = o.get("date_mesure") or o.get("timestamp_mesure")
        if niveau is None or instant is None:
            continue
        if isinstance(instant, (int, float)):     # horodatage en millisecondes
            instant = datetime.fromtimestamp(instant / 1000, timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ")
        sorties.append((code_bss, str(instant), float(niveau)))
    return sorties


def ecrire_piezo(lignes, dossier):
    """donnees/nappes/<code_bss>/<AAAA-MM>.csv — le code BSS contient un slash."""
    par_fichier = {}
    for bss, instant, niveau in lignes:
        par_fichier.setdefault((bss, instant[:7]), {})[instant] = niveau

    ajouts = 0
    for (bss, mois), mesures in sorted(par_fichier.items()):
        chemin = os.path.join(dossier, "nappes", bss.replace("/", "_"), f"{mois}.csv")
        os.makedirs(os.path.dirname(chemin), exist_ok=True)
        existant = {}
        if os.path.exists(chemin):
            with open(chemin, newline="") as f:
                for ligne in csv.DictReader(f):
                    existant[ligne["instant_utc"]] = float(ligne["niveau_ngf"])
        nouvelles = {k: v for k, v in mesures.items() if k not in existant}
        if not nouvelles:
            continue
        existant.update(nouvelles)
        ajouts += len(nouvelles)
        _reecrire(chemin, ["instant_utc", "niveau_ngf"],
                  [[i, round(existant[i], 3)] for i in sorted(existant)])
    return ajouts


def importer_csv(dossier):
    """Reconstruit la base SQLite locale a partir des CSV du depot.

    Parcourt donnees/<source>/<grandeur>/*.csv, quel que soit le libelle de la
    colonne de valeur : c'est toujours la seconde du fichier.
    """
    if not os.path.isdir(dossier):
        print(f"  Dossier introuvable : {dossier}")
        return 0
    init()
    total = 0
    for racine, _, fichiers in os.walk(dossier):
        csvs = sorted(f for f in fichiers if f.endswith(".csv"))
        if not csvs:
            continue
        relatif = os.path.relpath(racine, dossier).split(os.sep)
        # nappes/<bss>/ et radar/<bassin>/ portent la grandeur dans le premier
        # segment ; les stations hydrometriques dans le second (<station>/<H|Q>).
        if relatif[0] in ("nappes", "radar") and len(relatif) == 2:
            source = relatif[1]
            grandeur = {"nappes": "nappe", "radar": "lame"}[relatif[0]]
        elif len(relatif) == 2:
            source, grandeur = relatif
        else:
            continue
        lignes = []
        for nom in csvs:
            with open(os.path.join(racine, nom), newline="") as f:
                lecteur = csv.reader(f)
                entete = next(lecteur, None)
                if not entete or len(entete) < 2:
                    continue
                for ligne in lecteur:
                    try:
                        lignes.append((source, ligne[0], float(ligne[1])))
                    except (ValueError, IndexError):
                        continue
        ajoutees = enregistrer(lignes, grandeur)
        total += ajoutees
        print(f"  {source} [{grandeur}] : {len(lignes)} lues, {ajoutees} nouvelles")
    return total


def rapport():
    if not os.path.exists(FICHIER):
        print("Aucune base : lancez d'abord une collecte.")
        return
    with sqlite3.connect(FICHIER) as cx:
        lignes = cx.execute("""
            SELECT source, grandeur, COUNT(*) n, MIN(instant) d, MAX(instant) f
            FROM mesures GROUP BY source, grandeur ORDER BY source, grandeur
        """).fetchall()

    if not lignes:
        print("Base vide.")
        return

    print("\n  Archive locale")
    for station, grandeur, n, debut, fin in lignes:
        config = STATIONS.get(station)
        nom = f"{config['nom']} [{grandeur}]" if config else f"{station} [{grandeur}]"
        jours = (datetime.fromisoformat(fin.replace("Z", "+00:00"))
                 - datetime.fromisoformat(debut.replace("Z", "+00:00"))).days
        print(f"    {nom}")
        print(f"      {n:>8} mesures  du {debut[:10]} au {fin[:10]}  ({jours} jours)")
        if jours:
            print(f"      soit environ une mesure toutes les {jours * 24 * 60 / n:.0f} min")


def exporter(chemin):
    with sqlite3.connect(FICHIER) as cx:
        lignes = cx.execute(
            "SELECT source, grandeur, instant, valeur FROM mesures "
            "ORDER BY source, grandeur, instant"
        ).fetchall()
    with open(chemin, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "grandeur", "instant_utc", "valeur"])
        w.writerows(lignes)
    print(f"  {len(lignes)} lignes ecrites dans {chemin}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rapport", action="store_true", help="etat de l'archive, sans collecte")
    p.add_argument("--export", metavar="FICHIER.csv", help="export CSV de toute l'archive")
    p.add_argument("--diagnostic", action="store_true",
                   help="teste l'acces a Hub'Eau et affiche la reponse brute")
    p.add_argument("--verifier", action="store_true",
                   help="identifie chaque code configure dans le referentiel")
    p.add_argument("--depot", metavar="DOSSIER",
                   help="archive en CSV mensuels au lieu de SQLite (pour git)")
    p.add_argument("--importer", metavar="DOSSIER",
                   help="reconstruit historique.db a partir des CSV du depot")
    p.add_argument("--radar", action="store_true",
                   help="ajoute la lame d'eau radar Meteo-France (necessite la cle API)")
    p.add_argument("--sans-nappes", action="store_true",
                   help="saute la collecte piezometrique")
    args = p.parse_args()

    if args.verifier:
        verifier()
        sys.exit(0)

    if args.diagnostic:
        code = next(iter(STATIONS))
        base = "https://hubeau.eaufrance.fr/api"
        for libelle, url, taille in [
            ("v1 (arretee)", f"{base}/v1/hydrometrie/observations_tr", 1),
            ("v2, 1 mesure", f"{base}/v2/hydrometrie/observations_tr", 1),
            ("v2, 1000 mesures", f"{base}/v2/hydrometrie/observations_tr", 1000),
            ("v2, referentiel", f"{base}/v2/hydrometrie/referentiel/stations", 1),
        ]:
            try:
                r = requests.get(url, params={
                    "code_entite": code.replace(" ", ""),
                    "grandeur_hydro": "H", "size": taille,
                }, headers=ENTETES, timeout=30)
                extrait = r.text[:120].replace("\n", " ")
                print(f"  {libelle:<20} -> HTTP {r.status_code}  {extrait}")
            except Exception as e:
                print(f"  {libelle:<20} -> exception : {e}")
        sys.exit(0)

    if args.importer:
        print(f"Import depuis {args.importer}")
        importer_csv(args.importer)
        rapport()
        sys.exit(0)

    if args.depot:
        horodatage = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"[{horodatage}] Collecte vers {args.depot}")
        total = 0
        succes = 0
        echecs = 0

        for code, config in STATIONS.items():
            for grandeur in config["grandeurs"]:
                etiquette = f"{config['nom']} [{grandeur}]"
                try:
                    lignes = recuperer(code, grandeur)
                    decimales = 1 if grandeur == "H" else 3
                    ajoutees = ecrire_csv(lignes, args.depot, grandeur, decimales)
                    total += ajoutees
                    succes += 1
                    print(f"  {etiquette} : {len(lignes)} recues, {ajoutees} nouvelles")
                except Exception as e:
                    print(f"  {etiquette} : ECHEC ({e})")
                    echecs += 1

        if not args.sans_nappes:
            for bss, nom in PIEZOS.items():
                try:
                    lignes = recuperer_piezo(bss)
                    ajoutees = ecrire_piezo(lignes, args.depot)
                    total += ajoutees
                    succes += 1
                    print(f"  nappe {nom} : {len(lignes)} recues, {ajoutees} nouvelles")
                except Exception as e:
                    print(f"  nappe {nom} : ECHEC ({e})")
                    echecs += 1

        if args.radar:
            try:
                import radar as radar_mf
                if radar_mf.main(["--depot", args.depot]) == 0:
                    succes += 1
                else:
                    echecs += 1
            except Exception as e:
                print(f"  radar : ECHEC ({e})")
                echecs += 1

        print(f"  total ajoute : {total}")
        # Rouge seulement si rien n'a abouti : une source indisponible ne doit
        # pas masquer le succes des autres, mais un echec general doit se voir.
        sys.exit(1 if succes == 0 and echecs else 0)

    init()

    if args.export:
        exporter(args.export)
        sys.exit(0)

    if not args.rapport:
        horodatage = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"[{horodatage}] Collecte Hub'Eau")
        total = 0
        for code, config in STATIONS.items():
            for grandeur in config["grandeurs"]:
                try:
                    lignes = recuperer(code, grandeur)
                    nouvelles = enregistrer(lignes, grandeur)
                    total += nouvelles
                    print(f"  {config['nom']} [{grandeur}] : "
                          f"{len(lignes)} recues, {nouvelles} nouvelles")
                except Exception as e:
                    print(f"  {config['nom']} [{grandeur}] : ECHEC ({e})")
        if not args.sans_nappes:
            for bss, nom in PIEZOS.items():
                try:
                    lignes = recuperer_piezo(bss)
                    nouvelles = enregistrer(lignes, "nappe")
                    total += nouvelles
                    print(f"  nappe {nom} : {len(lignes)} recues, {nouvelles} nouvelles")
                except Exception as e:
                    print(f"  nappe {nom} : ECHEC ({e})")
        print(f"  total ajoute : {total}")

    rapport()
