#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Archivage local des hauteurs d'eau publiees par Hub'Eau.

Hub'Eau n'expose qu'environ un mois glissant en pas de temps fin. Ce script
recupere ce qui est disponible et l'accumule dans une base locale, de facon
idempotente : le relancer plusieurs fois par jour ne cree aucun doublon.

    python collecte.py                 # collecte et rapport
    python collecte.py --rapport       # rapport seul, sans appel reseau
    python collecte.py --export serie.csv

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

BASE = os.path.dirname(os.path.abspath(__file__))
FICHIER = os.path.join(BASE, "historique.db")

# Codes a 10 caracteres = station de mesure, a 8 = site (qui peut porter
# plusieurs stations). Hub'Eau accepte les deux dans code_entite.
# Verifiez toute nouvelle entree avec : python collecte.py --verifier
STATIONS = {
    "M703243010": "La Sevre Nantaise a Saint-Laurent-sur-Sevre",
    "M702241010": "La Sevre Nantaise a Saint-Mesmin",
    # "M7014010": "Le Sevreau a Cerizay",   # code a confirmer
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


def init():
    with sqlite3.connect(FICHIER) as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS hauteurs (
                station  TEXT NOT NULL,
                instant  TEXT NOT NULL,
                h_cm     REAL NOT NULL,
                PRIMARY KEY (station, instant)
            )
        """)
        cx.execute("CREATE INDEX IF NOT EXISTS idx_instant ON hauteurs(station, instant)")


def recuperer(code, taille=20000):
    """Toutes les observations de hauteur disponibles pour une station.

    La v2 pagine par curseur : chaque reponse porte l'URL de la page suivante
    dans son champ "next". On la suit telle quelle plutot que de reconstruire
    les parametres.
    """
    lot = []
    url = API
    params = {
        "code_entite": code.replace(" ", ""),
        "grandeur_hydro": "H",
        "size": min(taille, TAILLE_PAGE),
        "sort": "asc",
    }

    while url:
        r = requests.get(url, params=params, headers=ENTETES, timeout=60)
        if r.status_code == 404:
            return []
        if not r.ok:
            detail = r.text[:200].replace("\n", " ")
            raise RuntimeError(f"HTTP {r.status_code} — {detail}")

        contenu = r.json()
        donnees = contenu.get("data", [])
        lot.extend(donnees)

        if not donnees or len(lot) >= taille:
            break
        url = contenu.get("next")
        params = None      # l'URL "next" porte deja tous les parametres

    return [
        (code.replace(" ", ""), o["date_obs"], o["resultat_obs"] / 10.0)
        for o in lot
        if o.get("resultat_obs") is not None and o.get("date_obs")
    ]


REFERENTIEL = "https://hubeau.eaufrance.fr/api/v2/hydrometrie/referentiel"


def verifier():
    """Identifie chaque code configure : nom, commune, position, surface.

    Evite de collecter pendant des semaines sur un code errone, et confirme
    au passage la surface officielle du bassin versant.
    """
    for code, libelle in STATIONS.items():
        code = code.replace(" ", "")
        print(f"\n  {code}  (configure comme : {libelle})")
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


def enregistrer(lignes):
    """Insere sans ecraser : une mesure deja archivee fait foi."""
    if not lignes:
        return 0
    with sqlite3.connect(FICHIER) as cx:
        avant = cx.execute("SELECT COUNT(*) FROM hauteurs").fetchone()[0]
        cx.executemany(
            "INSERT OR IGNORE INTO hauteurs (station, instant, h_cm) VALUES (?,?,?)",
            lignes,
        )
        apres = cx.execute("SELECT COUNT(*) FROM hauteurs").fetchone()[0]
    return apres - avant


def ecrire_csv(lignes, dossier):
    """Archive en CSV mensuels, un dossier par station.

    Format texte plutot que SQLite : git versionne alors des ajouts de lignes
    au lieu de reecrire un binaire entier a chaque collecte.
    """
    par_fichier = {}
    for station, instant, h in lignes:
        cle = (station, instant[:7])           # AAAA-MM
        par_fichier.setdefault(cle, {})[instant] = h

    ajouts = 0
    for (station, mois), mesures in sorted(par_fichier.items()):
        chemin = os.path.join(dossier, station, f"{mois}.csv")
        os.makedirs(os.path.dirname(chemin), exist_ok=True)

        existant = {}
        if os.path.exists(chemin):
            with open(chemin, newline="") as f:
                for ligne in csv.DictReader(f):
                    existant[ligne["instant_utc"]] = float(ligne["hauteur_cm"])

        nouvelles = {k: v for k, v in mesures.items() if k not in existant}
        if not nouvelles:
            continue
        existant.update(nouvelles)
        ajouts += len(nouvelles)

        with open(chemin, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["instant_utc", "hauteur_cm"])
            for instant in sorted(existant):
                w.writerow([instant, round(existant[instant], 1)])

    return ajouts


def importer_csv(dossier):
    """Reconstruit la base SQLite locale a partir des CSV du depot."""
    if not os.path.isdir(dossier):
        print(f"  Dossier introuvable : {dossier}")
        return 0
    init()
    total = 0
    for station in sorted(os.listdir(dossier)):
        chemin_station = os.path.join(dossier, station)
        if not os.path.isdir(chemin_station):
            continue
        lignes = []
        for nom in sorted(os.listdir(chemin_station)):
            if not nom.endswith(".csv"):
                continue
            with open(os.path.join(chemin_station, nom), newline="") as f:
                for l in csv.DictReader(f):
                    lignes.append((station, l["instant_utc"], float(l["hauteur_cm"])))
        ajoutees = enregistrer(lignes)
        total += ajoutees
        print(f"  {station} : {len(lignes)} lues, {ajoutees} nouvelles")
    return total


def rapport():
    if not os.path.exists(FICHIER):
        print("Aucune base : lancez d'abord une collecte.")
        return
    with sqlite3.connect(FICHIER) as cx:
        lignes = cx.execute("""
            SELECT station, COUNT(*) n, MIN(instant) d, MAX(instant) f
            FROM hauteurs GROUP BY station
        """).fetchall()

    if not lignes:
        print("Base vide.")
        return

    print("\n  Archive locale")
    for station, n, debut, fin in lignes:
        nom = STATIONS.get(station, station)
        jours = (datetime.fromisoformat(fin.replace("Z", "+00:00"))
                 - datetime.fromisoformat(debut.replace("Z", "+00:00"))).days
        print(f"    {nom}")
        print(f"      {n:>8} mesures  du {debut[:10]} au {fin[:10]}  ({jours} jours)")
        if jours:
            print(f"      soit environ une mesure toutes les {jours * 24 * 60 / n:.0f} min")


def exporter(chemin):
    with sqlite3.connect(FICHIER) as cx:
        lignes = cx.execute(
            "SELECT station, instant, h_cm FROM hauteurs ORDER BY station, instant"
        ).fetchall()
    with open(chemin, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["station", "instant_utc", "hauteur_cm"])
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
        print(f"[{horodatage}] Collecte Hub'Eau vers {args.depot}")
        total = 0
        echec = False
        for code, nom in STATIONS.items():
            try:
                lignes = recuperer(code)
                ajoutees = ecrire_csv(lignes, args.depot)
                total += ajoutees
                print(f"  {nom} : {len(lignes)} recues, {ajoutees} nouvelles")
            except Exception as e:
                print(f"  {nom} : ECHEC ({e})")
                echec = True
        print(f"  total ajoute : {total}")
        # Sortie en erreur si toutes les stations ont echoue : la tache
        # planifiee doit etre visiblement rouge, pas silencieusement vide.
        sys.exit(1 if echec and total == 0 else 0)

    init()

    if args.export:
        exporter(args.export)
        sys.exit(0)

    if not args.rapport:
        horodatage = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"[{horodatage}] Collecte Hub'Eau")
        total = 0
        for code, nom in STATIONS.items():
            try:
                lignes = recuperer(code)
                nouvelles = enregistrer(lignes)
                total += nouvelles
                print(f"  {nom} : {len(lignes)} recues, {nouvelles} nouvelles")
            except Exception as e:
                print(f"  {nom} : ECHEC ({e})")
        print(f"  total ajoute : {total}")

    rapport()
