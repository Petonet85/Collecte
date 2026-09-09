#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Produit la page de prevision pour Rochereau et la depose dans site/.

Enchaine la chaine FloodCast sur les deux stations amont de la Sevre Nantaise,
convertit le resultat en cote NGF a Rochereau via le bief cale sur quatre
observations de terrain, et regenere la page consultable.

    python prevoir.py            # ecrit docs/index.html
    python prevoir.py --horizon 96
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
SITE = os.path.join(BASE, "docs")


def construire(horizon_h=72, verbose=True):
    from floodcast import sevre

    courbe, _, _ = sevre.relation_transfert()
    res = sevre.prevoir(horizon_h=horizon_h, verbose=verbose)
    detail = res.pop("_detail_amont", {}) or {}

    prevision = {
        "date_prevision": res["date_prevision"],
        "time": res["time"],
        "h_saint_laurent": res["H"],
        "propagation": res.get("propagation"),
        "transfert": res.get("transfert"),
        "observe_h": res["observe"]["H"][-1] if res["observe"]["H"] else None,
        "genere_le": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "z_rochereau": {},
    }
    # La cote a Rochereau se calcule sur la hauteur DECALEE du temps de parcours
    # Saint-Laurent -> Rochereau, pas sur celle de l'echelle de Saint-Laurent :
    # treize kilometres de riviere separent les deux.
    for quantile, valeurs in res.get("H_rochereau", res["H"]).items():
        cotes = sevre.niveau_rochereau(np.asarray(valeurs, dtype=float), courbe)
        prevision["z_rochereau"][quantile] = [round(float(v), 4) for v in cotes]

    # Debit prevu station par station : Saint-Mesmin porte a lui seul 62 % du
    # bassin de Saint-Laurent, et c'est la seule des deux ou l'on dispose d'une
    # mesure de debit a comparer directement a la prevision.
    prevision["stations"] = {}
    for code, bloc in detail.items():
        p = bloc.get("prevision") or {}
        if not p.get("prevision"):
            continue
        prevision["stations"][code] = {
            "nom": bloc["nom"], "surface_km2": bloc["surface_km2"],
            "time": p["prevision"]["time"],
            "Q": p["prevision"]["Q"],
            "observe": {"time": p["observe"]["time"], "Q": p["observe"]["Q"]},
        }
    return prevision


def archiver(prevision):
    """Fige la prevision emise, pour pouvoir la juger plus tard.

    Sans cette trace, chaque passage ecrase le precedent et l'on ne peut jamais
    mesurer la justesse reelle de la chaine — seulement la rejouer sur des crues
    passees en connaissant deja la pluie, ce qui la flatte.
    """
    try:
        import verification
        chemin = verification.archiver(prevision, prevision.get("observe_h"))
        print(f"  prevision archivee : {os.path.relpath(chemin, BASE)}")
    except Exception as exc:  # noqa: BLE001 - l'archive ne doit jamais bloquer la prevision
        print(f"  archivage impossible ({exc})")


def ecrire_tableau(prevision, horizon_h=72):
    """Tableau de bord : hydrogramme, seuils de la propriete, pluie du bassin."""
    import donnees_tableau

    donnees = donnees_tableau.assembler(prevision, horizon_h)
    with open(os.path.join(SITE, "gabarit_tableau.html"), encoding="utf-8") as fh:
        page = fh.read()
    page = page.replace("__DATA__", json.dumps(donnees, ensure_ascii=False, separators=(",", ":")))
    chemin = os.path.join(SITE, "index.html")
    with open(chemin, "w", encoding="utf-8") as fh:
        fh.write(page)
    with open(os.path.join(SITE, "tableau.json"), "w", encoding="utf-8") as fh:
        json.dump(donnees, fh, ensure_ascii=False)
    return chemin


def ecrire_page(prevision):
    with open(os.path.join(SITE, "gabarit.html"), encoding="utf-8") as fh:
        page = fh.read()
    carte = json.load(open(os.path.join(SITE, "carte.json"), encoding="utf-8"))
    carte["mnt"] = json.load(open(os.path.join(SITE, "mnt_fin.json"), encoding="utf-8"))
    carte["reperes"] = REPERES
    calage = json.load(open(os.path.join(SITE, "calage.json"), encoding="utf-8"))
    calage["prevision"] = prevision

    page = page.replace("__DATA__", json.dumps(carte, ensure_ascii=False, separators=(",", ":")))
    page = page.replace("__CALAGE__", json.dumps(calage, ensure_ascii=False, separators=(",", ":")))
    chemin = os.path.join(SITE, "carte.html")
    with open(chemin, "w", encoding="utf-8") as fh:
        fh.write(page)
    with open(os.path.join(SITE, "prevision.json"), "w", encoding="utf-8") as fh:
        json.dump(prevision, fh, ensure_ascii=False)
    return chemin


# Les deux reperes de crue releves sur le terrain, figes : ce sont des mesures.
REPERES = [
    {"lon": -0.99276, "lat": 47.000408, "z": 59.95, "evenement": "janvier 2025"},
    {"lon": -0.992706, "lat": 47.000358, "z": 59.46, "evenement": "octobre 2024",
     "z_predit": 59.46},
]


def main(argv=None):
    p = argparse.ArgumentParser(description="Prevision de crue a Rochereau")
    p.add_argument("--horizon", type=int, default=72, help="echeance en heures")
    p.add_argument("--silencieux", action="store_true")
    args = p.parse_args(argv)

    prevision = construire(args.horizon, verbose=not args.silencieux)
    # Archiver AVANT d'ecrire les pages : si la generation echoue plus loin, la
    # prevision emise reste tracee, ce qui est le seul moment ou on peut la figer.
    archiver(prevision)
    carte = ecrire_page(prevision)
    chemin = ecrire_tableau(prevision, args.horizon)

    z = prevision["z_rochereau"]["50"]
    haut = prevision["z_rochereau"]["90"]
    pic, pic90 = max(z), max(haut)
    seuils = json.load(open(os.path.join(SITE, "seuils.json"), encoding="utf-8"))
    print(f"\n  tableau de bord : {chemin}\n  carte : {carte}")
    print(f"  pic median {pic:.2f} m NGF (scenario haut {pic90:.2f})")
    for nom, cote in (("porte de la maison", seuils["maison"]), ("atelier", seuils["atelier"])):
        print(f"    {nom:<20} {100 * (pic - cote):+6.0f} cm   "
              f"(scenario haut {100 * (pic90 - cote):+.0f} cm)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
