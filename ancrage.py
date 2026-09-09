#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Raccorde la prevision a la derniere observation.

Les debits sortent deja recales : la correction d'erreur autoregressive du
modele pluie-debit les ramene sur la mesure. Mais la hauteur a Saint-Laurent
et la cote a Rochereau sont obtenues *ensuite*, en faisant passer ce debit par
des relations ajustees ailleurs — l'une sur des maxima mensuels au-dessus de
0,80 m, l'autre sur quatre crues. Chacune porte son propre biais, qui se voit
d'autant plus que l'on est loin de son domaine de calage : en etiage, la
relation de Saint-Laurent ne descend pas sous 0,576 m alors que l'echelle lit
0,542 m.

Le resultat est une prevision qui demarre a cote du dernier releve. C'est
faux au sens strict — la meilleure estimation de l'etat present est la mesure,
pas le modele — et c'est surtout illisible : l'oeil suit un trait qui saute.

On applique donc un decalage, egal a l'ecart constate a l'instant de la
prevision, qui s'efface a mesure que l'echeance s'allonge. Le raccord devient
exact, la dynamique prevue est preservee, et le biais de la relation cesse
d'etre impose au-dela de la duree ou la mesure presente renseigne encore.
"""
from __future__ import annotations

import numpy as np


def raccorder(valeurs_prevues, observe_final, horizon_decroissance_h=24.0,
              pas_h=1.0, plafond=None):
    """Applique un decalage decroissant pour que la prevision parte de l'observation.

    `horizon_decroissance_h` est la duree au bout de laquelle il ne reste qu'un
    tiers du decalage. On la cale sur le temps de reponse du bassin : au-dela,
    l'etat present ne renseigne plus, et forcer le raccord reviendrait a nier
    la dynamique prevue.

    `plafond` borne le decalage applique : au-dela, l'ecart ne releve plus d'un
    biais de conversion mais d'un probleme qu'il vaut mieux voir que masquer.
    """
    v = np.asarray(valeurs_prevues, dtype=float)
    if observe_final is None or not np.isfinite(observe_final) or not len(v):
        return v
    ecart = float(observe_final) - float(v[0])
    if plafond is not None and abs(ecart) > plafond:
        ecart = float(np.sign(ecart) * plafond)
    # Decalage plein sur le premier pas, pour un raccord exact a l'oeil.
    h = np.arange(len(v)) * pas_h
    return v + ecart * np.exp(-h / max(horizon_decroissance_h, 1e-6))


def raccorder_quantiles(quantiles: dict, observe_final, mediane="50", **kw) -> dict:
    """Raccorde un faisceau entier sur le meme decalage.

    Le decalage se calcule sur la mediane et s'applique tel quel a tous les
    quantiles : le corriger separement deformerait la largeur du faisceau,
    alors que l'incertitude de la prevision, elle, n'a pas change.
    """
    ref = quantiles.get(mediane)
    if ref is None or observe_final is None or not np.isfinite(observe_final):
        return quantiles
    v = np.asarray(ref, dtype=float)
    if not len(v):
        return quantiles
    ecart = float(observe_final) - float(v[0])
    plafond = kw.get("plafond")
    if plafond is not None and abs(ecart) > plafond:
        ecart = float(np.sign(ecart) * plafond)
    duree = kw.get("horizon_decroissance_h", 24.0)
    pas = kw.get("pas_h", 1.0)
    h = np.arange(len(v)) * pas
    glissement = ecart * np.exp(-h / max(duree, 1e-6))
    return {q: [round(float(x), 4) for x in (np.asarray(val, dtype=float) + glissement)]
            for q, val in quantiles.items()}
