"""Assimilation : correction des erreurs recentes du modele et incertitude.

Un modele pluie-debit derive lentement (etat des reservoirs, biais de pluie).
Les dernieres heures observees disent dans quel sens il se trompe : on prolonge
cette erreur en la faisant decroitre selon son autocorrelation estimee.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ErrorModel:
    rho: float            # autocorrelation a 1 h des residus (log)
    e0: float             # dernier residu observe (log)
    sigma: float          # ecart-type des residus autour de leur moyenne (log)
    n: int
    e_moy: float = 0.0    # residu moyen sur la fenetre : la cible du retour
    q_haut: float = 0.0   # debit le plus fort vu dans la fenetre d'assimilation

    def poids_regime(self, q_modele) -> np.ndarray:
        """Poids du biais persistant selon qu'on reste ou non dans le regime observe.

        Le biais mesure sur douze jours d'etiage ne dit rien du biais en crue. La
        mesure est meme formelle : le modele SURsimule d'un facteur deux en
        basses eaux et SOUSsimule de 23 a 58 % au pic des crues rejouees. Faire
        persister le biais d'etiage dans une crue diviserait par deux une
        prevision deja trop basse — le calcul donne -71 % du pic reel au lieu de
        -44 %, et c'est le mauvais sens pour se tromper.

        On garde donc le biais tant que le debit reste dans la gamme ou il a ete
        vu, et on l'efface progressivement au-dela : sur une decade de debit, le
        poids passe de 1 a 0 et la correction revient a son comportement neutre,
        c'est-a-dire une simple extinction de l'ecart du moment.
        """
        if not self.q_haut or self.q_haut <= 0:
            return np.ones_like(np.asarray(q_modele, dtype=float))
        q = np.maximum(np.asarray(q_modele, dtype=float), 1e-9)
        return np.clip(1.0 - np.log10(q / self.q_haut), 0.0, 1.0)

    def correction(self, horizons: np.ndarray, q_modele=None) -> np.ndarray:
        """Facteur multiplicatif a appliquer au debit simule, par echeance.

        Un AR(1) revient vers la MOYENNE de ses residus, pas vers zero. Revenir
        vers zero reviendrait a affirmer que le modele redevient sans biais en
        deux jours — ce que la fenetre d'assimilation dement souvent a plat : a
        Saint-Mesmin, sur douze jours, le residu reste entre -0,96 et -0,34 sans
        jamais approcher zero, avec 0,10 d'ecart-type. Le modele y est
        structurellement 50 a 58 % trop haut.

        L'erreur etait visible a l'oeil : sans une goutte de pluie prevue, le
        debit annonce doublait sur l'echeance, par la seule disparition de la
        correction. Le rho etait d'ailleurs deja estime sur les residus CENTRES,
        donc autour de leur moyenne ; seule l'extrapolation l'avait oubliee.
        """
        h = np.asarray(horizons, dtype=float)
        t = 1.0 if q_modele is None else self.poids_regime(q_modele)
        cible = t * self.e_moy
        return np.exp(cible + (self.e0 - cible) * self.rho ** h)

    def spread(self, horizons: np.ndarray) -> np.ndarray:
        """Ecart-type (log) de l'erreur residuelle, croissant avec l'echeance."""
        h = np.asarray(horizons, dtype=float)
        var = self.sigma ** 2 * (1 - self.rho ** (2 * h))
        return np.sqrt(np.maximum(var, (0.15 * self.sigma) ** 2))

    def to_dict(self) -> dict:
        return {"rho_1h": round(self.rho, 4), "q_haut_m3s": round(self.q_haut, 4),
                "biais_actuel_pct": round(100 * (float(np.exp(self.e0)) - 1), 1),
                "biais_persistant_pct": round(100 * (float(np.exp(self.e_moy)) - 1), 1),
                "sigma_log": round(self.sigma, 4), "n_pas": self.n}


def fit(q_obs: np.ndarray, q_sim: np.ndarray, min_q: float = 1e-3,
        sigma_floor: float = 0.0) -> ErrorModel:
    """Estime un AR(1) sur les residus logarithmiques des dernieres heures.

    `sigma_floor` est un plancher d'incertitude issu du rejeu long terme. Sans lui,
    une fenetre d'etiage calme produit une dispersion quasi nulle et un faisceau de
    prevision faussement etroit : l'erreur du modele en crue n'a rien a voir avec
    son erreur en basses eaux.
    """
    o = np.asarray(q_obs, dtype=float)
    s = np.asarray(q_sim, dtype=float)
    m = np.isfinite(o) & np.isfinite(s) & (o > min_q) & (s > min_q)
    if m.sum() < 12:
        return ErrorModel(rho=0.9, e0=0.0, sigma=max(0.35, sigma_floor), n=int(m.sum()),
                          e_moy=0.0, q_haut=0.0)
    e = np.log(o[m]) - np.log(s[m])
    e_c = e - e.mean()
    denom = float((e_c[:-1] ** 2).sum())
    rho = float((e_c[:-1] * e_c[1:]).sum() / denom) if denom > 0 else 0.9
    rho = float(np.clip(rho, 0.0, 0.995))
    sigma = float(max(e.std(), sigma_floor, 1e-3))
    return ErrorModel(rho=rho, e0=float(e[-1]), sigma=sigma, n=int(m.sum()),
                      e_moy=float(e.mean()), q_haut=float(np.percentile(o[m], 90)))
