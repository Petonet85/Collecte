"""Prevision de hauteur a Saint-Laurent-sur-Sevre, station sans debit.

Saint-Laurent ne publie aucun debit, ni en temps reel ni en historique : ni
modele pluie-debit calable sur place, ni courbe de tarage. Mais 73 % de son
bassin est jauge en amont — Saint-Mesmin (359 km²) et l'Ouin (61 km²), qui ont
respectivement 32 et 36 ans de debits journaliers.

La chaine contourne donc l'absence de debit local :
  1. un modele GR par station amont, cale sur son propre historique ;
  2. la somme des trajectoires amont, membre par membre (elles partagent les
     memes membres de pluie, donc restent coherentes) ;
  3. une relation debit amont -> hauteur aval, ajustee sur 21 ans de maxima
     mensuels : la seule facon d'atteindre la gamme des crues sans attendre
     qu'une crue survienne pendant la periode d'archivage ;
  4. des seuils en metres, par loi de Gumbel sur les maxima annuels de hauteur.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .forecast import Context, build_context, run
from .http import get_paginated
from .model import rating
from .model.metrics import kge
from .sources import hubeau as hb

CIBLE = "M703243010"          # Saint-Laurent-sur-Sevre, station limnimetrique
SITE_CIBLE = "M7032430"
AMONT = {"M702241010": ("Sèvre Nantaise à Saint-Mesmin", 359.0),
         "M704401010": ("Ouin à Mauléon", 61.0)}
SURFACE_CIBLE = 576.0
OBS_ELAB = "https://hubeau.eaufrance.fr/api/v2/hydrometrie/obs_elab"


def _elabore(site: str, grandeur: str, diviseur: float = 1000.0) -> pd.Series:
    rows = get_paginated(OBS_ELAB,
                         {"code_entite": site, "grandeur_hydro_elab": grandeur,
                          "size": 1000, "sort": "asc",
                          "fields": "date_obs_elab,resultat_obs_elab"},
                         ttl=7 * 86400)
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series([r["resultat_obs_elab"] for r in rows],
                  index=pd.to_datetime([r["date_obs_elab"] for r in rows],
                                       format="mixed")).sort_index() / diviseur
    return s[~s.index.duplicated(keep="last")]


def relation_transfert(seuil_h: float = 0.8, fin_calage: str = "2020-01-01"):
    """Ajuste Q_amont -> H_aval sur les maxima mensuels, et la valide hors periode.

    L'ajustement se fait dans le sens d'une courbe de tarage (Q en fonction de H)
    puis s'inverse : le debit couvre trois ordres de grandeur la ou la hauteur
    n'en couvre qu'un, ce qui donne aux crues le poids qu'elles doivent avoir.
    Ajuster dans l'autre sens laisse les basses eaux, dix fois plus nombreuses,
    imposer un exposant non physique et sous-estimer chaque crue.

    Seuls les mois depassant `seuil_h` entrent dans le calage : la relation ne
    sert qu'a prevoir des crues, et l'etiage n'y apporte que du bruit.
    """
    h = _elabore(SITE_CIBLE, "HIXM")
    q_amont = sum(_elabore(code[:8], "QIXM") for code in AMONT)
    df = pd.DataFrame({"H": h, "Q": q_amont}).dropna()
    calage = df[(df.index < fin_calage) & (df["H"] >= seuil_h)]
    courbe = rating.fit(calage["H"], calage["Q"])
    if courbe is None:
        raise RuntimeError("relation de transfert inajustable")

    plancher = float(calage["H"].min())
    validation = df[(df.index >= fin_calage) & (df["H"] >= 1.5)]
    erreurs = courbe.to_h(validation["Q"].values) - validation["H"].values
    diagnostic = {
        "formule": f"Q_amont = {courbe.a:.2f} × (H − {courbe.h0:.3f})^{courbe.b:.3f}",
        "R2": round(courbe.r2, 4),
        "mois_calage": len(calage),
        "periode_calage": [str(calage.index[0].date()), str(calage.index[-1].date())],
        "validation_n_crues": int(len(validation)),
        "validation_biais_m": round(float(erreurs.mean()), 3),
        "validation_ecart_type_m": round(float(erreurs.std()), 3),
        "hauteur_plancher_m": round(plancher, 2),
        "debit_plancher_m3s": round(float(courbe.to_q(plancher)), 1),
    }
    return courbe, df, diagnostic


def seuils_hauteur(periodes=(2, 5, 10, 20, 50)) -> dict:
    """Periodes de retour en metres, par Gumbel sur les maxima annuels observes.

    Aucun debit n'etant publie ici, les seuils se calculent directement sur la
    hauteur — ce qui evite au passage toute incertitude de courbe de tarage.
    """
    h = _elabore(SITE_CIBLE, "HIXM")
    annuels = h.groupby(h.index.year + (h.index.month >= 9).astype(int)).agg(["max", "size"])
    annuels = annuels[annuels["size"] >= 10]["max"]
    if len(annuels) < 8:
        return {}
    x = annuels.to_numpy(dtype=float)
    sigma = x.std(ddof=1) * np.sqrt(6) / np.pi
    mu = x.mean() - 0.5772 * sigma
    out = {int(T): round(float(mu + sigma * (-np.log(-np.log(1 - 1.0 / T)))), 2)
           for T in periodes}
    out["n_annees"] = int(len(annuels))
    out["max_observe"] = round(float(x.max()), 2)
    return out


def prevoir(horizon_h: int = 72, verbose: bool = True) -> dict:
    """Chaine complete : previsions amont, sommation, conversion en hauteur."""
    def log(msg):
        if verbose:
            print(f"  [sevre] {msg}", flush=True)

    courbe, maxima, diag_transfert = relation_transfert()
    log(f"transfert : {diag_transfert['formule']} (R²={diag_transfert['R2']}), "
        f"validation {diag_transfert['validation_biais_m']:+.3f} "
        f"± {diag_transfert['validation_ecart_type_m']:.3f} m sur "
        f"{diag_transfert['validation_n_crues']} crues")

    seuils = seuils_hauteur()
    log(f"seuils : {seuils['n_annees']} maxima annuels — "
        + ", ".join(f"T{T}={seuils[T]} m" for T in (2, 5, 10, 20, 50) if T in seuils))

    trajectoires = None
    horodatage = None
    detail = {}
    for code, (nom, surface) in AMONT.items():
        log(f"prevision amont : {nom}…")
        ctx = build_context(code, verbose=verbose)
        res = run(ctx, horizon_h=horizon_h, verbose=verbose, retour_trajectoires=True)
        traj = res.pop("_trajectoires")
        horodatage = res.pop("_horodatage")
        res.pop("_q_observe", None)
        detail[code] = {"nom": nom, "surface_km2": surface, "prevision": res}
        # Les deux bassins partagent les memes membres de pluie : on peut sommer
        # trajectoire par trajectoire au lieu d'additionner des quantiles, ce qui
        # supposerait leurs rangs parfaitement correles et gonflerait l'incertitude.
        trajectoires = traj if trajectoires is None else trajectoires + traj

    # Les 156 km² intermediaires ne sont pas jauges : on les represente au prorata
    # de la surface, ce que la relation de transfert a de toute facon deja absorbe
    # puisqu'elle a ete calee sur le debit amont, non sur le debit total.
    quantiles_q = {p: np.percentile(trajectoires, p, axis=0) for p in (5, 10, 25, 50, 75, 90, 95)}
    quantiles_h = {p: courbe.to_h(v) for p, v in quantiles_q.items()}

    # La relation n'a ete calee que sur les mois depassant 0,8 m : en dessous elle
    # s'aplatit et rend une hauteur constante, incapable meme de reproduire le
    # niveau actuel. Hors de sa gamme, elle ne doit rien affirmer. On le declare
    # plutot que d'afficher une precision qui n'existe pas.
    q_plancher = diag_transfert["debit_plancher_m3s"]
    dans_la_gamme = quantiles_q[90] >= q_plancher
    part_valide = float(dans_la_gamme.mean())
    if part_valide < 1.0:
        log(f"hors gamme de validite sur {100*(1-part_valide):.0f} % de l'echeance "
            f"(debit amont sous {q_plancher:.0f} m³/s) : la hauteur prevue n'y est "
            "qu'indicative")

    h_obs = hb.hourly(hb.observations_tr(CIBLE, "H", 30)).dropna()
    depassements = {
        f"T{T}": {"hauteur": seuils[T],
                  "proba_max": round(float((courbe.to_h(trajectoires.max(axis=1)) >= seuils[T]).mean()), 3)}
        for T in (2, 5, 10, 20, 50) if T in seuils
    }
    log(f"H mediane a +{horizon_h} h : {quantiles_h[50][-1]:.2f} m "
        f"(80 % : {quantiles_h[10][-1]:.2f}–{quantiles_h[90][-1]:.2f}) — "
        f"observee {float(h_obs.iloc[-1]):.2f} m")

    return {
        "station": "La Sèvre Nantaise à Saint-Laurent-sur-Sèvre",
        "code_station": CIBLE, "surface_bv_km2": SURFACE_CIBLE,
        "date_prevision": str(horodatage[0] - pd.Timedelta(hours=1)),
        "time": [d.isoformat() for d in horodatage],
        "Q_amont": {str(p): list(np.round(v, 2)) for p, v in quantiles_q.items()},
        "H": {str(p): list(np.round(v, 3)) for p, v in quantiles_h.items()},
        "observe": {"time": [d.isoformat() for d in h_obs.index],
                    "H": list(np.round(h_obs.to_numpy(), 3))},
        "seuils": seuils, "depassements": depassements,
        "validite": {"part_dans_la_gamme": round(part_valide, 3),
                     "hauteur_plancher_m": diag_transfert["hauteur_plancher_m"],
                     "debit_plancher_m3s": q_plancher,
                     "dans_la_gamme": [bool(x) for x in dans_la_gamme]},
        "transfert": diag_transfert,
        "amont": {c: {"nom": d["nom"], "surface_km2": d["surface_km2"]} for c, d in detail.items()},
        "_detail_amont": detail,
    }


# --------------------------------------------------------------------------- #
# Transfert vers Rochereau (Mortagne-sur-Sevre)
# --------------------------------------------------------------------------- #

# Bief cale sur quatre observations couvrant 2 a 290 m3/s :
#   - ligne d'eau du vol LiDAR du 24/03/2022, dont le debit ce jour-la est connu ;
#   - deux reperes de crue releves sur photos (octobre 2024, janvier 2025) ;
#   - la profondeur d'eau mesuree en etiage, qui fixe la cote du fond ;
#   - le niveau d'avril 1983, qui a revele que la relation hauteur-debit de
#     Saint-Laurent surestimait de 16 % les debits hors de son domaine calibre.
ROCHEREAU = {
    "z_fond": 56.05, "a": 0.3371, "b": 0.470,
    "h_calibre_max": 2.54,          # au-dela, la relation amont est corrigee
    "h_1983": 4.25, "q_1983": 290.0,
    "seuils": {"porte de la maison": 59.35, "atelier": 59.55},
}


def _correction_amont(h: float, rc, k: float) -> float:
    """Debit amont pour une hauteur a Saint-Laurent, corrige hors domaine calibre."""
    hc = ROCHEREAU["h_calibre_max"]
    part = 0.0 if h <= hc else (h - hc) / (ROCHEREAU["h_1983"] - hc)
    return float(rc.to_q(h)) * float(np.exp(k * part))


def niveau_rochereau(h_saint_laurent, rc=None):
    """Convertit une hauteur a Saint-Laurent (m a l'echelle) en cote NGF a Rochereau."""
    import numpy as _np
    if rc is None:
        rc, _, _ = relation_transfert()
    k = float(_np.log(ROCHEREAU["q_1983"] / float(rc.to_q(ROCHEREAU["h_1983"]))))
    h = _np.atleast_1d(_np.asarray(h_saint_laurent, dtype=float))
    q = _np.array([_correction_amont(float(x), rc, k) for x in h])
    return ROCHEREAU["z_fond"] + ROCHEREAU["a"] * q ** ROCHEREAU["b"]


def marges(z_ngf):
    """Hauteur d'eau au-dessus (ou en dessous) de chaque seuil de la propriete."""
    import numpy as _np
    z = _np.asarray(z_ngf, dtype=float)
    return {nom: _np.round(z - cote, 3) for nom, cote in ROCHEREAU["seuils"].items()}
