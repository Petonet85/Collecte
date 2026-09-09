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

import json
import os

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


class CourbeTransfert:
    """Relation debit amont <-> hauteur a Saint-Laurent, tabulee et monotone.

    Tabulee plutot qu'ajustee : la relation n'est pas une loi de puissance. Le
    lit deborde, la section change de nature, et un exposant unique impose sur
    trois ordres de grandeur de debit se trompe aux deux bouts a la fois.
    Monotone par construction, donc exactement inversible — ce qui compte, la
    chaine faisant l'aller-retour hauteur/debit pour atteindre Rochereau.

    Hors du domaine mesure on prolonge par la pente locale en log-debit. C'est
    une extrapolation et rien d'autre ; au-dela, le calage de Rochereau reprend
    la main en ancrant sur la crue de 1983.
    """

    def __init__(self, q, h, meta: dict | None = None):
        self.q = np.asarray(q, dtype=float)
        self.h = np.asarray(h, dtype=float)
        self.meta = meta or {}

    @staticmethod
    def _rendre(sortie, scalaire):
        return float(sortie[0]) if scalaire else sortie

    def to_h(self, q):
        scalaire = np.ndim(q) == 0
        q = np.atleast_1d(np.asarray(q, dtype=float))
        lq = np.log10(np.clip(q, 1e-6, None))
        out = np.interp(lq, np.log10(self.q), self.h)
        haut = q > self.q[-1]
        if np.any(haut):
            pente = (self.h[-1] - self.h[-2]) / (np.log10(self.q[-1]) - np.log10(self.q[-2]))
            out = np.where(haut, self.h[-1] + pente * (lq - np.log10(self.q[-1])), out)
        return self._rendre(out, scalaire)

    def to_q(self, h):
        scalaire = np.ndim(h) == 0
        h = np.atleast_1d(np.asarray(h, dtype=float))
        out = 10 ** np.interp(h, self.h, np.log10(self.q))
        haut = h > self.h[-1]
        if np.any(haut):
            pente = (np.log10(self.q[-1]) - np.log10(self.q[-2])) / (self.h[-1] - self.h[-2])
            out = np.where(haut, 10 ** (np.log10(self.q[-1]) + pente * (h - self.h[-1])), out)
        return self._rendre(out, scalaire)


def calage_transfert() -> dict | None:
    chemin = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "calage_transfert.json")
    try:
        with open(chemin, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def relation_transfert(seuil_h: float = 0.8, fin_calage: str = "2020-01-01"):
    """Relation debit amont <-> hauteur a Saint-Laurent.

    Elle vient d'abord de data/calage_transfert.json : 170 000 couples
    reellement simultanes, tires des chroniques instantanees de HydroPortail sur
    27 crues et 14 etiages, le debit amont etant ramene a l'heure ou il se
    manifeste a l'aval grace au retard cale par ailleurs. Voir caler_transfert.py.

    Sans ce fichier on retombe sur l'ajustement historique, sur les maxima
    mensuels. Il avait deux faiblesses que la version tabulee corrige. Les deux
    maxima d'un mois ne sont pas forcement le meme evenement, ce qui appariait
    parfois une pointe amont avec une crue aval venue d'ailleurs. Et il ne
    couvrait rien sous 0,80 m : la relation s'y aplatissait et rendait une
    hauteur constante, incapable de reproduire l'etiage du jour, ce qui obligeait
    le tableau de bord a la contourner.
    """
    cal = calage_transfert()
    if cal:
        courbe = CourbeTransfert(cal["q"], cal["h"], cal)
        v = cal.get("validation", {})
        diagnostic = {
            "forme": cal.get("forme", "table monotone"),
            "formule": (f"table de {len(cal['q'])} noeuds, "
                        f"Q {cal['domaine_q'][0]}–{cal['domaine_q'][1]} m³/s"),
            "R2": None,
            "n_couples": cal.get("n_couples"),
            "n_episodes": cal.get("n_episodes"),
            "domaine_h": cal.get("domaine_h"),
            "domaine_q": cal.get("domaine_q"),
            # Validation croisee par episode : biais et dispersion en metres,
            # sous les memes cles que l'ancien diagnostic pour ne rien casser.
            "validation_n_crues": v.get("n_episodes"),
            "validation_biais_m": round((v.get("table_biais_cm") or 0.0) / 100, 3),
            "validation_ecart_type_m": round((v.get("table_pic_ecart_type_cm") or 0.0) / 100, 3),
            "validation_pic_biais_m": round((v.get("table_pic_biais_cm") or 0.0) / 100, 3),
            "hauteur_plancher_m": cal["domaine_h"][0],
            "debit_plancher_m3s": cal["domaine_q"][0],
            "source": cal.get("source"),
        }
        return courbe, None, diagnostic

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


# --- Propagation vers l'aval -------------------------------------------------
#
# Cale sur les crues passees, pas suppose : voir caler_celerite.py, qui mesure
# le decalage entre l'amont et l'aval sur les chroniques instantanees de
# HydroPortail depuis 2010.
#
# Deux troncons, et deux natures differentes.
#
# 1. Debit amont -> hauteur a Saint-Laurent. Ce retard n'est pas un temps de
#    trajet. La mesure l'etablit sans ambiguite : le pic de l'Ouin precede celui
#    de Saint-Laurent de 14 a 24 h alors que l'Ouin n'est qu'a 18 km, ce qui
#    donnerait une onde a 0,2 m/s, physiquement impossible. C'est que l'Ouin est
#    un bassin nerveux de 61 km² qui culmine tot, quand Saint-Laurent, dix fois
#    plus grand, met bien plus longtemps a rassembler son eau. Ce qu'on mesure
#    melange donc le trajet et l'ecart de reponse entre bassins — et c'est
#    exactement le retard dont la chaine a besoin, puisqu'elle relie ces deux
#    memes grandeurs. Il decroit avec l'ampleur de la crue (p = 0,003) : versants
#    satures et lit plein rendent l'eau plus vite.
#
# 2. Saint-Laurent -> Rochereau. Aucune station a Rochereau, mais il y en a une
#    a Tiffauges, 25,1 km plus bas, et Rochereau est a 13,2 km sur ce trajet.
#    Meme riviere, meme regime : le retard mesure y est essentiellement un temps
#    de parcours, 4,41 h pour 25,1 km, soit 1,63 ± 0,29 m/s sur 16 crues. Aucune
#    dependance a la cote n'y est detectable (pente +0,09, p = 0,63), on garde
#    donc une celerite constante. C'est au passage la mesure qui montre que les
#    1,2 m/s supposes auparavant etaient 26 % trop lents.
_CALAGE_DEFAUT = {
    "amont_vers_saint_laurent": {"a": 14.455, "n": 0.469, "h_calage": [1.03, 2.63],
                                 "ecart_type_h": 1.21, "n_crues": 17, "R2_log": 0.458},
    "saint_laurent_vers_rochereau": {"distance_km": 13.2, "celerite_ms": 1.626,
                                     "retard_h": 2.26, "n_crues": 16},
}


def calage_propagation() -> dict:
    chemin = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "calage_propagation.json")
    try:
        with open(chemin, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return _CALAGE_DEFAUT


def retard_amont(h_pic: float, calage: dict | None = None) -> float:
    """Retard entre le debit amont et la cote a Saint-Laurent, en heures.

    Borne a la gamme de cotes sur laquelle la relation a ete calee : au-dela,
    la loi de puissance divergerait sans qu'aucune crue ne l'ait verifiee.
    """
    c = (calage or calage_propagation())["amont_vers_saint_laurent"]
    bas, haut = c.get("h_calage", [1.0, 2.7])
    h = float(np.clip(h_pic, bas, haut))
    return float(c["a"] / h ** c["n"])


def retard_rochereau(calage: dict | None = None) -> float:
    c = (calage or calage_propagation())["saint_laurent_vers_rochereau"]
    return float(c.get("retard_h") or c["distance_km"] * 1000 / c["celerite_ms"] / 3600)


RECOUTURE_H = 6.0        # duree d'effacement du raccord mesure -> modele
LISSAGE_CRUE_H = 3
LISSAGE_ETIAGE_H = 25


def _historique(q_observe, t0, n_heures: int, defaut, lissage_h: int = LISSAGE_CRUE_H):
    """Les `n_heures` derniers debits horaires mesures, se terminant a t0.

    Ces valeurs alimentent le debut de l'echeance : ce qui se manifestera bientot
    a l'aval decoule de debits amont deja passes. On veut en retenir la
    tendance, pas les soubresauts.

    La fenetre depend du regime, parce que le bruit n'a pas la meme periode.

    En crue, trois heures. Sur les crues archivees ce lissage coute de 0,3 a
    1,7 % sur la pointe de debit, loin sous les ±15 cm d'incertitude de la
    relation de transfert ; a cinq heures on perdrait jusqu'a 3,6 %.

    En etiage, vingt-cinq heures, parce que ce qu'il faut retirer est un cycle
    *journalier*. Les deux stations amont montrent le meme creux au meme moment,
    minimum vers 11 h et retour la nuit : c'est l'evapotranspiration de la
    ripisylve, qui pompe le jour et lache la nuit. Le cycle est reel, mais il
    est local, quotidien et se referme sur lui-meme — il n'annonce rien a
    l'aval. La courbe de tarage de l'Ouin l'amplifie enormement : 2 cm d'eau y
    font varier le debit de 700 %. Transporte quatorze heures plus loin par un
    retard cale sur des crues, ce cycle produisait a Saint-Laurent un creux
    d'un centimetre vers 3 h du matin qui n'avait aucun sens. Une fenetre de
    trois heures n'y pouvait rien : elle ne voit pas une periode de 24 h.
    """
    idx = pd.date_range(t0 - pd.Timedelta(hours=n_heures - 1), t0, freq="h")
    if q_observe is None or not len(q_observe):
        return np.full(n_heures, defaut, dtype=float)
    serie = q_observe.rolling(lissage_h, center=True, min_periods=1).mean()
    serie = serie.reindex(idx).interpolate(limit_direction="both")
    return np.nan_to_num(serie.to_numpy(dtype=float), nan=defaut)


def translater(traj, q_observe, t0, retard: float, pas_h: float = 1.0,
               lissage_h: int = LISSAGE_CRUE_H):
    """Decale des trajectoires de debit du retard mesure vers l'aval.

    Le decalage n'est pas un artifice d'affichage : sur les premieres heures de
    l'echeance, ce qui se manifestera a l'aval decoule de debits amont deja
    passes, et on les prend mesures plutot que simules. Sur ce laps, la
    prevision aval cesse de dependre de la pluie a venir.

    Il faut alors recoudre. Au bout du retard, la serie change de source : elle
    quittait la mesure pour le modele d'un seul pas de temps, et l'ecart entre
    les deux — le biais du modele a l'instant present — sortait sous forme de
    marche. On le rattrape donc a la couture, avec un decalage qui s'efface en
    quelques heures : au raccord c'est la mesure qui a raison, plus loin c'est
    le modele. Le meme decalage est applique a toutes les trajectoires, jamais
    un decalage par membre : sinon le faisceau se pincerait a la couture, ce qui
    ferait croire a une certitude que l'incertitude de transfert dement.
    """
    traj = np.atleast_2d(np.asarray(traj, dtype=float))
    n = traj.shape[1]
    h_prev = np.arange(1, n + 1) * pas_h
    if retard <= 0:
        return traj.copy()
    k = int(np.ceil(retard / pas_h)) + 2
    hist = _historique(q_observe, t0, k, float(np.median(traj[:, 0])), lissage_h)

    # Recouture : on corrige les trajectoires AVANT de les mettre bout a bout
    # avec la mesure. Corriger apres coup reviendrait a appliquer l'ecart une
    # seconde fois sur le pas qui chevauche le raccord, et creuserait la
    # marche au lieu de la combler.
    ecart = float(hist[-1]) - float(np.median(traj[:, 0]))
    recolle = traj + ecart * np.exp(-(h_prev - pas_h) / RECOUTURE_H)
    recolle = np.maximum(recolle, 0.0)

    axe = np.concatenate([np.arange(-(k - 1), 1) * pas_h, h_prev])
    vise = h_prev - retard
    sortie = np.empty_like(traj)
    for m in range(traj.shape[0]):
        sortie[m] = np.interp(vise, axe, np.concatenate([hist, recolle[m]]))
    return sortie


def incertitude_transfert() -> dict:
    """Ecart-type de l'erreur de la relation de transfert, selon la cote.

    Mesure par validation croisee dans caler_transfert.py : on retire un
    episode, on recale sur les autres, on regarde l'erreur au pic. Elle croit
    avec la cote — 6 cm en basses eaux, 19 cm pour une crue a 2,6 m — parce
    qu'elle vient surtout de la part variable des 156 km² non jauges, dont la
    contribution pese d'autant plus que l'episode est fort.
    """
    chemin = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "calage_incertitude_transfert.json")
    try:
        with open(chemin, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {"a": 4.36, "b": 5.73, "sigma_mini_cm": 1.0}


def _bruiter_transfert(h_membres, tirage, calage):
    """Ajoute l'erreur de la relation de transfert, une par membre.

    Une seule realisation par trajectoire, tenue sur toute l'echeance, et non un
    bruit blanc a chaque pas : la mesure montre que cette erreur est une
    propriete de l'episode — la pluie tombe-t-elle plutot sur la partie jaugee
    du bassin ou sur les 156 km² intermediaires — et non une agitation qui se
    compenserait d'une heure a l'autre. Un bruit blanc ferait vibrer la courbe
    sans elargir le faisceau la ou il faut.

    Sans ce terme le faisceau etait nul sur toute la duree du retard, la ou le
    debit amont est mesure : il annoncait une certitude que la conversion en
    hauteur ne permet pas.
    """
    sigma = np.maximum(calage["a"] + calage["b"] * h_membres,
                       calage.get("sigma_mini_cm", 1.0)) / 100.0
    return np.maximum(h_membres + tirage * sigma, 0.0)


def prevoir(horizon_h: int = 72, verbose: bool = True) -> dict:
    """Chaine complete : previsions amont, sommation, conversion en hauteur."""
    def log(msg):
        if verbose:
            print(f"  [sevre] {msg}", flush=True)

    courbe, maxima, diag_transfert = relation_transfert()
    log(f"transfert : {diag_transfert['formule']}"
        + (f" (R²={diag_transfert['R2']})" if diag_transfert.get("R2") else "")
        + f", validation croisee {diag_transfert['validation_biais_m']:+.3f} m "
          f"sur {diag_transfert['validation_n_crues']} episodes")

    seuils = seuils_hauteur()
    log(f"seuils : {seuils['n_annees']} maxima annuels — "
        + ", ".join(f"T{T}={seuils[T]} m" for T in (2, 5, 10, 20, 50) if T in seuils))

    trajectoires = None
    q_amont_obs = None     # debit amont mesure, pour alimenter le debut d'echeance
    horodatage = None
    detail = {}
    for code, (nom, surface) in AMONT.items():
        log(f"prevision amont : {nom}…")
        ctx = build_context(code, verbose=verbose)
        res = run(ctx, horizon_h=horizon_h, verbose=verbose, retour_trajectoires=True)
        traj = res.pop("_trajectoires")
        horodatage = res.pop("_horodatage")
        q_obs = res.pop("_q_observe", None)
        detail[code] = {"nom": nom, "surface_km2": surface, "prevision": res}
        # Les deux bassins partagent les memes membres de pluie : on peut sommer
        # trajectoire par trajectoire au lieu d'additionner des quantiles, ce qui
        # supposerait leurs rangs parfaitement correles et gonflerait l'incertitude.
        trajectoires = traj if trajectoires is None else trajectoires + traj
        if q_obs is not None and len(q_obs):
            q_amont_obs = q_obs if q_amont_obs is None else q_amont_obs.add(q_obs, fill_value=0.0)

    t0 = horodatage[0] - pd.Timedelta(hours=1)
    calage = calage_propagation()

    # Le retard depend de l'ampleur de la crue, donc de la cote qu'on cherche a
    # prevoir : on la calcule d'abord sans decalage, uniquement pour lire le pic
    # attendu, puis on applique le retard que ce pic commande. Une seule passe
    # suffit — decaler ne change pas la valeur du maximum, seulement son heure.
    h_sans_retard = courbe.to_h(np.percentile(trajectoires, 50, axis=0))
    h_pic = float(np.nanmax(h_sans_retard))
    tau = retard_amont(h_pic, calage)
    c_am = calage["amont_vers_saint_laurent"]
    log(f"propagation amont → Saint-Laurent : {tau:.1f} h pour un pic a {h_pic:.2f} m "
        f"(cale sur {c_am['n_crues']} crues, ± {c_am['ecart_type_h']:.1f} h)")
    # Meme critere que le plafonnement du retard : sous la gamme de calage, on
    # n'est pas en crue, et le cycle journalier doit disparaitre de l'entree.
    bas_calage = c_am.get("h_calage", [1.0, 2.7])[0]
    etiage = h_pic < bas_calage
    lissage = LISSAGE_ETIAGE_H if etiage else LISSAGE_CRUE_H
    if etiage:
        log(f"regime d'etiage (pic attendu {h_pic:.2f} m < {bas_calage} m) : debit "
            f"amont injecte lisse sur {lissage} h pour ecarter le cycle journalier")
    trajectoires = translater(trajectoires, q_amont_obs, t0, tau, lissage_h=lissage)

    quantiles_q = {p: np.percentile(trajectoires, p, axis=0) for p in (5, 10, 25, 50, 75, 90, 95)}

    # La conversion en hauteur a sa propre incertitude, mesuree, et elle
    # manquait au faisceau. On la tire une fois par trajectoire, avec une graine
    # fixe pour que deux calculs successifs ne fassent pas respirer la bande
    # sans raison.
    inc = incertitude_transfert()
    tirage = np.random.default_rng(20260909).standard_normal(trajectoires.shape[0])[:, None]
    h_membres = _bruiter_transfert(courbe.to_h(trajectoires), tirage, inc)
    quantiles_h = {p: np.percentile(h_membres, p, axis=0) for p in (5, 10, 25, 50, 75, 90, 95)}
    log(f"incertitude de transfert ajoutee : ± {inc['a'] + inc['b'] * float(np.median(h_membres)):.0f} cm "
        f"a la cote actuelle, ± {inc['a'] + inc['b'] * 2.6:.0f} cm pour une crue a 2,60 m")

    # Rochereau est 13,2 km sous Saint-Laurent : l'eau y arrive encore plus tard.
    tau_roch = retard_rochereau(calage)
    c_ro = calage["saint_laurent_vers_rochereau"]
    traj_roch = translater(trajectoires, None, t0, tau_roch, lissage_h=lissage)
    # Meme tirage : c'est la meme relation et le meme episode.
    h_roch = _bruiter_transfert(courbe.to_h(traj_roch), tirage, inc)
    quantiles_h_roch = {p: np.percentile(h_roch, p, axis=0) for p in (5, 10, 25, 50, 75, 90, 95)}
    log(f"propagation Saint-Laurent → Rochereau : {tau_roch:.1f} h "
        f"({c_ro['distance_km']} km a {c_ro['celerite_ms']:.2f} m/s, "
        f"cale sur {c_ro['n_crues']} crues)")
    propagation = {
        "retard_amont_h": round(tau, 1), "retard_rochereau_h": round(tau_roch, 1),
        "h_pic_attendu_m": round(h_pic, 2), "regime_etiage": bool(etiage),
        "lissage_entree_h": int(lissage),
        "amont": {k: c_am[k] for k in ("a", "n", "n_crues", "R2_log", "ecart_type_h", "h_calage")
                  if k in c_am},
        "rochereau": {k: c_ro[k] for k in ("distance_km", "celerite_ms", "n_crues") if k in c_ro},
        "source": calage.get("source"),
        "incertitude_transfert": inc,
    }

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
                  "proba_max": round(float((h_membres.max(axis=1) >= seuils[T]).mean()), 3)}
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
        # Quatre decimales : la cote a Rochereau se deduit de cette hauteur, et
        # un arrondi au millimetre y laisse des marches visibles a l'ecran.
        "H": {str(p): list(np.round(v, 4)) for p, v in quantiles_h.items()},
        # Meme echeance, mais l'eau vue a Rochereau : trois heures de riviere
        # plus tard que celle qui passe devant l'echelle de Saint-Laurent.
        "H_rochereau": {str(p): list(np.round(v, 4)) for p, v in quantiles_h_roch.items()},
        "observe": {"time": [d.isoformat() for d in h_obs.index],
                    "H": list(np.round(h_obs.to_numpy(), 3))},
        "seuils": seuils, "depassements": depassements,
        "validite": {"part_dans_la_gamme": round(part_valide, 3),
                     "hauteur_plancher_m": diag_transfert["hauteur_plancher_m"],
                     "debit_plancher_m3s": q_plancher,
                     "dans_la_gamme": [bool(x) for x in dans_la_gamme]},
        "transfert": diag_transfert,
        "propagation": propagation,
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
