"""Rejeu historique : mesurer la competence reelle de la chaine de prevision.

Une prevision ne vaut que ce que valent ses scores hors echantillon. Ce module
rejoue la chaine au pas journalier sur une periode jamais vue au calage, en se
placant a chaque date comme si c'etait aujourd'hui : etat du modele issu du seul
passe, correction d'erreur issue des seuls residus deja observes, puis integration
sur l'echeance demandee.

Deux forcages de pluie future sont testes, et l'ecart entre les deux mesure la
part d'incertitude qui vient de la meteo et non de l'hydrologie :
  * `parfaite` : la pluie reellement tombee (borne haute de competence) ;
  * `nulle`    : aucune pluie a venir (recession pure, borne basse) ;
  * `prevue`   : la pluie telle qu'elle etait annoncee a l'epoque, echeance par
                 echeance (runs archives Meteo-France). C'est le scenario reel.

La reference a battre est la persistance (Q reste a sa valeur du jour) : c'est
le concurrent qu'une prevision doit depasser pour justifier son existence.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .model import assim, gr
from .model.metrics import contingency, kge, nse


def hindcast(P: np.ndarray, E: np.ndarray, Qobs: np.ndarray, index: pd.DatetimeIndex,
             params: gr.GRParams, area_km2: float, leads=(1, 2, 3, 5),
             start_frac: float = 0.7, assimilation: bool = True,
             residual_window: int = 30, rain_scenario: str = "parfaite",
             rain_forecast: dict[int, pd.Series] | None = None) -> pd.DataFrame:
    """Rejoue la prevision a chaque date d'origine. Renvoie un tableau long.

    `rain_forecast` : {echeance_j -> serie de pluie prevue}, requis pour le
    scenario "prevue". Les valeurs manquantes retombent sur la pluie observee.
    """
    q_mm, trace = gr.run(P, E, params, return_trace=True)
    q_sim = gr.mm_to_m3s(q_mm, area_km2, 24.0)
    n = len(P)
    i_start = int(n * start_frac)
    max_lead = max(leads)
    rows = []

    fc = None
    if rain_scenario == "prevue":
        if not rain_forecast:
            raise ValueError("scenario 'prevue' : fournir rain_forecast")
        fc = np.full((max_lead, n), np.nan)
        for lead in range(1, max_lead + 1):
            ser = rain_forecast.get(lead)
            if ser is not None and len(ser):
                fc[lead - 1] = ser.reindex(index).to_numpy(dtype=float)
        # a defaut de run archive, on retombe sur la pluie observee
        fc = np.where(np.isfinite(fc), fc, P[None, :])

    for i in range(i_start, n - max_lead):
        state = trace[i].copy()
        if rain_scenario == "parfaite":
            p_fut = P[i + 1: i + 1 + max_lead]
        elif rain_scenario == "prevue":
            p_fut = np.array([fc[k, i + 1 + k] for k in range(max_lead)])
        else:
            p_fut = np.zeros(max_lead)
        e_fut = E[i + 1: i + 1 + max_lead]
        q_fut = gr.mm_to_m3s(gr.run(p_fut, e_fut, params, state=state), area_km2, 24.0)

        if assimilation:
            lo = max(i - residual_window + 1, 0)
            err = assim.fit(Qobs[lo: i + 1], q_sim[lo: i + 1])
            q_fut = q_fut * err.correction(np.arange(1, max_lead + 1))

        for lead in leads:
            j = i + lead
            if j >= n:
                continue
            rows.append({
                "origine": index[i], "echeance_j": lead, "date_cible": index[j],
                "obs": Qobs[j], "prev": q_fut[lead - 1],
                "persistance": Qobs[i], "brut": q_sim[j],
            })
    return pd.DataFrame(rows)


def score(hc: pd.DataFrame, seuil: float | None = None) -> pd.DataFrame:
    """Scores par echeance, compares a la persistance et au modele non corrige."""
    out = []
    for lead, g in hc.groupby("echeance_j"):
        g = g.dropna(subset=["obs", "prev", "persistance"])
        if g.empty:
            continue
        row = {
            "echeance_j": int(lead), "n": len(g),
            "KGE": round(kge(g["obs"], g["prev"]), 3),
            "NSE": round(nse(g["obs"], g["prev"]), 3),
            "KGE_persistance": round(kge(g["obs"], g["persistance"]), 3),
            "NSE_persistance": round(nse(g["obs"], g["persistance"]), 3),
            "KGE_sans_assimilation": round(kge(g["obs"], g["brut"]), 3),
            "biais_pct": round(100 * (g["prev"].mean() / g["obs"].mean() - 1), 1),
            "MAE": round(float((g["prev"] - g["obs"]).abs().mean()), 2),
            "MAE_persistance": round(float((g["persistance"] - g["obs"]).abs().mean()), 2),
        }
        row["gain_sur_persistance_pct"] = round(
            100 * (1 - row["MAE"] / row["MAE_persistance"]), 1) if row["MAE_persistance"] else None
        if seuil:
            c = contingency(g["obs"], g["prev"], seuil)
            row.update({"POD": c["POD"], "FAR": c["FAR"], "CSI": c["CSI"],
                        "n_depassements": c["hits"] + c["misses"]})
        out.append(row)
    return pd.DataFrame(out).sort_values("echeance_j").reset_index(drop=True)


def evenements(hc: pd.DataFrame, seuil: float, lead: int = 2) -> pd.DataFrame:
    """Detail des episodes ou l'observation depasse le seuil, a une echeance donnee."""
    g = hc[hc["echeance_j"] == lead].dropna(subset=["obs", "prev"])
    ev = g[g["obs"] >= seuil].copy()
    if ev.empty:
        return ev
    ev["erreur_pct"] = (100 * (ev["prev"] / ev["obs"] - 1)).round(1)
    ev["groupe"] = (ev["date_cible"].diff() > pd.Timedelta(days=3)).cumsum()
    agg = ev.groupby("groupe").agg(
        debut=("date_cible", "min"), pic_observe=("obs", "max"),
        pic_prevu=("prev", "max"), erreur_pic_pct=("erreur_pct", "median"))
    agg["erreur_pic_pct"] = (100 * (agg["pic_prevu"] / agg["pic_observe"] - 1)).round(1)
    num = agg.select_dtypes("number").columns
    agg[num] = agg[num].round(1)
    return agg.reset_index(drop=True)


def validate(ctx, leads=(1, 2, 3, 5), start_frac: float = 0.7,
             avec_pluie_prevue: bool = True, verbose: bool = True) -> dict:
    """Produit le bloc de competence validee affiche dans le tableau de bord.

    Trois colonnes de lecture, du plus optimiste au plus realiste :
      * pluie parfaite  -> ce que vaudrait la chaine si la meteo etait exacte ;
      * pluie prevue    -> ce qu'elle vaut reellement, avec les runs de l'epoque ;
      * persistance     -> la reference a battre.
    """
    import datetime as _dt

    from .sources import hubeau as _hb
    from .sources import meteo as _meteo

    b = ctx.basin
    end = (_dt.date.today() - _dt.timedelta(days=2)).isoformat()
    start = (_dt.date.today() - _dt.timedelta(days=365 * 20)).isoformat()
    met = _meteo.history_daily(b.meteo_points, start, end)
    qj = _hb.debits_journaliers(b.code_site, start)
    df = pd.DataFrame({"P": met["P"], "E": met["E"]}).join(qj.rename("Q"), how="left")
    df = df.dropna(subset=["P", "E"])

    seuil = ctx.thresholds.get(2)
    pf = ctx.thresholds.get("facteur_pointe", 1.0) or 1.0
    seuil_j = seuil / pf if seuil else None

    common = dict(index=df.index, params=ctx.params_day, area_km2=b.area_km2,
                  leads=leads, start_frac=start_frac)
    hc_parfait = hindcast(df["P"].values, df["E"].values, df["Q"].values, **common)
    out = {"periode": [str(hc_parfait["origine"].min().date()),
                       str(hc_parfait["origine"].max().date())],
           "n_previsions": int(len(hc_parfait)),
           "seuil_Q2_journalier": None if seuil_j is None else round(seuil_j, 0),
           "pluie_parfaite": score(hc_parfait, seuil_j).to_dict("records")}
    if verbose:
        print("  [validation] pluie parfaite :",
              {r["echeance_j"]: r["KGE"] for r in out["pluie_parfaite"]})

    if avec_pluie_prevue:
        rain_fc = {}
        for lead in leads:
            ser = _meteo.past_forecast_daily(b.meteo_points[:6], _meteo.OPERATIONAL_START,
                                             end, lead)
            if len(ser):
                rain_fc[lead] = ser
        if rain_fc:
            first = min(s.index.min() for s in rain_fc.values())
            frac = float(np.clip((df.index < first).mean(), 0.05, 0.95))
            hc_reel = hindcast(df["P"].values, df["E"].values, df["Q"].values,
                               index=df.index, params=ctx.params_day, area_km2=b.area_km2,
                               leads=leads, start_frac=frac, rain_scenario="prevue",
                               rain_forecast=rain_fc)
            out["pluie_prevue"] = score(hc_reel, seuil_j).to_dict("records")
            out["periode_pluie_prevue"] = [str(hc_reel["origine"].min().date()),
                                           str(hc_reel["origine"].max().date())]
            ev = evenements(hc_reel, seuil_j, lead=min(2, max(leads))) if seuil_j else pd.DataFrame()
            out["episodes"] = [] if ev.empty else json.loads(
                ev.assign(debut=ev["debut"].astype(str)).to_json(orient="records"))
            if verbose:
                print("  [validation] pluie prevue   :",
                      {r["echeance_j"]: r["KGE"] for r in out["pluie_prevue"]})
    return out
