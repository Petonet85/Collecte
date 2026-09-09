"""Chaine de prevision complete, de la collecte des donnees aux quantiles de crue.

Enchainement :
  1. bassin versant (surface Hub'Eau + relief RGE ALTI) et stations voisines ;
  2. calage long terme du modele GR sur 20 ans de debits journaliers ;
  3. rejeu horaire des 60 derniers jours avec la pluie analysee Meteo-France ;
  4. modulation de l'etat de saturation par le niveau des nappes ;
  5. correction d'erreur autoregressive sur les dernieres heures observees ;
  6. propagation amont (regression ridge multi-echeance) melangee selon sa competence ;
  7. ensemble de pluie multi-modeles + bruit residuel -> quantiles de debit ;
  8. conversion en hauteur par courbe de tarage ancree sur les crues historiques.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import archive
from . import basin as basin_mod
from .model import assim, calib, frequency, gr, propagation, rating
from .model.metrics import kge, kge_sqrt, nse
from .sources import hubeau as hb
from .sources import meteo
from .sources import vigicrues as vc

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CALIB_YEARS = 20
WARMUP_DAYS = 60          # rejeu horaire avant l'instant de prevision
PIEZO_GAIN = 0.15         # sensibilite de l'etat de saturation a l'anomalie de nappe


@dataclass
class Context:
    basin: basin_mod.Basin
    params_day: gr.GRParams
    params_hour: gr.GRParams
    calib: dict
    rating: rating.RatingCurve | None
    upstream_meta: dict
    piezo_code: str | None
    thresholds: dict
    crues_historiques: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def path(self) -> str:
        return os.path.join(DATA_DIR, f"context_{self.basin.code_station}.json")


# --------------------------------------------------------------------------- #
# Construction du contexte (couteux, mis en cache sur disque)
# --------------------------------------------------------------------------- #


def _params_horaires_cales(code: str, defaut: gr.GRParams, log) -> gr.GRParams:
    chemin = os.path.join(os.path.dirname(DATA_DIR), "data", f"calage_horaire_{code}.json")
    if not os.path.exists(chemin):
        chemin = os.path.join(DATA_DIR, f"calage_horaire_{code}.json")
    try:
        with open(chemin, encoding="utf-8") as fh:
            c = json.load(fh)
        p = gr.GRParams.from_array(c["params_hour"])
        v = c.get("validation_apres", {})
        log(f"parametres horaires cales sur les pointes : KGE {v.get('KGE')}, "
            f"biais pic {v.get('biais_pic_pct')} %, etiage x{v.get('sim_sur_obs_etiage')}")
        return p
    except (OSError, ValueError, KeyError):
        return defaut


def build_context(code: str, years: int = CALIB_YEARS, force: bool = False,
                  iters: int = 1200, verbose: bool = True) -> Context:
    def log(msg):
        if verbose:
            print(f"  [contexte] {msg}", flush=True)

    b = basin_mod.delineate(code)
    log(f"{b.name} — BV {b.area_km2:.0f} km², tc≈{b.tc_hours} h, {len(b.cells)} mailles")

    cache = os.path.join(DATA_DIR, f"context_{b.code_station}.json")
    cached = None
    if os.path.exists(cache) and not force:
        with open(cache, encoding="utf-8") as fh:
            cached = json.load(fh)
        age_days = (dt.datetime.now() - dt.datetime.fromtimestamp(os.path.getmtime(cache))).days
        if age_days > 45:
            cached = None

    end = (dt.date.today() - dt.timedelta(days=2)).isoformat()
    start = (dt.date.today() - dt.timedelta(days=365 * years)).isoformat()
    met = meteo.history_daily(b.meteo_points, start, end)
    qj = hb.debits_journaliers(b.code_site, start)
    df = pd.DataFrame({"P": met["P"], "E": met["E"]}).join(qj.rename("Q"), how="left")
    df = df.dropna(subset=["P", "E"])

    if cached:
        p_day = gr.GRParams.from_array(cached["params_day"])
        p_hour = gr.GRParams.from_array(cached["params_hour"])
        # Parametres horaires reellement cales, s'ils existent. Ils priment sur
        # ceux du cache, qui sont DERIVES des journaliers par une regle empirique
        # faute de chronique horaire au moment ou ce cache a ete ecrit. Voir
        # caler_horaire.py : le calage direct ramene le biais d'etiage de x2,2 a
        # x1,6 et le KGE horaire de 0,54 a 0,70 en validation independante.
        p_hour = _params_horaires_cales(b.code_station, p_hour, log)
        scores = cached["calib"]
        log(f"parametres relus du cache : {p_day.to_dict()}")
        if "forcage_operationnel" not in scores:
            # Cache anterieur a l'etape de recalage : on la rattrape sans tout recaler.
            p_day, scores = _refit_operational(b, p_day, scores, log)
            p_hour = calib.daily_to_hourly(p_day)
    else:
        n_ok = int(df["Q"].notna().sum())
        if n_ok < 365 * 3:
            raise RuntimeError(
                f"historique de debit insuffisant ({n_ok} jours) pour caler le modele")
        split = int(len(df) * 0.7)
        tr = df.iloc[:split]
        log(f"calage sur {len(tr)} jours ({tr.index[0].date()} → {tr.index[-1].date()})…")
        p_day, obj, _ = calib.calibrate(tr["P"].values, tr["E"].values, tr["Q"].values,
                                        b.area_km2, 24.0, iters=iters)
        q_sim = gr.mm_to_m3s(gr.run(df["P"].values, df["E"].values, p_day), b.area_km2, 24.0)
        va = np.arange(len(df)) >= split
        scores = {
            "objectif_calage": round(float(obj), 3),
            "validation_KGE": round(kge(df["Q"].values[va], q_sim[va]), 3),
            "validation_NSE": round(nse(df["Q"].values[va], q_sim[va]), 3),
            "validation_KGE_sqrt": round(kge_sqrt(df["Q"].values[va], q_sim[va]), 3),
            "periode_calage": [str(tr.index[0].date()), str(tr.index[-1].date())],
            "periode_validation": [str(df.index[split].date()), str(df.index[-1].date())],
            "sigma_log_long_terme": round(_residual_sigma(df["Q"].values[va], q_sim[va]), 3),
        }
        p_hour = calib.daily_to_hourly(p_day)
        log(f"{p_day.to_dict()} | KGE validation = {scores['validation_KGE']}")

        # Etape 2 : realignement sur le forcage reellement utilise en operationnel.
        p_day, scores = _refit_operational(b, p_day, scores, log)
        p_hour = calib.daily_to_hourly(p_day)

    # --- Courbe de tarage : temps reel + ancrage sur les crues historiques
    h_tr = hb.hourly(hb.observations_tr(b.code_station, "H", 30))
    q_tr = hb.hourly(hb.observations_tr(b.code_site, "Q", 30))
    crues = vc.crues_historiques(b.code_station)
    rc = _fit_rating(h_tr, q_tr, crues)
    if rc:
        log(f"tarage Q={rc.a:.3f}(H−{rc.h0:.2f})^{rc.b:.2f} R²={rc.r2:.3f} "
            f"(n={rc.n}, {len(crues)} crues de reference)")

    # --- Stations amont : liste officielle Vigicrues + candidats geographiques
    up_meta = _upstream_meta(b)
    log(f"{len(up_meta)} stations amont candidates")

    piezos = basin_mod.find_piezos(b)
    b.piezos = piezos
    piezo_code = str(piezos.iloc[0]["code_bss"]) if len(piezos) else None

    peak_factor = frequency.estimate_peak_factor(df["Q"].dropna(), crues)
    thresholds = frequency.gumbel_quantiles(df["Q"].dropna(), peak_factor=peak_factor)
    thresholds["facteur_pointe"] = round(peak_factor, 2)
    log(f"periodes de retour (pointe, facteur {peak_factor:.2f}) : "
        f"Q10={thresholds.get(10)} Q50={thresholds.get(50)} m³/s")

    ctx = Context(b, p_day, p_hour, scores, rc, up_meta, piezo_code, thresholds, crues)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as fh:
        json.dump({"params_day": list(p_day.as_array()),
                   "params_hour": list(p_hour.as_array()),
                   "calib": scores}, fh, indent=1)
    return ctx


def _refit_operational(b, p_day: gr.GRParams, scores: dict, log) -> tuple[gr.GRParams, dict]:
    """Recale CP et X1 sur les runs Meteo-France archives, puis valide hors periode."""
    end = (dt.date.today() - dt.timedelta(days=2)).isoformat()
    op = meteo.operational_history_daily(b.meteo_points, meteo.OPERATIONAL_START, end)
    if op.empty or len(op) < 400:
        log(f"archive operationnelle trop courte ({len(op)} j) : parametres ERA5 conserves")
        scores["forcage_operationnel"] = None
        return p_day, scores
    qj = hb.debits_journaliers(b.code_site, meteo.OPERATIONAL_START)
    d = op.join(qj.rename("Q"), how="left").dropna(subset=["P", "E"])
    n_ok = int(d["Q"].notna().sum())
    if n_ok < 300:
        scores["forcage_operationnel"] = None
        return p_day, scores
    split = int(len(d) * 0.65)
    tr = d.iloc[:split]
    p_op, obj = calib.refit_forcing(tr["P"].values, tr["E"].values, tr["Q"].values,
                                    p_day, b.area_km2)
    q_op = gr.mm_to_m3s(gr.run(d["P"].values, d["E"].values, p_op), b.area_km2, 24.0)
    q_era = gr.mm_to_m3s(gr.run(d["P"].values, d["E"].values, p_day), b.area_km2, 24.0)
    va = np.arange(len(d)) >= split
    kge_op = kge(d["Q"].values[va], q_op[va])
    kge_era = kge(d["Q"].values[va], q_era[va])
    scores["forcage_operationnel"] = {
        "periode": [str(d.index[0].date()), str(d.index[-1].date())],
        "validation_KGE_recale": round(float(kge_op), 3),
        "validation_KGE_parametres_ERA5": round(float(kge_era), 3),
        "CP_avant": round(p_day.cp, 3), "CP_apres": round(p_op.cp, 3),
        "X1_avant": round(p_day.x1, 1), "X1_apres": round(p_op.x1, 1),
    }
    if kge_op <= kge_era:
        log(f"recalage operationnel sans gain (KGE {kge_op:.3f} vs {kge_era:.3f}) : "
            "parametres ERA5 conserves")
        scores["forcage_operationnel"]["retenu"] = False
        return p_day, scores
    log(f"recalage sur forcage Meteo-France : CP {p_day.cp:.2f}→{p_op.cp:.2f}, "
        f"X1 {p_day.x1:.0f}→{p_op.x1:.0f}, KGE {kge_era:.3f}→{kge_op:.3f}")
    scores["forcage_operationnel"]["retenu"] = True
    scores["sigma_log_long_terme"] = round(_residual_sigma(d["Q"].values[va], q_op[va]), 3)
    return p_op, scores


def _residual_sigma(q_obs, q_sim, min_q: float = 1e-3) -> float:
    """Dispersion des residus logarithmiques sur la periode de validation.

    Sert de plancher a l'incertitude temps reel : c'est l'erreur du modele quand
    on ne dispose d'aucune correction, donc le minimum honnete a afficher.
    """
    o, s = np.asarray(q_obs, float), np.asarray(q_sim, float)
    m = np.isfinite(o) & np.isfinite(s) & (o > min_q) & (s > min_q)
    if m.sum() < 30:
        return 0.35
    return float(np.clip(np.std(np.log(o[m]) - np.log(s[m])), 0.05, 1.5))


def _fit_rating(h_tr: pd.Series, q_tr: pd.Series,
                crues: pd.DataFrame) -> rating.RatingCurve | None:
    """Les 30 jours temps reel ne couvrent que les basses eaux : on y ajoute les
    couples (H, Q) des crues de reference, repliques pour peser dans l'ajustement."""
    h, q = h_tr.copy(), q_tr.copy()
    if crues is not None and not crues.empty and {"h", "q"} <= set(crues.columns):
        n_rep = max(len(h.dropna()) // (4 * max(len(crues), 1)), 5)
        extra_idx = pd.date_range("1900-01-01", periods=len(crues) * n_rep, freq="D")
        hv = np.repeat(crues["h"].to_numpy(dtype=float), n_rep)
        qv = np.repeat(crues["q"].to_numpy(dtype=float), n_rep)
        h = pd.concat([h, pd.Series(hv, index=extra_idx)])
        q = pd.concat([q, pd.Series(qv, index=extra_idx)])
    return rating.fit(h, q)


def _upstream_meta(b: basin_mod.Basin) -> dict:
    """Fusionne la composition officielle du bassin (Vigicrues) et le filtre geographique."""
    meta: dict[str, dict] = {}
    off = vc.stations_bassin(b.code_station)
    for _, r in off.iterrows():
        if r["code_station"] == b.code_station:
            continue
        meta[r["code_station"]] = {"label": f"{r['cours_eau']} — {r['libelle']}",
                                   "source": "vigicrues", "dist_km": np.nan}
    geo = basin_mod.find_upstream_candidates(b)
    b.upstream = geo
    for _, r in geo.iterrows():
        code = r["code_station"]
        entry = meta.setdefault(code, {"label": r["libelle_station"], "source": "geo"})
        entry["dist_km"] = float(r["dist_km"])
        entry["surface_bv"] = None if pd.isna(r.get("surface_bv")) else float(r["surface_bv"])
    return meta


# --------------------------------------------------------------------------- #
# Prevision
# --------------------------------------------------------------------------- #


def run(ctx: Context, horizon_h: int = 72, n_noise: int = 3,
        verbose: bool = True, retour_trajectoires: bool = False) -> dict:
    """Prevision sur une station. `retour_trajectoires` ajoute la matrice complete
    des trajectoires, necessaire pour sommer proprement plusieurs bassins :
    additionner des quantiles supposerait leurs rangs parfaitement correles."""
    def log(msg):
        if verbose:
            print(f"  [prevision] {msg}", flush=True)

    b = ctx.basin
    now = pd.Timestamp.utcnow().tz_localize(None).floor("h")

    # --- 1. Observations
    q_obs = archive.update(f"{b.code_site}_Q", hb.hourly(hb.observations_tr(b.code_site, "Q", 30)))
    h_obs = archive.update(f"{b.code_station}_H", hb.hourly(hb.observations_tr(b.code_station, "H", 30)))
    if q_obs.empty and not h_obs.empty and ctx.rating:
        q_obs = pd.Series(ctx.rating.to_q(h_obs.values), index=h_obs.index)
        log("debit temps reel indisponible : reconstitue par la courbe de tarage")
    if q_obs.empty:
        raise RuntimeError("aucune observation temps reel exploitable sur cette station")
    t0 = min(q_obs.index[-1], now)
    log(f"derniere observation : {q_obs.index[-1]} — Q={float(q_obs.iloc[-1]):.1f} m³/s")

    # --- 2. Forcages horaires (analyse + prevision deterministe Meteo-France)
    fc_days = int(np.ceil(horizon_h / 24)) + 1
    met_h = meteo.recent(b.meteo_points, past_days=WARMUP_DAYS, forecast_days=fc_days)
    met_h = met_h.asfreq("h").interpolate(limit=3)
    hist_start = met_h.index[0]

    # --- 3. Etat initial : le modele journalier fournit l'etat des reservoirs
    day_start = (hist_start - pd.Timedelta(days=365 * 3)).date().isoformat()
    met_d = meteo.history_daily(b.meteo_points, day_start,
                                (hist_start - pd.Timedelta(days=1)).date().isoformat())
    _, state_day = gr.run(met_d["P"].values, met_d["E"].values, ctx.params_day,
                          return_state=True)
    state = gr.GRState(state_day.s, state_day.r,
                       np.zeros_like(gr.initial_state(ctx.params_hour).uh1),
                       np.zeros_like(gr.initial_state(ctx.params_hour).uh2))
    log(f"etat initial (rejeu journalier 3 ans) : S={state.s:.0f} mm, R={state.r:.0f} mm")

    # --- 4. Modulation par le niveau des nappes
    piezo = _piezo_series(ctx)
    if len(piezo):
        piezo = archive.update(f"{ctx.piezo_code}_nappe", piezo)
    nappe = _nappe_anomaly(piezo)
    if nappe is not None:
        factor = float(np.clip(1 + PIEZO_GAIN * nappe, 0.75, 1.30))
        state.s = float(np.clip(state.s * factor, 0, 0.98 * ctx.params_hour.x1))
        log(f"nappe : anomalie {nappe:+.2f} σ → reservoir de production ×{factor:.3f}")

    # --- 5. Rejeu horaire jusqu'a t0, puis etat de depart de la prevision
    past = met_h.loc[:t0]
    q_sim_mm, state_t0 = gr.run(past["P"].values, past["E"].values, ctx.params_hour,
                                state=state, return_state=True)
    q_sim_past = pd.Series(gr.mm_to_m3s(q_sim_mm, b.area_km2, 1.0), index=past.index)

    # --- 6. Modele d'erreur sur les 12 derniers jours
    tail = slice(t0 - pd.Timedelta(days=12), t0)
    sigma_floor = 0.55 * float(ctx.calib.get("sigma_log_long_terme", 0.4))
    err = assim.fit(q_obs.loc[tail].reindex(q_sim_past.loc[tail].index).values,
                    q_sim_past.loc[tail].values, sigma_floor=sigma_floor)
    log(f"erreur AR(1) : rho={err.rho:.3f}, biais actuel {100*(np.exp(err.e0)-1):+.0f} %, "
        f"biais persistant {100*(np.exp(err.e_moy)-1):+.0f} %, "
        f"sigma_log={err.sigma:.2f} (plancher long terme {sigma_floor:.2f})")

    # --- 7. Ensemble de pluie -> ensemble de debit
    ens = meteo.ensemble(b.meteo_points, forecast_days=fc_days)
    future_idx = pd.date_range(t0 + pd.Timedelta(hours=1), periods=horizon_h, freq="h")
    det_future = met_h.reindex(future_idx)
    e_future = det_future["E"].fillna(det_future["E"].mean()).fillna(0.05).values

    members: list[np.ndarray] = []
    if not ens.empty:
        ens = ens.reindex(future_idx).ffill().fillna(0.0)
        rain_cols = list(ens.columns)
    else:
        ens = pd.DataFrame({"det": det_future["P"].fillna(0.0)}, index=future_idx)
        rain_cols = ["det"]
        log("ensemble indisponible : repli sur la prevision deterministe")

    for col in rain_cols:
        q_mm = gr.run(ens[col].values, e_future, ctx.params_hour, state=state_t0.copy())
        members.append(gr.mm_to_m3s(q_mm, b.area_km2, 1.0))
    members = np.vstack(members)                            # (n_membres, horizon)
    horizons = np.arange(1, horizon_h + 1)

    # --- 8. Correction d'erreur + bruit residuel
    # La correction depend du debit simule : le biais d'etiage ne doit pas
    # survivre a une crue (voir ErrorModel.poids_regime).
    members = members * err.correction(horizons[None, :], q_modele=members)
    spread = err.spread(horizons)
    rng = np.random.default_rng(42)
    noisy = np.vstack([
        members * np.exp(rng.normal(0, 1, size=(len(members), 1)) * spread[None, :])
        for _ in range(max(n_noise, 1))
    ])
    log(f"{members.shape[0]} membres de pluie × {max(n_noise,1)} tirages = "
        f"{noisy.shape[0]} trajectoires")

    # --- 9. Propagation amont, melangee selon sa competence hors echantillon
    ups, up_used = _upstream_series(ctx)
    prop_model = prop_pred = None
    weights = np.zeros(horizon_h)
    if ups:
        prop_window = q_obs.loc[t0 - pd.Timedelta(days=180):t0]
        prop_model = propagation.fit(prop_window, ups, ctx.upstream_meta, horizons)
    if prop_model is not None and not prop_model.exploitable:
        log("propagation : fenetre temps reel sans dynamique (etiage) — poids nul")
    if prop_model is not None and prop_model.exploitable:
        prop_pred = propagation.predict(prop_model, prop_window, ups)
        for i, h in enumerate(horizons):
            weights[i] = float(np.clip(prop_model.skill.get(int(h), 0.0), 0.0, 0.85))
        med = np.median(noisy, axis=0)
        prop_vals = prop_pred.reindex(horizons).interpolate().bfill().ffill().values
        ok = np.isfinite(prop_vals) & (prop_vals > 0) & (med > 0)
        delta = np.zeros(horizon_h)
        delta[ok] = weights[ok] * (np.log(prop_vals[ok]) - np.log(med[ok]))
        noisy = noisy * np.exp(delta)[None, :]
        log(f"propagation : {len(prop_model.links)} stations, poids {weights.min():.2f}"
            f"–{weights.max():.2f} (R² hors echantillon)")

    # --- 10. Quantiles et conversion en hauteur
    qs = {p: np.percentile(noisy, p, axis=0) for p in (5, 10, 25, 50, 75, 90, 95)}
    det_q = np.median(members, axis=0)
    result = _assemble(ctx, t0, future_idx, qs, det_q, noisy, q_obs, h_obs, q_sim_past,
                       met_h, ens, err, prop_model, weights, piezo, nappe, up_used)
    log(f"Q median a +{horizon_h} h : {qs[50][-1]:.1f} m³/s "
        f"(intervalle 80 % : {qs[10][-1]:.1f}–{qs[90][-1]:.1f})")
    if retour_trajectoires:
        result["_trajectoires"] = noisy
        result["_horodatage"] = future_idx
        result["_q_observe"] = q_obs
    return result


def _piezo_series(ctx: Context) -> pd.Series:
    if not ctx.piezo_code:
        return pd.Series(dtype=float)
    try:
        return hb.piezo_chronique(ctx.piezo_code, days=1800)
    except Exception:  # noqa: BLE001
        return pd.Series(dtype=float)


def _nappe_anomaly(piezo: pd.Series) -> float | None:
    """Anomalie standardisee du niveau de nappe, calculee sur le mois calendaire."""
    s = piezo.dropna()
    if len(s) < 365:
        return None
    month = s.index.month
    cur_month = s.index[-1].month
    ref = s[month == cur_month]
    if len(ref) < 30 or ref.std() == 0:
        ref = s
    return float(np.clip((s.iloc[-1] - ref.mean()) / max(ref.std(), 1e-6), -3, 3))


def _upstream_series(ctx: Context) -> tuple[dict[str, pd.Series], list[str]]:
    """Recupere les chroniques temps reel des stations amont exploitables."""
    ups: dict[str, pd.Series] = {}
    used: list[str] = []
    for code in list(ctx.upstream_meta)[:14]:
        try:
            ser = hb.hourly(hb.observations_tr(code, "H", 30))
            if ser.dropna().size < 200:
                ser = hb.hourly(hb.observations_tr(code[:8], "Q", 30))
        except Exception:  # noqa: BLE001
            continue
        ser = archive.update(f"{code}_TR", ser.dropna())
        if len(ser) >= 200 and ser.std() > 1e-6:
            ups[code] = ser
            used.append(code)
    return ups, used


def _sig(x, chiffres: int = 4):
    """Arrondi a un nombre de chiffres significatifs plutot qu'a des decimales.

    Deux decimales conviennent a une crue de 200 m3/s et detruisent l'etiage :
    a 0,06 m3/s le pas d'arrondi vaut 15 % du debit, ce qui se lit a l'ecran
    comme un escalier que le modele n'a jamais produit. Un nombre fixe de
    chiffres significatifs garde la meme finesse relative a toutes les eaux.
    """
    out = []
    for v in np.asarray(x, dtype=float).ravel():
        out.append(float(v) if not np.isfinite(v) or v == 0
                   else float(f"%.{chiffres}g" % v))
    return out


def _assemble(ctx, t0, idx, qs, det_q, noisy, q_obs, h_obs, q_sim_past, met_h, ens,
              err, prop_model, weights, piezo, nappe, up_used) -> dict:
    b = ctx.basin
    rc = ctx.rating

    def to_h(arr):
        return list(np.round(rc.to_h(arr), 3)) if rc else None

    obs_win = q_obs.loc[t0 - pd.Timedelta(days=15):]
    sim_win = q_sim_past.loc[t0 - pd.Timedelta(days=15):]
    h_win = h_obs.loc[obs_win.index[0]:] if len(h_obs) else pd.Series(dtype=float)
    rain_past = met_h.loc[t0 - pd.Timedelta(days=15):t0, "P"]
    # ens est (temps x membres) : les quantiles se prennent sur les membres, donc axis=1.
    rain_q = {p: list(np.round(np.percentile(ens.to_numpy(), p, axis=1), 2))
              for p in (10, 50, 90)} if len(ens.columns) > 2 else \
             {50: list(np.round(ens.iloc[:, 0].to_numpy(), 2))}

    thr = {k: v for k, v in ctx.thresholds.items() if isinstance(k, int)}
    exceed = {}
    for label, val in thr.items():
        exceed[f"T{label}"] = {
            "debit": val,
            "hauteur": round(float(rc.to_h(val)), 2) if rc else None,
            "proba_max": round(float((noisy.max(axis=1) >= val).mean()), 3),
        }

    subset = noisy[np.random.default_rng(7).choice(len(noisy), min(60, len(noisy)),
                                                   replace=False)]
    return {
        "meta": {
            "station": b.name, "code_station": b.code_station, "code_site": b.code_site,
            "cours_eau": b.river, "lon": b.lon, "lat": b.lat,
            "surface_bv_km2": b.area_km2, "altitude_m": b.altitude_m,
            "relief": b.relief, "tc_hours": b.tc_hours,
            "emprise": b.cells[["lon", "lat", "z"]].round(4).to_dict("list")
            if not b.cells.empty else {},
            "points_meteo": b.meteo_points,
            "date_prevision": t0.isoformat() + "Z",
            "genere_le": pd.Timestamp.utcnow().tz_localize(None).isoformat() + "Z",
        },
        "observe": {
            "time": [d.isoformat() for d in obs_win.index],
            "Q": _sig(obs_win.to_numpy()),
            "H": list(np.round(h_win.to_numpy(), 3)),
            "H_time": [d.isoformat() for d in h_win.index],
            "Q_simule": _sig(sim_win.reindex(obs_win.index).to_numpy()),
        },
        "prevision": {
            "time": [d.isoformat() for d in idx],
            "Q": {str(p): _sig(v) for p, v in qs.items()},
            "H": {str(p): to_h(v) for p, v in qs.items()} if rc else {},
            "Q_deterministe": _sig(det_q),
            "membres": [_sig(m) for m in subset],
            "poids_propagation": list(np.round(weights, 3)),
        },
        "pluie": {
            "passe_time": [d.isoformat() for d in rain_past.index],
            "passe": list(np.round(rain_past.to_numpy(), 2)),
            "prev_time": [d.isoformat() for d in ens.index],
            "prev": {str(k): v for k, v in rain_q.items()},
            "cumul_median_mm": round(float(np.median(ens.to_numpy().sum(axis=0))), 1),
            "cumul_p90_mm": round(float(np.percentile(ens.to_numpy().sum(axis=0), 90)), 1),
            "cumul_passe_7j_mm": round(float(met_h.loc[t0 - pd.Timedelta(days=7):t0, "P"].sum()), 1),
        },
        "nappe": {
            "code_bss": ctx.piezo_code,
            "anomalie_sigma": None if nappe is None else round(nappe, 2),
            "time": [d.isoformat() for d in piezo.tail(730).index] if len(piezo) else [],
            "niveau_ngf": list(np.round(piezo.tail(730).to_numpy(), 3)) if len(piezo) else [],
        },
        "seuils": {"periodes_retour": ctx.thresholds, "depassements": exceed,
                   "crues_historiques": ctx.crues_historiques.to_dict("records")
                   if not ctx.crues_historiques.empty else []},
        "diagnostic": {
            "calage": ctx.calib,
            "parametres_journaliers": ctx.params_day.to_dict(),
            "parametres_horaires": ctx.params_hour.to_dict(),
            "courbe_tarage": ctx.rating.to_dict() if ctx.rating else None,
            "modele_erreur": err.to_dict(),
            "propagation": prop_model.to_dict() if prop_model else None,
            "stations_amont_utilisees": up_used,
            "n_trajectoires": int(noisy.shape[0]),
        },
    }
