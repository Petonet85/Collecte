"""Propagation amont -> aval : le predicteur le plus fiable a courte echeance.

Sur les premieres heures, la crue est deja dans le lit en amont : aucune prevision
de pluie n'est necessaire. On detecte le temps de propagation de chaque station
amont par correlation croisee, puis on ajuste une regression ridge multi-echeance
qui n'utilise que de l'information disponible a l'instant de la prevision.
Le poids accorde a ce modele est fixe par sa competence validee hors echantillon,
et non a dire d'expert.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MAX_LAG_H = 48


@dataclass
class UpstreamLink:
    code: str
    label: str
    lag_h: int
    corr: float
    dist_km: float


@dataclass
class PropagationModel:
    horizons: np.ndarray
    coefs: dict[int, np.ndarray]           # echeance -> vecteur de coefficients
    feature_names: list[str]
    links: list[UpstreamLink]
    skill: dict[int, float] = field(default_factory=dict)   # R2 hors echantillon
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None
    dynamique: float = 0.0          # amplitude log de la fenetre de calage
    exploitable: bool = True

    def to_dict(self) -> dict:
        best = [h for h, v in sorted(self.skill.items()) if v > 0.05]
        return {
            "exploitable": bool(self.exploitable),
            "amplitude_fenetre_log": round(float(self.dynamique), 2),
            "echeance_utile_h": int(best[-1]) if best else 0,
            "stations_amont": [
                {"code": l.code, "libelle": l.label, "temps_propagation_h": l.lag_h,
                 "correlation": round(l.corr, 3), "distance_km": round(l.dist_km, 1)}
                for l in self.links
            ],
            "competence_R2": {int(h): round(v, 3) for h, v in sorted(self.skill.items())},
        }


def detect_lag(target: pd.Series, upstream: pd.Series, max_lag: int = MAX_LAG_H,
               smooth: int = 3) -> tuple[int, float]:
    """Temps de propagation = decalage maximisant la correlation des variations.

    Les series sont lissees sur `smooth` heures avant differenciation : sans cela,
    la quantification du capteur (le cm) domine le signal en basses eaux et la
    correlation ne mesure plus rien.
    """
    df = pd.concat([target.rename("y"), upstream.rename("x")], axis=1).dropna()
    if len(df) < 72:
        return 0, 0.0
    df = df.rolling(smooth, min_periods=1).mean()
    y = np.diff(np.log(np.maximum(df["y"].to_numpy(), 1e-6)))
    x = np.diff(np.log(np.maximum(df["x"].to_numpy(), 1e-6)))
    if y.std() < 1e-9 or x.std() < 1e-9:
        return 0, 0.0
    best = (0, -1.0)
    for lag in range(0, min(max_lag, len(y) // 3)):
        a = y[lag:] if lag else y
        b = x[: len(x) - lag] if lag else x
        n = min(len(a), len(b))
        if n < 48 or a[:n].std() < 1e-9 or b[:n].std() < 1e-9:
            continue
        c = float(np.corrcoef(a[:n], b[:n])[0, 1])
        if c > best[1]:
            best = (lag, c)
    return int(best[0]), float(best[1])


def _design(target: pd.Series, ups: dict[str, pd.Series], links: list[UpstreamLink]
            ) -> tuple[pd.DataFrame, list[str]]:
    """Predicteurs disponibles a t0 : etat et tendance de l'aval et de chaque amont."""
    y = np.log(np.maximum(target, 1e-6))
    cols = {"y": y, "dy1": y.diff(1), "dy3": y.diff(3), "dy6": y.diff(6)}
    names = ["y", "dy1", "dy3", "dy6"]
    for l in links:
        s = np.log(np.maximum(ups[l.code].reindex(target.index).interpolate(limit=6), 1e-6))
        cols[f"u_{l.code}"] = s
        cols[f"du_{l.code}"] = s.diff(3)
        names += [f"u_{l.code}", f"du_{l.code}"]
    return pd.DataFrame(cols), names


def _ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    n_f = X.shape[1]
    A = X.T @ X + alpha * np.eye(n_f)
    return np.linalg.solve(A, X.T @ y)


def fit(target: pd.Series, upstream: dict[str, pd.Series], meta: dict[str, dict],
        horizons: np.ndarray, min_corr: float = 0.40, alpha: float = 3.0,
        min_dynamique: float = 0.25) -> PropagationModel | None:
    """Cale le modele de propagation sur la fenetre temps reel disponible.

    Si la fenetre ne contient aucune variation notable (etiage stable), les
    coefficients ne sont pas identifiables : on le declare au lieu de produire
    une competence illusoire. Le poids accorde au modele sera alors nul.
    """
    target = target.dropna()
    if len(target) < 200:
        return None
    ylog_full = np.log(np.maximum(target.to_numpy(), 1e-6))
    dynamique = float(np.percentile(ylog_full, 98) - np.percentile(ylog_full, 2))
    links: list[UpstreamLink] = []
    for code, ser in upstream.items():
        ser = ser.dropna()
        if len(ser) < 200:
            continue
        lag, corr = detect_lag(target, ser)
        if corr >= min_corr:
            info = meta.get(code, {})
            links.append(UpstreamLink(code, info.get("label", code), lag, corr,
                                      float(info.get("dist_km", np.nan))))
    links.sort(key=lambda l: -l.corr)
    links = links[:5]

    if dynamique < min_dynamique:
        return PropagationModel(np.asarray(horizons), {}, [], links, {}, None, None,
                                dynamique, exploitable=False)

    Xdf, names = _design(target, upstream, links)
    coefs: dict[int, np.ndarray] = {}
    skill: dict[int, float] = {}
    ylog = np.log(np.maximum(target, 1e-6))

    base = Xdf.copy()
    ok_all = base.notna().all(axis=1)
    if ok_all.sum() < 150:
        return None
    mu = base[ok_all].mean().to_numpy()
    sd = base[ok_all].std().replace(0, 1).to_numpy()

    for h in horizons:
        h = int(h)
        yt = ylog.shift(-h)
        m = ok_all & yt.notna()
        if m.sum() < 120:
            continue
        Xs = ((base[m].to_numpy() - mu) / sd)
        Xs = np.c_[Xs, np.ones(len(Xs))]
        yv = yt[m].to_numpy()
        # Validation temporelle : on cale sur les 70 % anciens, on teste sur le reste.
        cut = int(len(yv) * 0.7)
        w = _ridge(Xs[:cut], yv[:cut], alpha)
        pred = Xs[cut:] @ w
        ref = Xs[cut:, names.index("y")] * sd[names.index("y")] + mu[names.index("y")]
        sse = float(((yv[cut:] - pred) ** 2).sum())
        sse_pers = float(((yv[cut:] - ref) ** 2).sum())
        skill[h] = float(1 - sse / sse_pers) if sse_pers > 0 else 0.0
        coefs[h] = _ridge(Xs, yv, alpha)   # coefficients finaux sur toute la fenetre

    if not coefs:
        return None
    return PropagationModel(np.asarray(horizons), coefs, names, links, skill, mu, sd,
                            dynamique, exploitable=True)


def predict(model: PropagationModel, target: pd.Series, upstream: dict[str, pd.Series]
            ) -> pd.Series:
    """Prevision de debit (m3/s) par echeance, a partir du dernier etat connu."""
    if not model.exploitable or not model.coefs:
        return pd.Series(dtype=float)
    Xdf, names = _design(target, upstream, model.links)
    row = Xdf.ffill().iloc[-1].to_numpy()
    if not np.all(np.isfinite(row)):
        return pd.Series(dtype=float)
    xs = np.r_[(row - model.mu) / model.sd, 1.0]
    out = {}
    for h, w in model.coefs.items():
        out[h] = float(np.exp(xs @ w))
    return pd.Series(out).sort_index()
