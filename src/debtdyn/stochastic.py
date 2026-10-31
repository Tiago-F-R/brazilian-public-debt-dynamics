"""Stochastic debt sustainability analysis (OBR method): equation (10)."""
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from .identity import debt_step_nominal


def fit_var(X, maxlags=4, ic="aic"):
    """VAR(p) with a constant; p chosen by the information criterion, at least 1."""
    model = VAR(X.reset_index(drop=True))
    p = max(1, int(model.select_order(maxlags).selected_orders[ic]))
    return model.fit(p)


def simulate_var_bootstrap(res, history, horizon, n_paths, seed):
    """OBR steps 1-2: draws historical residual vectors with replacement and feeds them recursively into the VAR."""
    rng = np.random.default_rng(seed)
    p, k = res.k_ar, res.neqs
    A, c = res.coefs, res.intercept
    resid = np.asarray(res.resid)
    draws = rng.integers(0, resid.shape[0], size=(n_paths, horizon))
    state = np.repeat(np.asarray(history, float)[-p:][None, :, :], n_paths, axis=0)   # oldest lag first
    sims = np.empty((n_paths, horizon, k))
    for h in range(horizon):
        x = c + sum(state[:, -1 - j, :] @ A[j].T for j in range(p)) + resid[draws[:, h]]
        sims[:, h, :] = x
        state = np.concatenate([state[:, 1:, :], x[:, None, :]], axis=1)
    return sims, draws


def align_to_baseline(sims, baseline):
    """OBR step 4: shifts each variable so that its median across paths equals the baseline at every horizon."""
    return sims - np.median(sims, axis=0, keepdims=True) + np.asarray(baseline, float)[None, :, :]


def debt_paths_yoy(g, pi, i, s, sfa, d_hist):
    """OBR step 3 with year-on-year quarterly data: d_t = (1+i_t)/((1+g_t)(1+pi_t)) d_{t-4} - s_t + sfa_t."""
    n_paths, H = g.shape
    d = np.empty((n_paths, H + 4))
    d[:, :4] = np.asarray(d_hist, float)[-4:]
    for h in range(H):
        d[:, h + 4] = debt_step_nominal(d[:, h], i[:, h], g[:, h], pi[:, h], s[:, h], sfa=sfa[:, h])
    return d[:, 4:]


def fan_percentiles(paths, index, qs=(10, 20, 30, 40, 50, 60, 70, 80, 90)):
    return pd.DataFrame(np.percentile(paths, qs, axis=0).T, index=index, columns=[f"p{q}" for q in qs])


VAR_COLS = ["g", "pi", "i_dlsp", "s_cons"]


def dlsp_fan(Q, horizon=20, n_paths=5000, seed=2026, sfa_mode="demeaned", maxlags=4, baseline_overrides=None):
    """Runs the four OBR steps on the DLSP and returns the fan, the simulated paths and a description of the VAR."""
    hist = Q[VAR_COLS + ["sfa_dlsp", "d_dlsp"]].dropna()
    res = fit_var(hist[VAR_COLS], maxlags)
    p = res.k_ar
    sims, draws = simulate_var_bootstrap(res, hist[VAR_COLS].values, horizon, n_paths, seed)
    baseline = res.forecast(hist[VAR_COLS].values[-p:], steps=horizon)
    for name, value in (baseline_overrides or {}).items():
        baseline[:, VAR_COLS.index(name)] = value
    aligned = align_to_baseline(sims, baseline)
    sfa_hist = hist["sfa_dlsp"].values[p:]                    # same rows as the VAR residuals
    sfa_hist = {"demeaned": sfa_hist - sfa_hist.mean(), "raw": sfa_hist, "zero": 0 * sfa_hist}[sfa_mode]
    cols = [aligned[:, :, VAR_COLS.index(c)] for c in VAR_COLS]
    paths = debt_paths_yoy(*cols, sfa_hist[draws], hist["d_dlsp"].values)
    base_path = debt_paths_yoy(*[baseline[None, :, VAR_COLS.index(c)] for c in VAR_COLS], np.zeros((1, horizon)),
                               hist["d_dlsp"].values)[0]
    future = pd.period_range(hist.index[-1] + 1, periods=horizon, freq="Q")
    fan = fan_percentiles(paths, future)
    fan["baseline"] = base_path
    info = {"lags": p, "stable": bool(res.is_stable()), "n_obs": len(hist), "start": str(hist.index[0]), "end": str(hist.index[-1]),
            "last_d": float(hist["d_dlsp"].iloc[-1])}
    return {"fan": fan, "paths": paths, "info": info}
