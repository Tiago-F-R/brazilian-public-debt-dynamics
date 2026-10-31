"""Bohn (1998) fiscal reaction function: equation (8)."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .identity import debt_step_real


def bohn_regression(s, d_lag, controls=None, hac_lags=4):
    """Equation (8) by OLS with Newey-West (HAC) standard errors; the coefficient on lagged debt is named 'rho'."""
    parts = [s.rename("s"), d_lag.rename("rho")] + ([controls] if controls is not None else [])
    data = pd.concat(parts, axis=1).dropna()
    X = data.drop(columns="s")
    X = X.loc[:, [c for c in X.columns if c == "rho" or X[c].std() > 0]]   # drop controls constant in the sample
    return sm.OLS(data["s"], sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})


def rolling_bohn(s, d_lag, controls, window, hac_lags, benchmark=None, alpha=0.10):
    """Equation (8) on rolling windows: rho, its (1-alpha) interval and the mean benchmark in each window."""
    data = pd.concat([s.rename("s"), d_lag.rename("d_lag")] + ([controls] if controls is not None else []), axis=1).dropna()
    rows = []
    for end in range(window, len(data) + 1):
        sub = data.iloc[end - window:end]
        ctrl = sub.drop(columns=["s", "d_lag"]) if controls is not None else None
        res = bohn_regression(sub["s"], sub["d_lag"], ctrl, hac_lags)
        lo, hi = res.conf_int(alpha=alpha).loc["rho"]
        rows.append({"end": sub.index[-1], "rho": res.params["rho"], "lo": lo, "hi": hi,
                     "benchmark": benchmark.reindex(sub.index).mean() if benchmark is not None else np.nan})
    return pd.DataFrame(rows).set_index("end")


def simulate_fiscal_rule(d0, r, g, rho, s_bar, n):
    """Debt under the rule s_t = s_bar + rho d_{t-1}: d_t = (lambda - rho) d_{t-1} - s_bar."""
    d = np.empty(n + 1)
    d[0] = d0
    for t in range(1, n + 1):
        d[t] = debt_step_real(d[t - 1], r, g, s_bar + rho * d[t - 1])
    return d


def temporary_2020_share(index):
    """Share of the Apr-Dec 2020 months inside the 12-month window ending in each quarter (temporary-spending proxy)."""
    lo, hi = pd.Period("2020-04", "M"), pd.Period("2020-12", "M")
    shares = []
    for q in index:
        end = q.asfreq("M", "end")
        months = pd.period_range(end - 11, end, freq="M")
        shares.append(float(((months >= lo) & (months <= hi)).sum()) / 9)
    return pd.Series(shares, index=index, name="temporary_2020")


def bohn_controls(Q, gap):
    """Controls Z of equation (8): output gap and the 2020 temporary-spending proxy, on the quarterly index."""
    return pd.concat([gap.rename("gap").reindex(Q.index), temporary_2020_share(Q.index)], axis=1)


PERIMETERS = {
    "DLSP, consolidated primary": ("s_cons", "d_dlsp", "lambda_minus_1_dlsp"),
    "DBGG, general-government primary": ("s_gg", "d_dbgg", None),
}


def bohn_full_sample(Q, Z, hac_lags=4):
    """Full-sample estimates for both perimeters, with and without controls."""
    rows = []
    for name, (s_col, d_col, _) in PERIMETERS.items():
        for label, controls in (("with controls", Z), ("no controls", None)):
            res = bohn_regression(Q[s_col], Q[d_col].shift(4), controls, hac_lags)
            labels = res.model.data.row_labels
            rows.append({"specification": f"{name}, {label}", "rho": res.params["rho"], "HAC s.e.": res.bse["rho"],
                         "t": res.tvalues["rho"], "p-value": res.pvalues["rho"], "R2": res.rsquared, "N": int(res.nobs),
                         "sample": f"{labels[0]}–{labels[-1]}"})
    return pd.DataFrame(rows).set_index("specification")
