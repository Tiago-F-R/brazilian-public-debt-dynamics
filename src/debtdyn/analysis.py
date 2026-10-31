"""High-level calculations on the panels built by `data`, one function per output of the dashboard."""
import numpy as np
import pandas as pd

from .bohn import PERIMETERS, bohn_controls, bohn_full_sample, rolling_bohn
from .cycle import cyclically_adjusted_ratios, output_gap_hp
from .identity import decompose_debt_change, implicit_rate, real_rate
from .simulate import debt_limit, primary_for_target, project_debt, required_primary_endogenous, stabilizing_primary

FULL_PARTS = ["interest", "growth", "inflation", "primary", "sfa"]
PARTIAL_PARTS = ["growth", "inflation", "primary", "interest_plus_sfa"]


# ---------- M1: historical decomposition ----------
def annual_decomposition(A, aggregate, gross_interest=None):
    """Equation (3) on annual data. aggregate: 'dlsp' or 'dbgg'. For the DBGG without a gross interest bill,
    interest and the stock-flow adjustment are reported together as a single residual."""
    if aggregate == "dlsp":
        dec = decompose_debt_change(A["d_dlsp"], A["d_dlsp"].shift(1), A["g"], A["pi"], A["s_cons"], interest=A["interest_cons"])
        return dec.drop(columns="seigniorage").dropna(), FULL_PARTS
    d, d_prev = A["d_dbgg"], A["d_dbgg"].shift(1)
    if gross_interest is not None:
        dec = decompose_debt_change(d, d_prev, A["g"], A["pi"], A["s_gg"], interest=gross_interest.reindex(A.index))
        return dec.drop(columns="seigniorage").dropna(), FULL_PARTS
    dec = pd.DataFrame({
        "growth": -A["g"] * d_prev / (1 + A["g"]),
        "inflation": -A["pi"] * d_prev / ((1 + A["g"]) * (1 + A["pi"])),
        "primary": -A["s_gg"],
        "change": d - d_prev,
    })
    dec["interest_plus_sfa"] = dec["change"] - dec[["growth", "inflation", "primary"]].sum(axis=1)
    return dec.dropna(), PARTIAL_PARTS


def implicit_rates(A, gross_interest=None):
    """Annual implicit rates (equation 3 inverted) and real growth."""
    out = pd.DataFrame({"DLSP implicit nominal rate": A["i_dlsp"], "DLSP implicit real rate": A["r_dlsp"], "Real GDP growth": A["g"]})
    if gross_interest is not None:
        i_dbgg = implicit_rate(gross_interest.reindex(A.index), A["d_dbgg"].shift(1), A["g"], A["pi"])
        out["DBGG implicit real rate"] = real_rate(i_dbgg, A["pi"])
    return out.dropna(how="all")


def compare_with_bcb_factors(dec, factors):
    """Differences (p.p. of GDP) between our DLSP decomposition and the BCB conditioning factors."""
    bcb = factors.set_index("year") / 100
    return pd.DataFrame({
        "interest: ours - BCB": dec["interest"] - bcb["interest"],
        "primary: ours - BCB": dec["primary"] - bcb["primary"],
        "growth+inflation: ours - BCB GDP effect": dec["growth"] + dec["inflation"] - bcb["gdp_effect"],
        "SFA: ours - BCB (FX + other)": dec["sfa"] - bcb["fx_adjustment"] - bcb["other_adjustments"],
    }).dropna(how="all")


def headline_metrics(raw):
    """Latest monthly observation and its change over 12 months, in % of GDP."""
    def latest(series, sign=1.0):
        s = sign * series.dropna()
        return {"value": float(s.iloc[-1]), "change_12m": float(s.iloc[-1] - s.iloc[-13]), "date": s.index[-1]}

    s_gg = -(raw["primary_federal_bcb"] + raw["primary_regional"]).dropna()
    return {
        "dbgg": latest(raw["dbgg"]),
        "dlsp": latest(raw["dlsp"]),
        "primary_cons": latest(raw["primary_consolidated"], -1.0),
        "interest_cons": latest(raw["interest_consolidated"]),
        "primary_gg": {"value": float(s_gg.iloc[-1]), "date": s_gg.index[-1]},
    }


# ---------- M2: scenarios ----------
def scenario_summary(d0, r, g, pi, s, target, horizon, phi=0.0, sigma=0.0):
    """The chosen path, the path with the constant primary for the target, and the key numbers."""
    chosen = project_debt(d0, horizon, r, g, s, pi=pi, sigma=sigma, phi=phi, d_bar=d0)
    s_tgt = primary_for_target(d0, target, horizon, r, g, sigma)
    to_target = project_debt(d0, horizon, r, g, s_tgt, pi=pi, sigma=sigma, phi=phi, d_bar=d0)
    return {"chosen": chosen, "to_target": to_target, "s_star": stabilizing_primary(d0, r, g, sigma), "s_target": s_tgt,
            "d_end": float(chosen["d"].iloc[-1]), "i": (1 + r) * (1 + pi) - 1}


def scenario_set(d0, r, g, pi, horizon, phi_alt=0.05, r_low=0.01, g_high=0.03):
    """Five reference scenarios around the chosen r and g."""
    specs = {
        f"Zero primary (r = {100 * r:.1f}%, g = {100 * g:.1f}%)": dict(r=r, g=g, s=0.0),
        "Stabilising primary": dict(r=r, g=g, s=stabilizing_primary(d0, r, g)),
        "Target: -10 p.p. in the horizon": dict(r=r, g=g, s=primary_for_target(d0, d0 - 0.10, horizon, r, g)),
        f"r < g (r = {100 * r_low:.0f}%, g = {100 * g_high:.0f}%), zero primary": dict(r=r_low, g=g_high, s=0.0),
        f"Debt-dependent r (phi = {phi_alt:.2f}), zero primary": dict(r=r, g=g, s=0.0, phi=phi_alt),
    }
    return {name: project_debt(d0, horizon, pi=pi, d_bar=d0, **kw)["d"] for name, kw in specs.items()}


def endogenous_rate(d0, r_bar, g, s_max, phis=(0.0, 0.02, 0.05), d_grid=None):
    """Required primary s*(d) under a debt-dependent rate, and the debt limit for each phi (equation 6)."""
    d_grid = np.linspace(0.3, 1.6, 131) if d_grid is None else d_grid
    curves = {f"phi = {phi:.2f}": required_primary_endogenous(d_grid, r_bar, g, phi, d0) for phi in phis}
    limits = pd.DataFrame({"phi": list(phis), "debt limit": [debt_limit(s_max, r_bar, g, phi, d0) for phi in phis]}).set_index("phi")
    return d_grid, curves, limits


# ---------- M3: cycle and stance ----------
def annual_output_gap(A, lamb=100):
    return output_gap_hp(A["real_index_4q"], lamb)


def fiscal_stance(rtn, gap):
    """Cyclically adjusted balance, fiscal impulse, stance and cyclicality from central-government revenue and spending."""
    rtn = rtn.set_index("year").sort_index()
    return cyclically_adjusted_ratios(rtn["revenue_pct_gdp"] / 100, rtn["expenditure_pct_gdp"] / 100, gap.reindex(rtn.index))


# ---------- M4: Bohn ----------
def bohn_analysis(Q, perimeter, window=40, hac_lags=4, lamb=1600):
    s_col, d_col, bench_col = PERIMETERS[perimeter]
    gap = output_gap_hp(Q["real_index_4q"], lamb)["gap"]
    Z = bohn_controls(Q, gap)
    roll = rolling_bohn(Q[s_col], Q[d_col].shift(4), Z, window, hac_lags, benchmark=Q[bench_col] if bench_col else None)
    full = bohn_full_sample(Q, Z, hac_lags)
    summary = {"share_above_threshold": float((roll["rho"] > roll["benchmark"]).mean()) if bench_col else np.nan,
               "share_significant": float((roll["lo"] > 0).mean()), "n_windows": len(roll),
               "mean_threshold": float(Q[bench_col].dropna().mean()) if bench_col else np.nan}
    return {"rolling": roll, "full": full, "summary": summary}


# ---------- M6: rules ----------
def annual_federal_primary(Q):
    s = Q["s_fed"].dropna()
    s = s[s.index.quarter == 4]
    s.index = s.index.year
    return s
