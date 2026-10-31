"""Tests against the course: Lecture 9 (slides 20, 22, 26), the drills of the course summary, the worked example and
sign table of Lecture 3, and the structural properties derived from equations (2)-(10).

Used by tests/test_course.py (pytest) and by the dashboard's "Data & checks" tab.
"""
import numpy as np
import pandas as pd

from .bohn import bohn_regression, simulate_fiscal_rule
from .cycle import cyclically_adjusted_levels
from .identity import debt_step_nominal, debt_step_real, decompose_debt_change, implicit_rate
from .simulate import (debt_limit, present_value_primaries, primary_for_target, project_debt, required_primary_endogenous,
                       snowball, stabilizing_grid, stabilizing_primary, steady_states)
from .stochastic import fan_percentiles, fit_var, simulate_var_bootstrap

IFI_DEC2025 = pd.DataFrame(   # Lecture 9, slide 22: IFI, Table 13 (Dec/2025), DBGG at 79.0% of GDP, % of GDP
    [[0.0, 0.4, 1.2, 2.4, 2.8, 3.1, 3.7],
     [-0.4, 0.0, 0.8, 2.0, 2.3, 2.7, 3.3],
     [-0.8, -0.4, 0.4, 1.6, 1.9, 2.3, 2.9],
     [-1.3, -1.0, -0.2, 1.0, 1.4, 1.7, 2.3],
     [-1.5, -1.2, -0.4, 0.8, 1.2, 1.5, 2.1],
     [-1.9, -1.5, -0.8, 0.4, 0.8, 1.2, 1.7],
     [-2.3, -1.9, -1.1, 0.0, 0.4, 0.8, 1.3]],
    index=pd.Index([0.005, 0.010, 0.015, 0.022, 0.025, 0.030, 0.035], name="g"),
    columns=pd.Index([0.005, 0.010, 0.020, 0.035, 0.040, 0.045, 0.052], name="r"),
)


def run_course_tests(seed=2026):
    """Acceptance tests of the plan: numbers of Lecture 9 and the course drills, plus the structural properties of section 2."""
    rows = []

    def check(name, value, expected, tol, source):
        rows.append({"test": name, "value": value, "expected": expected, "tolerance": tol,
                     "pass": bool(abs(value - expected) <= tol), "source": source})

    def holds(name, condition, source):
        rows.append({"test": name, "value": np.nan, "expected": np.nan, "tolerance": np.nan,
                     "pass": bool(condition), "source": source})

    # Equation (4): stabilising primary
    check("s*: d=79%, r=5.2%, g=2.2% (IFI highlighted cell)", stabilizing_primary(0.79, 0.052, 0.022), 0.023, 5e-4, "L9 slide 22")
    slide20 = [(0.040, 0.025, 0.60, 0.0029, 0.0059), (0.030, 0.025, 0.60, 0.0029, 0.0001),
               (0.040, 0.035, 0.60, 0.0033, -0.0004), (0.040, 0.025, 0.50, 0.0029, 0.0044),
               (0.040, 0.025, 0.50, 0.0033, 0.0040), (0.040, 0.025, 0.50, 0.0057, 0.0016)]
    for k, (r, g, d, sig, x) in enumerate(slide20, 1):
        check(f"s* with seigniorage, case {k}", stabilizing_primary(d, r, g, sig), x, 1e-4, "L9 slide 20")
    check("s* with r<g: d=79%, r=1%, g=3%", stabilizing_primary(0.79, 0.01, 0.03), -0.0153, 1e-4, "L9 slide 31; drill D4")
    check("s*: d=90%, r=3%, g=1%", stabilizing_primary(0.90, 0.03, 0.01), 0.0178, 1e-4, "Course summary, drill D5")

    # The whole IFI table
    grid = 100 * stabilizing_grid(0.79, IFI_DEC2025.columns, IFI_DEC2025.index)
    check("IFI table: max |formula - published| (p.p.)", float((grid - IFI_DEC2025).abs().max().max()), 0.0, 0.1, "L9 slide 22")

    # Equation (5): primary for a target
    slide26 = [(0.04, 0.025, 0.60, 0.40, 10, 0.0029, 0.0246), (0.04, 0.025, 0.60, 0.50, 10, 0.0029, 0.0153),
               (0.04, 0.025, 0.60, 0.40, 20, 0.0029, 0.0146), (0.03, 0.025, 0.50, 0.40, 10, 0.0029, 0.0094)]
    for k, (r, g, d0, dt, n, sig, x) in enumerate(slide26, 1):
        check(f"target primary, case {k}", primary_for_target(d0, dt, n, r, g, sig), x, 1e-4, "L9 slide 26")
    check("target primary: 79% -> 70% in 10 years, r=5.2%, g=2.2%", primary_for_target(0.79, 0.70, 10, 0.052, 0.022), 0.031, 5e-4, "Course summary, drill D3")
    check("target = initial ratio collapses to s*", primary_for_target(0.79, 0.79, 10, 0.052, 0.022), stabilizing_primary(0.79, 0.052, 0.022), 1e-12, "Equation (5)")
    check("target primary when lambda = 1", primary_for_target(0.60, 0.50, 10, 0.02, 0.02), 0.01, 1e-12, "Equation (5), limit")
    path = project_debt(0.60, 10, 0.04, 0.025, primary_for_target(0.60, 0.40, 10, 0.04, 0.025))
    check("projection with the target primary ends at the target", path["d"].iloc[-1], 0.40, 1e-12, "Equations (2) and (5)")

    # Equation (2): projections and the nominal/real equivalence
    flat = project_debt(0.79, 30, 0.052, 0.022, stabilizing_primary(0.79, 0.052, 0.022), pi=0.04)
    check("projection with s* keeps the ratio constant", float((flat["d"] - 0.79).abs().max()), 0.0, 1e-12, "Equation (4)")
    check("nominal form = real form when 1+i = (1+r)(1+pi)",
          debt_step_nominal(0.79, 1.052 * 1.04 - 1, 0.022, 0.04, 0.01), debt_step_real(0.79, 0.052, 0.022, 0.01), 1e-14, "Equation (2)")

    # Equation (3): the decomposition closes and recovers a known stock-flow adjustment
    rng = np.random.default_rng(seed)
    n = 200
    d_prev, i_, g_, pi_, s_, sfa_ = (rng.uniform(0.4, 1.0, n), rng.uniform(0.0, 0.15, n), rng.uniform(-0.04, 0.06, n),
                                    rng.uniform(0.0, 0.10, n), rng.uniform(-0.05, 0.04, n), rng.normal(0, 0.02, n))
    d_new = debt_step_nominal(d_prev, i_, g_, pi_, s_, sfa=sfa_)
    idx = pd.RangeIndex(n)
    dec = decompose_debt_change(pd.Series(d_new, idx), pd.Series(d_prev, idx), pd.Series(g_, idx), pd.Series(pi_, idx),
                                pd.Series(s_, idx), i=pd.Series(i_, idx))
    check("decomposition recovers the stock-flow adjustment", float(np.max(np.abs(dec["sfa"] - sfa_))), 0.0, 1e-12, "Equation (3)")
    parts = dec[["interest", "growth", "inflation", "primary", "seigniorage", "sfa"]].sum(axis=1)
    check("contributions sum to the change in the ratio", float((parts - dec["change"]).abs().max()), 0.0, 1e-12, "Equation (3)")
    interest_ratio = i_ * d_prev / ((1 + g_) * (1 + pi_))
    check("implicit rate recovered from interest / GDP", float(np.max(np.abs(implicit_rate(interest_ratio, d_prev, g_, pi_) - i_))), 0.0, 1e-12, "Equation (3)")

    # Equation (7): intertemporal constraint
    s_star = stabilizing_primary(0.79, 0.052, 0.022)
    check("PV of constant s* equals the initial ratio", present_value_primaries(np.full(5000, s_star), 0.052, 0.022), 0.79, 1e-9, "Equation (7)")

    # Equation (6): debt-dependent rate
    lin = project_debt(0.79, 20, 0.052, 0.022, 0.01)
    endo0 = project_debt(0.79, 20, 0.052, 0.022, 0.01, phi=0.0)
    check("phi = 0 reproduces the linear model", float((lin["d"] - endo0["d"]).abs().max()), 0.0, 1e-15, "Equation (6)")
    ss = steady_states(0.02, 0.03, 0.02, 0.05, 0.79)
    check("a steady state solves s*(d) = s", float(required_primary_endogenous(ss["d"].max(), 0.03, 0.02, 0.05, 0.79)), 0.02, 1e-12, "Equation (6)")
    d_lim = debt_limit(0.02, 0.03, 0.02, 0.05, 0.79)
    above = project_debt(d_lim + 0.01, 60, 0.03, 0.02, 0.02, phi=0.05, d_bar=0.79)["d"].iloc[-1]
    below = project_debt(d_lim - 0.01, 60, 0.03, 0.02, 0.02, phi=0.05, d_bar=0.79)["d"].iloc[-1]
    holds("1 p.p. above the debt limit the ratio runs away; 1 p.p. below it falls", above > d_lim + 0.10 and below < d_lim - 0.10, "Equation (6)")

    # Equation (8): Bohn stability condition and recovery of rho
    lam_1 = snowball(0.052, 0.022) - 1
    conv = simulate_fiscal_rule(0.79, 0.052, 0.022, lam_1 + 0.01, 0.0, 300)[-1]
    div = simulate_fiscal_rule(0.79, 0.052, 0.022, lam_1 - 0.01, 0.0, 300)[-1]
    holds("rho > (r-g)/(1+g) converges; rho < (r-g)/(1+g) diverges", conv < 0.05 and div > 10 * 0.79, "Equation (8)")
    T = 400
    d_sim = pd.Series(rng.uniform(0.3, 0.9, T))
    gap_sim = pd.Series(rng.normal(0, 0.02, T), name="gap")
    s_sim = 0.01 + 0.05 * d_sim + 0.3 * gap_sim + pd.Series(rng.normal(0, 0.005, T))
    res = bohn_regression(s_sim, d_sim, gap_sim.to_frame(), hac_lags=4)
    holds("Bohn OLS recovers rho = 0.05 within 3 s.e.", abs(res.params["rho"] - 0.05) < 3 * res.bse["rho"], "Equation (8)")

    # Equation (9): worked example of Lecture 3 (course summary) and the sign table of slide 29
    ex = cyclically_adjusted_levels(R=230, G=240, Y=1000, YP=1020)
    check("CAPB, worked example (Y=1000, YP=1020, R=230, G=240)", ex["CAPB"], -5.4, 1e-9, "L3, worked example")
    check("CPB, worked example", ex["CPB"], -4.6, 1e-9, "L3, worked example")
    check("CPB/YP, first-order approximation", 0.23 * (1000 / 1020 - 1), -4.6 / 1020, 2e-4, "L3, worked example")
    base = cyclically_adjusted_levels(R=20.0, G=20.0, Y=100.0, YP=100.0)
    cases = {
        "recession, no new measures": (dict(R=20 * 0.97, G=20.0, Y=97.0, YP=100.0), ("<0", "=0", "=0")),
        "recession, emergency aid": (dict(R=20 * 0.97, G=21.0, Y=97.0, YP=100.0), ("<0", "<0", ">0")),
        "growth at potential, temporary tax cut": (dict(R=20 * 1.02 - 1.0, G=20.0, Y=102.0, YP=102.0), ("<0", "<0", ">0")),
        "growth above potential, temporary tax cut": (dict(R=20 * 1.05 - 0.5, G=20.0, Y=105.0, YP=102.0), (None, "<0", ">0")),
    }

    def sign_ok(x, rule):
        return rule is None or {"<0": x < -1e-12, ">0": x > 1e-12, "=0": abs(x) <= 1e-12}[rule]

    for label, (kw, (d_pb, d_capb, fi)) in cases.items():
        now = cyclically_adjusted_levels(**kw)
        dPB = now["PB"] / kw["YP"] - base["PB"] / 100
        dCAPB = now["CAPB"] / kw["YP"] - base["CAPB"] / 100
        holds(f"sign table: {label}", sign_ok(dPB, d_pb) and sign_ok(dCAPB, d_capb) and sign_ok(-dCAPB, fi), "L3 slide 29")

    # Equation (10): the bootstrap is reproducible and the fan is ordered
    Xs = pd.DataFrame(rng.normal(size=(120, 4)).cumsum(axis=0) * 0.01, columns=["g", "pi", "i", "s"])
    vres = fit_var(Xs.diff().dropna(), maxlags=2)
    a, _ = simulate_var_bootstrap(vres, Xs.diff().dropna().values, 8, 200, seed=1)
    b, _ = simulate_var_bootstrap(vres, Xs.diff().dropna().values, 8, 200, seed=1)
    holds("bootstrap with a fixed seed is reproducible", np.array_equal(a, b), "Equation (10)")
    fan = fan_percentiles(a[:, :, 0], range(8))
    holds("fan percentiles are ordered", bool((fan.diff(axis=1).iloc[:, 1:] >= 0).all().all()), "Equation (10)")

    table = pd.DataFrame(rows)
    return table
