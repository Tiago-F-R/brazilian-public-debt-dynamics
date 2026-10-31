"""Brazilian public debt dynamics: dashboard.

Run from the repository root:   streamlit run app/app.py

Every number shown here is computed by the debtdyn package (src/debtdyn); this file only arranges controls and charts.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from debtdyn import analysis, data, plots, stochastic  # noqa: E402
from debtdyn.coursetests import IFI_DEC2025, run_course_tests  # noqa: E402
from debtdyn.simulate import stabilizing_grid, stabilizing_primary  # noqa: E402

DATA_DIR = ROOT / "data"
R_IFI, G_IFI = 0.052, 0.022          # IFI highlighted cell, Lecture 9 slide 22
AGGREGATES = {"DLSP: net debt, consolidated public sector": "dlsp", "DBGG: gross debt, general government": "dbgg"}

st.set_page_config(page_title="Brazilian Public Debt Dynamics", layout="wide")

_ST_VERSION = tuple(int(x) for x in st.__version__.split(".")[:2])
STRETCH = {"width": "stretch"} if _ST_VERSION >= (1, 50) else {"use_container_width": True}


def chart(fig):
    st.plotly_chart(fig, theme=None, **STRETCH)


def table(df, formats=None, **kwargs):
    shown = df.style.format(formats, na_rep="") if formats else df
    st.dataframe(shown, **STRETCH, **kwargs)


def pct(x, digits=2):
    return "n/a" if x is None or pd.isna(x) else f"{100 * x:.{digits}f}%"


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load_dataset(refresh):
    return data.load_all(DATA_DIR, refresh=refresh)


@st.cache_data(show_spinner=False)
def course_tests():
    return run_course_tests()


@st.cache_data(show_spinner=False)
def bohn_cached(Q, perimeter, window, hac_lags):
    return analysis.bohn_analysis(Q, perimeter, window, hac_lags)


@st.cache_data(show_spinner=False)
def fan_cached(Q, horizon, n_paths, seed, sfa_mode):
    return stochastic.dlsp_fan(Q, horizon, n_paths, seed, sfa_mode)


def manual_input(name, label):
    """Upload box for an optional input; the file is kept in data/manual/ so later sessions find it."""
    spec = data.MANUAL_INPUTS[name]
    path = data.manual_path(DATA_DIR, name)
    with st.expander(f"{label} ({'loaded' if path.exists() else 'not supplied'})"):
        st.caption(spec["help"])
        st.caption("Columns: " + ", ".join(f"`{c}`" for c in spec["columns"]))
        uploaded = st.file_uploader(f"Upload {name}.csv", type="csv", key=f"upload_{name}")
        if uploaded is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(uploaded.getvalue())
            st.success(f"Saved as data/manual/{name}.csv")
        st.download_button("Download an empty template", data.template_csv(name), file_name=f"{name}.csv",
                           mime="text/csv", key=f"template_{name}")
        if path.exists() and st.button("Remove the saved file", key=f"remove_{name}"):
            path.unlink()
            st.rerun()
    try:
        return data.read_manual(DATA_DIR, name)
    except ValueError as exc:
        st.error(str(exc))
        return None


with st.sidebar:
    st.header("Controls")
    refresh = st.button("Download the data again", help="Downloads every series from the BCB and the IBGE and keeps the old files.")
if refresh:
    load_dataset.clear()
try:
    with st.spinner("Loading BCB and IBGE data (the first run downloads and caches them)..."):
        ds = load_dataset(refresh)
    data_error = None
except Exception as exc:  # no internet, API change or a failed verification: the simulator still works
    ds, data_error = None, f"{type(exc).__name__}: {exc}"

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    if ds is not None:
        A, Q, raw = ds["A"], ds["Q"], ds["raw"]
        first_year = int(A["d_dlsp"].first_valid_index())
        last_year = int(A["d_dlsp"].last_valid_index())
        year_range = st.slider("Period (history charts)", first_year, last_year, (max(first_year, 2003), last_year))
        aggregate_label = st.radio("Debt aggregate (decomposition)", list(AGGREGATES))
        st.divider()
        downloaded = sorted({v.get("downloaded", "") for v in ds["manifest"].values()})
        if downloaded:
            st.caption(f"Data downloaded on {', '.join(pd.to_datetime(downloaded).strftime('%d/%m/%Y'))}.")
        for msg in ds["messages"]:
            st.warning(msg)
    st.caption("Application of *Topics in Fiscal Policy* (Prof. Marco Cavalcanti, FGV EPGE, 2026.2). "
               "Trajectories and fan charts are conditional exercises, not projections.")

# ---------------------------------------------------------------- header
st.title("Brazilian Public Debt Dynamics")
st.caption("Debt decomposition, sustainability scenarios, fiscal stance, Bohn test and stochastic analysis. "
           "Sources: BCB (SGS) and IBGE (SIDRA); calculations by the author.")

if data_error:
    st.error(f"Data could not be loaded. {data_error}")
    st.info("The simulator tab works without data. The other tabs need a first download with internet access.")
else:
    m = analysis.headline_metrics(raw)
    s_star = stabilizing_primary(m["dbgg"]["value"] / 100, R_IFI, G_IFI)
    cols = st.columns(5)
    cols[0].metric("Gross debt (DBGG)", f"{m['dbgg']['value']:.1f}%", f"{m['dbgg']['change_12m']:+.1f} p.p. in 12 m",
                   delta_color="inverse", help="General government, % of GDP (SGS 13762).")
    cols[1].metric("Net debt (DLSP)", f"{m['dlsp']['value']:.1f}%", f"{m['dlsp']['change_12m']:+.1f} p.p. in 12 m",
                   delta_color="inverse", help="Consolidated public sector, % of GDP (SGS 4513).")
    cols[2].metric("Primary balance", f"{m['primary_cons']['value']:+.2f}%", f"{m['primary_cons']['change_12m']:+.2f} p.p. in 12 m",
                   help="Consolidated public sector, 12-month flow, % of GDP; surplus > 0 (minus SGS 5793).")
    cols[3].metric("Interest bill", f"{m['interest_cons']['value']:.2f}%", f"{m['interest_cons']['change_12m']:+.2f} p.p. in 12 m",
                   delta_color="inverse", help="Nominal interest of the consolidated public sector, 12-month flow, % of GDP (SGS 5760).")
    cols[4].metric("Primary to stabilise DBGG", f"{100 * s_star:.2f}%", f"{m['primary_gg']['value'] - 100 * s_star:+.2f} p.p. vs current",
                   help="Equation (4) at the latest DBGG with the IFI cell r = 5.2%, g = 2.2% (Lecture 9, slide 22). The delta is the "
                        "current general-government primary (12 months) minus the required one.")
    st.caption(f"All values in % of GDP. Latest BCB observation: {m['dbgg']['date']:%b %Y}; IBGE GDP through "
               f"{Q['real_index_4q'].last_valid_index()}.")

tabs = st.tabs(["Decomposition", "Simulator", "Fiscal stance", "Bohn test", "Fan chart", "Composition & rules", "Data & checks", "Method"])


def needs_data():
    st.info("This tab needs the BCB and IBGE data; see the message at the top of the page.")


# ---------------------------------------------------------------- M1
with tabs[0]:
    if ds is None:
        needs_data()
    else:
        aggregate = AGGREGATES[aggregate_label]
        gross = manual_input("dbgg_gross_interest", "Gross interest bill of the DBGG") if aggregate == "dbgg" else None
        gross_interest = None if gross is None else gross.set_index("year")["interest_pct_gdp"] / 100
        dec, parts = analysis.annual_decomposition(A, aggregate, gross_interest)
        dec = dec.loc[year_range[0]:year_range[1]]
        name = "net debt ratio (DLSP)" if aggregate == "dlsp" else "gross debt ratio (DBGG)"
        chart(plots.decomposition(dec, f"Contributions to the annual change in the {name}", parts))
        if aggregate == "dbgg" and gross_interest is None:
            st.caption("The SGS publishes no interest bill for the gross debt, so interest and the stock-flow adjustment appear "
                       "as one residual. Upload the gross interest bill above to separate them.")
        with st.expander("Table of contributions (p.p. of GDP)"):
            table((100 * dec).round(2), "{:+.2f}")

        if aggregate == "dlsp":
            large = dec.loc[dec["sfa"].abs() > 0.02, ["sfa", "change"]]
            st.markdown(f"**Stock-flow adjustment.** It closes the identity by construction, so its size is the diagnostic. "
                        f"Years above 2 p.p. of GDP in the period: {', '.join(map(str, large.index)) or 'none'}. Expected sources: "
                        "exchange-rate valuation (the interest series excludes it; since the public sector became a net foreign-currency "
                        "creditor, depreciations lower the DLSP), recognition of liabilities, privatisations, and the gap between the "
                        "BCB's GDP and the four-quarter IBGE GDP.")
            factors = manual_input("dlsp_conditioning_factors", "BCB conditioning factors of the DLSP (validation)")
            if factors is not None:
                table(100 * analysis.compare_with_bcb_factors(dec, factors), "{:+.2f}")

        st.divider()
        c1, c2 = st.columns(2, gap="large")
        with c1:
            monthly = {"DBGG (gross, general government)": raw["dbgg"].dropna() / 100,
                       "DLSP (net, consolidated public sector)": raw["dlsp"].dropna() / 100}
            monthly = {k: v[(v.index.year >= year_range[0]) & (v.index.year <= year_range[1])] for k, v in monthly.items()}
            chart(plots.lines(monthly, "Gross and net public debt", "% of GDP", markers=False, hover_fmt=".1f"))
        with c2:
            rates = analysis.implicit_rates(A, gross_interest).loc[year_range[0]:year_range[1]]
            chart(plots.lines(dict(rates.items()), "Implicit interest rates and real growth", "% per year"))
        st.caption("Review Question 1e: the same equation governs gross and net debt with different effective rates; the DLSP rate "
                   "nets the return on reserves, so r > g on gross debt says nothing by itself about the net ratio.")

# ---------------------------------------------------------------- M2
with tabs[1]:
    d0_default = round(2 * m["dbgg"]["value"]) / 2 if ds is not None else 79.0
    pi_default = round(100 * float(A["pi"].dropna().iloc[-1]), 1) if ds is not None else 4.0
    left, right = st.columns([1, 2.3], gap="large")
    with left:
        st.markdown("**Scenario**")
        d0 = st.slider("Initial debt ratio (% of GDP)", 20.0, 150.0, d0_default, 0.5)
        r = st.slider("Real interest rate r (%)", -2.0, 12.0, 5.2, 0.1)
        g = st.slider("Real GDP growth g (%)", -3.0, 6.0, 2.2, 0.1)
        pi = st.slider("GDP-deflator inflation (%)", 0.0, 15.0, pi_default, 0.1, help="Only affects the nominal split of interest.")
        s = st.slider("Primary balance (% of GDP)", -5.0, 6.0, 0.0, 0.1)
        sigma = st.slider("Seigniorage (% of GDP)", 0.0, 1.0, 0.0, 0.01)
        target = st.slider("Target ratio (% of GDP)", 20.0, 150.0, d0_default - 10, 0.5)
        horizon = st.slider("Horizon (years)", 1, 30, 10)
        phi = st.slider("phi: rise in r per unit of debt ratio", 0.0, 0.20, 0.0, 0.01,
                        help="Equation (6): r_t = r + phi (d_{t-1} - d0). phi = 0.05 adds 5 bp to r per p.p. of GDP of extra debt. Illustrative.")
    sc = analysis.scenario_summary(d0 / 100, r / 100, g / 100, pi / 100, s / 100, target / 100, horizon, phi, sigma / 100)
    with right:
        k = st.columns(4)
        k[0].metric("Stabilising primary", pct(sc["s_star"]))
        k[1].metric(f"Primary for {target:.1f}% in {horizon} years", pct(sc["s_target"]))
        k[2].metric(f"Ratio after {horizon} years", pct(sc["d_end"], 1), f"{100 * (sc['d_end'] - d0 / 100):+.1f} p.p.", delta_color="inverse")
        k[3].metric("Implied nominal rate", pct(sc["i"]))
        chart(plots.lines({f"Chosen primary ({s:.1f}%)": sc["chosen"]["d"], f"Primary for the target ({100 * sc['s_target']:.2f}%)": sc["to_target"]["d"]},
                          "Debt ratio under the chosen scenario (conditional exercise)", "% of GDP", "years ahead", height=400,
                          hover_fmt=".1f", dashes={f"Primary for the target ({100 * sc['s_target']:.2f}%)": "dash"}))
        if phi > 0:
            st.caption("With phi > 0 the primary for the target, derived under a constant rate (equation 5), no longer lands exactly on it.")
        parts = analysis.FULL_PARTS + (["seigniorage"] if sigma > 0 else [])
        chart(plots.decomposition(sc["chosen"].iloc[1:], "Contributions along the chosen path", parts))
    st.divider()
    c1, c2 = st.columns(2, gap="large")
    with c1:
        chart(plots.heatmap(stabilizing_grid(d0 / 100, IFI_DEC2025.columns, IFI_DEC2025.index),
                            f"Stabilising primary at d = {d0:.1f}% (IFI layout, L9 slide 22)"))
        st.caption("At d = 79% this grid reproduces the IFI table of December 2025 within 0.08 p.p. (47 of 49 cells identical after rounding).")
    with c2:
        s_max = st.slider("Ceiling on the feasible primary surplus, s_max (% of GDP)", 0.5, 6.0, 3.0, 0.1,
                          help="Illustrative parameter, not a claim about Brazil's fiscal capacity.")
        d_grid, curves, limits = analysis.endogenous_rate(d0 / 100, r / 100, g / 100, s_max / 100)
        chart(plots.endogenous_curves(d_grid, curves, s_max / 100, "Required primary when r rises with debt"))
        table(limits, {"debt limit": "{:.1%}"})
        st.caption("Above the debt limit no primary up to s_max stabilises the ratio (equation 6; derivation beyond the slides).")
    st.divider()
    chart(plots.lines(analysis.scenario_set(d0 / 100, r / 100, g / 100, pi / 100, horizon),
                      "Reference scenarios around the chosen r and g", "% of GDP", "years ahead", height=460, hover_fmt=".1f"))

# ---------------------------------------------------------------- M3
with tabs[2]:
    if ds is None:
        needs_data()
    else:
        lamb = float(st.radio("HP smoothing parameter (annual data)", ["6.25", "100"], index=1, horizontal=True))
        gap = analysis.annual_output_gap(A, lamb)
        chart(plots.bars(gap["gap"].loc[year_range[0]:], f"Output gap, HP filter (lambda = {lamb:g})", "% of potential output", "Output gap"))
        st.caption("The HP filter is unreliable at the end of the sample, precisely in the years most consulted; compare the latest gaps "
                   "with the IFI and FGV estimates.")
        rtn = manual_input("central_government_rtn", "Central-government revenue and primary expenditure (Treasury, RTN)")
        bench = manual_input("structural_balance_benchmarks", "Official structural results for comparison (SPE, IFI)")
        if rtn is None:
            st.info("The cyclical adjustment needs central-government revenue and expenditure (upload above). The functions are already "
                    "tested against the worked example and the sign table of Lecture 3 (Data & checks).")
        else:
            ca = analysis.fiscal_stance(rtn, gap["gap"])
            series = {"Primary balance": ca["pb"], "Cyclically adjusted primary balance": ca["capb"]}
            if bench is not None:
                for src, sub in bench.groupby("source"):
                    series[f"Structural result, {src}"] = sub.set_index("year")["value_pct_gdp"] / 100
            chart(plots.lines(series, "Central government: observed and cyclically adjusted primary balance", "% of potential GDP"))
            chart(plots.bars(ca["fiscal_impulse"].dropna(), "Fiscal impulse, FI = -ΔCAPB (positive = expansionary)",
                             "p.p. of potential GDP", "Fiscal impulse"))
            table(ca.dropna(subset=["fiscal_impulse"])[["gap", "pb", "capb", "cpb", "fiscal_impulse", "stance", "cyclicality"]],
                  {c: "{:+.2%}" for c in ("gap", "pb", "capb", "cpb", "fiscal_impulse")})
            st.caption("Benchmark elasticities eps_R = 1, eps_G = 0. No structural balance of the author's is presented as equivalent "
                       "to the official series, which also remove non-recurring items.")

# ---------------------------------------------------------------- M4
with tabs[3]:
    if ds is None:
        needs_data()
    else:
        c = st.columns(3)
        perimeter = c[0].selectbox("Perimeter", list(analysis.PERIMETERS))
        window = c[1].slider("Rolling window (quarters)", 24, 60, 40, 4)
        hac = c[2].slider("Newey-West lags", 1, 8, 4)
        res = bohn_cached(Q, perimeter, window, hac)
        chart(plots.rolling_bohn(res["rolling"], f"{perimeter}: fiscal reaction in rolling {window}-quarter windows"))
        summ = res["summary"]
        line = f"Windows with rho significantly above zero (10%): {summ['share_significant']:.0%} of {summ['n_windows']}."
        if not np.isnan(summ["share_above_threshold"]):
            line += (f" Windows in which rho exceeds (r-g)/(1+g): {summ['share_above_threshold']:.0%}. "
                     f"Mean threshold over the sample: {pct(summ['mean_threshold'])}.")
        else:
            line += " The stability threshold needs the implicit rate on gross debt, which the SGS does not publish."
        st.markdown(line)
        st.markdown("**Full sample**")
        table(res["full"], {"rho": "{:.4f}", "HAC s.e.": "{:.4f}", "t": "{:.2f}", "p-value": "{:.3f}", "R2": "{:.2f}"})
        st.caption("Equation (8) on quarterly observations of 12-month flows, debt at the start of each window (d_{t-4}); controls: HP output "
                   "gap and the share of Apr–Dec 2020 in each window (temporary spending). The test looks backward, uses a short sample "
                   "with regime changes and possibly non-stationary series, and says nothing about safe levels (Lecture 9). "
                   "rho > 0 is the course's condition for the intertemporal constraint; rho > (r-g)/(1+g) is the stricter condition for the "
                   "ratio to converge.")

# ---------------------------------------------------------------- M5
with tabs[4]:
    if ds is None:
        needs_data()
    else:
        c = st.columns(5)
        horizon_q = c[0].slider("Horizon (quarters)", 8, 40, 20, 4)
        n_paths = c[1].select_slider("Simulated paths", [1000, 2000, 5000, 10000], 5000)
        seed = int(c[2].number_input("Seed", 0, 10_000, 2026, 1))
        sfa_mode = c[3].selectbox("Stock-flow adjustments", ["demeaned", "raw", "zero"],
                                  help="Drawn with the residual of the same quarter (OBR step 2); demeaned by default.")
        extra = c[4].slider("Threshold (p.p. above the latest DLSP)", 0, 30, 10)
        out = fan_cached(Q, horizon_q, n_paths, seed, sfa_mode)
        info, fan = out["info"], out["fan"]
        past = Q["d_dlsp"].dropna().loc[f"{max(year_range[0], 2005)}Q1":]
        chart(plots.fan(past, fan, f"DLSP fan chart: {n_paths:,} bootstrap paths, seed {seed} (conditional exercise)"))
        threshold = info["last_d"] + extra / 100
        prob = pd.DataFrame({"P(d > threshold)": (out["paths"] > threshold).mean(axis=0), "p10": fan["p10"], "median": fan["p50"],
                             "p90": fan["p90"]}, index=fan.index.astype(str))
        st.markdown(f"VAR({info['lags']}) on {info['n_obs']} quarters ({info['start']}–{info['end']}); stable: {info['stable']}. "
                    f"Threshold: {threshold:.0%} of GDP.")
        if not info["stable"]:
            st.warning("The estimated VAR is not stable: the simulated distribution widens without bound.")
        table(prob.iloc[3::4], {c_: "{:.1%}" for c_ in prob.columns})
        st.caption("OBR method (Lecture 9, slides 40–43): VAR on (g, pi, i, s) for the DLSP, residuals and stock-flow adjustments drawn "
                   "with replacement, debt equation on year-on-year quarterly data (d_t from d_{t-4}), medians aligned to the VAR's own "
                   "forecast. The IFI fan chart refers to the DBGG around its own baseline, so only the width of the bands is comparable.")

# ---------------------------------------------------------------- M6
with tabs[5]:
    if ds is None:
        needs_data()
    else:
        st.markdown("The fixed-rate and long-maturity share of the debt is the base on which unexpected inflation erodes its real value "
                    "(Barro and Bianchi, 2025; Lecture 8). Both inputs below are optional uploads.")
        comp = manual_input("federal_debt_composition", "Federal debt by indexer (Treasury, monthly debt report)")
        if comp is not None:
            comp["date"] = pd.to_datetime(comp["date"])
            chart(plots.stacked_area(comp, "Federal public debt by indexer", "% of total"))
        targets = manual_input("primary_targets", "Primary-balance targets of the fiscal framework")
        if targets is not None:
            s_fed = analysis.annual_federal_primary(Q)
            observed = s_fed.loc[targets["year"].min() - 3:]
            chart(plots.target_band(targets, observed, "Primary balance against the target (indicative)"))
            st.caption("The BCB's below-the-line primary of the federal government and the BCB (SGS 5783) differs from the Treasury's "
                       "above-the-line measure on which the targets are formally assessed.")

# ---------------------------------------------------------------- checks
with tabs[6]:
    tests = course_tests()
    passed = int(tests["pass"].sum())
    (st.success if passed == len(tests) else st.error)(f"Course tests: {passed} of {len(tests)} passed.")
    with st.expander("Tests against the course (Lecture 9 slides 20, 22, 26; drills; Lecture 3 worked example and sign table)"):
        table(tests, {"value": "{:.6f}", "expected": "{:.6f}", "tolerance": "{:.0e}"})
    if ds is not None:
        st.markdown("**Verification of the series codes**")
        table(ds["verification"][["code", "official_title", "status"]])
        st.markdown("**Levels against the slides** (warning only: ratios are revised when the IBGE revises GDP)")
        table(ds["level_checks"], {"data": "{:.1%}", "course": "{:.1%}", "diff_pp": "{:+.2f}"})
        st.markdown("**Downloads**")
        table(pd.DataFrame(ds["manifest"]).T)
        c1, c2 = st.columns(2)
        c1.download_button("Quarterly panel (CSV)", Q.to_csv(), "quarterly_panel.csv", "text/csv")
        c2.download_button("Annual panel (CSV)", A.to_csv(), "annual_panel.csv", "text/csv")
    st.markdown("**Series dictionary**")
    table(pd.DataFrame([{"key": k, "code": v["code"], "title": v["title"], "unit": v["unit"], "perimeter": v["perimeter"],
                         "verification": v["verification"]} for k, v in data.SGS_SERIES.items()]).set_index("key"))

# ---------------------------------------------------------------- method
with tabs[7]:
    st.markdown(r"""
**Notation.** Ratios to nominal GDP, as decimals: $d$ debt, $s$ primary surplus, $i$ implicit nominal rate, $\pi$ GDP-deflator inflation,
$g$ real growth, $1+r=(1+i)/(1+\pi)$, $\sigma$ seigniorage, $sfa$ stock–flow adjustment, $\lambda=(1+r)/(1+g)$.

**Debt dynamics (Lecture 2).** $d_t = \dfrac{1+i_t}{(1+g_t)(1+\pi_t)}\,d_{t-1} - s_t - \sigma_t + sfa_t$, and the change splits into
interest $\dfrac{i_t d_{t-1}}{(1+g_t)(1+\pi_t)}$, growth $-\dfrac{g_t d_{t-1}}{1+g_t}$, inflation $-\dfrac{\pi_t d_{t-1}}{(1+g_t)(1+\pi_t)}$,
primary balance, seigniorage and the stock–flow adjustment, which is the residual. The interest term equals the interest bill over GDP of
the same period, as the BCB publishes it, so $i_t=(J_t/Y_t)(1+g_t)(1+\pi_t)/d_{t-1}$.

**Sustainability (Lecture 9).** Stabilising primary $s^*=\dfrac{r-g}{1+g}d-\sigma$; constant primary that reaches $d^*$ in $n$ years
$s=\dfrac{(\lambda^n d_0-d^*)(\lambda-1)}{\lambda^n-1}-\sigma$; with $r_t=\bar r+\phi(d_{t-1}-\bar d)$ the steady states solve
$\phi d^2+(\bar r-\phi\bar d-g)d-(1+g)(s+\sigma)=0$, and the largest root is the debt limit for the primary $s$.

**Bohn test (Lecture 9).** $s_t=\rho\,d_{t-1}+\gamma'Z_t+\varepsilon_t$. Under $s_t=\bar s+\rho d_{t-1}$, $d_t=(\lambda-\rho)d_{t-1}-\bar s$,
so the ratio converges if and only if $\rho>\lambda-1=(r-g)/(1+g)$, a stricter condition than the $\rho>0$ of the slides.

**Fiscal stance (Lecture 3).** $R^{CA}=R(Y^P/Y)^{\epsilon_R}$, $G^{CA}=G(Y^P/Y)^{\epsilon_G}$, $CAPB=R^{CA}-G^{CA}$, $FI=-\Delta CAPB$.

**Stochastic analysis (Lecture 9, OBR).** VAR on $(g,\pi,i,s)$, bootstrap of residuals and stock–flow adjustments, debt equation on
year-on-year quarterly data, medians aligned to the baseline.

**Perimeters.** DBGG with the general-government primary (SGS 5783 + 5786); DLSP with the consolidated primary (5793) and nominal
interest (5760). Seigniorage enters neither empirical identity: the monetary base is a BCB liability inside the DLSP, and the DBGG excludes the BCB.

**Limitations.** The stock–flow adjustment is a residual and needs reconciliation with the BCB conditioning factors; the SGS has no
interest bill for the gross debt; the general-government primary includes the BCB's small primary result; debt ratios use the BCB's GDP
while growth and the deflator use the IBGE's; the quarterly 12-month windows overlap (HAC errors in the Bohn test, not addressed in the
bootstrap); the Bohn test is backward-looking and regime-dependent; the HP gap is unreliable at the end of the sample; $\phi$ and $s_{\max}$
are illustrative; every path and fan chart is conditional on its assumptions, and none is a projection or an official estimate.

**References.** Burnside (2005); Bohn (1998); Blanchard (2019, 2021); Fedelino et al. (2009); Bornhorst et al. (2011); Debrun et al. (2020);
Mauro and Zhou (2020); Barro and Bianchi (2025); OBR stochastic method; IFI, *Relatório de Acompanhamento Fiscal*, Dec/2025.
""")
