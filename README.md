# Brazilian public debt dynamics

A simulator and dashboard of Brazilian public debt, built as an application of the course *Topics in Fiscal Policy: Theory, Evidence and the Case of Brazil* (Prof. Marco Cavalcanti, FGV EPGE, 2026.2). Every result follows from one accounting identity, the government budget constraint, and from its consequences for debt sustainability and for the measurement of the fiscal stance.

| Module | Lecture | Question |
|---|---|---|
| M1 Historical decomposition | 2 | Which forces moved the debt ratio each year: interest, growth, inflation, primary balance or stock–flow adjustments? |
| M2 Scenario simulator | 9 | Which primary balance stabilises the ratio, or brings it to a target, under given r, g and a debt-dependent rate? |
| M3 Cycle and fiscal stance | 3 | How much of the primary balance is cyclical, and what was the fiscal impulse? |
| M4 Bohn test | 9 | Did the primary balance react to debt, and strongly enough to stabilise the ratio? |
| M5 Stochastic analysis | 9 | What is the distribution of the debt ratio under historical shocks (OBR fan-chart method)? |
| M6 Composition and rules | 8, 10 | Composition of federal debt and the primary balance against the fiscal-framework target (optional inputs) |

## Run the dashboard

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

The dashboard opens in the browser at `http://localhost:8501`. The first run needs internet access: it downloads the series from the BCB (SGS) and the IBGE (SIDRA) and stores each one under `data/raw/` with its download date, so that later runs reproduce the same numbers even after the official series are revised. The button *Download the data again* in the sidebar fetches fresh data. The scenario simulator works without data.

## Repository layout

```text
brazil-debt-dynamics/
├── src/debtdyn/        all calculations: identity, simulate, cycle, bohn, stochastic, analysis, data, plots, coursetests
├── app/app.py          Streamlit dashboard (only calls the package)
├── data/raw/           downloads with their dates (created on first run)
├── data/processed/     quarterly and annual panels
└── data/manual/        optional inputs uploaded from the dashboard
```

## Data and verification

| Series | Source | Use |
|---|---|---|
| DBGG, gross debt of the general government, % of GDP | BCB SGS 13762 | M1, M2, M4 |
| DLSP, net debt of the consolidated public sector, % of GDP | BCB SGS 4513 | M1, M4, M5 |
| Primary balance, 12 months, % of GDP: consolidated / federal government and BCB / states and municipalities | BCB SGS 5793 / 5783 / 5786 | M1, M4, M5, M6 |
| Nominal interest, consolidated public sector, 12 months, % of GDP | BCB SGS 5760 | M1, M5 |
| GDP at current prices and chained volume index, quarterly | IBGE SIDRA 1846 and 1620 | all |

Before any series is used, its official title is compared with the expected content (BCB open-data portal for the SGS; names returned by the API for the IBGE), and the debt levels are compared with the figures shown in the course. The *Data & checks* tab shows both. Inputs that the public APIs do not provide (gross interest bill of the DBGG, BCB conditioning factors, Treasury revenue and expenditure, debt composition, fiscal targets) can be uploaded as CSV files from the dashboard; each upload box offers an empty template.

## Limitations

The stock–flow adjustment is a residual and needs reconciliation with the BCB conditioning factors; the SGS publishes no interest bill for the gross debt, so the complete decomposition and the fan chart use the net debt; the quarterly 12-month windows overlap; the Bohn test is backward-looking, uses a short sample with regime changes and says nothing about safe debt levels; the HP output gap is unreliable at the end of the sample; the debt-dependent rate parameters are illustrative. Every trajectory and fan chart is an exercise conditional on its assumptions, not a projection or an official estimate. The *Method* tab of the dashboard states these points in full.

## References

Burnside (2005), *Fiscal Sustainability in Theory and Practice*; Bohn (1998), QJE; Blanchard (2019, 2021); Fedelino et al. (2009) and Bornhorst et al. (2011), IMF technical notes on cyclically adjusted and structural balances; Debrun et al. (2020); Office for Budget Responsibility, stochastic fan-chart method; Instituição Fiscal Independente, *Relatório de Acompanhamento Fiscal* (December 2025); lecture slides of *Topics in Fiscal Policy*, FGV EPGE, 2026.2.
