"""Data ingestion: series dictionary, download with dated cache, verification of codes, and the analysis panels.

Sources: BCB SGS (api.bcb.gov.br), BCB open-data portal (titles), IBGE SIDRA (apisidra.ibge.gov.br).
Flows from the BCB are 12-month sums over 12-month GDP with the sign of the borrowing requirement (deficit > 0),
so the primary surplus used here is minus the published value.
"""
import json
import time
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .identity import decompose_debt_change, implicit_rate, real_rate
from .simulate import snowball

SGS_SERIES = {
    "dbgg": {
        "code": 13762,
        "title": "Dívida bruta do governo geral (% PIB) - Metodologia utilizada a partir de 2008",
        "must_contain": ["divida bruta do governo geral", "% pib"],
        "unit": "% of GDP, end-of-month stock",
        "perimeter": "General government (federal government, INSS, states, municipalities); excludes the BCB and state-owned enterprises",
        "verification": "Official title confirmed on the BCB open-data portal on 2026-10-30.",
        "offline_verified": True,
    },
    "dlsp": {
        "code": 4513,
        "title": "Dívida Líquida do Setor Público (% PIB) - Total - Setor público consolidado",
        "must_contain": ["divida liquida do setor publico", "% pib", "setor publico consolidado"],
        "unit": "% of GDP, end-of-month stock",
        "perimeter": "Consolidated public sector: general government, BCB and state-owned enterprises (BCB methodology)",
        "verification": "Official title confirmed on the BCB open-data portal on 2026-10-30.",
        "offline_verified": True,
    },
    "primary_consolidated": {
        "code": 5793,
        "title": "NFSP sem desvalorização cambial (% PIB) - Fluxo acumulado em 12 meses - Resultado primário - Total - Setor público consolidado",
        "must_contain": ["resultado primario", "12 meses", "setor publico consolidado"],
        "unit": "% of GDP, 12-month flow, deficit > 0",
        "perimeter": "Consolidated public sector (pairs with the DLSP)",
        "verification": "Official title confirmed on the BCB open-data portal on 2026-10-30.",
        "offline_verified": True,
    },
    "primary_federal_bcb": {
        "code": 5783,
        "title": "NFSP sem desvalorização cambial (% PIB) - Fluxo acumulado em 12 meses - Resultado primário - Total - Governo Federal e Banco Central",
        "must_contain": ["resultado primario", "12 meses", "governo federal e banco central"],
        "unit": "% of GDP, 12-month flow, deficit > 0",
        "perimeter": "Federal government (including INSS) and BCB; summed with 5786 to approximate the general government",
        "verification": "Official title confirmed on the BCB open-data portal on 2026-10-30.",
        "offline_verified": True,
    },
    "primary_regional": {
        "code": 5786,
        "title": "NFSP sem desvalorização cambial (% PIB) - Fluxo acumulado em 12 meses - Resultado primário - Total - Governos estaduais e municipais",
        "must_contain": ["resultado primario", "12 meses", "governos estaduais e municipais"],
        "unit": "% of GDP, 12-month flow, deficit > 0",
        "perimeter": "States and municipalities",
        "verification": "Official title confirmed on the BCB open-data portal on 2026-10-30.",
        "offline_verified": True,
    },
    "interest_consolidated": {
        "code": 5760,
        "title": "NFSP sem desvalorização cambial (% PIB) - Fluxo acumulado em 12 meses - Juros nominais - Total - Setor público consolidado",
        "must_contain": ["juros nominais", "12 meses", "setor publico consolidado"],
        "unit": "% of GDP, 12-month flow, expense > 0",
        "perimeter": "Consolidated public sector, net nominal interest (pairs with the DLSP)",
        "verification": ("Inferred: the portal lists 5760 in the family 'Juros nominais - Total'; 5754 (states) and 5755 "
                         "(municipalities) were confirmed, and the primary-balance family shows the same ordering with the "
                         "consolidated sector last (5793). Confirmed against the official title at run time."),
        "offline_verified": False,
    },
}

SIDRA_SERIES = {
    "gdp_nominal": {
        "table": 1846, "variable": 585, "classification": 11255, "category": 90707,
        "must_contain_variable": ["precos correntes"], "must_contain_category": ["pib a precos de mercado"],
        "unit": "R$ million, quarterly, current prices",
        "verification": "Variable and category names returned by the API are checked at run time.",
    },
    "gdp_real_index": {
        "table": 1620, "variable": 583, "classification": 11255, "category": 90707,
        "must_contain_variable": ["encadeada", "volume"], "must_contain_category": ["pib a precos de mercado"],
        "unit": "Chained volume index (1995 = 100), quarterly, not seasonally adjusted",
        "verification": "Variable and category names returned by the API are checked at run time.",
    },
}

# Levels taken from the slides: IFI (Dec/2025) chart in Lecture 9 slide 16, IFI table in slide 22, Lecture 1 indicator table.
COURSE_LEVELS = [
    ("d_dbgg", 2013, 0.515, "IFI (Dec/2025) chart, L9 slide 16"),
    ("d_dbgg", 2019, 0.744, "IFI (Dec/2025) chart, L9 slide 16"),
    ("d_dbgg", 2020, 0.869, "IFI (Dec/2025) chart, L9 slide 16"),
    ("d_dbgg", 2024, 0.763, "IFI (Dec/2025) chart, L9 slide 16"),
    ("d_dbgg", 2025, 0.790, "IFI table, L9 slide 22"),
    ("d_dlsp", 2025, 0.650, "Indicator table, L1 (about 65%)"),
]

SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
SIDRA_URL = "https://apisidra.ibge.gov.br/values/t/{table}/n1/all/v/{variable}/p/all/c{classification}/{category}"
CKAN_SEARCH_URL = "https://dadosabertos.bcb.gov.br/api/3/action/package_search"
HTTP_HEADERS = {"Accept": "application/json", "User-Agent": "brazil-debt-dynamics (academic project)"}


def _norm(text):
    """Lower case, accents removed, single spaces: used to compare official titles and labels."""
    text = unicodedata.normalize("NFKD", str(text))
    return " ".join("".join(ch for ch in text if not unicodedata.combining(ch)).lower().split())


def http_get_json(url, params=None, retries=3, timeout=60):
    """GET returning parsed JSON; None on HTTP 404 (the SGS answers 404 for windows without observations)."""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, headers=HTTP_HEADERS, timeout=timeout)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last_error}")


def fetch_sgs(code, start, end=None):
    """Downloads an SGS series in windows of at most ten years."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
    frames, lo = [], start
    while lo <= end:
        hi = min(lo + pd.DateOffset(years=10) - pd.Timedelta(days=1), end)
        payload = http_get_json(SGS_URL.format(code=code), params={
            "formato": "json", "dataInicial": lo.strftime("%d/%m/%Y"), "dataFinal": hi.strftime("%d/%m/%Y")})
        if isinstance(payload, list) and payload:
            frames.append(pd.DataFrame(payload))
        lo = hi + pd.Timedelta(days=1)
    if not frames:
        raise RuntimeError(f"SGS {code}: no observations returned")
    df = pd.concat(frames, ignore_index=True)
    values = pd.Series(pd.to_numeric(df["valor"], errors="coerce").values,
                       index=pd.to_datetime(df["data"], format="%d/%m/%Y"), name="value")
    return values[~values.index.duplicated()].sort_index()


def fetch_sidra(table, variable, classification, category):
    """Downloads a quarterly SIDRA series for Brazil; returns the series and the names returned by the API."""
    payload = http_get_json(SIDRA_URL.format(table=table, variable=variable, classification=classification, category=category))
    if not payload or len(payload) < 2:
        raise RuntimeError(f"SIDRA {table}: empty response")
    header, rows = payload[0], payload[1:]
    key_of = {_norm(label): key for key, label in header.items()}
    period_key, variable_key = key_of["trimestre (codigo)"], key_of["variavel"]
    category_key = next(k for k in header if k.startswith("D") and k.endswith("C") and str(rows[0].get(k)) == str(category))
    df = pd.DataFrame(rows)
    periods = pd.PeriodIndex([pd.Period(year=int(p[:4]), quarter=int(p[4:]), freq="Q") for p in df[period_key]])
    series = pd.Series(pd.to_numeric(df["V"], errors="coerce").values, index=periods, name="value").sort_index()
    info = {"variable_name": str(df[variable_key].iloc[0]), "category_name": str(df[category_key[:-1] + "N"].iloc[0]),
            "unit": str(df["MN"].iloc[0]) if "MN" in df else ""}
    return series, info


def _update_manifest(data_dir, key, entry):
    path = Path(data_dir) / "raw" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    manifest[key] = entry
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def read_manifest(data_dir):
    path = Path(data_dir) / "raw" / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _latest(folder, stem):
    files = sorted(folder.glob(f"{stem}_*.csv"))
    return files[-1] if files else None


def load_sgs(key, data_dir, refresh=False, start="2000-01-01"):
    """Cached SGS series: data/raw/sgs/sgs_<code>_<YYYYMMDD>.csv; downloads when absent or when refresh=True."""
    code = SGS_SERIES[key]["code"]
    folder = Path(data_dir) / "raw" / "sgs"
    folder.mkdir(parents=True, exist_ok=True)
    path = _latest(folder, f"sgs_{code}")
    if path is None or refresh:
        series = fetch_sgs(code, start)
        stamp = pd.Timestamp.today().strftime("%Y%m%d")
        path = folder / f"sgs_{code}_{stamp}.csv"
        series.to_csv(path, index_label="date")
        _update_manifest(data_dir, f"sgs_{code}", {"key": key, "file": path.name, "downloaded": stamp,
                                                   "n_obs": int(series.notna().sum()),
                                                   "first": str(series.index.min().date()), "last": str(series.index.max().date())})
    return pd.read_csv(path, parse_dates=["date"], index_col="date")["value"]


def load_sidra(key, data_dir, refresh=False):
    """Cached SIDRA series: data/raw/sidra/sidra_<table>_v<variable>_<YYYYMMDD>.csv (+ .json with the API names)."""
    meta = SIDRA_SERIES[key]
    folder = Path(data_dir) / "raw" / "sidra"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"sidra_{meta['table']}_v{meta['variable']}"
    path = _latest(folder, stem)
    if path is None or refresh:
        series, info = fetch_sidra(meta["table"], meta["variable"], meta["classification"], meta["category"])
        stamp = pd.Timestamp.today().strftime("%Y%m%d")
        path = folder / f"{stem}_{stamp}.csv"
        series.to_frame().assign(period=series.index.astype(str)).set_index("period").to_csv(path)
        path.with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        _update_manifest(data_dir, stem, {"key": key, "file": path.name, "downloaded": stamp, **info,
                                          "first": str(series.index.min()), "last": str(series.index.max())})
    df = pd.read_csv(path)
    series = pd.Series(df["value"].values, index=pd.PeriodIndex(df["period"], freq="Q"), name="value")
    info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return series, info


def bcb_official_title(code):
    payload = http_get_json(CKAN_SEARCH_URL, params={"q": str(code), "rows": 100}, retries=1, timeout=20)
    for pkg in ((payload or {}).get("result") or {}).get("results", []):
        if str(pkg.get("name", "")).startswith(f"{code}-"):
            return pkg.get("title")
    return None


def verify_sgs_series(strict=False):
    """Compares each official SGS title with the expected keywords. A mismatch always raises; an unreachable
    portal produces a message (or raises when strict=True) for codes not confirmed in advance."""
    rows, portal_error = [], None
    for key, meta in SGS_SERIES.items():
        title, error = None, portal_error
        if portal_error is None:
            try:
                title = bcb_official_title(meta["code"])
            except Exception as exc:  # portal unreachable: stop querying it; the decision is taken below
                portal_error = error = str(exc)[:120]
        if title is None:
            ok = meta["offline_verified"]
            status = "confirmed on 2026-10-30 (portal not reachable now)" if ok else "NOT VERIFIED"
        else:
            missing = [w for w in meta["must_contain"] if _norm(w) not in _norm(title)]
            ok = not missing
            status = "verified online" if ok else f"MISMATCH (missing: {', '.join(missing)})"
        rows.append({"key": key, "code": str(meta["code"]), "official_title": title, "status": status, "ok": ok, "error": error})
    table = pd.DataFrame(rows).set_index("key")
    mismatches = table[table["status"].str.startswith("MISMATCH")]
    if len(mismatches):
        raise RuntimeError("Series title does not match the expected content; fix SGS_SERIES before continuing:\n"
                           + mismatches[["code", "official_title", "status"]].to_string())
    messages = []
    unverified = table[~table["ok"]]
    if len(unverified):
        msg = ("Could not confirm online: " + ", ".join(f"{k} ({c})" for k, c in unverified["code"].items())
               + ". Check the titles at https://www3.bcb.gov.br/sgspub before relying on them.")
        if strict:
            raise RuntimeError(msg)
        messages.append(msg)
    return table, messages


def verify_sidra_info(key, info):
    meta = SIDRA_SERIES[key]
    missing = [w for w in meta["must_contain_variable"] if _norm(w) not in _norm(info["variable_name"])]
    missing += [w for w in meta["must_contain_category"] if _norm(w) not in _norm(info["category_name"])]
    if missing:
        raise RuntimeError(f"SIDRA {meta['table']}: unexpected names {info} (missing: {missing})")
    return {"key": key, "code": f"SIDRA {meta['table']}/v{meta['variable']}",
            "official_title": f"{info['variable_name']} | {info['category_name']}", "status": "verified online", "ok": True, "error": None}


def to_quarterly(monthly):
    """Last month of each quarter, indexed by quarter."""
    m = monthly.dropna()
    q = m[m.index.month % 3 == 0].copy()
    q.index = q.index.to_period("Q")
    return q


def build_panels(raw, gdp_nominal_q, gdp_real_q):
    """Quarterly panel of 12-month windows and the annual panel (its fourth-quarter rows)."""
    Q = pd.DataFrame({
        "d_dbgg": to_quarterly(raw["dbgg"]) / 100,
        "d_dlsp": to_quarterly(raw["dlsp"]) / 100,
        "s_cons": -to_quarterly(raw["primary_consolidated"]) / 100,
        "s_gg": -(to_quarterly(raw["primary_federal_bcb"]) + to_quarterly(raw["primary_regional"])) / 100,
        "s_fed": -to_quarterly(raw["primary_federal_bcb"]) / 100,
        "interest_cons": to_quarterly(raw["interest_consolidated"]) / 100,
    })
    Q = Q.join(pd.DataFrame({"gdp_nominal_4q": gdp_nominal_q.rolling(4).sum(), "real_index_4q": gdp_real_q.rolling(4).sum()}), how="outer")
    Q = Q.reindex(pd.period_range(Q.index.min(), Q.index.max(), freq="Q"))
    Q.index.name = "quarter"
    Q["g"] = Q["real_index_4q"] / Q["real_index_4q"].shift(4) - 1
    Q["nominal_growth"] = Q["gdp_nominal_4q"] / Q["gdp_nominal_4q"].shift(4) - 1
    Q["pi"] = (1 + Q["nominal_growth"]) / (1 + Q["g"]) - 1
    Q["i_dlsp"] = implicit_rate(Q["interest_cons"], Q["d_dlsp"].shift(4), Q["g"], Q["pi"])
    Q["r_dlsp"] = real_rate(Q["i_dlsp"], Q["pi"])
    Q["lambda_minus_1_dlsp"] = snowball(Q["r_dlsp"], Q["g"]) - 1
    dec = decompose_debt_change(Q["d_dlsp"], Q["d_dlsp"].shift(4), Q["g"], Q["pi"], Q["s_cons"], interest=Q["interest_cons"])
    Q["sfa_dlsp"] = dec["sfa"]
    A = Q[Q.index.quarter == 4].copy()
    A.index = pd.Index(A.index.year, name="year")
    return Q, A


def level_checks(A, tolerance=0.015, tolerance_dlsp=0.02):
    """Compares the data with the levels shown in the slides (warning only: ratios are revised with GDP)."""
    rows = []
    for col, year, expected, source in COURSE_LEVELS:
        value = A[col].get(year, np.nan)
        tol = tolerance_dlsp if col == "d_dlsp" else tolerance
        rows.append({"series": col, "year": year, "data": value, "course": expected, "diff_pp": 100 * (value - expected),
                     "within_tolerance": bool(abs(value - expected) <= tol) if pd.notna(value) else False, "source": source})
    return pd.DataFrame(rows)


def sanity_checks(A):
    """Stops on sign or unit mistakes."""
    if not A.loc[2020, "s_cons"] < 0:
        raise ValueError("Primary balance has the wrong sign: 2020 must show a consolidated primary deficit.")
    if not (A["interest_cons"].dropna() > 0).all():
        raise ValueError("Nominal interest must be a positive expense in every year.")
    if not (A["d_dlsp"].dropna().between(0.2, 1.2).all() and A["d_dbgg"].dropna().between(0.3, 1.5).all()):
        raise ValueError("Debt ratios outside the plausible range: check units.")


def load_all(data_dir, refresh=False, strict=False, start="2000-01-01"):
    """Verifies the codes, loads every series (downloading when needed) and builds the panels."""
    data_dir = Path(data_dir)
    for sub in ("raw", "processed", "manual"):
        (data_dir / sub).mkdir(parents=True, exist_ok=True)
    verification, messages = verify_sgs_series(strict)
    raw = {key: load_sgs(key, data_dir, refresh, start) for key in SGS_SERIES}
    gdp_nominal_q, info_nominal = load_sidra("gdp_nominal", data_dir, refresh)
    gdp_real_q, info_real = load_sidra("gdp_real_index", data_dir, refresh)
    verification = pd.concat([verification, pd.DataFrame([verify_sidra_info("gdp_nominal", info_nominal),
                                                           verify_sidra_info("gdp_real_index", info_real)]).set_index("key")])
    Q, A = build_panels(raw, gdp_nominal_q, gdp_real_q)
    sanity_checks(A)
    checks = level_checks(A)
    if not checks["within_tolerance"].all():
        messages.append("Some debt levels differ from the course by more than the tolerance (GDP revisions or a wrong code): see Data & checks.")
    Q.to_csv(data_dir / "processed" / "quarterly_panel.csv")
    A.to_csv(data_dir / "processed" / "annual_panel.csv")
    for msg in messages:
        warnings.warn(msg)
    return {"raw": raw, "Q": Q, "A": A, "verification": verification, "level_checks": checks,
            "manifest": read_manifest(data_dir), "messages": messages}


MANUAL_INPUTS = {
    "dbgg_gross_interest": {
        "columns": ["year", "interest_pct_gdp"],
        "help": ("Nominal interest appropriated on the DBGG, % of GDP, from the BCB table of conditioning factors of the gross "
                 "debt (Estatísticas fiscais, 'Dívida bruta do governo geral – fatores condicionantes', row 'Juros nominais apropriados')."),
    },
    "dlsp_conditioning_factors": {
        "columns": ["year", "interest", "primary", "fx_adjustment", "other_adjustments", "gdp_effect"],
        "help": "BCB conditioning factors of the DLSP, p.p. of GDP, BCB sign convention (positive raises debt).",
    },
    "central_government_rtn": {
        "columns": ["year", "revenue_pct_gdp", "expenditure_pct_gdp"],
        "help": "Central-government net revenue and primary expenditure, % of GDP (Tesouro Nacional, Resultado do Tesouro Nacional).",
    },
    "structural_balance_benchmarks": {
        "columns": ["year", "source", "value_pct_gdp"],
        "help": "Official structural primary results for comparison (e.g. SPE, IFI), % of GDP.",
    },
    "federal_debt_composition": {
        "columns": ["date", "indexer", "share_pct"],
        "help": "Federal public debt by indexer, % of total (Tesouro Nacional, Relatório Mensal da Dívida).",
    },
    "primary_targets": {
        "columns": ["year", "target_pct_gdp", "lower_pct_gdp", "upper_pct_gdp"],
        "help": "Primary-balance targets of the fiscal framework and tolerance band, % of GDP.",
    },
}


def manual_path(data_dir, name):
    return Path(data_dir) / "manual" / f"{name}.csv"


def read_manual(data_dir, name):
    """Optional manual input: returns a DataFrame with the expected columns, or None when the file is absent."""
    path = manual_path(data_dir, name)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    missing = [c for c in MANUAL_INPUTS[name]["columns"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    return df


def template_csv(name):
    return ",".join(MANUAL_INPUTS[name]["columns"]) + "\n"
