"""Chart builders (plotly). They only draw what the other modules computed.

One categorical order is used everywhere and never cycled; each decomposition component keeps its color in every chart.
"""
import numpy as np
import plotly.graph_objects as go

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0.0, "#184f95"], [0.5, "#f0efec"], [1.0, "#b8322f"]]

COMPONENTS = {
    "interest": ("Interest", PALETTE[0]),
    "growth": ("Real growth", PALETTE[1]),
    "inflation": ("Inflation (GDP deflator)", PALETTE[2]),
    "primary": ("Primary balance", PALETTE[3]),
    "sfa": ("Stock-flow adjustment", PALETTE[4]),
    "seigniorage": ("Seigniorage", PALETTE[5]),
    "interest_plus_sfa": ("Interest + stock-flow adjustment", PALETTE[6]),
}


def style(fig, title, ytitle=None, xtitle=None, height=440):
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0, xanchor="left", font=dict(size=16, color=INK["primary"])),
        paper_bgcolor=INK["surface"], plot_bgcolor=INK["surface"],
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", size=12, color=INK["secondary"]),
        height=height, margin=dict(l=64, r=32, t=64, b=40),
        # legend under the plot: in narrow columns a multi-row legend above the plot would cover the title
        legend=dict(orientation="h", yanchor="top", y=-0.28 if xtitle else -0.16, xanchor="left", x=0, title=None),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor=INK["axis"], title=xtitle)
    fig.update_yaxes(gridcolor=INK["grid"], zerolinecolor=INK["axis"], zerolinewidth=1, title=ytitle)
    return fig


def decomposition(dec, title, components, x=None):
    """Stacked contributions (p.p. of GDP) with the change in the ratio as a line."""
    x = dec.index if x is None else x
    fig = go.Figure()
    for col in components:
        label, color = COMPONENTS[col]
        fig.add_bar(x=x, y=100 * dec[col], name=label, marker=dict(color=color, line=dict(color=INK["surface"], width=1.5)),
                    hovertemplate="%{x}: %{y:+.2f} p.p.<extra>" + label + "</extra>")
    fig.add_scatter(x=x, y=100 * dec["change"], name="Change in the debt ratio", mode="lines+markers",
                    line=dict(color=INK["primary"], width=2),
                    marker=dict(size=8, color=INK["primary"], line=dict(color=INK["surface"], width=2)),
                    hovertemplate="%{x}: %{y:+.2f} p.p.<extra>Change</extra>")
    fig.update_layout(barmode="relative", bargap=0.3)
    return style(fig, title, "p.p. of GDP")


def lines(series, title, ytitle, xtitle=None, height=440, markers=True, scale=100, hover_fmt=".2f", dashes=None):
    """One line per (label, pandas Series) pair, colors in fixed order."""
    fig = go.Figure()
    for k, ((label, s), color) in enumerate(zip(series.items(), PALETTE)):
        x = s.index.to_timestamp() if hasattr(s.index, "to_timestamp") else s.index
        fig.add_scatter(x=x, y=scale * s.values, name=label, mode="lines+markers" if markers else "lines",
                        line=dict(color=color, width=2, dash=(dashes or {}).get(label)), marker=dict(size=7),
                        hovertemplate="%{y:" + hover_fmt + "}<extra>" + label + "</extra>")
    fig.update_layout(hovermode="x unified")
    return style(fig, title, ytitle, xtitle, height)


def heatmap(grid, title):
    """Stabilising primary on an (r, g) grid, % of GDP, diverging around zero."""
    z = 100 * grid.values
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{100 * r:.1f}%" for r in grid.columns], y=[f"{100 * g:.1f}%" for g in grid.index],
        text=np.round(z, 1), texttemplate="%{text:.1f}", colorscale=DIVERGING, zmid=0,
        colorbar=dict(title="% of GDP"), hovertemplate="r = %{x}, g = %{y}: s* = %{z:.2f}% of GDP<extra></extra>"))
    style(fig, title, "real GDP growth", "implicit real interest rate", height=460)
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category", autorange="reversed")
    return fig


def bars(series, title, ytitle, label, scale=100, height=400):
    x = series.index.to_timestamp() if hasattr(series.index, "to_timestamp") else series.index
    fig = go.Figure(go.Bar(x=x, y=scale * series.values, marker=dict(color=PALETTE[0], line=dict(color=INK["surface"], width=1.5)),
                           hovertemplate="%{x}: %{y:+.2f}<extra>" + label + "</extra>"))
    return style(fig, title, ytitle, height=height)


def rolling_bohn(roll, title):
    x = roll.index.to_timestamp()
    fig = go.Figure()
    fig.add_scatter(x=list(x) + list(x[::-1]), y=list(roll["hi"]) + list(roll["lo"][::-1]), fill="toself",
                    fillcolor="rgba(42,120,214,0.15)", mode="lines", line=dict(width=0), hoverinfo="skip", name="90% interval (HAC)")
    fig.add_scatter(x=x, y=roll["rho"], name="rho (rolling)", mode="lines", line=dict(color=PALETTE[0], width=2),
                    hovertemplate="%{y:.3f}<extra>rho</extra>")
    if roll["benchmark"].notna().any():
        fig.add_scatter(x=x, y=roll["benchmark"], name="(r-g)/(1+g), window mean", mode="lines", line=dict(color=PALETTE[1], width=2),
                        hovertemplate="%{y:.3f}<extra>(r-g)/(1+g)</extra>")
    fig.add_hline(y=0, line=dict(color=INK["axis"], width=1))
    fig.update_layout(hovermode="x unified")
    return style(fig, title, "coefficient", "end of window")


def fan(past, fan_df, title):
    fig = go.Figure()
    fig.add_scatter(x=past.index.to_timestamp(), y=100 * past.values, name="Observed", mode="lines",
                    line=dict(color=INK["primary"], width=2), hovertemplate="%{y:.1f}%<extra>observed</extra>")
    xf = fan_df.index.to_timestamp()
    for (lo, hi), color in zip([(10, 90), (20, 80), (30, 70), (40, 60)], BLUE_RAMP[:4]):
        fig.add_scatter(x=list(xf) + list(xf[::-1]), y=list(100 * fan_df[f"p{hi}"]) + list(100 * fan_df[f"p{lo}"][::-1]),
                        fill="toself", fillcolor=color, mode="lines", line=dict(width=0), name=f"{lo}th–{hi}th percentile",
                        hoverinfo="skip")
    fig.add_scatter(x=xf, y=100 * fan_df["p50"], name="Median", mode="lines", line=dict(color=BLUE_RAMP[5], width=2),
                    hovertemplate="%{y:.1f}%<extra>median</extra>")
    fig.add_scatter(x=xf, y=100 * fan_df["baseline"], name="Baseline (VAR forecast)", mode="lines",
                    line=dict(color=PALETTE[1], width=2, dash="dash"), hovertemplate="%{y:.1f}%<extra>baseline</extra>")
    return style(fig, title, "% of GDP", height=480)


def endogenous_curves(d_grid, curves, s_max, title):
    fig = go.Figure()
    for (label, values), color in zip(curves.items(), PALETTE):
        fig.add_scatter(x=100 * d_grid, y=100 * values, name=label, mode="lines", line=dict(color=color, width=2),
                        hovertemplate="d = %{x:.0f}%: s* = %{y:.2f}%<extra>" + label + "</extra>")
    fig.add_hline(y=100 * s_max, line=dict(color=INK["muted"], dash="dot", width=1.5),
                  annotation_text=f"s_max = {100 * s_max:.1f}%", annotation_position="top left")
    return style(fig, title, "s* (% of GDP)", "debt ratio (% of GDP)")


def stacked_area(df, title, ytitle):
    fig = go.Figure()
    order = df.groupby("indexer")["share_pct"].mean().sort_values(ascending=False).index[:8]   # at most eight categories
    for name, color in zip(order, PALETTE):
        sub = df[df["indexer"] == name].sort_values("date")
        fig.add_scatter(x=sub["date"], y=sub["share_pct"], name=str(name), mode="lines", stackgroup="one",
                        line=dict(color=color, width=0.5), hovertemplate="%{y:.1f}%<extra>" + str(name) + "</extra>")
    fig.update_layout(hovermode="x unified")
    return style(fig, title, ytitle)


def target_band(targets, observed, title):
    years = targets["year"].tolist()
    fig = go.Figure()
    fig.add_scatter(x=years + years[::-1], y=targets["upper_pct_gdp"].tolist() + targets["lower_pct_gdp"].tolist()[::-1],
                    fill="toself", fillcolor="rgba(137,135,129,0.18)", mode="lines", line=dict(width=0),
                    name="Tolerance band", hoverinfo="skip")
    fig.add_scatter(x=years, y=targets["target_pct_gdp"], name="Target", mode="lines+markers",
                    line=dict(color=PALETTE[1], width=2, dash="dash"))
    fig.add_scatter(x=observed.index, y=100 * observed.values, name="Primary, federal government and BCB (SGS 5783)",
                    mode="lines+markers", line=dict(color=PALETTE[0], width=2))
    return style(fig, title, "% of GDP")
