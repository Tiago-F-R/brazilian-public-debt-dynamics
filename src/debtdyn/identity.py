"""Government budget constraint in ratios to GDP: equations (2) and (3) of the course notes."""
import pandas as pd


def growth_factor(i, g, pi):
    """Nominal 'snowball' factor of equation (2): (1+i)/((1+g)(1+pi))."""
    return (1 + i) / ((1 + g) * (1 + pi))


def debt_step_nominal(d_prev, i, g, pi, s, sigma=0.0, sfa=0.0):
    """Equation (2): d_t = (1+i)/((1+g)(1+pi)) d_{t-1} - s - sigma + sfa."""
    return growth_factor(i, g, pi) * d_prev - s - sigma + sfa


def debt_step_real(d_prev, r, g, s, sigma=0.0, sfa=0.0):
    """Real form of the course: d_t = (1+r)/(1+g) d_{t-1} - s - sigma + sfa."""
    return (1 + r) / (1 + g) * d_prev - s - sigma + sfa


def real_rate(i, pi):
    return (1 + i) / (1 + pi) - 1


def implicit_rate(interest_ratio, d_prev, g, pi):
    """i_t = (J_t/Y_t)(1+g_t)(1+pi_t)/d_{t-1}, with J_t/Y_t the interest bill over the GDP of the same period."""
    return interest_ratio * (1 + g) * (1 + pi) / d_prev


def decompose_debt_change(d, d_prev, g, pi, s, interest=None, i=None, sigma=0.0):
    """Equation (3): contributions to the change in the debt ratio, the stock-flow adjustment being the residual.

    Pass either `interest` (interest bill / GDP of the same period) or the implicit nominal rate `i`.
    Scalars return a Series; pandas Series return a DataFrame with one row per period.
    """
    if (interest is None) == (i is None):
        raise ValueError("pass exactly one of `interest` or `i`")
    if interest is None:
        interest = i * d_prev / ((1 + g) * (1 + pi))
    parts = {
        "interest": interest,
        "growth": -g * d_prev / (1 + g),
        "inflation": -pi * d_prev / ((1 + g) * (1 + pi)),
        "primary": -s,
        "seigniorage": -sigma,
    }
    change = d - d_prev
    sfa = change - sum(parts.values())
    columns = {**parts, "sfa": sfa, "change": change}
    series_inputs = [v for v in (d, d_prev, g, pi, s, interest) if isinstance(v, pd.Series)]
    if series_inputs:
        index = series_inputs[0].index
        return pd.DataFrame({k: (v if isinstance(v, pd.Series) else pd.Series(v, index=index)) for k, v in columns.items()})
    return pd.Series(columns, dtype=float)
