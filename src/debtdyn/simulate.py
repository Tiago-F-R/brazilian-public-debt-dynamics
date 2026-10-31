"""Scenarios: stabilising primary (4), primary for a target (5), debt-dependent rate (6), intertemporal constraint (7)."""
import numpy as np
import pandas as pd

from .identity import debt_step_nominal, decompose_debt_change


def snowball(r, g):
    """lambda = (1+r)/(1+g); the slides write r_bar = lambda - 1."""
    return (1 + r) / (1 + g)


def stabilizing_primary(d, r, g, sigma=0.0):
    """Equation (4): s* = (r-g)/(1+g) d - sigma."""
    return (r - g) / (1 + g) * d - sigma


def primary_for_target(d0, d_target, n, r, g, sigma=0.0):
    """Equation (5): constant primary that takes the ratio from d0 to d_target in n periods."""
    lam = snowball(r, g)
    if np.isclose(lam, 1.0):
        return (d0 - d_target) / n - sigma
    return (lam**n * d0 - d_target) * (lam - 1) / (lam**n - 1) - sigma


def stabilizing_grid(d, r_values, g_values, sigma=0.0):
    """Equation (4) on a grid: rows are growth rates, columns are real interest rates."""
    R, G = np.meshgrid(np.asarray(r_values, float), np.asarray(g_values, float))
    return pd.DataFrame(stabilizing_primary(d, R, G, sigma),
                        index=pd.Index(list(g_values), name="g"), columns=pd.Index(list(r_values), name="r"))


def project_debt(d0, n, r, g, s, pi=0.0, sigma=0.0, sfa=0.0, phi=0.0, d_bar=None):
    """Projects the ratio for n periods with equation (2) and decomposes each step with equation (3).

    r, g, s, pi, sigma, sfa: scalars or sequences of length n (periods 1..n).
    phi > 0 makes the real rate depend on debt (equation 6): r_t = r_t^base + phi (d_{t-1} - d_bar).
    """
    def as_path(x, name):
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 0:
            return np.full(n, float(arr))
        if arr.shape != (n,):
            raise ValueError(f"{name} must be a scalar or have length {n}")
        return arr

    r_b, g_, s_, pi_, sig_, sfa_ = (as_path(x, k) for x, k in
                                    ((r, "r"), (g, "g"), (s, "s"), (pi, "pi"), (sigma, "sigma"), (sfa, "sfa")))
    d_bar = d0 if d_bar is None else d_bar
    rows = [{"period": 0, "d": d0}]
    d_prev = d0
    for t in range(n):
        r_t = r_b[t] + phi * (d_prev - d_bar)
        i_t = (1 + r_t) * (1 + pi_[t]) - 1
        d_t = debt_step_nominal(d_prev, i_t, g_[t], pi_[t], s_[t], sig_[t], sfa_[t])
        parts = decompose_debt_change(d_t, d_prev, g_[t], pi_[t], s_[t], i=i_t, sigma=sig_[t])
        rows.append({"period": t + 1, "d": d_t, "r": r_t, "i": i_t, "g": g_[t], "pi": pi_[t], "s": s_[t], **parts.to_dict()})
        d_prev = d_t
    return pd.DataFrame(rows).set_index("period")


def required_primary_endogenous(d, r_bar, g, phi, d_bar, sigma=0.0):
    """Steady-state primary when r = r_bar + phi (d - d_bar): s*(d) = (r(d) - g) d / (1+g) - sigma."""
    return (r_bar + phi * (d - d_bar) - g) * d / (1 + g) - sigma


def steady_states(s, r_bar, g, phi, d_bar, sigma=0.0):
    """Real roots of phi d^2 + (r_bar - phi d_bar - g) d - (1+g)(s+sigma) = 0, with the slope of the map at each root."""
    c = -(1 + g) * (s + sigma)
    if np.isclose(phi, 0.0):
        b = r_bar - g
        roots = [] if np.isclose(b, 0.0) else [-c / b]
    else:
        b = r_bar - phi * d_bar - g
        disc = b**2 - 4 * phi * c
        roots = [] if disc < 0 else sorted({(-b - np.sqrt(disc)) / (2 * phi), (-b + np.sqrt(disc)) / (2 * phi)})
    rows = []
    for root in roots:
        slope = (1 + r_bar + phi * (2 * root - d_bar)) / (1 + g)
        rows.append({"d": root, "slope": slope, "stable": abs(slope) < 1})
    return pd.DataFrame(rows, columns=["d", "slope", "stable"])


def debt_limit(s_max, r_bar, g, phi, d_bar, sigma=0.0):
    """Largest steady state for the primary s_max: above it no primary up to s_max stabilises the ratio."""
    ss = steady_states(s_max, r_bar, g, phi, d_bar, sigma)
    return float(ss["d"].max()) if len(ss) else np.nan


def present_value_primaries(s_path, r, g, sigma_path=0.0):
    """Equation (7), truncated at len(s_path): sum_k lambda^{-(k+1)} (s_k + sigma_k)."""
    s_path = np.asarray(s_path, dtype=float)
    sig = np.broadcast_to(np.asarray(sigma_path, dtype=float), s_path.shape)
    k = np.arange(len(s_path))
    return float(np.sum(snowball(r, g) ** -(k + 1.0) * (s_path + sig)))
