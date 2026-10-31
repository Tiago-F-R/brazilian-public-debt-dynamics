"""Cyclically adjusted primary balance and fiscal impulse: equation (9) (Fedelino et al., 2009)."""
import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter


def output_gap_hp(real_output, lamb):
    """HP trend of log output taken as potential output; gap = Y / Y^P - 1."""
    y = real_output.dropna()
    cycle, trend = hpfilter(np.log(y), lamb=lamb)
    return pd.DataFrame({"output": y, "potential": np.exp(trend), "gap": np.exp(cycle) - 1}, index=y.index)


def cyclically_adjusted_levels(R, G, Y, YP, eps_R=1.0, eps_G=0.0):
    """Equation (9) in levels."""
    ratio = YP / Y
    R_ca, G_ca = R * ratio**eps_R, G * ratio**eps_G
    pb, capb = R - G, R_ca - G_ca
    return {"R_ca": R_ca, "G_ca": G_ca, "PB": pb, "CAPB": capb, "CPB": pb - capb}


def cyclically_adjusted_ratios(rev_ratio, exp_ratio, gap, eps_R=1.0, eps_G=0.0):
    """Equation (9) with revenue and spending as ratios to actual GDP; balances returned as ratios to potential GDP."""
    q = 1 + gap
    rev_ca = rev_ratio * q ** (1 - eps_R)
    exp_ca = exp_ratio * q ** (1 - eps_G)
    pb_pot = (rev_ratio - exp_ratio) * q
    capb = rev_ca - exp_ca
    out = pd.DataFrame({
        "pb_pct_gdp": rev_ratio - exp_ratio,
        "pb": pb_pot,
        "capb": capb,
        "cpb": pb_pot - capb,
        "cpb_first_order_pct_gdp": (eps_R * rev_ratio - eps_G * exp_ratio) * gap,
        "fiscal_impulse": -capb.diff(),
        "gap": gap,
    })
    out["stance"] = np.select([out["fiscal_impulse"] > 0, out["fiscal_impulse"] < 0], ["expansionary", "contractionary"], "neutral")
    out["cyclicality"] = np.where(np.sign(out["fiscal_impulse"]) == -np.sign(out["gap"]), "countercyclical", "procyclical")
    return out


def fiscal_impulse(capb):
    """FI = -Delta CAPB: positive means a discretionary expansion."""
    return -capb.diff()
