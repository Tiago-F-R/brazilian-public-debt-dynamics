"""Brazilian public debt dynamics: an application of Topics in Fiscal Policy (FGV EPGE, 2026.2).

Every calculation lives in this package; the dashboard (app/app.py), the notebook and the tests only call it.
Modules: identity (equations 2-3), simulate (4-7), cycle (9), bohn (8), stochastic (10),
data (ingestion, cache, verification, panels), plots (chart builders), coursetests (acceptance tests).
"""

__version__ = "0.1.0"
