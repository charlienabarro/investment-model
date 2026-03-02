# src/optimizer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import cvxpy as cp
except Exception:
    cp = None


@dataclass
class OptResult:
    weights: pd.Series
    status: str


def _shrink_cov(returns: pd.DataFrame, shrink: float = 0.10) -> np.ndarray:
    """
    Simple covariance shrinkage:
    Sigma_shrunk = (1-shrink)*Sigma + shrink*diag(Sigma)
    """
    x = returns.dropna(how="any")
    if x.shape[0] < 30 or x.shape[1] < 2:
        v = np.nanvar(x.values, axis=0)
        v = np.where(np.isfinite(v) & (v > 1e-12), v, 1e-4)
        return np.diag(v)

    Sigma = np.cov(x.values, rowvar=False)
    Sigma = np.nan_to_num(Sigma, nan=0.0, posinf=0.0, neginf=0.0)

    diag = np.diag(np.diag(Sigma))
    return (1.0 - shrink) * Sigma + shrink * diag


def optimise_long_only(
    tickers: List[str],
    mu: pd.Series,
    returns: pd.DataFrame,
    caps: pd.Series,
    min_w: float,
    sector_map: Dict[str, str],
    sector_caps: Dict[str, float],
    group_caps: Dict[str, float],
    prev_w: Optional[pd.Series] = None,
    risk_aversion: float = 8.0,
    turnover_penalty: float = 0.5,
    cov_shrink: float = 0.10,
    l2_penalty: float = 0.05,
) -> OptResult:
    """
    Maximise: mu'w - risk_aversion * w' Σ w - turnover_penalty * ||w - prev_w||_1
    s.t. sum w = 1, min_w <= w_i <= caps_i, w_i >= 0
         sector caps and group caps as sum constraints
    """
    idx = [t.lower() for t in tickers]
    mu = mu.reindex(idx).fillna(0.0).astype(float)
    caps = caps.reindex(idx).fillna(0.0).astype(float)

    if prev_w is None:
        prev_w = pd.Series(0.0, index=idx)
    else:
        prev_w = prev_w.reindex(idx).fillna(0.0).astype(float)

    if cp is None:
        # Fallback: simple capped normalised positive mu
        raw = mu.clip(lower=0.0)
        if float(raw.sum()) <= 0:
            raw = pd.Series(1.0, index=idx)
        w = raw / float(raw.sum())
        w = w.clip(lower=min_w)
        w = w / float(w.sum())
        w = w.clip(upper=caps)
        w = w / float(w.sum())
        return OptResult(weights=w, status="no_cvxpy_fallback")

    # Build covariance from returns
    R = returns.reindex(columns=idx).dropna(how="any")
    Sigma = _shrink_cov(R, shrink=cov_shrink)

    n = len(idx)
    w = cp.Variable(n)

    mu_vec = mu.values
    prev_vec = prev_w.values
    caps_vec = caps.values

    quad = cp.quad_form(w, Sigma)
    turnover = cp.norm1(w - prev_vec)

    l2 = cp.sum_squares(w)
    obj = cp.Maximize(mu_vec @ w - risk_aversion * quad - turnover_penalty * turnover - l2_penalty * l2)

    cons = []
    cons.append(cp.sum(w) == 1.0)
    cons.append(w >= float(min_w))
    cons.append(w <= caps_vec)

    # Sector caps
    if sector_caps:
        sectors = [sector_map.get(t, "unknown") for t in idx]
        for sec, cap in sector_caps.items():
            mask = np.array([1.0 if s == sec else 0.0 for s in sectors])
            if mask.sum() > 0:
                cons.append(mask @ w <= float(cap))

    # Group caps (like semis total, commodities total)
    if group_caps:
        for name, (members, cap) in group_caps.items():
            mem = set([m.lower() for m in members])
            mask = np.array([1.0 if t in mem else 0.0 for t in idx])
            if mask.sum() > 0:
                cons.append(mask @ w <= float(cap))

    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except Exception:
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except Exception:
            return OptResult(weights=prev_w / float(prev_w.sum()) if float(prev_w.sum()) > 0 else pd.Series(1.0 / n, index=idx), status="solve_failed")

    if w.value is None:
        return OptResult(weights=prev_w / float(prev_w.sum()) if float(prev_w.sum()) > 0 else pd.Series(1.0 / n, index=idx), status="no_solution")

    out = np.array(w.value).reshape(-1)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    out = np.maximum(out, float(min_w))
    out = np.minimum(out, caps_vec)
    s = float(out.sum())
    if s <= 0:
        out = np.ones(n) / n
    else:
        out = out / s

    return OptResult(weights=pd.Series(out, index=idx), status=str(prob.status))