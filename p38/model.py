"""Core model: Manning inversion and constraint layers.

SI units. All logarithms are natural. D-numbers refer to decisions in DESIGN.md.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.optimize import brentq

# ---------------------------------------------------------------- dA (D5)

def delta_area(W, H):
    """Change in cross-sectional area dA(t) relative to the lowest observed stage.

    Robust (Theil-Sen) fit W(h) = a + b (h - Hmin); dA = a*dh + b/2*dh^2.
    A negative width slope is not physical; if b < 0, b = 0 is used.
    Returns: dA, a, b
    """
    W = np.asarray(W, float)
    H = np.asarray(H, float)
    dh = H - np.nanmin(H)
    if np.ptp(dh) > 0 and len(dh) >= 3:
        b, a, _, _ = stats.theilslopes(W, dh)
    else:
        a, b = np.nanmean(W), 0.0
    b = max(b, 0.0)
    a = max(a, 1.0)
    return a * dh + 0.5 * b * dh ** 2, a, b


# ------------------------------------------------------------ Manning

def manning_logq(logA0, logn, dA, W, S):
    """log Q = -log n + 5/3 log(A0+dA) - 2/3 log W + 1/2 log S."""
    A = np.exp(logA0) + dA
    return -logn + (5.0 / 3.0) * np.log(A) - (2.0 / 3.0) * np.log(W) + 0.5 * np.log(S)


def logq_sigma(A, W, S, sA, sW, sS):
    """First-order propagation of observation error to log Q (D7)."""
    return np.sqrt((5 / 3 * sA / A) ** 2 + (2 / 3 * sW / W) ** 2 + (0.5 * sS / S) ** 2)


# ------------------------------------------------------------ data structure

@dataclass
class Reach:
    """Observations of one reach aligned to a common time index (NaN = no observation)."""
    reach_id: int
    W: np.ndarray
    H: np.ndarray
    S: np.ndarray
    sW: np.ndarray
    sH: np.ndarray
    sS: np.ndarray
    dA: np.ndarray = field(init=False)

    def __post_init__(self):
        ok = np.isfinite(self.W) & np.isfinite(self.H) & np.isfinite(self.S)
        dA = np.full_like(self.W, np.nan, dtype=float)
        if ok.sum() >= 3:
            dA[ok], _, _ = delta_area(self.W[ok], self.H[ok])
        self.dA = dA

    @property
    def Wbar(self):
        return float(np.nanmean(self.W))

    @property
    def sA(self):
        return self.W * self.sH  # uncertainty of dA from the stage error


# ------------------------------------------------------------ priors (D6)

@dataclass
class Prior:
    logn_mu: float = np.log(0.03)
    logn_sd: float = 0.4
    depth_sd: float = 0.5  # log depth
    # Downstream hydraulic geometry: W = aW Q^bW, D = aD Q^bD
    # Moody & Troutman (2002, ESPL 27:1251-1266), Eqs. 15-16, p. 1261: W = 7.2 Q^0.50, D = 0.27 Q^0.39
    # (n = 226 rivers worldwide; log standard error 0.22 for W, 0.18 for D).
    aW: float = 7.2
    bW: float = 0.50
    aD: float = 0.27
    bD: float = 0.39

    def qbar_from_width(self, W):
        return (W / self.aW) ** (1.0 / self.bW)

    def depth_from_width(self, W):
        return self.aD * self.qbar_from_width(W) ** self.bD

    # D17: mean-flow prior. If None, the width-based Qbar (SWOT only) is used.
    qbar: float | None = None

    def qbar_of(self, reach: Reach):
        return self.qbar if self.qbar is not None else self.qbar_from_width(reach.Wbar)

    def logA0_mu(self, reach: Reach):
        """D17 + D24: the prior median of A0 is solved numerically so that, at the prior median n, the mean
        Manning discharge over the observed overpasses equals Qbar:
            mean_t [ n0^-1 (A0 + dA_t)^(5/3) W_t^(-2/3) S_t^(1/2) ] = Qbar.
        (A mean-state approximation gives a mean discharge about 16% too high because the flow law is
        nonlinear - Jensen's inequality.) Qbar is the only source of magnitude information."""
        # cache stored on the reach object itself (avoids the risk of id() reuse)
        cache = reach.__dict__.setdefault("_a0_cache", {})
        key = (self.qbar, self.logn_mu)
        if key in cache:
            return cache[key]
        ok = np.isfinite(reach.W) & np.isfinite(reach.S) & np.isfinite(reach.dA) & (reach.S > 0)
        W, S, dA = reach.W[ok], reach.S[ok], reach.dA[ok]
        qbar = self.qbar_of(reach)
        n0 = np.exp(self.logn_mu)
        f = lambda la0: np.log(np.mean((np.exp(la0) + dA) ** (5 / 3) * W ** (-2 / 3) * np.sqrt(S)) / n0) - np.log(qbar)
        lo, hi = np.log(1e-3 * np.nanmean(W)), np.log(1e4 * np.nanmean(W))
        if f(lo) > 0:          # even the smallest A0 exceeds Qbar (large dA) -> lower bound
            val = lo
        else:
            val = brentq(f, lo, hi)
        cache[key] = float(val)
        return float(val)

    def logpdf(self, theta, reaches):
        K = len(reaches)
        la0, ln = theta[:K], theta[K:]
        lp = stats.norm.logpdf(ln, self.logn_mu, self.logn_sd).sum()
        for i, r in enumerate(reaches):
            lp += stats.norm.logpdf(la0[i], self.logA0_mu(r), self.depth_sd)
        return lp

    def sample(self, reaches, size, rng):
        K = len(reaches)
        la0 = np.column_stack([rng.normal(self.logA0_mu(r), self.depth_sd, size) for r in reaches])
        ln = rng.normal(self.logn_mu, self.logn_sd, (size, K))
        return np.hstack([la0, ln])


# ------------------------------------------------------------ constraints (D7, D9)

@dataclass
class Config:
    """Which constraint layers are active (C0 = all off)."""
    ahg: bool = False          # C2
    mcfli: bool = False        # C3
    qbar_prior: float | None = None   # C4: Qbar prior (m3/s), None = off
    qbar_logsd: float = 0.5
    sigma_struct: float = 0.10  # McFLI structural error (log Q)
    # D16: AHG width exponent b in the open interval (0, 1).
    # b < 1: continuity, b + f + m = 1 with f, m >= 0 (Leopold & Maddock, 1953);
    # b > 0: hydraulic consistency, width increases with discharge (Andreadis et al., 2025).
    # [0.05, 0.9] is a sensitivity run.
    ahg_b_min: float = 0.0
    ahg_b_max: float = 1.0


def reach_logq(theta, reaches):
    K = len(reaches)
    return [manning_logq(theta[i], theta[K + i], r.dA, r.W, r.S) for i, r in enumerate(reaches)]


def log_likelihood(theta, reaches, cfg: Config, prior=None):
    """D15: the error variance of log Q does not depend on the parameters - A is evaluated at the prior
    median A0. (A parameter-dependent variance rewards large A0 through the log(s2) term.)"""
    K = len(reaches)
    logqs = reach_logq(theta, reaches)
    prior = prior or Prior()
    ll = 0.0
    if cfg.mcfli and K > 1:
        for i in range(K - 1):
            a, b = reaches[i], reaches[i + 1]
            m = np.isfinite(logqs[i]) & np.isfinite(logqs[i + 1])
            if m.sum() < 3:
                continue
            Ai = np.exp(prior.logA0_mu(a)) + a.dA
            Aj = np.exp(prior.logA0_mu(b)) + b.dA
            s2 = (logq_sigma(Ai, a.W, a.S, a.sA, a.sW, a.sS) ** 2
                  + logq_sigma(Aj, b.W, b.S, b.sA, b.sW, b.sS) ** 2
                  + cfg.sigma_struct ** 2)[m]
            d = (logqs[i] - logqs[i + 1])[m]
            ll += -0.5 * np.sum(d ** 2 / s2 + np.log(2 * np.pi * s2))
    if cfg.ahg:
        for i, r in enumerate(reaches):
            m = np.isfinite(logqs[i])
            if m.sum() < 4:
                continue
            x, y = logqs[i][m], np.log(r.W[m])
            b, a = np.polyfit(x, y, 1)
            if not (cfg.ahg_b_min < b < cfg.ahg_b_max):  # D16: degenerate or non-physical solution
                return -np.inf
            res = y - (a + b * x)
            s2 = (r.sW[m] / r.W[m]) ** 2
            ll += -0.5 * np.sum(res ** 2 / s2 + np.log(2 * np.pi * s2))
    if cfg.qbar_prior is not None:
        m = np.isfinite(logqs[0])
        mean_logq = np.log(np.mean(np.exp(logqs[0][m])))
        ll += stats.norm.logpdf(mean_logq, np.log(cfg.qbar_prior), cfg.qbar_logsd)
    return ll


def log_posterior(theta, reaches, cfg: Config, prior: Prior):
    lp = prior.logpdf(theta, reaches)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, reaches, cfg, prior)
    return lp + ll if np.isfinite(ll) else -np.inf
