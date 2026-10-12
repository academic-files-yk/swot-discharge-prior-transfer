"""Twin experiment: reaches with known truth plus SWOT-like noise.

The "truth" is generated with the full hydraulic radius (R = A/P); the inversion uses the wide-channel
assumption (R = A/W), so the P = W structural error is part of the experiment.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .model import Reach


@dataclass
class TrueReach:
    B: float      # bottom width (m)
    z: float      # side slope (horizontal/vertical)
    S0: float     # mean water-surface slope
    n: float      # Manning n
    zbed: float   # bed elevation (m)
    kappa: float = 0.1  # change of slope with discharge, S = S0 (Q/Qmed)^kappa
    # D25: cross-section shape. "trapezoid" (bottom B, side slope z) or "power": W(h) = B (h / 1 m)^p (p in z)
    shape: str = "trapezoid"

    def area(self, h):
        if self.shape == "power":
            return self.B * h ** (self.z + 1) / (self.z + 1)
        return self.B * h + self.z * h ** 2

    def width(self, h):
        if self.shape == "power":
            return self.B * h ** self.z
        return self.B + 2 * self.z * h

    def perim(self, h):
        if self.shape == "power":
            return self.width(h) + 2 * h  # approximate wetted perimeter
        return self.B + 2 * h * np.sqrt(1 + self.z ** 2)

    def depth(self, Q, S):
        f = lambda h: (1 / self.n) * self.area(h) ** (5 / 3) * self.perim(h) ** (-2 / 3) * S ** 0.5 - Q
        return brentq(f, 1e-3, 500.0)


def simulate(Q, reaches_true, rng, sH=0.10, sW_rel=0.05, sS=1.7e-5, noise=1.0):
    """Generate observations of K reaches from a true Q(t) series.

    Returns: (list of Reach, true A0 array, true n array)
    """
    Q = np.asarray(Q, float)
    Qmed = np.median(Q)
    obs, A0_true, n_true = [], [], []
    for tr in reaches_true:
        S = tr.S0 * (Q / Qmed) ** tr.kappa
        h = np.array([tr.depth(q, s) for q, s in zip(Q, S)])
        W = tr.width(h)
        H = tr.zbed + h
        A0_true.append(tr.area(h.min()))
        n_true.append(tr.n)
        sW = sW_rel * W
        sSv = np.full_like(S, sS)
        sHv = np.full_like(H, sH)
        Wo = W + noise * rng.normal(0, sW)
        Ho = H + noise * rng.normal(0, sHv)
        So = S + noise * rng.normal(0, sSv)
        So = np.where(So > 0, So, np.nan)  # negative slope observations are not used (D4)
        obs.append(Reach(reach_id=len(obs), W=Wo, H=Ho, S=So, sW=sW, sH=sHv, sS=sSv))
    return obs, np.array(A0_true), np.array(n_true)
