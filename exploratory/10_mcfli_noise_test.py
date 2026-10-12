"""10 - Diagnosis of the McFLI A0 bias: noise level 0 -> 1 (twin experiment; exploratory).

Hypothesis: under dA noise the McFLI likelihood prefers large A0 (as A0 grows, dA noise between reaches is damped
in the log Q difference) -> A0 and Q magnitude systematically high.
Expectation: A0_ratio ~ 1 at noise = 0, > 1 as the noise increases.
Output: results/mcfli_noise.csv
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

from p38.infer import kge_parts, q_ensemble, run
from p38.model import Config, Prior
from p38.paths import ROOT  # noqa: E402
from p38.synth import TrueReach, simulate

g = pd.read_excel(ROOT / "gauges/arcticgro/Lena_Kyusyur_Version_20260810.xlsx")
g["date"] = pd.to_datetime(g["date"])
g = g[(g.date >= "2023-05-01") & (g.date <= "2025-10-31") & g.date.dt.month.between(5, 10)]
Q = g.set_index("date")["discharge"].dropna().iloc[::10].values
TRUTH = [TrueReach(B=1800, z=40, S0=3.0e-5, n=0.028, zbed=10.0),
         TrueReach(B=2200, z=30, S0=2.6e-5, n=0.031, zbed=9.7),
         TrueReach(B=1500, z=50, S0=3.4e-5, n=0.026, zbed=9.4)]


def one(args):
    noise, sig, rep = args
    rng = np.random.default_rng(100 * rep + int(noise * 10))
    reaches, A0t, nt = simulate(Q, TRUTH, rng, noise=noise)
    s = run(reaches, Config(mcfli=True, sigma_struct=sig), Prior(), seed=rep, nwalkers=24, nsteps=2000, burn=800)
    ens = q_ensemble(s, reaches, idx=0, max_draws=800, seed=rep)
    kp = kge_parts(np.median(ens, axis=0), Q)
    return dict(noise=noise, sigma_struct=sig, rep=rep, A0_ratio=float(np.exp(np.median(s[:, 0])) / A0t[0]),
                n_ratio=float(np.exp(np.median(s[:, 3])) / nt[0]), beta=kp["beta"], r=kp["r"])


if __name__ == "__main__":
    jobs = [(nz, sg, t) for nz in (0.0, 0.25, 0.5, 1.0) for sg in (0.02, 0.10, 0.30) for t in range(4)]
    with Pool(10) as p:
        rows = p.map(one, jobs)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/mcfli_noise.csv", index=False)
    print(df.groupby(["noise", "sigma_struct"])[["A0_ratio", "n_ratio", "beta", "r"]].median().round(2))
