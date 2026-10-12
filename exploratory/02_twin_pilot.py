"""02 - Twin experiment pilot (exploratory; not used for the results in the paper).

True Q: ArcticGRO Lena at Kyusyur, daily discharge, open-water season 2023-05 .. 2025-10 (May-October),
sampled about every 10 days (close to the SWOT sampling frequency at high latitude).
3 consecutive trapezoidal reaches (known A0, n) -> SWOT-like noise -> C0, C2, C3, C3+C4 inference.
Output: results/twin_pilot.csv
"""
import numpy as np
import pandas as pd

from p38.infer import contraction, coverage, kge_parts, nse, q_ensemble, run
from p38.model import Config, Prior
from p38.paths import ROOT  # noqa: E402
from p38.synth import TrueReach, simulate

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

g = pd.read_excel(ROOT / "gauges/arcticgro/Lena_Kyusyur_Version_20260810.xlsx")
g["date"] = pd.to_datetime(g["date"])
g = g[(g.date >= "2023-05-01") & (g.date <= "2025-10-31") & g.date.dt.month.between(5, 10)]
# values after 2023 are provisional ('P'); 134 days missing in 2025 -> gaps dropped, not filled
Q = g.set_index("date")["discharge"].dropna().iloc[::10].values
print("samples:", len(Q), "Q range:", Q.min(), Q.max())

truth = [
    TrueReach(B=1800, z=40, S0=3.0e-5, n=0.028, zbed=10.0),
    TrueReach(B=2200, z=30, S0=2.6e-5, n=0.031, zbed=9.7),
    TrueReach(B=1500, z=50, S0=3.4e-5, n=0.026, zbed=9.4),
]
rng = np.random.default_rng(42)
reaches, A0_true, n_true = simulate(Q, truth, rng)
prior = Prior()
qbar_true = float(np.mean(Q))

experiments = {
    "C0_prior": Config(),
    "C2_AHG": Config(ahg=True),
    "C3_McFLI": Config(mcfli=True),
    "C3+C4_Qbar(exact)": Config(mcfli=True, qbar_prior=qbar_true),
    "C3+C4_Qbar(x2 wrong)": Config(mcfli=True, qbar_prior=2 * qbar_true),
}
prior_s = run(reaches, Config(), prior, seed=1)
rows = []
for name, cfg in experiments.items():
    s = run(reaches, cfg, prior, seed=1, nsteps=2500, burn=1000)
    ens = q_ensemble(s, reaches, idx=0)
    qmed = np.median(ens, axis=0)
    kp = kge_parts(qmed, Q)
    c = contraction(prior_s, s)
    rows.append(dict(experiment=name, **kp, nse=nse(qmed, Q), coverage90=coverage(ens, Q),
                     A0_med=float(np.exp(np.median(s[:, 0]))), A0_true=A0_true[0],
                     n_med=float(np.exp(np.median(s[:, 3]))), n_true=n_true[0],
                     contraction_A0=float(c[0]), contraction_n=float(c[3])))
    print(rows[-1])
pd.DataFrame(rows).to_csv(OUT / "twin_pilot.csv", index=False)
