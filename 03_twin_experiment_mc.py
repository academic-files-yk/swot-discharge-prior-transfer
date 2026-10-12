"""03 - Monte Carlo twin experiment.

Each scenario:
  * true hydrograph: one of 8 ArcticGRO stations at random, open-water season (May-October), one of the years
    2015-2024 with >= 170 daily values; sampled every 10 days (missing days dropped, not filled)
  * K=3 consecutive reaches; half trapezoidal, half power-law cross-sections (D25); geometry, slope and n random
    (priors fixed -> prior errors arise naturally); depth solved with the full hydraulic radius (A/P)
  * SWOT-like noise (sH = 0.10 m, sW = 5%, sS = 1.7e-5)
  * experiments: Qbar prior {exact, x exp(N(0, 0.25)), x exp(N(0, 0.5)), x exp(N(0, 1.0)), width-based}
    x constraint {C0, C2, C3}  -> 15 inversions per scenario
Output: results/twin_mc_N<scenarios>.csv (scenario x experiment rows)
Usage: python 03_twin_experiment_mc.py [n_scenarios] [n_cores]
"""
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from p38.infer import contraction, coverage, kge_parts, nse, q_ensemble, run
from p38.model import Config, Prior
from p38.paths import ROOT  # noqa: E402
from p38.synth import TrueReach, simulate

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
GDIR = ROOT / "gauges/arcticgro"
STATIONS = ["Lena_Kyusyur", "Ob_Salekhard", "Yenisei_Igarka", "Kolyma_Kolymskoe",
            "Yukon_PilotStation", "Mackenzie_ArcticRedRiver", "Pechora_UstTsilma",
            "NorthernDvina_UstPenega"]


def hydrographs():
    out = []
    for s in STATIONS:
        g = pd.read_excel(GDIR / f"{s}_Version_20260810.xlsx")
        g["date"] = pd.to_datetime(g["date"])
        for year in range(2015, 2025):
            q = g[(g.date.dt.year == year) & g.date.dt.month.between(5, 10)].set_index("date")["discharge"]
            if q.notna().sum() >= 170:  # nearly complete years
                out.append((s, year, q.dropna().iloc[::10].values))
    return out


def scenario(args):
    k, (station, year, Q) = args
    rng = np.random.default_rng(1000 + k)
    Qm = float(np.mean(Q))
    Wbar = 7.2 * Qm ** 0.5 * np.exp(rng.normal(0, 0.3))   # scatter around the Q-W relation (scenario generation)
    shape = "power" if rng.random() < 0.5 else "trapezoid"   # D25: half trapezoidal, half power-law sections
    base = dict(z=rng.uniform(5, 60) if shape == "trapezoid" else rng.uniform(0.2, 1.0),
                S0=np.exp(rng.uniform(np.log(1e-5), np.log(3e-4))),
                n=np.exp(rng.uniform(np.log(0.02), np.log(0.06))))
    truth = []
    for i in range(3):
        f = np.exp(rng.normal(0, 0.2, 4))
        if shape == "trapezoid":
            B = 0.8 * Wbar * f[0]
        else:  # W(h_mean) ~ Wbar at a mean depth close to Moody's D = 0.27 Q^0.39
            h_mean = 0.27 * Qm ** 0.39
            B = Wbar * f[0] / h_mean ** base["z"]
        truth.append(TrueReach(B=B, z=base["z"] * (f[1] if shape == "trapezoid" else 1.0), S0=base["S0"] * f[2],
                               n=base["n"] * f[3], zbed=10.0 - 0.3 * i, shape=shape))
    try:
        reaches, A0_true, n_true = simulate(Q, truth, rng)
    except ValueError:
        return []
    # D17: prior quality (scale of the Qbar error) x constraint layer
    priors = {"Qexact": Qm, "Qs025": Qm * float(np.exp(rng.normal(0, 0.25))),
              "Qs050": Qm * float(np.exp(rng.normal(0, 0.50))), "Qs100": Qm * float(np.exp(rng.normal(0, 1.00))),
              "Qwidth": None}
    constraints = {"C0": Config(), "C2": Config(ahg=True), "C3": Config(mcfli=True)}
    rows = []
    for p_name, qb in priors.items():
        prior = Prior(qbar=qb)
        qb_eff = prior.qbar_of(reaches[0])
        prior_s = run(reaches, Config(), prior, seed=k)
        for c_name, cfg in constraints.items():
            try:
                info = {}
                s = run(reaches, cfg, prior, seed=k, nwalkers=24, nsteps=2000, burn=800, info=info)
                ens = q_ensemble(s, reaches, idx=0, max_draws=800, seed=k)
            except Exception as e:
                rows.append(dict(scenario=k, prior=p_name, constraint=c_name, experiment=f"{p_name}_{c_name}",
                                 shape=shape, error=f"{type(e).__name__}: {e}")); continue
            qmed = np.median(ens, axis=0)
            c = contraction(prior_s, s)
            rows.append(dict(scenario=k, station=station, year=year, prior=p_name, constraint=c_name,
                             experiment=f"{p_name}_{c_name}",
                             qbar_prior_ratio=qb_eff / Qm, **kge_parts(qmed, Q), nse=nse(qmed, Q),
                             coverage90=coverage(ens, Q),
                             A0_ratio=float(np.exp(np.median(s[:, 0])) / A0_true[0]),
                             n_ratio=float(np.exp(np.median(s[:, 3])) / n_true[0]),
                             contraction_A0=float(c[0]), contraction_n=float(c[3]),
                             contraction_common_n=float(1 - s[:, 3:].mean(axis=1).var() / prior_s[:, 3:].mean(axis=1).var()),
                             n_true=n_true[0], S0=truth[0].S0, Wbar=Wbar, shape=shape,
                             acceptance=info.get("acceptance", np.nan), feasible_frac=info.get("feasible_frac", np.nan)))
    return rows


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    ncpu = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    H = hydrographs()
    print("eligible hydrographs:", len(H))
    rng = np.random.default_rng(7)
    sel = [H[i] for i in rng.integers(0, len(H), N)]
    with Pool(ncpu) as p:
        res = p.map(scenario, list(enumerate(sel)))
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(OUT / f"twin_mc_N{N}.csv", index=False)
    print(df.groupby(["prior", "constraint"])[["kge", "r", "beta", "gamma", "coverage90", "A0_ratio",
                                               "contraction_A0", "contraction_common_n"]].median().round(2))
