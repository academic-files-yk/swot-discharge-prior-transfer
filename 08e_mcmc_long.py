"""08e - Sensitivity to chain length: main setting (2500 steps, 1000 burn-in) versus a long run
(7000 steps, 2000 burn-in).

The same 20 chains as in 08d (same seed), width-based and climatology priors x constraints C2 and C3. KGE, beta,
90% coverage against the gauge at the target reach and the contraction of the common friction scale are computed
for both settings; the median and maximum of the differences are reported.
Output (TABLES): mcmc_long.csv
"""
import pickle
from multiprocessing import Pool

import numpy as np
import pandas as pd

from p38.infer import NoFeasibleStart, coverage, kge_parts, q_ensemble, run
from p38.model import Config, Prior, Reach
from p38.paths import ROOT, TABLES  # noqa: E402

N_CHAINS, SEED = 20, 20261008
SETTINGS = {"main": dict(nsteps=2500, burn=1000), "long": dict(nsteps=7000, burn=2000)}
_cf = ROOT / "data/climatology.csv"
CLIM = dict(zip(pd.read_csv(_cf, dtype={"gage_id": str}).gage_id, pd.read_csv(_cf, dtype={"gage_id": str}).qbar_clim))


def compute(cid):
    D = pickle.load(open(ROOT / f"data/chain_data/{cid}.pkl", "rb"))
    reaches = [Reach(reach_id=r["reach_id"], W=r["W"], H=r["H"], S=r["S"], sW=np.maximum(r["sW"], 0.05 * r["W"]),
                     sH=r["sH"], sS=r["sS"]) for r in D["reaches"]]
    K, Q = len(reaches), D["Qobs"]
    priors = {"width": None}
    k = CLIM.get(str(D["meta"]["gage_id"]), np.nan)
    if np.isfinite(k):
        priors["clim"] = float(k)
    rows = []
    for p_name, qb in priors.items():
        prior = Prior(qbar=qb)
        ps = run(reaches, Config(), prior, seed=cid)
        for c_name, cfg in {"C2": Config(ahg=True), "C3": Config(mcfli=True)}.items():
            for s_name, a in SETTINGS.items():
                try:
                    s = run(reaches, cfg, prior, seed=cid, nwalkers=24, **a)
                except NoFeasibleStart:  # C2 not applicable in this chain
                    break
                ens = q_ensemble(s, reaches, idx=1, max_draws=1000, seed=cid)
                kp = kge_parts(np.nanmedian(ens, axis=0), Q)
                dn = 1 - s[:, K:].mean(axis=1).var() / ps[:, K:].mean(axis=1).var()
                rows.append(dict(chain_id=cid, prior=p_name, constraint=c_name, setting=s_name, kge=kp["kge"],
                                 beta=kp["beta"], coverage90=coverage(ens, Q), contraction_common_n=dn))
    return rows


if __name__ == "__main__":
    g = pd.read_csv(ROOT / "results/chain_ablation.csv")
    loc = pd.read_csv(ROOT / "data/gauge_location.csv")
    ok = set(loc[loc.status == "OK"].chain_id)
    cand = sorted(set(g[(g["mode"] == "chain") & g.chain_id.isin(ok) & g.prior.notna()].chain_id))
    sel = np.random.default_rng(SEED).choice(cand, N_CHAINS, replace=False)
    with Pool(8) as p:
        res = p.map(compute, [int(c) for c in sel])
    d = pd.DataFrame([r for rr in res for r in rr])
    d.to_csv(TABLES / "mcmc_long.csv", index=False)
    metrics = ["kge", "beta", "coverage90", "contraction_common_n"]
    w = d.pivot_table(index=["chain_id", "prior", "constraint"], columns="setting", values=metrics)
    for m in metrics:
        f = (w[(m, "long")] - w[(m, "main")]).abs()
        print(f"{m:22s} |diff| median {f.median():.3f}  max {f.max():.3f}")
    print(d.groupby(["prior", "constraint", "setting"])[metrics].median().round(2))
