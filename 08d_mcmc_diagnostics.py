"""08d - MCMC convergence diagnostics (main-run settings: 24 walkers, 2500 steps, 1000 burn-in).

20 chains are drawn with a fixed seed from the chains that pass the location check; C2 and C3 are resampled
with the width-based and climatology priors. For each run: mean acceptance fraction, integrated
autocorrelation time tau per parameter (Goodman & Weare, 2010; emcee get_autocorr_time, tol=0), and
post-burn-in steps / max(tau).
C0 is sampled directly from the prior and needs no diagnostics.
Output (TABLES): mcmc_diagnostics.csv
"""
import pickle
from multiprocessing import Pool

import emcee
import numpy as np
import pandas as pd

from p38.infer import NoFeasibleStart, feasible_start
from p38.model import Config, Prior, Reach, log_posterior
from p38.paths import ROOT, TABLES  # noqa: E402

N_CHAINS, SEED = 20, 20261008
NW, NS, BURN = 24, 2500, 1000
_cf = ROOT / "data/climatology.csv"
CLIM = dict(zip(pd.read_csv(_cf, dtype={"gage_id": str}).gage_id, pd.read_csv(_cf, dtype={"gage_id": str}).qbar_clim))


def diagnose(cid):
    D = pickle.load(open(ROOT / f"data/chain_data/{cid}.pkl", "rb"))
    reaches = [Reach(reach_id=r["reach_id"], W=r["W"], H=r["H"], S=r["S"], sW=np.maximum(r["sW"], 0.05 * r["W"]),
                     sH=r["sH"], sS=r["sS"]) for r in D["reaches"]]
    K = len(reaches)
    priors = {"width": None}
    k = CLIM.get(str(D["meta"]["gage_id"]), np.nan)
    if np.isfinite(k):
        priors["clim"] = float(k)
    rows = []
    for p_name, qb in priors.items():
        prior = Prior(qbar=qb)
        for c_name, cfg in {"C2": Config(ahg=True), "C3": Config(mcfli=True)}.items():
            rng = np.random.default_rng(cid)
            try:
                p0, _ = feasible_start(reaches, cfg, prior, NW, rng)  # D26: same initialisation as the main run
            except NoFeasibleStart:  # C2 not applicable in this chain (also counted as failed in the main run)
                rows.append(dict(chain_id=cid, prior=p_name, constraint=c_name, not_applicable=True))
                continue
            s = emcee.EnsembleSampler(NW, 2 * K, log_posterior, args=(reaches, cfg, prior))
            s.random_state = np.random.RandomState(cid).get_state()
            s.run_mcmc(p0, NS, progress=False)
            tau = s.get_autocorr_time(discard=BURN, tol=0)
            rows.append(dict(chain_id=cid, prior=p_name, constraint=c_name,
                             acceptance=float(np.mean(s.acceptance_fraction)),
                             tau_max=float(np.max(tau)), tau_median=float(np.median(tau)),
                             steps_per_tau=float((NS - BURN) / np.max(tau)),
                             ess=float(NW * (NS - BURN) / np.max(tau))))
    return rows


if __name__ == "__main__":
    g = pd.read_csv(ROOT / "results/chain_ablation.csv")
    loc = pd.read_csv(ROOT / "data/gauge_location.csv")
    ok = set(loc[loc.status == "OK"].chain_id)
    cand = sorted(set(g[(g["mode"] == "chain") & g.chain_id.isin(ok) & g.prior.notna()].chain_id))
    sel = np.random.default_rng(SEED).choice(cand, N_CHAINS, replace=False)
    with Pool(8) as p:
        res = p.map(diagnose, [int(c) for c in sel])
    d = pd.DataFrame([r for rr in res for r in rr])
    d.to_csv(TABLES / "mcmc_diagnostics.csv", index=False)
    if "not_applicable" in d:
        print("runs not applicable:", int(d.not_applicable.fillna(False).sum()))
        d = d[d.not_applicable.isna()]
    print(d.groupby(["prior", "constraint"])[["acceptance", "tau_max", "steps_per_tau", "ess"]].median().round(2))
    print("runs with steps/tau < 50:", int((d.steps_per_tau < 50).sum()), "/", len(d))
