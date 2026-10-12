"""08 - Ablation on real data (gauged chains).

Prior source: width-based Qbar (SWOT only, Prior.qbar_from_width of the target width) and, where available,
the pre-SWOT gauge climatology 2013-2022 (04c).
Chain mode (3 reaches): constraints C0 priors only | C2 AHG | C3 mass conservation (McFLI).
Single-reach mode (ArcticGRO): C0 | C2.
Evaluation at the target reach against daily gauge discharge (only overpasses with a gauge value).
Output: results/chain_ablation<tag>.csv, results/chain_q/<chain_id>_<experiment><tag>.npz (median, 5%, 95%)
Usage: python 08_chain_inversion.py [n_cores] [--ahg-min B] [--ahg-max B] [--sw-floor F] [--tag TAG]
"""
import argparse
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from p38.infer import contraction, coverage, kge_parts, nse, q_ensemble, run
from p38.model import Config, Prior, Reach
from p38.paths import ROOT  # noqa: E402

QD = ROOT / "results/chain_q"
QD.mkdir(parents=True, exist_ok=True)
_cf = ROOT / "data/climatology.csv"
CLIM = (dict(zip(pd.read_csv(_cf, dtype={"gage_id": str}).gage_id, pd.read_csv(_cf, dtype={"gage_id": str}).qbar_clim))
        if _cf.exists() else {})

MIN_NEIGHBOUR = 10  # fixed in advance: >= 10 overpasses matched to the target in every neighbouring reach
# Sensitivity options: --ahg-min/--ahg-max (D16), --sw-floor (D13 width error floor), --tag (output suffix).
# Defaults: main run.
_ap = argparse.ArgumentParser()
_ap.add_argument("ncpu", nargs="?", type=int, default=8)
_ap.add_argument("--ahg-min", type=float, default=0.0)
_ap.add_argument("--ahg-max", type=float, default=1.0)
_ap.add_argument("--sw-floor", type=float, default=0.05)
_ap.add_argument("--tag", default="")
ARG, _ = _ap.parse_known_args()


def process(pkl):
    try:
        return _process(pkl)
    except Exception as e:  # an error in one chain must not stop the run; it is recorded
        return [dict(chain_id=Path(pkl).stem, experiment="ERROR", error=f"{type(e).__name__}: {e}")]


def _process(pkl):
    D = pickle.load(open(pkl, "rb"))
    meta = D["meta"]
    valid = [int(np.sum(np.isfinite(r["W"]) & np.isfinite(r["H"]) & np.isfinite(r["S"]))) for r in D["reaches"]]
    if min(valid) < MIN_NEIGHBOUR:
        return [dict(**meta, experiment="SKIPPED", error=f"too few neighbour passes {valid}")]
    cid = meta["chain_id"]
    reaches = [Reach(reach_id=r["reach_id"], W=r["W"], H=r["H"], S=r["S"],
                     sW=np.maximum(r["sW"], ARG.sw_floor * r["W"]), sH=r["sH"], sS=r["sS"])
               for r in D["reaches"]]
    single = meta["mode"] == "single"
    hidx = 0 if single else 1
    Wmed = float(np.nanmedian(reaches[hidx].W))
    qbar_w = Prior().qbar_from_width(reaches[hidx].Wbar)
    clim = CLIM.get(str(meta["gage_id"]), np.nan)
    priors = {"width": None}                      # SWOT only: width-based Qbar
    if np.isfinite(clim):
        priors["clim"] = float(clim)              # external: pre-SWOT climatology (2013-2022)
    constraints = {"C0": Config(), "C2": Config(ahg=True, ahg_b_min=ARG.ahg_min, ahg_b_max=ARG.ahg_max)}
    if not single:
        constraints["C3"] = Config(mcfli=True)
    Q = D["Qobs"]
    K = len(reaches)
    rows = []
    for p_name, qb in priors.items():
        prior = Prior(qbar=qb)
        prior_s = run(reaches, Config(), prior, seed=cid)
        for c_name, cfg in constraints.items():
            exp = f"{p_name}_{c_name}"
            try:
                info = {}
                s = run(reaches, cfg, prior, seed=cid, nwalkers=24, nsteps=2500, burn=1000, info=info)
                ens = q_ensemble(s, reaches, idx=hidx, max_draws=1000, seed=cid)
            except Exception as e:  # numerical error -> record, continue with the others
                rows.append(dict(**meta, experiment=exp, prior=p_name, constraint=c_name,
                                 error=f"{type(e).__name__}: {e}")); continue
            qmed = np.nanmedian(ens, axis=0)
            lo, hi = np.nanpercentile(ens, [5, 95], axis=0)
            np.savez(QD / f"{cid}_{exp}{ARG.tag}.npz", t=D["t"], qmed=qmed, lo=lo, hi=hi, Qobs=Q)
            c = contraction(prior_s, s)
            rows.append(dict(**meta, experiment=exp, prior=p_name, constraint=c_name, **kge_parts(qmed, Q),
                             nse=nse(qmed, Q), coverage90=coverage(ens, Q), W_med=Wmed, qbar_w=qbar_w,
                             qbar_clim=clim, qbar_prior=prior.qbar_of(reaches[hidx]), qobs_mean=float(np.nanmean(Q)),
                             A0_med=float(np.exp(np.median(s[:, hidx]))), n_med=float(np.exp(np.median(s[:, K + hidx]))),
                             contraction_A0=float(c[hidx]), contraction_n=float(c[K + hidx]),
                             contraction_common_n=float(1 - s[:, K:].mean(axis=1).var() / prior_s[:, K:].mean(axis=1).var()),
                             acceptance=info.get("acceptance", np.nan), feasible_frac=info.get("feasible_frac", np.nan)))
    return rows


if __name__ == "__main__":
    cs = pd.read_csv(ROOT / "data/chain_summary.csv")
    ready = cs[cs.status == "READY"].chain_id.tolist()
    files = [ROOT / f"data/chain_data/{c}.pkl" for c in ready]
    print("chains to process:", len(files), flush=True)
    with Pool(ARG.ncpu) as p:
        res = p.map(process, files)
    df = pd.DataFrame([r for rr in res for r in rr])
    df["setting"] = f"ahg=({ARG.ahg_min},{ARG.ahg_max}) sW_floor={ARG.sw_floor}"
    df.to_csv(ROOT / f"results/chain_ablation{ARG.tag}.csv", index=False)
    cols = ["kge", "r", "beta", "gamma", "nse", "coverage90", "contraction_A0", "contraction_common_n"]
    print(df.groupby(["mode", "prior", "constraint"])[cols].median().round(2))
    print(df.groupby(["mode", "experiment"]).size())
