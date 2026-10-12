"""11 - Source of every number in the text (nothing is copied by hand). Output: output/ (TABLES).

summary_twin.csv        : prior x constraint median [IQR]
summary_twin_shape.csv  : prior x constraint x cross-section shape median [IQR]
summary_chains.csv      : mode x prior x constraint median [IQR] (main run; chains passing the location check)
summary_sensitivity.csv : main run vs AHG [0.05, 0.9] vs sigma_W floor 15%
summary_numbers.json    : key numbers for the text (the transfer numbers from 12b are merged in)
"""
import json
import pickle

import numpy as np
import pandas as pd

from p38.paths import ROOT, TABLES  # noqa: E402

RES = ROOT / "results"
COLS = ["kge", "r", "beta", "gamma", "nse", "coverage90", "contraction_A0", "contraction_common_n"]


def med_iqr(x):
    x = pd.Series(x).astype(float).dropna()
    return f"{x.median():.2f} [{x.quantile(.25):.2f}, {x.quantile(.75):.2f}]" if len(x) else ""


def chain_results(tag=""):
    f = RES / f"chain_ablation{tag}.csv"
    if not f.exists():
        return None
    g = pd.read_csv(f)
    loc = pd.read_csv(ROOT / "data/gauge_location.csv")
    ok = set(loc[loc.status == "OK"].chain_id)
    g = g[g.get("prior").notna() & g.chain_id.isin(ok)] if "prior" in g else g.iloc[0:0]
    return g


if __name__ == "__main__":
    numbers = {}
    tw = pd.read_csv(RES / "twin_mc_N200.csv")
    grp = ["prior", "constraint"] + (["shape"] if "shape" in tw else [])
    tt = tw.groupby(["prior", "constraint"])[COLS].agg(med_iqr)
    tt["n"] = tw[tw.kge.notna()].groupby(["prior", "constraint"]).size()
    tt["failed"] = tw[tw.kge.isna()].groupby(["prior", "constraint"]).size().reindex(tt.index).fillna(0).astype(int)
    for c in ("acceptance", "feasible_frac"):
        if c in tw:
            tt[c] = tw.groupby(["prior", "constraint"])[c].median().round(4)
    tt.to_csv(TABLES / "summary_twin.csv")
    if "shape" in tw:
        ts = tw.groupby(grp)[["kge", "r", "beta", "coverage90"]].agg(med_iqr)
        ts.to_csv(TABLES / "summary_twin_shape.csv")
    numbers.update(twin_scenarios=int(tw.scenario.nunique()), twin_inversions=int(len(tw)),
                   twin_succeeded=int(tw.kge.notna().sum()),
                   twin_errors=int(tw.get("error", pd.Series(dtype=object)).notna().sum()),
                   twin_common_n_contraction_max_abs_median=round(float(
                       tw.groupby(["prior", "constraint"]).contraction_common_n.median().abs().max()), 3))

    g = chain_results()
    tg = g.groupby(["mode", "prior", "constraint"])[COLS].agg(med_iqr)
    tg["n"] = g[g.kge.notna()].groupby(["mode", "prior", "constraint"]).size()
    tg["failed"] = g[g.kge.isna()].groupby(["mode", "prior", "constraint"]).size().reindex(tg.index).fillna(0).astype(int)
    for c in ("acceptance", "feasible_frac"):
        if c in g:
            tg[c] = g.groupby(["mode", "prior", "constraint"])[c].median().round(4)
    tg.to_csv(TABLES / "summary_chains.csv")
    z = g[g["mode"] == "chain"]
    numbers.update(chains_width=int(z[z.prior == "width"].chain_id.nunique()),
                   chains_clim=int(z[z.prior == "clim"].chain_id.nunique()),
                   chains_per_continent=z[z.experiment == "width_C0"].groupby("continent").size().to_dict())

    sos = pd.read_csv(RES / "sos_comparison.csv")
    loc = pd.read_csv(ROOT / "data/gauge_location.csv")
    sos = sos[sos.chain_id.isin(set(loc[loc.status == "OK"].chain_id))]
    numbers.update(sos_consensus_n=int(sos.consensus_kge.notna().sum()), sos_consensus_kge=med_iqr(sos.consensus_kge),
                   sos_consensus_r=med_iqr(sos.consensus_r), sos_consensus_beta=med_iqr(sos.consensus_beta))
    same = {}
    for e in sorted(z.experiment.unique()):
        m = z[z.experiment == e].merge(sos[["chain_id", "consensus_kge"]], on="chain_id").dropna(
            subset=["kge", "consensus_kge"])
        same[e] = dict(n=len(m), ours=round(float(m.kge.median()), 2), sos=round(float(m.consensus_kge.median()), 2),
                       ours_better_frac=round(float((m.kge > m.consensus_kge).mean()), 2))
    numbers["sos_comparison_same_chains"] = same

    # D26: chains in which C2 cannot be applied (the AHG bounds are not met for any ln A0), and the paired
    # comparison of C0 and C2 on the same subset of chains
    def weakest_wh(cid):
        D = pickle.load(open(ROOT / f"data/chain_data/{int(cid)}.pkl", "rb"))
        cs = []
        for x in D["reaches"]:
            m = np.isfinite(x["W"]) & np.isfinite(x["H"])
            cs.append(np.corrcoef(x["W"][m], x["H"][m])[0, 1] if m.sum() > 3 else np.nan)
        return float(np.nanmin(cs))
    c2 = {}
    for p in ["width", "clim"]:
        d2 = z[z.experiment == f"{p}_C2"].set_index("chain_id")
        d0 = z[z.experiment == f"{p}_C0"].set_index("chain_id")
        ok2 = d2.index[d2.kge.notna()]
        wh = pd.Series({i: weakest_wh(i) for i in d2.index})
        c2[p] = dict(chains=int(len(d2)), not_applicable=int(d2.kge.isna().sum()),
                     weakest_wh_r_median_applicable=round(float(wh[ok2].median()), 2),
                     weakest_wh_r_median_not_applicable=round(float(wh[d2.index[d2.kge.isna()]].median()), 2),
                     not_applicable_negative_wh=int((wh[d2.index[d2.kge.isna()]] <= 0).sum()),
                     common_n=int(len(ok2)),
                     C0_kge=med_iqr(d0.loc[ok2].kge), C2_kge=med_iqr(d2.loc[ok2].kge),
                     C0_beta=med_iqr(d0.loc[ok2].beta), C2_beta=med_iqr(d2.loc[ok2].beta),
                     C0_r=med_iqr(d0.loc[ok2].r), C2_r=med_iqr(d2.loc[ok2].r),
                     C0_coverage=med_iqr(d0.loc[ok2].coverage90), C2_coverage=med_iqr(d2.loc[ok2].coverage90),
                     C2_feasible_frac_median=round(float(d2.loc[ok2].feasible_frac.median()), 4),
                     C2_better_kge_frac=round(float((d2.loc[ok2].kge > d0.loc[ok2].kge).mean()), 2))
    numbers["c2_applicability"] = c2
    k0 = z[z.experiment == "clim_C0"]
    numbers["clim_ratio_median"] = round(float((k0.qbar_clim / k0.qobs_mean).median()), 2)

    rows = []
    for tag, name in [("", "main: AHG (0,1), sW>=5%"), ("_ahg005_09", "AHG [0.05,0.9]"), ("_sw015", "sW>=15%")]:
        gg = chain_results(tag)
        if gg is None:
            continue
        zz = gg[gg["mode"] == "chain"]
        for (p, c), d in zz.groupby(["prior", "constraint"]):
            rows.append(dict(setting=name, prior=p, constraint=c, n=int(d.kge.notna().sum()),
                             failed=int(d.kge.isna().sum()), kge=d.kge.median(), r=d.r.median(),
                             beta=d.beta.median(), coverage90=d.coverage90.median()))
    if rows:
        pd.DataFrame(rows).round(3).to_csv(TABLES / "summary_sensitivity.csv", index=False)

    tf = TABLES / "transfer_numbers.json"
    if tf.exists():
        numbers["transfer"] = json.load(open(tf))
    json.dump(numbers, open(TABLES / "summary_numbers.json", "w"), indent=1, ensure_ascii=False, default=float)
    print(json.dumps(numbers, indent=1, ensure_ascii=False, default=float))
    print(tg[["kge", "r", "beta", "coverage90", "n"]].to_string())
