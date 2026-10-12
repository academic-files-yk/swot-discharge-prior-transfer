"""12b - Prior-error transfer: the Frasson et al. (2021) metric applied to real SWOT data (D20-D23).

Input: data/transfer/*.npz (12a), data/transfer_meta.csv, gauge series, swot/reach/*.csv
Output (TABLES): transfer_pairs.csv (pair x product rows), transfer_summary.csv (product x subset),
transfer_numbers.json
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from p38.paths import ROOT, TABLES  # noqa: E402

PRODUCTS = ["consensus", "metroman", "neobam", "hivdi", "momma", "sad", "sic4dvar"]
MIN_MATCHED = 10
NBOOT = 2000
SEED = 20261008
FRASSON = {"neobam": 0.85, "metroman": 0.73, "hivdi": 0.84, "sad": 0.53}  # BAM -> neoBAM


def gauge(gid):
    for sub in ("usgs", "wsc", "hubeau", "bom", "ana"):
        f = ROOT / f"gauges/{sub}/{gid}.csv"
        if f.exists():
            g = pd.read_csv(f)
            g["date"] = pd.to_datetime(g["date"]).dt.tz_localize(None).dt.normalize()
            return g.groupby("date")["discharge_m3s"].mean()
    return None


def hydraulic_consistency(rid):
    f = ROOT / f"swot/reach/{rid}.csv"
    if not f.exists():
        return np.nan, np.nan, 0
    d = pd.read_csv(f)
    m = ((d.reach_q <= 1) & (d.ice_clim_f == 0) & (d.dark_frac <= 0.5) & (d.xovr_cal_q <= 1)
         & (d.partial_f == 0) & d[["wse", "width"]].notna().all(axis=1))
    d = d[m]
    pw = float(np.nanmedian(d.p_width)) if len(d) else np.nan
    if len(d) < 5 or d.wse.std() == 0 or d.width.std() == 0:
        return np.nan, pw, len(d)
    return float(np.corrcoef(d.wse, d.width)[0, 1]), pw, len(d)


def pair_rows():
    meta = pd.read_csv(ROOT / "data/transfer_meta.csv", dtype={"gage_id": str})
    meta = meta[meta.status == "OK"]
    rows = []
    for _, m in meta.iterrows():
        D = np.load(ROOT / f"data/transfer/{m.reach_id}_{m.gage_id}.npz", allow_pickle=True)
        q = gauge(m.gage_id)
        if q is None:
            continue
        hc_r, pw, n_swot = hydraulic_consistency(int(m.reach_id))
        qm = float(D["qm_prior"])
        for u in PRODUCTS:
            if u not in D.files:
                continue
            t = pd.DatetimeIndex(D["t_consensus"] if u == "consensus" else D["t_reach"])
            est = np.asarray(D[u], float)
            obs = q.reindex(t.normalize()).values
            ok = np.isfinite(est) & np.isfinite(obs) & (obs > 0)
            if ok.sum() < MIN_MATCHED:
                continue
            e, o = est[ok], obs[ok]
            qo = o.mean()
            rows.append(dict(continent=m.continent, reach_id=m.reach_id, gage_id=m.gage_id, product=u, n=int(ok.sum()),
                             qobs_mean=qo, qest_mean=e.mean(), qm_prior=qm,
                             nbias=(e.mean() - qo) / qo, nbias_prior=(qm - qo) / qo if np.isfinite(qm) else np.nan,
                             log_beta=np.log(e.mean() / qo), log_prior=np.log(qm / qo) if qm > 0 else np.nan,
                             r=float(np.corrcoef(e, o)[0, 1]) if e.std() > 0 and o.std() > 0 else np.nan,
                             hydraulic_r=hc_r, p_width=pw, n_swot=n_swot))
    return pd.DataFrame(rows)


def slope(d, x, y):
    if len(d) < 5:
        return dict(n=len(d))
    X, Y = d[x].values, d[y].values
    b, a = np.polyfit(X, Y, 1)
    ts = stats.theilslopes(Y, X)[0]
    return dict(n=len(d), n_gauge=d.gage_id.nunique(), slope=b, intercept=a, r=np.corrcoef(X, Y)[0, 1], theil_sen=ts)


def cluster_bootstrap(d, x, y, rng):
    gs = d.gage_id.unique()
    idx = {g: np.where(d.gage_id.values == g)[0] for g in gs}
    slopes, better = [], []
    for _ in range(NBOOT):
        sel = rng.choice(gs, len(gs), replace=True)
        ii = np.concatenate([idx[g] for g in sel])
        X, Y = d[x].values[ii], d[y].values[ii]
        if np.ptp(X) == 0:
            continue
        slopes.append(np.polyfit(X, Y, 1)[0])
        better.append(np.mean(np.abs(d.nbias.values[ii]) < np.abs(d.nbias_prior.values[ii])))
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return q(slopes), q(better)


def subset_summary(df, rng):
    out = []
    subsets = {"all": lambda d: d,
               "hydraulically_consistent": lambda d: d[d.hydraulic_r > 0],
               "hydraulically_inconsistent": lambda d: d[d.hydraulic_r <= 0]}
    for k in ["na", "eu", "oc", "sa"]:
        subsets[f"continent_{k}"] = (lambda kk: (lambda d: d[d.continent == kk]))(k)
    for name, (lo, hi) in {"w<200": (0, 200), "w200-500": (200, 500), "w>500": (500, 1e9)}.items():
        subsets[name] = (lambda a, b: (lambda d: d[(d.p_width >= a) & (d.p_width < b)]))(lo, hi)
    for u in PRODUCTS:
        du = df[(df["product"] == u)].dropna(subset=["nbias", "nbias_prior"])
        for sname, f in subsets.items():
            d = f(du)
            if len(d) < 8:
                continue
            res = dict(product=u, subset=sname, **slope(d, "nbias_prior", "nbias"))
            dl = d.dropna(subset=["log_beta", "log_prior"])
            lg = slope(dl, "log_prior", "log_beta")
            res.update(log_slope=lg.get("slope"), log_r=lg.get("r"))
            res["improved_frac"] = float(np.mean(np.abs(d.nbias) < np.abs(d.nbias_prior)))
            res["r_median"] = float(d.r.median())
            res["abs_nbias_median"] = float(d.nbias.abs().median())
            res["abs_nbias_prior_median"] = float(d.nbias_prior.abs().median())
            if sname in ("all", "hydraulically_consistent"):
                (s_lo, s_hi), (i_lo, i_hi) = cluster_bootstrap(d, "nbias_prior", "nbias", rng)
                res.update(slope_ci_lo=s_lo, slope_ci_hi=s_hi, improved_ci_lo=i_lo, improved_ci_hi=i_hi)
                # paired test (D23): |nbias_product| - |nbias_prior| (negative = product less biased than prior)
                diff = (d.nbias.abs() - d.nbias_prior.abs()).values
                w = stats.wilcoxon(diff)
                bs = [np.median(rng.choice(diff, len(diff), replace=True)) for _ in range(NBOOT)]
                res.update(diff_median=float(np.median(diff)), diff_ci_lo=float(np.percentile(bs, 2.5)),
                           diff_ci_hi=float(np.percentile(bs, 97.5)), wilcoxon_p=float(w.pvalue))
            res["frasson_synthetic_slope"] = FRASSON.get(u)
            out.append(res)
    return pd.DataFrame(out)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    df = pair_rows()
    df.to_csv(TABLES / "transfer_pairs.csv", index=False)
    sm = subset_summary(df, rng)
    sm.to_csv(TABLES / "transfer_summary.csv", index=False)
    allp = sm[sm.subset == "all"]
    cons = df[df["product"] == "consensus"]
    numbers = dict(pairs=int(df[["reach_id", "gage_id"]].drop_duplicates().shape[0]),
                   gauges=int(df.gage_id.nunique()),
                   rows_per_product=df.groupby("product").size().to_dict(),
                   pairs_per_continent=cons.groupby("continent").size().to_dict(),
                   hydraulically_consistent_frac=float((cons.hydraulic_r > 0).mean()))
    json.dump(numbers, open(TABLES / "transfer_numbers.json", "w"), indent=1, ensure_ascii=False, default=float)
    pd.set_option("display.width", 220)
    print(json.dumps(numbers, indent=1, default=float))
    print(allp[["product", "n", "n_gauge", "slope", "slope_ci_lo", "slope_ci_hi", "r", "theil_sen", "log_slope",
                "improved_frac", "r_median", "abs_nbias_median", "abs_nbias_prior_median",
                "frasson_synthetic_slope"]].round(2).to_string())
    print(sm[sm["product"] == "consensus"][["subset", "n", "slope", "r", "improved_frac", "r_median"]].round(2).to_string())
