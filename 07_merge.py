"""07 - Assembly of the chain data sets.

For each passing chain (upstream, target, downstream):
  * D4 quality filter applied to each reach observation: reach_q <= 1, ice_clim_f == 0,
    dark_frac <= 0.5, xovr_cal_q <= 1, slope2 > 0, width > 0, partial_f == 0
  * upstream/downstream observations are matched to the nearest overpass within +-2 h of the target overpass
  * D13 lower bounds on the uncertainties: sH >= 0.10 m, sW >= 0.05 W, sS >= 1.7e-5
  * gauge discharge: daily mean of the overpass day (UTC); chains without a gauge are recorded as such
  * at least 10 valid overpasses at the target (fixed in advance)
ArcticGRO stations: single-reach mode (target reach only).
Output: data/chain_data/<chain_id>.pkl, data/chain_summary.csv (with counts removed at each step)
"""
import pickle

import numpy as np
import pandas as pd

from p38.paths import ROOT  # noqa: E402
OUTD = ROOT / "data/chain_data"
OUTD.mkdir(parents=True, exist_ok=True)
MIN_PASSES = 10


def read_reach(rid):
    f = ROOT / f"swot/reach/{rid}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    d["t"] = pd.to_datetime(d.time_str, utc=True)
    return d


def quality_filter(d):
    n0 = len(d)
    m = ((d.reach_q <= 1) & (d.ice_clim_f == 0) & (d.dark_frac <= 0.5) & (d.xovr_cal_q <= 1)
         & (d.slope2 > 0) & (d.width > 0) & (d.partial_f == 0)
         & d[["wse", "width", "slope2"]].notna().all(axis=1))
    return d[m].copy(), n0, int(m.sum())


def match(target, neighbour, tol=pd.Timedelta("2h")):
    k = neighbour.sort_values("t")
    h = target[["t"]].sort_values("t")
    m = pd.merge_asof(h, k, on="t", direction="nearest", tolerance=tol, suffixes=("", "_k"))
    return m


def gauge_series(gid, source):
    for sub in ("usgs", "wsc", "hubeau", "bom", "ana"):
        f = ROOT / f"gauges/{sub}/{gid}.csv"
        if f.exists():
            g = pd.read_csv(f)
            g["date"] = pd.to_datetime(g["date"]).dt.tz_localize(None).dt.normalize()
            return g.groupby("date")["discharge_m3s"].mean(), sub
    if source == "ArcticGRO":
        st = pd.read_csv(ROOT / "data/stations_arcticgro.csv")
        riv, stn = gid.split("_", 1)
        f = st[(st.river == riv) & (st.station == stn)].file.iloc[0]
        g = pd.read_excel(ROOT / f"gauges/arcticgro/{f}")
        g["date"] = pd.to_datetime(g["date"])
        return g.set_index("date")["discharge"], "arcticgro"
    return None, None


def sigma(d):
    return (np.maximum(d.wse_u.fillna(0), 0.10).values,
            np.maximum(d.width_u.fillna(0), 0.05 * d.width).values,
            np.maximum(d.slope2_u.fillna(0), 1.7e-5).values)


if __name__ == "__main__":
    z = pd.read_csv(ROOT / "data/chains.csv", dtype={"gage_id": str})
    summary = []
    for i, r in z.iterrows():
        single = r.source == "ArcticGRO"
        if not single and r.reason != "PASSED":
            continue
        target_id = int(r.selected_reach) if str(r.get("r6")) == "True" else int(r.reach_id)
        ids = [target_id] if single else [int(r.up), target_id, int(r.dn)]
        rec = dict(chain_id=i, source=r.source, continent=r.continent, gage_id=r.gage_id, target=target_id,
                   mode="single" if single else "chain")
        raw = [read_reach(x) for x in ids]
        if any(h is None for h in raw):
            summary.append(dict(**rec, status="no SWOT data")); continue
        filt = [quality_filter(h) for h in raw]
        for j, (_, n0, n1) in enumerate(filt):
            rec[f"n_raw_{j}"], rec[f"n_filtered_{j}"] = n0, n1
        H = filt[len(ids) // 2][0] if not single else filt[0][0]
        if len(H) < MIN_PASSES:
            summary.append(dict(**rec, status=f"< {MIN_PASSES} passes at target")); continue
        H = H.sort_values("t").reset_index(drop=True)
        series = []
        for j, x in enumerate(ids):
            d = filt[j][0]
            m = H[["t"]].assign(**{c: H[c] for c in d.columns if c != "t"}) if x == target_id else match(H, d)
            sH, sW, sS = sigma(m)
            series.append(dict(reach_id=x, W=m.width.values, H=m.wse.values, S=m.slope2.values,
                               sW=sW, sH=sH, sS=sS, p_width=float(np.nanmedian(m.p_width))))
        q, gs = gauge_series(r.gage_id, r.source)
        day = H.t.dt.tz_convert(None).dt.normalize()
        Qobs = q.reindex(day).values if q is not None else np.full(len(H), np.nan)
        rec.update(gauge_source=gs, n_passes=len(H), n_gauge=int(np.isfinite(Qobs).sum()))
        with open(OUTD / f"{i}.pkl", "wb") as fh:
            pickle.dump(dict(meta=rec, t=H.t.values, Qobs=Qobs, reaches=series), fh)
        summary.append(dict(**rec, status="READY" if rec["n_gauge"] >= MIN_PASSES else "too few gauge matches"))
    cs = pd.DataFrame(summary)
    cs.to_csv(ROOT / "data/chain_summary.csv", index=False)
    print(cs.groupby(["mode", "status"]).size())
    print(cs[cs.status == "READY"].groupby(["continent", "gauge_source"]).size())
