"""07b - Operational SoS V3 (unconstrained) discharge at the chains.

D18: the SWOT Discharge Product Quick Guide v3 (10 Dec 2025) states "DO NOT USE DATA FROM THE FOLLOWING
GROUPS: MOI, offline, postdiagnostics, validation - experimental and incorrect in this version".
Therefore: discharge = consensus/consensus_q, time = consensus/time_int (s since 2000-01-01);
algorithm series from their own groups (metroman/allq, neobam/q/q, hivdi/Q, momma/Q, sad/Qa, sic4dvar/Q_da)
when their length equals that of reaches/time. A0/n: metroman, sad, sic4dvar, hivdi groups.
MOI/offline/validation are NOT used.
KGE against the same gauge, on the SoS overpass days.
Output: results/sos_comparison.csv, results/sos_q/<chain_id>.npz
"""
import earthaccess
import h5py
import numpy as np
import pandas as pd

from p38.infer import kge_parts, nse
from p38.paths import ROOT  # noqa: E402

SD = ROOT / "results/sos_q"
SD.mkdir(parents=True, exist_ok=True)
FILL = -1e11
ALG = {"metroman": "metroman/allq", "neobam": "neobam/q/q", "hivdi": "hivdi/Q", "momma": "momma/Q",
       "sad": "sad/Qa", "sic4dvar": "sic4dvar/Q_da"}
PARAM = {"metroman_A0": "metroman/A0hat", "metroman_n": "metroman/nahat", "sad_A0": "sad/A0", "sad_n": "sad/n",
         "sic4dvar_A0": "sic4dvar/A0", "sic4dvar_n": "sic4dvar/n", "hivdi_A0": "hivdi/A0"}


def gauge(gid):
    for sub in ("usgs", "wsc", "hubeau", "bom", "ana"):
        f = ROOT / f"gauges/{sub}/{gid}.csv"
        if f.exists():
            g = pd.read_csv(f)
            g["date"] = pd.to_datetime(g["date"]).dt.tz_localize(None).dt.normalize()
            return g.groupby("date")["discharge_m3s"].mean()
    return None


def to_time(sec):
    sec = np.asarray(sec, float)
    t = pd.to_datetime(np.where(sec > 0, sec, np.nan), unit="s", origin="2000-01-01", errors="coerce")
    return pd.DatetimeIndex(t)


def clean(s):
    s = np.asarray(s, float)
    s[(s < FILL) | (s <= 0)] = np.nan
    return s


if __name__ == "__main__":
    earthaccess.login(strategy="netrc")
    cs = pd.read_csv(ROOT / "data/chain_summary.csv", dtype={"gage_id": str})
    z = pd.read_csv(ROOT / "data/chains.csv", dtype={"gage_id": str})
    ready = cs[(cs.status == "READY") & (cs["mode"] == "chain")]
    grans = earthaccess.search_data(short_name="SWOT_L4_HR_DAWG_SOS_DISCHARGE_V3", count=10)
    rows = []
    for continent, grp in ready.groupby("continent"):
        gk = [g for g in grans if g["umm"]["GranuleUR"].startswith(f"{continent}_")]
        H = h5py.File(earthaccess.open(gk)[0], "r")
        rid_all = H["reaches/reach_id"][:]
        for _, r in grp.iterrows():
            sos_rid = int(z.loc[r.chain_id, "reach_id"])
            w = np.where(rid_all == sos_rid)[0]
            if len(w) == 0:
                continue
            i = int(w[0])
            q = gauge(r.gage_id)
            rec = dict(chain_id=r.chain_id, continent=continent, gage_id=r.gage_id, sos_reach=sos_rid, target=r.target)
            for k, v in PARAM.items():
                x = float(H[v][i]); rec[k] = x if x > FILL else np.nan
            rec_npz = {}
            # consensus: its own time axis
            tc = to_time(H["consensus/time_int"][i])
            qc = clean(H["consensus/consensus_q"][i])
            Qobs_c = q.reindex(tc.normalize()).values if q is not None else np.full(len(tc), np.nan)
            rec_npz.update(t_consensus=np.asarray(tc, "datetime64[s]"), consensus=qc, Qobs_consensus=Qobs_c)
            m = np.isfinite(qc) & np.isfinite(Qobs_c)
            rec["consensus_n_matched"] = int(m.sum())
            if m.sum() >= 5:
                kp = kge_parts(qc, Qobs_c)
                rec.update({f"consensus_{k}": kp[k] for k in ("kge", "r", "beta", "gamma")})
                rec["consensus_nse"] = nse(qc, Qobs_c)
            # algorithms: only if the length equals that of reaches/time
            tr = to_time(H["reaches/time"][i])
            Qobs_r = q.reindex(tr.normalize()).values if q is not None else np.full(len(tr), np.nan)
            rec_npz.update(t_reach=np.asarray(tr, "datetime64[s]"), Qobs_reach=Qobs_r)
            for a, v in ALG.items():
                try:
                    s = clean(H[v][i])
                except Exception:
                    continue
                if len(s) != len(tr):
                    rec[f"{a}_length_mismatch"] = True
                    continue
                rec_npz[a] = s
                mm = np.isfinite(s) & np.isfinite(Qobs_r)
                if mm.sum() >= 5:
                    kp = kge_parts(s, Qobs_r)
                    rec.update({f"{a}_{k}": kp[k] for k in ("kge", "r", "beta", "gamma")})
            np.savez(SD / f"{r.chain_id}.npz", **rec_npz)
            rows.append(rec)
        print(continent, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/sos_comparison.csv", index=False)
    cols = [c for c in df.columns if c.endswith(("_kge", "_beta", "_r"))]
    print(df[cols].median().round(2))
    print("chains with consensus KGE:", df.get("consensus_kge", pd.Series(dtype=float)).notna().sum(), "/", len(df))
