"""12a - Data extraction for the prior-error transfer analysis.

For all candidate gauge-reach pairs that pass the location check (05c, 'OK'):
  * SoS V3 (permitted groups, D18): consensus/consensus_q + consensus/time_int; algorithm series
    (metroman/allq, neobam/q/q, hivdi/Q, momma/Q, sad/Qa, sic4dvar/Q_da; if the length equals that of
    reaches/time); operational mean-flow prior momma/input_Qm_prior (guide: "prior calibration target hosted
    in SWORD").
  * SWOT RiverSP 2.0 reach observations (Hydrocron; downloaded if missing) -> WSE and width for the hydraulic
    consistency check.
Output: data/transfer/<reach_id>_<gage_id>.npz, data/transfer_meta.csv
"""
import io
import time

import earthaccess
import h5py
import numpy as np
import pandas as pd
import requests

from p38.paths import ROOT  # noqa: E402

OUT = ROOT / "data/transfer"
OUT.mkdir(parents=True, exist_ok=True)
FILL = -1e11
ALG = {"metroman": "metroman/allq", "neobam": "neobam/q/q", "hivdi": "hivdi/Q", "momma": "momma/Q",
       "sad": "sad/Qa", "sic4dvar": "sic4dvar/Q_da"}
HC = "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries"
FIELDS = ("reach_id,time_str,wse,wse_u,width,width_u,slope,slope_u,slope2,slope2_u,d_x_area,d_x_area_u,"
          "reach_q,reach_q_b,dark_frac,ice_clim_f,xovr_cal_q,n_good_nod,obs_frac_n,partial_f,p_width,p_length")


def clean(s):
    s = np.asarray(s, float)
    s[(s < FILL) | (s <= 0)] = np.nan
    return s


def to_time(sec):
    sec = np.asarray(sec, float)
    return np.asarray(pd.to_datetime(np.where(sec > 0, sec, np.nan), unit="s", origin="2000-01-01", errors="coerce"),
                      "datetime64[s]")


def hydrocron(rid):
    f = ROOT / f"swot/reach/{rid}.csv"
    if f.exists():
        return "exists"
    for attempt in range(4):
        try:
            r = requests.get(HC, params=dict(feature="Reach", feature_id=str(rid), start_time="2023-03-01T00:00:00Z",
                                             end_time="2025-06-30T00:00:00Z", output="csv", fields=FIELDS), timeout=300)
            if r.status_code == 400 and "not found" in r.text:
                return "not found"
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.json()["results"]["csv"]))
            df = df[df.time_str != "no_data"]
            df = df[[c for c in df.columns if not c.endswith("_units")]].replace(-999999999999.0, np.nan)
            df.to_csv(f, index=False)
            return f"{len(df)}"
        except Exception:
            time.sleep(5 * (attempt + 1))
    return "error"


if __name__ == "__main__":
    loc = pd.read_csv(ROOT / "data/gauge_location_all.csv", dtype={"gage_id": str})
    loc = loc[loc.status == "OK"]
    earthaccess.login(strategy="netrc")
    grans = earthaccess.search_data(short_name="SWOT_L4_HR_DAWG_SOS_DISCHARGE_V3", count=10)
    meta = []
    for continent, grp in loc.groupby("continent"):
        H = h5py.File(earthaccess.open([g for g in grans if g["umm"]["GranuleUR"].startswith(f"{continent}_")])[0], "r")
        rid_all = H["reaches/reach_id"][:]
        for _, r in grp.iterrows():
            w = np.where(rid_all == r.reach_id)[0]
            if len(w) == 0:
                meta.append(dict(continent=continent, reach_id=r.reach_id, gage_id=r.gage_id, status="not in SoS"))
                continue
            i = int(w[0])
            d = dict(t_consensus=to_time(H["consensus/time_int"][i]), consensus=clean(H["consensus/consensus_q"][i]),
                     t_reach=to_time(H["reaches/time"][i]))
            mismatch = []
            for a, v in ALG.items():
                try:
                    s = clean(H[v][i])
                except Exception:
                    continue
                if len(s) == len(d["t_reach"]):
                    d[a] = s
                else:
                    mismatch.append(a)
            qm = float(H["momma/input_Qm_prior"][i])
            d["qm_prior"] = np.array(qm if qm > 0 else np.nan)
            np.savez(OUT / f"{r.reach_id}_{r.gage_id}.npz", **d)
            hc = hydrocron(int(r.reach_id))
            meta.append(dict(continent=continent, reach_id=r.reach_id, gage_id=r.gage_id, source=r.source,
                             km_reach=r.km_reach, qm_prior=d["qm_prior"].item(),
                             n_consensus=int(np.isfinite(d["consensus"]).sum()),
                             algorithms=",".join(a for a in ALG if a in d), length_mismatch=",".join(mismatch),
                             hydrocron=hc, status="OK"))
        print(continent, "done", flush=True)
    m = pd.DataFrame(meta)
    m.to_csv(ROOT / "data/transfer_meta.csv", index=False)
    print(m.groupby(["continent", "status"]).size())
    print("with qm_prior:", m.qm_prior.notna().sum(), "| with consensus:", (m.n_consensus > 0).sum())
