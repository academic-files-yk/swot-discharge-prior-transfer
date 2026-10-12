"""06 - SWOT RiverSP reach time series for the chain reaches (Hydrocron).

Collection: default RiverSP 2.0 (version C, SWORD v16) -> same reach ids as SoS V3.
Period 2023-03-01 .. 2025-06-30. 'no_data' rows are dropped, fill values (-999999999999) -> NaN.
Output: swot/reach/<reach_id>.csv, data/hydrocron_log.csv
"""
import io
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from p38.paths import ROOT  # noqa: E402
OUT = ROOT / "swot/reach"
OUT.mkdir(parents=True, exist_ok=True)
URL = "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries"
FIELDS = ("reach_id,time_str,wse,wse_u,width,width_u,slope,slope_u,slope2,slope2_u,d_x_area,d_x_area_u,"
          "reach_q,reach_q_b,dark_frac,ice_clim_f,xovr_cal_q,n_good_nod,obs_frac_n,partial_f,p_width,p_length")


def download(rid):
    out = OUT / f"{rid}.csv"
    if out.exists():
        return rid, "exists"
    params = dict(feature="Reach", feature_id=str(rid), start_time="2023-03-01T00:00:00Z",
                  end_time="2025-06-30T00:00:00Z", output="csv", fields=FIELDS)
    for attempt in range(4):
        try:
            r = requests.get(URL, params=params, timeout=300)
            if r.status_code == 400 and "not found" in r.text:
                return rid, "not found"
            r.raise_for_status()
            csv = r.json()["results"]["csv"]
            df = pd.read_csv(io.StringIO(csv))
            df = df[df.time_str != "no_data"]
            df = df[[c for c in df.columns if not c.endswith("_units")]]
            df = df.replace(-999999999999.0, np.nan)
            df.to_csv(out, index=False)
            return rid, f"{len(df)} passes"
        except Exception as e:  # transient network/server error -> wait and retry
            err = f"error {type(e).__name__}"
            time.sleep(5 * (attempt + 1))
    return rid, err


if __name__ == "__main__":
    z = pd.read_csv(ROOT / "data/chains.csv")
    ids = set()
    passed = z[z.reason == "PASSED"]
    for _, r in passed.iterrows():
        target = int(r.selected_reach) if (r.get("r6") is True or str(r.get("r6")) == "True") else int(r.reach_id)
        ids.update([target, int(r.up), int(r.dn)])
    ids.update(int(v) for v in z[z.source == "ArcticGRO"].reach_id)  # single-reach mode
    ids = sorted(ids)
    print("reaches:", len(ids), flush=True)
    with ThreadPoolExecutor(4) as ex:
        log = list(ex.map(download, ids))
    pd.DataFrame(log, columns=["reach_id", "status"]).to_csv(ROOT / "data/hydrocron_log.csv", index=False)
    s = pd.Series([d.split()[0] if d[0].isdigit() else d for _, d in log])
    print(s.apply(lambda v: "with passes" if v.isdigit() else v).value_counts())
