"""04 - Daily discharge for the gauges matched in SoS (North America).

USGS (8-15 digit numeric id) -> NWIS dv, parameter 00060 (ft3/s) -> m3/s (x 0.0283168466)
WSC (e.g. 05RB003)           -> api.weather.gc.ca hydrometric-daily-mean (m3/s)
Period: 2023-03-01 .. 2025-06-30 (SoS V3 coverage).
Output: gauges/usgs/<id>.csv, gauges/wsc/<id>.csv (date, discharge_m3s, qual)
"""
import io
import re
import time

import pandas as pd
import requests

from p38.paths import ROOT  # noqa: E402
START, END = "2023-03-01", "2025-06-30"
CFS2CMS = 0.0283168466


def usgs(site):
    out = ROOT / f"gauges/usgs/{site}.csv"
    if out.exists():
        return "exists"
    url = ("https://waterservices.usgs.gov/nwis/dv/?format=rdb&parameterCd=00060&statCd=00003"
           f"&sites={site}&startDT={START}&endDT={END}")
    r = requests.get(url, timeout=120)
    if r.status_code != 200:
        return f"http {r.status_code}"
    lines = [l for l in r.text.splitlines() if l and not l.startswith("#")]
    if len(lines) < 3:
        return "no data"
    df = pd.read_csv(io.StringIO("\n".join([lines[0]] + lines[2:])), sep="\t", dtype=str)
    vcol = [c for c in df.columns if c.endswith("_00060_00003")]
    if not vcol:
        return "no column"
    d = pd.DataFrame({"date": pd.to_datetime(df["datetime"]),
                      "discharge_m3s": pd.to_numeric(df[vcol[0]], errors="coerce") * CFS2CMS,
                      "qual": df[vcol[0] + "_cd"]})
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    return f"{d.discharge_m3s.notna().sum()} days"


def wsc(st):
    out = ROOT / f"gauges/wsc/{st}.csv"
    if out.exists():
        return "exists"
    url = ("https://api.weather.gc.ca/collections/hydrometric-daily-mean/items?f=json&limit=10000"
           f"&STATION_NUMBER={st}&datetime={START}/{END}")
    r = requests.get(url, timeout=120)
    if r.status_code != 200:
        return f"http {r.status_code}"
    feats = r.json().get("features", [])
    if not feats:
        return "no data"
    d = pd.DataFrame([{"date": f["properties"]["DATE"],
                       "discharge_m3s": f["properties"].get("DISCHARGE"),
                       "qual": f["properties"].get("DISCHARGE_SYMBOL_EN")} for f in feats])
    d["date"] = pd.to_datetime(d["date"])
    out.parent.mkdir(parents=True, exist_ok=True)
    d.sort_values("date").to_csv(out, index=False)
    return f"{d.discharge_m3s.notna().sum()} days"


if __name__ == "__main__":
    inv = pd.read_csv(ROOT / "data/sos_inventory_na.csv", dtype={"gage_id": str})
    ids = sorted(set(inv.loc[inv.gage_id != "NO_DATA", "gage_id"]))
    log = []
    for g in ids:
        try:
            if re.fullmatch(r"\d{8,15}", g):
                s = usgs(g); source = "usgs"
            elif re.fullmatch(r"\d{2}[A-Z]{2}\d{3}", g):
                s = wsc(g); source = "wsc"
            else:
                s, source = "unknown format", "?"
        except Exception as e:  # network error: record and continue
            s, source = f"error {type(e).__name__}", "?"
        log.append((g, source, s))
        print(g, source, s, flush=True)
        time.sleep(0.3)
    pd.DataFrame(log, columns=["gage_id", "source", "status"]).to_csv(ROOT / "data/gauge_download_log_na.csv",
                                                                     index=False)
