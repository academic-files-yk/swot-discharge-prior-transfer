"""04b - Daily discharge for France (Hub'Eau) and Australia (BoM KiWIS) at chain gauges.

Hub'Eau hydrometrie v2 obs_elab, grandeur QmnJ (daily mean discharge, L/s) -> m3/s.
BoM Water Data Online KiWIS: station_no -> 'DMQaQc.Merged.DailyMean.24HR' time series (m3/s).
Period 2023-03-01 .. 2025-06-30. Output: gauges/hubeau/<id>.csv, gauges/bom/<id>.csv
"""
import pandas as pd
import requests

from p38.paths import ROOT  # noqa: E402
START, END = "2023-03-01", "2025-06-30"


def hubeau(code):
    out = ROOT / f"gauges/hubeau/{code}.csv"
    if out.exists():
        return "exists"
    rows, url = [], "https://hubeau.eaufrance.fr/api/v2/hydrometrie/obs_elab"
    params = dict(code_entite=code, grandeur_hydro_elab="QmnJ", date_debut_obs_elab=START,
                  date_fin_obs_elab=END, size=20000)
    r = requests.get(url, params=params, timeout=120)
    if r.status_code not in (200, 206):
        return f"http {r.status_code}"
    for d in r.json().get("data", []):
        if d.get("resultat_obs_elab") is None:  # no measurement -> skipped (not filled)
            continue
        rows.append(dict(date=d["date_obs_elab"], discharge_m3s=d["resultat_obs_elab"] / 1000.0,
                         qual=d.get("libelle_qualification")))
    if not rows:
        return "no data"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("date").to_csv(out, index=False)
    return f"{len(rows)} days"


BOM = "http://www.bom.gov.au/waterdata/services"
BASE = dict(service="kisters", type="queryServices", format="json")


def bom(st):
    out = ROOT / f"gauges/bom/{st}.csv"
    if out.exists():
        return "exists"
    p = dict(BASE, request="getTimeseriesList", station_no=st, ts_name="DMQaQc.Merged.DailyMean.24HR",
             parametertype_name="Water Course Discharge", returnfields="ts_id,ts_name,ts_unitsymbol")
    r = requests.get(BOM, params=p, timeout=120)
    js = r.json()
    if not isinstance(js, list) or len(js) < 2:
        return "no series"
    ts_id, unit = js[1][0], js[1][2]
    p = dict(BASE, request="getTimeseriesValues", ts_id=ts_id, **{"from": START, "to": END},
             returnfields="Timestamp,Value,Quality Code")
    v = requests.get(BOM, params=p, timeout=120).json()
    data = v[0]["data"] if v and "data" in v[0] else []
    if not data:
        return "no data"
    df = pd.DataFrame(data, columns=["date", "discharge_m3s", "qual"])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    if unit.strip().lower() in ("ml/d", "ml/day"):  # megalitres per day -> m3/s
        df["discharge_m3s"] = df["discharge_m3s"] * 1000 / 86400
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return f"{len(df)} days ({unit})"


if __name__ == "__main__":
    z = pd.read_csv(ROOT / "data/chains.csv", dtype={"gage_id": str})
    g = z[z.reason == "PASSED"]
    log = []
    for gid in g[g.continent == "eu"].gage_id:
        if gid.startswith("n9999"):
            continue
        try:
            s = hubeau(gid)
        except Exception as e:
            s = f"error {type(e).__name__}"
        log.append((gid, "hubeau", s)); print(gid, s, flush=True)
    for gid in g[g.continent == "oc"].gage_id:
        try:
            s = bom(gid)
        except Exception as e:
            s = f"error {type(e).__name__}"
        log.append((gid, "bom", s)); print(gid, s, flush=True)
    pd.DataFrame(log, columns=["gage_id", "source", "status"]).to_csv(ROOT / "data/gauge_download_log_eu_oc.csv",
                                                                     index=False)
