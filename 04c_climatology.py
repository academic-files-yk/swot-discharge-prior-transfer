"""04c - Pre-SWOT climatological mean discharge: 2013-01-01 .. 2022-12-31.

Does not overlap the SWOT period (2023-03 .. 2025-06) -> no leakage into the evaluation data.
Sources: USGS NWIS dv, WSC hydrometric-daily-mean, Hub'Eau QmnJ, BoM KiWIS, ANA; ArcticGRO files are local.
Rule: at least 70% of days available over the period; otherwise no climatology (NaN).
Output: data/climatology.csv (gage_id, source, qbar_clim, n_days, completeness)
"""
import io
import time

import pandas as pd
import requests

from p38.paths import ROOT  # noqa: E402
START, END = "2013-01-01", "2022-12-31"
TOTAL_DAYS = (pd.Timestamp(END) - pd.Timestamp(START)).days + 1
CFS2CMS = 0.0283168466


def usgs(site):
    url = ("https://waterservices.usgs.gov/nwis/dv/?format=rdb&parameterCd=00060&statCd=00003"
           f"&sites={site}&startDT={START}&endDT={END}")
    r = requests.get(url, timeout=300)
    lines = [l for l in r.text.splitlines() if l and not l.startswith("#")]
    if len(lines) < 3:
        return None
    df = pd.read_csv(io.StringIO("\n".join([lines[0]] + lines[2:])), sep="\t", dtype=str)
    v = [c for c in df.columns if c.endswith("_00060_00003")]
    return pd.to_numeric(df[v[0]], errors="coerce") * CFS2CMS if v else None


def wsc(st):
    vals, off = [], 0
    while True:
        url = ("https://api.weather.gc.ca/collections/hydrometric-daily-mean/items?f=json&limit=10000"
               f"&offset={off}&STATION_NUMBER={st}&datetime={START}/{END}")
        fs = requests.get(url, timeout=300).json().get("features", [])
        vals += [f["properties"].get("DISCHARGE") for f in fs]
        if len(fs) < 10000:
            break
        off += 10000
    return pd.to_numeric(pd.Series(vals), errors="coerce")


def hubeau(code):
    url = "https://hubeau.eaufrance.fr/api/v2/hydrometrie/obs_elab"
    js = requests.get(url, params=dict(code_entite=code, grandeur_hydro_elab="QmnJ", date_debut_obs_elab=START,
                                       date_fin_obs_elab=END, size=20000), timeout=300).json()
    return pd.Series([d["resultat_obs_elab"] / 1000 for d in js.get("data", []) if d.get("resultat_obs_elab") is not None])


BOM = "http://www.bom.gov.au/waterdata/services"
B = dict(service="kisters", type="queryServices", format="json")


def bom(st):
    js = requests.get(BOM, params=dict(B, request="getTimeseriesList", station_no=st,
                                       ts_name="DMQaQc.Merged.DailyMean.24HR",
                                       parametertype_name="Water Course Discharge",
                                       returnfields="ts_id,ts_unitsymbol"), timeout=300).json()
    if len(js) < 2:
        return None
    v = requests.get(BOM, params=dict(B, request="getTimeseriesValues", ts_id=js[1][0], **{"from": START, "to": END},
                                      returnfields="Timestamp,Value"), timeout=300).json()
    return pd.to_numeric(pd.Series([d[1] for d in v[0].get("data", [])]), errors="coerce")


def ana(code):
    """ANA legacy service (no registration); daily values in monthly records Vazao01..31; level 2 preferred."""
    import calendar, re as _re
    r = requests.get("https://telemetriaws1.ana.gov.br/ServiceANA.asmx/HidroSerieHistorica",
                     params=dict(codEstacao=code, dataInicio="01/01/2013", dataFim="31/12/2022", tipoDados="3",
                                 nivelConsistencia=""), timeout=600)
    rows = []
    for block in _re.findall(r"<SerieHistorica\b[^>]*>(.*?)</SerieHistorica>", r.text, flags=_re.S):
        g = dict(_re.findall(r"<(\w+)>([^<]*)</\1>", block))
        month = pd.Timestamp(g["DataHora"][:10])
        for d in range(1, calendar.monthrange(month.year, month.month)[1] + 1):
            v = g.get(f"Vazao{d:02d}", "")
            if v:
                rows.append((month.replace(day=d), float(v), int(g["NivelConsistencia"])))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["date", "q", "qual"]).sort_values(["date", "qual"]).drop_duplicates("date", keep="last")
    return df.q


def arcticgro(gid):
    st = pd.read_csv(ROOT / "data/stations_arcticgro.csv")
    riv, stn = gid.split("_", 1)
    f = st[(st.river == riv) & (st.station == stn)].file.iloc[0]
    g = pd.read_excel(ROOT / f"gauges/arcticgro/{f}")
    g["date"] = pd.to_datetime(g["date"])
    return g[(g.date >= START) & (g.date <= END)]["discharge"]


if __name__ == "__main__":
    cs = pd.read_csv(ROOT / "data/chain_summary.csv", dtype={"gage_id": str})
    cs = cs[cs.status == "READY"]
    out = ROOT / "data/climatology.csv"
    old = pd.read_csv(out, dtype={"gage_id": str}) if out.exists() else pd.DataFrame(columns=["gage_id"])
    rows = old.to_dict("records")
    for _, r in cs.iterrows():
        gid, gs = r.gage_id, r.gauge_source
        if gid in set(old.gage_id):
            continue
        try:
            s = {"usgs": usgs, "wsc": wsc, "hubeau": hubeau, "bom": bom, "ana": ana, "arcticgro": arcticgro}[gs](gid)
        except Exception:
            s = None
        n = int(s.notna().sum()) if s is not None else 0
        compl = n / TOTAL_DAYS
        q = float(s.mean()) if (s is not None and compl >= 0.70) else float("nan")
        rows.append(dict(gage_id=gid, source=gs, qbar_clim=q, n_days=n, completeness=round(compl, 3)))
        print(gid, gs, n, round(compl, 2), q, flush=True)
        time.sleep(0.3)
    pd.DataFrame(rows).drop_duplicates("gage_id").to_csv(out, index=False)
