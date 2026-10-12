"""04d - Daily discharge for all candidate gauged reaches in Europe (France), Oceania and South America.

Scope (fixed in advance): gauge ids in the SoS V3 candidate list.
  EU: 10-character French Hydro codes -> Hub'Eau (other countries out of scope; no open API)
  OC: BoM Water Data Online (KiWIS)
  SA: ANA (Brazil) legacy HidroSerieHistorica service (no registration).
      Daily values in monthly records Vazao01..Vazao31; NivelConsistencia 2 (consistency-checked) if available,
      otherwise 1 (raw).
  AF: South Africa DWS - no open API -> out of scope.
Period 2023-03-01 .. 2025-06-30. Output: gauges/{hubeau,bom,ana}/<id>.csv, data/gauge_download_log_all.csv
"""
import calendar
import re
import time

import pandas as pd
import requests

from p38.paths import ROOT  # noqa: E402

START, END = "2023-03-01", "2025-06-30"
ANA_URL = "https://telemetriaws1.ana.gov.br/ServiceANA.asmx/HidroSerieHistorica"


def hubeau(code):
    out = ROOT / f"gauges/hubeau/{code}.csv"
    if out.exists():
        return "exists"
    js = requests.get("https://hubeau.eaufrance.fr/api/v2/hydrometrie/obs_elab",
                      params=dict(code_entite=code, grandeur_hydro_elab="QmnJ", date_debut_obs_elab=START,
                                  date_fin_obs_elab=END, size=20000), timeout=180).json()
    rows = [dict(date=d["date_obs_elab"], discharge_m3s=d["resultat_obs_elab"] / 1000.0,
                 qual=d.get("libelle_qualification")) for d in js.get("data", []) if d.get("resultat_obs_elab") is not None]
    if not rows:
        return "no data"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("date").to_csv(out, index=False)
    return f"{len(rows)} days"


BOM = "http://www.bom.gov.au/waterdata/services"
B = dict(service="kisters", type="queryServices", format="json")


def bom(st):
    out = ROOT / f"gauges/bom/{st}.csv"
    if out.exists():
        return "exists"
    js = requests.get(BOM, params=dict(B, request="getTimeseriesList", station_no=st,
                                       ts_name="DMQaQc.Merged.DailyMean.24HR",
                                       parametertype_name="Water Course Discharge",
                                       returnfields="ts_id,ts_unitsymbol"), timeout=180).json()
    if not isinstance(js, list) or len(js) < 2:
        return "no series"
    ts_id, unit = js[1][0], js[1][1]
    v = requests.get(BOM, params=dict(B, request="getTimeseriesValues", ts_id=ts_id, **{"from": START, "to": END},
                                      returnfields="Timestamp,Value,Quality Code"), timeout=180).json()
    data = v[0].get("data", []) if v else []
    if not data:
        return "no data"
    df = pd.DataFrame(data, columns=["date", "discharge_m3s", "qual"])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    if unit.strip().lower() in ("ml/d", "ml/day"):
        df["discharge_m3s"] = df["discharge_m3s"] * 1000 / 86400
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return f"{len(df)} days ({unit})"


def ana(code):
    out = ROOT / f"gauges/ana/{code}.csv"
    if out.exists():
        return "exists"
    params = dict(codEstacao=code, dataInicio="01/03/2023", dataFim="30/06/2025", tipoDados="3", nivelConsistencia="")
    r = requests.get(ANA_URL, params=params, timeout=300)
    r.raise_for_status()
    # the response is .NET DataSet XML (diffgr:/msdata: prefixes) -> record blocks are parsed directly
    rows = []
    for block in re.findall(r"<SerieHistorica\b[^>]*>(.*?)</SerieHistorica>", r.text, flags=re.S):
        g = dict(re.findall(r"<(\w+)>([^<]*)</\1>", block))
        month = pd.Timestamp(g["DataHora"][:10])
        nd = calendar.monthrange(month.year, month.month)[1]
        for d in range(1, nd + 1):
            v = g.get(f"Vazao{d:02d}", "")
            if v:
                rows.append(dict(date=month.replace(day=d), discharge_m3s=float(v), qual=int(g["NivelConsistencia"])))
    if not rows:
        return "no data"
    df = pd.DataFrame(rows)
    # for the same day the consistency-checked value (2) replaces the raw value (1)
    df = df.sort_values(["date", "qual"]).drop_duplicates("date", keep="last")
    df = df[(df.date >= START) & (df.date <= END)]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return f"{len(df)} days (level-2 fraction {float((df.qual == 2).mean()):.2f})"


if __name__ == "__main__":
    log = []
    for continent, fn, source, pattern in [("eu", hubeau, "hubeau", r"[A-Z]\d{9}"), ("oc", bom, "bom", r".+"),
                                           ("sa", ana, "ana", r"\d{8}")]:
        inv = pd.read_csv(ROOT / f"data/sos_inventory_{continent}.csv", dtype={"gage_id": str})
        for gid in sorted(set(inv.loc[inv.gage_id != "NO_DATA", "gage_id"])):
            if not re.fullmatch(pattern, gid):
                log.append((gid, continent, source, "out of scope (format)")); continue
            try:
                s = fn(gid)
            except Exception as e:
                s = f"error {type(e).__name__}"
            log.append((gid, continent, source, s)); print(continent, gid, s, flush=True)
            time.sleep(0.5)
    pd.DataFrame(log, columns=["gage_id", "continent", "source", "status"]).to_csv(
        ROOT / "data/gauge_download_log_all.csv", index=False)
