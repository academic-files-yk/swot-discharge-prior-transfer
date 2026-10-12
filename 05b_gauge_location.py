"""05b - Independent check of the gauge-reach match for the chains.

The SoS 'validation' group is flagged 'experimental and incorrect' in the product guide -> its gauge id match is
used only as a candidate list. Station coordinates come from the operating agencies: USGS site service, WSC
hydrometric-stations, Hub'Eau referentiel, BoM KiWIS getStationList, ANA HidroInventario, ArcticGRO
(stations_arcticgro.csv).
Distance: station -> SoS reach centre and -> target reach centre (SWORD v16 x, y).
Rule (fixed in advance): station - target reach centre <= 30 km (including the R6 shift); otherwise
'location_suspect'.
Output: data/gauge_location.csv
"""
import netCDF4 as nc
import numpy as np
import pandas as pd
import requests

from p38.paths import ROOT  # noqa: E402
MAX_KM = 30.0


def hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def usgs_coords(ids):
    out = {}
    for i in range(0, len(ids), 80):
        url = "https://waterservices.usgs.gov/nwis/site/?format=rdb&sites=" + ",".join(ids[i:i + 80])
        for l in requests.get(url, timeout=120).text.splitlines():
            p = l.split("\t")
            if len(p) > 5 and p[0] == "USGS":
                out[p[1]] = (float(p[4]), float(p[5]))
    return out


def wsc_coords(st):
    fs = requests.get("https://api.weather.gc.ca/collections/hydrometric-stations/items",
                      params=dict(STATION_NUMBER=st, f="json"), timeout=120).json()["features"]
    lon, lat = fs[0]["geometry"]["coordinates"][:2]
    return lat, lon


def hubeau_coords(code):
    d = requests.get("https://hubeau.eaufrance.fr/api/v2/hydrometrie/referentiel/stations",
                     params=dict(code_station=code, format="json"), timeout=120).json()["data"][0]
    return d["latitude_station"], d["longitude_station"]


def bom_coords(st):
    js = requests.get("http://www.bom.gov.au/waterdata/services",
                      params=dict(service="kisters", type="queryServices", format="json", request="getStationList",
                                  station_no=st, returnfields="station_latitude,station_longitude"), timeout=120).json()
    return float(js[1][0]), float(js[1][1])


def ana_coords(code):
    import re as _re
    url = ("https://telemetriaws1.ana.gov.br/ServiceANA.asmx/HidroInventario?codEstDE={c}&codEstATE={c}&tpEst=1"
           "&nmEst=&nmRio=&codSubBacia=&codBacia=&nmMunicipio=&nmEstado=&sgResp=&sgOper=&telemetrica=").format(c=code)
    t = requests.get(url, timeout=180).text
    return float(_re.search(r"<Latitude>([^<]+)", t).group(1)), float(_re.search(r"<Longitude>([^<]+)", t).group(1))


def sword_xy():
    xy = {}
    for k in ["na", "eu", "oc", "as", "sa"]:
        r = nc.Dataset(ROOT / f"sword/netcdf/{k}_sword_v16.nc")["reaches"]
        for rid, x, y in zip(r["reach_id"][:], r["x"][:], r["y"][:]):
            xy[int(rid)] = (float(y), float(x))
    return xy


if __name__ == "__main__":
    cs = pd.read_csv(ROOT / "data/chain_summary.csv", dtype={"gage_id": str})
    cs = cs[cs.status == "READY"]
    z = pd.read_csv(ROOT / "data/chains.csv", dtype={"gage_id": str})
    XY = sword_xy()
    us = usgs_coords(sorted(cs[cs.gauge_source == "usgs"].gage_id.unique()))
    st = pd.read_csv(ROOT / "data/stations_arcticgro.csv")
    rows = []
    for _, r in cs.iterrows():
        gid, gs = r.gage_id, r.gauge_source
        try:
            if gs == "usgs":
                ll = us.get(gid)
            elif gs == "wsc":
                ll = wsc_coords(gid)
            elif gs == "hubeau":
                ll = hubeau_coords(gid)
            elif gs == "bom":
                ll = bom_coords(gid)
            elif gs == "ana":
                ll = ana_coords(gid)
            else:
                riv, stn = gid.split("_", 1)
                s = st[(st.river == riv) & (st.station == stn)].iloc[0]
                ll = (s.lat, s.lon)
        except Exception:
            ll = None
        sos_rid = int(z.loc[r.chain_id, "reach_id"])
        rec = dict(chain_id=r.chain_id, gage_id=gid, source=gs, target=r.target, sos_reach=sos_rid)
        if ll is None:
            rec.update(status="no coordinates")
        else:
            rec.update(lat=ll[0], lon=ll[1])
            for name, rid in (("sos", sos_rid), ("target", int(r.target))):
                if rid in XY:
                    rec[f"km_{name}"] = float(hav(ll[0], ll[1], *XY[rid]))
            rec["status"] = "OK" if rec.get("km_target", np.inf) <= MAX_KM else "location_suspect"
        rows.append(rec)
        print(gid, rec.get("status"), round(rec.get("km_target", np.nan), 1), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data/gauge_location.csv", index=False)
    print(df.status.value_counts())
    print(df[["km_sos", "km_target"]].describe().round(1))
