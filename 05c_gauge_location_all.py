"""05c - Independent location check of all candidate gauge-reach pairs.

The gauge id in the SoS 'validation' group is only a candidate list. Station coordinates are taken from the
operating agency (USGS, WSC, Hub'Eau, BoM, ANA) and the distance to the SoS reach centre (SWORD v16 x, y) is
computed.
D19 (fixed in advance): 'OK' if station - SoS reach centre <= 20 km.
Output: data/gauge_location_all.csv
"""
import re
import time

import netCDF4 as nc
import numpy as np
import pandas as pd
import requests

from p38.paths import ROOT  # noqa: E402

MAX_KM = 20.0


def hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def usgs(ids):
    out = {}
    for i in range(0, len(ids), 80):
        url = "https://waterservices.usgs.gov/nwis/site/?format=rdb&sites=" + ",".join(ids[i:i + 80])
        for l in requests.get(url, timeout=180).text.splitlines():
            p = l.split("\t")
            if len(p) > 5 and p[0] == "USGS":
                out[p[1]] = (float(p[4]), float(p[5]))
    return out


def wsc(st):
    f = requests.get("https://api.weather.gc.ca/collections/hydrometric-stations/items",
                     params=dict(STATION_NUMBER=st, f="json"), timeout=180).json()["features"][0]
    lon, lat = f["geometry"]["coordinates"][:2]
    return lat, lon


def hubeau(code):
    d = requests.get("https://hubeau.eaufrance.fr/api/v2/hydrometrie/referentiel/stations",
                     params=dict(code_station=code, format="json"), timeout=180).json()["data"][0]
    return d["latitude_station"], d["longitude_station"]


def bom(st):
    js = requests.get("http://www.bom.gov.au/waterdata/services",
                      params=dict(service="kisters", type="queryServices", format="json", request="getStationList",
                                  station_no=st, returnfields="station_latitude,station_longitude"), timeout=180).json()
    return float(js[1][0]), float(js[1][1])


def ana(code):
    url = ("https://telemetriaws1.ana.gov.br/ServiceANA.asmx/HidroInventario?codEstDE={c}&codEstATE={c}&tpEst=1"
           "&nmEst=&nmRio=&codSubBacia=&codBacia=&nmMunicipio=&nmEstado=&sgResp=&sgOper=&telemetrica=").format(c=code)
    t = requests.get(url, timeout=180).text
    return float(re.search(r"<Latitude>([^<]+)", t).group(1)), float(re.search(r"<Longitude>([^<]+)", t).group(1))


def find_source(gid):
    for sub in ("usgs", "wsc", "hubeau", "bom", "ana"):
        if (ROOT / f"gauges/{sub}/{gid}.csv").exists():
            return sub
    return None


if __name__ == "__main__":
    rows = []
    for continent in ["na", "eu", "oc", "sa", "af", "as"]:
        inv = pd.read_csv(ROOT / f"data/sos_inventory_{continent}.csv", dtype={"gage_id": str})
        inv = inv[inv.gage_id != "NO_DATA"]
        if inv.empty:
            continue
        r = nc.Dataset(ROOT / f"sword/netcdf/{continent}_sword_v16.nc")["reaches"]
        xy = dict(zip(r["reach_id"][:].tolist(), zip(r["y"][:].tolist(), r["x"][:].tolist())))
        src = {g: find_source(g) for g in inv.gage_id}
        us = usgs(sorted(g for g, k in src.items() if k == "usgs"))
        for _, e in inv.iterrows():
            g, k = e.gage_id, src[e.gage_id]
            ll, status = None, None
            try:
                if k == "usgs":
                    ll = us.get(g)
                elif k in ("wsc", "hubeau", "bom", "ana"):
                    ll = {"wsc": wsc, "hubeau": hubeau, "bom": bom, "ana": ana}[k](g)
                    time.sleep(0.3)
                else:
                    status = "no gauge data"
            except Exception as ex:
                status = f"coordinate error {type(ex).__name__}"
            rec = dict(continent=continent, reach_id=int(e.reach_id), gage_id=g, source=k)
            if ll is not None and int(e.reach_id) in xy:
                km = float(hav(ll[0], ll[1], *xy[int(e.reach_id)]))
                rec.update(lat=ll[0], lon=ll[1], km_reach=km, status="OK" if km <= MAX_KM else "far")
            else:
                rec["status"] = status or "no coordinates"
            rows.append(rec)
        print(continent, "done", flush=True)
    df = pd.DataFrame(rows).drop_duplicates(["reach_id", "gage_id"])
    df.to_csv(ROOT / "data/gauge_location_all.csv", index=False)
    print(df.groupby(["continent", "status"]).size())
