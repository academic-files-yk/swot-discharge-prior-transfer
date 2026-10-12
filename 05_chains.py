"""05 - Three-reach chains around the gauged reach.

Target reaches:
  (a) reaches with a gauge match in the SoS V3 offline validation group (gage_id != NO_DATA)
  (b) ArcticGRO stations: among type-1 reaches within 20 km of the station, the one with the largest facc (main stem)
Chain = [upstream, target, downstream]. Rules (fixed in advance, independent of results):
  R1 all three reaches are type 1 (river) with lakeflag == 0
  R2 no junction at the links: up.n_rch_down == 1, target.n_rch_up == 1, target.n_rch_down == 1, down.n_rch_up == 1
  R3 facc(down) / facc(up) <= 1.10 (lateral inflow along the chain <= 10%)
  R4 trib_flag == 0 for all three reaches (no tributary missing from SWORD)
  R5 SWORD width >= 80 m (all three reaches)
  R6 if the target fails, the nearest passing chain along the main stem within +-50 km and facc difference <= 10%
Output: data/chains.csv (all candidates, passing or not, with the reason for rejection)
"""
import netCDF4 as nc
import numpy as np
import pandas as pd

from p38.paths import ROOT  # noqa: E402
SW = ROOT / "sword/netcdf"
CONTINENTS = ["na", "sa", "eu", "af", "as", "oc"]


def sword_reaches(continent):
    d = nc.Dataset(SW / f"{continent}_sword_v16.nc")
    r = d["reaches"]
    df = pd.DataFrame({k: r[k][:] for k in ["reach_id", "x", "y", "width", "facc", "slope", "reach_length",
                                             "n_rch_up", "n_rch_down", "lakeflag", "trib_flag", "dist_out"]})
    up = np.asarray(r["rch_id_up"][:]).T
    dn = np.asarray(r["rch_id_dn"][:]).T
    df["up1"], df["dn1"] = up[:, 0], dn[:, 0]
    df["reach_type"] = df.reach_id % 10
    rn = r["river_name"][:]
    df["river"] = [str(v) for v in rn]
    return df.set_index("reach_id")


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def check(S, rid):
    """Build the chain for a target reach; returns a result dict."""
    if rid not in S.index:
        return dict(reason="not in SWORD")
    h = S.loc[rid]
    if h.n_rch_up != 1 or h.n_rch_down != 1:
        return dict(reason="R2 junction at target")
    u, d = int(h.up1), int(h.dn1)
    if u not in S.index or d not in S.index:
        return dict(reason="neighbour not in SWORD")
    U, D = S.loc[u], S.loc[d]
    chain = [U, h, D]
    if any(c.reach_type != 1 or c.lakeflag != 0 for c in chain):
        return dict(reason="R1 type/lake", up=u, dn=d)
    if U.n_rch_down != 1 or D.n_rch_up != 1:
        return dict(reason="R2 junction at neighbour", up=u, dn=d)
    ratio = D.facc / U.facc if U.facc > 0 else np.inf
    if ratio > 1.10:
        return dict(reason="R3 facc", up=u, dn=d, facc_ratio=ratio)
    if any(c.trib_flag != 0 for c in chain):
        return dict(reason="R4 trib_flag", up=u, dn=d, facc_ratio=ratio)
    wmin = min(c.width for c in chain)
    if wmin < 80:
        return dict(reason="R5 width", up=u, dn=d, facc_ratio=ratio, w_min=wmin)
    return dict(reason="PASSED", up=u, dn=d, facc_ratio=ratio, w_min=wmin,
                w_target=h.width, slope_sword=h.slope, facc=h.facc, river=h.river)


def check_r6(S, rid, max_km=50.0, max_facc=0.10, steps=8):
    """R6: if the target fails, search along the main stem (single links) within +-max_km and a facc difference
    <= 10% for the nearest passing chain. Applied in the same way to all targets."""
    first = check(S, rid)
    if first["reason"] == "PASSED" or rid not in S.index:
        return dict(**first, r6=False, shift_km=0.0)
    h = S.loc[rid]
    candidates = []
    for direction in ("up1", "dn1"):
        cur = rid
        for _ in range(steps):
            nxt = int(S.loc[cur, direction])
            if nxt not in S.index or nxt == 0:
                break
            cur = nxt
            shift = abs(S.loc[cur, "dist_out"] - h.dist_out) / 1000.0
            if shift > max_km:
                break
            if abs(S.loc[cur, "facc"] / h.facc - 1) > max_facc:
                break
            result = check(S, cur)
            if result["reason"] == "PASSED":
                candidates.append((shift, cur, result))
                break
    if not candidates:
        return dict(**first, r6=False, shift_km=np.nan)
    shift, cur, result = min(candidates, key=lambda a: a[0])
    return dict(**result, r6=True, shift_km=shift, selected_reach=cur)


if __name__ == "__main__":
    rows = []
    st = pd.read_csv(ROOT / "data/stations_arcticgro.csv")
    for k in CONTINENTS:
        S = sword_reaches(k)
        invf = ROOT / f"data/sos_inventory_{k}.csv"
        if invf.exists():
            inv = pd.read_csv(invf, dtype={"gage_id": str})
            inv = inv[inv.gage_id != "NO_DATA"]
            for _, e in inv.iterrows():
                rows.append(dict(continent=k, source="SoS", gage_id=e.gage_id, reach_id=int(e.reach_id),
                                 **check_r6(S, int(e.reach_id))))
        # ArcticGRO stations (if within the SWORD coverage of this continent)
        for _, s in st.iterrows():
            dist = haversine_km(s.lat, s.lon, S.y.values, S.x.values)
            cand = S[(dist <= 20) & (S.reach_type == 1)]
            if cand.empty:
                continue
            rid = int(cand.facc.idxmax())
            rows.append(dict(continent=k, source="ArcticGRO", gage_id=f"{s.river}_{s.station}", reach_id=rid,
                             station_dist_km=float(dist[S.index.get_loc(rid)]), **check_r6(S, rid)))
        print(k, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data/chains.csv", index=False)
    print(df.groupby(["source", "reason"]).size())
    print(df[df.source == "ArcticGRO"][["gage_id", "reach_id", "station_dist_km", "reason", "w_min", "facc_ratio"]])
