"""01 - Inventory of the SoS V3 (unconstrained) product.

Lists, from the SWOT L4 SoS discharge files (per continent, read remotely), the reaches that are matched to a
gauge: reach_id, x, y, river_name, gauge id(s), number of observations and the offline (unconstrained) KGE of
the operational algorithms. The gauge match is used only as a candidate list (see 05b, 05c).

Output: data/sos_inventory_<continent>.csv
Data source: PO.DAAC, SWOT_L4_HR_DAWG_SOS_DISCHARGE_V3 (SWORD v16).
"""
import sys

import earthaccess
import h5py
import numpy as np
import pandas as pd

from p38.paths import ROOT  # noqa: E402
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)
SHORT_NAME = "SWOT_L4_HR_DAWG_SOS_DISCHARGE_V3"


def decode_chars(arr):
    """Convert an (n, nchar) |S1 array to a list of strings."""
    return ["".join(c.decode() for c in row).strip("\x00 ").strip() for row in arr]


def inventory(continent):
    earthaccess.login(strategy="netrc")
    grans = earthaccess.search_data(short_name=SHORT_NAME, count=20)
    grans = [g for g in grans if g["umm"]["GranuleUR"].startswith(f"{continent}_")]
    if not grans:
        raise SystemExit(f"no granule found for {continent}")
    fobj = earthaccess.open(grans)[0]
    with h5py.File(fobj, "r") as h:
        reach_id = h["reaches/reach_id"][:]
        x = h["reaches/x"][:]
        y = h["reaches/y"][:]
        river = [r.decode() if isinstance(r, bytes) else str(r) for r in h["reaches/river_name"][:]]
        v = h["validation/offline"]
        has_val = v["has_validation"][:]
        idx = np.where(has_val == 1)[0]
        gage = decode_chars(v["gageid"][idx, :])
        kge = v["kge"][idx, :]
        testn = v["testn"][idx, :]
        algo = [decode_chars(v["algo_names"][i, :, :]) for i in idx[:1]][0]
    df = pd.DataFrame({
        "reach_id": reach_id[idx], "x": x[idx], "y": y[idx],
        "river_name": [river[i] for i in idx], "gage_id": gage,
    })
    for j, a in enumerate(algo):
        if a:
            df[f"kge_{a}"] = kge[:, j]
            df[f"n_{a}"] = testn[:, j]
    df.to_csv(OUT / f"sos_inventory_{continent}.csv", index=False)
    print(continent, "reaches:", len(reach_id), "| with gauge:", len(idx))
    print("algorithms:", [a for a in algo if a])
    return df


if __name__ == "__main__":
    for k in sys.argv[1:] or ["oc"]:
        inventory(k)
