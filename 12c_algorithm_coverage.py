"""12c - Number of reaches for which each algorithm produces discharge in the SoS V3 (unconstrained) product,
per continent.

Purpose: HiVDI and SAD are absent from, or rare in, the transfer analysis (12b); this documents that the cause is
the product itself, not the data extraction. A series with at least one positive value counts as "has output".
Output (TABLES): algorithm_coverage.csv
"""
import earthaccess
import h5py
import numpy as np
import pandas as pd

from p38.paths import TABLES  # noqa: E402

ALG = {"consensus": "consensus/consensus_q", "metroman": "metroman/allq", "neobam": "neobam/q/q", "hivdi": "hivdi/Q",
       "momma": "momma/Q", "sad": "sad/Qa", "sic4dvar": "sic4dvar/Q_da"}

if __name__ == "__main__":
    earthaccess.login(strategy="netrc")
    grans = earthaccess.search_data(short_name="SWOT_L4_HR_DAWG_SOS_DISCHARGE_V3", count=10)
    rows = []
    for g in grans:
        continent = g["umm"]["GranuleUR"][:2]
        H = h5py.File(earthaccess.open([g])[0], "r")
        n = len(H["reaches/reach_id"])
        r = dict(continent=continent, reaches=n, granule=g["umm"]["GranuleUR"])
        for a, v in ALG.items():
            r[a] = int(sum(np.any(np.asarray(q, float) > 0) for q in H[v][:]))
        rows.append(r)
        print(r, flush=True)
    d = pd.DataFrame(rows).sort_values("continent")
    d.to_csv(TABLES / "algorithm_coverage.csv", index=False)
    print(d.to_string())
