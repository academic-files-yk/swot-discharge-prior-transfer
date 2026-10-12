"""13 - Figures of the prior-error transfer analysis (165/90 mm, 7-8 pt, PDF + 600 dpi PNG + 600 dpi TIFF).

Fig1_map            : gauge-reach pairs (Equal Earth, EPSG:8857); boundaries Natural Earth 1:50m Admin 0 v5.1.1
Fig2_prior_transfer : retrieved/gauge mean ratio against prior/gauge mean ratio (log-log), one panel per product
Fig3_improvement    : median of the paired difference |nbias_product| - |nbias_prior| (95% CI) + improvement fraction
Input: output/transfer_pairs.csv, output/transfer_summary.csv, data/gauge_location_all.csv,
       data/gauge_location.csv, results/chain_ablation.csv
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from p38.paths import FIGURES, ROOT, TABLES  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
import figure_quality as F  # noqa: E402

PRODUCT_NAME = {"consensus": "Consensus", "momma": "MOMMA", "sic4dvar": "SIC4DVar", "metroman": "MetroMan",
                "neobam": "neoBAM"}
FRASSON = {"neobam": ("BAM", 0.85), "metroman": ("MetroMan", 0.73)}
GREY = "#7a7974"


def fig1_map():
    loc = pd.read_csv(ROOT / "data/gauge_location_all.csv", dtype={"gage_id": str})
    tp = pd.read_csv(TABLES / "transfer_pairs.csv", dtype={"gage_id": str})
    used = loc.merge(tp[["reach_id", "gage_id"]].drop_duplicates(), on=["reach_id", "gage_id"])
    # gauges used in the chain experiment: same filter as 11_summary (location check OK, result in chain_ablation)
    ca = pd.read_csv(ROOT / "results/chain_ablation.csv", dtype={"gage_id": str})
    loc_c = pd.read_csv(ROOT / "data/gauge_location.csv", dtype={"gage_id": str})
    cg = ca[(ca["mode"] == "chain") & (ca.prior == "width") & ca.chain_id.isin(set(loc_c[loc_c.status == "OK"].chain_id))]
    cg = loc_c[loc_c.chain_id.isin(set(cg.chain_id))][["gage_id", "lat", "lon"]].drop_duplicates("gage_id")
    used["chain"] = used.gage_id.isin(set(cg.gage_id))
    chain_only = cg[~cg.gage_id.isin(set(used.gage_id))]
    st = pd.read_csv(ROOT / "data/stations_arcticgro.csv")
    world = gpd.read_file(ROOT / "maps/ne_50m_admin_0_countries.shp").to_crs(8857)
    pts = gpd.GeoDataFrame(used, geometry=gpd.points_from_xy(used.lon, used.lat), crs=4326).to_crs(8857)
    arc = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st.lon, st.lat), crs=4326).to_crs(8857)
    fig, ax = plt.subplots(figsize=(F.FULL, F.FULL * 0.50))
    world.plot(ax=ax, color="#eeeeec", edgecolor="#bdbdb8", linewidth=0.25)
    # graticule (30 degrees)
    for lat in range(-60, 61, 30):
        g = gpd.GeoSeries.from_xy(np.linspace(-180, 180, 361), np.full(361, lat), crs=4326).to_crs(8857)
        ax.plot(g.x, g.y, color="#d8d8d4", lw=0.3, zorder=0)
        if -60 < lat < 90:
            p = gpd.GeoSeries.from_xy([-180], [lat], crs=4326).to_crs(8857)
            ax.text(p.x.iloc[0], p.y.iloc[0], f"{abs(lat)}°{'N' if lat > 0 else 'S' if lat < 0 else ''}",
                    fontsize=6, color=F.INK2, ha="right", va="center")
    for lon in range(-180, 181, 60):
        g = gpd.GeoSeries.from_xy(np.full(181, lon), np.linspace(-90, 90, 181), crs=4326).to_crs(8857)
        ax.plot(g.x, g.y, color="#d8d8d4", lw=0.3, zorder=0)
    a = pts[~pts.chain]
    b = pts[pts.chain]
    ax.scatter(a.geometry.x, a.geometry.y, s=6, color=F.PALETTE[0], edgecolor="white", lw=0.2,
               label=f"Prior-transfer analysis only (n = {len(a)})", zorder=3)
    c = gpd.GeoDataFrame(chain_only, geometry=gpd.points_from_xy(chain_only.lon, chain_only.lat),
                         crs=4326).to_crs(8857)
    bc = pd.concat([b.geometry, c.geometry])
    ax.scatter(bc.x, bc.y, s=9, color=F.PALETTE[1], edgecolor="white", lw=0.2,
               label=f"Three-reach chain experiment (n = {len(bc)})", zorder=4)
    ax.scatter(arc.geometry.x, arc.geometry.y, s=14, marker="^", color=F.PALETTE[2], edgecolor="white", lw=0.3,
               label=f"ArcticGRO single-reach tests (n = {len(arc)})", zorder=4)
    ax.set_axis_off()
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")  # keep the text check clean
    ax.set_aspect("equal")
    top = gpd.GeoSeries.from_xy([0], [85], crs=4326).to_crs(8857).y.iloc[0]
    bottom = gpd.GeoSeries.from_xy([0], [-60], crs=4326).to_crs(8857).y.iloc[0]
    ax.set_ylim(bottom, top)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.10)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=6.5, handletextpad=0.3,
               columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))
    F.save(fig, "Fig1_map", FIGURES)


def fig2_transfer():
    tp = pd.read_csv(TABLES / "transfer_pairs.csv", dtype={"gage_id": str})
    sm = pd.read_csv(TABLES / "transfer_summary.csv")
    products = ["consensus", "momma", "sic4dvar", "metroman", "neobam"]
    fig, axs = plt.subplots(2, 3, figsize=(F.FULL, F.FULL * 0.66))
    axs = axs.ravel()
    lim = (0.05, 20)
    for k, (ax, u) in enumerate(zip(axs, products)):
        d = tp[(tp["product"] == u)].dropna(subset=["log_beta", "log_prior"])
        x, y = np.exp(d.log_prior), np.exp(d.log_beta)
        ax.scatter(x, y, s=4, color=F.PALETTE[0], alpha=0.55, edgecolor="none", rasterized=True)
        ax.plot(lim, lim, color=F.INK2, lw=0.6, ls="--")
        o = sm[(sm["product"] == u) & (sm.subset == "all")].iloc[0]
        xx = np.geomspace(*lim, 50)
        b = np.polyfit(d.log_prior, d.log_beta, 1)
        ax.plot(xx, np.exp(b[1]) * xx ** b[0], color=F.PALETTE[1], lw=1.0)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.grid(True, which="major")
        F.panel_title(ax, "abcde"[k], PRODUCT_NAME[u])
        label = f"n = {int(o.n)}\nslope = {o.log_slope:.2f}"
        ax.text(0.04, 0.96, label, transform=ax.transAxes, va="top", fontsize=6.5, color=F.INK,
                bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="none", alpha=0.85))
        if k % 3 == 0:
            ax.set_ylabel("Retrieved mean Q / gauge mean Q")
        if k >= 2:
            ax.set_xlabel("Prior mean Q / gauge mean Q")
    axs[5].set_axis_off()
    axs[5].plot([], [], color=F.INK2, lw=0.6, ls="--", label="1:1 (full transfer)")
    axs[5].plot([], [], color=F.PALETTE[1], lw=1.0, label="Log–log least-squares fit")
    axs[5].scatter([], [], s=6, color=F.PALETTE[0], label="Gauge–reach pair")
    axs[5].legend(loc="center", fontsize=7)
    fig.tight_layout()
    F.save(fig, "Fig2_prior_transfer", FIGURES)


def fig3_improvement():
    sm = pd.read_csv(TABLES / "transfer_summary.csv")
    products = ["consensus", "momma", "sic4dvar", "metroman", "neobam"]
    fig, axs = plt.subplots(1, 2, figsize=(F.FULL, F.FULL * 0.40), sharey=True)
    for j, (sub, mk, colour, lab) in enumerate([("all", "o", F.PALETTE[0], "All pairs"),
                                                ("hydraulically_consistent", "s", F.PALETTE[1],
                                                 "Hydraulically consistent")]):
        d = sm[sm.subset == sub].set_index("product").loc[products]
        yy = np.arange(len(products)) + (j - 0.5) * 0.25
        axs[0].errorbar(d.diff_median, yy, xerr=[d.diff_median - d.diff_ci_lo, d.diff_ci_hi - d.diff_median],
                        fmt=mk, ms=3.5, color=colour, lw=0.8, capsize=1.5, label=lab)
        axs[1].errorbar(d.improved_frac, yy, xerr=[d.improved_frac - d.improved_ci_lo,
                                                   d.improved_ci_hi - d.improved_frac],
                        fmt=mk, ms=3.5, color=colour, lw=0.8, capsize=1.5)
    axs[0].axvline(0, color=F.INK2, lw=0.6, ls="--")
    axs[1].axvline(0.5, color=F.INK2, lw=0.6, ls="--")
    axs[0].set_yticks(range(len(products)), [PRODUCT_NAME[u] for u in products])
    axs[0].invert_yaxis()
    axs[0].set_xlabel("Median of |nbias(product)| − |nbias(prior)|")
    axs[1].set_xlabel("Fraction of pairs with |nbias(product)| < |nbias(prior)|")
    F.panel_title(axs[0], "a", "Change in absolute bias relative to the prior")
    F.panel_title(axs[1], "b", "Pairs where the product improves on the prior")
    for ax in axs:
        ax.grid(True, axis="x")
    h, l = axs[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.0))
    F.save(fig, "Fig3_improvement", FIGURES)


if __name__ == "__main__":
    F.style()
    fig1_map()
    fig2_transfer()
    fig3_improvement()
    F.dpi_report(FIGURES)
