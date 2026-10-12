"""14 - Figures of the controlled inversion (165 mm, 7-8 pt, PDF + 600 dpi PNG + 600 dpi TIFF).

Fig4_twin        : twin experiment - KGE, r, beta, contraction of the common log n; x = prior quality,
                   colour = constraint
Fig5_chains      : gauged chains - same components; x = prior source (width, climatology); + SoS consensus (grey)
Fig6_beta_prior  : beta against Qbar_prior / observed mean (C0 and C2)
Fig7_hydrographs : example hydrographs (chains at the 25th/50th/75th percentile of SoS consensus KGE - selection
                   fixed in advance)
Input: results/twin_mc_N200.csv, results/chain_ablation.csv, results/sos_comparison.csv, data/gauge_location.csv
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from p38.paths import FIGURES, ROOT  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
import figure_quality as F  # noqa: E402

CC = {"C0": F.PALETTE[0], "C2": F.PALETTE[1], "C3": F.PALETTE[2]}
CL = {"C0": "C0 prior only", "C2": "C2 + bounded AHG", "C3": "C3 + mass conservation"}
GREY = F.PALETTE[5]
# prior-source colours (same in Fig6 and Fig7): width-based = dark violet; climatology = yellow
PRIOR_COLOUR = {"width": F.PALETTE[6], "clim": F.PALETTE[3]}


def boxes(ax, df, xs, xl, m, ref=None, log=False, sos=None):
    cs = [c for c in ("C0", "C2", "C3") if c in set(df.constraint)]
    w = 0.8 / len(cs)
    for j, c in enumerate(cs):
        pos = [i + (j - (len(cs) - 1) / 2) * w for i in range(len(xs))]
        data = [df[(df.prior == x) & (df.constraint == c)][m].astype(float).dropna().values for x in xs]
        bp = ax.boxplot(data, positions=pos, widths=w * 0.85, patch_artist=True, showfliers=False,
                        medianprops=dict(color=F.INK, lw=1.0), whiskerprops=dict(color=F.INK2, lw=0.6),
                        capprops=dict(color=F.INK2, lw=0.6), boxprops=dict(lw=0.5))
        for p in bp["boxes"]:
            p.set_facecolor(CC[c]); p.set_edgecolor("white")
        ax.plot([], [], "s", color=CC[c], label=CL[c])
    ticks, lab = list(range(len(xs))), [xl[x] for x in xs]
    if sos is not None:
        v = sos[m].astype(float).dropna().values
        bp = ax.boxplot([v], positions=[len(xs)], widths=0.35, patch_artist=True, showfliers=False,
                        medianprops=dict(color=F.INK, lw=1.0), whiskerprops=dict(color=F.INK2, lw=0.6),
                        capprops=dict(color=F.INK2, lw=0.6), boxprops=dict(lw=0.5))
        bp["boxes"][0].set_facecolor(GREY); bp["boxes"][0].set_edgecolor("white")
        ticks.append(len(xs)); lab.append("SWOT\nconsensus")
    ax.set_xticks(ticks, lab)
    if ref is not None:
        ax.axhline(ref, color=F.INK2, lw=0.6, ls="--", zorder=0)
    if log:
        ax.set_yscale("log")
        plain_log_axis(ax.yaxis)
    ax.grid(True, axis="y")


def plain_log_axis(axis, ticks=(0.1, 0.2, 0.5, 1, 2, 5, 10, 20)):
    """Plain numbers instead of 6x10^0 on a log axis (readability)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def four_panels(df, xs, xl, name, sos=None):
    fig, ax = plt.subplots(2, 2, figsize=(F.FULL, F.FULL * 0.70))
    ax = ax.ravel()
    boxes(ax[0], df, xs, xl, "kge", ref=1.0, sos=sos); ax[0].set_ylim(-2.5, 1.05); ax[0].set_ylabel("KGE")
    F.panel_title(ax[0], "a", "Kling–Gupta efficiency")
    boxes(ax[1], df, xs, xl, "r", ref=1.0, sos=sos); ax[1].set_ylabel("r")
    F.panel_title(ax[1], "b", "Correlation (dynamics)")
    boxes(ax[2], df, xs, xl, "beta", ref=1.0, log=True, sos=sos); ax[2].set_ylabel("β = mean Q_est / mean Q_obs")
    F.panel_title(ax[2], "c", "Bias ratio (magnitude)")
    boxes(ax[3], df, xs, xl, "contraction_common_n", ref=0.0); ax[3].set_ylabel("1 − Var_post / Var_prior")
    F.panel_title(ax[3], "d", "Contraction of the common friction scale")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 1.0))
    F.save(fig, name, FIGURES)


def chain_data():
    g = pd.read_csv(ROOT / "results/chain_ablation.csv")
    loc = pd.read_csv(ROOT / "data/gauge_location.csv")
    ok = set(loc[loc.status == "OK"].chain_id)
    g = g[g.prior.notna() & g.chain_id.isin(ok)]
    sos = pd.read_csv(ROOT / "results/sos_comparison.csv")
    sos = sos[sos.chain_id.isin(ok)].rename(columns={"consensus_kge": "kge", "consensus_r": "r", "consensus_beta": "beta"})
    return g[g["mode"] == "chain"], sos


def fig6(z):
    fig, ax = plt.subplots(1, 2, figsize=(F.FULL, F.FULL * 0.45), sharey=True)
    lim = (0.05, 50)
    for a, c, h in zip(ax, ("C0", "C2"), "ab"):
        d = z[z.constraint == c]
        for p, mk, lab in (("width", "o", "Width-based prior (SWOT only)"), ("clim", "^", "Pre-SWOT gauge climatology")):
            e = d[d.prior == p]
            a.scatter(e.qbar_prior / e.qobs_mean, e.beta, s=9, marker=mk, color=PRIOR_COLOUR[p],
                      edgecolor="white", lw=0.3, label=lab)
        a.plot(lim, lim, color=F.INK2, lw=0.6, ls="--")
        a.axhline(1, color=F.INK2, lw=0.4)
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lim); a.set_ylim(lim)
        plain_log_axis(a.xaxis, (0.1, 1, 10)); plain_log_axis(a.yaxis, (0.1, 1, 10))
        a.set_xlabel("Prior mean Q / gauge mean Q")
        a.grid(True)
        F.panel_title(a, h, CL[c])
    ax[0].set_ylabel("β = mean Q_est / mean Q_obs")
    ax[0].legend(loc="upper left", fontsize=6.5)
    fig.tight_layout()
    F.save(fig, "Fig6_beta_prior", FIGURES)


def fig7(sos):
    s = sos.dropna(subset=["kge"]).sort_values("kge")
    sel = [s.iloc[int(q * (len(s) - 1))] for q in (0.25, 0.5, 0.75)]
    fig, ax = plt.subplots(3, 1, figsize=(F.FULL, F.FULL * 0.95))
    for i, (a, r) in enumerate(zip(ax, sel)):
        cid = int(r.chain_id)
        files = {e: ROOT / f"results/chain_q/{cid}_{e}.npz" for e in ("width_C0", "clim_C0", "clim_C2")}
        v = {e: np.load(f, allow_pickle=True) for e, f in files.items() if f.exists()}
        if not v:
            continue
        ref = next(iter(v.values()))
        t = pd.to_datetime(ref["t"])
        if "clim_C0" in v:
            a.fill_between(t, v["clim_C0"]["lo"], v["clim_C0"]["hi"], color=F.PALETTE[3], alpha=0.25, lw=0,
                           label="Climatology prior, C0: 90% interval")
        styles = {"width_C0": ("o", PRIOR_COLOUR["width"], "Width-based prior, C0"),
                  "clim_C0": ("^", PRIOR_COLOUR["clim"], "Climatology prior, C0"),
                  "clim_C2": ("D", CC["C2"], "Climatology prior, C2")}
        for e, d in v.items():
            mk, c, lab = styles[e]
            a.plot(t, d["qmed"], mk, ms=2.8, color=c, ls="none", label=lab)
        Sq = np.load(ROOT / f"results/sos_q/{cid}.npz", allow_pickle=True)
        a.plot(pd.to_datetime(Sq["t_consensus"]), Sq["consensus"], "s", ms=2.5, color=GREY, ls="none",
               label="SWOT consensus")
        a.plot(t, ref["Qobs"], ".", ms=4, color=F.INK, label="Gauge (daily mean)")
        a.set_ylabel(r"Q (m$^{3}$ s$^{-1}$)")
        a.grid(True)
        F.panel_title(a, "abc"[i], f"Gauge {r.gage_id} ({r.continent.upper()}); consensus KGE = {r.kge:.2f}")
    h, l = ax[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, 1.0))
    F.save(fig, "Fig7_hydrographs", FIGURES)


if __name__ == "__main__":
    F.style()
    tw = pd.read_csv(ROOT / "results/twin_mc_N200.csv")
    xs = ["Qexact", "Qs025", "Qs050", "Qs100", "Qwidth"]
    xl = {"Qexact": "exact Q̄", "Qs025": "σ = 0.25", "Qs050": "σ = 0.5", "Qs100": "σ = 1.0", "Qwidth": "width-\nbased Q̄"}
    four_panels(tw, xs, xl, "Fig4_twin")
    z, sos = chain_data()
    four_panels(z, ["width", "clim"], {"width": "width-based\nprior", "clim": "climatology\nprior"}, "Fig5_chains", sos=sos)
    fig6(z)
    fig7(sos)
    F.dpi_report(FIGURES)
