"""15 - Paper tables (three-rule layout; formatting is done when the manuscript is compiled). Numbers come only
from the files in output/.

table1  : prior-error transfer per product (transfer_summary.csv, subset 'all')
table2  : chain experiment, main run + sensitivity (summary_chains.csv, summary_sensitivity.csv, summary_numbers.json)
tableS1 : number of reaches with discharge from each algorithm in SoS V3 (algorithm_coverage.csv)
tableS2 : MCMC diagnostics and sensitivity to chain length (mcmc_diagnostics.csv, mcmc_long.csv)
tableS3 : transfer by continent and width class (transfer_summary.csv)
Output: output/tables.json  {name: {"title": ..., "columns": [...], "rows": [[...]], "note": ...}}
"""
import json

import pandas as pd

from p38.paths import TABLES  # noqa: E402

NAME = {"consensus": "Consensus", "momma": "MOMMA", "sic4dvar": "SIC4DVar", "metroman": "MetroMan", "neobam": "neoBAM",
        "hivdi": "HiVDI", "sad": "SAD"}
ORDER = ["consensus", "momma", "sic4dvar", "metroman", "neobam"]
f2 = lambda x: "–" if pd.isna(x) else f"{x:.2f}".replace("-", "−")
f3 = lambda x: "–" if pd.isna(x) else ("0.000" if abs(x) < 0.0005 else f"{x:.3f}".replace("-", "−"))
pct = lambda x: "–" if pd.isna(x) else f"{100 * x:.0f}"


def p_format(p):
    return "< 0.001" if p < 0.001 else f"{p:.2f}" if p >= 0.01 else f"{p:.3f}"


def table1():
    d = pd.read_csv(TABLES / "transfer_summary.csv")
    d = d[d.subset == "all"].set_index("product")
    rows = []
    for u in ORDER:
        r = d.loc[u]
        rows.append([NAME[u], f"{int(r.n)}", f2(r.theil_sen), f"{f2(r.slope)} ({f2(r.slope_ci_lo)}, {f2(r.slope_ci_hi)})",
                     f2(r.log_slope), f"{f3(r.diff_median)} ({f3(r.diff_ci_lo)}, {f3(r.diff_ci_hi)})",
                     p_format(r.wilcoxon_p),
                     f"{pct(r.improved_frac)} ({pct(r.improved_ci_lo)}, {pct(r.improved_ci_hi)})",
                     f2(r.r_median)])
    return dict(
        title="Transfer of Prior Error in the Unconstrained SWOT Discharge Product (Version 3) at Gauged Reaches",
        columns=["Product", "n", "Theil–Sen slope", "LS slope (95% CI)", "Log slope", "Median Δ (95% CI)", "p",
                 "Improved, % (95% CI)", "Median r"],
        rows=rows,
        note=("n: gauge–reach pairs with ≥ 10 matched overpasses and a prior value. Slopes: nbias regressed on "
              "nbias_prior (Equations 1–2) by Theil–Sen and least squares (LS), and log(mean Q_est/mean Q_obs) on "
              "log(Q_prior/mean Q_obs) by least squares. Δ: paired difference |nbias| − |nbias_prior| (Equation 3); "
              "p: two-sided Wilcoxon signed-rank test; Improved: pairs with Δ < 0. Intervals: bootstrap over gauges "
              "(2000 resamples). Median r: correlation with gauge discharge on matched days. LS slopes in the "
              "synthetic benchmark of Frasson et al. (2021): MetroMan 0.73, BAM 0.85, HiVDI 0.84, SAD 0.53."))


def table2():
    g = pd.read_csv(TABLES / "summary_chains.csv")
    g = g[g["mode"] == "chain"]
    sens = pd.read_csv(TABLES / "summary_sensitivity.csv")
    num = json.load(open(TABLES / "summary_numbers.json"))
    PRI = {"width": "Width", "clim": "Climatology"}
    c2 = num["c2_applicability"]
    minus = lambda x: str(x).replace("-", "−")
    rows = []
    for p in ["width", "clim"]:
        for c in ["C0", "C2", "C3"]:
            r = g[(g.prior == p) & (g.constraint == c)].iloc[0]
            rows.append([PRI[p], c + (" ᵃ" if c == "C2" else ""), f"{int(r.n)}", minus(r.kge), minus(r.r),
                         r.beta, r.coverage90])
            if c == "C2":
                a = c2[p]
                rows.append([PRI[p], "C0, C2 chains ᵃ", f"{a['common_n']}", minus(a["C0_kge"]),
                             minus(a["C0_r"]), a["C0_beta"], a["C0_coverage"]])
    rows.append(["SWOT consensus", "–", f"{num['sos_consensus_n']}", num["sos_consensus_kge"].replace("-", "−"),
                 num["sos_consensus_r"], num["sos_consensus_beta"], "–"])
    for setting, label in [("AHG [0.05,0.9]", "C2, b ∈ [0.05, 0.9]"), ("sW>=15%", "C2, σW ≥ 15%")]:
        for p in ["width", "clim"]:
            r = sens[(sens.setting == setting) & (sens.prior == p) & (sens.constraint == "C2")].iloc[0]
            rows.append([PRI[p], label, f"{int(r.n)}", f2(r.kge), f2(r.r), f2(r.beta), f2(r.coverage90)])
    return dict(
        title="Controlled Inversion on Gauged Three-Reach Chains: Median Skill at the Gauged Reach",
        columns=["Prior", "Constraint", "n", "KGE", "r", "β", "Coverage"],
        rows=rows,
        note=("Prior: width-based (Moody and Troutman, 2002) or gauge climatology 2013–2022. Constraint: C0 priors only; "
              "C2 + AHG constraint with exponent bounds; C3 + mass conservation. "
              "Main runs: median [interquartile range]; sensitivity runs (last four rows): median. β: ratio of "
              "means; Coverage: fraction of gauge values inside the posterior 90% interval. The SWOT consensus "
              "row uses the chains for which the product had at least 10 matched overpasses. Sensitivity runs "
              "change one setting of the main run (AHG exponent in (0, 1), width error bounded below at 5%). "
              f"ᵃ C2 could not be applied to {c2['width']['not_applicable']} of {c2['width']['chains']} chains "
              f"(width-based prior) and {c2['clim']['not_applicable']} of {c2['clim']['chains']} (climatology prior), "
              "in which no value of A0 gave an exponent in (0, 1) for at least one reach; C2 statistics and the "
              "paired C0 row refer to the remaining chains."))


def tableS1():
    d = pd.read_csv(TABLES / "algorithm_coverage.csv")
    CONT = {"af": "Africa", "as": "Asia", "eu": "Europe", "na": "North America", "oc": "Oceania", "sa": "South America"}
    al = ["consensus", "metroman", "neobam", "hivdi", "momma", "sad", "sic4dvar"]
    rows = [[CONT[r.continent], f"{r.reaches:,}"] + [f"{getattr(r, a):,}" for a in al] for r in d.itertuples()]
    tot = d[["reaches"] + al].sum()
    rows.append(["Total", f"{tot.reaches:,}"] + [f"{tot[a]:,}" for a in al])
    return dict(title="Number of SWORD Reaches With Discharge From Each Algorithm in the SoS Discharge Product "
                      "(Version 3, Unconstrained)",
                columns=["Continent", "Reaches", "Consensus"] + [NAME[a] for a in al[1:]],
                rows=rows,
                note="A reach is counted if its series contains at least one positive value. Script 12c.")


def tableS2():
    t = pd.read_csv(TABLES / "mcmc_diagnostics.csv")
    if "not_applicable" in t:
        t = t[t.not_applicable.isna()]
    a = t.groupby(["prior", "constraint"]).agg(n=("chain_id", "size"), acceptance=("acceptance", "median"),
                                               tau=("tau_max", "median"), ratio=("steps_per_tau", "median"),
                                               ess=("ess", "median"))
    u = pd.read_csv(TABLES / "mcmc_long.csv")
    w = u.pivot_table(index=["chain_id", "prior", "constraint"], columns="setting",
                      values=["kge", "beta", "contraction_common_n"])
    diff = {m: (w[(m, "long")] - w[(m, "main")]).abs().groupby(level=["prior", "constraint"]).median()
            for m in ["kge", "beta", "contraction_common_n"]}
    PRI = {"width": "Width based", "clim": "Climatology"}
    rows = []
    for (p, c), r in a.iterrows():
        rows.append([PRI[p], c, f"{int(r.n)}", f2(r.acceptance), f"{r.tau:.0f}", f"{r.ratio:.0f}", f"{r.ess:.0f}",
                     f3(diff["kge"].loc[(p, c)]), f3(diff["beta"].loc[(p, c)]),
                     f3(diff["contraction_common_n"].loc[(p, c)])])
    return dict(title="MCMC Diagnostics for 20 Randomly Selected Chains and Sensitivity to Chain Length",
                columns=["Prior", "Constraint", "n", "Acceptance", "τ (steps)", "Steps/τ", "ESS",
                         "|ΔKGE|", "|Δβ|", "|Δ contraction|"],
                rows=rows,
                note=("Main setting: 24 walkers, 2500 steps, 1000 burn-in. τ: largest integrated autocorrelation time "
                      "over parameters (emcee); Steps/τ: post-burn-in steps divided by τ; ESS: effective sample size, "
                      "walkers × steps/τ. |Δ|: median absolute difference between the main setting and a long run "
                      "(7000 steps, 2000 burn-in) for KGE, β and the contraction of the common friction scale. "
                      "Medians over chains. Scripts 08d and 08e."))


def tableS3():
    d = pd.read_csv(TABLES / "transfer_summary.csv")
    SUB = {"continent_na": "North America", "continent_eu": "Europe", "continent_oc": "Oceania",
           "continent_sa": "South America", "w<200": "Width < 200 m", "w200-500": "Width 200–500 m",
           "w>500": "Width > 500 m", "hydraulically_consistent": "Hydraulically consistent",
           "hydraulically_inconsistent": "Hydraulically inconsistent"}
    rows = []
    for k, label in SUB.items():
        row = [label]
        for u in ORDER:
            r = d[(d["product"] == u) & (d.subset == k)]
            row.append("–" if r.empty else
                       f"{f2(r.theil_sen.iloc[0])} / {pct(r.improved_frac.iloc[0])} ({int(r.n.iloc[0])})")
        rows.append(row)
    return dict(title="Prior-Error Transfer by Continent, River Width and Hydraulic Consistency",
                columns=["Subset"] + [NAME[u] for u in ORDER],
                rows=rows,
                note=("Each cell: Theil–Sen slope / percentage of pairs improving on the prior (number of pairs). "
                      "Subsets with fewer than 8 pairs are not shown. Width: median SWOT reach width. Hydraulic "
                      "consistency (positive WSE–width correlation) requires at least 5 quality-controlled "
                      "observations."))


if __name__ == "__main__":
    out = {}
    for name, f in [("table1", table1), ("table2", table2), ("tableS1", tableS1), ("tableS2", tableS2),
                    ("tableS3", tableS3)]:
        try:
            out[name] = f()
            print(name, "-", len(out[name]["rows"]), "rows")
        except FileNotFoundError as e:
            print(name, "SKIPPED (missing input):", e.filename)
    json.dump(out, open(TABLES / "tables.json", "w"), ensure_ascii=False, indent=1)
