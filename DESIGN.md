# Design record

Every methodological decision is recorded here with its reason, in the order in which it was taken
(29 September – 8 October 2026). Decisions are numbered D1–D26; when a decision was changed, the change is
recorded as a dated update rather than by rewriting the original entry. The code refers to these numbers.

## Scientific framework

Manning's equation (wide channel, R ≈ A/W): Q = n⁻¹ (A0 + dA)^(5/3) W^(−2/3) S^(1/2)

Observed: W(t), H(t), S(t) and their uncertainties; dA(t) is derived from W–H. Unknown: A0 and n (per reach) and Q(t).

Identifiability hierarchy:
1. **Single reach, Manning only.** Formally not identifiable: every (A0, n) gives a Q(t). Information comes from
   the prior.
2. **Single reach + AHG linearity (C2).** The linearity of log Q against log W depends on A0 (curvature) but not on
   n (scale), so it can weakly constrain A0 but cannot constrain magnitude.
3. **Several reaches + mass conservation (McFLI, C3).** In consecutive reaches without tributaries,
   Q_i(t) ≈ Q_j(t). The A0 values and the ratios of n are identified; **the common factor of all n is not**
   (it scales Q uniformly and does not violate conservation).
4. **External information for magnitude (C4).** (a) width-based mean-flow prior from SWOT itself;
   (b) model-based mean flow (SoS). Hypothesis: the r and γ components of KGE come from SWOT, β from the prior.

## Decisions

- **D1 Data (SWOT).** Reach observations from the NASA Hydrocron API, RiverSP collection; fields wse, wse_u, width,
  width_u, slope2, slope2_u, d_x_area, reach_q, reach_q_b, dark_frac, ice_clim_f, xovr_cal_q, n_good_nod, partial_f,
  time_str. Slope is the official `slope2` (computed along the SWORD centreline).
  *Update:* collection RiverSP 2.0 (version C, SWORD v16), which has the same reach ids as SoS V3. Version D
  (SWORD v17) uses different ids.
- **D2 Gauges and matching.** Candidate reach–gauge matches from SoS V3; ArcticGRO (17 stations, version 20260810)
  and agency services for daily discharge. Matching rules fixed in advance: gauge–reach distance and flow
  accumulation (facc) difference below thresholds, no major tributary in between.
  *Update (R6):* if a target chain fails the rules, the nearest passing chain along the main stem within ±50 km and
  a facc difference ≤ 10% is used (same rule for all targets). Of the large Arctic rivers only the Ob passed the
  chain rules; the others are run in single-reach mode (no McFLI).
- **D3 Benchmark.** SoS V3 unconstrained consensus and the individual algorithms; A0 and n estimates are compared
  across algorithms where available.
- **D4 Quality filter (fixed in advance, independent of results).** reach_q ≤ 1, ice_clim_f == 0, dark_frac ≤ 0.5,
  slope2 > 0, xovr_cal_q ≤ 1, partial_f == 0. The fraction removed at each step is recorded.
- **D5 dA.** Robust linear W–H fit over the reach time series; dA(t) = ∫ W(h) dh from Hmin to H(t), analytically.
- **D6 Priors.** ln n ~ N(ln 0.03, 0.4²) (covers 0.015–0.08); ln A0 ~ N(μ, 0.5²), with μ set from the mean-flow
  prior Q̄ (D17, D24). The width-based Q̄ uses the downstream hydraulic geometry of Moody and Troutman (2002),
  W = 7.2 Q^0.50.
- **D7 Likelihood.** McFLI: Σ_t Σ_(i,j neighbours) [log Q_i(t) − log Q_j(t)]² / (2σ²_ij(t)); σ² propagated from the
  observation uncertainties by linearising Manning's equation, (5/3·σ_A/A)² + (2/3·σ_W/W)² + (1/2·σ_S/S)², plus a
  structural error term.
- **D8 Inference.** emcee (affine-invariant ensemble MCMC); convergence checked with integrated autocorrelation
  times; fixed seeds.
- **D9 Experimental design (ablation).** C0 prior only; C2 + AHG; C3 + McFLI; each experiment on the same set of
  reaches.
- **D10 Twin experiment.** Gauge discharge plus known A0, n and cross-section shape generate K consecutive reaches;
  noise is added according to SWOT uncertainties; recovery of parameters and Q is measured (sensitivity to input
  error).
- **D11 Metrics.** KGE and its components (r, β, γ), NSE, posterior contraction = 1 − Var_post/Var_prior, coverage of
  the 90% interval.
- **D12 Reporting.** All selected reaches are reported; no selection by score. Code is public.
- **D13 Uncertainty floors.** Product uncertainties are not realistic (width_u ≈ 0.7 m, slope2_u ≈ 3e−6) → lower
  bounds σ_H ≥ 0.10 m, σ_W ≥ 0.05 W, σ_S ≥ 1.7e−5. Sensitivity run: σ_W floor 15% (mission requirement;
  Andreadis et al., 2025).
- **D14 (superseded by D17).** A separate likelihood layer C4a for the width-based mean flow (log sd 0.7) was
  replaced by entering magnitude information only through the prior (D17).
- **D15 Parameter-independent variance.** In the McFLI likelihood the variance of log Q is evaluated at the prior
  median A0. A parameter-dependent variance rewards large A0 through the log(σ²) term (first real run: median β 6.45).
- **D16 AHG bounds.** Without bounds the AHG constraint is degenerate: as A0 → ∞, log Q ≈ c − (2/3) log W, so W–Q is
  perfectly linear with a negative exponent (first real run: median β 20.6). A high W–Q R² therefore says nothing
  about magnitude.
  *Update (8 Oct):* exponent b ∈ (0, 1), from continuity (Leopold and Maddock, 1953) and hydraulic consistency
  (Andreadis et al., 2025); [0.05, 0.9] as a sensitivity run.
- **D17 Single source of magnitude.** The A0 prior is derived from the mean-flow prior Q̄ through Manning's equation,
  so Q̄ is the only source of magnitude information. Design: prior source {width-based (SWOT only); pre-SWOT gauge
  climatology 2013–2022 (no overlap with the evaluation period)} × constraint {C0, C2, C3}. In the twin experiment,
  prior error scale {0, 0.25, 0.5, 1.0} plus width-based.
  *Diagnostic (exploratory/10):* even with zero noise, McFLI cannot move A0 away from its prior; A0 is largely not
  identifiable and the bias comes from the prior, not from noise.
- **D18 Permitted SoS groups.** The SWOT Discharge Product Quick Guide v3 (10 Dec 2025) states that the MOI, offline,
  postdiagnostics and validation groups are experimental and incorrect. The comparison therefore uses
  `consensus/consensus_q` with `consensus/time_int` and the algorithm groups; the gauge match in the validation group
  is used only as a candidate list and checked independently against agency station coordinates (05b: station to
  target reach ≤ 30 km). A first comparison based on the offline group was discarded.
- **D19 Transfer analysis sample.** Not limited to chains: all gauged reaches in the SoS V3 candidate list; station
  coordinates from the operating agency (USGS, WSC, Hub'Eau, BoM, ANA); station to SoS reach centre ≤ 20 km.
  Africa (DWS) has no open data and is out of scope. ANA data from the legacy service (about 36%
  consistency-checked, the rest raw).
- **D20 Transfer metric (fixed before looking at results).** Same metric as Frasson et al. (2021). Matching: gauge
  daily mean on the SWOT overpass day (UTC); ≥ 10 matches per product. nbias_product =
  (mean(Q_est) − mean(Q_obs))/mean(Q_obs) over matched pairs; nbias_prior = (Q_m_prior − mean(Q_obs))/mean(Q_obs) on
  the same matched set. Transfer slope: OLS (nbias_product ~ nbias_prior) with Pearson r; also Theil–Sen and a log
  version (log β ~ log(Q_m_prior/mean Q_obs)). Improvement fraction: P(|nbias_product| < |nbias_prior|).
- **D21 Confidence intervals.** Cluster bootstrap over gauges (one gauge may be matched to several reaches), 2000
  resamples, 95%, fixed seed.
- **D22 Subsets.** (i) hydraulic consistency: corr(WSE, width) > 0 over quality-filtered overpasses (criterion of
  Andreadis et al., 2025); (ii) continent; (iii) width class (SWORD p_width < 200, 200–500, > 500 m). Reference:
  synthetic slopes of Frasson et al. (2021) (BAM 0.85, MetroMan 0.73, HiVDI 0.84, SAD 0.53).
- **D23 Paired test.** "Does the product improve on the prior?": Wilcoxon signed-rank test of the paired difference
  |nbias_product| − |nbias_prior| and a bootstrap 95% CI of its median. Robust transfer measures (Theil–Sen, log) are
  the main result; the OLS slope used by Frasson et al. is very sensitive to outliers and is given for comparison.
  Caveat: MOMMA uses input_Qm_prior as its calibration target (slope 1 by construction); other algorithms may
  receive a differently processed prior.
- **D24 Exact prior calibration.** The A0 prior median is solved numerically so that, at the prior median n, the
  mean Manning discharge over the observed overpasses equals Q̄ exactly (the mean-state approximation of D17 gave a
  Jensen bias of about 16%). Check: with the true Q̄, β 1.10 → 1.001.
- **D25 Cross-section shape in the twin experiment.** Half trapezoidal, half power law W = B h^p
  (p ~ U(0.2, 1.0)), to test whether the success of AHG is specific to trapezoids. Prior uncertainty consistent with
  the log standard errors of Moody and Troutman (2002) (W: 0.22).
- **D26 MCMC initialisation (bug fix, 8 Oct).** Walkers start in the part of the prior that satisfies the
  constraints. Previously they started directly from the prior; with the width-based prior, in C2 all 24 walkers lay
  outside the AHG bounds (log posterior −∞) in 63 of 108 chains (37 of 83 with the climatology prior), and the
  sampler never moved (acceptance 0; found in the long-chain test 08e). Because the AHG exponent b depends only on
  ln A0 (ln n only shifts the intercept, Equation 9), the feasible set is a one-dimensional set of ln A0 per reach;
  it is found on a grid (step 0.005), ln A0 is drawn from the prior truncated to this set and ln n from the prior.
  The feasible prior mass (feasible_frac) and the acceptance fraction are recorded for every run; if the feasible set
  is empty the run fails with `NoFeasibleStart` and is reported separately. All affected steps were re-run
  (08, 08a, 08s, 03, 08d, 08e → 11, 13, 14, 15).

## Data provenance

ArcticGRO values after 2023 are provisional ('P'); the Lena record has 134 missing days in 2025. Gaps are not filled.
