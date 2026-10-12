"""MCMC inference and metrics (D11)."""
import emcee
import numpy as np

from .model import Config, Prior, log_posterior, manning_logq, reach_logq


class NoFeasibleStart(RuntimeError):
    """No point drawn from the prior has a finite posterior (the constraint excludes all prior mass)."""


def ahg_feasible_grid(reach, cfg, prior, step=0.005, sd_below=15.0, sd_above=8.0):
    """ln A0 values of one reach for which the AHG exponent b lies in (cfg.ahg_b_min, cfg.ahg_b_max).

    b is the slope of the regression of log W on model log Q; ln n only shifts the intercept, so b depends on
    ln A0 alone (Equation 9). The feasible set is therefore a one-dimensional set of ln A0 per reach and is
    found on a fine grid.
    Returns: (grid, feasible_mask, prior_mass); with fewer than 4 valid observations there is no constraint
    (mask all True).
    """
    mu, sd = prior.logA0_mu(reach), prior.depth_sd
    g = np.arange(mu - sd_below * sd, mu + sd_above * sd, step)
    m = np.isfinite(reach.W) & np.isfinite(reach.S) & np.isfinite(reach.dA) & (reach.S > 0)
    if m.sum() < 4:
        return g, np.ones_like(g, bool), 1.0
    lw = np.log(reach.W[m])
    ok = np.zeros_like(g, bool)
    for i, la0 in enumerate(g):
        lq = manning_logq(la0, prior.logn_mu, reach.dA[m], reach.W[m], reach.S[m])
        b = np.polyfit(lq, lw, 1)[0]
        ok[i] = cfg.ahg_b_min < b < cfg.ahg_b_max
    w = np.exp(-0.5 * ((g - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    return g, ok, float(np.sum(w[ok]) * step)


def feasible_start(reaches, cfg, prior, nwalkers, rng):
    """D26: walkers start in the part of the prior that satisfies the constraints.

    With AHG on: for each reach, ln A0 is drawn from the prior truncated to the feasible set (grid point plus a
    uniform offset within one grid step) and ln n from the prior; this is an exact draw from the intersection
    of the prior with the feasible set. With AHG off: directly from the prior.
    Each starting point is redrawn until its log posterior is finite. Raises NoFeasibleStart if the feasible
    set is empty. (Starting the walkers directly from the prior leaves all of them at -inf when the AHG bound
    excludes almost all prior mass, and the sampler never moves.)
    Returns: (p0, feasible_frac)  feasible_frac = share of prior mass that satisfies the constraints (1 without AHG).
    """
    K = len(reaches)
    if not cfg.ahg:
        x = prior.sample(reaches, nwalkers, rng)
        lp = np.array([log_posterior(t, reaches, cfg, prior) for t in x])
        for _ in range(1000):
            if np.all(np.isfinite(lp)):
                break
            j = ~np.isfinite(lp)
            x[j] = prior.sample(reaches, int(j.sum()), rng)
            lp[j] = [log_posterior(t, reaches, cfg, prior) for t in x[j]]
        if not np.all(np.isfinite(lp)):
            raise NoFeasibleStart("no finite starting point found in the prior")
        return x, 1.0
    grids = [ahg_feasible_grid(r, cfg, prior) for r in reaches]
    if any(not ok.any() for _, ok, _ in grids):
        raise NoFeasibleStart("in at least one reach the AHG exponent is outside the bounds for every ln A0")
    frac = float(np.prod([k for _, _, k in grids]))
    step = grids[0][0][1] - grids[0][0][0]
    out = []
    for _ in range(100 * nwalkers):
        la0 = []
        for (g, ok, _), r in zip(grids, reaches):
            mu, sd = prior.logA0_mu(r), prior.depth_sd
            gg = g[ok]
            w = np.exp(-0.5 * ((gg - mu) / sd) ** 2)
            la0.append(rng.choice(gg, p=w / w.sum()) + rng.uniform(-step / 2, step / 2))
        ln = rng.normal(prior.logn_mu, prior.logn_sd, K)
        th = np.concatenate([la0, ln])
        if np.isfinite(log_posterior(th, reaches, cfg, prior)):
            out.append(th)
            if len(out) == nwalkers:
                return np.array(out), frac
    raise NoFeasibleStart("could not generate enough finite starting points from the feasible set")


def run(reaches, cfg: Config, prior: Prior, seed=0, nwalkers=32, nsteps=3000, burn=1000, thin=10, info=None):
    """Posterior samples (N, 2K). For C0 (no constraint) samples are drawn directly from the prior.

    info (if a dict is given) is filled with acceptance (mean acceptance fraction) and feasible_frac (share of
    prior mass satisfying the constraints, from the initialisation).
    """
    rng = np.random.default_rng(seed)
    K = len(reaches)
    if not (cfg.ahg or cfg.mcfli or cfg.qbar_prior is not None):
        return prior.sample(reaches, nwalkers * (nsteps - burn) // thin, rng)
    p0, frac = feasible_start(reaches, cfg, prior, nwalkers, rng)
    sampler = emcee.EnsembleSampler(nwalkers, 2 * K, log_posterior, args=(reaches, cfg, prior))
    sampler.random_state = np.random.RandomState(seed).get_state()
    sampler.run_mcmc(p0, nsteps, progress=False)
    if info is not None:
        info.update(acceptance=float(np.mean(sampler.acceptance_fraction)), feasible_frac=float(frac))
    return sampler.get_chain(discard=burn, thin=thin, flat=True)


def q_ensemble(samples, reaches, idx=0, max_draws=2000, seed=0):
    """Ensemble of Q(t) for reach idx from posterior samples (draws, T)."""
    rng = np.random.default_rng(seed)
    sel = samples[rng.choice(len(samples), min(max_draws, len(samples)), replace=False)]
    return np.exp(np.array([reach_logq(th, reaches)[idx] for th in sel]))


def kge_parts(sim, obs):
    m = np.isfinite(sim) & np.isfinite(obs)
    s, o = sim[m], obs[m]
    r = np.corrcoef(s, o)[0, 1]
    beta = s.mean() / o.mean()
    gamma = (s.std() / s.mean()) / (o.std() / o.mean())
    kge = 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
    return dict(kge=kge, r=r, beta=beta, gamma=gamma, n=int(m.sum()))


def nse(sim, obs):
    m = np.isfinite(sim) & np.isfinite(obs)
    s, o = sim[m], obs[m]
    return 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)


def coverage(ens, obs, lo=5, hi=95):
    a, b = np.nanpercentile(ens, [lo, hi], axis=0)
    m = np.isfinite(obs)
    return float(np.mean((obs[m] >= a[m]) & (obs[m] <= b[m])))


def contraction(prior_samples, post_samples):
    """Posterior contraction per parameter: 1 - Var_post / Var_prior."""
    return 1 - post_samples.var(axis=0) / prior_samples.var(axis=0)
