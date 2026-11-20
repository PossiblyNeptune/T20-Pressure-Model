"""
Shared scoring logic used by step 5 (final rankings + bootstrap) and step 6 (robustness checks).

prepare()  -> per player-match sums from the adjusted ball scores
score()    -> per-player components and final score:
   pressure rating  = pressure_C * tau^2/(tau^2 + SE^2)        quality under pressure, shrunk for sample size
   consistency      = pressure-weighted share of matches "delivered" (match score > 0), shrunk to league rate
   volume           = pressure rating * total pressure faced (sum of w)   "pressure value above average"
   FINAL (legacy)   = wR*z(rating) + wC*z(consistency) + wV*z(volume)    z across eligible players

Two volume-aware alternatives (no hand-picked weights):
   replacement level = ROLE-SPECIFIC: pooled pressure performance of fringe players (60-299 balls) in the same
                       role (batting position / bowling phase x pace-spin), averaged over the balls the player faced
   VORP              = (rating - replacement) * pressure faced      -> "total pressure value over a fringe player"
   rating_inf        = volume-informed rating: shrink observed pressure performance toward a prior mean that
                       depends on playing time, mu(x) = a + c*log(pressure faced), fitted by weighted regression
                       across all players with 60+ balls (iterated with tau^2). Volume counts exactly as much as
                       it predicts quality.
   MARGIN (MAIN)     = "Reliable Pressure Margin": 90% lower bound of (rating - role replacement level)
                       = rating - repl_level - z * posterior_sd,  posterior_sd = sqrt(tau^2 * (1 - reliability))
                       "we are 90% sure this player beats a fringe player in his own role, under pressure, by at least this"
mode: "margin" | "vorp" | "informed" | "legacy" decides which column drives `rank`.
"""
import numpy as np, pandas as pd

SIDES = {"bat": ("striker_id", "striker_name"), "bowl": ("bowler_id", "bowler_name")}

W_MODE = "excess"         # pressure weight = LI - 1 for balls above normal pressure (capped LI 5)
TAU_C_FILE = None         # set by callers to data_work/clutch_tau.json (empirical clutch spread)
REPL_BAND = (60, 300)     # "fringe" players = 60-299 live balls in the data being ranked
ROLE_K = 2000             # role baseline needs ~2000 pressure-weighted balls of fringe evidence to be half-trusted

def role_of(d, side):
    if side == "bat":    # batting position the ball was faced at
        # top order = openers + No. 3 (one role: same job, and No. 3 alone has too few fringe players for a stable bar)
        return pd.cut(d.batting_order, [0, 3, 5, 7, 11], labels=["top", "4-5", "6-7", "tail"]).astype(str)
    return (d.phase.astype(str) + "-" + d.pace_or_spin.astype(str))      # bowling phase x pace/spin

def hier_replacement(d, fr, side, cap, pid, verbose=False):
    """Replacement level for each ball from fringe players in the SAME context, as specific as the data allows:
    role -> + competition (IPL / international) -> + innings -> + phase (batters) -> + pressure band.
    Each finer cell is shrunk toward its parent by empirical Bayes (true spread of cells vs their noise).
    Outlier guard: no single fringe player supplies more than `cap` of a cell's pressure weight."""
    for x in (d, fr):
        x["cg"] = np.where(x.comp_group == "IPL", "IPL", "INT")
        x["pband"] = pd.cut(x.li, [0, 1, 2, 3.5, np.inf], labels=False).astype(int)
    if hier_replacement.use_cg:
        lv = [["role"], ["role", "cg"], ["role", "cg", "innings"]]
        lv += [["role", "cg", "innings", "phase"], ["role", "cg", "innings", "phase", "pband"]] if side == "bat" else \
              [["role", "cg", "innings", "pband"]]
    else:   # the match SITUATION only: competition differences are already handled by the IPL-T20I bridge
        lv = [["role"], ["role", "innings"]]
        lv += [["role", "innings", "phase"], ["role", "innings", "phase", "pband"]] if side == "bat" else \
              [["role", "innings", "pband"]]
    f = fr[fr.w > 0].copy()
    est_prev = None
    info = []
    for keys in lv:
        pc = f.groupby(keys + [pid]).agg(wr=("wr", "sum"), w=("w", "sum"), w2v=("w2v", "sum")).reset_index()
        cw = pc.groupby(keys).w.transform("sum"); npl = pc.groupby(keys).w.transform("size")
        lim = np.maximum(cap, 1 / npl) * cw
        sc = np.minimum(1, lim / pc.w)                        # cap any one player's share of the cell
        pc["wr"], pc["w"], pc["w2v"] = pc.wr * sc, pc.w * sc, pc.w2v * sc ** 2
        c = pc.groupby(keys)[["wr", "w", "w2v"]].sum()
        c["m"], c["se2"] = c.wr / c.w, c.w2v / c.w ** 2
        if est_prev is None:
            mu = np.average(c.m, weights=c.w); tau2 = max(np.average((c.m - mu) ** 2, weights=c.w) - np.average(c.se2, weights=c.w), 1e-6)
            c["est"] = mu + tau2 / (tau2 + c.se2) * (c.m - mu)
        else:
            par = est_prev.reindex(pd.MultiIndex.from_frame(c.index.to_frame()[keys[:-1]]) if len(keys) > 2 else c.index.get_level_values(0) if len(keys) == 2 else None)
            par = pd.Series(np.asarray(par), index=c.index)
            tau2 = max(np.average((c.m - par) ** 2, weights=c.w) - np.average(c.se2, weights=c.w), 1e-6)
            c["est"] = par + tau2 / (tau2 + c.se2) * (c.m - par)
        info.append((" x ".join(keys), len(c), float(np.sqrt(tau2))))
        est_prev = c.est
    hier_replacement.info = info
    keys = lv[-1]
    idx = pd.MultiIndex.from_frame(d[keys]) if len(keys) > 1 else d[keys[0]]
    out = pd.Series(np.asarray(est_prev.reindex(idx)), index=d.index)
    # contexts with no fringe evidence at the finest level fall back to the closest coarser estimate
    return out.fillna(out.mean())

hier_replacement.use_cg = False

def prepare(b, side, r_col=None, w_cap=5.0, press_li=2.0, extra_mask=None, repl_band=REPL_BAND, role_repl=True, w_mode=W_MODE, stake_k=1.0, repl_mode="role", player_cap=0.10):
    """Per player-match sums. Also attaches a ROLE-SPECIFIC replacement baseline to every ball:
       baseline(role) = pooled pressure-weighted score of fringe players in that role, shrunk toward the
       all-roles fringe level. A player's replacement level = average baseline of the balls THEY faced/bowled."""
    pid, pname = SIDES[side]
    r = r_col or f"r_{side}_adj"
    m = b[r].notna() & (b.ball_faced if side == "bat" else True)
    if extra_mask is not None: m &= extra_mask
    d = b.loc[m, [pid, pname, "match_id", "li", "li_band", "year", "batting_order", "phase", "pace_or_spin", "knock_out", "innings", "comp_group", r]].copy()
    lc = d.li.clip(upper=w_cap)
    d["w"] = {"li": lc, "li2": lc ** 2, "excess": (lc - 1).clip(lower=0)}[w_mode]   # how pressure is weighted
    if stake_k != 1.0:                         # match stakes: knockout balls weigh stake_k times more
        d["w"] = d.w * np.where(d.knock_out == 1, stake_k, 1.0)
    d["v"] = d.groupby("li_band")[r].transform("var")
    d["wr"], d["w2v"], d["pb"] = d.w * d[r], d.w ** 2 * d.v, (d.li >= press_li).astype(int)
    d["wv"] = d.w * d.v
    d["role"] = role_of(d, side)
    nb = d.groupby(pid)[r].transform("size")
    fr = d[(nb >= repl_band[0]) & (nb < repl_band[1])]
    overall = fr.wr.sum() / fr.w.sum()
    if repl_mode == "hier":
        base = None
        d["repl"] = hier_replacement(d, fr, side, player_cap, pid)
    elif role_repl:
        rs = fr.groupby("role")[["wr", "w"]].sum()
        base = (rs.wr + ROLE_K * overall) / (rs.w + ROLE_K)
        d["repl"] = d.role.map(base).fillna(overall)
    elif not role_repl:
        base = pd.Series({"all": overall}); d["repl"] = overall
    d["wrepl"] = d.w * d.repl
    pm = (d.groupby([pid, "match_id"])
            .agg(S_wr=("wr", "sum"), S_w=("w", "sum"), S_w2v=("w2v", "sum"), S_r=(r, "sum"), S_wrepl=("wrepl", "sum"),
                 S_wv=("wv", "sum"), S_v=("v", "sum"), n=(r, "size"), npb=("pb", "sum"), year=("year", "first")).reset_index()
            .rename(columns={pid: "pid"}))
    pm.attrs["role_baselines"], pm.attrs["repl_overall"] = base, overall
    names = d.groupby(pid)[pname].first()
    return pm, names

def score(pm, names, side=None, tau_c2=None, weights=(0.6, 0.2, 0.2), min_balls=300, min_pb=30, mw=None, fixed=None, mode="margin", repl_band=REPL_BAND, z_conf=1.2816, robust=False):  # repl_band used only for the info value
    """mw: optional Series match_id -> weight (bootstrap). fixed: params from the real-data run."""
    ww = np.ones(len(pm)) if mw is None else pm.match_id.map(mw).fillna(0).values
    t = pm.assign(**{c: pm[c] * ww for c in ["S_wr", "S_w", "S_w2v", "S_r", "S_wrepl", "S_wv", "S_v", "n"]},
                  hitw=(pm.S_wr > 0) * pm.S_w * ww, w2=pm.S_w ** 2 * ww)
    g = t.groupby("pid")[["S_wr", "S_w", "S_w2v", "S_r", "S_wrepl", "S_wv", "S_v", "n", "hitw", "w2"]].sum()
    g = g[g.S_w > 0]
    g["pc"], g["se2"], g["usual"] = g.S_wr / g.S_w, g.S_w2v / g.S_w ** 2, g.S_r / g.n
    if robust:   # consistency built in: Huber M-estimate across a player's matches (one freak match can't carry him)
        t2 = t[t.S_w > 0].assign(m=lambda x: x.S_wr / x.S_w)
        mu0 = t2.pid.map(g.pc)
        s0 = 1.4826 * np.median(np.abs((t2.m - mu0) * np.sqrt(t2.S_w)))
        est = t2.pid.map(g.pc).values.astype(float)
        for _ in range(25):
            z = (t2.m.values - est) * np.sqrt(t2.S_w.values) / s0
            hw = np.where(np.abs(z) <= 1.345, 1.0, 1.345 / np.maximum(np.abs(z), 1e-9))
            ww = t2.S_w.values * hw
            num = pd.Series(ww * t2.m.values).groupby(t2.pid.values).sum(); den = pd.Series(ww).groupby(t2.pid.values).sum()
            est = t2.pid.map(num / den).values
        g["pc"] = (num / den).reindex(g.index)
    g["clutch"] = g.pc - g.usual
    g["se_u2"] = g.S_v / g.n ** 2
    g["se_c2"] = (g.S_w2v / g.S_w ** 2 - 2 * g.S_wv / (g.S_w * g.n) + g.S_v / g.n ** 2).clip(lower=1e-8)
    if fixed is None:
        tot = pm.groupby("pid")[["n", "npb"]].sum()
        elig = tot[(tot.n >= min_balls) & (tot.npb >= min_pb)].index
        base = g[g.n >= 100]
        tau2 = max(base.pc.var() - base.se2.mean(), 1e-6)
        ge = g.loc[g.index.intersection(elig)]
        lh = ge.hitw.sum() / ge.S_w.sum()
        rate_e, neff_e = ge.hitw / ge.S_w, ge.S_w ** 2 / ge.w2
        th2 = max(rate_e.var() - np.mean(lh * (1 - lh) / neff_e), 1e-5)
        fr = g[(g.n >= repl_band[0]) & (g.n < repl_band[1])]
        repl = fr.S_wr.sum() / fr.S_w.sum()
        # volume-informed prior: pc ~ a + c*log(S_w), weighted by 1/(tau_i^2 + se2), iterate for tau_i^2
        h = g[g.n >= 60]; X = np.column_stack([np.ones(len(h)), np.log(h.S_w)]); t2 = tau2
        for _ in range(20):
            wt = 1 / (t2 + h.se2.values)
            coef = np.linalg.lstsq(X * np.sqrt(wt)[:, None], h.pc.values * np.sqrt(wt), rcond=None)[0]
            res = h.pc.values - X @ coef
            t2 = max(np.average(res ** 2 - h.se2.values, weights=wt), 1e-6)
        bb = g[g.n >= 100]
        tau_u2 = max(bb.usual.var() - bb.se_u2.mean(), 1e-6)
        if tau_c2 is None:   # fallback: method of moments (callers pass the empirical split-half value)
            tau_c2 = max(bb.clutch.var() - bb.se_c2.mean(), 1e-6)
        fixed = dict(tau_u2=tau_u2, tau_c2=tau_c2, elig=elig, tau2=tau2, league_hit=lh, k=lh * (1 - lh) / th2 - 1, repl=repl,
                     prior_coef=coef, tau2_inf=t2)
    g = g.loc[g.index.intersection(fixed["elig"])].copy()
    g["rating"] = g.pc * fixed["tau2"] / (fixed["tau2"] + g.se2)
    g["rate"], g["neff"] = g.hitw / g.S_w, g.S_w ** 2 / g.w2
    g["consistency"] = (g.rate * g.neff + fixed["k"] * fixed["league_hit"]) / (g.neff + fixed["k"])
    g["volume"] = g.rating * g.S_w
    z = lambda x: (x - x.mean()) / x.std()
    wR, wC, wV = weights
    g["final_legacy"] = wR * z(g.rating) + wC * z(g.consistency) + wV * z(g.volume)
    g["repl_level"] = g.S_wrepl / g.S_w                 # player's own role-mix replacement level
    g["vorp"] = (g.rating - g.repl_level) * g.S_w
    a, c = fixed["prior_coef"]
    g["prior_mu"] = a + c * np.log(g.S_w)
    rel = fixed["tau2_inf"] / (fixed["tau2_inf"] + g.se2)
    g["rating_inf"] = g.prior_mu + rel * (g.pc - g.prior_mu)
    # BLEND: usual level + clutch deviation, each trusted exactly as much as its measured reliability
    g["rel_u"] = fixed["tau_u2"] / (fixed["tau_u2"] + g.se_u2); g["rel_c"] = fixed["tau_c2"] / (fixed["tau_c2"] + g.se_c2)
    g["usual_s"], g["clutch_s"] = g.usual * g.rel_u, g.clutch * g.rel_c
    g["rating_blend"] = g.usual_s + g.clutch_s
    g["post_sd"] = np.sqrt(fixed["tau_u2"] * (1 - g.rel_u) + fixed["tau_c2"] * (1 - g.rel_c))
    g["margin"] = g.rating_blend - g.repl_level - z_conf * g.post_sd
    for col, nm in (("margin", "rank_margin"), ("vorp", "rank_vorp"), ("rating_inf", "rank_inf"), ("final_legacy", "rank_legacy")):
        g[nm] = g[col].rank(ascending=False, method="first").astype(int)
    g["final"] = g[{"margin": "margin", "vorp": "vorp", "informed": "rating_inf", "legacy": "final_legacy"}[mode]]
    g["rank"] = g[{"margin": "rank_margin", "vorp": "rank_vorp", "informed": "rank_inf", "legacy": "rank_legacy"}[mode]]
    g["player"] = names.reindex(g.index).values
    return g.sort_values("rank"), fixed
