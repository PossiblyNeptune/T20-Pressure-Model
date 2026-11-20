"""
STEP 4 - Fair comparisons: opposition strength, IPL vs T20I check, sample-size shrinkage, eligibility.

(1) Opposition strength + ground. Ridge regression on per-ball scores:
        r_bat  ~ batting-team-year + bowling-team-year + team tier + venue    (fit on balls faced)
        r_bowl ~ bowling-team-year + batting-team-year + team tier + venue    (fit on all balls)
    Then every competition-year is centred on its own average (scoring environment), and the IPL <-> T20I
    level difference is set by the "bridge" of players who play in both.
    Fitting both teams at once means a side that mostly plays weak teams is not flattered.
    Ridge (alpha) shrinks team-years with few balls toward average.
    Adjusted ball score:  batter  r_bat_adj  = r_bat  - (effect of the bowling team faced)
                          bowler  r_bowl_adj = r_bowl - (effect of the batting team bowled to)
(2) Sample-size shrinkage (empirical Bayes). For each player:
        w = min(LI, 5)  (one ball counts at most 5x);  v_i = league per-ball variance for that ball's LI band
        usual       u  = mean r_adj                 SE_u^2 = sum(v_i) / n^2
        pressure_C  pc = sum(w r) / sum(w)          SE_p^2 = sum((w_i/W)^2 v_i)
        clutch      c  = pc - u                     SE_c^2 = sum((w_i/W - 1/n)^2 v_i)
    True-talent spread (tau^2) estimated from the data: var(observed) - avg(SE^2).
    Shrunk:  u* = u * tau_u^2/(tau_u^2+SE_u^2),   c* = c * tau_c^2/(tau_c^2+SE_c^2)
    (i)  rating   = u* + c*                      (talent view: usual level + shrunk clutch gap)
    (ii) rating_D = pc * tau_p^2/(tau_p^2+SE_p^2) (demonstrated view: pressure level shrunk to league avg)
Output: data_work/ratings_bat.csv, ratings_bowl.csv, team_strength.csv
"""
import pandas as pd, numpy as np
from pathlib import Path
from scipy import sparse
from sklearn.linear_model import Ridge
ROOT = Path(__file__).resolve().parents[1]; W = ROOT / "data_work"; D = ROOT / "Data Pack"
ALPHA, MIN_BALLS, MIN_PB = 1000, 300, 30
pd.set_option("display.width", 220)

b = pd.read_parquet(W / "balls_values.parquet")
b["year"] = pd.to_datetime(b.match_date).dt.year
tname = pd.read_csv(D / "teams.csv").drop_duplicates("team_id").set_index("team_id").team_name
b["bat_ty"] = b.batting_team_id.astype(str) + "_" + b.year.astype(str)
b["bowl_ty"] = b.bowling_team_id.astype(str) + "_" + b.year.astype(str)

# Team tiers (ICC status) - lets each team shrink toward ITS tier's average, not the global average.
FULL = {"Afghanistan", "Australia", "Bangladesh", "England", "India", "Ireland", "New Zealand",
        "Pakistan", "South Africa", "Sri Lanka", "West Indies", "Zimbabwe"}
def tier(team_id, comp_group):
    nm = str(tname.get(team_id, ""))
    if comp_group == "IPL": return "IPL"
    if nm.endswith(" A"): return "A_team"
    return "Full" if nm in FULL else "Associate"
b["bat_tier"] = [tier(t, g) for t, g in zip(b.batting_team_id, b.comp_group)]
b["bowl_tier"] = [tier(t, g) for t, g in zip(b.bowling_team_id, b.comp_group)]
TIER_SCALE = 30          # tier columns scaled up => ~900x weaker ridge penalty (tier effects barely shrunk)

def onehot(codes, n):
    cats, idx = np.unique(codes, return_inverse=True)
    return cats, sparse.csr_matrix((np.ones(n), (np.arange(n), idx)), shape=(n, len(cats)))
def two_way(d, y):
    n = len(d)
    ca, Xa = onehot(d.bat_ty.values, n); cb, Xb = onehot(d.bowl_ty.values, n)
    nt = d.comp_group != "IPL"            # IPL tier is not identifiable here (no IPL v national games) -> bridge later
    ta, Ta = onehot(np.where(nt, d.bat_tier, "Full").astype(str), n)
    tb, Tb = onehot(np.where(nt, d.bowl_tier, "Full").astype(str), n)
    keep_a, keep_b = ta != "Full", tb != "Full"   # Full Members = baseline tier
    cv, Xv = onehot(d.venue_id.fillna(-1).astype(int).values, n)     # ground effect (shrunk like teams)
    X = sparse.hstack([Xa, Xb, Ta[:, keep_a] * TIER_SCALE, Tb[:, keep_b] * TIER_SCALE, Xv]).tocsr()
    m = Ridge(alpha=ALPHA, fit_intercept=True).fit(X, d[y].values)
    co = m.coef_; na, nb = len(ca), len(cb)
    team_a, team_b = pd.Series(co[:na], index=ca), pd.Series(co[na:na + nb], index=cb)
    tier_a = pd.Series(co[na + nb:na + nb + keep_a.sum()] * TIER_SCALE, index=ta[keep_a])
    nta, ntb = keep_a.sum(), keep_b.sum()
    tier_b = pd.Series(co[na + nb + nta:na + nb + nta + ntb] * TIER_SCALE, index=tb[keep_b])
    venue = pd.Series(co[na + nb + nta + ntb:], index=cv)
    print(f"  tier effects on {y}: batting-side {tier_a.round(3).to_dict()} | bowling-side {tier_b.round(3).to_dict()}")
    # full opposition effect per team-year = own team effect + its tier effect
    ty_tier_a = d.groupby("bat_ty").bat_tier.first(); ty_tier_b = d.groupby("bowl_ty").bowl_tier.first()
    team_a = team_a + ty_tier_a.reindex(team_a.index).map(tier_a).fillna(0)
    team_b = team_b + ty_tier_b.reindex(team_b.index).map(tier_b).fillna(0)
    return team_a, team_b, venue

# --- (1) opposition strength ---
bat_d = b[b.ball_faced & b.r_bat.notna()]
eff_bat_team, eff_bowl_opp, ven_bat = two_way(bat_d, "r_bat")          # effects on BATTER score
eff_bat_opp, eff_bowl_team, ven_bowl = two_way(b[b.r_bowl.notna()], "r_bowl")   # effects on BOWLER score
vid = b.venue_id.fillna(-1).astype(int)
b["venue_adj_bat"], b["venue_adj_bowl"] = vid.map(ven_bat).fillna(0), vid.map(ven_bowl).fillna(0)
b["opp_adj_bat"] = b.bowl_ty.map(eff_bowl_opp).fillna(0) + b.venue_adj_bat
b["opp_adj_bowl"] = b.bat_ty.map(eff_bat_opp).fillna(0) + b.venue_adj_bowl
b["r_bat_adj"] = b.r_bat - b.opp_adj_bat
b["r_bowl_adj"] = b.r_bowl - b.opp_adj_bowl
hv = b[b.comp_group == "IPL"].groupby("venue_id").home_team.agg(lambda x: x.mode().iat[0]) if "home_team" in b else None
vb = b[b.ball_faced].groupby(vid).venue_adj_bat.first().to_frame("bat_effect").join(b.groupby(vid).size().rename("balls"))
vb = vb[vb.balls >= 1500].sort_values("bat_effect")
print("Venue effects on batter score (1500+ balls; + = batting-friendly): most bowler-friendly / most batter-friendly")
print(pd.concat([vb.head(3), vb.tail(3)]).round(3).to_string())

# Competition-year environment: centre every competition-year (e.g. IPL 2026, T20I 2024) on its own
# pressure-weighted average, so season-wide scoring inflation/deflation (Impact Player rule, pitches)
# doesn't flatter batters of high-scoring seasons or bowlers of low-scoring ones. The IPL <-> T20I level
# difference is then restored by the bridge below.
b["cy"] = b.comp_group + "_" + b.year.astype(str)
wcap = b.li.clip(upper=5)
for side, mask in (("bat", b.ball_faced), ("bowl", b.r_bowl.notna())):
    r = f"r_{side}_adj"; m = mask & b[r].notna()
    cy_mean = (wcap[m] * b.loc[m, r]).groupby(b.loc[m, "cy"]).sum() / wcap[m].groupby(b.loc[m, "cy"]).sum()
    off = b.cy.map(cy_mean).fillna(0)
    b[r] -= off; b[f"opp_adj_{side}"] += off

# IPL <-> international bridge: IPL teams never meet national teams, so the team model can't put them on one
# scale. Players who appear in both act as the bridge: within-player comparison of adjusted scores by group.
def bridge(d, pid, r):
    d = d[d[r].notna()]
    dm = d[r] - d.groupby(pid)[r].transform("mean")
    G = pd.get_dummies(d.comp_group)[["IPL", "OtherT20"]].astype(float)
    Gd = G - G.groupby(d[pid]).transform("mean")
    beta, *_ = np.linalg.lstsq(Gd.values, dm.values, rcond=None)
    return dict(zip(["IPL", "OtherT20"], beta))
for side, pid, mask in (("bat", "striker_id", b.ball_faced), ("bowl", "bowler_id", np.ones(len(b), bool))):
    br = bridge(b[mask], pid, f"r_{side}_adj")
    print(f"Bridge offsets for {side} (vs T20I, from players in both): " + ", ".join(f"{k} {v:+.3f}" for k, v in br.items()))
    off = b.comp_group.map(br).fillna(0)
    b[f"r_{side}_adj"] -= off
    b[f"opp_adj_{side}"] += off

print("Comp-year check after centring + bridge (pressure-weighted mean; should be flat within each competition):")
chk = pd.DataFrame({side: (wcap * b[f"r_{side}_adj"]).groupby(b.cy).sum() / wcap[b[f"r_{side}_adj"].notna()].groupby(b.cy).sum()
                    for side in ("bat", "bowl")})
print(chk.round(3).T.to_string())

# team strength table (positive = strong). Bowling strength = how much it suppresses batters, etc.
ts = pd.DataFrame({"batting_strength": -eff_bat_opp, "bowling_strength": -eff_bowl_opp}).dropna(how="all")
ts["team"] = [tname.get(int(i.split("_")[0]), i) for i in ts.index]; ts["year"] = [i.split("_")[1] for i in ts.index]
ts["balls"] = ts.index.map(b.groupby("bat_ty").size()).fillna(0).astype(int)
ts["overall"] = ts[["batting_strength", "bowling_strength"]].mean(axis=1)
ts.sort_values("overall", ascending=False).to_csv(W / "team_strength.csv")
q = ts[ts.balls >= 600].sort_values("overall", ascending=False)
print("TEAM-YEAR STRENGTH (>=600 balls batted) - top 8 and bottom 8:")
print(pd.concat([q.head(8), q.tail(8)])[["team", "year", "balls", "batting_strength", "bowling_strength", "overall"]].round(3).to_string(index=False))

# --- (2) IPL vs T20I check (after adjustment) ---
print("\nAverage adjusted score by competition group (should be near 0 if fair):")
print(pd.DataFrame({"bat": b[b.ball_faced].groupby("comp_group").r_bat_adj.mean(),
                    "bowl": b.groupby("comp_group").r_bowl_adj.mean(),
                    "bat_unadj": b[b.ball_faced].groupby("comp_group").r_bat.mean()}).round(3))
for side, pid, mask in (("bat", "striker_id", b.ball_faced), ("bowl", "bowler_id", np.ones(len(b), bool))):
    g = b[mask].groupby([pid, "comp_group"])[f"r_{side}_adj"].agg(["mean", "size"]).unstack()
    both = g[(g[("size", "IPL")] >= 200) & (g[("size", "T20I")] >= 200)]
    diff = both[("mean", "IPL")] - both[("mean", "T20I")]
    print(f"{side}: {len(both)} players with 200+ balls in both | avg (IPL - T20I) = {diff.mean():+.3f} "
          f"(SE {diff.std()/np.sqrt(len(diff)):.3f}) | corr of levels = {both[('mean','IPL')].corr(both[('mean','T20I')]):.2f}")

# --- (3) shrinkage ---
W_CAP = 5.0     # LI weight cap: one last-ball moment counts as at most 5 ordinary balls, not 40
def ratings(d, pid, pname, r):
    d = d[d[r].notna()].copy()
    d["w"] = d.li.clip(upper=W_CAP)
    # per-ball noise differs by pressure band (high-LI balls are less noisy once scaled) -> band-specific variance
    d["v"] = d.groupby("li_band")[r].transform("var")
    g = d.groupby(pid)
    n, W_ = g.size(), g.w.sum()
    u = g[r].mean()
    pc = d.assign(x=d.w * d[r]).groupby(pid).x.sum() / W_
    c = pc - u
    d = d.join(W_.rename("W"), on=pid).join(n.rename("N"), on=pid)
    se_u2 = d.assign(z=d.v / d.N ** 2).groupby(pid).z.sum()
    se_c2 = d.assign(z=(d.w / d.W - 1 / d.N) ** 2 * d.v).groupby(pid).z.sum()
    se_p2_ = d.assign(z=(d.w / d.W) ** 2 * d.v).groupby(pid).z.sum()
    pa = d[d.pressure_ball].groupby(pid)[r].agg(["mean", "size"])
    out = pd.DataFrame({"player": g[pname].agg(lambda x: x.mode().iat[0]), "balls": n,
                        "matches": g.match_id.nunique(), "pressure_balls": pa["size"].reindex(n.index).fillna(0).astype(int),
                        "avg_LI": g.li.mean(), "usual": u, "pressure_C": pc, "clutch": c,
                        "pressure_A": pa["mean"], "se_u": np.sqrt(se_u2), "se_c": np.sqrt(se_c2), "se_p": np.sqrt(se_p2_),
                        "ipl_share": g.comp_group.agg(lambda x: (x == "IPL").mean()),
                        "opp_adj": g[f"opp_adj_{r.split('_')[1]}"].mean()})
    base = out[out.balls >= 100]
    tau_u2 = max(base.usual.var() - (base.se_u ** 2).mean(), 1e-6)
    tau_c2 = max(base.clutch.var() - (base.se_c ** 2).mean(), 1e-6)
    out["rel_u"] = tau_u2 / (tau_u2 + out.se_u ** 2)
    out["rel_c"] = tau_c2 / (tau_c2 + out.se_c ** 2)
    out["usual_s"], out["clutch_s"] = out.usual * out.rel_u, out.clutch * out.rel_c
    out["rating"] = out.usual_s + out.clutch_s                      # (i) predictive talent view
    tau_p2 = max(base.pressure_C.var() - (base.se_p ** 2).mean(), 1e-6)
    out["rel_p"] = tau_p2 / (tau_p2 + out.se_p ** 2)
    out["rating_D"] = out.pressure_C * out.rel_p                      # (ii) demonstrated-under-pressure view
    out["rating_sd"] = np.sqrt(tau_u2 * (1 - out.rel_u) + tau_c2 * (1 - out.rel_c))
    out["eligible"] = (out.balls >= MIN_BALLS) & (out.pressure_balls >= MIN_PB)
    print(f"\n{pname.split('_')[0].upper()}: true-talent sd: usual {np.sqrt(tau_u2):.3f}, "
          f"clutch {np.sqrt(tau_c2):.3f}, pressure_C {np.sqrt(tau_p2):.3f} | median reliability (eligible): usual {out[out.eligible].rel_u.median():.2f}, "
          f"clutch {out[out.eligible].rel_c.median():.2f}, pressure_C {out[out.eligible].rel_p.median():.2f} | eligible players: {out.eligible.sum()}")
    return out.sort_values("rating", ascending=False)

bat = ratings(b[b.ball_faced], "striker_id", "striker_name", "r_bat_adj")
bowl = ratings(b, "bowler_id", "bowler_name", "r_bowl_adj")
bat.to_csv(W / "ratings_bat.csv"); bowl.to_csv(W / "ratings_bowl.csv")
b.to_parquet(W / "balls_values_adj.parquet", index=False)

cols = ["player", "balls", "pressure_balls", "usual", "pressure_C", "pressure_A", "clutch", "rating", "rating_D", "opp_adj", "ipl_share"]
for nm, t in (("BATTERS", bat), ("BOWLERS", bowl)):
    e = t[t.eligible].copy()
    e["rank_i"] = e.rating.rank(ascending=False).astype(int); e["rank_ii"] = e.rating_D.rank(ascending=False).astype(int)
    print(f"\n{nm} - top 15 by (ii) DEMONSTRATED pressure rating, with (i) talent-view rank alongside:")
    print(e.sort_values("rating_D", ascending=False).head(15)[cols + ["rank_i", "rank_ii"]].round(3).to_string())
    print(f"overlap of top-10s: {len(set(e.nsmallest(10,'rank_i').index) & set(e.nsmallest(10,'rank_ii').index))}/10 | "
          f"rank correlation (all eligible) {e.rating.corr(e.rating_D, method='spearman'):.2f}")
