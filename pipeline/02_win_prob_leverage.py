"""
STEP 2 - Win probability (WP) model + Leverage Index (LI) for every ball.

WP  = chance the BATTING team wins, given the situation before the ball.
LI  = how much the NEXT ball is expected to move WP, scaled so the average ball = 1.0.
      LI is computed from the situation only (not from what actually happened on the ball),
      so it measures the pressure the player FACES, not how well they handled it.

Two WP models (gradient-boosted trees with "common-sense" monotonic constraints):
  * Chase model (2nd innings): runs needed, balls left, wickets in hand, required rate, IPL flag
  * 1st-innings model: score, balls left, wickets in hand, run rate, scoring-environment par
Validated with match-grouped 5-fold cross-validation (balls of one match never in both train & test).

Output: data_work/balls_wp.parquet
"""
import pandas as pd, numpy as np, os, sys
CV = os.environ.get("CV", "0") == "1"   # CV=1: run validation only; default: fit + compute leverage
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "data_work"
import time; T0 = time.time()
b = pd.read_parquet(W / "balls_state.parquet")

b = b[~b.is_super_over].copy()
b["wk_in_hand"] = 10 - b.wkts_before
b["is_ipl"] = (b.comp_group == "IPL").astype(int)
b["year"] = pd.to_datetime(b.match_date).dt.year

# scoring environment: average completed 1st-innings total per competition group & year
inn1 = (b[(b.innings == 1) & ~b.no_result].groupby("match_id")
        .agg(tot=("total_runs", "sum"), grp=("comp_group", "first"), yr=("year", "first"),
             sched=("sched_balls", "first")))
inn1 = inn1[inn1.sched == 120]
par = inn1.groupby(["grp", "yr"]).tot.mean().rename("env_par")
b = b.join(par, on=["comp_group", "year"])
b["env_par"] = b.env_par.fillna(inn1.tot.mean())
print("Scoring environment (avg 1st-inns total):\n", par.round(1).unstack(0), "\n")

# ---------- training labels ----------
tie = b.result.str.contains("super over|tie", case=False, na=False)
train_ok = ~b.no_result & ~tie & ~b.chase_state_unreliable & b.win_team_id.notna()
b["y"] = b.batting_team_won.astype(int)

FEATS = {
    2: (["runs_needed", "balls_left", "wk_in_hand", "rrr_c", "is_ipl"], [-1, 1, 1, -1, 0]),
    # run rate deliberately NOT used in innings 1: it is very jumpy after 1-2 balls (a first-ball four = 24 rpo)
    # and creates fake early-innings swings. Score + balls left already carry the same information.
    1: (["score_before", "balls_left", "wk_in_hand", "env_par"], [1, 1, 1, -1]),
}
b["rrr_c"] = b.rrr.clip(upper=36)          # cap: beyond 36/over the chase is effectively dead
b["crr_c"] = b.crr.fillna(0).clip(upper=20)

def make_model(mono):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=200, l2_regularization=1.0,
                                          monotonic_cst=mono, random_state=0)

from sklearn.ensemble import HistGradientBoostingRegressor
def make_reg(mono):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=200, l2_regularization=1.0,
                                         monotonic_cst=mono, random_state=0)

def report(name, y, pred, nm):
    base = np.full(len(y), y.mean())
    print(f"--- {name} ({nm})")
    print(f"  Brier vs actual result {brier_score_loss(y, pred):.4f} (no-skill {brier_score_loss(y, base):.4f})"
          f" | LogLoss {log_loss(y, np.clip(pred,1e-4,1-1e-4)):.4f} | AUC {roc_auc_score(y, pred):.3f}")
    cal = pd.DataFrame({"pred": pred, "y": y}).assign(bin=lambda x: pd.cut(x.pred, np.linspace(0, 1, 11)))
    print("  Calibration (predicted vs actual win rate):")
    print("    " + cal.groupby("bin", observed=True).agg(pred=("pred", "mean"), actual=("y", "mean"), n=("y", "size"))
          .round(3).to_string().replace("\n", "\n    "))

models, b["wp"] = {}, np.nan

# ---- (a) chase model: trained on actual win/loss ----
feats, mono = FEATS[2]
d = b[(b.innings == 2) & train_ok]
X, y, grp = d[feats].values, d.y.values, d.match_id.values
if CV:
  oof = np.zeros(len(d))
  for tr, te in GroupKFold(5).split(X, y, grp):
    oof[te] = make_model(mono).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
  report("Innings 2 WP model", y, oof, f"out-of-fold, {d.match_id.nunique()} matches, {len(d)} balls")
models[2] = make_model(mono).fit(X, y)
idx = b.innings == 2
b.loc[idx, "wp"] = models[2].predict_proba(b.loc[idx, feats].values)[:, 1]

# ---- (b) 1st-innings model: trained on the chase model's verdict on the final total ----
# soft label = 1 - WP(chasing side at ball 1 of the chase), using the real target & chase length
start2 = b[b.innings == 2].groupby("match_id").first()
soft = 1 - models[2].predict_proba(start2[FEATS[2][0]].values)[:, 1]
b["y_soft"] = b.match_id.map(pd.Series(soft, index=start2.index))
feats, mono = FEATS[1]
d = b[(b.innings == 1) & train_ok & b.y_soft.notna()]
X, ys, y, grp = d[feats].values, d.y_soft.values, d.y.values, d.match_id.values
if CV:
  oof = np.zeros(len(d))
  for tr, te in GroupKFold(5).split(X, ys, grp):
    oof[te] = np.clip(make_reg(mono).fit(X[tr], ys[tr]).predict(X[te]), 0.001, 0.999)
  report("Innings 1 WP model", y, oof, f"out-of-fold, {d.match_id.nunique()} matches, {len(d)} balls")
class _Wrap:                       # give the regressor the same predict_proba interface
    def __init__(self, m): self.m = m
    def predict_proba(self, X):
        p = np.clip(self.m.predict(X), 0.001, 0.999); return np.column_stack([1 - p, p])
models[1] = _Wrap(make_reg(mono).fit(X, ys))
idx = b.innings == 1
b.loc[idx, "wp"] = models[1].predict_proba(b.loc[idx, feats].values)[:, 1]

if CV:
    sys.exit(0)

# ---------- next-ball outcome probabilities (by innings & phase) ----------
def outcome(r):
    if r.is_wide or r.is_noball: return "extra"
    if r.is_wicket: return "W"
    return str(min(int(r.bat_runs + r.extras1_runs), 6)).replace("5", "4")
cats = ["extra", "W", "0", "1", "2", "3", "4", "6"]
b["out_cat"] = np.select(
    [b.is_wide | b.is_noball, b.is_wicket],
    ["extra", "W"],
    default=(b.total_runs.clip(upper=6).replace(5, 4)).astype(int).astype(str))
P = (b.groupby(["innings", "phase"], observed=True).out_cat.value_counts(normalize=True)
     .unstack().reindex(columns=cats).fillna(0))
print("\nNext-ball outcome probabilities:\n", P.round(3))
RUNS = {"extra": 1, "W": 0, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "6": 6}

# ---------- ENDGAME: exact WP by dynamic programming for the last DP_BALLS balls of a chase ----------
# Tree models smooth over the last-ball "cliff" (e.g. 4 needed off 1 came out at 87%). For the final
# DP_BALLS balls we instead compute WP exactly from next-ball outcome odds, working back from the end.
DP_BALLS, N_MAX = 12, 150
# Odds depend on (a) required runs per ball - batters attack harder as the ask grows - and (b) batting depth.
RB_EDGES = [0, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 99]       # runs-needed-per-ball buckets
dd = b[(b.innings == 2) & (b.balls_left <= 24) & (b.runs_needed > 0) & ~b.chase_state_unreliable].copy()
dd["rb"] = pd.cut(dd.runs_needed / dd.balls_left, RB_EDGES, labels=False)
dd["wb"] = pd.cut(dd.wkts_before, [-1, 5, 7, 9], labels=False)
cnt = dd.groupby(["wb", "rb"]).size()
PD_full = dd.groupby(["wb", "rb"]).out_cat.value_counts(normalize=True).unstack().reindex(columns=cats).fillna(0)
PD_rb = dd.groupby("rb").out_cat.value_counts(normalize=True).unstack().reindex(columns=cats).fillna(0)
PD = {}
for wb_ in range(3):
    for rb_ in range(len(RB_EDGES) - 1):
        n_ = cnt.get((wb_, rb_), 0)
        # shrink sparse cells toward the all-wickets row for that ask-rate (weight = n/(n+300))
        w_ = n_ / (n_ + 300)
        full = PD_full.loc[(wb_, rb_)].values if n_ > 0 else PD_rb.loc[rb_].values
        PD[(wb_, rb_)] = w_ * full + (1 - w_) * PD_rb.loc[rb_].values
print("\nEndgame odds, 5 down or fewer, by runs-needed-per-ball bucket (W / 4 / 6 shown):")
print(pd.DataFrame({k[1]: v for k, v in PD.items() if k[0] == 0}, index=cats).T[["W", "0", "4", "6"]]
      .set_axis([f"{RB_EDGES[i]}-{RB_EDGES[i+1]}" for i in range(len(RB_EDGES) - 1)]).round(3))
V = np.zeros((N_MAX + 1, DP_BALLS + 1, 11))           # V[needed, balls_left, wickets_in_hand]
for bl in range(0, DP_BALLS + 1):
    for wk in range(0, 11):
        for n in range(0, N_MAX + 1):
            if n <= 0: V[n, bl, wk] = 1.0; continue
            if bl == 0 or wk == 0: V[n, bl, wk] = 0.5 if n == 1 else 0.0; continue
            wb_ = 0 if wk >= 5 else (1 if wk >= 3 else 2)
            rb_ = int(np.searchsorted(RB_EDGES, n / bl, side="left") - 1)
            pr = PD[(wb_, min(max(rb_, 0), len(RB_EDGES) - 2))]
            v = 0.0
            for j, k in enumerate(cats):
                r = RUNS[k]
                if k == "extra":   v += pr[j] * V[max(n - 1, 0), bl, wk]
                elif k == "W":     v += pr[j] * V[n, bl - 1, wk - 1]
                else:              v += pr[j] * V[max(n - r, 0), bl - 1, wk]
            V[n, bl, wk] = v
def wp_dp_raw(needed, balls, wk):
    n = np.clip(np.asarray(needed), 0, N_MAX).astype(int)
    return V[n, np.clip(np.asarray(balls), 0, DP_BALLS).astype(int), np.clip(np.asarray(wk), 0, 10).astype(int)]

# Recalibration (Platt scaling): in real close finishes batters score faster than average odds imply,
# so raw DP is too pessimistic for the chaser. A 2-parameter logistic correction on logit(DP) fixes the
# level while keeping DP's exact shape. Terminal states (won/lost/tied) stay exact.
from sklearn.linear_model import LogisticRegression
_lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
_e = b[(b.innings == 2) & (b.balls_left <= DP_BALLS) & (b.balls_left > 0) & (b.runs_needed > 0) & train_ok]
_Xe, _ye, _ge = _lg(wp_dp_raw(_e.runs_needed, _e.balls_left, _e.wk_in_hand)).reshape(-1, 1), _e.y.values, _e.match_id.values
_oof = np.zeros(len(_e))
for tr, te in GroupKFold(5).split(_Xe, _ye, _ge):
    _oof[te] = LogisticRegression().fit(_Xe[tr], _ye[tr]).predict_proba(_Xe[te])[:, 1]
CAL = LogisticRegression().fit(_Xe, _ye)
print(f"DP recalibration: a={CAL.intercept_[0]:.2f}, b={CAL.coef_[0][0]:.2f} | out-of-fold endgame Brier "
      f"raw {brier_score_loss(_ye, np.clip(np.exp(_Xe[:,0])/(1+np.exp(_Xe[:,0])),0,1)):.4f} -> calibrated {brier_score_loss(_ye, _oof):.4f}")
def wp_dp(needed, balls, wk):
    raw = wp_dp_raw(needed, balls, wk)
    out = CAL.predict_proba(_lg(raw).reshape(-1, 1))[:, 1]
    term = (raw >= 1) | (raw <= 0) | (raw == 0.5) & (np.asarray(balls) <= 0)
    return np.where(term, raw, out)

# compare tree vs DP on real endgame states against actual results
e = b[(b.innings == 2) & (b.balls_left <= DP_BALLS) & train_ok & (b.runs_needed > 0)]
dpv = wp_dp(e.runs_needed, e.balls_left, e.wk_in_hand)
print(f"Endgame ({len(e)} balls): Brier tree {brier_score_loss(e.y, e.wp):.4f} vs DP {brier_score_loss(e.y, dpv):.4f}")
chk = e.assign(dp=dpv, nb=pd.cut(e.runs_needed, [0, 3, 6, 10, 15, 20, 30]), bb=pd.cut(e.balls_left, [0, 1, 3, 6, 12]))
chk = chk[chk.wk_in_hand >= 3].groupby(["bb", "nb"], observed=True).agg(actual=("y", "mean"), DP=("dp", "mean"), tree=("wp", "mean"), n=("y", "size"))
print("Endgame check vs reality (3+ wickets in hand):\n", chk[chk.n >= 40].round(2))
ex = pd.DataFrame({"needed": [4, 3, 10, 16, 12], "balls": [1, 1, 6, 6, 12], "wk": [5, 5, 5, 5, 5]})
ex["tree_WP"] = models[2].predict_proba(np.column_stack([ex.needed, ex.balls, ex.wk, np.clip(ex.needed*6/ex.balls,None,36), np.ones(5)]))[:, 1].round(2)
ex["DP_WP"] = wp_dp(ex.needed, ex.balls, ex.wk).round(2)
print(ex.to_string(index=False))
idx = (b.innings == 2) & (b.balls_left <= DP_BALLS) & b.runs_needed.notna()
b.loc[idx, "wp"] = wp_dp(b.loc[idx, "runs_needed"], b.loc[idx, "balls_left"], b.loc[idx, "wk_in_hand"])

def wp_chase(needed, balls, wk, ipl):
    """WP for the chasing side with terminal rules."""
    needed, balls, wk = map(np.asarray, (needed, balls, wk))
    out = np.full(needed.shape, np.nan)
    won, done = needed <= 0, (wk <= 0) | (balls <= 0)
    out[won] = 1.0
    out[~won & done] = np.where(needed[~won & done] == 1, 0.5, 0.0)
    live = ~won & ~done
    endg = live & (balls <= DP_BALLS)
    if endg.any():
        out[endg] = wp_dp(needed[endg], balls[endg], wk[endg])
    live = live & ~endg
    if live.any():
        rrr = np.clip(needed[live] * 6 / balls[live], None, 36)
        X = np.column_stack([needed[live], balls[live], wk[live], rrr, np.asarray(ipl)[live]])
        out[live] = models[2].predict_proba(X)[:, 1]
    return out

def wp_first(score, balls, wk, env, ipl, sched_legal_bowled):
    score, balls, wk = map(np.asarray, (score, balls, wk))
    out = np.full(score.shape, np.nan)
    done = (wk <= 0) | (balls <= 0)
    if done.any():   # innings over -> value it with the chase model at the start of the chase
        out[done] = 1 - wp_chase(score[done] + 1, np.full(done.sum(), 120), np.full(done.sum(), 10),
                                 np.asarray(ipl)[done])
    live = ~done
    if live.any():
        X = np.column_stack([score[live], balls[live], wk[live], np.asarray(env)[live]])
        out[live] = models[1].predict_proba(X)[:, 1]
    return out

# ---------- Leverage: expected |change in WP| over the possible next-ball outcomes ----------
b["li_raw"] = 0.0
for inn in (1, 2):
    d = b[b.innings == inn]
    probs = P.loc[inn].reindex(d.phase.astype(str)).values          # rows aligned with d
    acc = np.zeros(len(d))
    for j, k in enumerate(cats):
        r, legal, wkt = RUNS[k], int(k != "extra"), int(k == "W")
        if inn == 2:
            new = wp_chase(d.runs_needed.values - r, d.balls_left.values - legal,
                           d.wk_in_hand.values - wkt, d.is_ipl.values)
        else:
            new = wp_first(d.score_before.values + r, d.balls_left.values - legal,
                           d.wk_in_hand.values - wkt, d.env_par.values, d.is_ipl.values,
                           d.legal_before.values + legal)
        acc += probs[:, j] * np.abs(new - d.wp.values)
    b.loc[d.index, "li_raw"] = acc
b.loc[b.chase_state_unreliable, ["wp", "li_raw"]] = np.nan
b["li"] = b.li_raw / b.li_raw.mean()

# ---------- WPA: how much each ball moved WP, credited to the batter and to the bowler ----------
# WPA = WP(after the ball) - WP(before). "After" is computed with the same WP functions, applying only
# the part of the outcome each player is responsible for:
#   batter view : runs off the bat + the striker's own dismissal (wides excluded - batter can't play them)
#   bowler view : runs charged to bowler (bat runs + wides + no-balls) + wickets credited to bowler
#                 (run outs / byes / leg-byes are not the bowler's doing). Bowler WPA = -(change in batting WP).
def wp_after(d, runs, legal, wkt):
    if d.innings.iloc[0] == 2:
        return wp_chase(d.runs_needed.values - runs, d.balls_left.values - legal,
                        d.wk_in_hand.values - wkt, d.is_ipl.values)
    return wp_first(d.score_before.values + runs, d.balls_left.values - legal,
                    d.wk_in_hand.values - wkt, d.env_par.values, d.is_ipl.values,
                    d.legal_before.values + legal)
b["striker_out"] = (b.is_wicket & (b.dismissed_player_id == b.striker_id)
                    & ~b.dismissal_type.isin(["Retire Out"]))
b["wpa_bat"], b["wpa_bowl"] = np.nan, np.nan
for inn in (1, 2):
    d = b[b.innings == inn]
    leg = d.is_legal.astype(int).values
    b.loc[d.index, "wpa_bat"] = wp_after(d, d.bat_runs.values, leg, d.striker_out.astype(int).values) - d.wp.values
    b.loc[d.index, "wpa_bowl"] = -(wp_after(d, d.bowler_runs.values, leg, d.bowler_wicket.astype(int).values) - d.wp.values)
b.loc[b.is_wide, "wpa_bat"] = np.nan                       # wides are not "faced"
b.loc[b.chase_state_unreliable, ["wpa_bat", "wpa_bowl"]] = np.nan

keep = [c for c in b.columns if c not in ("y", "y_soft")]
b["phase"] = b.phase.astype(str)
b[keep].to_parquet(W / "balls_wp.parquet", index=False)
print(f"[runtime so far {time.time()-T0:.0f}s]")

print("\nLeverage by innings & phase (mean, 90th pct):")
print(b.groupby(["innings", "phase"], observed=True).li.describe(percentiles=[.5, .9])[["mean", "50%", "90%"]].round(2))
print("\nShare of balls with LI > 2:", round((b.li > 2).mean(), 3))
# consistency check: WP after last 1st-innings ball vs WP at first chase ball (should be ~1 - each other)
last1 = b[b.innings == 1].groupby("match_id").wp.last()
first2 = b[b.innings == 2].groupby("match_id").wp.first()
j = pd.concat([last1, 1 - first2], axis=1, keys=["end_inn1", "start_inn2"]).dropna()
print("Hand-over consistency: mean |diff| =", round((j.end_inn1 - j.start_inn2).abs().mean(), 3))
