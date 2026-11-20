"""
STEP 5 - Final rankings, bootstrap uncertainty, plain-cricket evidence.

MAIN RANKING: Reliable Pressure Margin (rank_lib.py, mode="margin")
   pressure rating = usual level (shrunk) + clutch (shrunk by its empirically calibrated reliability, step 4b)
                     clutch = performance on above-normal-pressure balls (weight LI-1) minus usual level
   margin          = 90% lower bound of (pressure rating - role-specific replacement level)
Also reported: Pressure VORP (total value over a fringe player), consistency (evidence only).
Bootstrap: 1000 resamples of matches (Poisson(1) weights shared by all players in a match).
Output: outputs/final_batters.csv, outputs/final_bowlers.csv
"""
import pandas as pd, numpy as np, warnings, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); from rank_lib import prepare, score
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; W = ROOT / "data_work"; OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
N_BOOT = 1000
pd.set_option("display.width", 250)
b = pd.read_parquet(W / "balls_values_adj.parquet")

def plain_stats(d, pid, side):
    p = d.pressure_ball
    if side == "bat":
        f = lambda x: pd.Series({"balls": len(x), "SR": 100 * x.bat_runs.sum() / len(x),
                                 "balls_per_out": len(x) / max(x.striker_out.sum(), 1),
                                 "dot%": 100 * (x.bat_runs == 0).mean(), "bdry%": 100 * x.is_boundary.mean()})
    else:
        f = lambda x: pd.Series({"balls": x.is_legal.sum(), "econ": 6 * x.bowler_runs.sum() / max(x.is_legal.sum(), 1),
                                 "balls_per_wkt": x.is_legal.sum() / max(x.bowler_wicket.sum(), 1),
                                 "dot%": 100 * (x.is_legal & (x.bowler_runs == 0)).sum() / max(x.is_legal.sum(), 1),
                                 "bdry%": 100 * (x.is_boundary & ~x.is_wide).sum() / max(x.is_legal.sum(), 1)})
    pr = d[p].groupby(pid).apply(f).add_prefix("P_"); nr = d[~p].groupby(pid).apply(f).add_prefix("N_")
    lg = pd.concat([f(d[p]).add_prefix("P_"), f(d[~p]).add_prefix("N_")]).to_frame("-- league avg --").T
    return pr.join(nr), lg

for side, label in (("bat", "batters"), ("bowl", "bowlers")):
    TAU_C = json.load(open(W / "clutch_tau.json"))
    pm, names = prepare(b, side)
    res, fx = score(pm, names, tau_c2=TAU_C[side])
    matches = pm.match_id.unique(); rng = np.random.default_rng(42)
    R = np.zeros((N_BOOT, len(res))); Rv = R.copy()
    for i in range(N_BOOT):
        r_, _ = score(pm, names, mw=pd.Series(rng.poisson(1.0, len(matches)), index=matches), fixed=fx)
        r_ = r_.reindex(res.index); R[i], Rv[i] = r_["rank"], r_["rank_vorp"]
    res["p_top10"], res["p_top10_vorp"] = (R <= 10).mean(0), (Rv <= 10).mean(0)
    res["rank_p05"], res["rank_p50"], res["rank_p95"] = (np.percentile(R, q, axis=0).astype(int) for q in (5, 50, 95))
    rt = pd.read_csv(W / f"ratings_{side}.csv", index_col=0)
    res = res.join(rt[["matches", "pressure_balls", "pressure_A", "ipl_share"]])
    pid = "striker_id" if side == "bat" else "bowler_id"
    dd = b[b[f"r_{side}_adj"].notna() & (b.ball_faced if side == "bat" else True)]
    ps, lg = plain_stats(dd, pid, side); ps = ps[ps.index.isin(res.index)]
    res = res.join(ps)
    res.to_csv(OUT / f"final_{label}.csv")
    print(f"\n=== {label.upper()} ({len(res)} eligible) | role baselines: "
          + ", ".join(f"{k} {v:+.3f}" for k, v in pm.attrs["role_baselines"].items()))
    print(f"tau usual {np.sqrt(fx['tau_u2']):.3f} | tau clutch {np.sqrt(fx['tau_c2']):.3f} | median reliability: usual "
          f"{res.rel_u.median():.2f}, clutch {res.rel_c.median():.2f}")
    show = ["player", "n", "pressure_balls", "usual", "clutch", "usual_s", "clutch_s", "rating_blend", "repl_level", "post_sd", "margin", "rank", "p_top10",
            "rank_p05", "rank_p95", "vorp", "rank_vorp", "consistency"]
    print(res.head(15)[show].round(3).to_string())
    pc = [c for c in ps.columns if c not in ("P_balls", "N_balls")]
    print("Plain stats, Pressure (P_ = LI>=2) vs Normal (N_):")
    print(pd.concat([res.head(12).set_index("player")[pc], lg[pc]]).round(1).to_string())
