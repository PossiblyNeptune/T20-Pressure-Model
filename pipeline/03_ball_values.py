"""
STEP 3 - Turn every ball into a fair performance score for the batter and the bowler,
and summarise each player's usual level vs pressure level.

Per-ball score (both batter & bowler):
    s  = WPA / LI_raw          -> outcome measured in "typical swings" for that ball (pressure removed)
    s is capped to [-8, 8]     -> stops a single freak ball dominating
    r  = s - league average s in the same situation (innings x phase x pressure band)
         -> "better/worse than a typical player facing this exact kind of situation"
Dead balls (LI < 0.1: result effectively decided) are excluded - nothing at stake, no information.

Player summaries (FanGraphs 'Clutch' logic, adapted):
    usual      = plain average of r over all live balls         (every ball counts the same)
    pressure_C = sum(LI * r) / sum(LI)                         (high-pressure balls count more)  <- main
    pressure_A = average r on balls with LI >= 2                (simple "pressure ball" cut-off)
    normal_A   = average r on balls with LI < 2
    clutch_C   = pressure_C - usual ;  clutch_A = pressure_A - normal_A
Output: data_work/balls_values.parquet, data_work/players_raw_{bat,bowl}.csv
"""
import pandas as pd, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; W = ROOT / "data_work"

DEAD_LI, CAP, PRESS_LI = 0.1, 8.0, 2.0
LI_EDGES = [DEAD_LI, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, np.inf]

b = pd.read_parquet(W / "balls_wp.parquet")
b = b[b.li.notna() & (b.li >= DEAD_LI)].copy()
b["li_band"] = pd.cut(b.li, LI_EDGES, right=False, labels=False)
b["pressure_ball"] = b.li >= PRESS_LI

for side in ("bat", "bowl"):
    s = (b[f"wpa_{side}"] / b.li_raw).clip(-CAP, CAP)
    b[f"s_{side}"] = s
    ctx = ["innings", "phase", "li_band"]
    grp = b.groupby(ctx, observed=True)[f"s_{side}"]
    cell_mean, cell_n = grp.transform("mean"), grp.transform("count")
    broad = b.groupby(["innings", "phase"])[f"s_{side}"].transform("mean")
    k = 200                                                  # shrink small cells toward the broader mean
    expected = (cell_n * cell_mean + k * broad) / (cell_n + k)
    b[f"r_{side}"] = s - expected

b.to_parquet(W / "balls_values.parquet", index=False)

def summarise(d, pid, pname, r):
    d = d[d[r].notna()]
    g = d.assign(lr=d.li * d[r], pr=np.where(d.pressure_ball, d[r], np.nan),
                 nr=np.where(~d.pressure_ball, d[r], np.nan)).groupby(pid)
    out = pd.DataFrame({
        "player": g[pname].agg(lambda x: x.mode().iat[0]),
        "balls": g.size(), "matches": g.match_id.nunique(),
        "pressure_balls": g.pressure_ball.sum(), "avg_LI": g.li.mean(),
        "usual": g[r].mean(), "pressure_C": g.lr.sum() / g.li.sum(),
        "pressure_A": g.pr.mean(), "normal_A": g.nr.mean(),
        "ipl_share": g.comp_group.agg(lambda x: (x == "IPL").mean()),
    })
    out["clutch_C"] = out.pressure_C - out.usual
    out["clutch_A"] = out.pressure_A - out.normal_A
    return out

bat = summarise(b[b.ball_faced], "striker_id", "striker_name", "r_bat")
bowl = summarise(b, "bowler_id", "bowler_name", "r_bowl")
bat.to_csv(W / "players_raw_bat.csv"); bowl.to_csv(W / "players_raw_bowl.csv")

pd.set_option("display.width", 200)
print(f"Live balls used: {len(b)} | pressure balls (LI>={PRESS_LI}): {b.pressure_ball.sum()} ({b.pressure_ball.mean():.1%})")
print("League check - average r should be ~0:", round(b.r_bat.mean(), 4), round(b.r_bowl.mean(), 4))
for nm, t, mn in (("BATTERS", bat, 300), ("BOWLERS", bowl, 300)):
    q = t[(t.balls >= mn) & (t.pressure_balls >= 30)]
    print(f"\n{nm}: {len(t)} players, {len(q)} with >= {mn} balls & >= 30 pressure balls")
    print(q.describe().loc[["mean", "std", "min", "max"], ["balls", "pressure_balls", "usual", "pressure_C", "pressure_A", "clutch_C"]].round(3))
    print(f"PRELIMINARY top 12 by pressure_C (raw, no sample-size adjustment yet):")
    print(q.sort_values("pressure_C", ascending=False).head(12)[["player", "balls", "pressure_balls", "usual", "pressure_C", "pressure_A", "clutch_C", "ipl_share"]].round(3).to_string())
