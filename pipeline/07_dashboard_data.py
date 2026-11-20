"""
STEP 7 - Export everything the dashboard shows into one JSON file (no numbers are typed by hand).
Output: outputs/dashboard_data.json
"""
import pandas as pd, numpy as np, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; W = ROOT / "data_work"; OUT = ROOT / "outputs"
REPLAYS = [20232, 14123, 31242, 50375, 27060, 45512]

b = pd.read_parquet(W / "balls_values_adj.parquet")
f = lambda x, d=3: None if pd.isna(x) else round(float(x), d)

def main_role(side):
    if side == "bat":
        d = b[b.ball_faced]
        pos = pd.cut(d.batting_order, [0, 2, 3, 5, 7, 11], labels=["Opener", "No. 3", "No. 4-5", "Finisher (6-7)", "Lower order"]).astype(str)
        return pos.groupby(d.striker_id).agg(lambda x: x.mode().iat[0])
    r = (b.phase.str.replace("Powerplay", "Powerplay ").str.replace("Middle", "Middle-overs ").str.replace("Death", "Death ")
         + b.pace_or_spin.str.lower())
    return r.groupby(b.bowler_id).agg(lambda x: x.mode().iat[0])

def league(side):
    p = b.pressure_ball
    if side == "bat":
        d = b[b.ball_faced & b.r_bat_adj.notna()]
        g = lambda x: dict(SR=100 * x.bat_runs.sum() / len(x), bpo=len(x) / x.striker_out.sum(),
                           dot=100 * (x.bat_runs == 0).mean(), bdry=100 * x.is_boundary.mean())
    else:
        d = b[b.r_bowl_adj.notna()]
        g = lambda x: dict(econ=6 * x.bowler_runs.sum() / x.is_legal.sum(), bpw=x.is_legal.sum() / x.bowler_wicket.sum(),
                           dot=100 * (x.is_legal & (x.bowler_runs == 0)).sum() / x.is_legal.sum(),
                           bdry=100 * (x.is_boundary & ~x.is_wide).sum() / x.is_legal.sum())
    return {"P": {k: f(v, 1) for k, v in g(d[d.pressure_ball]).items()}, "N": {k: f(v, 1) for k, v in g(d[~d.pressure_ball]).items()}}

out = {"players": {}, "robust": {}, "league": {}, "replays": []}
for side, label in (("bat", "batters"), ("bowl", "bowlers")):
    d = pd.read_csv(OUT / f"final_{label}.csv", index_col=0).sort_values("rank")
    role = main_role(side)
    stat = ["SR", "balls_per_out", "dot%", "bdry%"] if side == "bat" else ["econ", "balls_per_wkt", "dot%", "bdry%"]
    key = ["SR", "bpo", "dot", "bdry"] if side == "bat" else ["econ", "bpw", "dot", "bdry"]
    rows = []
    for pid, r in d.iterrows():
        rows.append(dict(id=int(pid), name=r.player.replace("Namboori ", ""), rank=int(r["rank"]), balls=int(r.n),
                         matches=int(r.matches), pb=int(r.pressure_balls), role=role.get(pid, ""),
                         usual=f(r.usual), clutch=f(r.clutch), usual_s=f(r.usual_s), clutch_s=f(r.clutch_s),
                         rating=f(r.rating_blend), repl=f(r.repl_level), sd=f(r.post_sd), margin=f(r.margin),
                         p10=f(r.p_top10, 3), r05=int(r.rank_p05), r95=int(r.rank_p95), vorp_rank=int(r.rank_vorp),
                         ipl=f(r.ipl_share, 2),
                         P={k: f(r[f"P_{s}"], 1) for k, s in zip(key, stat)}, N={k: f(r[f"N_{s}"], 1) for k, s in zip(key, stat)}))
    out["players"][side] = rows
    out["league"][side] = league(side)
    R = pd.read_csv(OUT / f"robustness_{label}.csv", index_col=0)
    sc = [c for c in R.columns if c != "player"]
    out["robust"][side] = {"scenarios": sc, "ranks": {str(int(pid)): [None if pd.isna(v) else int(v) for v in R.loc[pid, sc]] for pid in R.index}}

# match replays: whole match, WP from the CHASING side's point of view
wp = pd.read_parquet(W / "balls_wp.parquet")
m = pd.read_csv(ROOT / "Data Pack" / "matches.csv").set_index("match_id")
tn = pd.read_csv(ROOT / "Data Pack" / "teams.csv").drop_duplicates("team_id").set_index("team_id").team_name
for mid in REPLAYS:
    x = wp[(wp.match_id == mid) & (wp.innings <= 2)].sort_values(["innings", "innings_ball_num"])
    bat1 = tn[x[x.innings == 1].batting_team_id.iat[0]]; bat2 = tn[x[x.innings == 2].batting_team_id.iat[0]]
    balls = []
    for _, r in x.iterrows():
        w2 = r.wp if r.innings == 2 else 1 - r.wp
        o = "W" if r.is_wicket else ("wd" if r.is_wide else ("nb" if r.is_noball else str(int(r.total_runs))))
        if (r.is_wide or r.is_noball) and r.total_runs > 1: o += f"+{int(r.total_runs) - 1}"
        balls.append([int(r.innings), f"{int(r.overid) - 1}.{int(r.over_ball_num)}", int(r.score_before), int(r.wkts_before),
                      None if pd.isna(r.target) else int(r.target), f(w2), f(r.li, 2), r.striker_name, r.bowler_name, o])
    res = str(m.loc[mid, "result"])
    final = 0.5 if "tie" in res.lower() else (1.0 if m.loc[mid, "win_team_id"] == x[x.innings == 2].batting_team_id.iat[0] else 0.0)
    out["replays"].append(dict(id=int(mid), comp=x.comp_name.iat[0], date=str(m.loc[mid, "match_date"]), bat1=bat1, bat2=bat2,
                               result=res, final=final, balls=balls))
json.dump(out, open(OUT / "dashboard_data.json", "w"), separators=(",", ":"))
print("players", {k: len(v) for k, v in out["players"].items()}, "| replays", [(r["bat1"], r["bat2"], len(r["balls"])) for r in out["replays"]])
print("league", out["league"]); print("size KB", round((OUT / "dashboard_data.json").stat().st_size / 1024))
