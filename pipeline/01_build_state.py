"""
STEP 1 - Clean the ball-by-ball data and rebuild the match situation before every ball.

Output: data_work/balls_state.parquet  (one row per delivery, with match context attached)

Key decisions (explained in the report):
  * Legal ball      = not a wide and not a no-ball ('wide', 'Overstep', 'other' are illegal).
  * Batter runs     = `runs` column (runs off the bat). Byes/leg-byes are not the batter's.
  * Ball faced      = every delivery except wides (a no-ball IS faced by the batter).
  * Bowler conceded = bat runs + wides + no-ball penalty (extras1 on wide/nb). Byes/leg-byes are NOT charged.
  * Bowler wicket   = any dismissal except run out, retired out, obstructing the field.
  * State columns describe the situation BEFORE the ball is bowled (that's what the player faces).
  * Super overs (innings 3/4) are kept in the file but flagged; they are excluded from the main model.
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data Pack"
OUT = ROOT / "data_work"
OUT.mkdir(exist_ok=True)

b = pd.read_csv(DATA / "ball_by_ball.csv", low_memory=False)
m = pd.read_csv(DATA / "matches.csv")
c = pd.read_csv(DATA / "competitions.csv")

# ---------------- match-level context ----------------
m = m.merge(c[["comp_id", "comp_name", "match_format", "season"]], on="comp_id", how="left")
m["comp_group"] = np.where(m.comp_name.str.contains("IPL"), "IPL",
                   np.where(m.match_format == "T20I", "T20I", "OtherT20"))
res = m.result.fillna("").str.lower()
m["no_result"] = res.str.contains("no result|abandon")
m["dls"] = res.str.contains("d/l|dls") | m.revised_target.notna()
m["super_over"] = res.str.contains("super over")

# ---------------- delivery-level basics ----------------
b = b.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)
ext = b.extras_type.fillna("none")
b["is_wide"] = ext.eq("wide")
b["is_noball"] = ext.isin(["Overstep", "other"])
b["is_legal"] = ~(b.is_wide | b.is_noball)
b["ball_faced"] = ~b.is_wide
b["total_runs"] = b.runs + b.extras1_runs + b.extras_2_runs
b["bat_runs"] = b.runs
b["bowler_runs"] = b.runs + np.where(b.is_wide | b.is_noball, b.extras1_runs, 0)
non_bowler = {"RunOut", "Retire Out", "Obstracting the field", "Obstructing the field"}
b["bowler_wicket"] = b.is_wicket & ~b.dismissal_type.isin(non_bowler)
b["is_dot"] = b.is_legal & (b.total_runs == 0)
b["is_boundary"] = b.is_four | b.is_six
b["is_super_over"] = b.innings > 2

# ---------------- running state BEFORE each ball ----------------
g = b.groupby(["match_id", "innings"], sort=False)
b["score_before"] = g.total_runs.cumsum() - b.total_runs
b["wkts_before"] = g.is_wicket.cumsum() - b.is_wicket.astype(int)
b["legal_before"] = g.is_legal.cumsum() - b.is_legal.astype(int)
b["over_no"] = b.overid                      # 1..20
b["phase"] = pd.cut(b.over_no, [0, 6, 15, 20], labels=["Powerplay", "Middle", "Death"])

# scheduled innings length (balls) - uses revised overs when the match was shortened
mm = m.set_index("match_id")
# NOTE: despite the name, *_revised_overs columns hold BALLS (114 = 19 overs, 30 = 5 overs).
def sched_balls(row_innings, revised):
    return np.where(revised.notna() & (revised > 0), revised, 120)
b["sched_balls"] = 120.0
for inn, col in [(1, "first_inn_revised_overs"), (2, "second_inn_revised_overs")]:
    idx = b.innings == inn
    rev = b.loc[idx, "match_id"].map(mm[col])
    b.loc[idx, "sched_balls"] = sched_balls(inn, rev)
b["balls_left"] = (b.sched_balls - b.legal_before).clip(lower=0)
b["crr"] = np.where(b.legal_before > 0, b.score_before * 6 / b.legal_before.replace(0, np.nan), np.nan)

# target for the chase: revised target if DLS gave one, else 1st innings total + 1
inn1_total = b[b.innings == 1].groupby("match_id").total_runs.sum()
target = (inn1_total + 1).rename("target").to_frame()
target["revised_target"] = target.index.map(mm.revised_target)
target["target"] = target.revised_target.fillna(target.target)
b["target"] = np.where(b.innings == 2, b.match_id.map(target.target), np.nan)
b["runs_needed"] = b.target - b.score_before
b["rrr"] = np.where(b.innings == 2,
                    b.runs_needed * 6 / b.balls_left.replace(0, np.nan), np.nan)

# ---------------- attach match context ----------------
keep = ["comp_id", "comp_name", "comp_group", "season", "match_date", "stage", "knock_out",
        "venue_id", "venue_country", "win_team_id", "result", "no_result", "dls", "super_over"]
b = b.merge(m[["match_id"] + keep], on="match_id", how="left")
b["batting_team_won"] = b.win_team_id == b.batting_team_id

# flag matches whose chase state is unreliable: DLS game where we don't know the revised target
b["chase_state_unreliable"] = (b.innings == 2) & b.dls & b.match_id.map(mm.revised_target).isna()
# ...or the recorded target is inconsistent: chasing side passed it but is not the winner (bad revised target)
bad = b[(b.innings == 2) & (b.runs_needed <= 0) & ~b.batting_team_won & ~b.no_result].match_id.unique()
b.loc[(b.innings == 2) & b.match_id.isin(bad), "chase_state_unreliable"] = True

b["phase"] = b.phase.astype(str)
b.to_parquet(OUT / "balls_state.parquet", index=False)

# ---------------- quick summary ----------------
main = b[~b.is_super_over]
print("deliveries:", len(b), "| matches:", b.match_id.nunique())
print("no-result matches with ball data:", main.loc[main.no_result, "match_id"].nunique())
print("DLS matches:", main.loc[main.dls, "match_id"].nunique(),
      "| of which chase target unknown:", main.loc[main.chase_state_unreliable, "match_id"].nunique())
print("\nmatches by competition group:\n", main.groupby("comp_group").match_id.nunique())
print("\nlegal balls by phase:\n", main[main.is_legal].phase.value_counts().sort_index())
