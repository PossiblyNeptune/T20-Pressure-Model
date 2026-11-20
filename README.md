# Reliable under pressure

**Which T20 batters and bowlers most surely beat a replacement-level player when the match is on the line?**

A ball-by-ball pressure model for IPL and T20I cricket, 2022 to 2026, built for SupeRR Selector Season 2 (Round 1) by Anirudh Chaitanya.

**[Open the interactive dashboard](https://possiblyneptune.github.io/T20-Pressure-Model/)** · **[Read the report (PDF)](report/Reliable_Under_Pressure.pdf)**

| 843 | 182,640 | 16,346 | 131 · 146 |
|---|---|---|---|
| completed matches | live deliveries scored | high-pressure balls | eligible batters · bowlers |

## Results

Players are ranked by their **Reliable Pressure Margin**: the amount by which we are 90% confident a player beats a fringe player in his own role when the match is on the line. The 90% rank range is the span of places a player reaches in 90% of 1,000 resampled versions of the matches.

| # | Batter | Margin | 90% rank range | | # | Bowler | Margin | 90% rank range |
|---|---|---|---|---|---|---|---|---|
| 1 | Suryakumar Yadav | 0.109 | 1–20 | | 1 | Josh Hazlewood | 0.122 | 1–17 |
| 2 | Heinrich Klaasen | 0.100 | 1–25 | | 2 | Jasprit Bumrah | 0.105 | 1–16 |
| 3 | Tim David | 0.098 | 1–37 | | 3 | Axar Patel | 0.098 | 1–19 |
| 4 | Abhishek Sharma | 0.098 | 1–29 | | 4 | Varun Chakravarthy | 0.097 | 1–21 |
| 5 | Shreyas Iyer | 0.095 | 1–45 | | 5 | Sunil Narine | 0.095 | 1–19 |
| 6 | Shubman Gill | 0.075 | 2–44 | | 6 | Krunal Pandya | 0.092 | 1–24 |
| 7 | Andre Russell | 0.071 | 2–64 | | 7 | Rashid Khan | 0.089 | 2–21 |
| 8 | Yashasvi Jaiswal | 0.071 | 3–43 | | 8 | Kuldeep Yadav | 0.085 | 1–29 |
| 9 | Tilak Varma | 0.069 | 3–47 | | 9 | Kagiso Rabada | 0.069 | 4–32 |
| 10 | Phil Salt | 0.064 | 2–63 | | 10 | Yuzvendra Chahal | 0.058 | 3–52 |

The top six batters and top eight bowlers stay in the top 10 under all 14 alternative method settings tested; places 9 and 10 in each list are the closest calls.

## How it works

The full reasoning, with every formula and the checks behind each decision, is in the [report](report/Reliable_Under_Pressure.pdf). In short:

1. **Win probability.** For every ball, a model estimates the batting side's chance of winning from the match situation before that ball. The chase model and the first-innings model are gradient-boosted trees with common-sense monotonic constraints (more wickets in hand never lowers the chance); the last 12 balls of a chase are worked out ball by ball from next-ball outcome odds, then recalibrated. Both are validated with match-grouped 5-fold cross-validation.
2. **Pressure.** The **Leverage Index (LI)** of a ball is how far the next delivery is expected to swing win probability, scaled so the average ball is 1. It depends only on the situation before the ball, so it measures the pressure a player faces, not how he handled it. A ball with LI of 2 or more counts as high pressure.
3. **Ball scores.** Each ball's change in win probability is credited to the batter and the bowler, divided by that ball's leverage, and compared with the league average in the same innings, phase and pressure band. Balls in decided games (LI below 0.1) are dropped.
4. **Fair comparisons.** A ridge regression adjusts for the strength of the opposition, team tier (Full Member, Associate, A-team) and the ground. Each competition-year is centred on its own average, and IPL and T20I scales are linked through players who play in both.
5. **Usual level vs pressure.** Each player's usual level is his average over all live balls. His pressure gap is how much better or worse he does on above-normal-pressure balls. Both are shrunk for sample size, and the pressure gap is counted only as far as it repeats between random halves of the matches.
6. **Reliable Pressure Margin.** The pressure rating minus the level of a fringe player (60–299 balls) in the same role — batting position for batters, phase and pace or spin for bowlers — less 1.28 standard deviations of uncertainty. A player needs 300+ live balls and 30+ high-pressure balls to be ranked.
7. **Robustness.** Ranks are re-computed under 18 alternative settings (14 method choices and 4 data subsets) and over 1,000 bootstrap resamples of matches.

## Repository layout

```
index.html                 Interactive dashboard (single self-contained file; served by GitHub Pages)
report/
  Reliable_Under_Pressure.pdf   Methodology report submitted for Round 1
pipeline/
  01_build_state.py        Clean the ball-by-ball data; rebuild score, wickets, balls left and target before every ball
  02_win_prob_leverage.py  Win-probability models, Leverage Index, per-ball win probability added
  03_ball_values.py        Per-ball scores against the league average in the same situation
  04_fair_ratings.py       Opposition, tier and venue adjustment; season centring; IPL–T20I bridge
  04b_calibrate_clutch.py  Split-half estimate of how much pressure performance repeats between players
  05_final_rankings.py     Final rankings, bootstrap rank ranges, plain-cricket statistics
  06_robustness.py         Rankings under the 18 alternative settings
  07_dashboard_data.py     Every number the dashboard shows, exported to one JSON file
  rank_lib.py              Shared scoring code used by steps 4b, 5 and 6
outputs/
  final_batters.csv        All 131 eligible batters: margin, rank, components, rank range, plain stats
  final_bowlers.csv        All 146 eligible bowlers, same columns
  robustness_batters.csv   Rank of every batter under each alternative setting
  robustness_bowlers.csv   Rank of every bowler under each alternative setting
  dashboard_data.json      Data embedded in the dashboard
```

The most useful columns in `final_*.csv` are `player`, `rank`, `margin` (the Reliable Pressure Margin), `usual_s` and `clutch_s` (usual level and pressure gap after shrinkage), `repl_level` (the role's fringe-player level), `post_sd` (uncertainty), `rank_p05`/`rank_p95` (90% rank range), and the plain statistics under pressure (`P_` columns) and in normal situations (`N_` columns).

## Reproducing the results

The competition data pack is not included in this repository. To rerun the analysis:

1. Put the supplied CSV files (`ball_by_ball.csv`, `matches.csv`, `competitions.csv`, `teams.csv`, `playing_eleven.csv`, `competition_team_player_mapping.csv`) in a folder called `Data Pack/` at the top of the repository.
2. Install the dependencies (Python 3.10 or later):
   ```
   pip install -r requirements.txt
   ```
3. Run the steps in order from the top of the repository:
   ```
   python pipeline/01_build_state.py
   python pipeline/02_win_prob_leverage.py
   python pipeline/03_ball_values.py
   python pipeline/04_fair_ratings.py
   python pipeline/04b_calibrate_clutch.py
   python pipeline/05_final_rankings.py
   python pipeline/06_robustness.py
   python pipeline/07_dashboard_data.py
   ```

The whole run takes a few minutes on a laptop. Intermediate files go to `data_work/` and results to `outputs/`. All random steps use fixed seeds, so a clean run reproduces the published rankings exactly. To see the win-probability models' cross-validation scores, run step 2 with `CV=1` set in the environment.

## Data

Ball-by-ball records for 867 matches from January 2022 to November 2026 (367 IPL, 470 T20I, 30 other T20), supplied by the SupeRR Selector organisers. 843 matches are used after removing no-results and super overs. T20 World Cup matches are not in the data, which is why match importance is not weighted.
