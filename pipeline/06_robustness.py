"""
STEP 6 - Robustness of the final ranking (Reliable Pressure Margin).
  (a) Sensitivity: does the top 10 survive reasonable changes to each method choice / data subset?
  (b) Time stability: 2022-23 vs 2024-26
  (c) Split-half reliability (random match halves) of each component and the final margin
Output: outputs/robustness_{batters,bowlers}.csv
"""
import pandas as pd, numpy as np, warnings, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); from rank_lib import prepare, score
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; W = ROOT / "data_work"; OUT = ROOT / "outputs"
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
b = pd.read_parquet(W / "balls_values_adj.parquet")
TAU_C = json.load(open(W / "clutch_tau.json"))
no_assoc = (b.bat_tier != "Associate") & (b.bowl_tier != "Associate")

def scen(side):
    t = TAU_C[side]
    return {  # name: (prepare kwargs, score kwargs)
        "BASE":                  ({}, {}),
        "80% confidence":        ({}, dict(z_conf=0.8416)),
        "95% confidence":        ({}, dict(z_conf=1.6449)),
        "LI cap 3":              (dict(w_cap=3.0), {}),
        "LI cap 10":             (dict(w_cap=10.0), {}),
        "pressure wt = LI":      (dict(w_mode="li"), {}),
        "pressure wt = LI^2":    (dict(w_mode="li2"), {}),
        "pressure ball LI>=1.5": (dict(press_li=1.5), {}),
        "pressure ball LI>=2.5": (dict(press_li=2.5), {}),
        "clutch trust x0.5":     ({}, dict(tau_c2=t * 0.5)),
        "clutch trust x2":       ({}, dict(tau_c2=t * 2)),
        "single replacement":    (dict(role_repl=False), {}),
        "fringe = 1-299 balls":  (dict(repl_band=(1, 300)), {}),
        "fringe = 60-149 balls": (dict(repl_band=(60, 150)), {}),
        "no adjustments":        ("RAW", {}),
        "no associate matches":  (dict(extra_mask=no_assoc), {}),
        "min 500 balls":         ({}, dict(min_balls=500, min_pb=50)),
        "IPL only":              (dict(extra_mask=b.comp_group == "IPL"), dict(min_balls=200, min_pb=20)),
        "T20I only":             (dict(extra_mask=b.comp_group == "T20I"), dict(min_balls=200, min_pb=20)),
    }

for side, label in (("bat", "batters"), ("bowl", "bowlers")):
    ranks, base = {}, None
    S = scen(side)
    for nm, (pk, sk) in S.items():
        pk = dict(r_col=f"r_{side}") if pk == "RAW" else pk
        sk = {"tau_c2": TAU_C[side], **sk}
        pm, names = prepare(b, side, **pk)
        res, _ = score(pm, names, **sk)
        if base is None: base = res
        ranks[nm] = res["rank"]
    R = pd.DataFrame(ranks).reindex(base.index); R.insert(0, "player", base.player)
    R.to_csv(OUT / f"robustness_{label}.csv")
    method = [c for c in S if c not in ("IPL only", "T20I only", "no associate matches", "min 500 balls")]
    summ = pd.DataFrame({nm: {"top10 same": int(((R[nm] <= 10) & (R.BASE <= 10)).sum()),
                              "spearman": R[["BASE", nm]].dropna().corr(method="spearman").iloc[0, 1]} for nm in S}).T
    print(f"\n=== {label.upper()}: rank of base top-14 under each scenario ('-' = not eligible) ===")
    print(R.head(14).rename(columns=lambda c: c[:12]).to_string(na_rep="-"))
    print(summ.round(2).T.rename(columns=lambda c: c[:12]).to_string())
    stay = R.head(10)[method].le(10).sum(1)
    print(f"Base top-10 kept in top 10 across the {len(method)} METHOD scenarios:", dict(zip(R.head(10).player, stay)))
    near = R[(R.BASE > 10) & (R[method].le(10).sum(1) >= 1)]
    print("Others who enter the top 10 in some method scenario:", dict(zip(near.player, R.loc[near.index, method].le(10).sum(1))))

    # (b) time stability
    out = []
    for msk in (b.year <= 2023, b.year >= 2024):
        pm, names = prepare(b, side, extra_mask=msk); r_, _ = score(pm, names, tau_c2=TAU_C[side], min_balls=150, min_pb=15); out.append(r_)
    j = out[0].join(out[1], lsuffix="_e", rsuffix="_l", how="inner")
    print(f"Time stability 2022-23 vs 2024-26 ({len(j)} players): " +
          ", ".join(f"{c} r={j[c+'_e'].corr(j[c+'_l']):.2f}" for c in ["usual", "pc", "clutch", "rating_blend", "margin"]))
    # (c) split-half
    ms = b.match_id.unique(); rows = []
    for seed in range(16):
        h = pd.Series(np.random.default_rng(seed).integers(0, 2, len(ms)), index=ms); H = []
        for k in (0, 1):
            pm, names = prepare(b, side, extra_mask=b.match_id.map(h) == k)
            r_, _ = score(pm, names, tau_c2=TAU_C[side], min_balls=150, min_pb=15); H.append(r_)
        j = H[0].join(H[1], lsuffix="_1", rsuffix="_2", how="inner")
        rows.append({c: j[c + "_1"].corr(j[c + "_2"]) for c in ["usual", "pc", "clutch", "rating_blend", "margin"]} | {"n": len(j)})
    sh = pd.DataFrame(rows).mean()
    print(f"Split-half (avg {sh.n:.0f} players) -> full-data reliability (Spearman-Brown): " +
          ", ".join(f"{c} {2*sh[c]/(1+sh[c]):.2f}" for c in ["usual", "pc", "clutch", "rating_blend", "margin"]))
