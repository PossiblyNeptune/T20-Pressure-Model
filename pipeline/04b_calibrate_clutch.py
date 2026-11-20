"""
STEP 4b - Empirical calibration of how much "clutch" (pressure-only performance minus usual level) really
varies between players. Uses split-half reliability: split matches randomly in two, measure each player's
clutch in both halves; the correlation x observed variance = true (repeatable) variance.
This avoids relying on the noise formula (which over-stated bowler clutch on the full data).
Output: data_work/clutch_tau.json  {side: tau_c^2 on the full-data scale}
"""
import pandas as pd, numpy as np, sys, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); from rank_lib import prepare, score, W_MODE
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]; W = ROOT / "data_work"
b = pd.read_parquet(W / "balls_values_adj.parquet"); ms = b.match_id.unique(); out = {}
for side in ("bat", "bowl"):
    rows = []
    for seed in range(16):
        h = pd.Series(np.random.default_rng(seed).integers(0, 2, len(ms)), index=ms); H = []
        for k in (0, 1):
            pm, nm = prepare(b, side, w_mode=W_MODE, extra_mask=b.match_id.map(h) == k)
            g, _ = score(pm, nm, min_balls=150, min_pb=10); H.append(g)
        j = H[0].join(H[1], lsuffix="_1", rsuffix="_2", how="inner")
        rows.append(dict(cov=np.cov(j.clutch_1, j.clutch_2)[0, 1], r=j.clutch_1.corr(j.clutch_2)))
    r = pd.DataFrame(rows)
    out[side] = float(max(r["cov"].mean(), 1e-6))     # covariance across halves = true variance
    print(f"{side}: split-half clutch r = {r.r.mean():.2f} (sd over splits {r.r.std():.2f}) -> true clutch sd {np.sqrt(out[side]):.3f}")
json.dump(out, open(W / "clutch_tau.json", "w"), indent=1)
