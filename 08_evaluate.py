"""
08_evaluate.py  -  error metrics of the paper (Table 3) and Bland-Altman plots

Reads results_all.csv (made by 07_vital_signs.py, one row per patient per recording).
Error = estimate - truth.

Metrics: RMSE, MAE, ErrorSD, ME, MAPE (%), Pearson r.
Condition D (apnea, RR = 0) and L (changing RR) are left out of the RR metrics.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from config import SHOW_PLOTS


# ============================================================
# SETTINGS OF THIS SCRIPT
# ============================================================

RESULTS_FILE = "results_all.csv"

RR_EXCLUDE = {"D", "L"}


# ============================================================
# 1. LOAD RESULTS
# ============================================================

rows = []

with open(RESULTS_FILE) as f:
    for r in csv.DictReader(f):
        rows.append(r)

print(f"{len(rows)} patient results from {len({r['recording'] for r in rows})} recording(s)")


def col(name, keep=None):
    out = []
    for r in rows:
        if keep is not None and r["cond"] in keep:
            continue
        out.append(float(r[name]) if r[name] not in ("", "nan") else np.nan)
    return np.array(out)


# ============================================================
# 2. METRICS
# ============================================================

def metrics(est, gt, skip_zero_truth=True):

    ok = ~np.isnan(est) & ~np.isnan(gt)

    est, gt = est[ok], gt[ok]

    e = est - gt

    out = {
        "n": len(e),
        "RMSE": np.sqrt(np.mean(e ** 2)),
        "MAE": np.mean(np.abs(e)),
        "ErrorSD": np.std(e),
        "ME": np.mean(e),
    }

    nz = gt != 0 if skip_zero_truth else np.ones_like(gt, dtype=bool)

    out["MAPE"] = 100 * np.mean(np.abs(e[nz]) / np.abs(gt[nz]))

    out["r"] = np.corrcoef(est, gt)[0, 1] if len(e) > 2 else np.nan

    out["LoA"] = (out["ME"] - 1.96 * out["ErrorSD"], out["ME"] + 1.96 * out["ErrorSD"])

    return out, est, gt


hr, hr_est, hr_gt = metrics(col("HR_est"), col("HR_true"))

rr, rr_est, rr_gt = metrics(col("RR_est", RR_EXCLUDE), col("RR_true", RR_EXCLUDE))


# ============================================================
# 3. PRINT  (with the paper's values for comparison)
# ============================================================

print(f"\n{'':10}{'n':>5}{'RMSE':>8}{'MAE':>8}{'ErrSD':>8}{'ME':>8}{'MAPE%':>8}{'r':>8}")

for name, m in (("HR (yours)", hr), ("RR (yours)", rr)):
    print(f"{name:10}{m['n']:5d}{m['RMSE']:8.2f}{m['MAE']:8.2f}{m['ErrorSD']:8.2f}"
          f"{m['ME']:8.2f}{m['MAPE']:8.2f}{m['r']:8.3f}")

print(f"{'HR (paper)':10}{'':5}{7.91:8.2f}{6.60:8.2f}{7.89:8.2f}{-0.63:8.2f}{6.11:8.2f}{0.92:8.3f}")
print(f"{'RR (paper)':10}{'':5}{1.80:8.2f}{1.47:8.2f}{1.71:8.2f}{0.54:8.2f}{8.43:8.2f}{0.72:8.3f}")

if hr["n"] < 10:
    print("\nNote: fewer than 10 points. Metrics from so few points are not reliable; "
          "process more recordings (run_all.py).")


# ============================================================
# 4. BLAND-ALTMAN PLOTS
# ============================================================

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

for a, name, est, gt, m, unit in (
    (ax[0], "Heart rate", hr_est, hr_gt, hr, "bpm"),
    (ax[1], "Breathing rate", rr_est, rr_gt, rr, "breaths/min"),
):
    d = est - gt
    avg = (est + gt) / 2

    a.plot(avg, d, "ko")
    a.axhline(m["ME"], color="r", linestyle="--", label=f"mean bias {m['ME']:.2f}")
    a.axhline(m["LoA"][1], color="g", linestyle="--", label=f"+1.96 SD  {m['LoA'][1]:.2f}")
    a.axhline(m["LoA"][0], color="g", linestyle="--", label=f"-1.96 SD  {m['LoA'][0]:.2f}")

    a.set_xlabel(f"Mean of estimate and truth ({unit})")
    a.set_ylabel("Estimate - truth")
    a.set_title(f"Bland-Altman: {name}")
    a.legend()
    a.grid(True)

plt.tight_layout()

plt.savefig("bland_altman.png", dpi=300)

if SHOW_PLOTS:
    plt.show()

print("\nSaved: bland_altman.png")
