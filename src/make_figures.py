"""Makes the static charts in docs/figures/ used by the README and case study.

Run: python src/make_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis  # noqa: E402
from config import HOURLY_FILE, INCIDENTS_FILE, METRICS_FILE, ROOT  # noqa: E402

OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
INK, AMBER, RED, TEAL, BLUE, GREY = "#1B2A41", "#F2A900", "#D1495B", "#2A9D8F", "#457B9D", "#A3ACB9"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "text.color": INK, "figure.dpi": 150})
inc = pd.read_csv(INCIDENTS_FILE, parse_dates=["timestamp"], dtype={"route": str})
hourly = pd.read_csv(HOURLY_FILE, parse_dates=["hour_ts"])
m = json.loads(METRICS_FILE.read_text())

# snowfall vs a normal hour
snow = analysis.compare_to_expected(analysis.snow_and_rain_bins(hourly), "snowfall")
fig, ax = plt.subplots(figsize=(6.4, 3.6))
cols = [RED if lo > 1 and r >= 1.05 else GREY for r, lo in zip(snow["ratio"], snow["ratio_low"])]
ax.bar(snow["snowfall"].astype(str), snow["ratio"], color=cols,
       yerr=[snow["ratio"] - snow["ratio_low"], snow["ratio_high"] - snow["ratio"]], ecolor=INK, capsize=3)
ax.axhline(1, color=INK, lw=1)
ax.set_ylabel("incidents vs a normal hour")
ax.set_title("Incidents per hour by snowfall (cm/h)", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "snow.png"); plt.close(fig)

# ablation
ab = {"final model": m["regressor"]["test"]["mae"], **{f"no {k}": v["mae"] for k, v in m["ablation"].items()}}
ab = {k.replace(" inputs", ""): v for k, v in ab.items()}
s = pd.Series(ab).sort_values()
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.barh(s.index, s.values, color=[AMBER if k == "final model" else BLUE for k in s.index])
ax.axvline(m["baselines"]["reg_global_median"]["mae"], color=RED, ls="--", lw=1)
ax.text(m["baselines"]["reg_global_median"]["mae"], len(s) - 0.45, " median baseline", color=RED, fontsize=8)
for i, v in enumerate(s.values): ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8)
ax.set_xlabel("MAE on 2024 (minutes, lower is better)"); ax.set_xlim(0, 9.5)
ax.set_title("Delay model error when inputs are removed", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "ablation.png"); plt.close(fig)

# share of major delays by hour
by_hour = inc.groupby("hour")["major_delay"].mean() * 100
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.bar(by_hour.index, by_hour.values, color=[RED if h < 5 else BLUE for h in by_hour.index])
ax.set_xlabel("hour of day"); ax.set_ylabel("% major (15+ min)")
ax.set_title("Share of incidents that become major delays", loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "night.png"); plt.close(fig)
print("saved docs/figures/snow.png, ablation.png, night.png")
