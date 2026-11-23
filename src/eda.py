"""
eda.py
------
Exploratory Data Analysis: saves charts to reports/figures/ and a short
summary table to reports/eda_summary.md.

Remember: the data is synthetic, so these patterns describe the simulation,
not a real city.

Run:
    python src/eda.py
"""

import matplotlib.pyplot as plt
import pandas as pd

import plots
from config import CLEAN_DATA, FIGURES_DIR, REPORTS_DIR

CHARTS = {
    "01_delay_distribution.png": plots.delay_distribution,
    "02_delay_by_transport.png": plots.delay_by_transport,
    "03_delay_by_route.png": plots.delay_by_route,
    "04_delay_by_weather.png": plots.delay_by_weather,
    "05_delay_by_event.png": plots.delay_by_event,
    "06_delay_by_hour.png": plots.delay_by_hour,
    "07_hour_weekday_heatmap.png": plots.hour_weekday_heatmap,
    "08_correlation_heatmap.png": plots.correlation_heatmap,
}


def summary_table(df, column):
    """Mean delay, % delayed and trip count for each value of a column."""
    return (df.groupby(column)
              .agg(trips=("trip_id", "count"),
                   mean_delay_min=("arrival_delay_min", "mean"),
                   pct_delayed=("delayed", "mean"))
              .assign(pct_delayed=lambda t: (t["pct_delayed"] * 100).round(1),
                      mean_delay_min=lambda t: t["mean_delay_min"].round(2))
              .sort_values("mean_delay_min", ascending=False))


def main():
    df = pd.read_csv(CLEAN_DATA, keep_default_na=False, na_values=[""])
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for name, func in CHARTS.items():
        fig = func(df)
        fig.savefig(FIGURES_DIR / name, dpi=120)
        plt.close(fig)
        print(f"Saved {name}")

    lines = ["# EDA summary (synthetic data)\n",
             f"Trips: {len(df):,}  |  Mean delay: {df['arrival_delay_min'].mean():.2f} min  |  "
             f"Delayed (> threshold): {df['delayed'].mean():.1%}\n"]
    for col in ["transport_type", "weather_condition", "event_type", "is_peak_hour", "is_weekend", "is_holiday"]:
        lines.append(f"\n## By {col}\n")
        lines.append(summary_table(df, col).to_markdown())
        lines.append("\n")
    (REPORTS_DIR / "eda_summary.md").write_text("\n".join(lines))
    print("Saved reports/eda_summary.md")


if __name__ == "__main__":
    main()
