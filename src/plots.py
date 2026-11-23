"""
plots.py
--------
Chart functions shared by the EDA script (eda.py) and the dashboard (app.py).
Every function takes the cleaned DataFrame and returns a Matplotlib figure,
so the same chart can be saved as a PNG or shown in Streamlit.
"""

import matplotlib
matplotlib.use("Agg")  # draw without a screen (needed for scripts/servers)
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401  (used for % axis labels)
import pandas as pd
import seaborn as sns

from config import DAY_ORDER, WEATHER_ORDER, EVENT_ORDER, DELAY_THRESHOLD_MIN

sns.set_theme(style="whitegrid")
MAIN_COLOR = "#2a6f97"


def _order(values, preferred):
    """Keep the preferred order but only for values present in the data."""
    present = set(values)
    return [v for v in preferred if v in present]


def delay_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["arrival_delay_min"], bins=50, color=MAIN_COLOR, ax=ax)
    ax.axvline(DELAY_THRESHOLD_MIN, color="crimson", ls="--",
               label=f"Delay threshold ({DELAY_THRESHOLD_MIN} min)")
    ax.set(title="Distribution of arrival delays", xlabel="Arrival delay (minutes)", ylabel="Number of trips")
    ax.legend()
    fig.tight_layout()
    return fig


def delay_by_transport(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    order = df.groupby("transport_type")["arrival_delay_min"].mean().sort_values().index
    sns.barplot(data=df, x="transport_type", y="arrival_delay_min", order=order,
                color=MAIN_COLOR, errorbar=("ci", 95), ax=ax)
    ax.set(title="Average delay by transport type", xlabel="", ylabel="Mean delay (min)")
    fig.tight_layout()
    return fig


def delay_by_route(df):
    order = df.groupby("route_id")["arrival_delay_min"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="route_id", y="arrival_delay_min", hue="transport_type",
                order=order, dodge=False, showfliers=False, ax=ax)
    ax.set(title="Delay distribution per route (outliers hidden)", xlabel="", ylabel="Arrival delay (min)")
    ax.tick_params(axis="x", rotation=60)
    ax.legend(title="Type", fontsize=8)
    fig.tight_layout()
    return fig


def delay_by_weather(df):
    order = _order(df["weather_condition"].unique(), WEATHER_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.barplot(data=df, x="weather_condition", y="arrival_delay_min", order=order,
                color=MAIN_COLOR, ax=axes[0])
    axes[0].set(title="Mean delay by weather", xlabel="", ylabel="Mean delay (min)")
    axes[0].tick_params(axis="x", rotation=30)
    sns.scatterplot(data=df.sample(min(len(df), 2000), random_state=0), x="precipitation_mm",
                    y="arrival_delay_min", hue="transport_type", alpha=0.5, s=15, ax=axes[1])
    axes[1].set(title="Rainfall vs delay (sample of trips)", xlabel="Precipitation (mm)", ylabel="Delay (min)")
    fig.tight_layout()
    return fig


def delay_by_event(df):
    order = _order(df["event_type"].unique(), EVENT_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.barplot(data=df, x="event_type", y="arrival_delay_min", order=order,
                color=MAIN_COLOR, ax=axes[0])
    axes[0].set(title="Mean delay by city event", xlabel="", ylabel="Mean delay (min)")
    axes[0].tick_params(axis="x", rotation=30)
    sns.barplot(data=df, x="event_type", y="arrival_delay_min", hue="transport_type",
                order=order, errorbar=None, ax=axes[1])
    axes[1].set(title="Event impact by transport type", xlabel="", ylabel="Mean delay (min)")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def delay_by_hour(df):
    hourly = df.groupby(["hour", "is_weekend"])["arrival_delay_min"].mean().reset_index()
    hourly["Day type"] = hourly["is_weekend"].map({0: "Weekday", 1: "Weekend"})
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.lineplot(data=hourly, x="hour", y="arrival_delay_min", hue="Day type", marker="o", ax=ax)
    for start, end in [(7, 9), (17, 19)]:
        ax.axvspan(start - 0.5, end + 0.5, color="orange", alpha=0.15)
    ax.set(title="Mean delay by hour of day (shaded = weekday rush hours)",
           xlabel="Hour of day", ylabel="Mean delay (min)")
    fig.tight_layout()
    return fig


def hour_weekday_heatmap(df):
    table = df.pivot_table(index="day_of_week", columns="hour", values="arrival_delay_min", aggfunc="mean")
    table = table.reindex(_order(table.index, DAY_ORDER))
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(table, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Mean delay (min)"})
    ax.set(title="Mean delay by weekday and hour", xlabel="Hour", ylabel="")
    fig.tight_layout()
    return fig


def correlation_heatmap(df):
    cols = ["arrival_delay_min", "precipitation_mm", "wind_speed_kmh", "temperature_c",
            "humidity_pct", "event_attendance", "traffic_congestion_index",
            "is_peak_hour", "is_weekend", "is_holiday", "distance_km"]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set(title="Correlation between numeric features")
    fig.tight_layout()
    return fig


def actual_vs_predicted(pred_df):
    """pred_df needs columns 'actual' and 'predicted' (from reports/test_predictions.csv)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pred_df["actual"], pred_df["predicted"], alpha=0.35, s=12, color=MAIN_COLOR)
    lim = [min(pred_df["actual"].min(), pred_df["predicted"].min()),
           max(pred_df["actual"].max(), pred_df["predicted"].max())]
    ax.plot(lim, lim, "r--", label="Perfect prediction")
    ax.set(title="Actual vs predicted delay (test set)", xlabel="Actual delay (min)",
           ylabel="Predicted delay (min)")
    ax.legend()
    fig.tight_layout()
    return fig


def explanation_waterfall(expl):
    """Waterfall chart: reference delay -> + each factor group -> predicted delay."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    running = 0.0
    labels = list(expl["factor"])
    for i, (label, value) in enumerate(zip(expl["factor"], expl["minutes"])):
        if i == 0 or i == len(expl) - 1:          # first and last bars start at zero
            ax.bar(i, value, color="#7f8c8d")
            running = value if i == 0 else running
            y_text = value
        else:
            color = "crimson" if value > 0 else "seagreen"
            ax.bar(i, value, bottom=running, color=color)
            running += value
            y_text = max(running, running - value)
        ax.text(i, y_text + 0.3, f"{value:+.1f}" if 0 < i < len(expl) - 1 else f"{value:.1f}",
                ha="center", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([lbl.replace(" on a ", "\non a ") for lbl in labels], fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, 0), hi + 0.12 * (hi - min(lo, 0)))   # room for the value labels
    ax.set(ylabel="Delay (min)", title="Red = adds delay, green = reduces it")
    fig.tight_layout()
    return fig


def what_if_chart(curve, feature, label, current_value):
    """Two side-by-side charts: expected minutes and chance of delay vs one input."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    categorical = not pd.api.types.is_numeric_dtype(curve[feature])
    for ax, col, title in [(axes[0], "expected_delay_min", "Expected delay (min)"),
                           (axes[1], "chance_of_delay", "Chance of delay")]:
        if categorical:
            colors = ["crimson" if v == current_value else MAIN_COLOR for v in curve[feature]]
            ax.bar(curve[feature].astype(str), curve[col], color=colors)
            ax.tick_params(axis="x", rotation=30)
        else:
            ax.plot(curve[feature], curve[col], marker="o", color=MAIN_COLOR)
            ax.axvline(current_value, color="crimson", ls="--", label="Your trip")
            ax.legend()
        ax.set(title=title, xlabel=label)
    axes[1].set_ylim(0, 1)
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    return fig


def route_plan_chart(plan):
    """Horizontal bars: expected delay for each route, coloured by transport type."""
    fig, ax = plt.subplots(figsize=(9, max(3, 0.32 * len(plan))))
    sns.barplot(data=plan, y="route", x="expected_delay_min", hue="type", dodge=False, ax=ax)
    ax.set(title="Expected delay per route for your conditions", xlabel="Expected delay (min)", ylabel="")
    ax.legend(title="Type", fontsize=8)
    fig.tight_layout()
    return fig
