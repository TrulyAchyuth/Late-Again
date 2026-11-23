"""
app.py
------
Streamlit dashboard for "Late Again": Public Transport Delays with Weather & Events.

Run (from the project folder, after running the pipeline):
    streamlit run app.py
"""

import datetime as dt
import json
import subprocess
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning

# Let us import our own modules from the src/ folder
sys.path.append(str(Path(__file__).resolve().parent / "src"))
import live_weather  # noqa: E402
import plots  # noqa: E402
from config import (CLEAN_DATA, EVENT_ORDER, FIGURES_DIR, MODELS_DIR,  # noqa: E402
                    REPORTS_DIR, WEATHER_ORDER)
from predict_tools import explain, plan_routes, predict, what_if  # noqa: E402

st.set_page_config(page_title="Late Again", page_icon="🚌", layout="wide")


# ---------------------------------------------------------------------------
# Loading data and models (cached so they load only once)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(CLEAN_DATA, keep_default_na=False, na_values=[""])


def build_project():
    """Run the whole pipeline (data -> models). Used when the models are missing
    or were saved with a different scikit-learn version (e.g. on a new server)."""
    root = Path(__file__).resolve().parent
    cmd = [sys.executable, str(root / "run_pipeline.py")]
    if (root / "data" / "raw" / "public_transport_delays.csv").exists():
        cmd.append("--skip-generate")
    subprocess.run(cmd, check=True, cwd=root)


def _read_models():
    # Turn "model saved with another scikit-learn version" into an error we can catch
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        clf = joblib.load(MODELS_DIR / "best_classifier.joblib")
        reg = joblib.load(MODELS_DIR / "best_regressor.joblib")
    meta = json.loads((MODELS_DIR / "app_metadata.json").read_text())
    return clf, reg, meta


@st.cache_resource
def load_models():
    try:
        return _read_models()
    except Exception:
        with st.spinner("Training the models for the first time on this computer (about 30 seconds)..."):
            build_project()
        return _read_models()


def show(fig):
    """Display a Matplotlib figure and free its memory."""
    st.pyplot(fig)
    plt.close(fig)


clf, reg, meta = load_models()   # also builds data + models if they are missing
df_all = load_data()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🚌 Late Again")
st.sidebar.warning("All data in this app is **synthetic** (computer-generated). "
                   "Patterns shown are not real-world statistics.")
types = sorted(df_all["transport_type"].unique())
chosen_types = st.sidebar.multiselect("Transport types", types, default=types)
df = df_all[df_all["transport_type"].isin(chosen_types)]
st.sidebar.caption(f"Showing {len(df):,} of {len(df_all):,} trips")

st.title("🚌 Late Again")
st.caption("Public Transport Delays with Weather & Events")

if df.empty:
    st.info("Select at least one transport type in the sidebar.")
    st.stop()

tabs = st.tabs(["📊 Overview", "🛣️ Routes", "🌧️ Weather", "🎉 Events",
                "⏰ Peak hours", "🤖 Model results", "🔮 Predict & plan"])

# ---- Overview ----
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trips", f"{len(df):,}")
    c2.metric("Mean delay", f"{df['arrival_delay_min'].mean():.1f} min")
    c3.metric("Median delay", f"{df['arrival_delay_min'].median():.0f} min")
    c4.metric(f"Delayed (> {meta['delay_threshold_min']} min)", f"{df['delayed'].mean():.0%}")
    left, right = st.columns(2)
    with left:
        show(plots.delay_distribution(df))
    with right:
        show(plots.delay_by_transport(df))
    with st.expander("Show a sample of the data"):
        st.dataframe(df.head(200), width="stretch")

# ---- Routes ----
with tabs[1]:
    show(plots.delay_by_route(df))
    route_table = (df.groupby(["route_id", "transport_type"])
                     .agg(trips=("trip_id", "count"),
                          mean_delay=("arrival_delay_min", "mean"),
                          pct_delayed=("delayed", "mean"))
                     .sort_values("mean_delay", ascending=False).reset_index())
    route_table["mean_delay"] = route_table["mean_delay"].round(2)
    route_table["pct_delayed"] = (route_table["pct_delayed"] * 100).round(1)
    st.dataframe(route_table, width="stretch", hide_index=True)

# ---- Weather ----
with tabs[2]:
    show(plots.delay_by_weather(df))
    st.caption("In this synthetic data, storms, heavy rain and snow were set up to add the most delay, "
               "and metro lines were set up to be least affected.")

# ---- Events ----
with tabs[3]:
    show(plots.delay_by_event(df))
    ev = df[df["event_type"] != "No Event"]
    if not ev.empty:
        st.write(f"Trips during an event: **{len(ev):,}**, mean delay "
                 f"**{ev['arrival_delay_min'].mean():.1f} min** vs "
                 f"**{df[df['event_type'] == 'No Event']['arrival_delay_min'].mean():.1f} min** without.")

# ---- Peak hours ----
with tabs[4]:
    show(plots.delay_by_hour(df))
    show(plots.hour_weekday_heatmap(df))

# ---- Model results ----
with tabs[5]:
    st.subheader("How well do the models predict?")
    st.write(f"Models were trained on {meta['train_rows']:,} trips and tested on "
             f"{meta['test_rows']:,} trips they had never seen. Numbers below come from "
             "`reports/` and are produced by `src/train.py`.")
    b = meta["baselines"]
    st.info(f"Baselines without ML: always guessing the most common class gives "
            f"accuracy {b['classification_majority_class_accuracy']:.3f}; always guessing the average delay "
            f"gives MAE {b['regression_predict_mean_mae']:.2f} min.")

    left, right = st.columns(2)
    with left:
        st.markdown(f"**Classification** (best: {meta['best_classifier']})")
        st.dataframe(pd.read_csv(REPORTS_DIR / "metrics_classification.csv"), hide_index=True)
        st.image(str(FIGURES_DIR / "11_confusion_matrix.png"))
    with right:
        st.markdown(f"**Regression** (best: {meta['best_regressor']})")
        st.dataframe(pd.read_csv(REPORTS_DIR / "metrics_regression.csv"), hide_index=True)
        preds = pd.read_csv(REPORTS_DIR / "test_predictions.csv")
        preds = preds[preds["transport_type"].isin(chosen_types)]
        if not preds.empty:
            show(plots.actual_vs_predicted(preds))
    st.image(str(FIGURES_DIR / "12_feature_importance.png"), width=600)


# ---- Prediction, explanation, what-if and route planner ----
routes = pd.DataFrame(meta["routes"]).set_index("route_id")
route_labels = {rid: f"{rid} ({r.transport_type}, {r.distance_km} km)" for rid, r in routes.iterrows()}

# Default values for the input widgets, stored in st.session_state so that the
# "live weather" button can overwrite them.
DEFAULTS = {
    "route_id": routes.index[0], "trip_date": dt.date.today(), "trip_time": dt.time(8, 0),
    "is_holiday": False, "weather_condition": "Clear", "temperature_c": 15.0,
    "humidity_pct": 65.0, "wind_speed_kmh": 12.0, "precipitation_mm": 0.0,
    "event_type": "No Event", "event_attendance": 20_000, "traffic_congestion_index": 3.0,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def fill_live_weather():
    """Button callback: download real weather and put it into the weather widgets."""
    city = st.session_state["city"]
    lat, lon = live_weather.CITIES[city]
    try:
        w = live_weather.get_weather(lat, lon, st.session_state["trip_date"],
                                     st.session_state["trip_time"].hour)
    except Exception as err:  # no internet, API down, date out of range...
        st.session_state["weather_msg"] = ("error", f"Could not get live weather: {err}")
        return
    st.session_state["weather_condition"] = w["weather_condition"]
    st.session_state["temperature_c"] = float(min(max(w["temperature_c"], -15), 40))
    st.session_state["humidity_pct"] = float(min(max(w["humidity_pct"], 20), 100))
    st.session_state["wind_speed_kmh"] = float(min(max(w["wind_speed_kmh"], 0), 100))
    st.session_state["precipitation_mm"] = float(min(max(w["precipitation_mm"], 0), 50))
    st.session_state["weather_msg"] = (
        "success", f"Filled with Open-Meteo weather for {city} at {w['time']} "
                   f"({w['weather_condition']}, {w['temperature_c']:.1f} °C, "
                   f"{w['precipitation_mm']:.1f} mm rain that day).")


with tabs[6]:
    st.subheader("Describe a trip")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Trip**")
        st.selectbox("Route", routes.index, format_func=route_labels.get, key="route_id")
        st.date_input("Date", key="trip_date")
        st.time_input("Departure time", key="trip_time", step=3600)
        st.checkbox("Public holiday", key="is_holiday")
    with col2:
        st.markdown("**Weather**")
        with st.expander("🌐 Use real weather (Open-Meteo)"):
            st.selectbox("City", list(live_weather.CITIES), key="city")
            st.button("Fetch weather for this date & hour", on_click=fill_live_weather)
            if "weather_msg" in st.session_state:
                kind, text = st.session_state["weather_msg"]
                (st.success if kind == "success" else st.error)(text)
        st.selectbox("Condition", WEATHER_ORDER, key="weather_condition")
        st.slider("Temperature (°C)", -15.0, 40.0, key="temperature_c")
        st.slider("Humidity (%)", 20.0, 100.0, key="humidity_pct")
        st.slider("Wind speed (km/h)", 0.0, 100.0, key="wind_speed_kmh")
        st.slider("Precipitation that day (mm)", 0.0, 50.0, key="precipitation_mm")
    with col3:
        st.markdown("**Events & traffic**")
        st.selectbox("City event", EVENT_ORDER, key="event_type")
        st.slider("Expected attendance", 0, 60_000, step=1_000, key="event_attendance",
                  disabled=(st.session_state["event_type"] == "No Event"))
        st.slider("Traffic congestion (0 = empty, 10 = gridlock)", 0.0, 10.0, key="traffic_congestion_index")

    # Collect everything into one plain dict for predict_tools.py
    ss = st.session_state
    inputs = {
        "route_id": ss["route_id"], "date": ss["trip_date"], "hour": ss["trip_time"].hour,
        "is_holiday": ss["is_holiday"], "weather_condition": ss["weather_condition"],
        "temperature_c": ss["temperature_c"], "humidity_pct": ss["humidity_pct"],
        "wind_speed_kmh": ss["wind_speed_kmh"], "precipitation_mm": ss["precipitation_mm"],
        "event_type": ss["event_type"], "event_attendance": ss["event_attendance"],
        "traffic_congestion_index": ss["traffic_congestion_index"],
    }
    st.caption("Models were trained on synthetic data, so all numbers below are a demo, not a real forecast.")

    sub = st.tabs(["🎯 Prediction & why", "🎚️ What-if", "🧭 Route planner"])

    # ---- Prediction + explanation ----
    with sub[0]:
        prob, minutes = predict(clf, reg, inputs, routes)
        r1, r2, r3 = st.columns(3)
        r1.metric("Chance of delay", f"{prob:.0%}")
        r2.metric("Expected delay", f"{minutes:.1f} min")
        r3.metric("Verdict", "Likely delayed" if prob >= 0.5 else "Likely on time")

        st.markdown("**Why this prediction?**")
        expl = explain(reg, inputs, routes)
        show(plots.explanation_waterfall(expl))
        st.caption("We start from this route on a typical calm day (weekday noon, clear, no event, "
                   "normal traffic) and add each group of your trip's factors. Each bar is that "
                   "group's average effect over every order of adding them (a Shapley value), so the "
                   "bars add up exactly to the prediction.")

    # ---- What-if explorer ----
    with sub[1]:
        options = {
            "Precipitation (mm)": ("precipitation_mm", [float(x) for x in range(0, 51, 2)]),
            "Event attendance": ("event_attendance", list(range(0, 60_001, 2_500))),
            "Traffic congestion": ("traffic_congestion_index", [x / 2 for x in range(0, 21)]),
            "Temperature (°C)": ("temperature_c", [float(x) for x in range(-15, 41, 2)]),
            "Wind speed (km/h)": ("wind_speed_kmh", [float(x) for x in range(0, 101, 5)]),
            "Departure hour": ("hour", list(range(5, 24))),
            "Weather condition": ("weather_condition", WEATHER_ORDER),
        }
        choice = st.selectbox("What do you want to change?", list(options))
        feature, values = options[choice]
        if feature == "event_attendance" and inputs["event_type"] == "No Event":
            st.info("Pick a city event above first; with 'No Event' attendance is always 0.")
        else:
            curve = what_if(clf, reg, inputs, routes, feature, values)
            show(plots.what_if_chart(curve, feature, choice, inputs[feature]))
            st.caption("Everything else stays as you set it above. Only the selected input changes. "
                       "Tree-based models (like Gradient Boosting) cannot extrapolate, so curves go flat "
                       "beyond values that were common in the training data.")

    # ---- Route planner ----
    with sub[2]:
        st.write("Same date, time, weather, event and traffic on every route. Which is likely to run best?")
        pick = st.multiselect("Transport types to compare", sorted(routes["transport_type"].unique()),
                              default=sorted(routes["transport_type"].unique()))
        sort_names = {"expected_delay_min": "Expected delay", "expected_total_min": "Total travel time",
                      "chance_of_delay_%": "Chance of delay"}
        sort_by = st.radio("Sort by", list(sort_names), format_func=sort_names.get, horizontal=True)
        ids = routes.index[routes["transport_type"].isin(pick)]
        if len(ids) == 0:
            st.info("Select at least one transport type.")
        else:
            plan = plan_routes(clf, reg, inputs, routes, ids).sort_values(sort_by).reset_index(drop=True)
            best = plan.iloc[0]
            st.success(f"Best by {sort_names[sort_by].lower()}: **{best['route']}** ({best['type']}), "
                       f"expected delay {best['expected_delay_min']} min, "
                       f"{best['chance_of_delay_%']}% chance of being late.")
            st.dataframe(plan, hide_index=True, width="stretch")
            show(plots.route_plan_chart(plan))
            st.caption("In this synthetic city each route links different stations, so this compares "
                       "routes in general rather than alternatives for one journey.")
