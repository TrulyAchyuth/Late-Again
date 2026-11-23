"""
app.py
------
Late Again: a dashboard about real Toronto bus delays (TTC, 2023-2024),
the weather and events around them, and a model that predicts how long a
reported incident will hold the next bus up.

Run:
    streamlit run app.py
"""

import datetime as dt
import json
import subprocess
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))
import analysis  # noqa: E402
import live_weather  # noqa: E402
import plots  # noqa: E402
from config import (HOURLY_FILE, INCIDENTS_FILE, MAJOR_DELAY_MIN, METRICS_FILE,  # noqa: E402
                    MODELS_DIR, REPORTS_DIR, TTC_DATASET_PAGE, WEATHER_ORDER)
from predict_tools import TYPICAL, explain, predict, what_if  # noqa: E402

st.set_page_config(page_title="Late Again · Toronto bus delays", page_icon="🚌", layout="wide")
st.markdown(f"<style>{(ROOT / 'assets' / 'style.css').read_text()}</style>", unsafe_allow_html=True)
CHART = {"displayModeBar": False}


# ---------------------------------------------------------------------------
# Data and models (cached: loaded once per server)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    inc = pd.read_csv(INCIDENTS_FILE, parse_dates=["timestamp"], dtype={"route": str})
    hourly = pd.read_csv(HOURLY_FILE, parse_dates=["hour_ts"])
    preds = pd.read_csv(REPORTS_DIR / "test_predictions.csv.gz")
    importance = pd.read_csv(REPORTS_DIR / "feature_importance.csv")
    metrics = json.loads(METRICS_FILE.read_text())
    return inc, hourly, preds, importance, metrics


def _read_models():
    with warnings.catch_warnings():   # a model saved by another scikit-learn version -> retrain
        warnings.simplefilter("error", InconsistentVersionWarning)
        clf = joblib.load(MODELS_DIR / "major_delay_classifier.joblib")
        reg = joblib.load(MODELS_DIR / "delay_regressor.joblib")
    medians = joblib.load(MODELS_DIR / "incident_medians.joblib")
    return clf, reg, medians


@st.cache_resource
def load_models():
    try:
        return _read_models()
    except Exception:
        with st.spinner("Training the models on this machine (about a minute)..."):
            subprocess.run([sys.executable, str(ROOT / "src" / "train.py")], check=True, cwd=ROOT / "src")
        return _read_models()


inc, hourly, preds, importance, metrics = load_data()
clf, reg, incident_medians = load_models()
k = analysis.kpis(inc)


def stat(value, label, kind=""):
    return f'<div class="stat {kind}"><div class="value">{value}</div><div class="label">{label}</div></div>'


def stats_row(*cards):
    st.markdown('<div class="stats">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def finding(text, kind=""):
    st.markdown(f'<div class="finding {kind}">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar: where the data comes from
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### About the data")
    st.markdown(
        f"""
**{k['incidents']:,} real delay incidents** reported on Toronto's TTC bus network,
{k['first']:%b %Y} to {k['last']:%b %Y}.

- Delays: [TTC Bus Delay Data]({TTC_DATASET_PAGE}), City of Toronto Open Data
  (Open Government Licence – Toronto)
- Weather: hourly, downtown Toronto, [Open-Meteo](https://open-meteo.com) (CC BY 4.0)
- Events: Blue Jays home games at Rogers Centre, MLB Stats API
- Holidays: Ontario public holidays

Each row is an **incident** (a breakdown, a diversion, a security call...), not a
trip. "Delay" is how late it made the next bus.

Independent student project, not affiliated with the TTC.
"""
    )

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    f"""
<div class="hero">
  <h1>Late again.</h1>
  <p>Toronto's buses logged <strong>{k['incidents']:,} delay incidents</strong> in 2023 and 2024.
  Half of them held the next bus up <strong>{k['median_delay']:.0f} minutes or more</strong>,
  and {k['major_share']:.0%} caused a major delay of {MAJOR_DELAY_MIN}+ minutes.
  What makes it worse: the snow, the route, or the time of night?</p>
  <div class="source">Real TTC open data, joined with hourly weather and city events.</div>
</div>
""",
    unsafe_allow_html=True,
)

tabs = st.tabs(["Overview", "When it happens", "Routes & causes", "Weather & events",
                "How good is the model", "Predict an incident"])

# ---- Overview ----
with tabs[0]:
    per_day = len(inc) / inc["timestamp"].dt.normalize().nunique()
    stats_row(stat(f"{per_day:.0f}", "incidents on an average day"),
              stat(f"{k['median_delay']:.0f} min", "median delay to the next bus"),
              stat(f"{k['major_share']:.0%}", f"incidents with a {MAJOR_DELAY_MIN}+ min delay", "late"),
              stat(f"{k['routes']}", "routes with at least one incident"))
    left, right = st.columns([3, 2])
    left.plotly_chart(plots.monthly_trend(inc), width="stretch", config=CHART)
    right.plotly_chart(plots.delay_histogram(inc), width="stretch", config=CHART)

# ---- When ----
with tabs[1]:
    table = analysis.hour_weekday_table(hourly, "incidents")
    st.plotly_chart(plots.heatmap(table, "Incidents per hour, by weekday and hour",
                                  [[0, "#FFFFFF"], [1, plots.ASPHALT]], "incidents/h"),
                    width="stretch", config=CHART)
    major = inc.pivot_table(index="day_of_week", columns="hour", values="major_delay", aggfunc="mean") * 100
    st.plotly_chart(plots.heatmap(major, "Share of incidents that become major delays (%)",
                                  [[0, "#FFFFFF"], [1, plots.LATE]], "% major"),
                    width="stretch", config=CHART)
    night = inc[inc["hour"].isin([0, 1, 2, 3, 4])]["major_delay"].mean()
    rush = inc[inc["is_rush_hour"] == 1]["major_delay"].mean()
    finding(f"Rush hours have the <b>most</b> incidents, but late night has the <b>worst</b> ones: "
            f"{night:.0%} of incidents between midnight and 5 am become major delays, against {rush:.0%} "
            f"in weekday rush hours. Night buses run less often, so one problem leaves a longer gap.", "late")

# ---- Routes & causes ----
with tabs[2]:
    left, right = st.columns(2)
    left.plotly_chart(plots.route_bars(analysis.route_table(inc)), width="stretch", config=CHART)
    right.plotly_chart(plots.incident_bars(analysis.incident_table(inc)), width="stretch", config=CHART)
    with st.expander("All routes with 200+ incidents"):
        rt = analysis.route_table(inc)
        rt["major_share"] = (rt["major_share"] * 100).round(1)
        st.dataframe(rt.rename(columns={"major_share": "% major", "median_delay": "median delay (min)"}),
                     hide_index=True, width="stretch")

# ---- Weather & events ----
with tabs[3]:
    st.markdown("Each bar compares the number of incidents in those hours with a **normal hour at the same "
                "time of the week** (same year, weekday and hour). 1.0 means no difference. Thin lines "
                "show a 90% bootstrap range: wide ranges mean only a few hours had that weather.")
    binned = analysis.snow_and_rain_bins(hourly)
    snow = analysis.compare_to_expected(binned, "snowfall")
    rain = analysis.compare_to_expected(binned, "rainfall")
    left, right = st.columns(2)
    left.plotly_chart(plots.ratio_chart(snow, "snowfall", "Snowfall"), width="stretch", config=CHART)
    right.plotly_chart(plots.ratio_chart(rain, "rainfall", "Rain"), width="stretch", config=CHART)
    heavy = snow[snow["snowfall"] == "Heavy (>1.5)"].iloc[0]
    finding(f"Snow is the clearest weather effect in the data. In the {heavy['hours']:.0f} hours with heavy "
            f"snowfall the network logged <b>{heavy['ratio']:.1f}x</b> the usual number of incidents. "
            f"Moderate rain adds about {rain.loc[rain['rainfall'] == 'Moderate (1-4)', 'ratio'].iloc[0] - 1:.0%}.",
            "late")
    games = analysis.compare_to_expected(hourly.assign(
        when=hourly["jays_game_window"].map({0: "No game", 1: "Blue Jays home game (1 h before to 4 h after)"})), "when")
    hols = analysis.compare_to_expected(hourly.assign(
        when=hourly["is_holiday"].map({0: "Regular day", 1: "Ontario public holiday"})), "when")
    left, right = st.columns(2)
    left.plotly_chart(plots.ratio_chart(games, "when", "Blue Jays home games"), width="stretch", config=CHART)
    right.plotly_chart(plots.ratio_chart(hols, "when", "Public holidays"), width="stretch", config=CHART)
    finding("Blue Jays games made <b>no measurable difference</b> to the number of incidents. "
            "Public holidays have about 40% fewer, because fewer buses run.", "calm")
    finding("Weather changes <b>how many</b> incidents happen, but not <b>how long</b> each one lasts: "
            "retraining the delay model without any weather inputs left its 2024 error unchanged "
            f"({metrics['ablation']['weather inputs']['mae']:.2f} vs {metrics['regressor']['test']['mae']:.2f} min).")

# ---- Model ----
with tabs[4]:
    c, r, b = metrics["classifier"], metrics["regressor"], metrics["baselines"]
    rows = metrics["rows"]
    st.markdown(f"Trained on Jan–Sep 2023 ({rows['train']:,} incidents), model chosen on Oct–Dec 2023 "
                f"({rows['valid']:,}), then refit on all of 2023 and **tested once on all of 2024** "
                f"({rows['test']:,} incidents it had never seen). No random shuffling, so the model "
                "is always predicting the future from the past.")
    gain = 1 - r["test"]["mae"] / b["reg_global_median"]["mae"]
    stats_row(stat(f"{r['test']['mae']:.1f} min", f"average error, delay model ({r['best']})", "signal"),
              stat(f"{gain:.0%}", f"less error than guessing the median ({b['reg_global_median']['mae']:.1f} min)", "calm"),
              stat(f"{c['test']['roc_auc']:.2f}", f"ROC-AUC, major-delay model (rule of thumb: {b['clf_incident_rule']['roc_auc']:.2f})"),
              stat(f"{c['test']['f1']:.2f}", f"F1 for major delays (rule of thumb: {b['clf_incident_rule']['f1']:.2f})"))
    left, right = st.columns(2)
    left.plotly_chart(plots.actual_vs_predicted(preds), width="stretch", config=CHART)
    right.plotly_chart(plots.importance_bars(importance), width="stretch", config=CHART)

    st.markdown("#### Every model, on the validation months (Oct–Dec 2023)")
    left, right = st.columns(2)
    left.dataframe(pd.DataFrame(c["validation"]).set_index("model"), width="stretch")
    right.dataframe(pd.DataFrame(r["validation"]).set_index("model"), width="stretch")
    st.markdown("#### What happens if we take inputs away? (2024 test set)")
    ab = pd.DataFrame(metrics["ablation"]).T.rename(columns={"roc_auc": "ROC-AUC (major delay)", "mae": "MAE (min)"})
    ab.loc["nothing removed"] = [c["test"]["roc_auc"], r["test"]["mae"]]
    st.dataframe(ab.iloc[::-1], width="stretch")
    st.caption("The route matters most. Routes that run less often leave longer gaps when something goes wrong.")

# ---- Predict ----
with tabs[5]:
    st.markdown("Describe an incident as a dispatcher would report it. The model estimates how late the "
                "next bus will be. It learned from 2023 data, so treat it as an estimate, not a timetable.")
    routes = analysis.route_table(inc, min_incidents=100)["route"].tolist()
    incident_types = analysis.incident_table(inc)["incident"].tolist()

    defaults = {"incident": "Mechanical", "route": routes[0], "direction": "N", "at_station": False,
                "date": dt.date.today(), "hour": 8, "game": False, "weather": "Cloudy",
                "temperature_c": 5.0, "precipitation_mm": 0.0, "snowfall_cm": 0.0, "snow_depth_cm": 0.0,
                "wind_kmh": 15.0, "gust_kmh": 30.0}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    def fill_weather():
        try:
            w = live_weather.get_weather(st.session_state["date"], st.session_state["hour"])
        except Exception as err:
            st.session_state["weather_msg"] = ("error", f"Couldn't get weather from Open-Meteo: {err}. "
                                                        "Enter it below instead.")
            return
        for key in ["weather", "temperature_c", "precipitation_mm", "snowfall_cm", "snow_depth_cm",
                    "wind_kmh", "gust_kmh"]:
            st.session_state[key] = w[key] if key == "weather" else round(float(w[key]), 1)
        st.session_state["weather_msg"] = ("ok", f"Toronto weather for {w['time'].replace('T', ' ')}: "
                                                 f"{w['weather'].lower()}, {w['temperature_c']:.0f} °C.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("##### The incident")
        st.selectbox("What happened", incident_types, key="incident")
        st.selectbox("Route", routes, key="route")
        st.selectbox("Direction", ["N", "S", "E", "W", "Unknown"], key="direction")
        st.toggle("At a station", key="at_station")
    with c2:
        st.markdown("##### When")
        st.date_input("Date", key="date")
        st.slider("Hour of day", 0, 23, key="hour")
        st.toggle("Blue Jays home game around this time", key="game")
        st.button("Use Toronto weather for this date and hour", on_click=fill_weather)
        if "weather_msg" in st.session_state:
            kind, text = st.session_state["weather_msg"]
            (st.success if kind == "ok" else st.warning)(text)
    with c3:
        st.markdown("##### Weather")
        st.selectbox("Conditions", WEATHER_ORDER, key="weather")
        st.number_input("Temperature (°C)", -35.0, 40.0, key="temperature_c", step=1.0)
        st.number_input("Snowfall this hour (cm)", 0.0, 10.0, key="snowfall_cm", step=0.1)
        st.number_input("Precipitation this hour (mm)", 0.0, 50.0, key="precipitation_mm", step=0.5)
        with st.expander("More weather"):
            st.number_input("Snow on the ground (cm)", 0.0, 100.0, key="snow_depth_cm", step=1.0)
            st.number_input("Wind (km/h)", 0.0, 120.0, key="wind_kmh", step=1.0)
            st.number_input("Gusts (km/h)", 0.0, 150.0, key="gust_kmh", step=1.0)

    inputs = {key: st.session_state[key] for key in defaults}
    chance, minutes = predict(clf, reg, inputs)
    typical = incident_medians.get(inputs["incident"], k["median_delay"])
    stats_row(stat(f"{minutes:.0f} min", "expected delay to the next bus", "signal"),
              stat(f"{chance:.0%}", f"chance of a major delay ({MAJOR_DELAY_MIN}+ min)",
                   "late" if chance >= 0.5 else "calm"),
              stat(f"{typical:.0f} min", f"typical for “{inputs['incident']}” across all routes"))

    sub = st.tabs(["Why", "What if", "Compare routes"])
    with sub[0]:
        st.plotly_chart(plots.waterfall(explain(reg, inputs)), width="stretch", config=CHART)
        st.caption(f"Starts from a typical incident (a mechanical problem on route {TYPICAL['route']}, "
                   "weekday noon, mild and dry) and adds your inputs group by group. Each bar is that "
                   "group's average effect over every order of adding them (a Shapley value), so the bars "
                   "add up to the prediction.")
    with sub[1]:
        left, right = st.columns(2)
        curve = what_if(clf, reg, inputs, "hour", list(range(24)))
        left.plotly_chart(plots.what_if_line(curve, "hour", "Hour of day", inputs["hour"]),
                          width="stretch", config=CHART)
        kinds = what_if(clf, reg, inputs, "incident", incident_types)
        right.plotly_chart(plots.compare_bars(kinds, "incident", inputs["incident"],
                                              "Same moment, different incident"),
                           width="stretch", config=CHART)
    with sub[2]:
        top = analysis.route_table(inc, min_incidents=100)["route"].head(20).tolist()
        if inputs["route"] not in top:
            top.append(inputs["route"])
        by_route = what_if(clf, reg, inputs, "route", top)
        by_route["route"] = "Route " + by_route["route"].astype(str)
        st.plotly_chart(plots.compare_bars(by_route, "route", f"Route {inputs['route']}",
                                           "The same incident on the 20 busiest routes"),
                        width="stretch", config=CHART)
