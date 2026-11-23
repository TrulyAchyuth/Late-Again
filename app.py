"""
Late Again: dashboard for TTC bus delays in Toronto (2023-2024).

Run: streamlit run app.py
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

ASSETS = ROOT / "assets"
REPO = "https://github.com/TrulyAchyuth/Late-Again"
CHART = {"displayModeBar": False}

st.set_page_config(page_title="Late Again", page_icon=str(ASSETS / "icon.png"), layout="wide")
st.logo(str(ASSETS / "logo.svg"), size="large", link=REPO)
st.html("<style>.block-container {padding-top: 4rem;}</style>")


# Data and models, loaded once
@st.cache_data
def load_data():
    inc = pd.read_csv(INCIDENTS_FILE, parse_dates=["timestamp"], dtype={"route": str})
    hourly = pd.read_csv(HOURLY_FILE, parse_dates=["hour_ts"])
    preds = pd.read_csv(REPORTS_DIR / "test_predictions.csv.gz")
    importance = pd.read_csv(REPORTS_DIR / "feature_importance.csv")
    metrics = json.loads(METRICS_FILE.read_text())
    return inc, hourly, preds, importance, metrics


def _read_models():
    # if the models were saved with another scikit-learn version, retrain them
    with warnings.catch_warnings():
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
        with st.spinner("Training the models (about a minute)..."):
            subprocess.run([sys.executable, str(ROOT / "src" / "train.py")], check=True, cwd=ROOT / "src")
        return _read_models()


inc, hourly, preds, importance, metrics = load_data()
clf, reg, incident_medians = load_models()
k = analysis.kpis(inc)


def header(title, text):
    st.header(title, anchor=False)
    st.caption(text)


def chart(fig, where=st):
    where.plotly_chart(fig, width="stretch", config=CHART)


def note(text):
    with st.container(border=True):
        st.markdown(text)


# Pages

def overview():
    header("Overview", f"{k['incidents']:,} TTC bus delay incidents from {k['first']:%B %Y} to "
                       f"{k['last']:%B %Y}. Each one is something that held up the next bus.")
    per_day = len(inc) / inc["timestamp"].dt.normalize().nunique()
    a, b, c, d = st.columns(4)
    a.metric("Incidents per day", f"{per_day:.0f}", border=True)
    b.metric("Median delay", f"{k['median_delay']:.0f} min", border=True)
    c.metric(f"Major delays ({MAJOR_DELAY_MIN}+ min)", f"{k['major_share']:.0%}", border=True)
    d.metric("Routes affected", f"{k['routes']}", border=True)
    left, right = st.columns([3, 2])
    chart(plots.monthly_trend(inc), left)
    chart(plots.delay_histogram(inc), right)


def timing():
    header("Timing", "When incidents happen, and when they turn into long delays.")
    left, right = st.columns(2)
    table = analysis.hour_weekday_table(hourly, "incidents")
    chart(plots.heatmap(table, "Incidents per hour", [[0, "#FFFFFF"], [0.5, plots.BLUE], [1, plots.INK]],
                        "incidents/h"), left)
    major = inc.pivot_table(index="day_of_week", columns="hour", values="major_delay", aggfunc="mean") * 100
    chart(plots.heatmap(major, "Share that become major delays (%)",
                        [[0, "#FFFFFF"], [0.5, plots.AMBER], [1, plots.RED]], "% major"), right)
    night = inc[inc["hour"].isin([0, 1, 2, 3, 4])]["major_delay"].mean()
    rush = inc[inc["is_rush_hour"] == 1]["major_delay"].mean()
    note(f"Rush hour has the most incidents, but nights have the worst ones. Between midnight and 5 am, "
         f"**{night:.0%}** of incidents become major delays, compared with {rush:.0%} at rush hour. "
         "Night buses run less often, so one problem leaves a bigger gap.")


def routes():
    header("Routes", "Which routes have the most incidents, and which kinds of incident take longest.")
    rt = analysis.route_table(inc)
    left, right = st.columns(2)
    chart(plots.route_bars(rt), left)
    chart(plots.incident_bars(analysis.incident_table(inc)), right)
    with st.expander("All routes with 200+ incidents"):
        rt = rt.assign(major_share=(rt["major_share"] * 100).round(1))
        st.dataframe(rt.rename(columns={"major_share": "% major", "median_delay": "median delay (min)"}),
                     hide_index=True, width="stretch")


def weather():
    header("Weather", "Incidents in each kind of hour compared with a normal hour at the same time of the "
                      "week. 1.0 means no change. The thin lines show a 90% bootstrap range.")
    binned = analysis.snow_and_rain_bins(hourly)
    snow = analysis.compare_to_expected(binned, "snowfall")
    rain = analysis.compare_to_expected(binned, "rainfall")
    games = analysis.compare_to_expected(hourly.assign(
        when=hourly["jays_game_window"].map({0: "No game", 1: "Game window"})), "when")
    hols = analysis.compare_to_expected(hourly.assign(
        when=hourly["is_holiday"].map({0: "Regular day", 1: "Holiday"})), "when")

    left, right = st.columns(2)
    chart(plots.ratio_chart(snow, "snowfall", "Snowfall (cm per hour)"), left)
    chart(plots.ratio_chart(rain, "rainfall", "Rain (mm per hour)"), right)
    left, right = st.columns(2)
    chart(plots.ratio_chart(games, "when", "Blue Jays home games"), left)
    chart(plots.ratio_chart(hols, "when", "Ontario public holidays"), right)

    heavy = snow[snow["snowfall"] == "Heavy (>1.5)"].iloc[0]
    mod_rain = rain.loc[rain["rainfall"] == "Moderate (1-4)", "ratio"].iloc[0]
    no_weather = metrics["ablation"]["weather inputs"]["mae"]
    note(f"- Heavy snow ({heavy['hours']:.0f} hours) came with **{heavy['ratio']:.1f}x** the usual number "
         f"of incidents. Moderate rain was about {mod_rain:.1f}x.\n"
         "- Blue Jays games made no clear difference. Holidays had about 40% fewer incidents.\n"
         f"- Weather changes how many incidents happen, not how long they last. The delay model without "
         f"weather inputs had the same error ({no_weather:.2f} vs {metrics['regressor']['test']['mae']:.2f} min).")


def model():
    c, r, b, rows = metrics["classifier"], metrics["regressor"], metrics["baselines"], metrics["rows"]
    header("Model", f"Trained on Jan to Sep 2023 ({rows['train']:,} incidents), tuned on Oct to Dec 2023 "
                    f"({rows['valid']:,}) and tested once on 2024 ({rows['test']:,}).")
    gain = 1 - r["test"]["mae"] / b["reg_global_median"]["mae"]
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Delay error (MAE)", f"{r['test']['mae']:.1f} min", border=True,
              help=f"{r['best']}. Guessing the median every time gives {b['reg_global_median']['mae']:.1f} min.")
    a2.metric("Better than baseline", f"{gain:.0%}", border=True)
    a3.metric("Major delay ROC-AUC", f"{c['test']['roc_auc']:.2f}", border=True,
              help=f"Rule based on incident type: {b['clf_incident_rule']['roc_auc']:.2f}")
    a4.metric("Major delay F1", f"{c['test']['f1']:.2f}", border=True,
              help=f"Rule based on incident type: {b['clf_incident_rule']['f1']:.2f}")
    left, right = st.columns(2)
    chart(plots.actual_vs_predicted(preds), left)
    chart(plots.importance_bars(importance), right)

    with st.expander("All models on the validation months"):
        left, right = st.columns(2)
        left.dataframe(pd.DataFrame(c["validation"]).set_index("model"), width="stretch")
        right.dataframe(pd.DataFrame(r["validation"]).set_index("model"), width="stretch")
    with st.expander("Ablation: retrained without some inputs (2024)"):
        ab = pd.DataFrame(metrics["ablation"]).T.rename(columns={"roc_auc": "ROC-AUC", "mae": "MAE (min)"})
        ab.loc["nothing removed"] = [c["test"]["roc_auc"], r["test"]["mae"]]
        st.dataframe(ab.iloc[::-1], width="stretch")
        st.caption("The route matters most. Less frequent routes leave longer gaps when something goes wrong.")


def predictor():
    header("Predict", "Describe an incident and the model estimates how late the next bus will be.")
    route_list = analysis.route_table(inc, min_incidents=100)["route"].tolist()
    incident_types = analysis.incident_table(inc)["incident"].tolist()

    defaults = {"incident": "Mechanical", "route": route_list[0], "direction": "N", "at_station": False,
                "date": dt.date.today(), "hour": 8, "game": False, "weather": "Cloudy",
                "temperature_c": 5.0, "precipitation_mm": 0.0, "snowfall_cm": 0.0, "snow_depth_cm": 0.0,
                "wind_kmh": 15.0, "gust_kmh": 30.0}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    def fill_weather():
        try:
            w = live_weather.get_weather(st.session_state["date"], st.session_state["hour"])
        except Exception as err:
            st.session_state["weather_msg"] = ("error", f"Couldn't get weather from Open-Meteo ({err}). "
                                                        "Enter it by hand instead.")
            return
        for key in ["weather", "temperature_c", "precipitation_mm", "snowfall_cm", "snow_depth_cm",
                    "wind_kmh", "gust_kmh"]:
            st.session_state[key] = w[key] if key == "weather" else round(float(w[key]), 1)
        st.session_state["weather_msg"] = ("ok", f"Toronto weather for {w['time'].replace('T', ' ')}: "
                                                 f"{w['weather'].lower()}, {w['temperature_c']:.0f} °C.")

    form, result = st.columns([1, 2], gap="large")
    with form:
        with st.container(border=True):
            st.subheader("Incident", anchor=False)
            st.selectbox("Type", incident_types, key="incident")
            a, b = st.columns(2)
            a.selectbox("Route", route_list, key="route")
            b.selectbox("Direction", ["N", "S", "E", "W", "Unknown"], key="direction")
            st.toggle("At a station", key="at_station")
            st.subheader("Time", anchor=False)
            st.date_input("Date", key="date")
            st.slider("Hour", 0, 23, key="hour")
            st.toggle("Blue Jays home game", key="game")
            st.subheader("Weather", anchor=False)
            st.button("Get Toronto weather for this hour", on_click=fill_weather, width="stretch")
            if "weather_msg" in st.session_state:
                kind, text = st.session_state["weather_msg"]
                (st.success if kind == "ok" else st.warning)(text)
            st.selectbox("Conditions", WEATHER_ORDER, key="weather")
            a, b = st.columns(2)
            a.number_input("Temperature (°C)", -35.0, 40.0, key="temperature_c", step=1.0)
            b.number_input("Snowfall (cm/h)", 0.0, 10.0, key="snowfall_cm", step=0.1)
            with st.expander("More"):
                st.number_input("Precipitation (mm/h)", 0.0, 50.0, key="precipitation_mm", step=0.5)
                st.number_input("Snow on the ground (cm)", 0.0, 100.0, key="snow_depth_cm", step=1.0)
                st.number_input("Wind (km/h)", 0.0, 120.0, key="wind_kmh", step=1.0)
                st.number_input("Gusts (km/h)", 0.0, 150.0, key="gust_kmh", step=1.0)

    inputs = {key: st.session_state[key] for key in defaults}
    chance, minutes = predict(clf, reg, inputs)
    typical = incident_medians.get(inputs["incident"], k["median_delay"])
    with result:
        a, b, c = st.columns(3)
        a.metric("Expected delay", f"{minutes:.0f} min", border=True)
        b.metric(f"Chance of {MAJOR_DELAY_MIN}+ min", f"{chance:.0%}", border=True)
        c.metric(f"Usual for {inputs['incident'].lower()}", f"{typical:.0f} min", border=True)

        why, whatif, compare = st.tabs(["Why", "What if", "Routes"])
        with why:
            chart(plots.waterfall(explain(reg, inputs)))
            st.caption(f"Starts from a typical incident (mechanical, route {TYPICAL['route']}, weekday noon, "
                       "mild and dry) and adds each group of inputs. The bars are Shapley values, so they "
                       "add up to the prediction.")
        with whatif:
            curve = what_if(clf, reg, inputs, "hour", list(range(24)))
            chart(plots.what_if_line(curve, "hour", "Hour of day", inputs["hour"]))
            kinds = what_if(clf, reg, inputs, "incident", incident_types)
            chart(plots.compare_bars(kinds, "incident", inputs["incident"], "Same time, different incident"))
        with compare:
            top = analysis.route_table(inc, min_incidents=100)["route"].head(20).tolist()
            if inputs["route"] not in top:
                top.append(inputs["route"])
            by_route = what_if(clf, reg, inputs, "route", top)
            by_route["route"] = "Route " + by_route["route"].astype(str)
            chart(plots.compare_bars(by_route, "route", f"Route {inputs['route']}",
                                     "The same incident on the 20 busiest routes"))
    st.caption("The model learned from 2023 data, so treat this as an estimate.")


def about():
    header("About", "Where the data comes from and how the project fits together.")
    st.image(str(ASSETS / "pipeline.png"), width=1000)
    left, right = st.columns(2)
    with left:
        st.subheader("Data", anchor=False)
        st.markdown(f"""
| Source | Licence |
|---|---|
| [TTC Bus Delay Data]({TTC_DATASET_PAGE}), City of Toronto | Open Government Licence, Toronto |
| [Open-Meteo](https://open-meteo.com) hourly weather | CC BY 4.0 |
| [MLB Stats API](https://statsapi.mlb.com) Blue Jays schedule | free public API |
| `holidays` Python package | MIT |

Each row in the TTC data is an incident (a breakdown, a diversion, a security call and so on),
not a trip. Buses that ran on time are not in the data. "Delay" is how late the incident made
the next bus.
""")
    with right:
        st.subheader("Limits", anchor=False)
        st.markdown("""
- On-time buses aren't recorded, so the model can't say how likely a trip is to be late.
- Weather is from one point downtown.
- Events are only Blue Jays home games.
- The delay model explains about 28% of the variation (R² 0.28).
- Delays over 120 minutes were removed as outliers.
""")
        st.markdown(f"Code and write-up on [GitHub]({REPO}). Student project, not affiliated with the TTC.")


pages = [
    st.Page(overview, title="Overview", icon=":material/dashboard:", default=True),
    st.Page(timing, title="Timing", icon=":material/schedule:", url_path="timing"),
    st.Page(routes, title="Routes", icon=":material/route:", url_path="routes"),
    st.Page(weather, title="Weather", icon=":material/ac_unit:", url_path="weather"),
    st.Page(model, title="Model", icon=":material/insights:", url_path="model"),
    st.Page(predictor, title="Predict", icon=":material/directions_bus:", url_path="predict"),
    st.Page(about, title="About", icon=":material/info:", url_path="about"),
]
st.navigation(pages, position="top").run()
