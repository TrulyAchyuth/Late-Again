# Case study: what really makes Toronto's buses late?

## The question
Commuters blame the snow, the crowds and the time of day. I wanted to check those claims against real data, and to build a tool that estimates how long a bus delay will last the moment an incident is reported.

## The data
The City of Toronto publishes every TTC bus delay incident: 115,850 of them in 2023–2024, each with a time, route, location, incident type and the delay it caused to the next bus. I joined these with hourly weather for downtown Toronto (Open-Meteo), the start times of all 162 Blue Jays home games (MLB Stats API) and Ontario public holidays.

After cleaning (unreadable times, duplicate reports, and delays over 2 hours, which were mostly long diversions and typing errors), 111,750 incidents (96.5%) were left.

The first thing I had to get right was what a row means. The data lists **incidents**, not trips, so on-time buses never appear. That ruled out the naive question "will my bus be late?" and shaped two honest ones:
1. Given an incident, how late will the next bus be, and is it a major (15+ minute) delay?
2. Does weather or an event change **how many** incidents happen? For this I built a table of all 17,544 hours in the two years, including quiet ones.

## Approach
- **No peeking at the future.** I trained on January–September 2023, chose the model on October–December 2023, and tested once on all of 2024.
- **Baselines first.** Every model had to beat a "guess the median" rule and an "average for this incident type" rule.
- **Fair comparisons for weather.** A snowy 8 am Monday was compared with an ordinary 8 am Monday (same year, weekday and hour), not with the yearly average, with a bootstrap range to show uncertainty.
- **Ablation.** I retrained the final models without the weather, event, route and incident-type inputs to see what each one really adds.

## Results (2024 test year)
- The delay model (Gradient Boosting) is off by **6.3 minutes** on average, **21% better** than guessing the median (8.0 minutes).
- The major-delay model reaches **ROC-AUC 0.84** and F1 0.70, against 0.56 and 0.24 for a rule of thumb based on incident type.
- **Heavy snowfall** (over 1.5 cm per hour) came with **2.2x** the normal number of incidents; moderate rain about 1.2x.
- **Blue Jays games made no measurable difference.** Holidays had about 40% fewer incidents.

## What surprised me
- **Weather does not make delays longer.** Removing every weather input left the delay model's error unchanged (6.29 min). Snow causes *more* incidents, but each one takes about as long to clear.
- **The route is the strongest input** (removing it raised the error to 7.6 minutes). Routes with less frequent service leave a longer gap when a bus drops out.
- **Night is worse than rush hour.** Rush hours have the most incidents, but 50% of night-time incidents become major delays, against 32% at rush hour.

## Limitations and next steps
- The model explains only part of the variation (R² 0.28). Missing pieces include spare-bus availability and live traffic.
- Events cover only baseball games; adding concerts and other sports would need a reliable historical source.
- The TTC's 2025 data uses incident codes instead of categories. Mapping them would extend the test to a third year.
- A natural extension is predicting incident *counts* per route and hour from weather forecasts, to help plan spare buses on snowy days.

## Tools
Python, pandas, scikit-learn, Plotly, Streamlit, pytest, GitHub Actions, Docker. Live demo: [late-again.streamlit.app](https://late-again.streamlit.app).
