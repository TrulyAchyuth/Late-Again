"""
plots.py
--------
Interactive Plotly charts for the dashboard. All charts share one small theme
(colours and fonts) so the app looks consistent.
"""

import plotly.graph_objects as go
import plotly.io as pio

from config import DAY_ORDER

# ---- Design tokens (also used in assets/style.css) ----
ASPHALT = "#1F2A36"   # text, dark surfaces
SLATE = "#6B7785"     # secondary text, neutral bars
LINE = "#D9DEE3"      # gridlines and borders
SIGNAL = "#F5C400"    # accent: the thing to look at
LATE = "#C2253A"      # major delays / more incidents than normal
CALM = "#1D7874"      # fewer incidents than normal / on time
FONT = "Barlow, 'Segoe UI', sans-serif"

pio.templates["late_again"] = go.layout.Template(layout=dict(
    font=dict(family=FONT, color=ASPHALT, size=14),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=[ASPHALT, SIGNAL, LATE, CALM, SLATE],
    xaxis=dict(showgrid=False, linecolor=LINE, ticks="outside", tickcolor=LINE),
    yaxis=dict(gridcolor=LINE, zeroline=False),
    margin=dict(l=10, r=10, t=40, b=10),
    hoverlabel=dict(font_family=FONT, bgcolor="white"),
    title=dict(font=dict(size=17, family=FONT), x=0, xanchor="left"),
))
pio.templates.default = "late_again"


def monthly_trend(inc):
    m = inc.set_index("timestamp").resample("MS").agg(incidents=("delay_min", "size"),
                                                     major=("major_delay", "sum"))
    fig = go.Figure()
    fig.add_bar(x=m.index, y=m["incidents"], name="All incidents", marker_color=LINE,
                hovertemplate="%{x|%b %Y}: %{y:,} incidents<extra></extra>")
    fig.add_bar(x=m.index, y=m["major"], name="Major (15+ min)", marker_color=LATE,
                hovertemplate="%{x|%b %Y}: %{y:,} major<extra></extra>")
    fig.update_layout(barmode="overlay", title="Delay incidents per month", height=340,
                      legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
    return fig


def delay_histogram(inc):
    fig = go.Figure(go.Histogram(x=inc["delay_min"], xbins=dict(start=0, end=121, size=2),
                                 marker_color=ASPHALT, hovertemplate="%{x} min: %{y:,}<extra></extra>"))
    fig.add_vline(x=15, line_dash="dot", line_color=LATE,
                  annotation_text="major delay (15 min)", annotation_position="top right")
    fig.update_layout(title="How late was the next bus?", xaxis_title="Delay (minutes)",
                      yaxis_title="Incidents", height=340, bargap=0.05)
    return fig


def heatmap(table, title, colorscale, zlabel):
    table = table.reindex([d for d in DAY_ORDER if d in table.index])
    fig = go.Figure(go.Heatmap(z=table.values, x=table.columns, y=table.index, colorscale=colorscale,
                               colorbar=dict(title=zlabel, thickness=10),
                               hovertemplate="%{y} %{x}:00<br>" + zlabel + ": %{z:.1f}<extra></extra>"))
    fig.update_layout(title=title, xaxis_title="Hour of day", height=330,
                      xaxis=dict(dtick=2), yaxis=dict(autorange="reversed"))
    return fig


def route_bars(routes, top=15):
    r = routes.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(
        y="Route " + r["route"].astype(str), x=r["incidents"], orientation="h",
        marker=dict(color=r["major_share"], colorscale=[[0, LINE], [1, LATE]], cmin=0.2, cmax=0.6,
                    colorbar=dict(title="% major", tickformat=".0%", thickness=10)),
        customdata=r[["median_delay", "major_share"]],
        hovertemplate="%{y}: %{x:,} incidents<br>median %{customdata[0]:.0f} min, "
                      "%{customdata[1]:.0%} major<extra></extra>"))
    fig.update_layout(title=f"The {top} routes with the most delay incidents", height=470,
                      xaxis_title="Incidents (2023-2024)")
    return fig


def incident_bars(table):
    t = table.iloc[::-1]
    fig = go.Figure()
    fig.add_bar(y=t["incident"], x=t["median_delay"], orientation="h", marker_color=ASPHALT,
                customdata=t[["incidents"]],
                hovertemplate="%{y}: median %{x:.0f} min<br>%{customdata[0]:,} incidents<extra></extra>")
    fig.update_layout(title="Median delay by incident type", xaxis_title="Median delay (minutes)", height=470)
    return fig


def ratio_chart(table, col, title):
    """Actual vs expected incidents, as a ratio with a 90% interval. 1.0 = a normal hour."""
    # Colour only clear, meaningful differences: the 90% range excludes 1 AND the effect is 5%+
    colors = [LATE if lo > 1 and r >= 1.05 else CALM if hi < 1 and r <= 0.95 else SLATE
              for r, lo, hi in zip(table["ratio"], table["ratio_low"], table["ratio_high"])]
    labels = table[col].astype(str)
    fig = go.Figure(go.Bar(
        x=labels, y=table["ratio"], marker_color=colors,
        error_y=dict(type="data", symmetric=False, array=table["ratio_high"] - table["ratio"],
                     arrayminus=table["ratio"] - table["ratio_low"], color=SLATE, thickness=1.2),
        customdata=table[["hours", "incidents_per_hour", "expected_per_hour"]],
        hovertemplate="%{x}: %{y:.2f}x normal<br>%{customdata[0]:,} hours<br>"
                      "%{customdata[1]:.1f} incidents/h vs %{customdata[2]:.1f} expected<extra></extra>"))
    fig.add_hline(y=1, line_color=ASPHALT, line_width=1)
    fig.update_layout(title=title, yaxis_title="Incidents vs a normal hour (x)", height=360)
    return fig


def actual_vs_predicted(preds, n=4000):
    s = preds.sample(min(n, len(preds)), random_state=0)
    fig = go.Figure(go.Scattergl(x=s["delay_min"], y=s["predicted_delay_min"], mode="markers",
                                 marker=dict(color=ASPHALT, opacity=0.18, size=5),
                                 hovertemplate="actual %{x} min, predicted %{y:.1f}<extra></extra>"))
    fig.add_shape(type="line", x0=0, y0=0, x1=120, y1=120, line=dict(color=LATE, dash="dot"))
    fig.update_layout(title="Actual vs predicted delay (2024 test set)", height=380,
                      xaxis_title="Actual delay (min)", yaxis_title="Predicted (min)",
                      yaxis_range=[0, 70])
    return fig


def importance_bars(imp):
    t = imp[imp["importance"] > 0.001].iloc[::-1]
    fig = go.Figure(go.Bar(y=t["feature"], x=t["importance"], orientation="h", marker_color=ASPHALT,
                           hovertemplate="%{y}: +%{x:.2f} min MAE when shuffled<extra></extra>"))
    fig.update_layout(title="What the delay model relies on", height=360,
                      xaxis_title="Extra error (min) when this input is scrambled")
    return fig


def waterfall(expl):
    measures = ["absolute"] + ["relative"] * (len(expl) - 2) + ["total"]
    fig = go.Figure(go.Waterfall(
        x=expl["factor"], y=expl["minutes"], measure=measures,
        text=[f"{v:+.1f}" if m == "relative" else f"{v:.1f}" for v, m in zip(expl["minutes"], measures)],
        textposition="outside",
        increasing=dict(marker=dict(color=LATE)), decreasing=dict(marker=dict(color=CALM)),
        totals=dict(marker=dict(color=ASPHALT)), connector=dict(line=dict(color=LINE))))
    fig.update_layout(title="Why this many minutes?", yaxis_title="Delay (min)", height=380,
                      showlegend=False)
    return fig


def what_if_line(curve, key, label, current):
    fig = go.Figure()
    fig.add_scatter(x=curve[key], y=curve["expected_delay"], mode="lines+markers",
                    line=dict(color=ASPHALT, width=2.5), name="Expected delay",
                    hovertemplate="%{x}: %{y:.1f} min<extra></extra>")
    fig.add_vline(x=current, line_color=SIGNAL, line_width=3)
    fig.update_layout(title=f"Expected delay as {label.lower()} changes", xaxis_title=label,
                      yaxis_title="Minutes", height=340, showlegend=False)
    return fig


def compare_bars(curve, key, current, title):
    c = curve.sort_values("expected_delay")
    colors = [SIGNAL if str(v) == str(current) else LINE for v in c[key]]
    fig = go.Figure(go.Bar(y=c[key].astype(str), x=c["expected_delay"], orientation="h",
                           marker_color=colors, marker_line=dict(color=ASPHALT, width=0.6),
                           customdata=c[["major_chance"]],
                           hovertemplate="%{y}: %{x:.1f} min, %{customdata[0]:.0%} major<extra></extra>"))
    fig.update_layout(title=title, xaxis_title="Expected delay (min)", height=max(320, 26 * len(c)))
    return fig
