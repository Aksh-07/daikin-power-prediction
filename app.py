"""Client proof-of-concept: one number for the next day's electricity."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import forecast

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"

st.set_page_config(page_title="Next day's AC electricity", layout="wide")
st.markdown(
    """
    <style>
    .block-container { max-width: 980px; padding-top: 1.5rem; }
    div[data-testid="stMetricValue"] { font-size: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def pretty_date(value) -> str:
    day = pd.Timestamp(value)
    return f"{day.strftime('%A')}, {day.day} {day.strftime('%B %Y')}"


def units(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    number = float(value)
    if abs(number) >= 10 or abs(number - round(number)) < 0.05:
        return f"{round(number):.0f}"
    return f"{number:.1f}"


def load() -> tuple[dict, pd.DataFrame]:
    summary = json.loads((ART / "summary.json").read_text(encoding="utf-8"))
    replay = pd.read_csv(ART / "replay.csv", parse_dates=["date"])
    return summary, replay


if not (ART / "summary.json").exists():
    st.error("The forecast has not been built yet. Run: python forecast.py")
    st.stop()

summary, replay = load()
replay = replay.sort_values("date")
calendar_metrics = summary["calendar"]
weather_metrics = summary["weather"]

st.title("Next day's air-conditioner electricity")
st.write(
    "Each morning this gives **one number**: how many units of electricity the "
    "outdoor unit is likely to use that day, and a **range**. "
    "About 8 times out of 10, the real use should land inside the range. "
    "A unit here is a kilowatt-hour, the same unit as on an electricity bill."
)
st.info(
    f"Proof of concept, using real readings from {pretty_date(summary['period_start'])} "
    f"to {pretty_date(summary['period_end'])}, counted midnight to midnight, India time. "
    "Pick a morning. You will see the forecast as it would have been given "
    "before that day unfolded, then what the unit actually used."
)

mode = st.radio(
    "What do we know that morning?",
    ["The calendar and recent days", "Also how hot that day will be"],
    horizontal=True,
    help="The second choice stands in for a weather forecast. It uses that day's real outdoor temperature, so a live weather forecast would be a little less exact.",
)
if abs(calendar_metrics["mae"] - weather_metrics["mae"]) < 0.5:
    st.caption(
        "On these months, knowing today's temperature changes a typical miss by less than one unit. "
        "Yesterday's weather already carries most of that signal."
    )
use_weather = mode.startswith("Also")
prefix = "weather" if use_weather else "calendar"
metrics = weather_metrics if use_weather else calendar_metrics

upcoming = replay[replay["kind"] == "upcoming"]
history = replay[replay["kind"] == "replay"].sort_values("date", ascending=False)
options = []
if len(upcoming):
    options.append(upcoming.iloc[0]["date"])
options.extend(history["date"].tolist())

def option_label(value) -> str:
    text = pretty_date(value)
    row = replay.loc[replay["date"] == value].iloc[0]
    if row["kind"] == "upcoming":
        return f"Next morning after the recordings — {text}"
    return text

default_index = 1 if len(upcoming) and len(options) > 1 else 0
chosen = st.selectbox(
    "Choose a morning",
    options=options,
    index=default_index,
    format_func=option_label,
)
row = replay.loc[replay["date"] == chosen].iloc[0]
pred = float(row[f"pred_{prefix}"])
low = float(row[f"low_{prefix}"])
high = float(row[f"high_{prefix}"])
actual = row["actual_kwh"]
reasons = [row[f"reason_1_{prefix}"], row[f"reason_2_{prefix}"], row[f"reason_3_{prefix}"]]
reasons = [text for text in reasons if isinstance(text, str) and text.strip()]

left, middle, right = st.columns(3)
left.metric("Forecast for the day", f"{units(pred)} units")
middle.metric("Likely range", f"{units(low)} – {units(high)}")
right.metric("Actually used", f"{units(actual)} units" if pd.notna(actual) else "Not in yet")

if pd.isna(actual):
    st.info(
        f"On the morning of {pretty_date(chosen)}, before any of that day's readings, "
        f"we would say about **{units(pred)} units**, and that the day is likely to fall "
        f"between **{units(low)} and {units(high)} units**."
    )
else:
    actual_f = float(actual)
    if abs(actual_f) >= 10 and abs(pred) >= 10:
        gap = round(actual_f) - round(pred)
    else:
        gap = actual_f - pred
    if gap > 0.5:
        direction = f"{units(gap)} units more than the forecast"
    elif gap < -0.5:
        direction = f"{units(abs(gap))} units less than the forecast"
    else:
        direction = "almost the same as the forecast"
    inside = low <= actual_f <= high
    where = "inside the range" if inside else "outside the range"
    message = (
        f"The unit used **{units(actual_f)} units**, which is {direction}. "
        f"That result is **{where}**."
    )
    if inside:
        st.success(message)
    else:
        st.warning(message)

last_week = row["baseline_last_week"]
if pd.notna(last_week):
    st.caption(
        f"A simple rule, repeating this weekday from last week, would have said {units(last_week)} units."
    )
elif pd.notna(row["baseline_typical_weekday"]):
    st.caption(
        f"Last week is missing for this weekday. The recent typical {pd.Timestamp(chosen).strftime('%A')} "
        f"was {units(row['baseline_typical_weekday'])} units."
    )

if use_weather and pd.isna(row["outdoor_today"]):
    st.caption(
        "This morning does not have a full outdoor temperature for the day, so the weather choice leans on recent days instead."
    )
elif use_weather and pd.notna(row["outdoor_today"]):
    st.caption(
        f"This choice is shown the day's outdoor temperature, about {float(row['outdoor_today']):.0f} degrees, "
        "as if a weather forecast had already arrived."
    )

st.subheader("Why this number")
if reasons:
    for text in reasons:
        st.write(f"- {text}")
else:
    st.write("This looks like an ordinary day for this point in the week.")

st.subheader("The mornings we checked")
st.write(
    f"The chart is {pretty_date(summary['test_start'])} to {pretty_date(summary['test_end'])}. "
    "Each morning's number uses only the healthy days before that morning, "
    "the same way it would run in real life. Gaps are days we refused to score: "
    "the meter was stuck, or the logger was off."
)

chart = replay[replay["kind"] == "replay"].sort_values("date")
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=chart["date"],
        y=chart[f"high_{prefix}"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=chart["date"],
        y=chart[f"low_{prefix}"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        name="Likely range",
        fillcolor="rgba(31, 119, 180, 0.18)",
        hovertemplate="Range %{y:.1f} units<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=chart["date"],
        y=chart[f"pred_{prefix}"],
        mode="lines+markers",
        name="Forecast that morning",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="Forecast %{y:.1f} units<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=chart["date"],
        y=chart["actual_kwh"],
        mode="lines+markers",
        name="Electricity actually used",
        line=dict(color="#d95f02", width=2),
        hovertemplate="Used %{y:.1f} units<extra></extra>",
    )
)
fig.add_vline(x=pd.Timestamp(chosen), line_dash="dot", line_color="#666666", line_width=1)
fig.update_layout(
    template="plotly_white",
    height=420,
    margin=dict(l=8, r=8, t=48, b=8),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    yaxis_title="Electricity for the day (kWh)",
    xaxis_title="Day",
    hovermode="x unified",
)
st.plotly_chart(fig, width="stretch")

st.subheader("How close it was")
miss_per_100 = metrics["wape"] * 100
bias = metrics["bias"]
if bias > 0.4:
    bias_text = f"On average the forecast was {units(bias)} units high."
elif bias < -0.4:
    bias_text = f"On average the forecast was {units(abs(bias))} units low."
else:
    bias_text = "On average it was not consistently high or low."

base_gap = metrics["baseline_mae"] - metrics["mae"]
if base_gap > 0.3:
    versus = (
        f"Copying the same weekday from last week missed by {units(metrics['baseline_mae'])} units "
        f"on a typical day. This forecast is closer, by about {units(base_gap)} units."
    )
elif base_gap < -0.3:
    versus = (
        f"Copying the same weekday from last week missed by {units(metrics['baseline_mae'])} units "
        f"on a typical day. That simple rule was closer than this forecast, by about {units(abs(base_gap))} units. "
        "We would keep the simple rule until the forecast beats it."
    )
else:
    versus = (
        f"Copying the same weekday from last week missed by about the same amount, "
        f"{units(metrics['baseline_mae'])} units on a typical day."
    )

st.write(
    f"On the {metrics['n']} days in the chart, a typical miss was **{units(metrics['mae'])} units**. "
    f"If the unit used 100 units over many days, the misses would add up to about "
    f"**{miss_per_100:.0f} units**. {bias_text}"
)
st.write(
    f"The range contained the real use on **{metrics['n_in_range']} of {metrics['n']} days**. "
    "The aim is about 8 out of 10."
)
st.write(versus)

st.subheader("What usually moves the number")
for factor in metrics["factors"]:
    st.write(f"- {factor['label']}")
st.caption("Listed from the strongest influence to a weaker one, on the days in the chart.")

st.subheader("Days the meter got stuck")
st.write(
    f"On **{summary['n_frozen']} days** the power meter repeated one high number all day, "
    "either about 1.2 or 2.5 kilowatts, for every reading. A live compressor does not do that. "
    f"Those readings add up to about {units(summary['frozen_recorded_kwh'])} units in the file, "
    "and we do not treat them as real."
)
st.write(
    "They are left out of the lessons the forecast learns, and left out of the score above. "
    "They are not left out of the service. The next morning we still give a number, "
    "and we use the last healthy day instead of the stuck number."
)
st.write(
    f"Quiet days stay in. On **{summary['n_quiet']} days** the unit truly used under "
    f"{units(3)} units, often because it was the weekend or the weather was mild. "
    "A typical weekday in the recordings is about "
    f"**{units(summary['weekday_median_kwh'])} units**, and a typical weekend day is about "
    f"**{units(summary['weekend_median_kwh'])} units**."
)
if summary["frozen_examples"]:
    examples = ", ".join(pretty_date(d) for d in summary["frozen_examples"])
    st.caption(f"Examples of stuck days: {examples}.")

st.subheader("What these readings are")
st.write(
    "This is one Daikin outdoor unit, sampled every 5 minutes, together with the indoor unit "
    "it serves. Most of the year it is cooling. In the cold weeks it heats. "
    "It behaves like an office: busy on weekdays, nearly asleep on Sundays, and it works "
    "harder when the outdoor air is hot."
)
st.write(
    "The sensors inside the machine — compressor temperature, pipe pressure, the expansion valve — "
    "tell us what the unit is doing right now. They are not used to guess tomorrow, "
    "because tomorrow's sensor readings do not exist yet. The forecast uses the day of the week, "
    "how much electricity the recent healthy days used, and how hot it has been."
)
st.caption(
    "Hour-by-hour shape of the day is a later step. This proof is the daily total only."
)

st.subheader("Add new readings")
st.write(
    "Upload a file with the **same columns** as the original readings. "
    "New times are added to the history. If a time is already there, the new file replaces it. "
    "The next morning's number is then rebuilt from every healthy day before that morning."
)
if summary.get("extra_rows"):
    st.caption(
        f"{summary['extra_rows']} readings have already been added on top of the original file."
    )
if st.session_state.get("upload_message"):
    st.success(st.session_state.pop("upload_message"))
upload = st.file_uploader("Readings file", type=["csv"])
if upload is not None:
    st.session_state["pending_upload"] = upload.getvalue()
    st.session_state["pending_name"] = upload.name
if st.session_state.get("pending_upload") and st.button("Update the forecast"):
    try:
        with st.spinner("Adding the readings and rebuilding tomorrow's number. This takes about half a minute."):
            fresh = forecast.read_new_file(BytesIO(st.session_state["pending_upload"]))
            stats = forecast.append_readings([fresh])
            forecast.main()
        st.session_state["pending_upload"] = None
        st.session_state["upload_message"] = (
            f"Added {stats['added']} new readings"
            + (f" and replaced {stats['replaced']} existing ones." if stats["replaced"] else ".")
            + " The morning list now includes the new day."
        )
        st.rerun()
    except ValueError as exc:
        st.error(str(exc))
st.caption(
    "To do this without opening the page, drop the same kind of file into the incoming folder "
    "next to the original readings, then run python forecast.py. "
    "A morning schedule can run that command and the new day appears on the next refresh."
)
