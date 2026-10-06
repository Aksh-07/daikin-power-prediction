"""Next-day electricity forecast for one Daikin outdoor unit.

The product is one number per local day (kWh), plus an 80% range.
Stuck-meter days are not used as answers. Real off days are kept.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data_D.csv"
EXTRA_PATH = ROOT / "data_extra.csv"
INCOMING_DIR = ROOT / "incoming"
ART_DIR = ROOT / "artifacts"
ZONE = "Asia/Kolkata"
SLOTS_PER_DAY = 288
STEP_HOURS = 5 / 60
MIN_COVERAGE = 0.90
# Meter stuck on a high constant all day. Standby near 0.1 kW is a real off day.
FROZEN_MIN_SAMPLES = 200
FROZEN_MAX_LEVELS = 2
FROZEN_MIN_MEDIAN_KW = 0.2
FROZEN_MAX_STD_KW = 0.05
QUIET_KWH = 3.0
# Score April–June 2023. Each morning is trained only on healthy days before it.
TEST_START = pd.Timestamp("2023-04-01")
BURN_DAYS = 35
LOOKBACK_DAYS = 28
MIN_HISTORY_DAYS = 14
WEEKDAY_LOOKBACK = 4

FEATURES = [
    "dow",
    "is_weekend",
    "month",
    "energy_yesterday",
    "energy_two_days_ago",
    "energy_last_week",
    "energy_typical_this_weekday",
    "energy_past_week",
    "energy_last_healthy",
    "days_since_healthy",
    "outdoor_yesterday",
    "outdoor_past_week",
    "hours_running_yesterday",
    "heating_yesterday",
]
FEATURES_WEATHER = FEATURES + ["outdoor_today"]

DAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
MONTH_NAMES = [
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def expected_columns() -> list[str]:
    return list(pd.read_csv(DATA_PATH, nrows=0).columns)


def read_new_file(source) -> pd.DataFrame:
    """Read an uploaded or dropped file. It must use the same columns as data_D.csv."""
    try:
        fresh = pd.read_csv(source)
    except Exception as exc:
        raise ValueError("This file could not be read as a CSV.") from exc
    expected = expected_columns()
    missing = [column for column in expected if column not in fresh.columns]
    if missing:
        raise ValueError(
            "This file is missing columns the original readings have: "
            + ", ".join(missing)
            + "."
        )
    if fresh.empty:
        raise ValueError("This file has no rows.")
    fresh = fresh[expected].copy()
    fresh["timestamp"] = pd.to_datetime(fresh["timestamp"], utc=True, errors="coerce")
    if fresh["timestamp"].isna().any():
        raise ValueError("Some timestamps could not be read. Use the same time format as the original file.")
    return fresh


def append_readings(frames: list[pd.DataFrame]) -> dict:
    """Store new rows beside the original file. A repeated timestamp keeps the newer row."""
    expected = expected_columns()
    if EXTRA_PATH.exists() and EXTRA_PATH.stat().st_size > 0:
        extra = pd.read_csv(EXTRA_PATH)
        extra["timestamp"] = pd.to_datetime(extra["timestamp"], utc=True)
    else:
        extra = pd.DataFrame(columns=expected)
    original_ts = set(pd.to_datetime(pd.read_csv(DATA_PATH, usecols=["timestamp"])["timestamp"], utc=True))
    already = original_ts | set(extra["timestamp"]) if len(extra) else original_ts
    incoming = pd.concat(frames, ignore_index=True)
    incoming["timestamp"] = pd.to_datetime(incoming["timestamp"], utc=True)
    new_ts = set(incoming["timestamp"])
    added = len(new_ts - already)
    replaced = len(new_ts & already)
    combined = pd.concat([extra, incoming], ignore_index=True)
    combined = combined.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    combined.to_csv(EXTRA_PATH, index=False)
    return {"added": added, "replaced": replaced, "stored_rows": int(len(combined))}


def absorb_incoming() -> dict | None:
    """Pick up CSV files dropped in the incoming folder, then move them aside."""
    if not INCOMING_DIR.exists():
        return None
    files = sorted(path for path in INCOMING_DIR.glob("*.csv") if path.is_file())
    if not files:
        return None
    frames = [read_new_file(path) for path in files]
    result = append_readings(frames)
    processed = INCOMING_DIR / "processed"
    processed.mkdir(exist_ok=True)
    for path in files:
        dest = processed / path.name
        if dest.exists():
            dest = processed / f"{path.stem}_{path.stat().st_mtime_ns}{path.suffix}"
        path.replace(dest)
    result["files"] = len(files)
    return result


def load_samples() -> pd.DataFrame:
    frames = [pd.read_csv(DATA_PATH)]
    if EXTRA_PATH.exists() and EXTRA_PATH.stat().st_size > 0:
        frames.append(pd.read_csv(EXTRA_PATH))
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    df["ts"] = df["timestamp"].dt.tz_convert(ZONE)
    df["date"] = df["ts"].dt.floor("D").dt.tz_localize(None)
    return df


def _day_record(date: pd.Timestamp, g: pd.DataFrame) -> dict:
    power = g["power"].dropna()
    n = int(power.shape[0])
    std = float(power.std(ddof=0)) if n else np.nan
    median = float(power.median()) if n else np.nan
    nunique = int(power.nunique()) if n else 0
    frozen = bool(
        n >= FROZEN_MIN_SAMPLES
        and nunique <= FROZEN_MAX_LEVELS
        and median > FROZEN_MIN_MEDIAN_KW
        and std < FROZEN_MAX_STD_KW
    )
    coverage = n / SLOTS_PER_DAY
    usable = bool(coverage >= MIN_COVERAGE and not frozen)
    outdoor = g["odu_outdoortemp"].dropna()
    mode = g["operatingmode"].dropna()
    return {
        "date": date,
        "kwh": float((power * STEP_HOURS).sum()) if n else np.nan,
        "n_power": n,
        "coverage": coverage,
        "power_median": median,
        "power_std": std,
        "power_nunique": nunique,
        "frozen": frozen,
        "usable": usable,
        "outdoor_mean": float(outdoor.mean()) if len(outdoor) else np.nan,
        "outdoor_max": float(outdoor.max()) if len(outdoor) else np.nan,
        "outdoor_n": int(len(outdoor)),
        "outdoor_nunique": int(outdoor.nunique()) if len(outdoor) else 0,
        "runtime_hours": float((power > 0.2).sum() * STEP_HOURS) if n else np.nan,
        "heat_share": float((mode == 1).mean()) if len(mode) else np.nan,
    }


def build_daily(samples: pd.DataFrame) -> pd.DataFrame:
    records = [
        _day_record(date, group)
        for date, group in samples.groupby("date", sort=True)
    ]
    daily = pd.DataFrame.from_records(records).set_index("date").sort_index()
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full)
    daily.index.name = "date"
    daily["frozen"] = daily["frozen"].eq(True)
    daily["usable"] = daily["usable"].eq(True)
    for col in ("n_power", "outdoor_n", "outdoor_nunique", "power_nunique"):
        daily[col] = daily[col].fillna(0).astype(int)
    daily["coverage"] = daily["coverage"].fillna(0.0)
    return daily


def _outdoor_ok(row: pd.Series) -> bool:
    return bool(row["outdoor_n"] >= 72 and row["outdoor_nunique"] >= 3)


def _past_usable(daily: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    return daily.loc[(daily.index < day) & daily["usable"]]


def features_for_day(daily: pd.DataFrame, day: pd.Timestamp) -> dict:
    usable = _past_usable(daily, day)
    yesterday = day - pd.Timedelta(days=1)
    two_ago = day - pd.Timedelta(days=2)
    last_week = day - pd.Timedelta(days=7)

    def energy_if_usable(when: pd.Timestamp) -> float:
        if when in daily.index and bool(daily.at[when, "usable"]):
            return float(daily.at[when, "kwh"])
        return np.nan

    same_weekday = usable[usable.index.dayofweek == day.dayofweek].tail(WEEKDAY_LOOKBACK)
    past_week = usable[usable.index >= day - pd.Timedelta(days=7)]
    outdoor_days = daily.loc[
        (daily.index < day) & (daily.index >= day - pd.Timedelta(days=7))
    ]
    outdoor_days = outdoor_days[outdoor_days.apply(_outdoor_ok, axis=1)]

    if len(usable):
        last = usable.iloc[-1]
        last_day = usable.index[-1]
        energy_last_healthy = float(last["kwh"])
        days_since_healthy = int((day - last_day).days)
    else:
        energy_last_healthy = np.nan
        days_since_healthy = np.nan

    if yesterday in daily.index and bool(daily.at[yesterday, "usable"]):
        hours_running = float(daily.at[yesterday, "runtime_hours"])
        heat_share = daily.at[yesterday, "heat_share"]
        heating = float(heat_share >= 0.5) if pd.notna(heat_share) else np.nan
    else:
        hours_running = np.nan
        heating = np.nan

    outdoor_yesterday = np.nan
    if yesterday in daily.index and _outdoor_ok(daily.loc[yesterday]):
        outdoor_yesterday = float(daily.at[yesterday, "outdoor_mean"])

    outdoor_today = np.nan
    if (
        day in daily.index
        and daily.at[day, "outdoor_n"] >= 200
        and daily.at[day, "outdoor_nunique"] >= 3
    ):
        outdoor_today = float(daily.at[day, "outdoor_mean"])

    return {
        "date": day,
        "dow": int(day.dayofweek),
        "is_weekend": int(day.dayofweek >= 5),
        "month": int(day.month),
        "energy_yesterday": energy_if_usable(yesterday),
        "energy_two_days_ago": energy_if_usable(two_ago),
        "energy_last_week": energy_if_usable(last_week),
        "energy_typical_this_weekday": (
            float(same_weekday["kwh"].median()) if len(same_weekday) else np.nan
        ),
        "energy_past_week": float(past_week["kwh"].mean()) if len(past_week) else np.nan,
        "energy_last_healthy": energy_last_healthy,
        "days_since_healthy": days_since_healthy,
        "outdoor_yesterday": outdoor_yesterday,
        "outdoor_past_week": (
            float(outdoor_days["outdoor_mean"].mean()) if len(outdoor_days) else np.nan
        ),
        "hours_running_yesterday": hours_running,
        "heating_yesterday": heating,
        "outdoor_today": outdoor_today,
        "actual_kwh": (
            float(daily.at[day, "kwh"]) if day in daily.index and bool(daily.at[day, "usable"]) else np.nan
        ),
        "baseline_last_week": energy_if_usable(last_week),
        "baseline_typical_weekday": (
            float(same_weekday["kwh"].median()) if len(same_weekday) else np.nan
        ),
    }


def build_frame(daily: pd.DataFrame) -> pd.DataFrame:
    start = daily.index.min() + pd.Timedelta(days=MIN_HISTORY_DAYS)
    days = daily.index[daily.index >= start]
    frame = pd.DataFrame([features_for_day(daily, day) for day in days])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date").sort_index()


def make_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.06,
        min_child_weight=8,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
    )


def _group_key(is_weekend: pd.Series) -> pd.Series:
    return np.where(is_weekend.astype(bool), "weekend", "weekday")


def _offsets(residuals: np.ndarray) -> tuple[float, float] | None:
    if len(residuals) < 8:
        return None
    ordered = np.sort(residuals)
    n = len(ordered)
    lo_i = max(0, int(np.floor(0.10 * (n + 1))) - 1)
    hi_i = min(n - 1, int(np.ceil(0.90 * (n + 1))) - 1)
    return float(ordered[lo_i]), float(ordered[hi_i])


def _interval_model(cal: pd.DataFrame, pred_cal: np.ndarray) -> dict[str, tuple[float, float]]:
    """80% range around a bias-adjusted forecast, split by weekday and weekend."""
    resid = cal["actual_kwh"].to_numpy() - pred_cal
    groups = _group_key(cal["is_weekend"])
    pooled = _offsets(resid - np.median(resid))
    pooled_shift = float(np.median(resid))
    out: dict[str, tuple[float, float]] = {}
    for name in ("weekday", "weekend"):
        mask = groups == name
        if mask.sum() < 8 or pooled is None:
            out[name] = (pooled_shift, pooled if pooled is not None else (-5.0, 5.0))
            continue
        part = resid[mask]
        shift = float(np.median(part))
        bounds = _offsets(part - shift) or pooled
        out[name] = (shift, bounds)
    out["pooled_shift"] = pooled_shift  # type: ignore[assignment]
    return out


def apply_interval(pred: np.ndarray, is_weekend: pd.Series, interval: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point = pred.copy()
    low = np.zeros_like(pred)
    high = np.zeros_like(pred)
    groups = _group_key(is_weekend)
    for i, group in enumerate(groups):
        shift, bounds = interval[group]
        if isinstance(bounds, tuple):
            lo_off, hi_off = bounds
        else:
            lo_off, hi_off = -5.0, 5.0
        guess = max(0.0, float(pred[i]) + shift)
        lo = min(guess, guess + lo_off)
        hi = max(guess, guess + hi_off)
        lo = max(0.0, lo)
        hi = max(hi, lo)
        point[i] = guess
        low[i] = lo
        high[i] = hi
    return point, low, high


def _contributions(model: XGBRegressor, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    matrix = xgb.DMatrix(frame[features], feature_names=features)
    return model.get_booster().predict(matrix, pred_contribs=True)


def _sentence(name: str, value: float, effect: float, dow: int) -> str | None:
    if not np.isfinite(effect) or abs(effect) < 0.35:
        return None
    higher = effect > 0
    missing = not np.isfinite(value)

    if name == "is_weekend":
        if value >= 0.5 and not higher:
            return "It is the weekend. The office is usually empty, so the forecast stays low."
        if value >= 0.5 and higher:
            return "It is the weekend, but recent weekends used more than usual, so the forecast is higher."
        if not higher:
            return "Even though it is a weekday, other signs point to a quieter day."
        return "It is a weekday. The office usually runs the cooling, so the forecast is higher."

    if name == "dow":
        day_name = DAY_NAMES[int(value)] if np.isfinite(value) else "this weekday"
        if higher:
            return f"{day_name}s tend to use more electricity, so the forecast moves up."
        return f"{day_name}s tend to use less electricity, so the forecast moves down."

    if name == "month":
        month = MONTH_NAMES[int(value)] if np.isfinite(value) else "This month"
        if higher:
            return f"{month} is a heavier month for this unit, so the forecast moves up."
        return f"{month} is a lighter month for this unit, so the forecast moves down."

    if name == "energy_yesterday":
        if missing:
            return "Yesterday has no trustworthy electricity reading, so that clue is left out."
        bit = "pushes the guess up" if higher else "pushes the guess down"
        return f"Yesterday the unit used about {_fmt(value)} units, which {bit}."

    if name == "energy_two_days_ago":
        if missing:
            return None
        bit = "pushes the guess up" if higher else "pushes the guess down"
        return f"Two days ago the unit used about {_fmt(value)} units, which {bit}."

    if name == "energy_last_week":
        if missing:
            return "This same weekday last week has no trustworthy reading, so that clue is left out."
        bit = "pushes the guess up" if higher else "pushes the guess down"
        return f"This same weekday last week used about {_fmt(value)} units, which {bit}."

    if name == "energy_typical_this_weekday":
        if missing:
            return None
        day_name = DAY_NAMES[dow]
        bit = "pushes the guess up" if higher else "pushes the guess down"
        return f"Recent {day_name}s used about {_fmt(value)} units, which {bit}."

    if name == "energy_past_week":
        if missing:
            return None
        bit = "pushes the guess up" if higher else "pushes the guess down"
        return f"The past week averaged about {_fmt(value)} units a day, which {bit}."

    if name == "energy_last_healthy":
        if missing:
            return None
        bit = "pushes the guess up" if higher else "pushes the guess down"
        return f"The last trustworthy day used about {_fmt(value)} units, which {bit}."

    if name == "days_since_healthy":
        if missing:
            return None
        return (
            f"The last trustworthy reading is {int(value)} days old, so the forecast is less sure of recent habits."
        )

    if name == "outdoor_yesterday":
        if missing:
            return None
        bit = "pushes the guess up" if higher else "pushes the guess down"
        return f"Yesterday it was about {value:.0f} degrees outside, which {bit}."

    if name == "outdoor_past_week":
        if missing:
            return None
        bit = "pushes the guess up" if higher else "pushes the guess down"
        return f"The past week averaged about {value:.0f} degrees outside, which {bit}."

    if name == "outdoor_today":
        if missing:
            return None
        bit = "pushes the guess up" if higher else "pushes the guess down"
        return f"Today is about {value:.0f} degrees outside, which {bit}."

    if name == "hours_running_yesterday":
        if missing:
            return None
        bit = "pushes the guess up" if higher else "pushes the guess down"
        return f"Yesterday the unit ran for about {value:.0f} hours, which {bit}."

    if name == "heating_yesterday":
        if missing:
            return None
        if value >= 0.5 and higher:
            return "Yesterday the unit was heating, and that raises the forecast."
        if value >= 0.5:
            return "Yesterday the unit was heating, and that lowers the forecast."
        if higher:
            return "Yesterday the unit was cooling, and that raises the forecast."
        return "Yesterday the unit was cooling, and that lowers the forecast."

    return None


def _fmt(value: float) -> str:
    if abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.1f}"


def reasons_for_row(names: list[str], values: np.ndarray, contrib: np.ndarray) -> list[str]:
    # Last column of pred_contribs is the bias term.
    order = np.argsort(-np.abs(contrib[:-1]))
    dow = int(values[names.index("dow")]) if "dow" in names else 0
    sentences: list[str] = []
    for idx in order:
        sentence = _sentence(names[idx], float(values[idx]), float(contrib[idx]), dow)
        if sentence and sentence not in sentences:
            sentences.append(sentence)
        if len(sentences) == 3:
            break
    if not sentences:
        sentences.append("This looks like an ordinary day for this point in the week.")
    while len(sentences) < 3:
        sentences.append("")
    return sentences


def score(actual: np.ndarray, pred: np.ndarray, low: np.ndarray, high: np.ndarray) -> dict:
    err = np.abs(actual - pred)
    return {
        "n": int(len(actual)),
        "mae": float(err.mean()),
        "wape": float(err.sum() / actual.sum()) if actual.sum() else None,
        "bias": float((pred - actual).mean()),
        "n_in_range": int(((actual >= low) & (actual <= high)).sum()),
        "coverage": float(((actual >= low) & (actual <= high)).mean()),
    }


def baseline_score(frame: pd.DataFrame) -> dict:
    pred = frame["baseline_last_week"].copy()
    missing = pred.isna()
    pred.loc[missing] = frame.loc[missing, "baseline_typical_weekday"]
    still = pred.isna()
    pred.loc[still] = frame.loc[still, "energy_last_healthy"]
    pred = np.maximum(0.0, pred.fillna(0.0).to_numpy())
    actual = frame["actual_kwh"].to_numpy()
    err = np.abs(actual - pred)
    return {
        "mae": float(err.mean()),
        "wape": float(err.sum() / actual.sum()) if actual.sum() else None,
        "bias": float((pred - actual).mean()),
    }


def _importance(contrib: np.ndarray, features: list[str]) -> list[dict]:
    mean_abs = np.abs(contrib[:, :-1]).mean(axis=0)
    total = float(mean_abs.sum()) or 1.0
    order = np.argsort(-mean_abs)
    labels = {
        "dow": "Which day of the week it is",
        "is_weekend": "Weekday or weekend",
        "month": "Time of year",
        "energy_yesterday": "Yesterday's electricity",
        "energy_two_days_ago": "Electricity two days ago",
        "energy_last_week": "This weekday last week",
        "energy_typical_this_weekday": "What this weekday usually uses",
        "energy_past_week": "The past week's average",
        "energy_last_healthy": "The last trustworthy day",
        "days_since_healthy": "Days since a trustworthy reading",
        "outdoor_yesterday": "Yesterday's outdoor temperature",
        "outdoor_past_week": "How hot the recent days were",
        "hours_running_yesterday": "How long the unit ran yesterday",
        "heating_yesterday": "Heating or cooling yesterday",
        "outdoor_today": "How hot this day is",
    }
    rows = []
    for idx in order[:5]:
        rows.append(
            {
                "key": features[idx],
                "label": labels.get(features[idx], features[idx]),
                "share": float(mean_abs[idx] / total),
            }
        )
    return rows


def _range_from_residuals(pred: float, residuals: list[float]) -> tuple[float, float]:
    if len(residuals) < 8:
        lo_off, hi_off = -8.0, 8.0
    else:
        lo_off = float(np.quantile(residuals, 0.10))
        hi_off = float(np.quantile(residuals, 0.90))
    low = max(0.0, min(pred, pred + lo_off))
    high = max(pred, pred + hi_off)
    return low, high


def run_model(frame: pd.DataFrame, features: list[str]) -> dict:
    """Forecast each morning from healthy days before that morning only."""
    labeled = frame[frame["actual_kwh"].notna()].sort_index()
    if labeled.index.max() < TEST_START or (labeled.index < TEST_START).sum() < 60:
        raise RuntimeError("Not enough history before the checked months.")

    burn_start = TEST_START - pd.Timedelta(days=BURN_DAYS)
    upcoming_day = labeled.index.max() + pd.Timedelta(days=1)
    predict_days = list(labeled.index[labeled.index >= burn_start])
    if upcoming_day in frame.index:
        predict_days.append(upcoming_day)

    history: list[dict] = []
    latest_model = None
    for day in predict_days:
        train = labeled[labeled.index < day]
        model = make_model()
        model.fit(train[features], train["actual_kwh"])
        latest_model = model
        row = frame.loc[[day]]
        pred = float(max(0.0, model.predict(row[features])[0]))
        contrib = _contributions(model, row, features)[0]
        reasons = reasons_for_row(features, row[features].to_numpy()[0], contrib)
        actual = float(row["actual_kwh"].iloc[0]) if pd.notna(row["actual_kwh"].iloc[0]) else np.nan
        weekend = int(row["is_weekend"].iloc[0])
        window_start = day - pd.Timedelta(days=LOOKBACK_DAYS)
        past = [
            item
            for item in history
            if window_start <= item["date"] < day and np.isfinite(item["actual"])
        ]
        same = [item for item in past if item["is_weekend"] == weekend]
        chosen = same if len(same) >= 8 else past
        low, high = _range_from_residuals(pred, [item["actual"] - item["pred"] for item in chosen])
        history.append(
            {
                "date": day,
                "pred": pred,
                "low": low,
                "high": high,
                "actual": actual,
                "reasons": reasons,
                "is_weekend": weekend,
                "contrib": contrib,
                "baseline_last_week": row["baseline_last_week"].iloc[0],
                "baseline_typical_weekday": row["baseline_typical_weekday"].iloc[0],
                "outdoor_today": row["outdoor_today"].iloc[0],
            }
        )

    scored = [item for item in history if item["date"] >= TEST_START and np.isfinite(item["actual"])]
    if len(scored) < 10:
        raise RuntimeError(f"Only {len(scored)} days to check.")

    replay = labeled.loc[[item["date"] for item in scored]].copy()
    replay["pred_kwh"] = [item["pred"] for item in scored]
    replay["low_kwh"] = [item["low"] for item in scored]
    replay["high_kwh"] = [item["high"] for item in scored]
    replay["reason_1"] = [item["reasons"][0] for item in scored]
    replay["reason_2"] = [item["reasons"][1] for item in scored]
    replay["reason_3"] = [item["reasons"][2] for item in scored]
    replay["in_range"] = (replay["actual_kwh"] >= replay["low_kwh"]) & (
        replay["actual_kwh"] <= replay["high_kwh"]
    )

    upcoming_item = next(item for item in history if item["date"] == upcoming_day)
    upcoming_pack = {
        "date": upcoming_day,
        "pred": upcoming_item["pred"],
        "low": upcoming_item["low"],
        "high": upcoming_item["high"],
        "reasons": upcoming_item["reasons"],
        "is_weekend": upcoming_item["is_weekend"],
        "baseline_last_week": (
            None
            if pd.isna(upcoming_item["baseline_last_week"])
            else float(upcoming_item["baseline_last_week"])
        ),
        "baseline_typical_weekday": (
            None
            if pd.isna(upcoming_item["baseline_typical_weekday"])
            else float(upcoming_item["baseline_typical_weekday"])
        ),
        "outdoor_today": (
            None if pd.isna(upcoming_item["outdoor_today"]) else float(upcoming_item["outdoor_today"])
        ),
    }

    metrics = score(
        replay["actual_kwh"].to_numpy(),
        replay["pred_kwh"].to_numpy(),
        replay["low_kwh"].to_numpy(),
        replay["high_kwh"].to_numpy(),
    )
    metrics["baseline"] = baseline_score(replay)
    metrics["train_mae"] = None
    metrics["cal_n"] = int(sum(1 for item in history if item["date"] < TEST_START))
    metrics["train_n"] = int((labeled.index < TEST_START).sum())
    test_contrib = np.vstack(
        [item["contrib"] for item in history if item["date"] >= TEST_START and np.isfinite(item["actual"])]
    )
    metrics["factors"] = _importance(test_contrib, features)

    return {
        "model": latest_model,
        "metrics": metrics,
        "replay": replay,
        "upcoming": upcoming_pack,
        "train_end": str((TEST_START - pd.Timedelta(days=1)).date()),
        "cal_start": str(burn_start.date()),
        "cal_end": str((TEST_START - pd.Timedelta(days=1)).date()),
    }


def _round(value, digits=3):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, (float, np.floating)):
        return round(float(value), digits)
    return value


def save_outputs(daily: pd.DataFrame, calendar: dict, weather: dict) -> None:
    ART_DIR.mkdir(exist_ok=True)
    daily.reset_index().to_csv(ART_DIR / "daily.csv", index=False)

    def pack(result: dict, prefix: str) -> pd.DataFrame:
        replay = result["replay"][
            [
                "actual_kwh",
                "pred_kwh",
                "low_kwh",
                "high_kwh",
                "reason_1",
                "reason_2",
                "reason_3",
                "in_range",
                "is_weekend",
                "baseline_last_week",
                "baseline_typical_weekday",
                "outdoor_today",
            ]
        ].copy()
        replay["kind"] = "replay"
        upcoming = result["upcoming"]
        if upcoming is not None:
            extra = pd.DataFrame(
                [
                    {
                        "date": upcoming["date"],
                        "actual_kwh": np.nan,
                        "pred_kwh": upcoming["pred"],
                        "low_kwh": upcoming["low"],
                        "high_kwh": upcoming["high"],
                        "reason_1": upcoming["reasons"][0],
                        "reason_2": upcoming["reasons"][1],
                        "reason_3": upcoming["reasons"][2],
                        "in_range": np.nan,
                        "is_weekend": upcoming["is_weekend"],
                        "baseline_last_week": upcoming["baseline_last_week"],
                        "baseline_typical_weekday": upcoming["baseline_typical_weekday"],
                        "outdoor_today": upcoming["outdoor_today"],
                        "kind": "upcoming",
                    }
                ]
            ).set_index("date")
            replay = pd.concat([replay, extra])
        replay = replay.rename(
            columns={
                "pred_kwh": f"pred_{prefix}",
                "low_kwh": f"low_{prefix}",
                "high_kwh": f"high_{prefix}",
                "reason_1": f"reason_1_{prefix}",
                "reason_2": f"reason_2_{prefix}",
                "reason_3": f"reason_3_{prefix}",
                "in_range": f"in_range_{prefix}",
            }
        )
        return replay

    merged = pack(calendar, "calendar").join(
        pack(weather, "weather")[
            [
                "pred_weather",
                "low_weather",
                "high_weather",
                "reason_1_weather",
                "reason_2_weather",
                "reason_3_weather",
                "in_range_weather",
            ]
        ],
        how="outer",
    )
    merged.reset_index().to_csv(ART_DIR / "replay.csv", index=False)
    calendar["model"].save_model(ART_DIR / "model_calendar.json")
    weather["model"].save_model(ART_DIR / "model_weather.json")

    frozen = daily[daily["frozen"]]
    usable = daily[daily["usable"]]
    quiet = usable[usable["kwh"] < QUIET_KWH]
    summary = {
        "zone": ZONE,
        "period_start": str(daily.index.min().date()),
        "period_end": str(daily.index.max().date()),
        "n_days": int(len(daily)),
        "n_usable": int(usable.shape[0]),
        "n_frozen": int(frozen.shape[0]),
        "n_quiet": int(quiet.shape[0]),
        "frozen_recorded_kwh": _round(float(frozen["kwh"].sum()) if len(frozen) else 0.0, 1),
        "frozen_examples": [str(d.date()) for d in frozen.index[:6]],
        "median_usable_kwh": _round(float(usable["kwh"].median()), 1),
        "weekday_median_kwh": _round(float(usable.loc[usable.index.dayofweek < 5, "kwh"].median()), 1),
        "weekend_median_kwh": _round(float(usable.loc[usable.index.dayofweek >= 5, "kwh"].median()), 1),
        "train_end": calendar["train_end"],
        "cal_start": calendar["cal_start"],
        "cal_end": calendar["cal_end"],
        "test_start": str(calendar["replay"].index.min().date()),
        "test_end": str(calendar["replay"].index.max().date()),
        "calendar": _metrics_block(calendar),
        "weather": _metrics_block(weather),
        "upcoming_date": (
            str(calendar["upcoming"]["date"].date()) if calendar["upcoming"] else None
        ),
        "extra_rows": (
            int(len(pd.read_csv(EXTRA_PATH, usecols=["timestamp"])))
            if EXTRA_PATH.exists() and EXTRA_PATH.stat().st_size > 0
            else 0
        ),
    }
    (ART_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _metrics_block(result: dict) -> dict:
    metrics = result["metrics"]
    return {
        "n": metrics["n"],
        "train_n": metrics["train_n"],
        "cal_n": metrics["cal_n"],
        "mae": _round(metrics["mae"], 2),
        "wape": _round(metrics["wape"], 3),
        "bias": _round(metrics["bias"], 2),
        "n_in_range": metrics["n_in_range"],
        "coverage": _round(metrics["coverage"], 3),
        "train_mae": _round(metrics["train_mae"], 2),
        "baseline_mae": _round(metrics["baseline"]["mae"], 2),
        "baseline_wape": _round(metrics["baseline"]["wape"], 3),
        "baseline_bias": _round(metrics["baseline"]["bias"], 2),
        "factors": [
            {"label": row["label"], "share": _round(row["share"], 3)}
            for row in metrics["factors"]
        ],
    }


def main() -> None:
    absorb_incoming()
    samples = load_samples()
    daily = build_daily(samples)
    frame = build_frame(daily)
    calendar = run_model(frame, FEATURES)
    weather = run_model(frame, FEATURES_WEATHER)
    save_outputs(daily, calendar, weather)


if __name__ == "__main__":
    main()
