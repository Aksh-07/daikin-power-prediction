# Next-day air-conditioner electricity

Proof of concept for one Daikin outdoor unit. Each morning it gives **one number**: how many units of electricity (kWh) the unit is likely to use that day, and a **range**. About 8 times out of 10, the real use should fall inside the range.

The screen is written for a non-technical reader. Pick any morning from April through June 2023 to see the forecast as it would have been given before that day unfolded, then what the unit actually used.

## What is in the box

- `data_D.csv` — 5-minute readings from 19 April 2022 through 30 June 2023 (UTC timestamps). Days on the screen are midnight to midnight, India time.
- `forecast.py` — turns those readings into daily energy, trains the morning-by-morning forecast, and writes `artifacts/`.
- `app.py` — the Streamlit page.
- `artifacts/` — the forecast already built, so the page opens without retraining.

The model is [XGBoost](https://xgboost.readthedocs.io/). It sees the calendar, recent healthy days, and recent outdoor temperature. It does not see same-time compressor temperature or refrigerant pressure, because those describe electricity already being used.

## What the check showed

The forecast for each morning in April–June 2023 was trained only on healthy days before that morning (263 healthy days before 1 April). Stuck-meter days are left out of the score. Real off days stay in.

| Question | Result |
|---|---|
| Typical miss | about 5.1 kWh |
| Misses as a share of electricity used | 25% |
| Average bias | about 1.5 kWh low |
| Range contained the real day | 54 of 71 days |
| Copying the same weekday last week | about 10.3 kWh typical miss |

Precision, recall, and a confusion matrix do not apply. This predicts a quantity, not a yes-or-no class.

## Run it

Python 3.11 or newer.

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open http://localhost:8501.

Rebuild the forecast from the readings (about half a minute):

```powershell
python forecast.py
```

## Add new readings

The file must have the same columns as `data_D.csv`.

**On the page.** Scroll to **Add new readings**, upload the CSV, and choose **Update the forecast**. New times are added. A time that already exists is replaced. The original `data_D.csv` is not overwritten. Added rows are stored in `data_extra.csv`, which is gitignored.

**Without the page.** Drop the CSV into the `incoming` folder and run:

```powershell
python forecast.py
```

A daily schedule can run that command after yesterday’s file arrives. Refresh the page and the new morning is there. The smallest useful columns are `timestamp`, `power` (kW), and `odu_outdoortemp`.

## Layout

```
data_D.csv          original 5-minute readings
forecast.py         daily table, model, artifacts
app.py              client page
requirements.txt
artifacts/          forecast, range, and reasons
incoming/           drop a new CSV here for the next rebuild
```
