import configparser
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import pandas as pd
from datetime import datetime, timedelta, timezone
from dateutil.parser import parse as parse_datetime
from dateutil.relativedelta import relativedelta
import time

# --- Load credentials securely from config file ---
config = configparser.ConfigParser()
config.read("oanda.cfg")
account_id = config['oanda']['account_id']
access_token = config['oanda']['access_token']

# --- Initialize OANDA API client ---
client = API(access_token=access_token)

instrument = "EUR_USD"
granularity = "H4"  

# --- Define time periods in days ---
periods = {
    "10_years": 365 * 10 # (6 / 7)
}

# EXPERIMENT RULE:
# - Do NOT test on Dec 2025.
# - So set an experiment cutoff at the start of Dec 2025.
# - You may still DOWNLOAD through yesterday/today/etc. (fetch_end),
#   but the backtest/test must use experiment_cutoff as the end boundary.
#
experiment_cutoff = datetime(2025, 12, 1, 0, 0, 0, tzinfo=timezone.utc)  # EXCLUDE Dec 2025 from testing
fetch_end = datetime(2025, 12, 25, 23, 59, 59, tzinfo=timezone.utc)      # can be yesterday/today/05-12 etc.


def _align_to_grid(dt, granularity):
    if granularity.startswith("M"):
        n = int(granularity[1:])
        return dt.replace(minute=(dt.minute // n) * n, second=0, microsecond=0)
    if granularity.endswith("H"):
        n = int(granularity[:-1])
        return dt.replace(hour=(dt.hour // n) * n, minute=0, second=0, microsecond=0)
    return dt

def fetch_candles(start, end, granularity):
    start = _align_to_grid(start, granularity)  # NEW: snap to 00/15/30/45 for M15
    all_data = []
    current_time = start
    previous_last_time = None

    while current_time < end:
        params = {
            "from": current_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "granularity": granularity,
            "count": 5000,
            "price": "MBA"  # Request mid, bid, and ask prices
        }

        try:
            r = instruments.InstrumentsCandles(instrument=instrument, params=params)
            client.request(r)
            candles = r.response.get("candles")
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

        if not candles:
            print("No more data returned.")
            break

        batch = []
        for candle in candles:
            if not candle["complete"]:
                continue

            time_str = candle["time"]
            candle_time = parse_datetime(time_str)

            if candle_time >= end:
                break

            mid = candle.get("mid", {})
            bid = candle.get("bid", {})
            ask = candle.get("ask", {})

            try:
                spread = float(ask["c"]) - float(bid["c"])
            except KeyError:
                spread = None

            batch.append({
                "time": time_str,
                "mid_open": mid.get("o"),
                "mid_high": mid.get("h"),
                "mid_low": mid.get("l"),
                "mid_close": mid.get("c"),
                "bid_open": bid.get("o"),
                "bid_close": bid.get("c"),
                "ask_open": ask.get("o"),
                "ask_close": ask.get("c"),
                "spread": spread,
                "volume": candle.get("volume"),  # OANDA provides this
                "complete": candle["complete"]
            })
            
        if not batch:
            print("No new candles before end time.")
            break

        last_time = parse_datetime(batch[-1]["time"])

        if previous_last_time and last_time == previous_last_time:
            print(f"Last candle time {last_time} repeated. Breaking to avoid infinite loop.")
            break

        all_data.extend(batch)
        previous_last_time = last_time
        current_time = last_time + timedelta(seconds=1)

        print(f"Fetched {len(batch)} candles ending at {last_time.isoformat()}")
        time.sleep(1)

    return pd.DataFrame(all_data)



for label, days in periods.items():
    # Anchor the 10y window to the experiment cutoff (so testing ends at 2025-12-01)
    # This gives 2015-12-01 -> 2025-12-01 for the 10y experiment window.
    start_date = experiment_cutoff - relativedelta(years=10)
    print(f"\nFetching {label} from {start_date.date()} to {fetch_end.date()} (download end)...")
    print(f"Experiment cutoff (test end, exclusive): {experiment_cutoff.isoformat()}  -> last test month = 11/2025")
    df = fetch_candles(start=start_date, end=fetch_end, granularity=granularity)
    if df.empty:
        print(f"No data retrieved for {label}.")
        continue

    # --- Add this block ---
    if "complete" in df.columns:
        df = df.drop(columns=["complete"])
    cols = [
        "time", "mid_open", "mid_high", "mid_low", "mid_close",
        "bid_open", "bid_close", "ask_open", "ask_close", "spread", "volume"
    ]
    df = df[cols]

    # --- Patch: Ensure datetime type and clean data integrity ---
    df["time"] = pd.to_datetime(df["time"])  # Convert to datetime
    df = df.drop_duplicates("time")          # Drop duplicate timestamps
    df = df.sort_values("time")              # Sort chronologically
    df = df.reset_index(drop=True)           # Reset index after sort
    # ------------------------------------------------------------

    filename = f"csv_data/EURUSD_{label}_H4_OANDA.csv"
    df.to_csv(filename, index=False)
    print(f"Saved: {filename}")