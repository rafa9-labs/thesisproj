csv_data/README.txt
===================

Purpose
-------
This folder contains example CSV price data so reviewers can run the pipeline
without external accounts (e.g., OANDA). You can replace the CSVs with any
equivalent dataset, as long as you keep the required column format.

Quick start (using the included sample CSV)
-------------------------------------------
1) Leave the sample CSV(s) in this folder as-is.
2) Run from project root:

   python MLBacktesterNoWFO.py

If the project is configured for a specific instrument/timeframe, make sure the
sample CSV filename and the config match what the loader expects (see below).

Required CSV format (minimum)
-----------------------------
Your CSV file must be regular time-series bars with at least:

- time (timestamp)
- open
- high
- low
- close

Recommended (if available):
- volume

Column name conventions
-----------------------
The loader expects standard OHLC naming. Use ONE of these common conventions:

Option A (preferred):
- time, open, high, low, close, volume

Option B (also common):
- datetime, Open, High, Low, Close, Volume

Timestamps must be:
- strictly increasing
- no duplicates
- parseable by pandas (ISO 8601 is safest)

Timeframe expectations
----------------------
This thesis pipeline was evaluated on a fixed bar grid (e.g., 30-minute bars).
If you provide data at a different timeframe, results will differ and you should
update the config accordingly.

Where to change which file is loaded
------------------------------------
The dataset path / symbol / timeframe mapping is defined in the project config
and/or loader code.

Start here:
- configs/feature_config.json   (feature + gating settings; may include symbol/timeframe)
- MLBacktesterNoWFO.py          (main entrypoint and dataset wiring)

If you replace the sample CSV:
------------------------------
To test with your own data:

1) Put your CSV into this folder: csv_data/
2) Ensure it follows the OHLC format above.
3) Update any dataset path/symbol setting in the config/code so the loader points
   to your new file.
4) Run:

   python MLBacktesterNoWFO.py

Notes / common pitfalls
-----------------------
- Missing bars: If your data has gaps, the pipeline may behave differently because
  indicators and rolling windows assume consistent spacing.
- Timezones: Use a single timezone (UTC recommended).
- Extra columns are fine (they will be ignored unless explicitly used).

Optional: Getting data from external sources
--------------------------------------------
You can use any provider (broker export, Dukascopy, HistData, Kaggle, etc.).
As long as the file matches the OHLC format and is on a consistent bar grid,
the pipeline can be used for a functional test run.
