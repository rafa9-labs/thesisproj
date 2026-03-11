"""
DataFactory: Multi-source data loading with CSV and Oanda support.
Handles multiple timeframes (M5, H1, D1) with validation and logging.
"""

import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dateutil.parser import parse as parse_datetime
from typing import Optional, Dict, Literal, Union, List
import time

from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments

from ..core.config import AppConfig


logger = logging.getLogger(__name__)


class DataFactory:
    """
    Factory for loading forex data from multiple sources.
    Supports CSV files and live Oanda API with multi-timeframe capabilities.
    """
    
    STANDARD_COLUMNS = [
        "time", "mid_open", "mid_high", "mid_low", "mid_close",
        "bid_open", "bid_close", "ask_open", "ask_close", "spread", "volume"
    ]
    
    GRANULARITY_MAP = {
        "M5": "M5",
        "M15": "M15",
        "M30": "M30",
        "H1": "H1",
        "H4": "H4",
        "D1": "D",
    }
    
    def __init__(self, config: AppConfig):
        """
        Initialize DataFactory with configuration.
        
        Args:
            config: AppConfig instance with Oanda credentials and settings
        """
        self.config = config
        self._oanda_client: Optional[API] = None
        
        logger.info("DataFactory initialized")
    
    @property
    def oanda_client(self) -> API:
        """Lazy initialization of Oanda API client"""
        if self._oanda_client is None:
            if not self.config.oanda.access_token:
                raise ValueError("Oanda access token not configured")
            
            self._oanda_client = API(access_token=self.config.oanda.access_token)
            logger.info(f"Oanda API client initialized (account: {self.config.oanda.account_type})")
        
        return self._oanda_client
    
    def load_csv(
        self,
        filepath: Union[str, Path],
        timeframe: Optional[str] = None,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Load data from CSV file with validation.
        
        Args:
            filepath: Path to CSV file
            timeframe: Optional timeframe label (for logging)
            validate: Whether to validate and clean data
            
        Returns:
            DataFrame with standardized columns
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.error(f"CSV file not found: {filepath}")
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} rows from {filepath.name}")
            
            if validate:
                df = self._validate_and_clean(df, source="csv")
            
            if timeframe:
                logger.debug(f"Timeframe {timeframe}: {len(df)} candles")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV {filepath}: {e}", exc_info=True)
            raise
    
    def fetch_oanda(
        self,
        instrument: str,
        granularity: str,
        start: datetime,
        end: datetime,
        is_live: bool = False,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Fetch data from Oanda API.
        
        Args:
            instrument: Instrument code (e.g., "EUR_USD")
            granularity: Oanda granularity (M5, H1, H4, D, etc.)
            start: Start datetime (UTC)
            end: End datetime (UTC)
            is_live: If True, prepare for real-time streaming (future use)
            validate: Whether to validate and clean data
            
        Returns:
            DataFrame with standardized columns
        """
        if is_live:
            logger.warning("Live mode requested but streaming not yet implemented. Fetching historical data.")
        
        try:
            start = self._align_to_grid(start, granularity)
            all_data = []
            current_time = start
            previous_last_time = None
            
            logger.info(f"Fetching {instrument} {granularity} from {start.date()} to {end.date()}")
            
            while current_time < end:
                params = {
                    "from": current_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    "granularity": granularity,
                    "count": 5000,
                    "price": "MBA"
                }
                
                try:
                    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
                    self.oanda_client.request(r)
                    candles = r.response.get("candles")
                except Exception as e:
                    logger.error(f"Oanda API error: {e}", exc_info=True)
                    break
                
                if not candles:
                    logger.warning("No more data returned from Oanda")
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
                    except (KeyError, TypeError):
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
                        "volume": candle.get("volume"),
                    })
                
                if not batch:
                    logger.debug("No new candles before end time")
                    break
                
                last_time = parse_datetime(batch[-1]["time"])
                
                if previous_last_time and last_time == previous_last_time:
                    logger.warning(f"Last candle time {last_time} repeated. Breaking to avoid infinite loop.")
                    break
                
                all_data.extend(batch)
                previous_last_time = last_time
                current_time = last_time + timedelta(seconds=1)
                
                logger.debug(f"Fetched {len(batch)} candles ending at {last_time.isoformat()}")
                time.sleep(1)
            
            df = pd.DataFrame(all_data)
            logger.info(f"Fetched total {len(df)} candles from Oanda")
            
            if validate:
                df = self._validate_and_clean(df, source="oanda")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Oanda data: {e}", exc_info=True)
            raise
    
    def get_data(
        self,
        source: Literal["csv", "oanda"],
        is_live: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Unified interface for loading data from any source.
        
        Args:
            source: Data source ("csv" or "oanda")
            is_live: If True, prepare for real-time data (Oanda only)
            **kwargs: Source-specific parameters
                For CSV: csv_path, filepath, timeframe
                For Oanda: instrument, granularity, start, end
        
        Returns:
            DataFrame with standardized columns
        """
        if source == "csv":
            # Map csv_path to filepath for load_csv compatibility
            if "csv_path" in kwargs:
                kwargs["filepath"] = kwargs.pop("csv_path")
            return self.load_csv(**kwargs)
        elif source == "oanda":
            return self.fetch_oanda(is_live=is_live, **kwargs)
        else:
            raise ValueError(f"Unknown source: {source}. Use 'csv' or 'oanda'")
    
    def load_multi_timeframe(
        self,
        timeframes: List[str],
        source: Literal["csv", "oanda"],
        is_live: bool = False,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple timeframes simultaneously.
        
        Args:
            timeframes: List of timeframes (e.g., ["M5", "H1", "D1"])
            source: Data source ("csv" or "oanda")
            is_live: If True, prepare for real-time data (Oanda only)
            **kwargs: Source-specific parameters
                For CSV: base_path, instrument (for filename pattern)
                For Oanda: instrument, start, end
        
        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        logger.info(f"Loading {len(timeframes)} timeframes: {timeframes}")
        
        result = {}
        
        for tf in timeframes:
            try:
                if source == "csv":
                    base_path = kwargs.get("base_path", "csv_data")
                    instrument = kwargs.get("instrument", "EURUSD")
                    
                    # Construct filename pattern: EURUSD_10_years_H4_OANDA.csv
                    period = kwargs.get("period", "10_years")
                    filename = f"{instrument}_{period}_{tf}_OANDA.csv"
                    filepath = Path(base_path) / filename
                    
                    result[tf] = self.load_csv(filepath, timeframe=tf)
                    
                elif source == "oanda":
                    granularity = self.GRANULARITY_MAP.get(tf, tf)
                    
                    result[tf] = self.fetch_oanda(
                        instrument=kwargs["instrument"],
                        granularity=granularity,
                        start=kwargs["start"],
                        end=kwargs["end"],
                        is_live=is_live
                    )
                
                logger.info(f"✓ {tf}: {len(result[tf])} candles loaded")
                
            except Exception as e:
                logger.error(f"✗ Failed to load {tf}: {e}")
                result[tf] = pd.DataFrame()
        
        return result
    
    def _validate_and_clean(
        self,
        df: pd.DataFrame,
        source: str
    ) -> pd.DataFrame:
        """
        Validate and clean DataFrame.
        
        Args:
            df: Raw DataFrame
            source: Data source name (for logging)
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            logger.warning(f"Empty DataFrame from {source}")
            return df
        
        original_len = len(df)
        
        # Remove 'complete' column if present
        if "complete" in df.columns:
            df = df.drop(columns=["complete"])
        
        # Ensure standard columns exist
        for col in self.STANDARD_COLUMNS:
            if col not in df.columns:
                logger.warning(f"Missing column '{col}' in {source} data")
        
        # Reorder columns to standard format
        available_cols = [c for c in self.STANDARD_COLUMNS if c in df.columns]
        df = df[available_cols]
        
        # Convert time to datetime
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
        
        # Convert price columns to numeric
        price_cols = ["mid_open", "mid_high", "mid_low", "mid_close", 
                     "bid_open", "bid_close", "ask_open", "ask_close", "spread", "volume"]
        for col in price_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop duplicates
        if "time" in df.columns:
            df = df.drop_duplicates("time")
            duplicates_removed = original_len - len(df)
            if duplicates_removed > 0:
                logger.debug(f"Removed {duplicates_removed} duplicate timestamps")
        
        # Sort chronologically
        if "time" in df.columns:
            df = df.sort_values("time")
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            logger.debug(f"Missing values detected:\n{missing_counts[missing_counts > 0]}")
        
        logger.debug(f"Validation complete: {len(df)} rows (cleaned from {original_len})")
        
        return df
    
    @staticmethod
    def _align_to_grid(dt: datetime, granularity: str) -> datetime:
        """
        Align datetime to granularity grid.
        
        Args:
            dt: Datetime to align
            granularity: Oanda granularity (M5, H1, etc.)
            
        Returns:
            Aligned datetime
        """
        if granularity.startswith("M"):
            n = int(granularity[1:])
            return dt.replace(minute=(dt.minute // n) * n, second=0, microsecond=0)
        
        if granularity.endswith("H") or granularity.startswith("H"):
            n = int(granularity.replace("H", ""))
            return dt.replace(hour=(dt.hour // n) * n, minute=0, second=0, microsecond=0)
        
        if granularity == "D":
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        
        return dt


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure logging for the data module.
    
    Args:
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger.setLevel(level)
