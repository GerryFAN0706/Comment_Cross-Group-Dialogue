import pandas as pd
import numpy as np

def parse_ts(series: pd.Series, tz: str = "Asia/Shanghai") -> pd.Series:
    """
    Parse timestamp strings of format 'YYYY-MM-DD HH:MM:SS' into tz-aware pandas Timestamps.
    If strings vary, consider using pd.to_datetime with errors='coerce'.
    """
    dt = pd.to_datetime(series, errors="coerce")
    if tz:
        # Assume naive times are in local tz already
        dt = dt.dt.tz_localize(tz, nonexistent="NaT", ambiguous="NaT")
    return dt

def hour_of_day(ts: pd.Series) -> pd.Series:
    return ts.dt.hour

def day_of_week(ts: pd.Series) -> pd.Series:
    # Monday=0
    return ts.dt.dayofweek

def to_minute(ts: pd.Series) -> pd.Series:
    return ts.dt.floor("T")

def safe_timedelta_minutes(ts: pd.Series, ref: pd.Timestamp) -> pd.Series:
    # returns minutes difference ts - ref
    return (ts - ref).dt.total_seconds() / 60.0
