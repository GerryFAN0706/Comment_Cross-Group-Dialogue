from __future__ import annotations

from typing import Dict, Tuple, Optional

import pandas as pd


def median_timestamp(values: pd.Series) -> pd.Timestamp:
    """
    Return the (lower) median timestamp from a Series.
    - Drops NaT/NaN
    - Sorts ascending
    - Returns the middle element (lower median for even n)
    """
    s = pd.to_datetime(values, errors="coerce").dropna().sort_values()
    if s.empty:
        return pd.NaT
    return s.iloc[(len(s) - 1) // 2]


def median_timedelta(values: pd.Series) -> pd.Timedelta:
    """
    Return the (lower) median timedelta from a Series.
    - Drops NaT/NaN
    - Sorts ascending
    - Returns the middle element (lower median for even n)
    """
    s = pd.to_timedelta(values, errors="coerce").dropna().sort_values()
    if s.empty:
        return pd.NaT
    return s.iloc[(len(s) - 1) // 2]


def build_control_anchor_map(
    matched_pairs: pd.DataFrame,
    tstar_map: Dict[str, pd.Timestamp],
    *,
    strategy: str = "matched_median",
    created_at_map: Optional[Dict[str, pd.Timestamp]] = None,
) -> Tuple[Dict[str, pd.Timestamp], Dict[str, int]]:
    """
    Build pseudo-anchor times for control threads from their matched treated counterparts.

    Parameters
    - matched_pairs: DataFrame with columns ['mblogid_t','mblogid_c', ...]
    - tstar_map: dict treated_mblogid -> tstar timestamp
    - strategy:
        - 'matched_median' (recommended): median of matched treated t*
        - 'matched_median_latency': median of matched treated (t* - thread_created_at),
          then add to the control thread's created_at. This keeps anchors *within* each
          control thread's own timeline and avoids absolute-time mismatches.
    - created_at_map: dict mblogid -> thread created_at (required for matched_median_latency)

    Returns
    - anchor_map: control_mblogid -> pseudo anchor timestamp (NaT if unavailable)
    - n_matches: control_mblogid -> number of matched treated threads used
    """
    if matched_pairs is None or matched_pairs.empty:
        return {}, {}

    if not {"mblogid_t", "mblogid_c"}.issubset(matched_pairs.columns):
        return {}, {}

    strategy = (strategy or "matched_median").strip().lower()
    if strategy not in {"matched_median", "matched_median_latency"}:
        # Fallback to matched_median if unknown strategy
        strategy = "matched_median"

    anchor_map: Dict[str, pd.Timestamp] = {}
    n_matches: Dict[str, int] = {}

    grouped = matched_pairs.groupby("mblogid_c")["mblogid_t"].apply(list)
    for ctrl_id, treated_list in grouped.items():
        if strategy == "matched_median_latency":
            # Need thread-level created_at to compute treated reply latency and to place
            # the pseudo-anchor within the control's own thread timeline.
            if not created_at_map:
                anchor_map[ctrl_id] = pd.NaT
                n_matches[ctrl_id] = 0
                continue
            ctrl_t0 = pd.to_datetime(created_at_map.get(ctrl_id, pd.NaT), errors="coerce")
            if pd.isna(ctrl_t0):
                anchor_map[ctrl_id] = pd.NaT
                n_matches[ctrl_id] = 0
                continue

            latencies = []
            for tid in treated_list:
                tstar = pd.to_datetime(tstar_map.get(tid, pd.NaT), errors="coerce")
                t0 = pd.to_datetime(created_at_map.get(tid, pd.NaT), errors="coerce")
                if pd.isna(tstar) or pd.isna(t0):
                    continue
                latencies.append(tstar - t0)

            med_lat = median_timedelta(pd.Series(latencies))
            anchor_map[ctrl_id] = (ctrl_t0 + med_lat) if not pd.isna(med_lat) else pd.NaT
            n_matches[ctrl_id] = int(pd.Series(latencies).notna().sum())
        else:
            # matched_median: absolute median of matched treated t*
            tvals = [tstar_map.get(tid, pd.NaT) for tid in treated_list]
            med = median_timestamp(pd.Series(tvals))
            anchor_map[ctrl_id] = med
            n_matches[ctrl_id] = int(pd.Series(tvals).notna().sum())

    return anchor_map, n_matches

