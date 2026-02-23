"""Domain model module for the Snow Removal Deployment Tracking system.

Maps WhatsApp chat export data (pandas DataFrame) into formalized snow removal
domain entities such as job logs, crew summaries, route segments, and deployment
burndown metrics.

This module is standalone and does NOT depend on dashboard_web.py or Dash.
"""

import json
import math
import os
import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Haversine distance calculation
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_location_coords(location_registry):
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower().strip())
    coords = {}
    for entry in location_registry:
        lat = entry.get("lat")
        lon = entry.get("lon")
        if lat is None or lon is None or lat == "" or lon == "":
            continue
        try:
            lat_f, lon_f = float(lat), float(lon)
        except (ValueError, TypeError):
            continue
        for field in ["_matched_raw", "location_name", "address"]:
            val = entry.get(field, "")
            if val:
                key = norm(val)
                if key and key not in coords:
                    coords[key] = (lat_f, lon_f)
        mcl = entry.get("matched_chat_location", "")
        if mcl and "(" in mcl:
            raw = mcl.split("(")[0].strip()
            key = norm(raw)
            if key and key not in coords:
                coords[key] = (float(lat), float(lon))
    return coords


CITY_AVG_SPEED_KMH = 25.0


def estimate_travel_mins(distance_km, speed_kmh=None):
    if speed_kmh is None:
        speed_kmh = CITY_AVG_SPEED_KMH
    if distance_km <= 0 or speed_kmh <= 0:
        return 0.0
    return (distance_km / speed_kmh) * 60.0


# ---------------------------------------------------------------------------
# Helper: auto-detect location type from chat name
# ---------------------------------------------------------------------------

def infer_location_type(chat_name):
    """Infer the location type from a chat/group name.

    Returns:
        str: "Sidewalk" if the name contains 'sidewalk' (case-insensitive),
             "Parking Lot" if it contains 'parking', otherwise "Unknown".
    """
    if not chat_name:
        return "Unknown"
    name_lower = chat_name.lower()
    if "sidewalk" in name_lower:
        return "Sidewalk"
    if "parking" in name_lower:
        return "Parking Lot"
    return "Unknown"


# ---------------------------------------------------------------------------
# 1. Configuration Loading
# ---------------------------------------------------------------------------

def load_config(config_path):
    """Load a JSON configuration file for the snow removal domain model.

    The config file may contain:
        - deployment_types: dict mapping deployment label -> type enum
        - location_types: dict mapping normalized location name -> type
        - crew_config: dict mapping crew/chat name -> {crew_size, machines}
        - non_trackable_senders: list of sender names to exclude from KPIs
        - standard_travel_times: dict mapping "from|to" -> minutes
        - expected_service_times: dict mapping location_type -> minutes

    If the file does not exist, sensible defaults (empty dicts/lists) are
    returned.

    Args:
        config_path: Path to the JSON config file.

    Returns:
        dict with all configuration keys populated.
    """
    defaults = {
        "deployment_types": {},
        "location_types": {},
        "crew_config": {},
        "non_trackable_senders": [],
        "standard_travel_times": {},
        "expected_service_times": {},
    }

    if not os.path.exists(config_path):
        return defaults

    try:
        with open(config_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return defaults

    for key in defaults:
        if key not in data:
            data[key] = defaults[key]

    return data


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_location_type(location, chat_name, config):
    """Resolve location type from config or auto-detect from chat name."""
    loc_types = config.get("location_types", {})
    if location and location in loc_types:
        return loc_types[location]
    return infer_location_type(chat_name)


def _empty_df(columns):
    """Return an empty DataFrame with the given columns."""
    return pd.DataFrame(columns=columns)


# ---------------------------------------------------------------------------
# 2. Domain Builder Functions
# ---------------------------------------------------------------------------

def build_job_logs(df, config):
    """Build job log records from raw chat messages.

    Each image-type message (type == "image") or message with a non-empty
    location field is treated as a job completion report.

    Args:
        df: Raw messages DataFrame with columns from dashboard_web.py.
        config: Configuration dict (output of load_config).

    Returns:
        DataFrame with columns: crew, deployment, date, location,
        location_type, picture_submitted_at, chat, is_recall,
        recall_added_time_mins.
    """
    cols = [
        "crew", "deployment", "date", "location", "location_type",
        "picture_submitted_at", "chat", "is_recall", "recall_added_time_mins",
    ]

    if df is None or df.empty:
        return _empty_df(cols)

    required_cols = ["type", "location", "sender_resolved", "chat", "time", "msg_date"]
    for rc in required_cols:
        if rc not in df.columns:
            return _empty_df(cols)

    mask = (df["type"] == "image") | (df["location"].astype(str).str.len() > 0)
    subset = df.loc[mask].copy()

    if subset.empty:
        return _empty_df(cols)

    job_logs = pd.DataFrame()
    job_logs["crew"] = subset["sender_resolved"]
    job_logs["deployment"] = subset.get("deployment", pd.Series(dtype="object"))
    job_logs["date"] = pd.to_datetime(subset["msg_date"])
    job_logs["location"] = subset["location"]
    job_logs["chat"] = subset["chat"]
    job_logs["picture_submitted_at"] = pd.to_datetime(subset["time"])

    job_logs["location_type"] = subset.apply(
        lambda row: _resolve_location_type(
            row.get("location", ""), row.get("chat", ""), config
        ),
        axis=1,
    )

    job_logs = job_logs.reset_index(drop=True)

    job_logs["is_recall"] = False
    job_logs["recall_added_time_mins"] = np.nan

    job_logs_sorted = job_logs.sort_values(
        ["crew", "deployment", "location", "picture_submitted_at"]
    ).reset_index(drop=True)

    for idx in range(len(job_logs_sorted)):
        row = job_logs_sorted.iloc[idx]
        if not row["crew"] or not row["location"] or pd.isna(row["deployment"]):
            continue
        earlier = job_logs_sorted.iloc[:idx]
        same = earlier[
            (earlier["crew"] == row["crew"])
            & (earlier["deployment"] == row["deployment"])
            & (earlier["location"] == row["location"])
        ]
        if not same.empty:
            job_logs_sorted.at[idx, "is_recall"] = True
            last_visit = same["picture_submitted_at"].max()
            if pd.notna(last_visit) and pd.notna(row["picture_submitted_at"]):
                delta = (
                    row["picture_submitted_at"] - last_visit
                ).total_seconds() / 60.0
                job_logs_sorted.at[idx, "recall_added_time_mins"] = delta

    return job_logs_sorted[cols].reset_index(drop=True)


def build_crew_summary(job_logs_df, config, location_coords=None):
    """Build per-crew summary statistics from job logs.

    Args:
        job_logs_df: DataFrame output of build_job_logs.
        config: Configuration dict (output of load_config).
        location_coords: dict mapping normalized location name to (lat, lon).

    Returns:
        DataFrame with columns: crew, days_active, total_sites, total_recalls,
        avg_sites_per_hour, avg_transition_min, crew_size, machines.
    """
    cols = [
        "crew", "days_active", "total_sites", "total_recalls",
        "avg_sites_per_hour", "avg_transition_min", "crew_size", "machines",
    ]

    if job_logs_df is None or job_logs_df.empty:
        return _empty_df(cols)

    trackable = filter_trackable(job_logs_df, config, sender_col="crew")
    if trackable.empty:
        return _empty_df(cols)

    crew_config = config.get("crew_config", {})
    rows = []

    for crew, grp in trackable.groupby("crew"):
        days_active = grp["date"].dt.date.nunique()
        total_sites = len(grp)
        total_recalls = int(grp["is_recall"].sum())

        times = grp["picture_submitted_at"].dropna().sort_values()
        if len(times) >= 2:
            total_hours = (times.max() - times.min()).total_seconds() / 3600.0
            avg_sites_per_hour = (
                total_sites / total_hours if total_hours > 0 else 0.0
            )
        else:
            avg_sites_per_hour = 0.0

        segments = build_route_segments(grp, config, location_coords)
        avg_transition_min = (
            segments["actual_duration_mins"].mean()
            if not segments.empty
            else 0.0
        )

        cc = crew_config.get(crew, {})
        crew_size = cc.get("crew_size", 1)
        machines = cc.get("machines", [])

        rows.append({
            "crew": crew,
            "days_active": days_active,
            "total_sites": total_sites,
            "total_recalls": total_recalls,
            "avg_sites_per_hour": round(avg_sites_per_hour, 2),
            "avg_transition_min": round(avg_transition_min, 1),
            "crew_size": crew_size,
            "machines": machines,
        })

    return pd.DataFrame(rows, columns=cols)


def build_route_segments(job_logs_df, config, location_coords=None):
    """Build route segments from consecutive job logs for the same crew/day.

    Each pair of consecutive job log entries for the same crew on the same
    day forms a route segment.

    Args:
        job_logs_df: DataFrame output of build_job_logs.
        config: Configuration dict (output of load_config).
        location_coords: dict mapping normalized location name to (lat, lon).

    Returns:
        DataFrame with columns: crew, date, origin_location,
        destination_location, standard_travel_time_mins,
        actual_duration_mins, is_delayed, distance_km,
        estimated_travel_mins, travel_efficiency.
    """
    cols = [
        "crew", "date", "origin_location", "destination_location",
        "start_time", "end_time",
        "standard_travel_time_mins", "actual_duration_mins", "is_delayed",
        "distance_km", "estimated_travel_mins", "travel_efficiency",
    ]

    if job_logs_df is None or job_logs_df.empty:
        return _empty_df(cols)

    if location_coords is None:
        location_coords = {}

    _norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower().strip())

    def _lookup_coords(loc_name):
        key = _norm(loc_name)
        if key in location_coords:
            return location_coords[key]
        for k, v in location_coords.items():
            if key in k or k in key:
                return v
        return None

    sorted_df = job_logs_df.sort_values(
        ["crew", "date", "picture_submitted_at"]
    ).reset_index(drop=True)

    standard_times = config.get("standard_travel_times", {})
    expected_service = config.get("expected_service_times", {})
    rows = []

    for (crew, dt), grp in sorted_df.groupby(
        ["crew", sorted_df["date"].dt.date]
    ):
        grp = grp.reset_index(drop=True)
        for i in range(1, len(grp)):
            origin = grp.iloc[i - 1]
            dest = grp.iloc[i]

            origin_loc = origin["location"]
            dest_loc = dest["location"]

            key = f"{origin_loc}|{dest_loc}"
            std_travel = standard_times.get(key)

            t0 = origin["picture_submitted_at"]
            t1 = dest["picture_submitted_at"]
            if pd.notna(t0) and pd.notna(t1):
                actual_mins = (t1 - t0).total_seconds() / 60.0
            else:
                actual_mins = np.nan

            dist_km = np.nan
            est_travel = np.nan
            efficiency = np.nan
            o_coords = _lookup_coords(origin_loc)
            d_coords = _lookup_coords(dest_loc)
            if o_coords and d_coords:
                dist_km = haversine_km(o_coords[0], o_coords[1], d_coords[0], d_coords[1])
                est_travel = estimate_travel_mins(dist_km)
                if std_travel is None and pd.notna(est_travel):
                    std_travel = round(est_travel, 1)

            is_delayed = False
            if pd.notna(actual_mins) and std_travel is not None:
                origin_type = origin.get("location_type", "Unknown")
                svc_time = expected_service.get(origin_type, 0)
                if actual_mins > svc_time + std_travel:
                    is_delayed = True

            if pd.notna(actual_mins) and pd.notna(est_travel) and est_travel > 0:
                efficiency = round(est_travel / actual_mins * 100, 1) if actual_mins > 0 else 100.0

            rows.append({
                "crew": crew,
                "date": pd.Timestamp(dt),
                "origin_location": origin_loc,
                "destination_location": dest_loc,
                "start_time": t0,
                "end_time": t1,
                "standard_travel_time_mins": std_travel,
                "actual_duration_mins": round(actual_mins, 1) if pd.notna(actual_mins) else np.nan,
                "is_delayed": is_delayed,
                "distance_km": round(dist_km, 2) if pd.notna(dist_km) else np.nan,
                "estimated_travel_mins": round(est_travel, 1) if pd.notna(est_travel) else np.nan,
                "travel_efficiency": efficiency if pd.notna(efficiency) else np.nan,
            })

    if not rows:
        return _empty_df(cols)

    return pd.DataFrame(rows, columns=cols)


def build_deployment_burndown(job_logs_df, deployments_list, config):
    """Build deployment burndown data comparing actual vs expected pace.

    For each deployment, computes cumulative sites completed over time
    against an expected linear pace.

    Args:
        job_logs_df: DataFrame output of build_job_logs.
        deployments_list: List of deployment dicts with keys: label,
            start_date, end_date.
        config: Configuration dict (output of load_config).

    Returns:
        DataFrame with columns: deployment, timestamp, cumulative_completed,
        expected_completed, pct_complete.
    """
    cols = [
        "deployment", "timestamp", "cumulative_completed",
        "expected_completed", "pct_complete",
    ]

    if job_logs_df is None or job_logs_df.empty or not deployments_list:
        return _empty_df(cols)

    expected_duration_hours = config.get("expected_deployment_hours", 12.0)
    rows = []

    for dep in deployments_list:
        label = dep.get("label", "")
        dep_logs = job_logs_df[job_logs_df["deployment"] == label].copy()
        if dep_logs.empty:
            continue

        dep_logs = dep_logs.sort_values("picture_submitted_at").reset_index(drop=True)
        total_sites = len(dep_logs)
        start_time = dep_logs["picture_submitted_at"].min()

        if pd.isna(start_time):
            continue

        for i, (_, row) in enumerate(dep_logs.iterrows(), 1):
            ts = row["picture_submitted_at"]
            if pd.isna(ts):
                continue
            elapsed_hours = (ts - start_time).total_seconds() / 3600.0
            expected = (
                total_sites / expected_duration_hours * elapsed_hours
            )
            pct = (i / total_sites * 100.0) if total_sites > 0 else 0.0
            rows.append({
                "deployment": label,
                "timestamp": ts,
                "cumulative_completed": i,
                "expected_completed": round(expected, 2),
                "pct_complete": round(pct, 2),
            })

    if not rows:
        return _empty_df(cols)

    return pd.DataFrame(rows, columns=cols)


def build_location_type_stats(job_logs_df):
    """Compute average service time per location type.

    Uses time gaps between consecutive job logs at the same location type
    as a proxy for service duration.

    Args:
        job_logs_df: DataFrame output of build_job_logs.

    Returns:
        DataFrame with columns: location_type, avg_duration_min, count,
        std_duration_min.
    """
    cols = ["location_type", "avg_duration_min", "count", "std_duration_min"]

    if job_logs_df is None or job_logs_df.empty:
        return _empty_df(cols)

    sorted_df = job_logs_df.sort_values(
        ["crew", "date", "picture_submitted_at"]
    ).reset_index(drop=True)

    durations = []
    for (crew, dt), grp in sorted_df.groupby(
        ["crew", sorted_df["date"].dt.date]
    ):
        grp = grp.reset_index(drop=True)
        for i in range(1, len(grp)):
            t0 = grp.iloc[i - 1]["picture_submitted_at"]
            t1 = grp.iloc[i]["picture_submitted_at"]
            loc_type = grp.iloc[i - 1]["location_type"]
            if pd.notna(t0) and pd.notna(t1):
                dur = (t1 - t0).total_seconds() / 60.0
                if dur > 0:
                    durations.append({
                        "location_type": loc_type,
                        "duration_min": dur,
                    })

    if not durations:
        return _empty_df(cols)

    dur_df = pd.DataFrame(durations)
    stats = dur_df.groupby("location_type")["duration_min"].agg(
        avg_duration_min="mean",
        count="count",
        std_duration_min="std",
    ).reset_index()

    stats["avg_duration_min"] = stats["avg_duration_min"].round(1)
    stats["std_duration_min"] = stats["std_duration_min"].round(1).fillna(0.0)
    stats["count"] = stats["count"].astype(int)

    return stats[cols]


def build_traffic_analysis(route_segments_df):
    """Compute average actual travel time between location pairs.

    Args:
        route_segments_df: DataFrame output of build_route_segments.

    Returns:
        DataFrame with columns: origin, destination, avg_travel_min, count,
        std_travel_min, distance_km, est_travel_min, avg_efficiency.
    """
    cols = ["origin", "destination", "avg_travel_min", "count",
            "std_travel_min", "distance_km", "est_travel_min", "avg_efficiency"]

    if route_segments_df is None or route_segments_df.empty:
        return _empty_df(cols)

    valid = route_segments_df.dropna(subset=["actual_duration_mins"]).copy()
    if valid.empty:
        return _empty_df(cols)

    agg_dict = {
        "actual_duration_mins": ["mean", "count", "std"],
    }
    if "distance_km" in valid.columns:
        agg_dict["distance_km"] = "first"
    if "estimated_travel_mins" in valid.columns:
        agg_dict["estimated_travel_mins"] = "first"
    if "travel_efficiency" in valid.columns:
        agg_dict["travel_efficiency"] = "mean"

    grouped = valid.groupby(
        ["origin_location", "destination_location"]
    ).agg(agg_dict)
    grouped.columns = ["_".join(c) if c[1] else c[0] for c in grouped.columns]
    grouped = grouped.reset_index()

    stats = pd.DataFrame()
    stats["origin"] = grouped["origin_location"]
    stats["destination"] = grouped["destination_location"]
    stats["avg_travel_min"] = grouped["actual_duration_mins_mean"].round(1)
    stats["count"] = grouped["actual_duration_mins_count"].astype(int)
    stats["std_travel_min"] = grouped["actual_duration_mins_std"].round(1).fillna(0.0)
    stats["distance_km"] = grouped.get("distance_km_first", pd.Series(dtype=float)).round(2) if "distance_km_first" in grouped.columns else np.nan
    stats["est_travel_min"] = grouped.get("estimated_travel_mins_first", pd.Series(dtype=float)).round(1) if "estimated_travel_mins_first" in grouped.columns else np.nan
    stats["avg_efficiency"] = grouped.get("travel_efficiency_mean", pd.Series(dtype=float)).round(1) if "travel_efficiency_mean" in grouped.columns else np.nan

    return stats[cols]


def build_delay_report(route_segments_df, config):
    """Build a report of delayed route segments.

    A segment is delayed when actual_duration exceeds expected service time
    plus standard travel time.

    Args:
        route_segments_df: DataFrame output of build_route_segments.
        config: Configuration dict (output of load_config).

    Returns:
        DataFrame with columns: crew, date, origin, destination,
        expected_min, actual_min, delay_min.
    """
    cols = [
        "crew", "date", "origin", "destination",
        "expected_min", "actual_min", "delay_min",
    ]

    if route_segments_df is None or route_segments_df.empty:
        return _empty_df(cols)

    delayed = route_segments_df[route_segments_df["is_delayed"] == True].copy()
    if delayed.empty:
        return _empty_df(cols)

    standard_times = config.get("standard_travel_times", {})
    expected_service = config.get("expected_service_times", {})

    rows = []
    for _, seg in delayed.iterrows():
        key = f"{seg['origin_location']}|{seg['destination_location']}"
        std_travel = standard_times.get(key, 0)
        svc_time = expected_service.get("Unknown", 0)
        expected_min = std_travel + svc_time
        actual_min = seg["actual_duration_mins"]
        delay = actual_min - expected_min if pd.notna(actual_min) else np.nan

        rows.append({
            "crew": seg["crew"],
            "date": seg["date"],
            "origin": seg["origin_location"],
            "destination": seg["destination_location"],
            "expected_min": round(expected_min, 1),
            "actual_min": round(actual_min, 1) if pd.notna(actual_min) else np.nan,
            "delay_min": round(delay, 1) if pd.notna(delay) else np.nan,
        })

    if not rows:
        return _empty_df(cols)

    return pd.DataFrame(rows, columns=cols)


def filter_trackable(df, config, sender_col="sender_resolved"):
    """Remove rows where sender is in the non-trackable senders list.

    Args:
        df: Any DataFrame containing a sender column.
        config: Configuration dict (output of load_config).
        sender_col: Name of the column to check against the exclusion list.

    Returns:
        Filtered DataFrame with non-trackable senders removed.
    """
    if df is None or df.empty:
        return df

    non_trackable = config.get("non_trackable_senders", [])
    if not non_trackable:
        return df

    return df[~df[sender_col].isin(non_trackable)].reset_index(drop=True)
