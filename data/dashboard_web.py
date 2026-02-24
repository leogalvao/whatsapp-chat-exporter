#!/usr/bin/env python3
"""Browser-based interactive WhatsApp chat dashboard using Plotly Dash.

Reads all whatsapp_*.json exports, applies noise filtering, and provides
a rich interactive dashboard at http://0.0.0.0:5000.

Usage:
    pip install dash dash-bootstrap-components
    python3 dashboard_web.py
"""

import base64
import glob
import json
import os
import re
import unicodedata
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from flask import request as flask_request, jsonify as flask_jsonify
from dash import ALL, Dash, Input, Output, State, callback_context, dash_table, dcc, html, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from invoice_parser import (
    parse_invoice_bytes,
    match_invoice_to_deployment,
    reconcile_invoice_with_chat,
)

from domain_model import (
    load_config, build_job_logs, build_crew_summary, build_route_segments,
    build_deployment_burndown, build_location_type_stats, build_traffic_analysis,
    build_delay_report, infer_location_type,
    build_location_coords, haversine_km,
    count_billable_routes, get_billable_routes_df, billable_route_key,
)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Timestamp parsing ────────────────────────────────────────────────────────

def parse_msg_datetime(msg, export_date=None):
    """Resolve a message's datetime using the best available field.

    Priority:
      1. ``dateTime`` (ISO 8601, e.g. "2026-02-10T14:40:00")
      2. ``date`` + ``timestamp`` combined
      3. ``timestamp`` + *export_date* fallback
    """
    # 1. Try dateTime field (most accurate)
    dt_str = msg.get("dateTime")
    if dt_str:
        try:
            return datetime.fromisoformat(dt_str)
        except (ValueError, TypeError):
            pass

    ts_str = (msg.get("timestamp") or "").strip()
    date_str = msg.get("date")  # e.g. "2026-02-10"

    # 2. Try date + timestamp combined
    if date_str and ts_str:
        try:
            d = date.fromisoformat(date_str)
            t = datetime.strptime(ts_str, "%I:%M %p")
            return t.replace(year=d.year, month=d.month, day=d.day)
        except (ValueError, TypeError):
            pass

    # 3. Fallback: timestamp + export_date
    if ts_str:
        try:
            t = datetime.strptime(ts_str, "%I:%M %p")
            if export_date:
                return t.replace(year=export_date.year, month=export_date.month,
                                 day=export_date.day)
            return t.replace(year=2026, month=2, day=8)
        except ValueError:
            pass

    return None


# ── Data loading & cleaning ──────────────────────────────────────────────────

# Patterns for content cleaning
RE_PHONE = re.compile(r"\+1\s*\(\d{3}\)\s*\d{3}[- ]?\d{4}")
UI_TOKENS = ["wds-ic-hd-filled", "tail-in", "forward-refreshed",
             "default-group-refreshed", "camera-v2"]
RE_TRAILING_TIME = re.compile(r"\d{1,2}:\d{2}\s*[AP]M\s*$")

# Patterns for detecting fake senders (locations / captions the exporter
# mistakenly placed in the sender field).
_RE_ADDR_SUFFIX = re.compile(
    r"\b(st|ave|rd|street|drive|dr|road)\s+(nw|ne|sw|se)\b", re.I)
_RE_INTERSECTION = re.compile(
    r"\b(st|ave|av|rd|street)\b.*&|\&.*\b(st|ave|av|rd|street)\b", re.I)
_RE_LOCATION_WORD = re.compile(
    r"\b(school|elementary|housing|annex|camp|education|church|park)\b", re.I)
_RE_PHONE_SENDER = re.compile(r"^[@+]")
_RE_PHOTO_COUNT = re.compile(r"^\d+\s+photos?$", re.I)


def _is_fake_sender(name):
    """Return True if *name* looks like a location, address, or UI artifact
    rather than a real person's name."""
    if not name:
        return False
    if _RE_ADDR_SUFFIX.search(name):
        return True
    if _RE_INTERSECTION.search(name):
        return True
    if _RE_LOCATION_WORD.search(name):
        return True
    if _RE_PHONE_SENDER.match(name):
        return True
    if _RE_PHOTO_COUNT.match(name):
        return True
    return False


def clean_content(text):
    """Strip CSS blobs, phone numbers, UI tokens, trailing timestamps."""
    if not text:
        return ""
    # Remove CSS blocks
    text = re.sub(r"\.cdf40d50ba[\s\S]*", "", text)
    # Remove phone numbers
    text = RE_PHONE.sub("", text)
    # Remove UI tokens
    for tok in UI_TOKENS:
        text = text.replace(tok, "")
    # Remove trailing duplicate timestamps
    text = RE_TRAILING_TIME.sub("", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Location extraction ──────────────────────────────────────────────────────

# Street address pattern: optional number, street name, suffix + direction
_RE_STREET_ADDR = re.compile(
    r"\d{1,5}[-–]?\d{0,5}\s+[\w\s]{2,40}\b"
    r"(st|ave|avenue|rd|road|drive|dr|street|blvd|boulevard|pl|place|ct|court"
    r"|way|ln|lane)\s+(nw|ne|sw|se)\b",
    re.I,
)

# Named-place pattern: content contains a known place keyword
_RE_NAMED_PLACE = re.compile(
    r"\b(school|elementary|recreation\s+center|community\s+center"
    r"|church|housing|annex|camp|park|center|library|plaza)\b",
    re.I,
)

# Things that look like locations but aren't
_RE_SUPPLY_COUNT = re.compile(r"^\d+\s+(bags?|boxes?|loads?|pallets?|tons?)$", re.I)
_RE_IMAGE_TAG = re.compile(r"^\[Image\]$", re.I)


def _normalize_location(loc):
    """Lowercase, collapse whitespace, normalize street abbreviations."""
    loc = loc.lower().strip()
    loc = re.sub(r"\s+", " ", loc)
    # Normalize common abbreviations
    loc = re.sub(r"\bavenue\b", "ave", loc)
    loc = re.sub(r"\bstreet\b", "st", loc)
    loc = re.sub(r"\broad\b", "rd", loc)
    loc = re.sub(r"\bdrive\b", "dr", loc)
    loc = re.sub(r"\bboulevard\b", "blvd", loc)
    loc = re.sub(r"\bplace\b", "pl", loc)
    loc = re.sub(r"\bcourt\b", "ct", loc)
    loc = re.sub(r"\blane\b", "ln", loc)
    return loc


_ABBREV_MAP = {
    "street": "st", "st": "street",
    "avenue": "ave", "ave": "avenue",
    "drive": "dr", "dr": "drive",
    "road": "rd", "rd": "road",
    "boulevard": "blvd", "blvd": "boulevard",
    "place": "pl", "pl": "place",
    "court": "ct", "ct": "court",
    "lane": "ln", "ln": "lane",
}


def _fuzzy_match_location(name, known_locations):
    if not name or not known_locations:
        return ("", 0)
    norm = _normalize_location(name)
    for loc in known_locations:
        if norm == _normalize_location(loc):
            return (loc, 100)
    for loc in known_locations:
        nloc = _normalize_location(loc)
        if norm in nloc or nloc in norm:
            return (loc, 80)
    norm_words = norm.split()
    expanded_variants = set()
    expanded_variants.add(norm)
    for i, w in enumerate(norm_words):
        if w in _ABBREV_MAP:
            variant = norm_words[:i] + [_ABBREV_MAP[w]] + norm_words[i+1:]
            expanded_variants.add(" ".join(variant))
    for loc in known_locations:
        nloc = _normalize_location(loc)
        for variant in expanded_variants:
            if variant in nloc or nloc in variant:
                return (loc, 60)
    return ("", 0)


def _extract_crew_from_chat(chat_name):
    """Extract crew name and type (sidewalk/parking) from chat group name."""
    name = chat_name.strip()
    is_sidewalk = bool(re.search(r'sidewalk', name, re.IGNORECASE))
    is_parking = bool(re.search(r'parking', name, re.IGNORECASE))
    crew = re.sub(r'\s*(sidewalk|parking\s*lot|parking)s?\s*$', '', name, flags=re.IGNORECASE).strip()
    crew = re.sub(r'\s*QAQC\s*$', '', crew, flags=re.IGNORECASE).strip()
    return crew, is_sidewalk, is_parking


def _build_location_crew_map(df):
    """Build mapping: normalized_location -> {sidewalk_crews: set, parking_crews: set}."""
    loc_crew = {}
    if df.empty:
        return loc_crew
    loc_df = df[(df["location"].str.len() > 0) & (df["noise_type"] == "clean")]
    for _, row in loc_df.iterrows():
        loc = row["location"]
        chat = row["chat"]
        crew, is_sw, is_pl = _extract_crew_from_chat(chat)
        if not crew:
            continue
        norm_loc = _normalize_location(loc)
        if norm_loc not in loc_crew:
            loc_crew[norm_loc] = {"sidewalk": set(), "parking": set()}
        if is_sw:
            loc_crew[norm_loc]["sidewalk"].add(crew)
        elif is_pl:
            loc_crew[norm_loc]["parking"].add(crew)
    return loc_crew


def _find_crews_for_location(search_terms, loc_crew_map):
    """Try multiple search terms to find crew assignments from chat data."""
    for term in search_terms:
        if not term:
            continue
        norm = _normalize_location(term)
        crews = loc_crew_map.get(norm)
        if crews and (crews["sidewalk"] or crews["parking"]):
            return crews
        for nloc, c in loc_crew_map.items():
            if norm in nloc or nloc in norm:
                if c["sidewalk"] or c["parking"]:
                    return c
    return None


def _auto_assign_crews(registry, df):
    """Auto-assign crews to registry locations based on chat location reports."""
    loc_crew_map = _build_location_crew_map(df)
    if not loc_crew_map:
        return registry, 0

    assigned = 0
    for entry in registry:
        matched_raw = entry.get("_matched_raw", "")
        if not matched_raw:
            mcl = entry.get("matched_chat_location", "")
            if mcl and "(" in mcl:
                matched_raw = mcl.split("(")[0].strip()

        search_terms = [matched_raw, entry.get("address", ""), entry.get("location_name", "")]
        crews = _find_crews_for_location(search_terms, loc_crew_map)
        if not crews:
            continue

        changed = False
        if crews["sidewalk"] and not entry.get("crew_sidewalk", ""):
            entry["crew_sidewalk"] = ", ".join(sorted(crews["sidewalk"]))
            changed = True
        if crews["parking"] and not entry.get("crew_parking_lot", ""):
            entry["crew_parking_lot"] = ", ".join(sorted(crews["parking"]))
            changed = True
        if changed:
            assigned += 1

    return registry, assigned


def extract_location(content):
    """Extract a location from message content, or return empty string.

    Two-tier approach:
      1. Street addresses like "310-0324 Kennedy St NW"
      2. Named places containing keywords like "school", "recreation center"

    Excludes [Image] tags, supply counts ("5 bags"), and very short/long strings.
    """
    if not content:
        return ""
    # Skip non-location content
    if _RE_IMAGE_TAG.match(content):
        return ""
    if _RE_SUPPLY_COUNT.match(content):
        return ""
    # Too short or too long to be a useful location
    if len(content) < 5 or len(content) > 200:
        return ""

    # Tier 1: street address
    m = _RE_STREET_ADDR.search(content)
    if m:
        return _normalize_location(m.group(0))

    # Tier 2: named place — use the full content as the location
    if _RE_NAMED_PLACE.search(content):
        # Trim to first line / sentence for cleanliness
        loc = content.split("\n")[0].strip()
        if len(loc) > 100:
            loc = loc[:100]
        return _normalize_location(loc)

    return ""


def classify_noise(msg):
    """Classify a message into a noise type.

    Accepts the full message dict so v1.5+ fields (dateTime, date,
    isDeleted, isForwarded) can be used for filtering.
    """
    if msg.get("isDeleted"):
        return "deleted"

    sender = msg.get("sender", "") or ""
    ts_str = msg.get("timestamp", "") or ""
    content = msg.get("content", "") or ""

    if not sender and not ts_str:
        return "system_metadata"
    if len(content) > 500 and ".cdf40d50ba" in content:
        return "css_html"
    if "This message couldn\u0027t load" in content or \
       "This message couldn\u2019t load" in content:
        return "load_error"
    if ts_str and not sender:
        if msg.get("dateTime") or msg.get("date"):
            return "clean"
        return "empty_sender_caption"
    return "clean"


def parse_export_date(iso_str):
    """Extract date from ISO 8601 exportDate string."""
    try:
        return datetime.fromisoformat(iso_str.replace("Z", "+00:00")).date()
    except (ValueError, AttributeError):
        return date(2026, 2, 8)


EXCLUDE_DIRS = {".claude", "__pycache__", "node_modules", "config"}


def _normalize_chat_name(name):
    """Normalize a chat name for de-duplication.

    Folds accents (ñ→n, é→e) and lowercases so that variants like
    "Julian Ordoñez Sidewalks" / "Julian Ordonez sidewalks" merge.
    """
    nfkd = unicodedata.normalize("NFKD", name)
    stripped = "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn")
    return stripped.lower().strip()


def _classify_json_file(path):
    """Classify a JSON file by its schema.

    Returns one of:
      - 'combined'  : Combined_Messages.json (unified multi-chat export)
      - 'legacy'    : Per-chat WhatsApp export (exportInfo + messages)
      - 'metrics'   : metrics_report.json (pre-computed metrics)
      - 'dispatches': OCR dispatch data
      - 'unknown'   : Unrecognised format
    """
    basename = os.path.basename(path)
    if basename.endswith("_dispatches.json"):
        return "dispatches"
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return "unknown"

    if not isinstance(data, dict):
        return "unknown"

    if "summary" in data and "crew_metrics" in data:
        return "metrics"

    if "exportInfo" not in data or "messages" not in data:
        return "unknown"

    msgs = data.get("messages", [])
    if msgs and isinstance(msgs[0], dict) and "chatName" in msgs[0]:
        return "combined"

    return "legacy"


def _discover_json_files(data_dir):
    """Find all JSON files under *data_dir* and classify them.

    Returns (chat_paths, metrics_path) where chat_paths is a list of
    legacy or combined chat export files, and metrics_path is the path
    to a metrics_report.json if found (else None).
    """
    chat_paths = []
    metrics_path = None

    all_json = []
    for p in glob.glob(os.path.join(data_dir, "archive", "*.json")):
        all_json.append(p)
    for root, dirs, files in os.walk(data_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and d != "archive"]
        for fn in files:
            if fn.endswith(".json"):
                all_json.append(os.path.join(root, fn))

    for p in sorted(set(all_json)):
        kind = _classify_json_file(p)
        if kind in ("legacy", "combined"):
            chat_paths.append(p)
        elif kind == "metrics":
            metrics_path = p

    return chat_paths, metrics_path


def _process_message(m, chat_name, export_date, prev_sender_ref):
    """Process a single message dict into a row dict for the DataFrame."""
    msg_id = m.get("id", "")
    sender = m.get("sender", "") or ""
    ts_str = m.get("timestamp", "") or ""
    content_raw = m.get("content", "") or ""
    msg_type = m.get("type", "text") or "text"
    is_outgoing = bool(m.get("isOutgoing", False))
    is_forwarded = bool(m.get("isForwarded", False))
    is_deleted = bool(m.get("isDeleted", False))
    ts = parse_msg_datetime(m, export_date)

    media_obj = m.get("media")
    media_type = media_obj.get("type", "") if isinstance(media_obj, dict) else ""
    media_src = media_obj.get("exportedFilename", "") if isinstance(media_obj, dict) else ""
    has_media = media_obj is not None

    if _is_fake_sender(sender):
        if not content_raw or content_raw == "[Image]":
            content_raw = sender
        sender = ""

    noise = classify_noise(m)
    content_clean = clean_content(content_raw)

    sender_resolved = sender if sender else prev_sender_ref[0]
    if sender:
        prev_sender_ref[0] = sender

    msg_date = ts.date() if ts else (export_date if export_date else None)

    return {
        "chat": chat_name,
        "sender": sender,
        "sender_resolved": sender_resolved,
        "timestamp": ts_str,
        "time": ts,
        "hour": ts.hour + ts.minute / 60.0 if ts else None,
        "hour_int": ts.hour if ts else None,
        "msg_date": msg_date,
        "type": msg_type,
        "content_raw": content_raw,
        "content": content_clean,
        "content_len": len(content_clean),
        "location": extract_location(content_clean),
        "noise_type": noise,
        "export_date": export_date,
        "msg_id": msg_id,
        "is_outgoing": is_outgoing,
        "is_forwarded": is_forwarded,
        "is_deleted": is_deleted,
        "has_media": has_media,
        "media_type": media_type,
        "media_file": media_src,
    }


def _load_combined_file(path):
    """Load a Combined_Messages.json (unified multi-chat export).

    Each message has a ``chatName`` field identifying which crew/chat it
    belongs to. Messages are de-duplicated by ``id``.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    export_date = parse_export_date(data.get("exportInfo", {}).get("exportDate", ""))
    msgs = data.get("messages", [])

    rows = []
    seen_ids = set()
    prev_by_chat = {}
    for m in msgs:
        msg_id = m.get("id", "")
        if msg_id:
            if msg_id in seen_ids:
                continue
            seen_ids.add(msg_id)
        chat_name = m.get("chatName", data.get("exportInfo", {}).get("chatName", "Unknown"))
        if chat_name not in prev_by_chat:
            prev_by_chat[chat_name] = [""]
        row = _process_message(m, chat_name, export_date, prev_by_chat[chat_name])
        rows.append(row)
    return rows


def _load_legacy_files(paths):
    """Load legacy per-chat WhatsApp JSON exports.

    De-duplicates by normalized chatName (accent-folded, lowercased),
    keeping the file with the most messages.
    """
    chat_files = {}
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if "exportInfo" not in data or "messages" not in data:
                continue
        except (json.JSONDecodeError, KeyError):
            continue
        name = data["exportInfo"]["chatName"]
        key = _normalize_chat_name(name)
        msgs = data["messages"]
        export_date = parse_export_date(data["exportInfo"].get("exportDate", ""))
        if key not in chat_files or len(msgs) > len(chat_files[key][1]):
            chat_files[key] = (path, msgs, export_date, name)

    rows = []
    for _key, (path, msgs, export_date, chat_name) in chat_files.items():
        prev_sender_ref = [""]
        seen_ids = set()
        for m in msgs:
            msg_id = m.get("id", "")
            if msg_id:
                if msg_id in seen_ids:
                    continue
                seen_ids.add(msg_id)
            row = _process_message(m, chat_name, export_date, prev_sender_ref)
            rows.append(row)
    return rows


def load_all_data(data_dir):
    """Load all JSON chat files, build DataFrame.

    Supports three formats:
      1. Combined_Messages.json — unified multi-chat export with per-message chatName
      2. Legacy per-chat exports — individual files with exportInfo.chatName
      3. v2 CrewChatData format — legacy with id, isOutgoing, isForwarded, isDeleted fields

    Chat names are normalized (case-insensitive, accent-folded) for de-duplication.
    Messages are de-duplicated by their ``id`` field to prevent inflation from re-exports.
    """
    chat_paths, metrics_path = _discover_json_files(data_dir)

    combined_paths = []
    legacy_paths = []
    for p in chat_paths:
        kind = _classify_json_file(p)
        if kind == "combined":
            combined_paths.append(p)
        else:
            legacy_paths.append(p)

    rows = []
    for cp in combined_paths:
        try:
            rows.extend(_load_combined_file(cp))
        except Exception as e:
            print(f"  Error loading combined file {cp}: {e}")
    if legacy_paths:
        rows.extend(_load_legacy_files(legacy_paths))

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=[
            "chat", "sender", "sender_resolved", "timestamp", "time",
            "hour", "hour_int", "msg_date", "type", "content_raw",
            "content", "content_len", "location", "noise_type", "export_date",
            "msg_id", "is_outgoing", "is_forwarded", "is_deleted",
            "has_media", "media_type", "media_file",
        ])
    df["export_date"] = pd.to_datetime(df["export_date"])
    df["msg_date"] = pd.to_datetime(df["msg_date"])

    config_path = os.path.join(data_dir, "config", "snow_removal.json")
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        crew_merges = cfg.get("crew_merges", {})
        if crew_merges and not df.empty:
            df["chat"] = df["chat"].replace(crew_merges)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    return df


def load_metrics_report(data_dir):
    """Load metrics_report.json if present.

    Returns dict with keys: summary, crew_metrics, daily_breakdown,
    productivity_scores, site_visits.  Returns None if not found.
    """
    _, metrics_path = _discover_json_files(data_dir)
    if not metrics_path:
        return None
    try:
        with open(metrics_path, encoding="utf-8") as f:
            data = json.load(f)
        required = {"summary", "crew_metrics"}
        if not required.issubset(data.keys()):
            return None
        return data
    except (json.JSONDecodeError, OSError) as e:
        print(f"  Error loading metrics report: {e}")
        return None


# ── Build global DataFrame ───────────────────────────────────────────────────

DF_ALL = load_all_data(DATA_DIR)
METRICS_REPORT = load_metrics_report(DATA_DIR)
SNOW_CONFIG = load_config(os.path.join(DATA_DIR, "config", "snow_removal.json"))


def load_pricing(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


PRICING_DATA = load_pricing(os.path.join(DATA_DIR, "config", "pricing.json"))
PRICING_INDEX = {}
for p in PRICING_DATA:
    addr_norm = _normalize_location(p.get("address", ""))
    if addr_norm:
        PRICING_INDEX[addr_norm] = p
    name_norm = _normalize_location(p.get("building_name", ""))
    if name_norm:
        PRICING_INDEX[name_norm] = p


def _match_location_to_pricing(location, pricing_index):
    if not location:
        return None
    norm = _normalize_location(location)
    if not norm or len(norm) < 4:
        return None
    if norm in pricing_index:
        return pricing_index[norm]
    for key, entry in pricing_index.items():
        if len(key) < 6 or len(norm) < 6:
            continue
        if norm in key or key in norm:
            return entry
    return None


def _compute_financials(df, deployments_list, snow_config, pricing_index):
    finance_cfg = snow_config.get("finance_config", {})
    labor_rate_sw = finance_cfg.get("labor_rate_sidewalk", 25.0)
    labor_rate_pl = finance_cfg.get("labor_rate_parking", 30.0)
    machine_rate = finance_cfg.get("machine_hourly_rate", 75.0)
    salt_cost_lb = finance_cfg.get("salt_cost_per_lb", 0.15)
    salt_lbs_sw = finance_cfg.get("salt_lbs_per_site_sidewalk", 50.0)
    salt_lbs_pl = finance_cfg.get("salt_lbs_per_site_parking", 200.0)
    workers_sw = finance_cfg.get("workers_sidewalk", 3)
    workers_pl = finance_cfg.get("workers_parking", 2)
    overhead_pct = finance_cfg.get("overhead_pct", 10.0)
    default_tier = finance_cfg.get("default_snow_tier", "price_under_6in")
    deployment_tiers = finance_cfg.get("deployment_snow_tiers", {})

    empty_result = {
        "deployment_financials": [], "crew_financials": [],
        "total_revenue": 0, "total_costs": 0, "total_profit": 0,
        "profit_margin": 0, "sites_serviced": 0, "sites_contracted": len(PRICING_DATA),
        "avg_revenue_per_site": 0, "avg_cost_per_crew": 0,
        "revenue_per_deployment": 0, "total_salt_lbs": 0,
        "cost_per_hour": 0, "revenue_per_hour": 0, "profit_per_hour": 0,
    }
    if df.empty or not deployments_list:
        return empty_result

    df_clean = df[df["noise_type"] == "clean"].copy()
    non_trackable = set(snow_config.get("non_trackable_senders", []))
    if non_trackable:
        df_clean = df_clean[~df_clean["sender_resolved"].isin(non_trackable)]

    dep_financials = []
    crew_data = {}
    total_revenue = 0
    total_costs = 0
    total_salt = 0
    total_hours = 0
    all_serviced_locations = set()

    chat_crew_type = {}
    for chat in df_clean["chat"].unique():
        _, is_sw, is_pl = _extract_crew_from_chat(chat)
        if is_pl:
            chat_crew_type[chat] = "Parking Lot"
        elif is_sw:
            chat_crew_type[chat] = "Sidewalk"
        else:
            chat_crew_type[chat] = "Sidewalk"

    invoices_list = snow_config.get("invoices", [])
    invoice_by_dep = {}
    for inv in invoices_list:
        dep = inv.get("matched_deployment")
        if dep:
            if dep not in invoice_by_dep:
                invoice_by_dep[dep] = []
            invoice_by_dep[dep].append(inv)

    for dep in deployments_list:
        label = dep["label"]
        start = pd.Timestamp(dep["start_date"])
        end = pd.Timestamp(dep["end_date"])

        dep_invoices = invoice_by_dep.get(label, [])
        if dep_invoices:
            tier = dep_invoices[0].get("snow_tier", default_tier)
        else:
            tier = deployment_tiers.get(label, default_tier)

        dep_df = df_clean[(df_clean["msg_date"] >= start) & (df_clean["msg_date"] <= end + pd.Timedelta(days=1))]
        if dep_df.empty:
            continue

        dep_locations = dep_df[dep_df["location"].str.len() > 0]["location"].unique()

        if dep_invoices:
            dep_revenue = sum(inv.get("total_billed", 0) for inv in dep_invoices)
            matched_sites = sum(inv.get("site_count", 0) for inv in dep_invoices)
            for loc in dep_locations:
                all_serviced_locations.add(loc)
        else:
            dep_revenue = 0
            matched_sites = 0
            loc_service_pairs = set()
            for loc in dep_locations:
                loc_chats = dep_df[dep_df["location"] == loc]["chat"].unique()
                sas = set()
                for ch in loc_chats:
                    sas.add(chat_crew_type.get(ch, "Sidewalk"))
                for sa in sas:
                    loc_service_pairs.add((loc, sa))
            for loc, sa in loc_service_pairs:
                pricing = _match_location_to_pricing(loc, pricing_index)
                if pricing:
                    price = pricing.get(tier, pricing.get("price_under_6in", 0))
                    dep_revenue += price
                    matched_sites += 1
                    all_serviced_locations.add(loc)

        dep_crews = dep_df[dep_df["sender_resolved"] != ""]["sender_resolved"].unique()
        n_crews = len(dep_crews)

        dep_labor = 0
        dep_machine = 0
        dep_salt = 0
        dep_hours = 0
        sw_sites = 0
        pl_sites = 0

        billable_routes_seen = set()
        for loc in dep_locations:
            loc_chats = dep_df[dep_df["location"] == loc]["chat"].unique()
            service_areas_at_loc = set()
            for ch in loc_chats:
                ct = chat_crew_type.get(ch, "Sidewalk")
                service_areas_at_loc.add(ct)
            for sa in service_areas_at_loc:
                route_key = f"{label}|{loc}|{sa}"
                if route_key not in billable_routes_seen:
                    billable_routes_seen.add(route_key)
                    if sa == "Parking Lot":
                        pl_sites += 1
                    else:
                        sw_sites += 1

        billable_site_count = sw_sites + pl_sites
        dep_salt_lbs = (sw_sites * salt_lbs_sw) + (pl_sites * salt_lbs_pl)
        dep_salt = dep_salt_lbs * salt_cost_lb

        labor_overrides = snow_config.get("labor_overrides", {}).get(label, {})

        for crew in dep_crews:
            crew_msgs = dep_df[dep_df["sender_resolved"] == crew]
            if len(crew_msgs) < 2:
                continue
            time_range = crew_msgs["time"].dropna()
            if len(time_range) < 2:
                continue
            hrs = (time_range.max() - time_range.min()).total_seconds() / 3600
            dep_hours += hrs

            crew_chats = crew_msgs["chat"].unique()
            crew_type = "Sidewalk"
            for ch in crew_chats:
                if chat_crew_type.get(ch) == "Parking Lot":
                    crew_type = "Parking Lot"
                    break

            crew_chat_name = crew_chats[0] if len(crew_chats) > 0 else ""
            override = labor_overrides.get(crew_chat_name, {})
            ov_rate = override.get("labor_rate")
            ov_workers = override.get("workers")
            ov_hours = override.get("hours")

            actual_hrs = ov_hours if ov_hours else hrs

            if crew_type == "Parking Lot":
                w = ov_workers if ov_workers else min(workers_pl, 2)
                rate = ov_rate if ov_rate else labor_rate_pl
                crew_labor = actual_hrs * rate * w
                crew_machine = actual_hrs * machine_rate
            else:
                w = ov_workers if ov_workers else workers_sw
                rate = ov_rate if ov_rate else labor_rate_sw
                crew_labor = actual_hrs * rate * w
                crew_machine = 0

            dep_labor += crew_labor
            dep_machine += crew_machine

            if crew not in crew_data:
                crew_data[crew] = {"hours": 0, "sites": 0, "revenue": 0, "deployments": 0,
                                   "labor_cost": 0, "machine_cost": 0, "salt_cost": 0,
                                   "crew_type": crew_type, "workers": w}
            crew_data[crew]["hours"] += actual_hrs
            crew_data[crew]["deployments"] += 1
            crew_data[crew]["labor_cost"] += crew_labor
            crew_data[crew]["machine_cost"] += crew_machine

        for crew in dep_crews:
            if crew in crew_data:
                crew_data[crew]["sites"] += matched_sites / max(n_crews, 1)
                crew_data[crew]["revenue"] += dep_revenue / max(n_crews, 1)
                crew_data[crew]["salt_cost"] += dep_salt / max(n_crews, 1)

        direct_cost = dep_labor + dep_machine + dep_salt
        dep_overhead = direct_cost * overhead_pct / 100
        dep_total_cost = direct_cost + dep_overhead
        dep_profit = dep_revenue - dep_total_cost
        dep_margin = (dep_profit / dep_revenue * 100) if dep_revenue > 0 else 0

        rev_source = "Invoice" if dep_invoices else "Estimated"
        dep_type = dep_invoices[0].get("deployment_type", "Snow Removal") if dep_invoices else "Snow Removal"

        dep_financials.append({
            "deployment": label,
            "type": dep_type,
            "snow_tier": tier.replace("price_", "").replace("_", " ").title(),
            "source": rev_source,
            "sites_serviced": matched_sites,
            "total_sites": billable_site_count,
            "sw_routes": sw_sites,
            "pl_routes": pl_sites,
            "crews": n_crews,
            "hours": round(dep_hours, 1),
            "revenue": round(dep_revenue, 2),
            "labor_cost": round(dep_labor, 2),
            "machine_cost": round(dep_machine, 2),
            "material_cost": round(dep_salt, 2),
            "overhead": round(dep_overhead, 2),
            "total_cost": round(dep_total_cost, 2),
            "profit": round(dep_profit, 2),
            "margin": round(dep_margin, 1),
            "salt_lbs": round(dep_salt_lbs, 0),
        })

        total_revenue += dep_revenue
        total_costs += dep_total_cost
        total_salt += dep_salt_lbs
        total_hours += dep_hours

    crew_financials = []
    for crew, data in sorted(crew_data.items(), key=lambda x: x[1]["revenue"], reverse=True):
        crew_total_cost = data["labor_cost"] + data["machine_cost"] + data["salt_cost"]
        crew_profit = data["revenue"] - crew_total_cost
        crew_financials.append({
            "crew": crew,
            "type": data["crew_type"],
            "workers": data["workers"],
            "deployments": data["deployments"],
            "sites": round(data["sites"], 1),
            "hours": round(data["hours"], 1),
            "revenue": round(data["revenue"], 2),
            "labor_cost": round(data["labor_cost"], 2),
            "machine_cost": round(data["machine_cost"], 2),
            "total_cost": round(crew_total_cost, 2),
            "profit": round(crew_profit, 2),
            "rate_per_hour": round(data["revenue"] / max(data["hours"], 0.1), 2),
        })

    total_profit = total_revenue - total_costs
    n_deps = len([d for d in dep_financials])
    n_crews_total = len(crew_data)

    return {
        "deployment_financials": dep_financials,
        "crew_financials": crew_financials,
        "total_revenue": round(total_revenue, 2),
        "total_costs": round(total_costs, 2),
        "total_profit": round(total_profit, 2),
        "profit_margin": round((total_profit / total_revenue * 100) if total_revenue > 0 else 0, 1),
        "sites_serviced": len(all_serviced_locations),
        "sites_contracted": len(PRICING_DATA),
        "avg_revenue_per_site": round(total_revenue / max(len(all_serviced_locations), 1), 2),
        "avg_cost_per_crew": round(total_costs / max(n_crews_total, 1), 2),
        "revenue_per_deployment": round(total_revenue / max(n_deps, 1), 2),
        "total_salt_lbs": round(total_salt, 0),
        "cost_per_hour": round(total_costs / max(total_hours, 0.1), 2),
        "revenue_per_hour": round(total_revenue / max(total_hours, 0.1), 2),
        "profit_per_hour": round(total_profit / max(total_hours, 0.1), 2),
    }


# Precompute lists for filters
ALL_CHATS = sorted(DF_ALL["chat"].unique()) if not DF_ALL.empty else []
_non_trackable_init = set(SNOW_CONFIG.get("non_trackable_senders", []))
ALL_SENDERS = sorted(s for s in DF_ALL[DF_ALL["sender"] != ""]["sender"].unique()
                     if s not in _non_trackable_init) if not DF_ALL.empty else []
ALL_SENDERS_RESOLVED = sorted(s for s in DF_ALL[DF_ALL["sender_resolved"] != ""]["sender_resolved"].unique()
                              if s not in _non_trackable_init) if not DF_ALL.empty else []
ALL_SENDERS_RESOLVED_FULL = sorted(
    DF_ALL[DF_ALL["sender_resolved"] != ""]["sender_resolved"].unique()) if not DF_ALL.empty else []
ALL_TYPES = sorted(DF_ALL["type"].unique()) if not DF_ALL.empty else []
NOISE_TYPES = sorted(DF_ALL["noise_type"].unique()) if not DF_ALL.empty else []

# Date range (prefer msg_date when available, fall back to export_date)
_date_col = DF_ALL["msg_date"].dropna()
if _date_col.empty:
    _date_col = DF_ALL["export_date"].dropna()
DATE_MIN = _date_col.min().date() if not _date_col.empty else date(2026, 2, 8)
DATE_MAX = _date_col.max().date() if not _date_col.empty else date(2026, 2, 8)
ALL_DATES = sorted(DF_ALL["msg_date"].dropna().dt.date.unique()) if not DF_ALL.empty else []


# ── Deployment computation ────────────────────────────────────────────────────

def _compute_deployments(dates, gap_hours=24):
    """Group sorted dates into deployments separated by gaps > gap_hours.

    Returns list of dicts: {id, label, start_date, end_date, days}.
    """
    if not dates:
        return []
    deployments = []
    current_start = dates[0]
    current_end = dates[0]
    for d in dates[1:]:
        gap_days = (d - current_end).days
        if gap_days <= (gap_hours / 24):
            current_end = d
        else:
            deployments.append({
                "start_date": current_start,
                "end_date": current_end,
            })
            current_start = d
            current_end = d
    deployments.append({
        "start_date": current_start,
        "end_date": current_end,
    })
    for i, dep in enumerate(deployments, 1):
        dep["id"] = i
        s = dep["start_date"]
        e = dep["end_date"]
        dep["days"] = (e - s).days + 1
        if s.month == e.month:
            dep["label"] = f"{s.strftime('%b %d')}\u2013{e.strftime('%d')}"
        else:
            dep["label"] = f"{s.strftime('%b %d')}\u2013{e.strftime('%b %d')}"
    return deployments


ALL_DEPLOYMENTS_LIST = _compute_deployments(ALL_DATES)

_date_to_deployment = {}
for _dep in ALL_DEPLOYMENTS_LIST:
    _d = _dep["start_date"]
    while _d <= _dep["end_date"]:
        _date_to_deployment[_d] = _dep["label"]
        _d += timedelta(days=1)

DF_ALL["deployment"] = DF_ALL["msg_date"].dt.date.map(_date_to_deployment)
ALL_DEPLOYMENTS = [dep["label"] for dep in ALL_DEPLOYMENTS_LIST]

# Stats
TOTAL_MSGS = len(DF_ALL)
CLEAN_MSGS = len(DF_ALL[DF_ALL["noise_type"] == "clean"])
NOISE_MSGS = TOTAL_MSGS - CLEAN_MSGS
NUM_CHATS = DF_ALL["chat"].nunique()
NUM_SENDERS = len(ALL_SENDERS_RESOLVED)


# ── Helper: filtered DataFrame ───────────────────────────────────────────────

def get_filtered_df(chats, senders, noise_types, msg_types,
                    time_range, use_resolved, date_start=None, date_end=None,
                    deployments=None):
    """Apply all sidebar filters to DF_ALL and return filtered copy."""
    df = DF_ALL.copy()

    non_trackable = SNOW_CONFIG.get("non_trackable_senders", [])
    if non_trackable:
        df = df[~df["sender_resolved"].isin(non_trackable)]

    # Date range filter (use msg_date for accurate per-message filtering)
    if date_start:
        df = df[df["msg_date"] >= pd.Timestamp(date_start)]
    if date_end:
        df = df[df["msg_date"] <= pd.Timestamp(date_end)]

    # Deployment filter
    if deployments:
        df = df[df["deployment"].isin(deployments)]

    # Chat filter
    if chats:
        df = df[df["chat"].isin(chats)]

    # Sender filter
    sender_col = "sender_resolved" if use_resolved else "sender"
    if senders:
        df = df[df[sender_col].isin(senders)]

    # Noise type filter
    if noise_types:
        df = df[df["noise_type"].isin(noise_types)]

    # Message type filter
    if msg_types:
        df = df[df["type"].isin(msg_types)]

    # Time range filter (hour slider)
    if time_range and len(time_range) == 2:
        lo, hi = time_range
        df = df[(df["hour"].notna()) & (df["hour"] >= lo) & (df["hour"] < hi)]

    return df


# ── Dash app ─────────────────────────────────────────────────────────────────

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css",
    ],
    suppress_callback_exceptions=True,
)
app.title = "WhatsApp Chat Dashboard"
server = app.server
server.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


@server.route("/api/export-report", methods=["GET"])
def api_export_report():
    try:
        return _do_export_report()
    except Exception as e:
        print(f"[export] Error: {e}")
        return flask_jsonify({"error": str(e)}), 500


def _do_export_report():
    fmt = flask_request.args.get("format", "json")
    df = DF_ALL.copy()
    if df.empty:
        return flask_jsonify({"error": "No data loaded"}), 400

    non_trackable = set(SNOW_CONFIG.get("non_trackable_senders", []))
    if non_trackable and "sender_resolved" in df.columns:
        df = df[~df["sender_resolved"].isin(non_trackable)]

    scol = "sender_resolved"
    ds = _build_daily_summary(df, scol)
    visits_df = _build_site_visits(df, scol)
    trans_df = _build_transitions(visits_df) if not visits_df.empty else pd.DataFrame()
    sc = _build_crew_scorecard(visits_df, ds) if not visits_df.empty and not ds.empty else pd.DataFrame()
    prod_df = _build_daily_productivity_score(df, scol)

    total_msgs = len(df)
    clean_msgs = len(df[df["noise_type"] == "clean"])
    active_crews = df["chat"].nunique()
    total_sites = _count_billable_routes_from_df(df)
    avg_sites_hr = round(sc["avg_sites_per_hour"].mean(), 2) if not sc.empty else 0
    avg_trans = round(sc["avg_transition_min"].mean(), 1) if not sc.empty else 0
    avg_first_str = "N/A"
    if not ds.empty:
        avg_first = ds["first_hour"].mean()
        fh = int(avg_first)
        fm = int((avg_first - fh) * 60)
        period = "AM" if fh < 12 else "PM"
        dh = fh % 12 or 12
        avg_first_str = f"{dh}:{fm:02d} {period}"

    crew_metrics = []
    if not sc.empty:
        for _, r in sc.iterrows():
            crew_metrics.append({
                "crew": r["sender"],
                "days_active": int(r["days_active"]),
                "total_sites": int(r["total_sites"]),
                "avg_sites_per_day": r["avg_sites_per_day"],
                "avg_sites_per_hour": r["avg_sites_per_hour"],
                "avg_transition_min": r["avg_transition_min"],
                "total_active_hrs": r["total_active_hrs"],
            })

    daily_breakdown = []
    if not ds.empty:
        for _, r in ds.iterrows():
            daily_breakdown.append({
                "crew": r["sender"],
                "date": str(r["date"]),
                "first_report": r["first_time"].strftime("%I:%M %p") if pd.notna(r["first_time"]) else "N/A",
                "last_report": r["last_time"].strftime("%I:%M %p") if pd.notna(r["last_time"]) else "N/A",
                "window_hrs": round(r["window_hrs"], 1),
                "messages": int(r["msg_count"]),
                "avg_gap_min": round(r["avg_gap_min"], 1),
            })

    productivity_scores = []
    if not prod_df.empty:
        for _, r in prod_df.iterrows():
            d = r["date"]
            date_str = str(d.date()) if hasattr(d, "date") and callable(d.date) else str(d)
            productivity_scores.append({
                "crew": r["sender"],
                "date": date_str,
                "productivity_score": float(r["productivity_score"]),
                "pace_score": round(float(r["pace_score"]), 1),
                "punctuality_score": round(float(r["punctuality_score"]), 1),
                "coverage_score": round(float(r.get("coverage", 0)), 1),
                "sites_visited": int(r["sites_visited"]),
                "sites_per_hour": round(float(r["sites_per_hour"]), 2),
            })

    site_details = []
    if not visits_df.empty:
        for _, r in visits_df.iterrows():
            d = r["date"]
            date_str = str(d.date()) if hasattr(d, "date") and callable(d.date) else str(d)
            site_details.append({
                "crew": r["sender"],
                "date": date_str,
                "location": r["location"],
                "start_time": r["start_time"].strftime("%I:%M %p") if pd.notna(r["start_time"]) else "",
                "end_time": r["end_time"].strftime("%I:%M %p") if pd.notna(r["end_time"]) else "",
                "duration_min": round(r["duration_min"], 1),
                "messages": int(r["msg_count"]),
            })

    transitions = []
    if not trans_df.empty:
        for _, r in trans_df.iterrows():
            d = r["date"]
            date_str = str(d.date()) if hasattr(d, "date") and callable(d.date) else str(d)
            transitions.append({
                "crew": r["sender"],
                "date": date_str,
                "from": r["from_location"],
                "to": r["to_location"],
                "transition_min": round(r["transition_min"], 1),
            })

    report = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_messages": total_msgs,
            "clean_messages": clean_msgs,
            "noise_messages": total_msgs - clean_msgs,
            "active_crews": active_crews,
            "total_sites_visited": total_sites,
            "avg_sites_per_hour": avg_sites_hr,
            "avg_first_report": avg_first_str,
            "avg_transition_min": avg_trans,
        },
        "crew_metrics": crew_metrics,
        "daily_breakdown": daily_breakdown,
        "productivity_scores": productivity_scores,
        "site_visits": site_details,
        "transitions": transitions,
    }

    job_logs = build_job_logs(df, SNOW_CONFIG)
    if not job_logs.empty and job_logs["is_recall"].any():
        recalls = job_logs[job_logs["is_recall"]]
        crew_recall_counts = recalls.groupby("crew").agg(
            recall_count=("is_recall", "sum"),
            total_added_min=("recall_added_time_mins", "sum"),
        ).reset_index()
        crew_recall_counts["total_added_min"] = crew_recall_counts["total_added_min"].round(1)
        report["recalls"] = crew_recall_counts.to_dict("records")
    else:
        report["recalls"] = []

    loc_stats = build_location_type_stats(job_logs)
    if not loc_stats.empty:
        report["location_type_stats"] = loc_stats.to_dict("records")
    else:
        report["location_type_stats"] = []

    if fmt == "csv":
        import io, csv
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Section: Summary"])
        for k, v in report["summary"].items():
            writer.writerow([k, v])
        writer.writerow([])

        if crew_metrics:
            writer.writerow(["Section: Crew Metrics"])
            writer.writerow(crew_metrics[0].keys())
            for row in crew_metrics:
                writer.writerow(row.values())
            writer.writerow([])

        if daily_breakdown:
            writer.writerow(["Section: Daily Breakdown"])
            writer.writerow(daily_breakdown[0].keys())
            for row in daily_breakdown:
                writer.writerow(row.values())
            writer.writerow([])

        if productivity_scores:
            writer.writerow(["Section: Productivity Scores"])
            writer.writerow(productivity_scores[0].keys())
            for row in productivity_scores:
                writer.writerow(row.values())
            writer.writerow([])

        if site_details:
            writer.writerow(["Section: Site Visits"])
            writer.writerow(site_details[0].keys())
            for row in site_details:
                writer.writerow(row.values())
            writer.writerow([])

        if transitions:
            writer.writerow(["Section: Transitions"])
            writer.writerow(transitions[0].keys())
            for row in transitions:
                writer.writerow(row.values())
            writer.writerow([])

        if report.get("recalls"):
            writer.writerow(["Section: Recalls"])
            writer.writerow(report["recalls"][0].keys())
            for row in report["recalls"]:
                writer.writerow(row.values())
            writer.writerow([])

        if report.get("location_type_stats"):
            writer.writerow(["Section: Location Type Stats"])
            writer.writerow(report["location_type_stats"][0].keys())
            for row in report["location_type_stats"]:
                writer.writerow(row.values())

        resp = server.make_response(output.getvalue())
        resp.headers["Content-Type"] = "text/csv"
        resp.headers["Content-Disposition"] = "attachment; filename=metrics_report.csv"
        return resp

    resp = server.make_response(json.dumps(report, indent=2, ensure_ascii=False))
    resp.headers["Content-Type"] = "application/json"
    resp.headers["Content-Disposition"] = "attachment; filename=metrics_report.json"
    return resp


@server.route("/api/upload-folder", methods=["POST"])
def api_upload_folder():
    archive_dir = os.path.join(DATA_DIR, "archive")
    os.makedirs(archive_dir, exist_ok=True)
    saved = []
    skipped = []
    errors = []
    files = flask_request.files.getlist("files")
    best_by_chat = {}
    for f in files:
        try:
            raw = f.read()
            data = json.loads(raw.decode("utf-8"))

            if isinstance(data, dict) and "summary" in data and "crew_metrics" in data:
                save_path = os.path.join(archive_dir, "metrics_report.json")
                with open(save_path, "w", encoding="utf-8") as out:
                    json.dump(data, out, ensure_ascii=False)
                saved.append("metrics_report.json")
                continue

            if "exportInfo" not in data or "messages" not in data:
                errors.append(f.filename)
                continue
            chat_name = data.get("exportInfo", {}).get("chatName", "").strip().lower()
            msg_count = len(data.get("messages", []))
            safe_name = os.path.basename(f.filename)
            safe_name = re.sub(r"[^\w.\-]", "_", safe_name)
            if not safe_name.endswith(".json"):
                safe_name += ".json"
            if chat_name and chat_name in best_by_chat:
                prev_count, prev_name, _ = best_by_chat[chat_name]
                if msg_count > prev_count:
                    skipped.append(prev_name)
                    best_by_chat[chat_name] = (msg_count, safe_name, data)
                else:
                    skipped.append(safe_name)
                    continue
            else:
                key = chat_name if chat_name else safe_name
                best_by_chat[key] = (msg_count, safe_name, data)
        except Exception as e:
            errors.append(str(f.filename))
    for _, (_, safe_name, data) in best_by_chat.items():
        try:
            save_path = os.path.join(archive_dir, safe_name)
            existing_count = 0
            if os.path.exists(save_path):
                try:
                    with open(save_path, "r", encoding="utf-8") as ef:
                        existing = json.load(ef)
                    existing_count = len(existing.get("messages", []))
                except Exception:
                    pass
                if len(data.get("messages", [])) <= existing_count:
                    skipped.append(safe_name)
                    continue
            with open(save_path, "w", encoding="utf-8") as out:
                json.dump(data, out, ensure_ascii=False)
            saved.append(safe_name)
        except Exception as e:
            errors.append(safe_name)
    if saved:
        _reload_global_data()
    return flask_jsonify({"saved": saved, "skipped": len(skipped), "errors": errors})


# ── Main content (tabs) ─────────────────────────────────────────────────────

main_content = dbc.Tabs(
    id="main-tabs",
    active_tab="tab-overview",
    children=[
        dbc.Tab(label="Overview", tab_id="tab-overview", children=[
            html.Div([
                html.H4("Overview", className="mb-1"),
                html.P("High-level view of message volume, types, and activity patterns across all crews.",
                       className="text-muted mb-3", style={"fontSize": "14px"}),
            ], className="mt-3 mb-2 px-1"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-msgs-per-chat"), md=6),
                dbc.Col(dcc.Graph(id="chart-type-donut"), md=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-heatmap"), md=6),
                dbc.Col(dcc.Graph(id="chart-sender-chat"), md=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-timeline"), md=12),
            ]),
        ]),
        dbc.Tab(label="Productivity", tab_id="tab-productivity", children=[
            html.Div([
                html.H4("Productivity", className="mb-1"),
                html.P("Track crew efficiency with daily scores, punctuality, pace, and idle time analysis.",
                       className="text-muted mb-3", style={"fontSize": "14px"}),
            ], className="mt-3 mb-2 px-1"),
            dbc.Row([
                dbc.Col(html.Div(id="efficiency-report-card"), md=12),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-first-report"), md=12),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-report-window-box"), md=6),
                dbc.Col(dcc.Graph(id="chart-daily-count-trend"), md=6),
            ]),
            dbc.Row([
                dbc.Col(html.Div(id="productivity-score-container"), md=12),
            ], className="mt-3"),
            dbc.Row([
                dbc.Col(html.Div(id="crew-leaderboard-container"), md=12),
            ], className="mt-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-idle-gaps"), md=12),
            ]),
        ]),
        dbc.Tab(label="Crew Analysis", tab_id="tab-crew", children=[
            html.Div([
                html.H4("Crew Analysis", className="mb-1"),
                html.P("Deep dive into individual crew performance, site visits, routes, and transition times.",
                       className="text-muted mb-3", style={"fontSize": "14px"}),
            ], className="mt-3 mb-2 px-1"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-msgs-per-sender"), md=6),
                dbc.Col(dcc.Graph(id="chart-gap-box"), md=6),
            ]),
            dbc.Row([
                dbc.Col(html.Div(id="crew-scorecard-container"), md=12),
            ], className="mt-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-sites-per-hour"), md=6),
                dbc.Col(dcc.Graph(id="chart-transition-time"), md=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-route-timeline"), md=12),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-pace-consistency"), md=6),
                dbc.Col(dcc.Graph(id="chart-top-locations"), md=6),
            ]),
        ]),
        dbc.Tab(label="Deployments", tab_id="tab-deployments", children=[
            html.Div([
                html.H4("Deployments", className="mb-1"),
                html.P("Compare crew performance across deployment periods with timeline and trend analysis.",
                       className="text-muted mb-3", style={"fontSize": "14px"}),
            ], className="mt-3 mb-2 px-1"),
            dbc.Row([
                dbc.Col(
                    dbc.Button("Download Report", id="btn-download-deployment-pdf",
                               color="primary", size="sm"),
                    width="auto",
                ),
                dcc.Download(id="download-deployment-pdf"),
            ], className="mb-2 ms-1"),
            dbc.Row([
                dbc.Col(html.Div(id="deployment-summary-container"), md=12),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-deployment-timeline"), md=12),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-deployment-first-report"), md=6),
                dbc.Col(dcc.Graph(id="chart-deployment-transition-box"), md=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-deployment-crew-trend"), md=12),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-deployment-crew-comparison"), md=6),
                dbc.Col(dcc.Graph(id="chart-deployment-sites-heatmap"), md=6),
            ]),
            html.Hr(className="my-3"),
            html.H5("Deployment Breakdown", className="mb-2"),
            html.P("Select a deployment to see which locations were serviced and by which crew. Reassign crews using the dropdowns.",
                   className="text-muted mb-2", style={"fontSize": "13px"}),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Deployment", className="mb-1"),
                    dcc.Dropdown(
                        id="dep-breakdown-select",
                        options=[{"label": d, "value": d} for d in ALL_DEPLOYMENTS],
                        value=ALL_DEPLOYMENTS[-1] if ALL_DEPLOYMENTS else None,
                        clearable=False,
                    ),
                ], md=4),
                dbc.Col([
                    dbc.Button("Save Crew Assignments", id="dep-save-crews-btn", 
                               color="success", size="sm", className="mt-4", n_clicks=0),
                    html.Span(id="dep-save-crews-status", className="ms-2 align-middle"),
                ], md=4),
            ], className="mb-3"),
            html.Div(id="dep-breakdown-table-container"),
        ]),
        dbc.Tab(label="Operations", tab_id="tab-operations", children=[
            html.Div([
                html.H4("Operations", className="mb-1"),
                html.P("Snow removal operations analysis: routing, burndown, location performance, and delay tracking.",
                       className="text-muted mb-3", style={"fontSize": "14px"}),
            ], className="mt-3 mb-2 px-1"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-routing-gantt"), md=12),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-burndown"), md=12),
            ]),
            dbc.Row([
                dbc.Col(html.Div(id="location-type-stats-container"), md=6),
                dbc.Col(html.Div(id="traffic-analysis-container"), md=6),
            ], className="mt-3"),
            dbc.Row([
                dbc.Col(html.Div(id="delay-report-container"), md=12),
            ], className="mt-3"),
            dbc.Row([
                dbc.Col(html.Div(id="recall-summary-container"), md=12),
            ], className="mt-3"),
        ]),
        dbc.Tab(label="Map", tab_id="tab-map", children=[
            html.Div([
                html.H4("DC Service Map", className="mb-1"),
                html.P("Geographic view of service locations. Filter by deployment to see which sites were active.",
                       className="text-muted mb-3", style={"fontSize": "14px"}),
            ], className="mt-3 mb-2 px-1"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Filter by Deployment", className="mb-1"),
                    dcc.Dropdown(
                        id="map-deployment-filter",
                        options=[{"label": "All Deployments", "value": "ALL"}] + 
                                [{"label": d, "value": d} for d in ALL_DEPLOYMENTS],
                        value="ALL",
                        clearable=False,
                    ),
                ], md=4),
            ], className="mb-2"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-service-map", style={"height": "700px"}), md=12),
            ]),
            dcc.Store(id="map-data-store"),
        ]),
        dbc.Tab(label="Finances", tab_id="tab-finances", children=[
            html.Div([
                html.H4("Financial Overview", className="mb-1"),
                html.P("Revenue forecasting and cost analysis with crew-type-aware costing (labor, machines, salt, overhead).",
                       className="text-muted mb-3", style={"fontSize": "14px"}),
            ], className="mt-3 mb-2 px-1"),
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardBody([
                    html.P("Total Revenue", className="text-muted mb-1", style={"fontSize": "11px"}),
                    html.H5(id="fin-kpi-revenue", className="mb-0", style={"color": "#28a745"}),
                ])], className="shadow-sm"), md=3, lg=True),
                dbc.Col(dbc.Card([dbc.CardBody([
                    html.P("Total Costs", className="text-muted mb-1", style={"fontSize": "11px"}),
                    html.H5(id="fin-kpi-costs", className="mb-0", style={"color": "#dc3545"}),
                ])], className="shadow-sm"), md=3, lg=True),
                dbc.Col(dbc.Card([dbc.CardBody([
                    html.P("Profit", className="text-muted mb-1", style={"fontSize": "11px"}),
                    html.H5(id="fin-kpi-profit", className="mb-0"),
                ])], className="shadow-sm"), md=3, lg=True),
                dbc.Col(dbc.Card([dbc.CardBody([
                    html.P("Margin %", className="text-muted mb-1", style={"fontSize": "11px"}),
                    html.H5(id="fin-kpi-margin", className="mb-0"),
                ])], className="shadow-sm"), md=3, lg=True),
                dbc.Col(dbc.Card([dbc.CardBody([
                    html.P("Sites Matched", className="text-muted mb-1", style={"fontSize": "11px"}),
                    html.H5(id="fin-kpi-sites", className="mb-0"),
                ])], className="shadow-sm"), md=3, lg=True),
                dbc.Col(dbc.Card([dbc.CardBody([
                    html.P("Salt Used (lbs)", className="text-muted mb-1", style={"fontSize": "11px"}),
                    html.H5(id="fin-kpi-salt", className="mb-0", style={"color": "#6f42c1"}),
                ])], className="shadow-sm"), md=3, lg=True),
                dbc.Col(dbc.Card([dbc.CardBody([
                    html.P("Cost / Hour", className="text-muted mb-1", style={"fontSize": "11px"}),
                    html.H5(id="fin-kpi-cost-hr", className="mb-0", style={"color": "#e83e8c"}),
                ])], className="shadow-sm"), md=3, lg=True),
                dbc.Col(dbc.Card([dbc.CardBody([
                    html.P("Rev / Deployment", className="text-muted mb-1", style={"fontSize": "11px"}),
                    html.H5(id="fin-kpi-rev-dep", className="mb-0", style={"color": "#17a2b8"}),
                ])], className="shadow-sm"), md=3, lg=True),
            ], className="mb-3 g-2"),
            html.H5("Cost Configuration", className="mt-3 mb-2"),
            html.P("Separate rates for sidewalk and parking lot crews. Parking lot workers capped at 2 (truck-based).",
                   className="text-muted mb-2", style={"fontSize": "13px"}),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Sidewalk Labor ($/hr)", className="mb-1", style={"fontSize": "12px"}),
                    dbc.Input(id="fin-labor-sw", type="number",
                              value=SNOW_CONFIG.get("finance_config", {}).get("labor_rate_sidewalk", 25.0),
                              min=0, step=0.5, size="sm"),
                ], md=2, lg=True),
                dbc.Col([
                    dbc.Label("Parking Labor ($/hr)", className="mb-1", style={"fontSize": "12px"}),
                    dbc.Input(id="fin-labor-pl", type="number",
                              value=SNOW_CONFIG.get("finance_config", {}).get("labor_rate_parking", 30.0),
                              min=0, step=0.5, size="sm"),
                ], md=2, lg=True),
                dbc.Col([
                    dbc.Label("Machine ($/hr)", className="mb-1", style={"fontSize": "12px"}),
                    dbc.Input(id="fin-machine-rate", type="number",
                              value=SNOW_CONFIG.get("finance_config", {}).get("machine_hourly_rate", 75.0),
                              min=0, step=1, size="sm"),
                ], md=2, lg=True),
                dbc.Col([
                    dbc.Label("Salt ($/lb)", className="mb-1", style={"fontSize": "12px"}),
                    dbc.Input(id="fin-salt-cost", type="number",
                              value=SNOW_CONFIG.get("finance_config", {}).get("salt_cost_per_lb", 0.15),
                              min=0, step=0.01, size="sm"),
                ], md=2, lg=True),
                dbc.Col([
                    dbc.Label("Salt lbs/site (SW)", className="mb-1", style={"fontSize": "12px"}),
                    dbc.Input(id="fin-salt-sw", type="number",
                              value=SNOW_CONFIG.get("finance_config", {}).get("salt_lbs_per_site_sidewalk", 50.0),
                              min=0, step=5, size="sm"),
                ], md=2, lg=True),
                dbc.Col([
                    dbc.Label("Salt lbs/site (PL)", className="mb-1", style={"fontSize": "12px"}),
                    dbc.Input(id="fin-salt-pl", type="number",
                              value=SNOW_CONFIG.get("finance_config", {}).get("salt_lbs_per_site_parking", 200.0),
                              min=0, step=10, size="sm"),
                ], md=2, lg=True),
            ], className="mb-2 g-2"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Workers (Sidewalk)", className="mb-1", style={"fontSize": "12px"}),
                    dbc.Input(id="fin-workers-sw", type="number",
                              value=SNOW_CONFIG.get("finance_config", {}).get("workers_sidewalk", 3),
                              min=1, max=10, step=1, size="sm"),
                ], md=2, lg=True),
                dbc.Col([
                    dbc.Label("Workers (Parking)", className="mb-1", style={"fontSize": "12px"}),
                    dbc.Input(id="fin-workers-pl", type="number",
                              value=SNOW_CONFIG.get("finance_config", {}).get("workers_parking", 2),
                              min=1, max=2, step=1, size="sm"),
                ], md=2, lg=True),
                dbc.Col([
                    dbc.Label("Overhead %", className="mb-1", style={"fontSize": "12px"}),
                    dbc.Input(id="fin-overhead", type="number",
                              value=SNOW_CONFIG.get("finance_config", {}).get("overhead_pct", 10.0),
                              min=0, max=100, step=0.5, size="sm"),
                ], md=2, lg=True),
                dbc.Col([
                    dbc.Label("Default Snow Tier", className="mb-1", style={"fontSize": "12px"}),
                    dcc.Dropdown(
                        id="fin-default-tier",
                        options=[
                            {"label": "Melt Only", "value": "price_melt_only"},
                            {"label": "Under 6\"", "value": "price_under_6in"},
                            {"label": "6\"-12\"", "value": "price_6_to_12in"},
                            {"label": "12\"-24\"", "value": "price_12_to_24in"},
                        ],
                        value=SNOW_CONFIG.get("finance_config", {}).get("default_snow_tier", "price_under_6in"),
                        clearable=False,
                    ),
                ], md=2, lg=True),
                dbc.Col([
                    dbc.Button("Recalculate", id="fin-recalc-btn", color="primary",
                               size="sm", className="mt-4 me-1", n_clicks=0),
                    dbc.Button("Save", id="fin-save-btn", color="success",
                               size="sm", className="mt-4", n_clicks=0),
                    html.Span(id="fin-save-status", className="ms-2 align-middle"),
                ], md=2, lg=True),
            ], className="mb-3 g-2"),
            html.Hr(className="my-3"),
            html.H5("Deployment Financials", className="mb-2"),
            html.Div(id="fin-deployment-table-container"),
            html.Hr(className="my-3"),
            html.H5("Crew Financials", className="mb-2"),
            html.Div(id="fin-crew-table-container"),
            html.Hr(className="my-3"),
            html.H5("Revenue vs Costs by Deployment", className="mb-2"),
            dcc.Graph(id="fin-chart-revenue"),
            html.Hr(className="my-3"),
            html.H5("Per-Deployment Labor Costs", className="mb-2"),
            html.P("Enter actual labor costs per crew per deployment. Leave blank to use default rates above.",
                   className="text-muted mb-2", style={"fontSize": "13px"}),
            html.Div(id="fin-labor-overrides-container"),
            dbc.Button("Save Labor Overrides", id="fin-save-labor-overrides", color="success",
                       size="sm", className="mt-2 mb-3", n_clicks=0),
            html.Span(id="fin-labor-overrides-status", className="ms-2"),
            html.Hr(className="my-3"),
            html.H5("Invoice Reconciliation", className="mb-2"),
            html.P("Compare invoiced sites against chat data to identify discrepancies.",
                   className="text-muted mb-2", style={"fontSize": "13px"}),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(id="fin-recon-deployment", placeholder="Select deployment to reconcile...",
                                 style={"fontSize": "13px"}),
                ], md=4),
                dbc.Col([
                    dbc.Button("Reconcile", id="fin-recon-btn", color="primary",
                               size="sm", className="mt-0", n_clicks=0),
                ], md=2),
            ], className="mb-2"),
            html.Div(id="fin-recon-container"),
            html.Hr(className="my-3"),
            html.H5("Completion Verification", className="mb-2"),
            html.P("Cross-reference portal completion reports with chat data to validate crew claims.",
                   className="text-muted mb-2", style={"fontSize": "13px"}),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(id="fin-verify-deployment", placeholder="Select deployment to verify...",
                                 style={"fontSize": "13px"}),
                ], md=4),
                dbc.Col([
                    dbc.Button("Verify", id="fin-verify-btn", color="info",
                               size="sm", className="mt-0", n_clicks=0),
                ], md=2),
            ], className="mb-2"),
            html.Div(id="fin-verify-container"),
        ]),
        dbc.Tab(label="Data Quality", tab_id="tab-quality", children=[
            html.Div([
                html.H4("Data Quality", className="mb-1"),
                html.P("Inspect raw data, noise filtering results, and message-level details.",
                       className="text-muted mb-3", style={"fontSize": "14px"}),
            ], className="mt-3 mb-2 px-1"),
            dbc.Row([
                dbc.Col(dcc.Graph(id="chart-quality"), md=12),
            ]),
            dbc.Row([
                dbc.Col(html.Div(id="data-table-container"), md=12),
            ], className="mt-3"),
        ]),
        dbc.Tab(label="Settings", tab_id="tab-settings", children=[
            html.Div([
                html.H4("Settings", className="mb-1"),
                html.P("Configure crew location types and sender tracking. Changes are saved automatically and applied to all analytics.",
                       className="text-muted mb-3", style={"fontSize": "14px"}),
            ], className="mt-3 mb-2 px-1"),
            dbc.Row([
                dbc.Col([
                    html.H5("Crew Location Types", className="mb-2"),
                    html.P("Assign each crew/chat group to Sidewalk or Parking Lot. 'Auto-detect' infers from the chat name.",
                           className="text-muted mb-2", style={"fontSize": "13px"}),
                    html.Div(id="settings-crew-types-container"),
                ], md=7),
                dbc.Col([
                    html.H5("Non-Trackable Senders", className="mb-2"),
                    html.P("Check senders whose messages should NOT be counted in operations metrics (e.g. supervisors, admins).",
                           className="text-muted mb-2", style={"fontSize": "13px"}),
                    html.Div(id="settings-non-trackable-container"),
                ], md=5),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Button("Save Settings", id="settings-save-btn", color="primary",
                               className="mt-3 me-2", n_clicks=0),
                    html.Span(id="settings-save-status", className="ms-2 align-middle"),
                ], md=12),
            ], className="mt-2"),
            html.Hr(className="my-3"),
            dbc.Row([
                dbc.Col([
                    html.H5("Expected Deployment Duration", className="mb-2"),
                    html.P("Hours expected for a full deployment (used for burn-down baseline).",
                           className="text-muted mb-2", style={"fontSize": "13px"}),
                    dbc.InputGroup([
                        dbc.Input(id="settings-expected-hours", type="number",
                                  value=SNOW_CONFIG.get("expected_deployment_hours", 12.0),
                                  min=1, max=48, step=0.5),
                        dbc.InputGroupText("hours"),
                    ], style={"maxWidth": "250px"}),
                ], md=4),
                dbc.Col([
                    html.H5("Expected Service Times (minutes)", className="mb-2"),
                    html.P("Average minutes to complete service at each location type.",
                           className="text-muted mb-2", style={"fontSize": "13px"}),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Sidewalk", className="mb-1"),
                            dbc.Input(id="settings-svc-sidewalk", type="number",
                                      value=SNOW_CONFIG.get("expected_service_times", {}).get("Sidewalk", 30),
                                      min=1, max=240, step=1),
                        ], md=4),
                        dbc.Col([
                            dbc.Label("Parking Lot", className="mb-1"),
                            dbc.Input(id="settings-svc-parking", type="number",
                                      value=SNOW_CONFIG.get("expected_service_times", {}).get("Parking Lot", 45),
                                      min=1, max=240, step=1),
                        ], md=4),
                    ]),
                ], md=6),
            ]),
            html.Hr(className="my-3"),
            dbc.Row([
                dbc.Col([
                    html.H5("Crew Merge", className="mb-2"),
                    html.P("Merge two crews that are the same team but appear under different chat names across deployments. "
                           "Select the primary crew (name to keep) and the secondary crew (to merge into primary). "
                           "All messages from the secondary crew will be reassigned to the primary.",
                           className="text-muted mb-2", style={"fontSize": "13px"}),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Primary Crew (keep this name)", className="mb-1 fw-bold"),
                            dcc.Dropdown(id="merge-primary-crew", placeholder="Select primary crew...",
                                         style={"fontSize": "13px"}),
                        ], md=5),
                        dbc.Col([
                            dbc.Label("Secondary Crew (merge into primary)", className="mb-1 fw-bold"),
                            dcc.Dropdown(id="merge-secondary-crew", placeholder="Select crew to merge...",
                                         style={"fontSize": "13px"}),
                        ], md=5),
                        dbc.Col([
                            dbc.Button("Merge", id="merge-crews-btn", color="warning",
                                       className="mt-4", n_clicks=0),
                        ], md=2),
                    ], className="mb-2"),
                    html.Div(id="merge-status", className="mb-2"),
                    html.Div(id="merge-list-container"),
                ], md=12),
            ]),
            html.Hr(className="my-3"),
            dbc.Row([
                dbc.Col([
                    html.H5("Invoice Import", className="mb-2"),
                    html.P("Upload deployment invoice files (.xlsx or .csv). Each invoice is matched to a deployment by date. "
                           "Supports snow removal, ice removal, and pre-treatment formats.",
                           className="text-muted mb-2", style={"fontSize": "13px"}),
                    dcc.Upload(
                        id="upload-invoice",
                        children=html.Div([
                            "Drag & drop invoice .xlsx or .csv files here, or ",
                            html.A("click to browse", className="text-primary fw-bold"),
                        ], style={"lineHeight": "40px"}),
                        style={
                            "borderWidth": "2px", "borderStyle": "dashed",
                            "borderRadius": "8px", "borderColor": "#adb5bd",
                            "textAlign": "center", "padding": "8px",
                            "backgroundColor": "#fafbfc", "cursor": "pointer",
                        },
                        multiple=True,
                    ),
                    html.Div(id="invoice-upload-status", className="mt-2"),
                    html.Div(id="invoice-preview-container", className="mt-2"),
                    html.Div(id="invoice-list-container", className="mt-3"),
                ], md=12),
            ]),
            html.Hr(className="my-3"),
            dbc.Row([
                dbc.Col([
                    html.H5("Location Registry", className="mb-2"),
                    html.P("Upload a CSV of locations (Building Name, Billing Street, Crew Sidewalk, Crew Parking Lot). "
                           "Locations are fuzzy-matched to chat data automatically.",
                           className="text-muted mb-2", style={"fontSize": "13px"}),
                    dcc.Upload(
                        id="upload-locations-csv",
                        children=html.Div([
                            "Drag & drop a CSV file here, or ",
                            html.A("click to browse", className="text-primary fw-bold"),
                        ], style={"lineHeight": "40px"}),
                        style={
                            "borderWidth": "2px",
                            "borderStyle": "dashed",
                            "borderRadius": "8px",
                            "borderColor": "#adb5bd",
                            "textAlign": "center",
                            "padding": "10px 15px",
                            "cursor": "pointer",
                            "backgroundColor": "#f8f9fa",
                            "height": "60px",
                        },
                        multiple=False,
                        accept=".csv",
                    ),
                    html.Div(id="locations-upload-status", className="mt-2"),
                    html.Div(
                        dash_table.DataTable(
                            id="locations-registry-table",
                            columns=[
                                {"name": "Building Name", "id": "location_name", "editable": False},
                                {"name": "Address", "id": "address", "editable": False},
                                {"name": "Matched Chat Location", "id": "matched_chat_location", "editable": False},
                                {"name": "Crew Sidewalk", "id": "crew_sidewalk", "editable": True},
                                {"name": "Crew Parking Lot", "id": "crew_parking_lot", "editable": True},
                            ],
                            data=[
                                {k: r.get(k, "") for k in ["location_name", "address", "matched_chat_location", "crew_sidewalk", "crew_parking_lot"]}
                                for r in SNOW_CONFIG.get("location_registry", [])
                            ],
                            style_table={"overflowX": "auto"},
                            style_cell={"textAlign": "left", "padding": "5px", "fontSize": "13px"},
                            style_header={"fontWeight": "bold", "backgroundColor": "#f1f3f5"},
                            page_size=15,
                            editable=True,
                        ),
                        className="mt-3",
                    ),
                    dbc.Button("Auto-Assign Crews", id="locations-auto-assign-btn", color="info",
                               className="mt-2 me-2", n_clicks=0),
                    dbc.Button("Save Locations", id="locations-save-btn", color="success",
                               className="mt-2 me-2", n_clicks=0),
                    html.Span(id="locations-auto-assign-status", className="ms-2 align-middle"),
                    html.Span(id="locations-save-status", className="ms-2 align-middle"),
                ], md=12),
            ]),
        ]),
    ],
)


# ── Upload component ─────────────────────────────────────────────────────────

upload_section = html.Div([
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id="upload-data",
                children=html.Div([
                    "Drag & drop JSON files here, or ",
                    html.A("click to browse files", className="text-primary fw-bold"),
                ], style={"lineHeight": "40px"}),
                style={
                    "borderWidth": "2px",
                    "borderStyle": "dashed",
                    "borderRadius": "8px",
                    "borderColor": "#adb5bd",
                    "textAlign": "center",
                    "padding": "10px 15px",
                    "cursor": "pointer",
                    "backgroundColor": "#f8f9fa",
                    "height": "60px",
                },
                multiple=True,
                accept=".json",
            ),
        ], width=5),
        dbc.Col([
            html.Div(
                [
                    html.Span("or "),
                    html.A("Upload a Folder", id="folder-upload-btn",
                           className="text-primary fw-bold",
                           style={"cursor": "pointer", "textDecoration": "underline"}),
                    html.Span(" containing JSON files"),
                ],
                style={
                    "borderWidth": "2px",
                    "borderStyle": "dashed",
                    "borderRadius": "8px",
                    "borderColor": "#adb5bd",
                    "textAlign": "center",
                    "padding": "10px 15px",
                    "backgroundColor": "#f8f9fa",
                    "height": "60px",
                    "lineHeight": "40px",
                },
            ),
        ], width=5),
        dbc.Col([
            html.Div(id="upload-status", className="text-center",
                     style={"lineHeight": "60px"}),
        ], width=2),
    ], className="g-2 mb-2"),
])

_FOLDER_UPLOAD_JS = """
<script>
(function() {
    function setupFolderUpload() {
        var btn = document.getElementById('folder-upload-btn');
        if (!btn) { setTimeout(setupFolderUpload, 500); return; }
        if (btn.dataset.bound) return;
        btn.dataset.bound = '1';

        var inp = document.createElement('input');
        inp.type = 'file';
        inp.setAttribute('webkitdirectory', '');
        inp.setAttribute('directory', '');
        inp.setAttribute('multiple', '');
        inp.style.display = 'none';
        document.body.appendChild(inp);

        var totalUploaded = 0;
        var folderCount = 0;
        var uploading = false;

        var doneBtn = document.createElement('a');
        doneBtn.textContent = 'Done — Reload Dashboard';
        doneBtn.style.cssText = 'display:none;cursor:pointer;text-decoration:underline;color:#198754;font-weight:bold;margin-left:10px;';
        btn.parentNode.appendChild(document.createElement('br'));
        btn.parentNode.appendChild(doneBtn);

        doneBtn.addEventListener('click', function() { window.location.reload(); });

        btn.addEventListener('click', function() {
            if (uploading) return;
            inp.click();
        });

        inp.addEventListener('change', function() {
            if (!inp.files || inp.files.length === 0) return;
            var formData = new FormData();
            var count = 0;
            for (var i = 0; i < inp.files.length; i++) {
                if (inp.files[i].name.endsWith('.json')) {
                    formData.append('files', inp.files[i]);
                    count++;
                }
            }
            if (count === 0) {
                alert('No JSON files found in the selected folder.');
                inp.value = '';
                return;
            }
            uploading = true;
            btn.textContent = 'Uploading ' + count + ' file(s)...';
            btn.style.color = '#0d6efd';

            fetch('/api/upload-folder', { method: 'POST', body: formData })
            .then(function(r) { return r.json(); })
            .then(function(data) {
                uploading = false;
                if (data.saved && data.saved.length > 0) {
                    totalUploaded += data.saved.length;
                    folderCount++;
                    var msg = totalUploaded + ' file(s) from ' + folderCount + ' folder(s) uploaded';
                    if (data.skipped && data.skipped > 0) msg += ' (' + data.skipped + ' duplicates skipped)';
                    msg += '<br>Click to add another folder';
                    btn.innerHTML = msg;
                    btn.style.color = '#198754';
                    doneBtn.style.display = 'inline';
                } else if (data.skipped && data.skipped > 0) {
                    btn.innerHTML = data.skipped + ' duplicate(s) skipped — no new files<br>Click to add another folder';
                    btn.style.color = '#6c757d';
                    if (totalUploaded > 0) doneBtn.style.display = 'inline';
                } else {
                    btn.textContent = 'No valid exports found — click to try again';
                    btn.style.color = '#dc3545';
                }
            })
            .catch(function(e) {
                uploading = false;
                btn.textContent = 'Upload failed — click to retry';
                btn.style.color = '#dc3545';
            });
            inp.value = '';
        });
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupFolderUpload);
    } else {
        setupFolderUpload();
    }
})();
</script>
"""

app.index_string = app.index_string.replace("</body>", _FOLDER_UPLOAD_JS + "</body>")


# ── App layout (function so it refreshes after data upload) ─────────────────

def _build_sidebar():
    return dbc.Card(
        [
            dbc.CardHeader(html.H5("Filters", className="mb-0")),
            dbc.CardBody(
                [
                    dbc.Accordion(
                        [
                            dbc.AccordionItem(
                                [
                                    html.Label("Chat Groups", className="fw-bold mb-1"),
                                    dcc.Dropdown(
                                        id="filter-chats",
                                        options=[{"label": c, "value": c} for c in ALL_CHATS],
                                        value=ALL_CHATS,
                                        multi=True,
                                        placeholder="Select chats...",
                                        className="mb-3",
                                    ),
                                    html.Label("Senders", className="fw-bold mb-1"),
                                    dcc.Dropdown(
                                        id="filter-senders",
                                        options=[{"label": s, "value": s}
                                                 for s in ALL_SENDERS_RESOLVED],
                                        value=ALL_SENDERS_RESOLVED,
                                        multi=True,
                                        placeholder="Select senders...",
                                        className="mb-3",
                                    ),
                                    dbc.Switch(
                                        id="filter-resolved",
                                        label="Use resolved senders",
                                        value=True,
                                        className="mb-1",
                                    ),
                                ],
                                title="People & Chats",
                            ),
                            dbc.AccordionItem(
                                [
                                    html.Label("Time Range (hour)", className="fw-bold mb-1"),
                                    dcc.RangeSlider(
                                        id="filter-time",
                                        min=0, max=24, step=1,
                                        value=[0, 24],
                                        marks={h: f"{h % 12 or 12}{'a' if h < 12 else 'p'}"
                                               for h in range(0, 25, 3)},
                                        className="mb-3",
                                    ),
                                    html.Label("Date Range", className="fw-bold mb-1"),
                                    dcc.DatePickerRange(
                                        id="filter-dates",
                                        min_date_allowed=DATE_MIN,
                                        max_date_allowed=DATE_MAX,
                                        start_date=DATE_MIN,
                                        end_date=DATE_MAX,
                                        display_format="YYYY-MM-DD",
                                        className="mb-3",
                                        style={"fontSize": "12px"},
                                    ),
                                    html.Label("Deployment", className="fw-bold mb-1"),
                                    dcc.Dropdown(
                                        id="filter-deployment",
                                        options=[{"label": d, "value": d} for d in ALL_DEPLOYMENTS],
                                        value=ALL_DEPLOYMENTS,
                                        multi=True,
                                        placeholder="Select deployments...",
                                        className="mb-1",
                                    ),
                                ],
                                title="Time & Dates",
                            ),
                            dbc.AccordionItem(
                                [
                                    html.Label("Message Type", className="fw-bold mb-1"),
                                    dbc.Checklist(
                                        id="filter-types",
                                        options=[{"label": t.title(), "value": t}
                                                 for t in ALL_TYPES],
                                        value=ALL_TYPES,
                                        inline=True,
                                        className="mb-3",
                                    ),
                                    html.Label("Data Quality", className="fw-bold mb-1"),
                                    dbc.RadioItems(
                                        id="filter-quality",
                                        options=[
                                            {"label": "Clean only", "value": "clean"},
                                            {"label": "All messages", "value": "all"},
                                            {"label": "Noise only", "value": "noise"},
                                        ],
                                        value="clean",
                                        className="mb-1",
                                    ),
                                ],
                                title="Message Filters",
                            ),
                        ],
                        always_open=True,
                        active_item=["item-0"],
                        className="mb-3",
                    ),
                    html.Div(id="summary-stats"),
                ],
                style={"overflowY": "auto", "maxHeight": "85vh"},
            ),
        ],
        className="shadow-sm",
        style={"height": "100vh"},
    )


def serve_layout():
    return dbc.Container(
        [
            upload_section,
            dbc.Row([
                dbc.Col(html.Div(id="kpi-bar"), md=10),
                dbc.Col(html.Div([
                    html.A(
                        [html.I(className="bi bi-download me-1"), "Export JSON"],
                        href="/api/export-report?format=json",
                        className="btn btn-outline-primary btn-sm d-block mb-1",
                        download="metrics_report.json",
                    ),
                    html.A(
                        [html.I(className="bi bi-file-earmark-spreadsheet me-1"), "Export CSV"],
                        href="/api/export-report?format=csv",
                        className="btn btn-outline-success btn-sm d-block",
                        download="metrics_report.csv",
                    ),
                ], className="d-flex flex-column align-items-end pt-1"), md=2),
            ], className="mb-2 g-0"),
            dbc.Row(
                [
                    dbc.Col(_build_sidebar(), md=3, className="pe-0"),
                    dbc.Col(main_content, md=9, className="ps-3"),
                ],
                className="g-0",
            ),
            dcc.Location(id="page-reload", refresh=True),
        ],
        fluid=True,
        className="p-2",
    )


app.layout = serve_layout


# ── Upload callback ──────────────────────────────────────────────────────────

def _reload_global_data():
    """Reload all global data variables after new files are uploaded."""
    global DF_ALL, ALL_CHATS, ALL_SENDERS, ALL_SENDERS_RESOLVED, ALL_SENDERS_RESOLVED_FULL, ALL_TYPES
    global NOISE_TYPES, DATE_MIN, DATE_MAX, ALL_DATES, ALL_DEPLOYMENTS_LIST
    global ALL_DEPLOYMENTS, _date_to_deployment
    global TOTAL_MSGS, CLEAN_MSGS, NOISE_MSGS, NUM_CHATS, NUM_SENDERS
    global SNOW_CONFIG
    global PRICING_DATA, PRICING_INDEX
    global METRICS_REPORT

    DF_ALL = load_all_data(DATA_DIR)
    METRICS_REPORT = load_metrics_report(DATA_DIR)
    ALL_CHATS = sorted(DF_ALL["chat"].unique()) if not DF_ALL.empty else []
    _nt = set(SNOW_CONFIG.get("non_trackable_senders", []))
    ALL_SENDERS = sorted(s for s in DF_ALL[DF_ALL["sender"] != ""]["sender"].unique()
                         if s not in _nt) if not DF_ALL.empty else []
    ALL_SENDERS_RESOLVED = sorted(s for s in DF_ALL[DF_ALL["sender_resolved"] != ""]["sender_resolved"].unique()
                                  if s not in _nt) if not DF_ALL.empty else []
    ALL_SENDERS_RESOLVED_FULL = sorted(
        DF_ALL[DF_ALL["sender_resolved"] != ""]["sender_resolved"].unique()) if not DF_ALL.empty else []
    ALL_TYPES = sorted(DF_ALL["type"].unique()) if not DF_ALL.empty else []
    NOISE_TYPES = sorted(DF_ALL["noise_type"].unique()) if not DF_ALL.empty else []

    _date_col = DF_ALL["msg_date"].dropna()
    if _date_col.empty:
        _date_col = DF_ALL["export_date"].dropna()
    DATE_MIN = _date_col.min().date() if not _date_col.empty else date(2026, 2, 8)
    DATE_MAX = _date_col.max().date() if not _date_col.empty else date(2026, 2, 8)
    ALL_DATES = sorted(DF_ALL["msg_date"].dropna().dt.date.unique()) if not DF_ALL.empty else []

    ALL_DEPLOYMENTS_LIST = _compute_deployments(ALL_DATES)
    _date_to_deployment = {}
    for _dep in ALL_DEPLOYMENTS_LIST:
        _d = _dep["start_date"]
        while _d <= _dep["end_date"]:
            _date_to_deployment[_d] = _dep["label"]
            _d += timedelta(days=1)

    if not DF_ALL.empty:
        DF_ALL["deployment"] = DF_ALL["msg_date"].dt.date.map(_date_to_deployment)
    ALL_DEPLOYMENTS = [dep["label"] for dep in ALL_DEPLOYMENTS_LIST]

    TOTAL_MSGS = len(DF_ALL)
    CLEAN_MSGS = len(DF_ALL[DF_ALL["noise_type"] == "clean"]) if not DF_ALL.empty else 0
    NOISE_MSGS = TOTAL_MSGS - CLEAN_MSGS
    NUM_CHATS = DF_ALL["chat"].nunique() if not DF_ALL.empty else 0
    NUM_SENDERS = len(ALL_SENDERS_RESOLVED)
    SNOW_CONFIG = load_config(os.path.join(DATA_DIR, "config", "snow_removal.json"))

    PRICING_DATA = load_pricing(os.path.join(DATA_DIR, "config", "pricing.json"))
    PRICING_INDEX = {}
    for p in PRICING_DATA:
        addr_norm = _normalize_location(p.get("address", ""))
        if addr_norm:
            PRICING_INDEX[addr_norm] = p
        name_norm = _normalize_location(p.get("building_name", ""))
        if name_norm:
            PRICING_INDEX[name_norm] = p


@app.callback(
    Output("upload-status", "children"),
    Output("page-reload", "href"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents_list, filenames_list):
    if not contents_list:
        return no_update, no_update

    archive_dir = os.path.join(DATA_DIR, "archive")
    os.makedirs(archive_dir, exist_ok=True)

    saved = []
    errors = []
    for content, filename in zip(contents_list, filenames_list):
        try:
            content_type, content_string = content.split(",", 1)
            decoded = base64.b64decode(content_string)
            data = json.loads(decoded.decode("utf-8"))

            if isinstance(data, dict) and "summary" in data and "crew_metrics" in data:
                save_path = os.path.join(archive_dir, "metrics_report.json")
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
                saved.append("metrics_report.json")
                continue

            if not isinstance(data, dict) or "exportInfo" not in data or "messages" not in data:
                errors.append(f"{filename}: not a valid chat export or metrics file")
                continue

            safe_name = os.path.basename(filename)
            safe_name = re.sub(r"[^\w.\-]", "_", safe_name)
            if not safe_name.endswith(".json"):
                safe_name += ".json"
            save_path = os.path.join(archive_dir, safe_name)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            saved.append(safe_name)
        except Exception as e:
            errors.append(f"{filename}: {str(e)[:50]}")

    _reload_global_data()

    msgs = []
    if saved:
        msgs.append(html.Span(
            f"Uploaded {len(saved)} file(s). ",
            className="text-success fw-bold"
        ))
    if errors:
        msgs.append(html.Span(
            f"{len(errors)} error(s). ",
            className="text-danger"
        ))

    if saved:
        return html.Div(msgs), "/"
    return html.Div(msgs), None


# ── Shared filter inputs ─────────────────────────────────────────────────────

FILTER_INPUTS = [
    Input("filter-chats", "value"),
    Input("filter-senders", "value"),
    Input("filter-quality", "value"),
    Input("filter-types", "value"),
    Input("filter-time", "value"),
    Input("filter-resolved", "value"),
    Input("filter-dates", "start_date"),
    Input("filter-dates", "end_date"),
    Input("filter-deployment", "value"),
]


def _apply_quality_filter(quality):
    """Convert quality radio selection to list of noise_types."""
    if quality == "clean":
        return ["clean"]
    elif quality == "noise":
        return [n for n in NOISE_TYPES if n != "clean"]
    else:  # "all"
        return NOISE_TYPES


def _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
            date_start=None, date_end=None, deployments=None):
    """Convenience wrapper that translates quality radio to noise_types."""
    noise_types = _apply_quality_filter(quality)
    return get_filtered_df(chats, senders, noise_types, msg_types,
                           time_range, use_resolved, date_start, date_end, deployments)


def _sender_col(use_resolved):
    return "sender_resolved" if use_resolved else "sender"


# ── Callback: Summary stats ─────────────────────────────────────────────────

@app.callback(
    Output("summary-stats", "children"),
    FILTER_INPUTS,
)
def update_summary(chats, senders, quality, msg_types, time_range,
                   use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    scol = _sender_col(use_resolved)
    n_clean = len(df[df["noise_type"] == "clean"])
    n_noise = len(df[df["noise_type"] != "clean"])
    n_senders = df[df[scol] != ""][scol].nunique()
    n_chats = df["chat"].nunique()

    # Compute average interval per person
    df_timed = df[(df[scol] != "") & df["time"].notna()]
    gap_parts = []
    for sender, grp in df_timed.groupby(scol):
        times = grp["time"].sort_values()
        if len(times) >= 2:
            diffs = times.diff().dropna().dt.total_seconds() / 60.0
            gap_parts.append(diffs.mean())
    avg_gap = f"{np.mean(gap_parts):.1f} min" if gap_parts else "N/A"

    # Compute efficiency metrics
    ds = _build_daily_summary(df, scol)
    if not ds.empty:
        avg_first = ds["first_hour"].mean()
        avg_first_h = int(avg_first)
        avg_first_m = int((avg_first - avg_first_h) * 60)
        period = "AM" if avg_first_h < 12 else "PM"
        disp_h = avg_first_h % 12 or 12
        avg_first_str = f"{disp_h}:{avg_first_m:02d} {period}"
        avg_window = f"{ds['window_hrs'].mean():.1f} hrs"
    else:
        avg_first_str = "N/A"
        avg_window = "N/A"

    return dbc.Card(
        dbc.CardBody([
            html.H6("Summary", className="card-title"),
            html.P(f"Total: {len(df)}", className="mb-1"),
            html.P(f"Clean: {n_clean}", className="mb-1 text-success"),
            html.P(f"Noise: {n_noise}", className="mb-1 text-danger"),
            html.P(f"Senders: {n_senders}", className="mb-1"),
            html.P(f"Chats: {n_chats}", className="mb-1"),
            html.P(f"Avg interval: {avg_gap}", className="mb-1 text-info"),
            html.Hr(),
            html.H6("Efficiency", className="card-title"),
            html.P(f"Avg first report: {avg_first_str}", className="mb-1"),
            html.P(f"Avg report window: {avg_window}", className="mb-0"),
        ]),
        color="light",
    )


# ── Chart 1: Messages per Chat ──────────────────────────────────────────────

@app.callback(Output("chart-msgs-per-chat", "figure"), FILTER_INPUTS)
def chart_msgs_per_chat(chats, senders, quality, msg_types, time_range,
                        use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    scol = _sender_col(use_resolved)
    df_valid = df[df[scol] != ""]
    if df_valid.empty:
        return _empty_fig("Messages per Chat")

    counts = (df_valid.groupby(["chat", scol])
              .size().reset_index(name="count"))
    # Totals per chat for labels
    totals = counts.groupby("chat")["count"].sum().reset_index()
    totals.columns = ["chat", "total"]
    fig = px.bar(
        counts, y="chat", x="count", color=scol,
        orientation="h", text="count",
        title="Messages per Chat",
        labels={"count": "Messages", "chat": "Chat Group", scol: "Sender"},
    )
    fig.update_traces(textposition="inside", textfont_size=10)
    # Add total annotations on the right
    for _, row in totals.iterrows():
        fig.add_annotation(
            x=row["total"], y=row["chat"],
            text=f"  {int(row['total'])}",
            showarrow=False, font=dict(size=11, color="#2c3e50"),
            xanchor="left",
        )
    fig.update_layout(
        barmode="stack",
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=10, r=40, t=40, b=10),
        legend=dict(font=dict(size=10)),
        height=400,
    )
    return fig


# ── Chart 2: Type Distribution (donut) ──────────────────────────────────────

@app.callback(Output("chart-type-donut", "figure"), FILTER_INPUTS)
def chart_type_donut(chats, senders, quality, msg_types, time_range,
                     use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    if df.empty:
        return _empty_fig("Message Type Distribution")

    counts = df["type"].value_counts().reset_index()
    counts.columns = ["type", "count"]
    fig = px.pie(
        counts, names="type", values="count", hole=0.4,
        title="Message Type Distribution",
    )
    fig.update_traces(textinfo="label+value+percent", textfont_size=12)
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
    )
    return fig


# ── Chart 3: Hourly Heatmap (sender × hour) ─────────────────────────────────

@app.callback(Output("chart-heatmap", "figure"), FILTER_INPUTS)
def chart_heatmap(chats, senders, quality, msg_types, time_range,
                  use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    scol = _sender_col(use_resolved)
    df_valid = df[(df[scol] != "") & df["hour_int"].notna()]
    if df_valid.empty:
        return _empty_fig("Hourly Activity Heatmap")

    df_valid = df_valid.copy()
    df_valid["hour_int"] = df_valid["hour_int"].astype(int)
    ct = pd.crosstab(df_valid[scol], df_valid["hour_int"])
    # Ensure all 24 hours
    for h in range(24):
        if h not in ct.columns:
            ct[h] = 0
    ct = ct[sorted(ct.columns)]
    hour_labels = [f"{h % 12 or 12}{'a' if h < 12 else 'p'}" for h in range(24)]

    fig = go.Figure(data=go.Heatmap(
        z=ct.values,
        x=hour_labels,
        y=ct.index.tolist(),
        colorscale="YlOrRd",
        text=ct.values,
        texttemplate="%{text}",
        hovertemplate="Sender: %{y}<br>Hour: %{x}<br>Messages: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title="Hourly Activity Heatmap",
        xaxis_title="Hour of Day",
        yaxis_title="Sender",
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
    )
    return fig


# ── Chart 4: Sender × Chat Matrix (bubble) ──────────────────────────────────

@app.callback(Output("chart-sender-chat", "figure"), FILTER_INPUTS)
def chart_sender_chat(chats, senders, quality, msg_types, time_range,
                      use_resolved, date_start, date_end, deployments):
    try:
        df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                     date_start, date_end, deployments)
        scol = _sender_col(use_resolved)
        if scol not in df.columns:
            return _empty_fig("Sender × Chat Participation")
        df_valid = df[df[scol] != ""]
        if df_valid.empty:
            return _empty_fig("Sender × Chat Participation")

        counts = (df_valid.groupby([scol, "chat"])
                  .size().reset_index(name="count"))
        top_combos = counts.nlargest(50, "count")
        fig = px.bar(
            top_combos, x="chat", y="count", color=scol,
            text="count", barmode="group",
            title="Sender × Chat Participation",
            labels={"count": "Messages", "chat": "Chat", scol: "Sender"},
        )
        fig.update_traces(textposition="outside", textfont_size=10)
        fig.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            height=400,
            xaxis_tickangle=-30,
            legend=dict(font=dict(size=10)),
        )
        return fig
    except Exception as e:
        print(f"[chart-sender-chat] Error: {e}")
        return _empty_fig("Sender × Chat Participation")


# ── Chart 5: Activity Timeline ──────────────────────────────────────────────

@app.callback(Output("chart-timeline", "figure"), FILTER_INPUTS)
def chart_timeline(chats, senders, quality, msg_types, time_range,
                   use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    scol = _sender_col(use_resolved)
    df_valid = df[(df[scol] != "") & df["time"].notna()].copy()
    if df_valid.empty:
        return _empty_fig("Activity Timeline")

    fig = px.scatter(
        df_valid, x="time", y=scol, color="chat",
        hover_data=["content", "type", "noise_type"],
        title="Activity Timeline",
        labels={"time": "Time", scol: "Sender", "chat": "Chat"},
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        height=450,
        xaxis_tickformat="%I:%M %p",
        legend=dict(font=dict(size=10)),
    )
    fig.update_traces(marker=dict(size=10, opacity=0.7))
    return fig


# ── Chart 6: KDE Density ────────────────────────────────────────────────────

@app.callback(Output("chart-kde", "figure"), FILTER_INPUTS)
def chart_kde(chats, senders, quality, msg_types, time_range, use_resolved,
              date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    scol = _sender_col(use_resolved)
    df_valid = df[(df[scol] != "") & df["hour"].notna()]
    if df_valid.empty:
        return _empty_fig("Message Density by Sender")

    fig = go.Figure()
    sender_list = sorted(df_valid[scol].unique())
    colors = px.colors.qualitative.Plotly
    for i, sender in enumerate(sender_list):
        hours = df_valid[df_valid[scol] == sender]["hour"].values
        if len(hours) < 2:
            continue
        # Compute KDE using numpy histogram for smoothing
        hist, bin_edges = np.histogram(hours, bins=48, range=(0, 24),
                                       density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # Simple moving average smoothing
        kernel_size = 3
        smoothed = np.convolve(hist, np.ones(kernel_size) / kernel_size,
                               mode="same")
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=bin_centers, y=smoothed,
            mode="lines", name=sender,
            fill="tozeroy", opacity=0.4,
            line=dict(color=color, width=2),
        ))

    fig.update_layout(
        title="Message Density by Sender (KDE-style)",
        xaxis_title="Hour of Day",
        yaxis_title="Density",
        xaxis=dict(range=[0, 24], dtick=2,
                   ticktext=[f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}"
                             for h in range(0, 25, 2)],
                   tickvals=list(range(0, 25, 2))),
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
        legend=dict(font=dict(size=10)),
    )
    return fig


# ── Chart 6b: Hourly Message Count (line) ───────────────────────────────────

@app.callback(Output("chart-line-hourly", "figure"), FILTER_INPUTS)
def chart_line_hourly(chats, senders, quality, msg_types, time_range,
                      use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    scol = _sender_col(use_resolved)
    df_valid = df[(df[scol] != "") & df["hour_int"].notna()].copy()
    if df_valid.empty:
        return _empty_fig("Messages per Hour (Line)")

    df_valid["hour_int"] = df_valid["hour_int"].astype(int)
    counts = (df_valid.groupby([scol, "hour_int"])
              .size().reset_index(name="count"))
    # Fill missing hours with 0 then compute cumulative sum per sender
    sender_list = counts[scol].unique()
    full_index = pd.MultiIndex.from_product(
        [sender_list, range(24)], names=[scol, "hour_int"])
    counts = (counts.set_index([scol, "hour_int"])
              .reindex(full_index, fill_value=0)
              .reset_index())
    counts = counts.sort_values([scol, "hour_int"])
    counts["cumulative"] = counts.groupby(scol)["count"].cumsum()

    hour_labels = [f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}" for h in range(24)]

    fig = px.line(
        counts, x="hour_int", y="cumulative", color=scol,
        markers=True,
        title="Cumulative Messages by Hour per Sender",
        labels={"hour_int": "Hour of Day", "cumulative": "Cumulative Messages", scol: "Sender"},
    )
    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(0, 24, 2)),
            ticktext=[hour_labels[h] for h in range(0, 24, 2)],
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
        legend=dict(font=dict(size=10)),
    )
    return fig


# ── Chart 9b: Content Length vs Hour (scatter) ───────────────────────────────

@app.callback(Output("chart-scatter-len-hour", "figure"), FILTER_INPUTS)
def chart_scatter_len_hour(chats, senders, quality, msg_types, time_range,
                           use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    scol = _sender_col(use_resolved)
    df_valid = df[(df[scol] != "") & df["hour"].notna() &
                  (df["content_len"] > 0)].copy()
    if df_valid.empty:
        return _empty_fig("Content Length vs Hour (Scatter)")

    fig = px.scatter(
        df_valid, x="hour", y="content_len", color=scol,
        opacity=0.6,
        hover_data=["chat", "type", "content"],
        title="Content Length vs Hour of Day",
        labels={"hour": "Hour of Day", "content_len": "Message Length (chars)",
                scol: "Sender"},
    )
    fig.update_layout(
        xaxis=dict(range=[0, 24], dtick=2,
                   ticktext=[f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}"
                             for h in range(0, 25, 2)],
                   tickvals=list(range(0, 25, 2))),
        margin=dict(l=10, r=10, t=40, b=10),
        height=450,
        legend=dict(font=dict(size=10)),
    )
    return fig


# ── Chart 7: Messages per Sender ────────────────────────────────────────────

@app.callback(Output("chart-msgs-per-sender", "figure"), FILTER_INPUTS)
def chart_msgs_per_sender(chats, senders, quality, msg_types, time_range,
                          use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    scol = _sender_col(use_resolved)
    df_valid = df[df[scol] != ""]
    if df_valid.empty:
        return _empty_fig("Messages per Sender")

    # Count + average interval per sender
    counts = df_valid[scol].value_counts().reset_index()
    counts.columns = ["sender", "count"]
    df_timed = df_valid[df_valid["time"].notna()]
    avg_gaps = {}
    for sender, grp in df_timed.groupby(scol):
        times = grp["time"].sort_values()
        if len(times) >= 2:
            avg_gaps[sender] = times.diff().dropna().dt.total_seconds().mean() / 60.0
    counts["avg_gap"] = counts["sender"].map(avg_gaps)
    counts["label"] = counts.apply(
        lambda r: f"{r['count']}  (avg {r['avg_gap']:.0f}m)"
        if pd.notna(r["avg_gap"]) else str(r["count"]),
        axis=1)

    fig = px.bar(
        counts, y="sender", x="count", orientation="h",
        text="label",
        title="Messages per Sender (count + avg interval)",
        labels={"count": "Messages", "sender": "Sender"},
        color="count",
        color_continuous_scale="Viridis",
    )
    fig.update_traces(textposition="outside", textfont_size=10)
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=10, r=80, t=40, b=10),
        height=400,
    )
    return fig


# ── Chart 8: Message Gap Analysis (box plot) ────────────────────────────────

@app.callback(Output("chart-gap-box", "figure"), FILTER_INPUTS)
def chart_gap_box(chats, senders, quality, msg_types, time_range,
                  use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    scol = _sender_col(use_resolved)
    df_valid = df[(df[scol] != "") & df["time"].notna()].copy()
    if df_valid.empty:
        return _empty_fig("Message Gap Analysis")

    # Compute gaps per sender
    gap_rows = []
    for sender, grp in df_valid.groupby(scol):
        times = grp["time"].sort_values()
        if len(times) < 2:
            continue
        diffs = times.diff().dropna().dt.total_seconds() / 60.0  # minutes
        for gap in diffs:
            gap_rows.append({"sender": sender, "gap_min": gap})

    if not gap_rows:
        return _empty_fig("Message Gap Analysis")

    df_gaps = pd.DataFrame(gap_rows)
    fig = px.box(
        df_gaps, x="sender", y="gap_min", points="all",
        title="Message Gap Analysis (minutes between messages)",
        labels={"gap_min": "Gap (minutes)", "sender": "Sender"},
    )
    # Add mean annotation per sender
    means = df_gaps.groupby("sender")["gap_min"].mean()
    for sender, avg in means.items():
        fig.add_annotation(
            x=sender, y=avg,
            text=f"avg {avg:.0f}m",
            showarrow=True, arrowhead=2, arrowcolor="#e74c3c",
            font=dict(size=10, color="#e74c3c"),
            ax=30, ay=-20,
        )
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
        xaxis_tickangle=-30,
    )
    return fig


# ── Chart 9: Content Length Distribution ─────────────────────────────────────

@app.callback(Output("chart-content-len", "figure"), FILTER_INPUTS)
def chart_content_len(chats, senders, quality, msg_types, time_range,
                      use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    scol = _sender_col(use_resolved)
    df_valid = df[df[scol] != ""].copy()
    if df_valid.empty:
        return _empty_fig("Content Length Distribution")

    fig = px.histogram(
        df_valid, x="content_len", color=scol,
        marginal="box",
        title="Content Length Distribution",
        labels={"content_len": "Content Length (chars)", scol: "Sender"},
        nbins=30,
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
        legend=dict(font=dict(size=10)),
    )
    return fig


# ── Chart 10: Data Quality Overview (always shows all data) ─────────────────

@app.callback(
    Output("chart-quality", "figure"),
    [Input("filter-chats", "value"),
     Input("filter-dates", "start_date"),
     Input("filter-dates", "end_date")],
)
def chart_quality(chats, date_start, date_end):
    df = DF_ALL.copy()
    if date_start:
        df = df[df["msg_date"] >= pd.Timestamp(date_start)]
    if date_end:
        df = df[df["msg_date"] <= pd.Timestamp(date_end)]
    if chats:
        df = df[df["chat"].isin(chats)]
    if df.empty:
        return _empty_fig("Data Quality Overview")

    counts = (df.groupby(["chat", "noise_type"])
              .size().reset_index(name="count"))

    color_map = {
        "clean": "#2ecc71",
        "css_html": "#e74c3c",
        "load_error": "#e67e22",
        "system_metadata": "#9b59b6",
        "empty_sender_caption": "#3498db",
    }

    fig = px.bar(
        counts, x="chat", y="count", color="noise_type",
        text="count",
        title="Data Quality Overview (noise types per chat)",
        labels={"count": "Messages", "chat": "Chat", "noise_type": "Noise Type"},
        color_discrete_map=color_map,
        barmode="stack",
    )
    fig.update_traces(textposition="inside", textfont_size=10)
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
        xaxis_tickangle=-30,
        legend=dict(font=dict(size=10)),
    )
    return fig


# ── Data Table ───────────────────────────────────────────────────────────────

@app.callback(Output("data-table-container", "children"), FILTER_INPUTS)
def update_data_table(chats, senders, quality, msg_types, time_range,
                      use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    scol = _sender_col(use_resolved)

    # Select columns for display
    display_cols = ["chat", scol, "timestamp", "type", "noise_type",
                    "content_len", "content"]
    df_display = df[display_cols].copy()
    df_display.columns = ["Chat", "Sender", "Time", "Type", "Noise",
                          "Length", "Content"]
    # Truncate content for display
    df_display["Content"] = df_display["Content"].str[:120]

    table = dash_table.DataTable(
        data=df_display.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df_display.columns],
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        page_size=20,
        style_table={"overflowX": "auto"},
        style_cell={
            "textAlign": "left",
            "padding": "8px",
            "fontSize": "13px",
            "maxWidth": "300px",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
        },
        style_header={
            "backgroundColor": "#2c3e50",
            "color": "white",
            "fontWeight": "bold",
        },
        style_data_conditional=[
            {
                "if": {
                    "filter_query": "{Noise} != 'clean'",
                },
                "backgroundColor": "#fde8e8",
                "color": "#c0392b",
            },
            {
                "if": {
                    "filter_query": "{Noise} = 'clean'",
                },
                "backgroundColor": "#eafaf1",
            },
        ],
    )
    return table


# ── Efficiency: helper to build daily summary per sender ─────────────────────

def _build_daily_summary(df, scol):
    """Return a DataFrame with one row per (sender, date) with efficiency stats."""
    df_valid = df[(df[scol] != "") & df["time"].notna() & df["msg_date"].notna()].copy()
    if df_valid.empty:
        return pd.DataFrame()

    rows = []
    for (sender, d), grp in df_valid.groupby([scol, df_valid["msg_date"].dt.date]):
        times = grp["time"].sort_values()
        first = times.iloc[0]
        last = times.iloc[-1]
        window_hrs = (last - first).total_seconds() / 3600.0
        n = len(times)
        avg_gap = (times.diff().dropna().dt.total_seconds().mean() / 60.0
                   if n >= 2 else 0.0)
        rows.append({
            "sender": sender,
            "date": d,
            "first_time": first,
            "last_time": last,
            "first_hour": first.hour + first.minute / 60.0,
            "last_hour": last.hour + last.minute / 60.0,
            "window_hrs": window_hrs,
            "msg_count": n,
            "avg_gap_min": avg_gap,
        })
    return pd.DataFrame(rows)


# ── Crew Metrics: site visit analysis ─────────────────────────────────────────

def _count_billable_routes_from_df(df):
    if df is None or df.empty:
        return 0
    loc_df = df[df["location"].astype(str).str.len() > 0].copy()
    if loc_df.empty:
        return 0
    def _sa(chat):
        _, is_sw, is_pl = _extract_crew_from_chat(chat)
        if is_pl:
            return "Parking Lot"
        return "Sidewalk"
    loc_df["_service_area"] = loc_df["chat"].apply(_sa)
    dep_col = "deployment" if "deployment" in loc_df.columns else None
    if dep_col:
        routes = loc_df.groupby([dep_col, "location", "_service_area"]).ngroups
    else:
        routes = loc_df.groupby(["location", "_service_area"]).ngroups
    return routes


def _build_site_visits(df, scol):
    """Build a DataFrame of site visits from messages with locations.

    Walks messages sorted by (sender, date, time). A new location or a new
    calendar date = new site visit.  Messages without a location inherit the
    current site.

    Returns DataFrame: sender, date, location, start_time, end_time,
                       duration_min, msg_count, visit_order
    """
    df_valid = df[
        (df[scol] != "") & df["time"].notna() & df["msg_date"].notna()
    ].copy()
    if df_valid.empty:
        return pd.DataFrame()

    df_valid = df_valid.sort_values([scol, "msg_date", "time"])

    def _flush(visits, sender, current_loc, visit_start, visit_end,
               visit_msgs, visit_date, visit_order):
        if current_loc and visit_start is not None:
            dur = (visit_end - visit_start).total_seconds() / 60.0
            visits.append({
                "sender": sender,
                "date": visit_date,
                "location": current_loc,
                "start_time": visit_start,
                "end_time": visit_end,
                "duration_min": max(dur, 1.0),
                "msg_count": visit_msgs,
                "visit_order": visit_order,
            })

    visits = []
    for sender, grp in df_valid.groupby(scol):
        current_loc = ""
        visit_start = None
        visit_end = None
        visit_msgs = 0
        visit_date = None
        visit_order = 0
        current_day = None

        for _, row in grp.iterrows():
            loc = row["location"]
            t = row["time"]
            d = row["msg_date"]
            row_day = d.date() if hasattr(d, "date") else d

            # Reset on date change
            if current_day is not None and row_day != current_day:
                _flush(visits, sender, current_loc, visit_start, visit_end,
                       visit_msgs, visit_date, visit_order)
                current_loc = ""
                visit_start = None
                visit_end = None
                visit_msgs = 0
                visit_order = 0

            current_day = row_day

            if loc and loc != current_loc:
                # Save previous visit if any
                _flush(visits, sender, current_loc, visit_start, visit_end,
                       visit_msgs, visit_date, visit_order)
                # Start new visit
                current_loc = loc
                visit_start = t
                visit_end = t
                visit_msgs = 1
                visit_date = d
                visit_order += 1
            elif current_loc:
                # Continue current visit
                visit_end = t
                visit_msgs += 1

        # Save last visit
        _flush(visits, sender, current_loc, visit_start, visit_end,
               visit_msgs, visit_date, visit_order)

    return pd.DataFrame(visits)


def _build_transitions(visits_df):
    """Compute transition times between consecutive site visits per sender.

    Returns DataFrame: sender, date, from_location, to_location, transition_min
    """
    if visits_df.empty:
        return pd.DataFrame()

    rows = []
    for sender, grp in visits_df.groupby("sender"):
        grp_sorted = grp.sort_values(["date", "start_time"])
        prev = None
        for _, row in grp_sorted.iterrows():
            if prev is not None:
                gap = (row["start_time"] - prev["end_time"]).total_seconds() / 60.0
                # Only count transitions within same day and reasonable gap
                if gap >= 0 and gap < 480:  # max 8 hours
                    rows.append({
                        "sender": sender,
                        "date": row["date"],
                        "from_location": prev["location"],
                        "to_location": row["location"],
                        "transition_min": gap,
                    })
            prev = row

    return pd.DataFrame(rows)


_WORK_BLOCK_GAP_MIN = 120  # gaps > 2 h split into separate work blocks


def _compute_active_hours(day_visits):
    """Sum work-block durations for one sender-day.

    Consecutive site visits with gaps < _WORK_BLOCK_GAP_MIN are grouped into
    a work block.  Off-duty gaps (> threshold) start a new block.
    """
    vs = day_visits.sort_values("start_time")
    total_sec = 0.0
    block_start = vs.iloc[0]["start_time"]
    block_end = vs.iloc[0]["end_time"]

    for i in range(1, len(vs)):
        row = vs.iloc[i]
        gap = (row["start_time"] - block_end).total_seconds() / 60.0
        if gap < _WORK_BLOCK_GAP_MIN:
            # Extend current work block
            block_end = max(block_end, row["end_time"])
        else:
            # Flush current block, start new one
            total_sec += (block_end - block_start).total_seconds()
            block_start = row["start_time"]
            block_end = row["end_time"]

    # Flush last block
    total_sec += (block_end - block_start).total_seconds()
    return max(total_sec / 3600.0, 1 / 60.0)  # at least 1 minute


def _build_crew_scorecard(visits_df, ds_df):
    """Aggregate crew efficiency metrics per sender.

    Active hours are the sum of work-block durations per day: consecutive
    site visits with gaps under 2 h are grouped into a work block; longer
    off-duty gaps are excluded.

    Returns DataFrame: sender, days_active, total_sites, avg_sites_per_day,
                       avg_sites_per_hour, avg_transition_min, total_active_hrs
    """
    if visits_df.empty:
        return pd.DataFrame()

    trans_df = _build_transitions(visits_df)

    rows = []
    for sender, grp in visits_df.groupby("sender"):
        total_sites = len(grp)
        days_active = grp["date"].nunique()
        avg_sites_day = total_sites / days_active if days_active else 0

        # Active hours = sum of work-block spans per day
        total_hrs = 0.0
        for _d, day_grp in grp.groupby(grp["date"].dt.date):
            total_hrs += _compute_active_hours(day_grp)

        avg_sites_hr = total_sites / total_hrs if total_hrs > 0 else 0

        # Average transition time
        if not trans_df.empty and sender in trans_df["sender"].values:
            avg_trans = trans_df[trans_df["sender"] == sender]["transition_min"].mean()
        else:
            avg_trans = 0.0

        rows.append({
            "sender": sender,
            "days_active": days_active,
            "total_sites": total_sites,
            "avg_sites_per_day": round(avg_sites_day, 1),
            "avg_sites_per_hour": round(avg_sites_hr, 1),
            "avg_transition_min": round(avg_trans, 1),
            "total_active_hrs": round(total_hrs, 1),
        })

    return pd.DataFrame(rows).sort_values("total_sites", ascending=False)


# ── Efficiency Chart 1: Daily Report Card (table) ───────────────────────────

@app.callback(Output("efficiency-report-card", "children"), FILTER_INPUTS)
def efficiency_report_card(chats, senders, quality, msg_types, time_range,
                           use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    scol = _sender_col(use_resolved)
    ds = _build_daily_summary(df, scol)
    if ds.empty:
        return html.P("No data for current filters", className="text-muted p-3")

    ds_display = ds.copy()
    ds_display["first_time"] = ds_display["first_time"].dt.strftime("%I:%M %p")
    ds_display["last_time"] = ds_display["last_time"].dt.strftime("%I:%M %p")
    ds_display["window_hrs"] = ds_display["window_hrs"].round(1)
    ds_display["avg_gap_min"] = ds_display["avg_gap_min"].round(1)
    ds_display["date"] = ds_display["date"].astype(str)
    ds_display = ds_display[["sender", "date", "first_time", "last_time",
                              "window_hrs", "msg_count", "avg_gap_min"]]
    ds_display.columns = ["Sender", "Date", "First Msg", "Last Msg",
                           "Window (hrs)", "Messages", "Avg Gap (min)"]

    table = dash_table.DataTable(
        data=ds_display.to_dict("records"),
        columns=[{"name": c, "id": c} for c in ds_display.columns],
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        page_size=20,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "8px", "fontSize": "13px"},
        style_header={
            "backgroundColor": "#2c3e50", "color": "white", "fontWeight": "bold",
        },
        style_data_conditional=[
            {"if": {"filter_query": "{Window (hrs)} < 1"},
             "backgroundColor": "#fde8e8", "color": "#c0392b"},
            {"if": {"filter_query": "{Window (hrs)} >= 4"},
             "backgroundColor": "#eafaf1"},
        ],
    )
    return html.Div([
        html.H5("Daily Report Card", className="mb-2"),
        table,
    ])


# ── Efficiency Chart 2: First Report Time scatter ───────────────────────────

@app.callback(Output("chart-first-report", "figure"), FILTER_INPUTS)
def chart_first_report(chats, senders, quality, msg_types, time_range,
                       use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    scol = _sender_col(use_resolved)
    ds = _build_daily_summary(df, scol)
    if ds.empty:
        return _empty_fig("First Report Time by Sender")

    ds["date_str"] = ds["date"].astype(str)
    fig = px.scatter(
        ds, x="date_str", y="first_hour", color="sender",
        size="msg_count",
        hover_data={"first_hour": False, "date_str": False,
                    "first_time": True, "msg_count": True, "window_hrs": ":.1f"},
        title="First Report Time by Sender (per day)",
        labels={"date_str": "Date", "first_hour": "First Message (hour)",
                "sender": "Sender"},
        size_max=18,
    )
    fig.update_layout(
        yaxis=dict(
            range=[24, 0], dtick=2,
            ticktext=[f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}"
                      for h in range(0, 25, 2)],
            tickvals=list(range(0, 25, 2)),
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        height=420,
        legend=dict(font=dict(size=10)),
    )
    return fig


# ── Efficiency Chart 3: Reporting Window box plot ────────────────────────────

@app.callback(Output("chart-report-window-box", "figure"), FILTER_INPUTS)
def chart_report_window_box(chats, senders, quality, msg_types, time_range,
                            use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    scol = _sender_col(use_resolved)
    ds = _build_daily_summary(df, scol)
    if ds.empty:
        return _empty_fig("Reporting Window by Sender")

    fig = px.box(
        ds, x="sender", y="window_hrs", points="all",
        title="Reporting Window by Sender (hours)",
        labels={"window_hrs": "Window (hours)", "sender": "Sender"},
        color="sender",
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
        xaxis_tickangle=-30,
        showlegend=False,
    )
    return fig


# ── Efficiency Chart 4: Daily Message Count Trend ────────────────────────────

@app.callback(Output("chart-daily-count-trend", "figure"), FILTER_INPUTS)
def chart_daily_count_trend(chats, senders, quality, msg_types, time_range,
                            use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    scol = _sender_col(use_resolved)
    ds = _build_daily_summary(df, scol)
    if ds.empty:
        return _empty_fig("Daily Message Count Trend")

    ds["date_str"] = ds["date"].astype(str)
    fig = px.line(
        ds.sort_values("date"), x="date_str", y="msg_count",
        color="sender", markers=True,
        title="Daily Message Count per Sender",
        labels={"date_str": "Date", "msg_count": "Messages", "sender": "Sender"},
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
        legend=dict(font=dict(size=10)),
    )
    return fig


# ── Crew Metrics Callback 1: Crew Scorecard Table ────────────────────────────

@app.callback(Output("crew-scorecard-container", "children"), FILTER_INPUTS)
def crew_scorecard(chats, senders, quality, msg_types, time_range,
                   use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    visits_df = _build_site_visits(df, "chat")
    if visits_df.empty:
        return html.P("No site visit data for current filters",
                       className="text-muted p-3")

    ds = _build_daily_summary(df, "chat")
    sc = _build_crew_scorecard(visits_df, ds)
    if sc.empty:
        return html.P("No crew scorecard data", className="text-muted p-3")

    sc_display = sc.rename(columns={
        "sender": "Crew",
        "days_active": "Days Active",
        "total_sites": "Total Sites",
        "avg_sites_per_day": "Sites/Day",
        "avg_sites_per_hour": "Sites/Hour",
        "avg_transition_min": "Avg Transition (min)",
        "total_active_hrs": "Active Hours",
    })

    table = dash_table.DataTable(
        data=sc_display.to_dict("records"),
        columns=[{"name": c, "id": c} for c in sc_display.columns],
        sort_action="native",
        sort_mode="multi",
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "8px", "fontSize": "13px"},
        style_header={
            "backgroundColor": "#2c3e50", "color": "white", "fontWeight": "bold",
        },
        style_data_conditional=[
            {"if": {"filter_query": "{Sites/Hour} >= 3"},
             "backgroundColor": "#eafaf1"},
            {"if": {"filter_query": "{Sites/Hour} < 1"},
             "backgroundColor": "#fde8e8", "color": "#c0392b"},
        ],
    )
    return html.Div([
        html.H5("Crew Scorecard", className="mb-2"),
        table,
    ])


# ── Crew Metrics Callback 2: Sites per Hour Bar Chart ────────────────────────

@app.callback(Output("chart-sites-per-hour", "figure"), FILTER_INPUTS)
def chart_sites_per_hour(chats, senders, quality, msg_types, time_range,
                         use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    visits_df = _build_site_visits(df, "chat")
    if visits_df.empty:
        return _empty_fig("Sites per Hour")

    ds = _build_daily_summary(df, "chat")
    sc = _build_crew_scorecard(visits_df, ds)
    if sc.empty:
        return _empty_fig("Sites per Hour")

    sc_sorted = sc.sort_values("avg_sites_per_hour")
    fig = px.bar(
        sc_sorted, y="sender", x="avg_sites_per_hour", orientation="h",
        text="avg_sites_per_hour",
        title="Average Sites per Hour by Crew",
        labels={"avg_sites_per_hour": "Sites / Hour", "sender": "Crew"},
        color="avg_sites_per_hour",
        color_continuous_scale="RdYlGn",
    )
    fig.update_traces(textposition="outside", textfont_size=11,
                      texttemplate="%{text:.1f}")
    fig.update_layout(
        margin=dict(l=10, r=60, t=40, b=10),
        height=400,
        yaxis={"categoryorder": "total ascending"},
    )
    return fig


# ── Crew Metrics Callback 3: Transition Time Box Plot ────────────────────────

@app.callback(Output("chart-transition-time", "figure"), FILTER_INPUTS)
def chart_transition_time(chats, senders, quality, msg_types, time_range,
                          use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    visits_df = _build_site_visits(df, "chat")
    if visits_df.empty:
        return _empty_fig("Transition Time Between Sites")

    trans_df = _build_transitions(visits_df)
    if trans_df.empty:
        return _empty_fig("Transition Time Between Sites")

    fig = px.box(
        trans_df, x="sender", y="transition_min", points="all",
        title="Transition Time Between Sites (minutes)",
        labels={"transition_min": "Transition (min)", "sender": "Crew"},
        color="sender",
    )
    # Add mean annotations
    means = trans_df.groupby("sender")["transition_min"].mean()
    for sender, avg in means.items():
        fig.add_annotation(
            x=sender, y=avg,
            text=f"avg {avg:.0f}m",
            showarrow=True, arrowhead=2, arrowcolor="#e74c3c",
            font=dict(size=10, color="#e74c3c"),
            ax=30, ay=-20,
        )
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
        xaxis_tickangle=-30,
        showlegend=False,
    )
    return fig


# ── Crew Metrics Callback 4: Route Timeline (Gantt-style) ────────────────────

@app.callback(Output("chart-route-timeline", "figure"), FILTER_INPUTS)
def chart_route_timeline(chats, senders, quality, msg_types, time_range,
                         use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    visits_df = _build_site_visits(df, "chat")
    if visits_df.empty:
        return _empty_fig("Route Timeline")

    # Build Gantt-style timeline using px.timeline
    gantt_data = visits_df.copy()
    gantt_data["start_str"] = gantt_data["start_time"].dt.strftime("%Y-%m-%d %H:%M")
    # Ensure end > start for visibility (min 2-minute block)
    gantt_data["end_adj"] = gantt_data.apply(
        lambda r: r["end_time"] if (r["end_time"] - r["start_time"]).total_seconds() > 60
        else r["start_time"] + pd.Timedelta(minutes=2),
        axis=1,
    )
    gantt_data["end_str"] = gantt_data["end_adj"].dt.strftime("%Y-%m-%d %H:%M")

    fig = px.timeline(
        gantt_data,
        x_start="start_time",
        x_end="end_adj",
        y="sender",
        color="location",
        hover_data=["location", "msg_count", "duration_min"],
        title="Route Timeline (site visits over time)",
        labels={"sender": "Crew"},
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        height=450,
        legend=dict(font=dict(size=9), orientation="h", yanchor="bottom",
                    y=-0.3, xanchor="center", x=0.5),
        xaxis_tickformat="%I:%M %p",
    )
    return fig


# ── Crew Metrics Callback 5: Pace Consistency ────────────────────────────────

@app.callback(Output("chart-pace-consistency", "figure"), FILTER_INPUTS)
def chart_pace_consistency(chats, senders, quality, msg_types, time_range,
                           use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    visits_df = _build_site_visits(df, "chat")
    if visits_df.empty:
        return _empty_fig("Pace Consistency")

    # For each sender-day, compute cumulative sites vs hours into shift
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    sender_list = sorted(visits_df["sender"].unique())

    for i, sender in enumerate(sender_list):
        sv = visits_df[visits_df["sender"] == sender].sort_values("start_time")
        if sv.empty:
            continue

        # Group by date, compute hours-into-shift and cumulative site count
        for d, day_visits in sv.groupby("date"):
            day_sorted = day_visits.sort_values("start_time")
            shift_start = day_sorted["start_time"].iloc[0]
            hours_in = []
            cum_sites = []
            for j, (_, row) in enumerate(day_sorted.iterrows(), 1):
                h = (row["start_time"] - shift_start).total_seconds() / 3600.0
                hours_in.append(h)
                cum_sites.append(j)

            color = colors[i % len(colors)]
            date_str = str(d)[:10] if hasattr(d, 'strftime') else str(d)[:10]
            fig.add_trace(go.Scatter(
                x=hours_in, y=cum_sites,
                mode="lines+markers",
                name=f"{sender} ({date_str})",
                line=dict(color=color),
                marker=dict(size=6),
                legendgroup=sender,
                showlegend=(d == day_sorted["date"].iloc[0]),
            ))

    fig.update_layout(
        title="Pace Consistency (cumulative sites vs hours into shift)",
        xaxis_title="Hours Into Shift",
        yaxis_title="Cumulative Sites Visited",
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
        legend=dict(font=dict(size=9)),
    )
    return fig


# ── Crew Metrics Callback 6: Most Visited Locations ──────────────────────────

@app.callback(Output("chart-top-locations", "figure"), FILTER_INPUTS)
def chart_top_locations(chats, senders, quality, msg_types, time_range,
                        use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    visits_df = _build_site_visits(df, "chat")
    if visits_df.empty:
        return _empty_fig("Most Visited Locations")

    loc_counts = (visits_df.groupby("location")
                  .agg(visits=("sender", "size"),
                       crews=("sender", "nunique"))
                  .sort_values("visits", ascending=False)
                  .head(20)
                  .reset_index())

    fig = px.bar(
        loc_counts, y="location", x="visits", orientation="h",
        text="visits",
        title="Top 20 Most Visited Locations",
        labels={"visits": "Visit Count", "location": "Location"},
        color="crews",
        color_continuous_scale="Blues",
        hover_data=["crews"],
    )
    fig.update_traces(textposition="outside", textfont_size=10)
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=10, r=60, t=40, b=10),
        height=max(400, len(loc_counts) * 25),
        coloraxis_colorbar_title="Unique<br>Crews",
    )
    return fig


# ── Deployment builder functions ──────────────────────────────────────────────


def _build_deployment_summary_data(df):
    """Build deployment summary DataFrame and per-deployment crew breakdowns.

    Returns (dep_df, crew_breakdowns) where crew_breakdowns is a dict
    mapping deployment label to its crew scorecard DataFrame.
    """
    if df.empty or df["deployment"].isna().all():
        return pd.DataFrame(), {}

    rows = []
    crew_breakdowns = {}
    for dep_label, grp in df.groupby("deployment"):
        dep_info = next(
            (d for d in ALL_DEPLOYMENTS_LIST if d["label"] == dep_label), None)
        days = dep_info["days"] if dep_info else 0
        crews = grp["chat"].nunique()
        n_msgs = len(grp)
        visits_df = _build_site_visits(grp, "chat")
        total_sites = _count_billable_routes_from_df(grp)
        ds = _build_daily_summary(grp, "chat")
        sc = _build_crew_scorecard(visits_df, ds)
        avg_sites_hr = sc["avg_sites_per_hour"].mean() if not sc.empty else 0

        if not ds.empty:
            avg_first = ds["first_hour"].mean()
            fh = int(avg_first)
            fm = int((avg_first - fh) * 60)
            period = "AM" if fh < 12 else "PM"
            dh = fh % 12 or 12
            avg_first_str = f"{dh}:{fm:02d} {period}"
            avg_window = round(ds["window_hrs"].mean(), 1)
            consistency = round(ds["window_hrs"].std(), 1) if len(ds) > 1 else 0.0
        else:
            avg_first_str = "N/A"
            avg_window = 0.0
            consistency = 0.0
        avg_trans = round(sc["avg_transition_min"].mean(), 1) if not sc.empty else 0.0
        active_per_crew = round(sc["total_active_hrs"].mean(), 1) if not sc.empty else 0.0

        rows.append({
            "Deployment": dep_label,
            "Days": days,
            "Crews Active": crews,
            "Total Messages": n_msgs,
            "Billable Routes": total_sites,
            "Avg Sites/Hr": round(avg_sites_hr, 1),
            "Avg First Report": avg_first_str,
            "Avg Window (hrs)": avg_window,
            "Avg Transition (min)": avg_trans,
            "Active Hrs/Crew": active_per_crew,
            "Consistency (std)": consistency,
        })
        if not sc.empty:
            crew_breakdowns[dep_label] = sc

    return pd.DataFrame(rows), crew_breakdowns


def _build_deployment_timeline_fig(df):
    """Build deployment timeline Gantt chart."""
    if df.empty or df["deployment"].isna().all():
        return _empty_fig("Deployment Timeline")

    dep_stats = []
    for dep_info in ALL_DEPLOYMENTS_LIST:
        dep_label = dep_info["label"]
        grp = df[df["deployment"] == dep_label]
        if grp.empty:
            continue
        dep_stats.append({
            "Deployment": dep_label,
            "Start": pd.Timestamp(dep_info["start_date"]),
            "End": pd.Timestamp(dep_info["end_date"]) + pd.Timedelta(days=1),
            "Messages": len(grp),
            "Crews": grp["chat"].nunique(),
        })

    if not dep_stats:
        return _empty_fig("Deployment Timeline")

    dep_df = pd.DataFrame(dep_stats)
    fig = px.timeline(
        dep_df,
        x_start="Start",
        x_end="End",
        y="Deployment",
        color="Deployment",
        hover_data=["Messages", "Crews"],
        title="Deployment Timeline",
        color_discrete_sequence=px.colors.qualitative.T10,
    )
    for i, row in dep_df.iterrows():
        fig.add_annotation(
            x=row["Start"] + (row["End"] - row["Start"]) / 2,
            y=row["Deployment"],
            text=f"{row['Messages']} msgs / {row['Crews']} crews",
            showarrow=False,
            font=dict(size=11, color="white"),
        )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        height=300,
        showlegend=False,
        yaxis={"categoryorder": "array",
               "categoryarray": [d["label"] for d in ALL_DEPLOYMENTS_LIST][::-1]},
    )
    return fig


def _build_deployment_first_report_fig(df):
    """Build grouped bar chart of avg first report time by deployment."""
    if df.empty or df["deployment"].isna().all():
        return _empty_fig("First Report Time by Deployment")

    dep_order = [d["label"] for d in ALL_DEPLOYMENTS_LIST]
    rows = []
    for dep_label, grp in df.groupby("deployment"):
        ds = _build_daily_summary(grp, "chat")
        if ds.empty:
            continue
        for crew, crew_grp in ds.groupby("sender"):
            rows.append({
                "Deployment": dep_label,
                "Crew": crew,
                "Avg First Hour": crew_grp["first_hour"].mean(),
            })

    if not rows:
        return _empty_fig("First Report Time by Deployment")

    fr_df = pd.DataFrame(rows)
    overall_mean = fr_df["Avg First Hour"].mean()

    fig = px.bar(
        fr_df, x="Deployment", y="Avg First Hour", color="Crew",
        barmode="group",
        title="Avg First Report Time by Deployment",
        labels={"Avg First Hour": "First Report (hour)", "Crew": "Crew"},
        color_discrete_sequence=px.colors.qualitative.T10,
    )
    fig.add_hline(
        y=overall_mean, line_dash="dash", line_color="#2c3e50",
        annotation_text=f"Overall avg {int(overall_mean)}:{int((overall_mean % 1) * 60):02d}",
        annotation_position="top left",
    )
    fig.update_layout(
        template="plotly_white",
        yaxis=dict(
            autorange="reversed",
            dtick=1,
            ticktext=[f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}"
                      for h in range(0, 25)],
            tickvals=list(range(0, 25)),
        ),
        xaxis={"categoryorder": "array", "categoryarray": dep_order},
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
        legend=dict(font=dict(size=10)),
    )
    return fig


def _build_deployment_transition_box_fig(df):
    """Build box plot of transition times by deployment."""
    if df.empty or df["deployment"].isna().all():
        return _empty_fig("Transition Time by Deployment")

    dep_order = [d["label"] for d in ALL_DEPLOYMENTS_LIST]
    all_trans = []
    for dep_label, grp in df.groupby("deployment"):
        visits_df = _build_site_visits(grp, "chat")
        if visits_df.empty:
            continue
        trans_df = _build_transitions(visits_df)
        if trans_df.empty:
            continue
        trans_df = trans_df.copy()
        trans_df["Deployment"] = dep_label
        all_trans.append(trans_df)

    if not all_trans:
        return _empty_fig("Transition Time by Deployment")

    trans_all = pd.concat(all_trans, ignore_index=True)
    overall_median = trans_all["transition_min"].median()

    fig = px.box(
        trans_all, x="Deployment", y="transition_min", points="all",
        title="Transition Time by Deployment (minutes)",
        labels={"transition_min": "Transition (min)", "Deployment": "Deployment"},
        color="Deployment",
        color_discrete_sequence=px.colors.qualitative.T10,
    )
    fig.add_hline(
        y=overall_median, line_dash="dash", line_color="#2c3e50",
        annotation_text=f"Median {overall_median:.0f} min",
        annotation_position="top left",
    )
    fig.update_layout(
        template="plotly_white",
        xaxis={"categoryorder": "array", "categoryarray": dep_order},
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
        showlegend=False,
    )
    return fig


def _build_deployment_crew_comparison_fig(df):
    """Build cross-deployment crew comparison bar chart."""
    if df.empty or df["deployment"].isna().all():
        return _empty_fig("Cross-Deployment Crew Comparison")

    rows = []
    for dep_label, grp in df.groupby("deployment"):
        visits_df = _build_site_visits(grp, "chat")
        if visits_df.empty:
            continue
        for crew, crew_visits in visits_df.groupby("sender"):
            rows.append({
                "Crew": crew,
                "Deployment": dep_label,
                "Sites": len(crew_visits),
            })

    if not rows:
        return _empty_fig("Cross-Deployment Crew Comparison")

    comp_df = pd.DataFrame(rows)
    fig = px.bar(
        comp_df, x="Crew", y="Sites", color="Deployment",
        barmode="group",
        title="Cross-Deployment Crew Comparison (Sites Visited)",
        labels={"Sites": "Sites Visited", "Crew": "Crew"},
        color_discrete_sequence=px.colors.qualitative.T10,
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
        xaxis_tickangle=-30,
        legend=dict(font=dict(size=10)),
    )
    return fig


def _build_deployment_sites_heatmap_fig(df):
    """Build deployment x sites heatmap figure."""
    title = "Deployment \u00d7 Sites Heatmap"
    if df.empty or df["deployment"].isna().all():
        return _empty_fig(title)

    visits_df = _build_site_visits(df, "chat")
    if visits_df.empty:
        return _empty_fig(title)

    visits_df = visits_df.copy()
    visits_df["deployment"] = visits_df["date"].apply(
        lambda d: _date_to_deployment.get(
            d.date() if hasattr(d, "date") else d, ""))
    visits_df = visits_df[visits_df["deployment"] != ""]

    heat = (visits_df.groupby(["deployment", "location"])
            .size().reset_index(name="visits"))
    top_locs = (heat.groupby("location")["visits"].sum()
                .sort_values(ascending=False).head(20).index.tolist())
    heat = heat[heat["location"].isin(top_locs)]

    if heat.empty:
        return _empty_fig(title)

    pivot = heat.pivot_table(index="deployment", columns="location",
                             values="visits", fill_value=0)
    dep_order = [d["label"] for d in ALL_DEPLOYMENTS_LIST
                 if d["label"] in pivot.index]
    pivot = pivot.reindex(dep_order)
    col_order = [c for c in top_locs if c in pivot.columns]
    pivot = pivot[col_order]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[loc[:30] for loc in pivot.columns],
        y=pivot.index.tolist(),
        colorscale="Tealgrn",
        text=pivot.values,
        texttemplate="%{text}",
        hovertemplate=(
            "Deployment: %{y}<br>Location: %{x}<br>"
            "Visits: %{z}<extra></extra>"),
    ))
    fig.update_layout(
        template="plotly_white",
        title=f"{title} (top 20 locations)",
        xaxis_title="Location",
        yaxis_title="Deployment",
        margin=dict(l=10, r=10, t=40, b=80),
        height=400,
        xaxis_tickangle=-45,
    )
    return fig


def _build_deployment_crew_trend_fig(df):
    """Build crew performance trend across deployments."""
    if df.empty or df["deployment"].isna().all():
        return _empty_fig("Crew Performance Trend Across Deployments")

    dep_order = [d["label"] for d in ALL_DEPLOYMENTS_LIST]

    rows = []
    for dep_label, grp in df.groupby("deployment"):
        visits_df = _build_site_visits(grp, "chat")
        if visits_df.empty:
            continue
        ds = _build_daily_summary(grp, "chat")
        sc = _build_crew_scorecard(visits_df, ds)
        if sc.empty:
            continue
        for _, crew_row in sc.iterrows():
            rows.append({
                "Crew": crew_row["sender"],
                "Deployment": dep_label,
                "Sites/Hr": crew_row["avg_sites_per_hour"],
                "Total Sites": crew_row["total_sites"],
                "Active Hrs": crew_row["total_active_hrs"],
            })

    if not rows:
        return _empty_fig("Crew Performance Trend Across Deployments")

    trend_df = pd.DataFrame(rows)
    dep_avg = (trend_df.groupby("Deployment")["Sites/Hr"]
               .mean().reset_index(name="Avg Sites/Hr"))

    trend_df["dep_idx"] = trend_df["Deployment"].map(
        {label: i for i, label in enumerate(dep_order)})
    trend_df = trend_df.sort_values("dep_idx")
    dep_avg["dep_idx"] = dep_avg["Deployment"].map(
        {label: i for i, label in enumerate(dep_order)})
    dep_avg = dep_avg.sort_values("dep_idx")

    fig = go.Figure()
    tableau = px.colors.qualitative.T10
    crews = sorted(trend_df["Crew"].unique())

    for i, crew in enumerate(crews):
        crew_data = trend_df[trend_df["Crew"] == crew]
        color = tableau[i % len(tableau)]
        fig.add_trace(go.Scatter(
            x=crew_data["Deployment"],
            y=crew_data["Sites/Hr"],
            mode="lines+markers",
            name=crew,
            line=dict(color=color, width=2),
            marker=dict(size=8, symbol="circle"),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Deployment: %{x}<br>"
                "Sites/Hr: %{y:.1f}<br>"
                "<extra></extra>"),
        ))

    fig.add_trace(go.Scatter(
        x=dep_avg["Deployment"],
        y=dep_avg["Avg Sites/Hr"],
        mode="lines+markers",
        name="Deployment Avg",
        line=dict(color="#2c3e50", width=3, dash="dash"),
        marker=dict(size=10, symbol="diamond", color="#2c3e50"),
    ))

    fig.update_layout(
        template="plotly_white",
        title="Crew Performance Trend Across Deployments (Avg Sites/Hr)",
        xaxis_title="Deployment",
        yaxis_title="Sites per Hour",
        xaxis={"categoryorder": "array", "categoryarray": dep_order},
        margin=dict(l=10, r=10, t=40, b=10),
        height=450,
        legend=dict(font=dict(size=10), orientation="h",
                    yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        hovermode="x unified",
    )
    return fig


# ── Deployment Callbacks (thin wrappers) ─────────────────────────────────────

@app.callback(Output("deployment-summary-container", "children"), FILTER_INPUTS)
def deployment_summary_table(chats, senders, quality, msg_types, time_range,
                             use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    dep_df, _ = _build_deployment_summary_data(df)
    if dep_df.empty:
        return html.P("No deployment data for current filters",
                       className="text-muted p-3")
    table = dash_table.DataTable(
        data=dep_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in dep_df.columns],
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "8px", "fontSize": "13px"},
        style_header={
            "backgroundColor": "#2c3e50", "color": "white", "fontWeight": "bold",
        },
    )
    return html.Div([html.H5("Deployment Summary", className="mb-2"), table])


@app.callback(Output("chart-deployment-timeline", "figure"), FILTER_INPUTS)
def chart_deployment_timeline(chats, senders, quality, msg_types, time_range,
                              use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    return _build_deployment_timeline_fig(df)


@app.callback(Output("chart-deployment-first-report", "figure"), FILTER_INPUTS)
def chart_deployment_first_report(chats, senders, quality, msg_types, time_range,
                                  use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    return _build_deployment_first_report_fig(df)


@app.callback(Output("chart-deployment-transition-box", "figure"), FILTER_INPUTS)
def chart_deployment_transition_box(chats, senders, quality, msg_types, time_range,
                                    use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    return _build_deployment_transition_box_fig(df)

@app.callback(Output("chart-deployment-crew-comparison", "figure"), FILTER_INPUTS)
def chart_deployment_crew_comparison(chats, senders, quality, msg_types,
                                     time_range, use_resolved, date_start,
                                     date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    return _build_deployment_crew_comparison_fig(df)


@app.callback(Output("chart-deployment-sites-heatmap", "figure"), FILTER_INPUTS)
def chart_deployment_sites_heatmap(chats, senders, quality, msg_types,
                                   time_range, use_resolved, date_start,
                                   date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    return _build_deployment_sites_heatmap_fig(df)


@app.callback(Output("chart-deployment-crew-trend", "figure"), FILTER_INPUTS)
def chart_deployment_crew_trend(chats, senders, quality, msg_types, time_range,
                                use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    return _build_deployment_crew_trend_fig(df)


# ── Deployment Download Callback ─────────────────────────────────────────────

FILTER_STATES = [
    State("filter-chats", "value"),
    State("filter-senders", "value"),
    State("filter-quality", "value"),
    State("filter-types", "value"),
    State("filter-time", "value"),
    State("filter-resolved", "value"),
    State("filter-dates", "start_date"),
    State("filter-dates", "end_date"),
    State("filter-deployment", "value"),
]


@app.callback(
    Output("download-deployment-pdf", "data"),
    Input("btn-download-deployment-pdf", "n_clicks"),
    FILTER_STATES,
    prevent_initial_call=True,
)
def download_deployment_report(n_clicks, chats, senders, quality, msg_types,
                               time_range, use_resolved, date_start, date_end,
                               deployments):
    if not n_clicks:
        return None

    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)

    # Build all figures
    dep_df, crew_breakdowns = _build_deployment_summary_data(df)
    fig_timeline = _build_deployment_timeline_fig(df)
    fig_first_report = _build_deployment_first_report_fig(df)
    fig_transition = _build_deployment_transition_box_fig(df)
    fig_crew_trend = _build_deployment_crew_trend_fig(df)
    fig_crew_comp = _build_deployment_crew_comparison_fig(df)
    fig_heatmap = _build_deployment_sites_heatmap_fig(df)

    # Convert figures to embedded HTML
    pjs = "cdn"
    chart_timeline = pio.to_html(fig_timeline, full_html=False, include_plotlyjs=pjs)
    chart_first = pio.to_html(fig_first_report, full_html=False, include_plotlyjs=False)
    chart_trans = pio.to_html(fig_transition, full_html=False, include_plotlyjs=False)
    chart_trend = pio.to_html(fig_crew_trend, full_html=False, include_plotlyjs=False)
    chart_comp = pio.to_html(fig_crew_comp, full_html=False, include_plotlyjs=False)
    chart_heat = pio.to_html(fig_heatmap, full_html=False, include_plotlyjs=False)

    # Summary table HTML
    summary_html = ""
    if not dep_df.empty:
        summary_html = dep_df.to_html(index=False, classes="summary-table",
                                       border=0)

    # Per-deployment crew breakdown tables
    breakdown_html = ""
    for dep_label, sc in crew_breakdowns.items():
        sc_display = sc.rename(columns={
            "sender": "Crew",
            "days_active": "Days",
            "total_sites": "Sites",
            "avg_sites_per_day": "Sites/Day",
            "avg_sites_per_hour": "Sites/Hr",
            "avg_transition_min": "Transition (min)",
            "total_active_hrs": "Active Hrs",
        })
        breakdown_html += f"<h3>{dep_label}</h3>\n"
        breakdown_html += sc_display.to_html(index=False, classes="breakdown-table",
                                              border=0)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    filename = f"deployment_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Deployment Report — {now_str}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 2em; color: #2c3e50; }}
  h1 {{ color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 0.3em; }}
  h2 {{ color: #34495e; margin-top: 2em; }}
  h3 {{ color: #7f8c8d; }}
  .summary-table, .breakdown-table {{
    border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 13px;
  }}
  .summary-table th, .breakdown-table th {{
    background: #2c3e50; color: white; padding: 8px 12px; text-align: left;
  }}
  .summary-table td, .breakdown-table td {{
    padding: 6px 12px; border-bottom: 1px solid #ecf0f1;
  }}
  .summary-table tr:nth-child(even), .breakdown-table tr:nth-child(even) {{
    background: #f8f9fa;
  }}
  .chart-row {{ display: flex; gap: 1em; margin: 1em 0; }}
  .chart-row > div {{ flex: 1; }}
  @media print {{
    body {{ margin: 0.5em; font-size: 11px; }}
    .chart-row {{ page-break-inside: avoid; }}
    h2 {{ page-break-before: auto; }}
    .plotly-graph-div {{ max-height: 350px !important; }}
    .modebar {{ display: none !important; }}
  }}
</style>
</head>
<body>
<h1>Deployment Report</h1>
<p>Generated: {now_str}</p>

<h2>Summary</h2>
{summary_html}

<h2>Deployment Timeline</h2>
{chart_timeline}

<div class="chart-row">
  <div>
    <h2>First Report Time</h2>
    {chart_first}
  </div>
  <div>
    <h2>Transition Time</h2>
    {chart_trans}
  </div>
</div>

<h2>Crew Performance Trend</h2>
{chart_trend}

<div class="chart-row">
  <div>
    <h2>Crew Comparison</h2>
    {chart_comp}
  </div>
  <div>
    <h2>Sites Heatmap</h2>
    {chart_heat}
  </div>
</div>

<h2>Per-Deployment Crew Breakdown</h2>
{breakdown_html}

</body>
</html>"""

    return dcc.send_bytes(html_content.encode("utf-8"), filename)


# ── KPI Bar Callback ─────────────────────────────────────────────────────────

@app.callback(Output("kpi-bar", "children"), FILTER_INPUTS)
def update_kpi_cards(chats, senders, quality, msg_types, time_range,
                     use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    scol = _sender_col(use_resolved)

    total_msgs = len(df)
    active_crews = df["chat"].nunique()

    visits_df = _build_site_visits(df, "chat")
    ds = _build_daily_summary(df, scol)

    total_sites = _count_billable_routes_from_df(df)
    avg_sites_hr = 0.0
    avg_trans = 0.0
    if not visits_df.empty:
        sc = _build_crew_scorecard(visits_df, ds)
        if not sc.empty:
            avg_sites_hr = sc["avg_sites_per_hour"].mean()
            avg_trans = sc["avg_transition_min"].mean()

    avg_first_str = "N/A"
    if not ds.empty:
        avg_first = ds["first_hour"].mean()
        fh = int(avg_first)
        fm = int((avg_first - fh) * 60)
        period = "AM" if fh < 12 else "PM"
        dh = fh % 12 or 12
        avg_first_str = f"{dh}:{fm:02d} {period}"

    kpi_data = [
        ("Total Messages", f"{total_msgs:,}", "#3498db"),
        ("Active Crews", str(active_crews), "#2ecc71"),
        ("Avg Sites/Hour", f"{avg_sites_hr:.1f}", "#e67e22"),
        ("Avg First Report", avg_first_str, "#9b59b6"),
        ("Billable Routes", f"{total_sites:,}", "#1abc9c"),
        ("Avg Transition", f"{avg_trans:.1f} min", "#e74c3c"),
    ]

    cards = []
    for label, value, color in kpi_data:
        cards.append(dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H3(value, className="mb-0 fw-bold text-center",
                            style={"fontSize": "1.5rem"}),
                    html.P(label, className="text-muted text-center mb-0",
                           style={"fontSize": "12px"}),
                ], className="py-2 px-2"),
                style={"borderTop": f"3px solid {color}"},
                className="shadow-sm",
            ),
            md=2, sm=4, xs=6, className="mb-2",
        ))

    return dbc.Row(cards, className="g-2")


# ── Productivity: Daily Productivity Score ───────────────────────────────────

def _build_daily_productivity_score(df, scol):
    ds = _build_daily_summary(df, scol)
    visits_df = _build_site_visits(df, scol)
    if ds.empty or visits_df.empty:
        return pd.DataFrame()

    site_counts = (visits_df.groupby(["sender", "date"])
                   .size().reset_index(name="sites_visited"))
    site_counts["date"] = pd.to_datetime(site_counts["date"])

    active_hrs = []
    for (sender, d), day_grp in visits_df.groupby(
            ["sender", visits_df["date"].dt.date]):
        hrs = _compute_active_hours(day_grp)
        active_hrs.append({
            "sender": sender,
            "date": pd.Timestamp(d),
            "active_hrs": hrs,
        })
    active_df = pd.DataFrame(active_hrs) if active_hrs else pd.DataFrame(
        columns=["sender", "date", "active_hrs"])

    ds["date"] = pd.to_datetime(ds["date"])
    merged = ds.merge(site_counts, on=["sender", "date"], how="left")
    merged["sites_visited"] = merged["sites_visited"].fillna(0).astype(int)

    if not active_df.empty:
        merged = merged.merge(active_df, on=["sender", "date"], how="left")
        merged["active_hrs"] = merged["active_hrs"].fillna(1 / 60)
    else:
        merged["active_hrs"] = 1 / 60

    merged["sites_per_hour"] = merged["sites_visited"] / merged["active_hrs"]

    merged["punctuality_score"] = merged["first_hour"].apply(
        lambda h: max(0, min(100, 100 - (h - 7) * 50)) if h >= 7 else 100
    )

    max_sph = merged["sites_per_hour"].max()
    if max_sph > 0:
        merged["pace_score"] = (merged["sites_per_hour"] / max_sph * 100).clip(0, 100)
    else:
        merged["pace_score"] = 0.0

    avg_sites = merged["sites_visited"].mean()
    if avg_sites > 0:
        merged["coverage"] = (merged["sites_visited"] / avg_sites * 100).clip(0, 100)
    else:
        merged["coverage"] = 0.0

    merged["productivity_score"] = (
        0.5 * merged["pace_score"]
        + 0.3 * merged["punctuality_score"]
        + 0.2 * merged["coverage"]
    ).round(1)

    result = merged[["sender", "date", "sites_visited", "sites_per_hour",
                      "first_hour", "punctuality_score", "pace_score",
                      "coverage", "productivity_score"]].copy()
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    result["sites_per_hour"] = result["sites_per_hour"].round(1)
    result["first_hour"] = result["first_hour"].round(2)
    result["punctuality_score"] = result["punctuality_score"].round(0).astype(int)
    result["pace_score"] = result["pace_score"].round(0).astype(int)
    return result


@app.callback(Output("productivity-score-container", "children"), FILTER_INPUTS)
def update_productivity_score(chats, senders, quality, msg_types, time_range,
                              use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    scol = _sender_col(use_resolved)
    ps = _build_daily_productivity_score(df, scol)
    if ps.empty:
        return html.P("No productivity data for current filters",
                       className="text-muted p-3")

    ps_display = ps.rename(columns={
        "sender": "Sender", "date": "Date", "sites_visited": "Sites",
        "sites_per_hour": "Sites/Hr", "first_hour": "First Hour",
        "punctuality_score": "Punctuality", "pace_score": "Pace",
        "productivity_score": "Score",
    })

    table = dash_table.DataTable(
        data=ps_display.to_dict("records"),
        columns=[{"name": c, "id": c} for c in ps_display.columns],
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        page_size=20,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "8px", "fontSize": "13px"},
        style_header={
            "backgroundColor": "#2c3e50", "color": "white", "fontWeight": "bold",
        },
        style_data_conditional=[
            {"if": {"filter_query": "{Score} >= 70"},
             "backgroundColor": "#eafaf1"},
            {"if": {"filter_query": "{Score} >= 40 && {Score} < 70"},
             "backgroundColor": "#fef9e7"},
            {"if": {"filter_query": "{Score} < 40"},
             "backgroundColor": "#fde8e8", "color": "#c0392b"},
        ],
    )
    return html.Div([
        html.H5("Daily Productivity Score", className="mb-2"),
        table,
    ])


# ── Productivity: Crew Leaderboard ───────────────────────────────────────────

@app.callback(Output("crew-leaderboard-container", "children"), FILTER_INPUTS)
def update_crew_leaderboard(chats, senders, quality, msg_types, time_range,
                            use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    scol = _sender_col(use_resolved)
    ps = _build_daily_productivity_score(df, scol)
    if ps.empty:
        return html.P("No leaderboard data for current filters",
                       className="text-muted p-3")

    agg = ps.groupby("sender").agg(
        avg_score=("productivity_score", "mean"),
        best_day=("productivity_score", "max"),
        days=("date", "count"),
    ).reset_index()

    if len(ps["date"].unique()) >= 2:
        dates_sorted = sorted(ps["date"].unique())
        mid = len(dates_sorted) // 2
        early_dates = set(dates_sorted[:mid])
        late_dates = set(dates_sorted[mid:])
        trends = []
        for sender in agg["sender"]:
            s_data = ps[ps["sender"] == sender]
            early_avg = s_data[s_data["date"].isin(early_dates)]["productivity_score"].mean()
            late_avg = s_data[s_data["date"].isin(late_dates)]["productivity_score"].mean()
            if pd.isna(early_avg) or pd.isna(late_avg):
                trends.append("─ stable")
            elif late_avg - early_avg > 3:
                trends.append("↑ improving")
            elif early_avg - late_avg > 3:
                trends.append("↓ declining")
            else:
                trends.append("─ stable")
        agg["trend"] = trends
    else:
        agg["trend"] = "─ stable"

    agg = agg.sort_values("avg_score", ascending=False).reset_index(drop=True)
    agg.insert(0, "rank", range(1, len(agg) + 1))
    agg["avg_score"] = agg["avg_score"].round(1)
    agg["best_day"] = agg["best_day"].round(1)

    lb_display = agg.rename(columns={
        "rank": "Rank", "sender": "Crew", "avg_score": "Avg Score",
        "best_day": "Best Day", "trend": "Trend", "days": "Days",
    })

    table = dash_table.DataTable(
        data=lb_display.to_dict("records"),
        columns=[{"name": c, "id": c} for c in lb_display.columns],
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "8px", "fontSize": "13px"},
        style_header={
            "backgroundColor": "#2c3e50", "color": "white", "fontWeight": "bold",
        },
        style_data_conditional=[
            {"if": {"filter_query": "{Avg Score} >= 70"},
             "backgroundColor": "#eafaf1"},
            {"if": {"filter_query": "{Avg Score} >= 40 && {Avg Score} < 70"},
             "backgroundColor": "#fef9e7"},
            {"if": {"filter_query": "{Avg Score} < 40"},
             "backgroundColor": "#fde8e8", "color": "#c0392b"},
        ],
    )
    return html.Div([
        html.H5("Crew Leaderboard", className="mb-2"),
        table,
    ])


# ── Productivity: Idle Time Detection ────────────────────────────────────────

def _build_idle_gaps(df, scol, threshold_min=45):
    df_valid = df[(df[scol] != "") & df["time"].notna() & df["msg_date"].notna()].copy()
    if df_valid.empty:
        return pd.DataFrame()

    df_valid = df_valid.sort_values([scol, "msg_date", "time"])
    rows = []
    for sender, grp in df_valid.groupby(scol):
        for d, day_grp in grp.groupby(grp["msg_date"].dt.date):
            times = day_grp["time"].sort_values()
            if len(times) < 2:
                continue
            diffs = times.diff().dropna()
            for idx, gap in diffs.items():
                gap_min = gap.total_seconds() / 60.0
                if gap_min >= threshold_min:
                    gap_end_time = times.loc[idx]
                    gap_start_time = gap_end_time - gap
                    rows.append({
                        "sender": sender,
                        "date": str(d),
                        "gap_start": gap_start_time.strftime("%I:%M %p"),
                        "gap_end": gap_end_time.strftime("%I:%M %p"),
                        "gap_minutes": round(gap_min, 1),
                    })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["sender", "date", "gap_start", "gap_end", "gap_minutes"])


@app.callback(Output("chart-idle-gaps", "figure"), FILTER_INPUTS)
def chart_idle_gaps(chats, senders, quality, msg_types, time_range,
                    use_resolved, date_start, date_end, deployments):
    df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                 date_start, date_end, deployments)
    scol = _sender_col(use_resolved)
    gaps = _build_idle_gaps(df, scol)
    if gaps.empty:
        return _empty_fig("Idle Time Detection")

    incident_counts = gaps.groupby("sender").agg(
        incidents=("gap_minutes", "size"),
        avg_gap=("gap_minutes", "mean"),
        max_gap=("gap_minutes", "max"),
    ).reset_index().sort_values("incidents")

    fig = px.bar(
        incident_counts, y="sender", x="incidents", orientation="h",
        text="incidents",
        title="Idle Time Incidents per Crew (gaps > 45 min)",
        labels={"incidents": "Idle Incidents", "sender": "Crew"},
        color="avg_gap",
        color_continuous_scale="OrRd",
        hover_data={"avg_gap": ":.1f", "max_gap": ":.1f"},
    )
    fig.update_traces(textposition="outside", textfont_size=11)
    fig.update_layout(
        margin=dict(l=10, r=60, t=40, b=10),
        height=400,
        yaxis={"categoryorder": "total ascending"},
        coloraxis_colorbar_title="Avg Gap<br>(min)",
    )
    return fig


# ── Operations Tab Callbacks ──────────────────────────────────────────────────

@app.callback(Output("chart-routing-gantt", "figure"), FILTER_INPUTS)
def chart_routing_gantt(chats, senders, quality, msg_types, time_range,
                        use_resolved, date_start, date_end, deployments):
    try:
        df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                     date_start, date_end, deployments)
        job_logs = build_job_logs(df, SNOW_CONFIG)
        if job_logs.empty:
            return _empty_fig("Crew Routing Timeline")
        loc_coords = build_location_coords(SNOW_CONFIG.get("location_registry", []))
        segments = build_route_segments(job_logs, SNOW_CONFIG, loc_coords)
        if segments.empty:
            return _empty_fig("Crew Routing Timeline")
        gantt_data = []
        for _, seg in segments.iterrows():
            t0 = seg["start_time"]
            t1 = seg["end_time"]
            if pd.isna(t0) or pd.isna(t1):
                continue
            dist = seg.get("distance_km", np.nan) if "distance_km" in segments.columns else np.nan
            dur = seg.get("actual_duration_mins", np.nan)
            eff = seg.get("travel_efficiency", np.nan) if "travel_efficiency" in segments.columns else np.nan
            hover_extra = ""
            if pd.notna(dist):
                hover_extra += f" | {dist:.1f}km"
            if pd.notna(dur):
                hover_extra += f" | {dur:.0f}min"
            if pd.notna(eff):
                hover_extra += f" | eff:{eff:.0f}%"
            gantt_data.append({
                "Crew": seg["crew"],
                "Start": t0,
                "Finish": t1,
                "Location": seg["destination_location"] or "Unknown",
                "Details": f"{seg['origin_location']} → {seg['destination_location']}{hover_extra}",
            })
        if not gantt_data:
            return _empty_fig("Crew Routing Timeline")
        gantt_df = pd.DataFrame(gantt_data)
        fig = px.timeline(
            gantt_df, x_start="Start", x_end="Finish", y="Crew", color="Location",
            title="Crew Routing Timeline",
            hover_data=["Details"] if "Details" in gantt_df.columns else None,
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            height=450,
            yaxis={"categoryorder": "category ascending"},
        )
        return fig
    except Exception as e:
        print(f"[chart-routing-gantt] Error: {e}")
        return _empty_fig("Crew Routing Timeline")


@app.callback(Output("chart-burndown", "figure"), FILTER_INPUTS)
def chart_burndown(chats, senders, quality, msg_types, time_range,
                   use_resolved, date_start, date_end, deployments):
    try:
        df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                     date_start, date_end, deployments)
        job_logs = build_job_logs(df, SNOW_CONFIG)
        burndown = build_deployment_burndown(job_logs, ALL_DEPLOYMENTS_LIST, SNOW_CONFIG)
        if burndown.empty:
            return _empty_fig("Deployment Burn-Down")
        fig = go.Figure()
        for dep_label in burndown["deployment"].unique():
            dep_data = burndown[burndown["deployment"] == dep_label]
            crew_count = int(dep_data["crew_count"].iloc[0]) if "crew_count" in dep_data.columns and not dep_data.empty else 1
            total_sites = int(dep_data["cumulative_completed"].max()) if not dep_data.empty else 0
            base_hrs = SNOW_CONFIG.get("expected_deployment_hours", 12.0)
            adj_hrs = base_hrs / max(crew_count, 1)
            fig.add_trace(go.Scatter(
                x=dep_data["timestamp"], y=dep_data["cumulative_completed"],
                mode="lines+markers",
                name=f"{dep_label} actual ({crew_count} crews, {total_sites} sites)",
                line=dict(dash="solid"),
            ))
            fig.add_trace(go.Scatter(
                x=dep_data["timestamp"], y=dep_data["expected_completed"],
                mode="lines",
                name=f"{dep_label} expected ({adj_hrs:.1f}h adj.)",
                line=dict(dash="dash"),
            ))
        fig.update_layout(
            title="Deployment Burn-Down: Actual vs Expected (crew-adjusted)",
            xaxis_title="Time", yaxis_title="Cumulative Sites Completed",
            margin=dict(l=10, r=10, t=40, b=10),
            height=450,
            legend=dict(font=dict(size=11)),
        )
        return fig
    except Exception as e:
        print(f"[chart-burndown] Error: {e}")
        return _empty_fig("Deployment Burn-Down")


@app.callback(Output("location-type-stats-container", "children"), FILTER_INPUTS)
def location_type_stats(chats, senders, quality, msg_types, time_range,
                        use_resolved, date_start, date_end, deployments):
    try:
        df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                     date_start, date_end, deployments)
        job_logs = build_job_logs(df, SNOW_CONFIG)
        stats = build_location_type_stats(job_logs)
        if stats.empty:
            return html.P("No location type data available.", className="text-muted p-3")
        fig = px.bar(
            stats, x="location_type", y="avg_duration_min",
            text="avg_duration_min", color="location_type",
            title="Avg Service Time by Location Type",
            labels={"avg_duration_min": "Avg Duration (min)", "location_type": "Type"},
        )
        fig.update_traces(textposition="outside", texttemplate="%{text:.1f}")
        fig.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            height=350, showlegend=False,
        )
        return dcc.Graph(figure=fig)
    except Exception as e:
        print(f"[location-type-stats] Error: {e}")
        return html.P("Error loading location type stats.", className="text-muted p-3")


@app.callback(Output("traffic-analysis-container", "children"), FILTER_INPUTS)
def traffic_analysis(chats, senders, quality, msg_types, time_range,
                     use_resolved, date_start, date_end, deployments):
    try:
        df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                     date_start, date_end, deployments)
        job_logs = build_job_logs(df, SNOW_CONFIG)
        loc_coords = build_location_coords(SNOW_CONFIG.get("location_registry", []))
        segments = build_route_segments(job_logs, SNOW_CONFIG, loc_coords)
        traffic = build_traffic_analysis(segments)
        if traffic.empty:
            return html.P("No traffic data available.", className="text-muted p-3")

        col_labels = {
            "origin": "From", "destination": "To",
            "avg_travel_min": "Avg Actual (min)", "count": "Trips",
            "std_travel_min": "Std Dev", "distance_km": "Distance (km)",
            "est_travel_min": "Est. Travel (min)", "avg_efficiency": "Efficiency %",
        }
        display_cols = [c for c in traffic.columns if not traffic[c].isna().all()]
        table_cols = [{"name": col_labels.get(c, c), "id": c} for c in display_cols]

        children = [
            html.H6("Travel Time vs Distance Between Locations", className="mb-2"),
            dash_table.DataTable(
                data=traffic.to_dict("records"),
                columns=table_cols,
                style_table={"overflowX": "auto", "maxHeight": "350px", "overflowY": "auto"},
                style_cell={"textAlign": "left", "padding": "6px", "fontSize": "13px"},
                style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
                style_data_conditional=[
                    {"if": {"filter_query": "{avg_efficiency} < 50", "column_id": "avg_efficiency"},
                     "color": "#e74c3c", "fontWeight": "bold"},
                    {"if": {"filter_query": "{avg_efficiency} >= 80", "column_id": "avg_efficiency"},
                     "color": "#27ae60", "fontWeight": "bold"},
                ],
                page_size=15,
                sort_action="native",
            ),
        ]

        has_dist = "distance_km" in segments.columns
        dist_segs = segments.dropna(subset=["distance_km", "actual_duration_mins"]) if has_dist else pd.DataFrame()
        if not dist_segs.empty and len(dist_segs) >= 3:
            fig_scatter = px.scatter(
                dist_segs,
                x="distance_km",
                y="actual_duration_mins",
                color="crew",
                hover_data=["origin_location", "destination_location"],
                labels={"distance_km": "Distance (km)", "actual_duration_mins": "Actual Travel (min)", "crew": "Crew"},
                title="Actual Travel Time vs Distance",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            max_dist = dist_segs["distance_km"].max()
            expected_x = [0, max_dist]
            expected_y = [0, max_dist / 25.0 * 60]
            fig_scatter.add_trace(go.Scatter(
                x=expected_x, y=expected_y,
                mode="lines", name="Expected (25 km/h)",
                line=dict(dash="dash", color="gray", width=2),
            ))
            fig_scatter.update_layout(height=400, template="plotly_white")
            children.append(dcc.Graph(figure=fig_scatter))

        return html.Div(children)
    except Exception as e:
        print(f"[traffic-analysis] Error: {e}")
        return html.P("Error loading traffic analysis.", className="text-muted p-3")


@app.callback(Output("delay-report-container", "children"), FILTER_INPUTS)
def delay_report(chats, senders, quality, msg_types, time_range,
                 use_resolved, date_start, date_end, deployments):
    try:
        df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                     date_start, date_end, deployments)
        job_logs = build_job_logs(df, SNOW_CONFIG)
        loc_coords = build_location_coords(SNOW_CONFIG.get("location_registry", []))
        segments = build_route_segments(job_logs, SNOW_CONFIG, loc_coords)
        delays = build_delay_report(segments, SNOW_CONFIG)
        if delays.empty:
            return html.P("No delayed segments found.", className="text-muted p-3")
        display = delays.copy()
        if "date" in display.columns:
            display["date"] = display["date"].astype(str)
        return html.Div([
            html.H6("Delay Report — Segments Exceeding Expected Time", className="mb-2"),
            dash_table.DataTable(
                data=display.to_dict("records"),
                columns=[{"name": c, "id": c} for c in display.columns],
                style_table={"overflowX": "auto", "maxHeight": "350px", "overflowY": "auto"},
                style_cell={"textAlign": "left", "padding": "6px", "fontSize": "13px"},
                style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
                page_size=15,
            ),
        ])
    except Exception as e:
        print(f"[delay-report] Error: {e}")
        return html.P("Error loading delay report.", className="text-muted p-3")


@app.callback(Output("recall-summary-container", "children"), FILTER_INPUTS)
def recall_summary(chats, senders, quality, msg_types, time_range,
                   use_resolved, date_start, date_end, deployments):
    try:
        df = _get_df(chats, senders, quality, msg_types, time_range, use_resolved,
                     date_start, date_end, deployments)
        job_logs = build_job_logs(df, SNOW_CONFIG)
        if job_logs.empty or not job_logs["is_recall"].any():
            return html.P("No recalls found.", className="text-muted p-3")
        recalls = job_logs[job_logs["is_recall"]].copy()
        total_recalls = len(recalls)
        total_added = recalls["recall_added_time_mins"].sum()
        crew_recalls = recalls.groupby("crew").agg(
            recall_count=("is_recall", "sum"),
            total_added_min=("recall_added_time_mins", "sum"),
        ).reset_index()
        crew_recalls["total_added_min"] = crew_recalls["total_added_min"].round(1)
        loc_recalls = recalls.groupby("location")["is_recall"].count().reset_index()
        loc_recalls.columns = ["location", "recall_count"]
        loc_recalls = loc_recalls.sort_values("recall_count", ascending=False).head(10)
        return html.Div([
            html.H6("Recall Summary", className="mb-2"),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H3(str(total_recalls), className="text-primary"),
                    html.Small("Total Recalls"),
                ]), className="text-center"), md=3),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H3(f"{total_added:.0f}" if pd.notna(total_added) else "N/A", className="text-warning"),
                    html.Small("Total Added Minutes"),
                ]), className="text-center"), md=3),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H3(str(crew_recalls["crew"].nunique()), className="text-info"),
                    html.Small("Crews with Recalls"),
                ]), className="text-center"), md=3),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.P("Recalls by Crew", className="fw-bold mb-1"),
                    dash_table.DataTable(
                        data=crew_recalls.to_dict("records"),
                        columns=[{"name": c, "id": c} for c in crew_recalls.columns],
                        style_cell={"textAlign": "left", "padding": "6px", "fontSize": "13px"},
                        style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
                        page_size=10,
                    ),
                ], md=6),
                dbc.Col([
                    html.P("Top Recalled Locations", className="fw-bold mb-1"),
                    dash_table.DataTable(
                        data=loc_recalls.to_dict("records"),
                        columns=[{"name": c, "id": c} for c in loc_recalls.columns],
                        style_cell={"textAlign": "left", "padding": "6px", "fontSize": "13px"},
                        style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa"},
                        page_size=10,
                    ),
                ], md=6),
            ]),
        ])
    except Exception as e:
        print(f"[recall-summary] Error: {e}")
        return html.P("Error loading recall summary.", className="text-muted p-3")


# ── Settings Tab Callbacks ───────────────────────────────────────────────────

@app.callback(
    Output("settings-crew-types-container", "children"),
    Input("main-tabs", "active_tab"),
)
def render_crew_type_settings(active_tab):
    if active_tab != "tab-settings":
        raise PreventUpdate
    loc_types = SNOW_CONFIG.get("location_types", {})
    rows = []
    for chat in ALL_CHATS:
        current = loc_types.get(chat, "")
        auto = infer_location_type(chat)
        auto_label = f"Auto-detect ({auto})"
        rows.append(
            dbc.Row([
                dbc.Col(html.Span(chat, style={"fontSize": "13px"}), md=6),
                dbc.Col(dcc.Dropdown(
                    id={"type": "crew-type-dd", "chat": chat},
                    options=[
                        {"label": auto_label, "value": ""},
                        {"label": "Sidewalk", "value": "Sidewalk"},
                        {"label": "Parking Lot", "value": "Parking Lot"},
                    ],
                    value=current,
                    clearable=False,
                    style={"fontSize": "13px"},
                ), md=6),
            ], className="mb-1 align-items-center")
        )
    return html.Div(rows, style={"maxHeight": "500px", "overflowY": "auto"})


@app.callback(
    Output("settings-non-trackable-container", "children"),
    Input("main-tabs", "active_tab"),
)
def render_non_trackable_settings(active_tab):
    if active_tab != "tab-settings":
        raise PreventUpdate
    non_trackable = SNOW_CONFIG.get("non_trackable_senders", [])
    items = []
    for sender in ALL_SENDERS_RESOLVED_FULL:
        items.append(
            dbc.Checkbox(
                id={"type": "non-track-cb", "sender": sender},
                label=sender,
                value=sender in non_trackable,
                className="mb-1",
                style={"fontSize": "13px"},
            )
        )
    return html.Div(items, style={"maxHeight": "500px", "overflowY": "auto"})


@app.callback(
    Output("settings-save-status", "children"),
    Input("settings-save-btn", "n_clicks"),
    [State({"type": "crew-type-dd", "chat": ALL}, "value"),
     State({"type": "crew-type-dd", "chat": ALL}, "id"),
     State({"type": "non-track-cb", "sender": ALL}, "value"),
     State({"type": "non-track-cb", "sender": ALL}, "id"),
     State("settings-expected-hours", "value"),
     State("settings-svc-sidewalk", "value"),
     State("settings-svc-parking", "value")],
    prevent_initial_call=True,
)
def save_settings(n_clicks, crew_values, crew_ids, track_values, track_ids,
                  expected_hours, svc_sidewalk, svc_parking):
    global SNOW_CONFIG
    if not n_clicks:
        raise PreventUpdate

    loc_types = {}
    if crew_ids and crew_values:
        for cid, val in zip(crew_ids, crew_values):
            if val:
                loc_types[cid["chat"]] = val

    non_trackable = []
    if track_ids and track_values:
        for tid, val in zip(track_ids, track_values):
            if val:
                non_trackable.append(tid["sender"])

    SNOW_CONFIG["location_types"] = loc_types
    SNOW_CONFIG["non_trackable_senders"] = non_trackable
    if expected_hours is not None:
        SNOW_CONFIG["expected_deployment_hours"] = float(expected_hours)
    svc = SNOW_CONFIG.get("expected_service_times", {})
    if svc_sidewalk is not None:
        svc["Sidewalk"] = int(svc_sidewalk)
    if svc_parking is not None:
        svc["Parking Lot"] = int(svc_parking)
    SNOW_CONFIG["expected_service_times"] = svc

    config_path = os.path.join(DATA_DIR, "config", "snow_removal.json")
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(SNOW_CONFIG, f, indent=2)
        return html.Span("Settings saved!", style={"color": "green", "fontWeight": "bold"})
    except Exception as e:
        return html.Span(f"Error saving: {e}", style={"color": "red"})


# ── Crew Merge Callbacks ─────────────────────────────────────────────────────

@app.callback(
    [Output("merge-primary-crew", "options"),
     Output("merge-secondary-crew", "options"),
     Output("merge-list-container", "children")],
    Input("main-tabs", "active_tab"),
    Input("merge-status", "children"),
)
def render_crew_merge_ui(active_tab, _status_trigger):
    if active_tab != "tab-settings":
        raise PreventUpdate

    all_chats_full = sorted(DF_ALL["chat"].unique()) if not DF_ALL.empty else []
    options = [{"label": c, "value": c} for c in all_chats_full]

    merges = SNOW_CONFIG.get("crew_merges", {})
    if merges:
        rows = []
        for secondary, primary in merges.items():
            rows.append(
                dbc.Row([
                    dbc.Col(html.Span(f"{secondary}", style={"fontSize": "13px"}), md=5),
                    dbc.Col(html.Span(f" \u2192 {primary}", style={"fontSize": "13px", "fontWeight": "bold"}), md=5),
                    dbc.Col(
                        dbc.Button("\u00d7", id={"type": "unmerge-btn", "crew": secondary},
                                   color="danger", size="sm", outline=True, n_clicks=0),
                        md=2),
                ], className="mb-1 align-items-center")
            )
        merge_list = html.Div([
            html.H6("Active Merges", className="mt-2 mb-1"),
            html.Div(rows, style={"maxHeight": "200px", "overflowY": "auto"}),
        ])
    else:
        merge_list = html.Div()

    return options, options, merge_list


@app.callback(
    Output("merge-status", "children", allow_duplicate=True),
    Input("merge-crews-btn", "n_clicks"),
    State("merge-primary-crew", "value"),
    State("merge-secondary-crew", "value"),
    prevent_initial_call=True,
)
def merge_crews(n_clicks, primary, secondary):
    global SNOW_CONFIG, DF_ALL
    if not n_clicks or not primary or not secondary:
        raise PreventUpdate
    if primary == secondary:
        return html.Span("Primary and secondary crews cannot be the same.", style={"color": "red"})

    merges = SNOW_CONFIG.get("crew_merges", {})
    merges[secondary] = primary
    SNOW_CONFIG["crew_merges"] = merges

    config_path = os.path.join(DATA_DIR, "config", "snow_removal.json")
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(SNOW_CONFIG, f, indent=2)
    except Exception as e:
        return html.Span(f"Error saving: {e}", style={"color": "red"})

    if not DF_ALL.empty:
        DF_ALL.loc[DF_ALL["chat"] == secondary, "chat"] = primary
    _reload_global_data()

    return html.Span(
        f"Merged \"{secondary}\" into \"{primary}\". Data reloaded.",
        style={"color": "green", "fontWeight": "bold"}
    )


@app.callback(
    Output("merge-status", "children", allow_duplicate=True),
    Input({"type": "unmerge-btn", "crew": ALL}, "n_clicks"),
    State({"type": "unmerge-btn", "crew": ALL}, "id"),
    prevent_initial_call=True,
)
def unmerge_crew(n_clicks_list, id_list):
    global SNOW_CONFIG
    if not n_clicks_list or not any(n_clicks_list):
        raise PreventUpdate

    clicked_idx = None
    for i, n in enumerate(n_clicks_list):
        if n and n > 0:
            clicked_idx = i
            break
    if clicked_idx is None:
        raise PreventUpdate

    secondary = id_list[clicked_idx]["crew"]
    merges = SNOW_CONFIG.get("crew_merges", {})
    if secondary in merges:
        removed_primary = merges.pop(secondary)
        SNOW_CONFIG["crew_merges"] = merges

        config_path = os.path.join(DATA_DIR, "config", "snow_removal.json")
        try:
            with open(config_path, "w") as f:
                json.dump(SNOW_CONFIG, f, indent=2)
        except Exception:
            pass

        _reload_global_data()
        return html.Span(
            f"Removed merge: \"{secondary}\" is now separate from \"{removed_primary}\". Data reloaded.",
            style={"color": "green", "fontWeight": "bold"}
        )
    raise PreventUpdate


# ── Location CSV upload callback ──────────────────────────────────────────────

@app.callback(
    [Output("locations-registry-table", "data"),
     Output("locations-upload-status", "children")],
    Input("upload-locations-csv", "contents"),
    State("upload-locations-csv", "filename"),
    prevent_initial_call=True,
)
def upload_locations_csv(contents, filename):
    global SNOW_CONFIG
    if contents is None:
        raise PreventUpdate
    try:
        import io
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string).decode("utf-8")
        csv_df = pd.read_csv(io.StringIO(decoded))
        csv_df.columns = [c.strip() for c in csv_df.columns]

        is_routes_format = "Building Name" in csv_df.columns
        if is_routes_format:
            name_col = "Building Name"
            addr_col = "Billing Street"
        else:
            name_col = "location_name" if "location_name" in csv_df.columns else None
            addr_col = "address" if "address" in csv_df.columns else None

        if name_col is None or name_col not in csv_df.columns:
            return no_update, html.Span(
                "CSV must have 'Building Name' or 'location_name' column.", style={"color": "red"})

        known_locations = sorted(
            DF_ALL[DF_ALL["location"].str.len() > 0]["location"].unique()
        ) if not DF_ALL.empty else []
        registry = []
        for _, row in csv_df.iterrows():
            name = str(row.get(name_col, "")).strip()
            if not name:
                continue
            address = str(row.get(addr_col, "")).strip() if addr_col and addr_col in csv_df.columns else ""
            if address.lower() == "nan":
                address = ""

            crew_sw = ""
            crew_pl = ""
            if is_routes_format:
                crew_sw = str(row.get("Crew Sidewalk", "")).strip()
                crew_pl = str(row.get("Crew Parking Lot", "")).strip()
            else:
                crew_sw = str(row.get("crew_sidewalk", "")).strip() if "crew_sidewalk" in csv_df.columns else ""
                crew_pl = str(row.get("crew_parking_lot", "")).strip() if "crew_parking_lot" in csv_df.columns else ""
            if crew_sw.lower() == "nan":
                crew_sw = ""
            if crew_pl.lower() == "nan":
                crew_pl = ""

            lat = row.get("lat", None) if "lat" in csv_df.columns else None
            lon = row.get("lon", None) if "lon" in csv_df.columns else None
            if pd.isna(lat) if lat is not None else True:
                lat = None
            if pd.isna(lon) if lon is not None else True:
                lon = None

            matched, score = _fuzzy_match_location(name, known_locations)
            if not matched and address:
                matched, score = _fuzzy_match_location(address, known_locations)
            match_display = f"{matched} ({score}%)" if matched else ""
            registry.append({
                "location_name": name,
                "address": address,
                "lat": float(lat) if lat is not None else "",
                "lon": float(lon) if lon is not None else "",
                "matched_chat_location": match_display,
                "crew_sidewalk": crew_sw,
                "crew_parking_lot": crew_pl,
                "_matched_raw": matched,
                "_match_score": score,
            })
        registry, auto_count = _auto_assign_crews(registry, DF_ALL)
        SNOW_CONFIG["location_registry"] = registry
        table_data = [
            {k: r.get(k, "") for k in ["location_name", "address", "matched_chat_location", "crew_sidewalk", "crew_parking_lot"]}
            for r in registry
        ]
        matched_count = sum(1 for r in registry if r.get("_matched_raw"))
        parts = [f"Loaded {len(registry)} locations from {filename}.",
                 f"{matched_count} matched to chat data."]
        if auto_count:
            parts.append(f"Auto-assigned crews to {auto_count} locations.")
        status = html.Span(
            " ".join(parts),
            style={"color": "green", "fontWeight": "bold"})
        return table_data, status
    except Exception as e:
        return no_update, html.Span(f"Error parsing CSV: {e}", style={"color": "red"})


@app.callback(
    [Output("locations-registry-table", "data", allow_duplicate=True),
     Output("locations-auto-assign-status", "children")],
    Input("locations-auto-assign-btn", "n_clicks"),
    State("locations-registry-table", "data"),
    prevent_initial_call=True,
)
def auto_assign_crews(n_clicks, table_data):
    global SNOW_CONFIG
    if not n_clicks:
        raise PreventUpdate
    try:
        registry = SNOW_CONFIG.get("location_registry", [])
        if not registry and table_data:
            registry = []
            for row in table_data:
                entry = dict(row)
                mcl = entry.get("matched_chat_location", "")
                if mcl and "(" in mcl:
                    entry["_matched_raw"] = mcl.split("(")[0].strip()
                registry.append(entry)

        if not registry:
            return no_update, html.Span("No locations loaded. Upload a CSV first.",
                                         style={"color": "orange"})

        registry, assigned = _auto_assign_crews(registry, DF_ALL)
        SNOW_CONFIG["location_registry"] = registry
        updated_table = [
            {k: r.get(k, "") for k in ["location_name", "address", "matched_chat_location", "crew_sidewalk", "crew_parking_lot"]}
            for r in registry
        ]
        status = html.Span(
            f"Auto-assigned crews to {assigned} locations. Review and edit as needed, then Save.",
            style={"color": "green", "fontWeight": "bold"})
        return updated_table, status
    except Exception as e:
        return no_update, html.Span(f"Error: {e}", style={"color": "red"})


@app.callback(
    Output("locations-save-status", "children"),
    Input("locations-save-btn", "n_clicks"),
    State("locations-registry-table", "data"),
    prevent_initial_call=True,
)
def save_locations_registry(n_clicks, table_data):
    global SNOW_CONFIG
    if not n_clicks:
        raise PreventUpdate
    try:
        if table_data:
            existing = {r.get("location_name", ""): r for r in SNOW_CONFIG.get("location_registry", [])}
            clean_registry = []
            for row in table_data:
                name = row.get("location_name", "")
                prev = existing.get(name, {})
                entry = {
                    "location_name": name,
                    "address": row.get("address", prev.get("address", "")),
                    "lat": prev.get("lat", ""),
                    "lon": prev.get("lon", ""),
                    "matched_chat_location": row.get("matched_chat_location", prev.get("matched_chat_location", "")),
                    "crew_sidewalk": row.get("crew_sidewalk", ""),
                    "crew_parking_lot": row.get("crew_parking_lot", ""),
                    "_matched_raw": prev.get("_matched_raw", ""),
                    "_match_score": prev.get("_match_score", 0),
                }
                if not entry["_matched_raw"]:
                    mcl = entry["matched_chat_location"]
                    if mcl and "(" in mcl:
                        entry["_matched_raw"] = mcl.split("(")[0].strip()
                clean_registry.append(entry)
            SNOW_CONFIG["location_registry"] = clean_registry
        config_path = os.path.join(DATA_DIR, "config", "snow_removal.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(SNOW_CONFIG, f, indent=2)
        return html.Span("Locations saved!", style={"color": "green", "fontWeight": "bold"})
    except Exception as e:
        return html.Span(f"Error saving: {e}", style={"color": "red"})


# ── DC Boundary helper ────────────────────────────────────────────────────────

def _load_dc_boundary():
    cache_path = os.path.join(DATA_DIR, "config", "dc_boundary.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    try:
        import urllib.request
        url = ("https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/"
               "Administrative_Other_Boundaries_WebMercator/MapServer/10/"
               "query?where=1%3D1&outFields=*&outSR=4326&f=json")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return data
    except Exception as e:
        print(f"[map] Failed to load DC boundary: {e}")
        return None


# ── Deployment Breakdown callback ─────────────────────────────────────────────

@app.callback(
    Output("dep-breakdown-table-container", "children"),
    Input("dep-breakdown-select", "value"),
)
def update_deployment_breakdown(selected_dep):
    if not selected_dep:
        raise PreventUpdate
    try:
        dep_df = DF_ALL[(DF_ALL["deployment"] == selected_dep) & (DF_ALL["noise_type"] == "clean")]
        non_trackable = set(SNOW_CONFIG.get("non_trackable_senders", []))
        if non_trackable:
            dep_df = dep_df[~dep_df["sender_resolved"].isin(non_trackable)]

        if dep_df.empty:
            return html.P("No data for this deployment.", className="text-muted")

        locations = dep_df[dep_df["location"].str.len() > 0]["location"].unique()
        
        all_crews = sorted(set(
            crew for chat in dep_df["chat"].unique()
            for crew, _, _ in [_extract_crew_from_chat(chat)]
            if crew
        ))

        registry = SNOW_CONFIG.get("location_registry", [])
        registry_map = {}
        for r in registry:
            raw = r.get("_matched_raw", "")
            if raw:
                registry_map[_normalize_location(raw)] = r

        rows = []
        for loc in sorted(locations):
            loc_msgs = dep_df[dep_df["location"] == loc]
            crews_in_loc = loc_msgs["chat"].unique()
            detected_crews = []
            detected_type = "Unknown"
            for ch in crews_in_loc:
                crew, is_sw, is_pl = _extract_crew_from_chat(ch)
                if crew:
                    detected_crews.append(crew)
                    if is_pl:
                        detected_type = "Parking Lot"
                    elif is_sw:
                        detected_type = "Sidewalk"

            reg = registry_map.get(_normalize_location(loc), {})
            assigned_sw = reg.get("crew_sidewalk", "")
            assigned_pl = reg.get("crew_parking_lot", "")

            visits = len(loc_msgs)
            senders = loc_msgs["sender_resolved"].nunique()

            rows.append({
                "location": loc,
                "type": detected_type,
                "visits": visits,
                "senders": senders,
                "detected_crew": ", ".join(detected_crews) if detected_crews else "Unknown",
                "crew_sidewalk": assigned_sw,
                "crew_parking_lot": assigned_pl,
            })

        if not rows:
            return html.P("No locations found for this deployment.", className="text-muted")

        all_crew_options = sorted(set(
            crew for chat in ALL_CHATS
            for crew, _, _ in [_extract_crew_from_chat(chat)]
            if crew
        ))

        table = dash_table.DataTable(
            id="dep-breakdown-table",
            columns=[
                {"name": "Location", "id": "location", "editable": False},
                {"name": "Type", "id": "type", "editable": False},
                {"name": "Visits", "id": "visits", "type": "numeric", "editable": False},
                {"name": "Reporters", "id": "senders", "type": "numeric", "editable": False},
                {"name": "Detected Crew", "id": "detected_crew", "editable": False},
                {"name": "Assigned SW Crew", "id": "crew_sidewalk",
                 "editable": True, "presentation": "dropdown"},
                {"name": "Assigned PL Crew", "id": "crew_parking_lot",
                 "editable": True, "presentation": "dropdown"},
            ],
            data=rows,
            dropdown={
                "crew_sidewalk": {
                    "options": [{"label": "", "value": ""}] + 
                               [{"label": c, "value": c} for c in all_crew_options],
                },
                "crew_parking_lot": {
                    "options": [{"label": "", "value": ""}] + 
                               [{"label": c, "value": c} for c in all_crew_options],
                },
            },
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "6px", "fontSize": "12px",
                        "minWidth": "80px", "maxWidth": "250px", "overflow": "hidden",
                        "textOverflow": "ellipsis"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f1f3f5", "fontSize": "12px"},
            style_data_conditional=[
                {"if": {"column_id": "crew_sidewalk"}, "backgroundColor": "#f0fff0"},
                {"if": {"column_id": "crew_parking_lot"}, "backgroundColor": "#fff0f0"},
            ],
            page_size=30,
            sort_action="native",
            filter_action="native",
        )

        summary = html.Div([
            html.P([
                html.Strong(f"{len(rows)} locations"),
                f" serviced by ",
                html.Strong(f"{len(all_crews)} crews"),
                f" | {sum(r['visits'] for r in rows)} total visits",
            ], className="mb-2", style={"fontSize": "13px"}),
        ])

        return html.Div([summary, table])

    except Exception as e:
        print(f"[Deployment Breakdown] Error: {e}")
        import traceback; traceback.print_exc()
        return html.P("Error loading deployment breakdown.", className="text-muted")


@app.callback(
    Output("dep-save-crews-status", "children"),
    Input("dep-save-crews-btn", "n_clicks"),
    State("dep-breakdown-table", "data"),
    prevent_initial_call=True,
)
def save_deployment_crew_assignments(n_clicks, table_data):
    global SNOW_CONFIG
    if not n_clicks or not table_data:
        raise PreventUpdate
    try:
        registry = SNOW_CONFIG.get("location_registry", [])
        registry_by_norm = {}
        for r in registry:
            raw = r.get("_matched_raw", "")
            if raw:
                registry_by_norm[_normalize_location(raw)] = r
            mcl = r.get("matched_chat_location", "")
            if mcl and "(" in mcl:
                mcl_raw = mcl.split("(")[0].strip()
                if mcl_raw:
                    registry_by_norm.setdefault(_normalize_location(mcl_raw), r)
            loc_name = r.get("location_name", "")
            if loc_name:
                registry_by_norm.setdefault(_normalize_location(loc_name), r)

        updated = 0
        for row in table_data:
            loc = row.get("location", "")
            norm = _normalize_location(loc)
            if norm in registry_by_norm:
                entry = registry_by_norm[norm]
                new_sw = row.get("crew_sidewalk", "")
                new_pl = row.get("crew_parking_lot", "")
                if entry.get("crew_sidewalk", "") != new_sw or entry.get("crew_parking_lot", "") != new_pl:
                    entry["crew_sidewalk"] = new_sw
                    entry["crew_parking_lot"] = new_pl
                    updated += 1

        config_path = os.path.join(DATA_DIR, "config", "snow_removal.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(SNOW_CONFIG, f, indent=2, ensure_ascii=False)
        return html.Span(f"Saved! {updated} assignments updated.", style={"color": "green"})
    except Exception as e:
        return html.Span(f"Error: {e}", style={"color": "red"})


# ── Map callback ──────────────────────────────────────────────────────────────

@app.callback(
    Output("chart-service-map", "figure"),
    Input("main-tabs", "active_tab"),
    Input("map-deployment-filter", "value"),
    prevent_initial_call=True,
)
def render_service_map(active_tab, dep_filter):
    if active_tab != "tab-map":
        raise PreventUpdate
    try:
        fig = go.Figure()

        boundary_data = _load_dc_boundary()
        if boundary_data and "features" in boundary_data:
            for feature in boundary_data["features"]:
                geom = feature.get("geometry", {})
                rings = geom.get("rings", [])
                for ring in rings:
                    lats = [pt[1] for pt in ring]
                    lons = [pt[0] for pt in ring]
                    lats.append(lats[0])
                    lons.append(lons[0])
                    fig.add_trace(go.Scattermap(
                        lat=lats,
                        lon=lons,
                        mode="lines",
                        line=dict(width=2, color="#1f77b4"),
                        name="DC Boundary",
                        showlegend=True,
                        hoverinfo="skip",
                    ))

        registry = SNOW_CONFIG.get("location_registry", [])

        if dep_filter and dep_filter != "ALL":
            dep_df = DF_ALL[(DF_ALL["deployment"] == dep_filter) & (DF_ALL["noise_type"] == "clean")]
            non_trackable = set(SNOW_CONFIG.get("non_trackable_senders", []))
            if non_trackable:
                dep_df = dep_df[~dep_df["sender_resolved"].isin(non_trackable)]
            active_locations = set(dep_df[dep_df["location"].str.len() > 0]["location"].unique())
            loc_counts = dep_df[dep_df["location"].str.len() > 0]["location"].value_counts().to_dict()
        else:
            active_locations = None
            loc_counts = {}
            if not DF_ALL.empty:
                counts = DF_ALL[DF_ALL["location"].str.len() > 0]["location"].value_counts()
                loc_counts = counts.to_dict()

        locs_with_coords = [
            r for r in registry
            if r.get("lat") and r.get("lon")
            and r["lat"] != "" and r["lon"] != ""
        ]

        if active_locations is not None:
            filtered_locs = []
            for r in locs_with_coords:
                m = r.get("_matched_raw", "")
                if not m:
                    mc = r.get("matched_chat_location", "")
                    if mc and "(" in mc:
                        m = mc.split("(")[0].strip()
                norm_m = _normalize_location(m) if m else ""
                if norm_m and any(_normalize_location(al) == norm_m for al in active_locations):
                    filtered_locs.append(r)
                elif not norm_m:
                    loc_name_norm = _normalize_location(r.get("location_name", ""))
                    for al in active_locations:
                        if _normalize_location(al) == loc_name_norm:
                            filtered_locs.append(r)
                            break
            locs_with_coords = filtered_locs

        if locs_with_coords:
            BOLD_COLORS = [
                "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
                "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
                "#dcbeff", "#9A6324", "#800000", "#aaffc3", "#808000",
                "#000075", "#ff4500", "#00ced1",
            ]

            all_individual_crews = set()
            for r in locs_with_coords:
                for field in ["crew_sidewalk", "crew_parking_lot"]:
                    val = r.get(field, "")
                    if val:
                        for c in val.split(","):
                            c = c.strip()
                            if c:
                                all_individual_crews.add(c)
            all_individual_crews = sorted(all_individual_crews)
            if not all_individual_crews:
                all_individual_crews = ["Unassigned"]
            indiv_color = {c: BOLD_COLORS[i % len(BOLD_COLORS)] for i, c in enumerate(all_individual_crews)}
            indiv_color["Unassigned"] = "#888888"

            def _primary_crew(r):
                sw = r.get("crew_sidewalk", "")
                pl = r.get("crew_parking_lot", "")
                val = sw if sw else (pl if pl else "")
                if not val:
                    return "Unassigned"
                return val.split(",")[0].strip()

            crew_groups = {}
            for r in locs_with_coords:
                pc = _primary_crew(r)
                crew_groups.setdefault(pc, []).append(r)

            for crew in sorted(crew_groups.keys()):
                crew_locs = crew_groups[crew]
                lats = [float(r["lat"]) for r in crew_locs]
                lons = [float(r["lon"]) for r in crew_locs]
                sizes = []
                hovers = []
                for r in crew_locs:
                    m = r.get("_matched_raw", "")
                    if not m:
                        mc = r.get("matched_chat_location", "")
                        if mc and "(" in mc:
                            m = mc.split("(")[0].strip()
                    cnt = loc_counts.get(m, 1)
                    sizes.append(max(10, min(30, 8 + cnt * 2)))
                    sw = r.get("crew_sidewalk", "")
                    pl = r.get("crew_parking_lot", "")
                    hover_parts = [f"<b>{r.get('location_name', '')}</b>"]
                    hover_parts.append(f"Visits: {loc_counts.get(m, 0)}")
                    if sw:
                        hover_parts.append(f"Sidewalk: {sw}")
                    if pl:
                        hover_parts.append(f"Parking: {pl}")
                    if not sw and not pl:
                        hover_parts.append("Crew: Unassigned")
                    hovers.append("<br>".join(hover_parts))
                fig.add_trace(go.Scattermap(
                    lat=lats,
                    lon=lons,
                    mode="markers",
                    marker=dict(
                        size=sizes,
                        color=indiv_color.get(crew, "#888888"),
                        opacity=0.95,
                    ),
                    text=hovers,
                    hoverinfo="text",
                    name=crew,
                    showlegend=True,
                ))

        title = "DC Service Map"
        if dep_filter and dep_filter != "ALL":
            title += f" — {dep_filter}"
            n_locs = len(locs_with_coords) if locs_with_coords else 0
            title += f" ({n_locs} sites)"

        fig.update_layout(
            map=dict(
                style="open-street-map",
                center=dict(lat=38.9072, lon=-77.0369),
                zoom=11,
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            height=700,
            title=title,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        return fig
    except Exception as e:
        print(f"[map] Error rendering map: {e}")
        return _empty_fig("DC Service Map")


# ── Helper: empty figure ─────────────────────────────────────────────────────

def _empty_fig(title):
    fig = go.Figure()
    fig.add_annotation(
        text="No data for current filters",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=16, color="gray"),
    )
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
    )
    return fig


# ── Finances callbacks ───────────────────────────────────────────────────────

@app.callback(
    Output("fin-kpi-revenue", "children"),
    Output("fin-kpi-costs", "children"),
    Output("fin-kpi-profit", "children"),
    Output("fin-kpi-profit", "style"),
    Output("fin-kpi-margin", "children"),
    Output("fin-kpi-margin", "style"),
    Output("fin-kpi-sites", "children"),
    Output("fin-kpi-salt", "children"),
    Output("fin-kpi-cost-hr", "children"),
    Output("fin-kpi-rev-dep", "children"),
    Output("fin-deployment-table-container", "children"),
    Output("fin-crew-table-container", "children"),
    Output("fin-chart-revenue", "figure"),
    Input("main-tabs", "active_tab"),
    Input("fin-recalc-btn", "n_clicks"),
    State("fin-labor-sw", "value"),
    State("fin-labor-pl", "value"),
    State("fin-machine-rate", "value"),
    State("fin-salt-cost", "value"),
    State("fin-salt-sw", "value"),
    State("fin-salt-pl", "value"),
    State("fin-workers-sw", "value"),
    State("fin-workers-pl", "value"),
    State("fin-overhead", "value"),
    State("fin-default-tier", "value"),
)
def update_finances(active_tab, n_clicks, labor_sw, labor_pl, machine_rate,
                    salt_cost, salt_sw, salt_pl, workers_sw, workers_pl,
                    overhead, default_tier):
    if active_tab != "tab-finances":
        raise PreventUpdate
    try:
        temp_config = dict(SNOW_CONFIG)
        temp_config["finance_config"] = {
            "labor_rate_sidewalk": labor_sw or 25.0,
            "labor_rate_parking": labor_pl or 30.0,
            "machine_hourly_rate": machine_rate or 75.0,
            "salt_cost_per_lb": salt_cost or 0.15,
            "salt_lbs_per_site_sidewalk": salt_sw or 50.0,
            "salt_lbs_per_site_parking": salt_pl or 200.0,
            "workers_sidewalk": workers_sw or 3,
            "workers_parking": min(workers_pl or 2, 2),
            "overhead_pct": overhead or 10.0,
            "default_snow_tier": default_tier or "price_under_6in",
            "deployment_snow_tiers": SNOW_CONFIG.get("finance_config", {}).get("deployment_snow_tiers", {}),
        }

        fin = _compute_financials(DF_ALL, ALL_DEPLOYMENTS_LIST, temp_config, PRICING_INDEX)

        rev_str = f"${fin['total_revenue']:,.0f}"
        cost_str = f"${fin['total_costs']:,.0f}"
        profit_str = f"${fin['total_profit']:,.0f}"
        profit_color = "#28a745" if fin['total_profit'] >= 0 else "#dc3545"
        margin_str = f"{fin['profit_margin']:.1f}%"
        margin_color = "#28a745" if fin['profit_margin'] >= 0 else "#dc3545"
        sites_str = f"{fin['sites_serviced']} / {fin['sites_contracted']}"
        salt_str = f"{fin['total_salt_lbs']:,.0f}"
        cost_hr_str = f"${fin['cost_per_hour']:,.0f}"
        rev_dep_str = f"${fin['revenue_per_deployment']:,.0f}"

        dep_table = dash_table.DataTable(
            columns=[
                {"name": "Deployment", "id": "deployment"},
                {"name": "Type", "id": "type"},
                {"name": "Tier", "id": "snow_tier"},
                {"name": "Source", "id": "source"},
                {"name": "Routes", "id": "total_sites", "type": "numeric"},
                {"name": "SW", "id": "sw_routes", "type": "numeric"},
                {"name": "PL", "id": "pl_routes", "type": "numeric"},
                {"name": "Crews", "id": "crews", "type": "numeric"},
                {"name": "Hours", "id": "hours", "type": "numeric"},
                {"name": "Revenue", "id": "revenue", "type": "numeric"},
                {"name": "Labor", "id": "labor_cost", "type": "numeric"},
                {"name": "Machine", "id": "machine_cost", "type": "numeric"},
                {"name": "Salt", "id": "material_cost", "type": "numeric"},
                {"name": "Overhead", "id": "overhead", "type": "numeric"},
                {"name": "Total Cost", "id": "total_cost", "type": "numeric"},
                {"name": "Profit", "id": "profit", "type": "numeric"},
                {"name": "Margin %", "id": "margin", "type": "numeric"},
                {"name": "Salt (lbs)", "id": "salt_lbs", "type": "numeric"},
            ],
            data=fin["deployment_financials"],
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "6px", "fontSize": "12px"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f1f3f5", "fontSize": "12px"},
            style_data_conditional=[
                {"if": {"filter_query": "{profit} < 0"}, "color": "#dc3545", "fontWeight": "bold"},
                {"if": {"filter_query": "{margin} >= 20"}, "color": "#28a745"},
            ],
            page_size=20,
        ) if fin["deployment_financials"] else html.P("No deployment financial data available.", className="text-muted")

        crew_table = dash_table.DataTable(
            columns=[
                {"name": "Crew", "id": "crew"},
                {"name": "Type", "id": "type"},
                {"name": "Workers", "id": "workers", "type": "numeric"},
                {"name": "Deps", "id": "deployments", "type": "numeric"},
                {"name": "Sites", "id": "sites", "type": "numeric"},
                {"name": "Hours", "id": "hours", "type": "numeric"},
                {"name": "Revenue", "id": "revenue", "type": "numeric"},
                {"name": "Labor", "id": "labor_cost", "type": "numeric"},
                {"name": "Machine", "id": "machine_cost", "type": "numeric"},
                {"name": "Total Cost", "id": "total_cost", "type": "numeric"},
                {"name": "Profit", "id": "profit", "type": "numeric"},
                {"name": "$/Hour", "id": "rate_per_hour", "type": "numeric"},
            ],
            data=fin["crew_financials"],
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "6px", "fontSize": "12px"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f1f3f5", "fontSize": "12px"},
            style_data_conditional=[
                {"if": {"filter_query": "{profit} < 0"}, "color": "#dc3545"},
            ],
            page_size=20,
        ) if fin["crew_financials"] else html.P("No crew financial data available.", className="text-muted")

        if fin["deployment_financials"]:
            dep_df_chart = pd.DataFrame(fin["deployment_financials"])
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Revenue", x=dep_df_chart["deployment"], y=dep_df_chart["revenue"],
                                marker_color="#28a745"))
            fig.add_trace(go.Bar(name="Labor", x=dep_df_chart["deployment"], y=dep_df_chart["labor_cost"],
                                marker_color="#fd7e14"))
            fig.add_trace(go.Bar(name="Machine", x=dep_df_chart["deployment"], y=dep_df_chart["machine_cost"],
                                marker_color="#6f42c1"))
            fig.add_trace(go.Bar(name="Salt", x=dep_df_chart["deployment"], y=dep_df_chart["material_cost"],
                                marker_color="#20c997"))
            fig.add_trace(go.Bar(name="Profit", x=dep_df_chart["deployment"], y=dep_df_chart["profit"],
                                marker_color="#17a2b8"))
            fig.update_layout(barmode="group", template="plotly_white",
                            margin=dict(l=40, r=20, t=30, b=40),
                            legend=dict(orientation="h", y=1.1),
                            yaxis_title="Amount ($)")
        else:
            fig = go.Figure()
            fig.update_layout(template="plotly_white",
                            annotations=[{"text": "No financial data", "showarrow": False,
                                         "font": {"size": 16, "color": "gray"}}])

        return (rev_str, cost_str, profit_str, {"color": profit_color},
                margin_str, {"color": margin_color},
                sites_str, salt_str, cost_hr_str, rev_dep_str,
                dep_table, crew_table, fig)
    except Exception as e:
        print(f"[Finances] Error: {e}")
        import traceback; traceback.print_exc()
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_white")
        return ("$0", "$0", "$0", {}, "0%", {}, "0 / 0", "0", "$0", "$0",
                html.P("Error computing financials.", className="text-muted"),
                html.P("Error computing financials.", className="text-muted"),
                empty_fig)


@app.callback(
    Output("fin-save-status", "children"),
    Input("fin-save-btn", "n_clicks"),
    State("fin-labor-sw", "value"),
    State("fin-labor-pl", "value"),
    State("fin-machine-rate", "value"),
    State("fin-salt-cost", "value"),
    State("fin-salt-sw", "value"),
    State("fin-salt-pl", "value"),
    State("fin-workers-sw", "value"),
    State("fin-workers-pl", "value"),
    State("fin-overhead", "value"),
    State("fin-default-tier", "value"),
    prevent_initial_call=True,
)
def save_finance_config(n_clicks, labor_sw, labor_pl, machine_rate,
                        salt_cost, salt_sw, salt_pl, workers_sw, workers_pl,
                        overhead, default_tier):
    if not n_clicks:
        raise PreventUpdate
    try:
        config_path = os.path.join(DATA_DIR, "config", "snow_removal.json")
        config = load_config(config_path)
        config["finance_config"] = {
            "labor_rate_sidewalk": labor_sw or 25.0,
            "labor_rate_parking": labor_pl or 30.0,
            "machine_hourly_rate": machine_rate or 75.0,
            "salt_cost_per_lb": salt_cost or 0.15,
            "salt_lbs_per_site_sidewalk": salt_sw or 50.0,
            "salt_lbs_per_site_parking": salt_pl or 200.0,
            "workers_sidewalk": workers_sw or 3,
            "workers_parking": min(workers_pl or 2, 2),
            "overhead_pct": overhead or 10.0,
            "default_snow_tier": default_tier or "price_under_6in",
            "deployment_snow_tiers": config.get("finance_config", {}).get("deployment_snow_tiers", {}),
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        global SNOW_CONFIG
        SNOW_CONFIG = config
        return html.Span("Saved!", style={"color": "green"})
    except Exception as e:
        return html.Span(f"Error: {e}", style={"color": "red"})


# ── Invoice List on Settings Load ────────────────────────────────────────────

@app.callback(
    Output("invoice-list-container", "children"),
    Input("main-tabs", "active_tab"),
)
def show_invoice_list_on_load(active_tab):
    if active_tab != "tab-settings":
        raise PreventUpdate
    return _render_invoice_list(SNOW_CONFIG.get("invoices", []))


# ── Invoice Upload Callback ──────────────────────────────────────────────────

@app.callback(
    [Output("invoice-upload-status", "children"),
     Output("invoice-preview-container", "children"),
     Output("invoice-list-container", "children", allow_duplicate=True)],
    Input("upload-invoice", "contents"),
    State("upload-invoice", "filename"),
    prevent_initial_call=True,
)
def handle_invoice_upload(contents_list, filenames_list):
    global SNOW_CONFIG
    if not contents_list:
        raise PreventUpdate
    if isinstance(contents_list, str):
        contents_list = [contents_list]
        filenames_list = [filenames_list]

    config_path = os.path.join(DATA_DIR, "config", "snow_removal.json")
    invoices_registry = SNOW_CONFIG.get("invoices", [])

    all_parsed = []
    errors = []
    for content, fname in zip(contents_list, filenames_list):
        try:
            _, content_string = content.split(",")
            decoded = base64.b64decode(content_string)
            parsed = parse_invoice_bytes(decoded, fname)
            if parsed:
                for inv in parsed:
                    matched_dep = match_invoice_to_deployment(
                        inv.get("deployment_date"), ALL_DEPLOYMENTS_LIST, tolerance_days=3)
                    inv["matched_deployment"] = matched_dep
                    inv_record = {
                        "filename": fname,
                        "deployment_type": inv["deployment_type"],
                        "snow_tier": inv["snow_tier"],
                        "deployment_date": inv["deployment_date"],
                        "matched_deployment": matched_dep,
                        "total_billed": inv["total_billed"],
                        "site_count": inv["site_count"],
                        "format": inv["format"],
                        "line_items": inv["line_items"],
                    }
                    existing = [i for i in invoices_registry
                                if i.get("filename") == fname and i.get("deployment_date") == inv.get("deployment_date")]
                    if existing:
                        invoices_registry.remove(existing[0])
                    invoices_registry.append(inv_record)
                all_parsed.extend(parsed)
            else:
                errors.append(f"{fname}: No invoice data found")
        except Exception as e:
            errors.append(f"{fname}: {e}")

    SNOW_CONFIG["invoices"] = invoices_registry
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(SNOW_CONFIG, f, indent=2, ensure_ascii=False)
    except Exception as e:
        errors.append(f"Config save error: {e}")

    if errors:
        status = html.Div([
            html.Span(f"Imported {len(all_parsed)} invoice(s). ", style={"color": "green"}),
            html.Span(f"Errors: {'; '.join(errors)}", style={"color": "red"}),
        ])
    else:
        status = html.Span(f"Successfully imported {len(all_parsed)} invoice(s).", style={"color": "green"})

    preview_rows = []
    for inv in all_parsed:
        preview_rows.append(
            dbc.Card([
                dbc.CardBody([
                    html.H6(f"{inv['filename']}", className="mb-1"),
                    html.P(f"Type: {inv['deployment_type']} | Tier: {inv['snow_tier'].replace('price_', '').replace('_', ' ').title()} | "
                           f"Sites: {inv['site_count']} | Total: ${inv['total_billed']:,.2f}",
                           className="mb-1", style={"fontSize": "13px"}),
                    html.P(f"Date: {inv.get('deployment_date', 'Unknown')} | "
                           f"Matched Deployment: {inv.get('matched_deployment', 'No match')} | "
                           f"Format: {inv['format']}",
                           className="text-muted mb-0", style={"fontSize": "12px"}),
                ])
            ], className="mb-2 shadow-sm")
        )

    inv_list = _render_invoice_list(invoices_registry)

    return status, html.Div(preview_rows), inv_list


def _render_invoice_list(invoices_registry):
    if not invoices_registry:
        return html.P("No invoices imported yet.", className="text-muted", style={"fontSize": "13px"})

    rows = []
    for inv in invoices_registry:
        tier_label = inv.get("snow_tier", "").replace("price_", "").replace("_", " ").title()
        rows.append({
            "filename": inv.get("filename", ""),
            "type": inv.get("deployment_type", ""),
            "tier": tier_label,
            "date": inv.get("deployment_date", ""),
            "deployment": inv.get("matched_deployment", ""),
            "sites": inv.get("site_count", 0),
            "total": f"${inv.get('total_billed', 0):,.2f}",
        })

    return dash_table.DataTable(
        columns=[
            {"name": "File", "id": "filename"},
            {"name": "Type", "id": "type"},
            {"name": "Tier", "id": "tier"},
            {"name": "Date", "id": "date"},
            {"name": "Deployment", "id": "deployment"},
            {"name": "Sites", "id": "sites", "type": "numeric"},
            {"name": "Total Billed", "id": "total"},
        ],
        data=rows,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "6px", "fontSize": "12px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f1f3f5", "fontSize": "12px"},
        page_size=10,
    )


# ── Per-Deployment Labor Overrides Callback ─────────────────────────────────

@app.callback(
    Output("fin-labor-overrides-container", "children"),
    Input("main-tabs", "active_tab"),
    Input("fin-labor-overrides-status", "children"),
)
def render_labor_overrides(active_tab, _trigger):
    if active_tab != "tab-finances":
        raise PreventUpdate

    if not ALL_DEPLOYMENTS_LIST:
        return html.P("No deployments available.", className="text-muted", style={"fontSize": "13px"})

    labor_overrides = SNOW_CONFIG.get("labor_overrides", {})
    all_crews = sorted(DF_ALL["chat"].unique()) if not DF_ALL.empty else []

    rows_data = []
    for dep in ALL_DEPLOYMENTS_LIST:
        label = dep["label"]
        dep_overrides = labor_overrides.get(label, {})
        dep_df = DF_ALL[(DF_ALL.get("deployment") == label) if "deployment" in DF_ALL.columns else DF_ALL["msg_date"].dt.date.between(dep["start_date"], dep["end_date"])]
        dep_crews = sorted(dep_df["chat"].unique()) if not dep_df.empty else all_crews[:5]

        for crew in dep_crews:
            crew_override = dep_overrides.get(crew, {})
            rows_data.append({
                "deployment": label,
                "crew": crew,
                "labor_rate": crew_override.get("labor_rate", ""),
                "workers": crew_override.get("workers", ""),
                "hours": crew_override.get("hours", ""),
            })

    if not rows_data:
        return html.P("No crew data for labor overrides.", className="text-muted", style={"fontSize": "13px"})

    return dash_table.DataTable(
        id="fin-labor-overrides-table",
        columns=[
            {"name": "Deployment", "id": "deployment"},
            {"name": "Crew", "id": "crew"},
            {"name": "Labor $/hr", "id": "labor_rate", "type": "numeric", "editable": True},
            {"name": "Workers", "id": "workers", "type": "numeric", "editable": True},
            {"name": "Hours", "id": "hours", "type": "numeric", "editable": True},
        ],
        data=rows_data,
        editable=True,
        style_table={"overflowX": "auto", "maxHeight": "400px", "overflowY": "auto"},
        style_cell={"textAlign": "left", "padding": "6px", "fontSize": "12px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f1f3f5", "fontSize": "12px",
                      "position": "sticky", "top": 0},
        style_data_conditional=[
            {"if": {"column_editable": True}, "backgroundColor": "#fffde7"},
        ],
        page_size=50,
    )


@app.callback(
    Output("fin-labor-overrides-status", "children"),
    Input("fin-save-labor-overrides", "n_clicks"),
    State("fin-labor-overrides-table", "data"),
    prevent_initial_call=True,
)
def save_labor_overrides(n_clicks, table_data):
    global SNOW_CONFIG
    if not n_clicks or not table_data:
        raise PreventUpdate

    labor_overrides = {}
    for row in table_data:
        dep = row.get("deployment", "")
        crew = row.get("crew", "")
        if not dep or not crew:
            continue
        lr = row.get("labor_rate")
        wk = row.get("workers")
        hrs = row.get("hours")
        if lr or wk or hrs:
            if dep not in labor_overrides:
                labor_overrides[dep] = {}
            override = {}
            if lr and lr != "":
                try:
                    override["labor_rate"] = float(lr)
                except (ValueError, TypeError):
                    pass
            if wk and wk != "":
                try:
                    override["workers"] = int(wk)
                except (ValueError, TypeError):
                    pass
            if hrs and hrs != "":
                try:
                    override["hours"] = float(hrs)
                except (ValueError, TypeError):
                    pass
            if override:
                labor_overrides[dep][crew] = override

    SNOW_CONFIG["labor_overrides"] = labor_overrides

    config_path = os.path.join(DATA_DIR, "config", "snow_removal.json")
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(SNOW_CONFIG, f, indent=2, ensure_ascii=False)
        return html.Span("Labor overrides saved!", style={"color": "green"})
    except Exception as e:
        return html.Span(f"Error: {e}", style={"color": "red"})


# ── Invoice Reconciliation Callback ─────────────────────────────────────────

@app.callback(
    Output("fin-recon-deployment", "options"),
    Input("main-tabs", "active_tab"),
)
def populate_recon_deployments(active_tab):
    if active_tab != "tab-finances":
        raise PreventUpdate
    invoices = SNOW_CONFIG.get("invoices", [])
    options = []
    seen = set()
    for inv in invoices:
        dep = inv.get("matched_deployment")
        if dep and dep not in seen:
            seen.add(dep)
            options.append({"label": f"{dep} ({inv.get('deployment_type', '')})", "value": dep})
    if not options:
        for dep in ALL_DEPLOYMENTS_LIST:
            options.append({"label": dep["label"], "value": dep["label"]})
    return options


@app.callback(
    Output("fin-recon-container", "children"),
    Input("fin-recon-btn", "n_clicks"),
    State("fin-recon-deployment", "value"),
    prevent_initial_call=True,
)
def reconcile_deployment(n_clicks, dep_label):
    if not n_clicks or not dep_label:
        raise PreventUpdate

    invoices = SNOW_CONFIG.get("invoices", [])
    matched_invoices = [inv for inv in invoices if inv.get("matched_deployment") == dep_label]
    if not matched_invoices:
        return html.P(f"No invoices found for deployment '{dep_label}'. Upload invoices in Settings first.",
                      className="text-muted")

    dep_info = next((d for d in ALL_DEPLOYMENTS_LIST if d["label"] == dep_label), None)
    if not dep_info:
        return html.P("Deployment not found.", className="text-muted")

    start = pd.Timestamp(dep_info["start_date"])
    end = pd.Timestamp(dep_info["end_date"])
    df_clean = DF_ALL[(DF_ALL["noise_type"] == "clean") &
                      (DF_ALL["msg_date"] >= start) &
                      (DF_ALL["msg_date"] <= end + pd.Timedelta(days=1))]
    chat_locations = list(df_clean[df_clean["location"].str.len() > 0]["location"].unique()) if not df_clean.empty else []

    job_logs = build_job_logs(df_clean, SNOW_CONFIG)
    dep_routes = get_billable_routes_df(job_logs, deployment=dep_label) if not job_logs.empty else None

    results_children = []
    for inv in matched_invoices:
        recon = reconcile_invoice_with_chat(inv, chat_locations, PRICING_INDEX, _normalize_location,
                                            billable_routes=dep_routes)

        route_info = ""
        if recon.get("billable_routes_total", 0) > 0:
            route_info = (f" | Billable Routes: {recon['billable_routes_total']} "
                         f"(SW: {recon['sw_routes']}, PL: {recon['pl_routes']})")

        summary = dbc.Alert([
            html.Strong(f"{inv.get('filename', 'Invoice')} — {inv.get('deployment_type', '')}"),
            html.Br(),
            html.Span(f"Invoiced: {recon['sites_invoiced']} sites | "
                      f"Found in chat: {recon['sites_in_chat']} | "
                      f"Not in chat: {recon['sites_not_in_chat']} | "
                      f"Price mismatches: {recon['price_mismatches']} | "
                      f"Chat but not invoiced: {recon['chat_not_invoiced_count']}"
                      f"{route_info}"),
        ], color="info" if recon["sites_not_in_chat"] == 0 else "warning", className="mb-2")

        issue_rows = [r for r in recon["line_items"] if r["status"] != "ok"]
        if issue_rows:
            issue_table = dash_table.DataTable(
                columns=[
                    {"name": "Building", "id": "building_name"},
                    {"name": "Address", "id": "address"},
                    {"name": "Billed", "id": "billed_amount", "type": "numeric"},
                    {"name": "Contract", "id": "contract_price", "type": "numeric"},
                    {"name": "In Chat?", "id": "in_chat_data"},
                    {"name": "Service Areas", "id": "service_areas"},
                    {"name": "Status", "id": "status"},
                    {"name": "Notes", "id": "notes"},
                ],
                data=[{k: (f"${v:,.2f}" if k in ("billed_amount", "contract_price") and v else v)
                       for k, v in r.items() if k != "matched_chat_location"} for r in issue_rows],
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "padding": "4px", "fontSize": "11px"},
                style_header={"fontWeight": "bold", "backgroundColor": "#f1f3f5", "fontSize": "11px"},
                style_data_conditional=[
                    {"if": {"filter_query": '{status} = "not_in_chat"'}, "backgroundColor": "#fff3cd"},
                    {"if": {"filter_query": '{status} = "price_mismatch"'}, "backgroundColor": "#f8d7da"},
                ],
                page_size=20,
            )
        else:
            issue_table = html.P("All invoiced sites match chat data.", className="text-success",
                                style={"fontSize": "13px"})

        chat_missing = recon.get("chat_not_invoiced", [])
        if chat_missing:
            missing_list = html.Details([
                html.Summary(f"{len(chat_missing)} site(s) in chat but not invoiced",
                            style={"fontSize": "13px", "color": "#856404", "cursor": "pointer"}),
                html.Ul([html.Li(loc, style={"fontSize": "12px"}) for loc in chat_missing[:30]]),
            ], className="mt-1")
        else:
            missing_list = html.Div()

        results_children.extend([summary, issue_table, missing_list])

    return html.Div(results_children)


# ── Completion Verification Callback ────────────────────────────────────────

@app.callback(
    Output("fin-verify-deployment", "options"),
    Input("main-tabs", "active_tab"),
)
def populate_verify_deployments(active_tab):
    if active_tab != "tab-finances":
        raise PreventUpdate
    invoices = SNOW_CONFIG.get("invoices", [])
    options = []
    seen = set()
    for inv in invoices:
        if inv.get("format") in ("completion_report", "pretreatment_report"):
            dep = inv.get("matched_deployment")
            if dep and dep not in seen:
                seen.add(dep)
                options.append({"label": f"{dep} ({inv.get('deployment_type', '')})", "value": dep})
    if not options:
        for dep in ALL_DEPLOYMENTS_LIST:
            options.append({"label": dep["label"], "value": dep["label"]})
    return options


@app.callback(
    Output("fin-verify-container", "children"),
    Input("fin-verify-btn", "n_clicks"),
    State("fin-verify-deployment", "value"),
    prevent_initial_call=True,
)
def verify_completion(n_clicks, dep_label):
    if not n_clicks or not dep_label:
        raise PreventUpdate

    invoices = SNOW_CONFIG.get("invoices", [])
    completion_invoices = [inv for inv in invoices
                          if inv.get("matched_deployment") == dep_label
                          and inv.get("format") in ("completion_report", "pretreatment_report")]

    if not completion_invoices:
        return html.P(f"No completion/pre-treatment reports found for '{dep_label}'. "
                      "Upload portal completion reports (Excel) in Settings.",
                      className="text-muted")

    dep_info = next((d for d in ALL_DEPLOYMENTS_LIST if d["label"] == dep_label), None)
    if not dep_info:
        return html.P("Deployment not found.", className="text-muted")

    start = pd.Timestamp(dep_info["start_date"])
    end = pd.Timestamp(dep_info["end_date"])
    df_clean = DF_ALL[(DF_ALL["noise_type"] == "clean") &
                      (DF_ALL["msg_date"] >= start) &
                      (DF_ALL["msg_date"] <= end + pd.Timedelta(days=1))]

    chat_locs_by_crew = {}
    if not df_clean.empty:
        for _, row in df_clean[df_clean["location"].str.len() > 0].iterrows():
            crew = row["chat"]
            loc = row["location"]
            if crew not in chat_locs_by_crew:
                chat_locs_by_crew[crew] = set()
            chat_locs_by_crew[crew].add(_normalize_location(loc))

    results_children = []
    for inv in completion_invoices:
        verify_rows = []
        for item in inv.get("line_items", []):
            bname = item.get("building_name", "")
            bname_norm = _normalize_location(bname)

            portal_crew = item.get("removal_crew") or item.get("pretreatment_crew") or item.get("created_by", "")
            portal_time = item.get("created_time", "")
            pct_done = item.get("pct_completed", "")
            sw_done = item.get("sidewalks_cleared") or item.get("sidewalks_pretreated", "")
            pl_done = item.get("parking_cleared") or item.get("parking_pretreated", "")

            chat_crew_found = None
            chat_confirmed = False
            for crew, locs in chat_locs_by_crew.items():
                if bname_norm in locs:
                    chat_crew_found = crew
                    chat_confirmed = True
                    break
                for loc in locs:
                    if bname_norm and (bname_norm in loc or loc in bname_norm):
                        chat_crew_found = crew
                        chat_confirmed = True
                        break
                if chat_confirmed:
                    break

            status = "verified" if chat_confirmed else "unverified"
            crew_match = ""
            if portal_crew and chat_crew_found:
                crew_match = "match" if _normalize_location(portal_crew) in _normalize_location(chat_crew_found) else "different"
            elif not portal_crew and chat_crew_found:
                crew_match = "chat only"
            elif portal_crew and not chat_crew_found:
                crew_match = "portal only"

            verify_rows.append({
                "building": bname,
                "portal_crew": portal_crew,
                "portal_time": str(portal_time),
                "completion": str(pct_done) if pct_done else "",
                "sw": str(sw_done),
                "pl": str(pl_done),
                "chat_crew": chat_crew_found or "",
                "status": status,
                "crew_match": crew_match,
            })

        verified = sum(1 for r in verify_rows if r["status"] == "verified")
        unverified = sum(1 for r in verify_rows if r["status"] == "unverified")

        summary = dbc.Alert([
            html.Strong(f"{inv.get('filename', 'Report')} — {inv.get('deployment_type', '')}"),
            html.Br(),
            html.Span(f"Total sites: {len(verify_rows)} | "
                      f"Verified in chat: {verified} | "
                      f"Unverified: {unverified}"),
        ], color="success" if unverified == 0 else "warning", className="mb-2")

        vtable = dash_table.DataTable(
            columns=[
                {"name": "Building", "id": "building"},
                {"name": "Portal Crew", "id": "portal_crew"},
                {"name": "Portal Time", "id": "portal_time"},
                {"name": "Completion", "id": "completion"},
                {"name": "SW", "id": "sw"},
                {"name": "PL", "id": "pl"},
                {"name": "Chat Crew", "id": "chat_crew"},
                {"name": "Status", "id": "status"},
                {"name": "Crew Match", "id": "crew_match"},
            ],
            data=verify_rows,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "4px", "fontSize": "11px"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f1f3f5", "fontSize": "11px"},
            style_data_conditional=[
                {"if": {"filter_query": '{status} = "unverified"'}, "backgroundColor": "#fff3cd"},
                {"if": {"filter_query": '{status} = "verified"'}, "backgroundColor": "#d4edda"},
                {"if": {"filter_query": '{crew_match} = "different"'}, "color": "#dc3545", "fontWeight": "bold"},
            ],
            page_size=30,
        )

        results_children.extend([summary, vtable, html.Hr(className="my-2")])

    return html.Div(results_children)


# ── Apply Labor Overrides in Financial Calculations ─────────────────────────

def _get_crew_labor_override(dep_label, crew_chat, snow_config):
    overrides = snow_config.get("labor_overrides", {})
    dep_overrides = overrides.get(dep_label, {})
    return dep_overrides.get(crew_chat, {})


# ── Entry point ──────────────────────────────────────────────────────────────

def print_report():
    """Print analytic metrics report to terminal."""
    sep = "─" * 72
    print(f"\n{'═' * 72}")
    print("  WhatsApp Chat Dashboard — Data Report")
    print(f"{'═' * 72}\n")

    # ── Overall stats
    pct_clean = CLEAN_MSGS / TOTAL_MSGS * 100 if TOTAL_MSGS else 0
    pct_noise = NOISE_MSGS / TOTAL_MSGS * 100 if TOTAL_MSGS else 0
    print(f"  Total messages:  {TOTAL_MSGS}")
    print(f"  Clean:           {CLEAN_MSGS}  ({pct_clean:.1f}%)")
    print(f"  Noise:           {NOISE_MSGS}  ({pct_noise:.1f}%)")
    print(f"  Unique chats:    {NUM_CHATS}")
    print(f"  Unique senders:  {NUM_SENDERS}")
    print(f"  Message types:   {', '.join(ALL_TYPES)}")

    # ── Deployment summary
    print(f"\n{sep}")
    print("  Deployment Summary")
    print(sep)
    if ALL_DEPLOYMENTS_LIST:
        print(f"  {'#':>2s}  {'Date Range':<20s} {'Days':>4s} {'Crews':>5s} "
              f"{'Sites':>5s} {'Msgs':>5s}")
        print(f"  {'─' * 2}  {'─' * 20} {'─' * 4} {'─' * 5} {'─' * 5} {'─' * 5}")
        for dep in ALL_DEPLOYMENTS_LIST:
            grp = DF_ALL[DF_ALL["deployment"] == dep["label"]]
            grp_clean = grp[grp["noise_type"] == "clean"]
            n_crews = grp_clean["chat"].nunique()
            n_msgs = len(grp_clean)
            visits_df = _build_site_visits(grp_clean, "chat")
            n_sites = len(visits_df)
            print(f"  {dep['id']:>2d}  {dep['label']:<20s} {dep['days']:>4d} "
                  f"{n_crews:>5d} {n_sites:>5d} {n_msgs:>5d}")
    else:
        print("  No deployments detected.")

    # ── Noise breakdown
    print(f"\n{sep}")
    print("  Noise Breakdown")
    print(sep)
    noise_counts = DF_ALL["noise_type"].value_counts()
    for ntype in sorted(noise_counts.index):
        cnt = noise_counts[ntype]
        pct = cnt / TOTAL_MSGS * 100
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"  {ntype:<25s} {cnt:>4d}  ({pct:5.1f}%)  {bar}")

    # ── Per-chat breakdown
    print(f"\n{sep}")
    print("  Per-Chat Breakdown")
    print(sep)
    print(f"  {'Chat':<35s} {'Total':>5s} {'Clean':>5s} {'Noise':>5s} "
          f"{'%Clean':>6s}  {'Senders'}")
    print(f"  {'─' * 35} {'─' * 5} {'─' * 5} {'─' * 5} {'─' * 6}  {'─' * 20}")
    for chat in ALL_CHATS:
        sub = DF_ALL[DF_ALL["chat"] == chat]
        n_total = len(sub)
        n_clean = len(sub[sub["noise_type"] == "clean"])
        n_noise = n_total - n_clean
        pct = n_clean / n_total * 100 if n_total else 0
        senders_in = sorted(
            sub[sub["sender_resolved"] != ""]["sender_resolved"].unique())
        senders_str = ", ".join(senders_in)
        if len(senders_str) > 20:
            senders_str = senders_str[:17] + "..."
        chat_short = chat[:35]
        print(f"  {chat_short:<35s} {n_total:>5d} {n_clean:>5d} {n_noise:>5d} "
              f"{pct:>5.1f}%  {senders_str}")

    # ── Per-sender stats (resolved)
    print(f"\n{sep}")
    print("  Per-Sender Stats (resolved)")
    print(sep)
    df_with_sender = DF_ALL[DF_ALL["sender_resolved"] != ""]
    sender_stats = (df_with_sender.groupby("sender_resolved")
                    .agg(total=("noise_type", "size"),
                         clean=("noise_type", lambda x: (x == "clean").sum()),
                         chats=("chat", "nunique"),
                         avg_len=("content_len", "mean"))
                    .sort_values("total", ascending=False))
    print(f"  {'Sender':<25s} {'Total':>5s} {'Clean':>5s} {'Chats':>5s} "
          f"{'Avg Len':>7s}")
    print(f"  {'─' * 25} {'─' * 5} {'─' * 5} {'─' * 5} {'─' * 7}")
    for sender, row in sender_stats.iterrows():
        sender_short = str(sender)[:25]
        print(f"  {sender_short:<25s} {int(row['total']):>5d} "
              f"{int(row['clean']):>5d} {int(row['chats']):>5d} "
              f"{row['avg_len']:>7.1f}")

    # ── Time distribution
    print(f"\n{sep}")
    print("  Hourly Distribution (clean messages)")
    print(sep)
    df_clean_timed = DF_ALL[
        (DF_ALL["noise_type"] == "clean") & DF_ALL["hour_int"].notna()].copy()
    if not df_clean_timed.empty:
        df_clean_timed["hour_int"] = df_clean_timed["hour_int"].astype(int)
        hour_counts = df_clean_timed["hour_int"].value_counts().sort_index()
        max_count = hour_counts.max() if len(hour_counts) else 1
        for h in range(24):
            cnt = hour_counts.get(h, 0)
            bar_len = int(cnt / max_count * 40) if max_count else 0
            label = f"{h % 12 or 12:>2d} {'AM' if h < 12 else 'PM'}"
            bar = "█" * bar_len
            print(f"  {label}  {cnt:>3d}  {bar}")

    # ── De-duplication report
    print(f"\n{sep}")
    print("  File De-duplication")
    print(sep)
    all_files, _ = _discover_json_files(DATA_DIR)
    chat_file_count = {}
    for path in all_files:
        try:
            with open(path) as f:
                data = json.load(f)
            if "exportInfo" not in data:
                continue
            name = data["exportInfo"]["chatName"]
            key = _normalize_chat_name(name)
            chat_file_count.setdefault(key, []).append(
                os.path.relpath(path, DATA_DIR))
        except (json.JSONDecodeError, KeyError):
            continue
    dupes = {k: v for k, v in chat_file_count.items() if len(v) > 1}
    if dupes:
        for name, files in sorted(dupes.items()):
            print(f"  {name}: {len(files)} files (kept largest)")
            for fn in files:
                print(f"    - {fn}")
    else:
        print("  No duplicate chat names found.")

    # ── Reporting efficiency
    print(f"\n{sep}")
    print("  Reporting Efficiency (per sender per day)")
    print(sep)
    ds = _build_daily_summary(DF_ALL[DF_ALL["noise_type"] == "clean"],
                              "sender_resolved")
    if not ds.empty:
        print(f"  {'Sender':<25s} {'Days':>4s} {'Avg 1st':>8s} "
              f"{'Avg Window':>10s} {'Avg Msgs':>8s}")
        print(f"  {'─' * 25} {'─' * 4} {'─' * 8} {'─' * 10} {'─' * 8}")
        for sender, grp in ds.groupby("sender"):
            n_days = len(grp)
            avg_first = grp["first_hour"].mean()
            fh = int(avg_first)
            fm = int((avg_first - fh) * 60)
            period = "AM" if fh < 12 else "PM"
            dh = fh % 12 or 12
            avg_win = grp["window_hrs"].mean()
            avg_msgs = grp["msg_count"].mean()
            sender_short = str(sender)[:25]
            print(f"  {sender_short:<25s} {n_days:>4d} "
                  f"{dh:>2d}:{fm:02d} {period} "
                  f"{avg_win:>9.1f}h {avg_msgs:>8.1f}")
    else:
        print("  No daily efficiency data available.")

    # ── Crew Metrics
    print(f"\n{sep}")
    print("  Crew Metrics (site visits per crew/chat)")
    print(sep)
    df_clean = DF_ALL[DF_ALL["noise_type"] == "clean"]
    visits_df = _build_site_visits(df_clean, "chat")
    if not visits_df.empty:
        ds_crew = _build_daily_summary(df_clean, "chat")
        sc = _build_crew_scorecard(visits_df, ds_crew)
        if not sc.empty:
            print(f"  {'Crew':<35s} {'Sites':>5s} {'Sites/Day':>9s} "
                  f"{'Sites/Hr':>8s} {'Avg Trans':>9s} {'Hrs':>5s}")
            print(f"  {'─' * 35} {'─' * 5} {'─' * 9} {'─' * 8} "
                  f"{'─' * 9} {'─' * 5}")
            for _, row in sc.iterrows():
                s = str(row["sender"])[:35]
                print(f"  {s:<35s} {int(row['total_sites']):>5d} "
                      f"{row['avg_sites_per_day']:>9.1f} "
                      f"{row['avg_sites_per_hour']:>8.1f} "
                      f"{row['avg_transition_min']:>8.1f}m "
                      f"{row['total_active_hrs']:>5.1f}")

        # Top locations
        loc_counts = (visits_df["location"].value_counts().head(10)
                      .reset_index())
        loc_counts.columns = ["location", "visits"]
        if not loc_counts.empty:
            print(f"\n  Top 10 Locations:")
            for _, row in loc_counts.iterrows():
                loc = str(row["location"])[:50]
                print(f"    {row['visits']:>3d}  {loc}")
    else:
        print("  No site visit data available.")

    print(f"\n{'═' * 72}")
    print(f"  Dashboard ready at http://0.0.0.0:5000")
    print(f"{'═' * 72}\n")


if __name__ == "__main__":
    print_report()
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
