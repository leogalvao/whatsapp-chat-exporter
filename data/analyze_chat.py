#!/usr/bin/env python3
"""Analyze WhatsApp chat export: display message table and plot message timeline."""

import json
import sys
from datetime import datetime
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

JSON_FILE = "whatsapp_Israel_Sidewalks_2026-02-08.json"


def parse_timestamp(ts_str):
    """Parse a timestamp like '2:29 PM' into a datetime object (date fixed to export day)."""
    if not ts_str:
        return None
    try:
        t = datetime.strptime(ts_str.strip(), "%I:%M %p")
        # Attach the export date so matplotlib can plot it on a time axis
        return t.replace(year=2026, month=2, day=8)
    except ValueError:
        return None


def load_messages(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def print_table(messages):
    """Print a formatted table of messages."""
    header = f"{'#':<4} {'Time':<10} {'Sender':<16} {'Type':<7} {'Content (truncated)'}"
    sep = "-" * 90
    print(sep)
    print(header)
    print(sep)
    idx = 0
    for msg in messages:
        ts = msg.get("timestamp", "")
        sender = msg.get("sender", "") or "(system)"
        mtype = msg.get("type", "")
        content = msg.get("content", "") or ""
        # Truncate long content for display
        content_short = content[:50].replace("\n", " ")
        if len(content) > 50:
            content_short += "..."
        if not ts:
            continue  # skip metadata rows with no timestamp
        idx += 1
        print(f"{idx:<4} {ts:<10} {sender:<16} {mtype:<7} {content_short}")
    print(sep)
    print(f"Total displayed: {idx}")


def plot_timeline(messages):
    """Plot message time vs cumulative messages per sender."""
    # Collect timestamped messages grouped by sender
    sender_times = defaultdict(list)
    for msg in messages:
        ts = parse_timestamp(msg.get("timestamp", ""))
        sender = msg.get("sender", "")
        if ts is None or not sender:
            continue
        sender_times[sender].append(ts)

    # Sort each sender's timestamps
    for sender in sender_times:
        sender_times[sender].sort()

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10.colors
    for i, (sender, times) in enumerate(sorted(sender_times.items())):
        cumulative = list(range(1, len(times) + 1))
        color = colors[i % len(colors)]
        ax.step(times, cumulative, where="post", label=sender, color=color,
                linewidth=2, marker="o", markersize=5)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%I:%M %p"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    fig.autofmt_xdate(rotation=45)

    ax.set_xlabel("Time of Day")
    ax.set_ylabel("Cumulative Messages Sent")
    ax.set_title("WhatsApp Messages Timeline — Israel Sidewalks")
    ax.legend(title="Sender")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("chat_timeline.png", dpi=150)
    print("\nGraph saved to chat_timeline.png")
    plt.show()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else JSON_FILE
    data = load_messages(path)
    messages = data.get("messages", [])

    print(f"\nChat: {data['exportInfo']['chatName']}")
    print(f"Export date: {data['exportInfo']['exportDate']}")
    print(f"Total messages: {data['exportInfo']['totalMessages']}")
    print(f"Total media: {data['exportInfo']['totalMedia']}\n")

    print_table(messages)
    plot_timeline(messages)


if __name__ == "__main__":
    main()
