#!/usr/bin/env python3
"""Interactive WhatsApp chat dashboard using Seaborn + matplotlib widgets.
Reads all whatsapp_*.json exports and provides rich filtering and chart types."""

import glob
import json
import os
from datetime import datetime, date
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import CheckButtons, RadioButtons

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Seaborn theme ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="notebook", font_scale=0.9,
              rc={"axes.facecolor": "#fafafa", "figure.facecolor": "#f0f0f0"})

PALETTE_NAME = "tab10"


# ── Data loading ─────────────────────────────────────────────────────────────

EXCLUDE_DIRS = {".claude", "__pycache__", "node_modules"}


def parse_export_date(iso_str):
    """Extract date from ISO 8601 exportDate string."""
    try:
        return datetime.fromisoformat(iso_str.replace("Z", "+00:00")).date()
    except (ValueError, AttributeError):
        return date(2026, 2, 8)


def parse_msg_datetime(msg, export_date=None):
    """Resolve a message's datetime using the best available field.

    Priority:
      1. ``dateTime`` (ISO 8601, e.g. "2026-02-10T14:40:00")
      2. ``date`` + ``timestamp`` combined
      3. ``timestamp`` + *export_date* fallback
    """
    dt_str = msg.get("dateTime")
    if dt_str:
        try:
            return datetime.fromisoformat(dt_str)
        except (ValueError, TypeError):
            pass

    ts_str = (msg.get("timestamp") or "").strip()
    date_str = msg.get("date")

    if date_str and ts_str:
        try:
            d = date.fromisoformat(date_str)
            t = datetime.strptime(ts_str, "%I:%M %p")
            return t.replace(year=d.year, month=d.month, day=d.day)
        except (ValueError, TypeError):
            pass

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


def _discover_json_files(data_dir):
    """Find all WhatsApp JSON exports under *data_dir*."""
    paths = []
    for p in glob.glob(os.path.join(data_dir, "archive", "whatsapp_*.json")):
        paths.append(p)
    for root, dirs, files in os.walk(data_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and d != "archive"]
        for fn in files:
            if fn.endswith(".json"):
                paths.append(os.path.join(root, fn))
    return sorted(set(paths))


def load_all_chats(data_dir):
    chats = {}
    for path in _discover_json_files(data_dir):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if "exportInfo" not in data or "messages" not in data:
                continue
        except (json.JSONDecodeError, KeyError):
            continue
        name = data["exportInfo"]["chatName"]
        msgs = data["messages"]
        export_date = parse_export_date(data["exportInfo"].get("exportDate", ""))
        if name not in chats or len(msgs) > len(chats[name][0]):
            chats[name] = (msgs, export_date)
    return chats


def build_dataframe(chats):
    """Build a single DataFrame from all chats."""
    rows = []
    for chat_name, (msgs, export_date) in chats.items():
        for m in msgs:
            sender = m.get("sender", "")
            ts = parse_msg_datetime(m, export_date)
            if ts is None or not sender:
                continue
            content = m.get("content", "") or ""
            msg_date = ts.date() if ts else None
            rows.append({
                "chat":     chat_name,
                "sender":   sender,
                "time":     ts,
                "hour":     ts.hour + ts.minute / 60.0,
                "hour_int": ts.hour,
                "msg_date": msg_date,
                "type":     m.get("type", "text"),
                "content_len": len(content),
            })
    return pd.DataFrame(rows)


# ── Filter constants ─────────────────────────────────────────────────────────
TIME_RANGES = {
    "All day":    (0, 24),
    "AM (0-12)":  (0, 12),
    "PM (12-24)": (12, 24),
    "Morning":    (6, 12),
    "Afternoon":  (12, 18),
    "Evening":    (18, 24),
    "Night":      (0, 6),
}

CONTENT_LEN = {
    "Any length":  (0, 999999),
    "Short (<20)": (0, 20),
    "Medium":      (20, 100),
    "Long (>100)": (100, 999999),
}


# ── Dashboard ────────────────────────────────────────────────────────────────
class Dashboard:
    def __init__(self, df_all):
        self.df_all = df_all

        self.chat_names = sorted(df_all["chat"].unique())
        self.all_senders = sorted(df_all["sender"].unique())
        self.all_types = sorted(df_all["type"].unique())

        # Active filters
        self.active_chats = set(self.chat_names)
        self.active_senders = set(self.all_senders)
        self.active_types = set(self.all_types)
        self.time_range = "All day"
        self.content_len = "Any length"

        # Chart settings
        self.chart_mode = "Cumulative"
        self.y_scale = "Linear"
        self.group_by = "Sender"

        self._build_ui()
        self._redraw()

    # ── Filtered DataFrame ───────────────────────────────────────────────
    def _filtered_df(self):
        df = self.df_all
        df = df[df["chat"].isin(self.active_chats)]
        df = df[df["sender"].isin(self.active_senders)]
        df = df[df["type"].isin(self.active_types)]

        lo_h, hi_h = TIME_RANGES[self.time_range]
        df = df[(df["hour"] >= lo_h) & (df["hour"] < hi_h)]

        lo_cl, hi_cl = CONTENT_LEN[self.content_len]
        df = df[(df["content_len"] >= lo_cl) & (df["content_len"] < hi_cl)]

        return df.copy()

    @property
    def _hue_col(self):
        return {"Sender": "sender", "Chat": "chat", "Type": "type"}[self.group_by]

    # ── UI layout — 3 control columns + plot ─────────────────────────────
    def _build_ui(self):
        self.fig = plt.figure(figsize=(20, 11))
        self.fig.canvas.manager.set_window_title(
            "WhatsApp Chat Dashboard (Seaborn)")

        # Main plot area
        self.ax = self.fig.add_axes([0.32, 0.08, 0.66, 0.84])

        # Geometry
        ITEM_H  = 0.028
        TITLE_H = 0.022
        GAP     = 0.015
        TOP     = 0.96
        COL_W   = 0.095
        COL_X   = [0.005, 0.105, 0.205]

        def _h(n):
            return TITLE_H + ITEM_H * n

        def _place(col, cur, title, n):
            h = _h(n)
            cur -= h
            ax = self.fig.add_axes([COL_X[col], cur, COL_W, h])
            ax.set_title(title, fontsize=8, fontweight="bold", loc="left",
                         pad=2)
            return ax, cur - GAP

        # ── COL 0 — data filters ────────────────────────────────────────
        c0 = TOP
        ax, c0 = _place(0, c0, "Chats", len(self.chat_names))
        self.chk_chats = CheckButtons(
            ax, self.chat_names, [True] * len(self.chat_names))
        self.chk_chats.on_clicked(self._on_chat_toggle)

        ax, c0 = _place(0, c0, "Senders", len(self.all_senders))
        self.chk_senders = CheckButtons(
            ax, self.all_senders, [True] * len(self.all_senders))
        self.chk_senders.on_clicked(self._on_sender_toggle)

        ax, c0 = _place(0, c0, "Msg Type", len(self.all_types))
        self.chk_types = CheckButtons(
            ax, self.all_types, [True] * len(self.all_types))
        self.chk_types.on_clicked(self._on_type_toggle)

        # ── COL 1 — more filters ────────────────────────────────────────
        c1 = TOP
        time_labels = list(TIME_RANGES.keys())
        ax, c1 = _place(1, c1, "Time of Day", len(time_labels))
        self.radio_time = RadioButtons(ax, time_labels)
        self.radio_time.on_clicked(self._on_time_change)

        clen_labels = list(CONTENT_LEN.keys())
        ax, c1 = _place(1, c1, "Content Length", len(clen_labels))
        self.radio_clen = RadioButtons(ax, clen_labels)
        self.radio_clen.on_clicked(self._on_clen_change)

        # ── COL 2 — chart options ────────────────────────────────────────
        c2 = TOP
        modes = [
            "Cumulative", "Scatter", "Bar", "Stacked Bar",
            "Pie", "Histogram", "Heatmap", "Area",
            "Violin", "Box", "Swarm", "KDE",
        ]
        ax, c2 = _place(2, c2, "Chart Type", len(modes))
        self.radio_mode = RadioButtons(ax, modes)
        self.radio_mode.on_clicked(self._on_mode_change)

        scales = ["Linear", "Log (Y)", "SymLog (Y)"]
        ax, c2 = _place(2, c2, "Y Scale", len(scales))
        self.radio_scale = RadioButtons(ax, scales)
        self.radio_scale.on_clicked(self._on_scale_change)

        groups = ["Sender", "Chat", "Type"]
        ax, c2 = _place(2, c2, "Group By", len(groups))
        self.radio_grp = RadioButtons(ax, groups)
        self.radio_grp.on_clicked(self._on_group_change)

    # ── Callbacks ────────────────────────────────────────────────────────
    def _toggle(self, s, label):
        s.discard(label) if label in s else s.add(label)

    def _on_chat_toggle(self, l):   self._toggle(self.active_chats, l);   self._redraw()
    def _on_sender_toggle(self, l): self._toggle(self.active_senders, l); self._redraw()
    def _on_type_toggle(self, l):   self._toggle(self.active_types, l);   self._redraw()
    def _on_time_change(self, l):   self.time_range = l;   self._redraw()
    def _on_clen_change(self, l):   self.content_len = l;  self._redraw()
    def _on_mode_change(self, l):   self.chart_mode = l;   self._redraw()
    def _on_scale_change(self, l):  self.y_scale = l;      self._redraw()
    def _on_group_change(self, l):  self.group_by = l;     self._redraw()

    # ── Drawing ──────────────────────────────────────────────────────────
    def _redraw(self):
        self.ax.clear()
        # Remove leftover colorbars from heatmap
        for cb in getattr(self, "_colorbars", []):
            cb.remove()
        self._colorbars = []

        df = self._filtered_df()
        hue = self._hue_col

        dispatch = {
            "Cumulative":  self._draw_cumulative,
            "Scatter":     self._draw_scatter,
            "Bar":         self._draw_bar,
            "Stacked Bar": self._draw_stacked_bar,
            "Pie":         self._draw_pie,
            "Histogram":   self._draw_histogram,
            "Heatmap":     self._draw_heatmap,
            "Area":        self._draw_area,
            "Violin":      self._draw_violin,
            "Box":         self._draw_box,
            "Swarm":       self._draw_swarm,
            "KDE":         self._draw_kde,
        }
        draw_fn = dispatch.get(self.chart_mode, self._draw_cumulative)
        draw_fn(df, hue)

        # Apply scale (skip where inapplicable)
        if self.chart_mode not in ("Pie", "Heatmap", "Scatter", "Swarm"):
            self._apply_scale()

        chats_str = ", ".join(sorted(self.active_chats)) or "None"
        if len(chats_str) > 80:
            chats_str = chats_str[:77] + "..."
        scale_tag = f" [{self.y_scale}]" if self.y_scale != "Linear" else ""
        self.ax.set_title(f"WhatsApp Dashboard — {chats_str}{scale_tag}",
                          fontsize=10)
        self.fig.canvas.draw_idle()

    def _apply_scale(self):
        if self.y_scale == "Log (Y)":
            self.ax.set_yscale("log")
            self.ax.yaxis.set_major_formatter(plt.ScalarFormatter())
        elif self.y_scale == "SymLog (Y)":
            self.ax.set_yscale("symlog", linthresh=1)
            self.ax.yaxis.set_major_formatter(plt.ScalarFormatter())

    def _time_axis(self):
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%I:%M %p"))
        self.ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        self.fig.autofmt_xdate(rotation=45)

    def _no_data(self):
        self.ax.text(0.5, 0.5, "No data for current filters",
                     transform=self.ax.transAxes, ha="center", va="center",
                     fontsize=14, color="gray")

    def _palette(self, df, hue):
        keys = sorted(df[hue].unique())
        return dict(zip(keys, sns.color_palette(PALETTE_NAME, len(keys))))

    # ── 1. Cumulative step ───────────────────────────────────────────────
    def _draw_cumulative(self, df, hue):
        if df.empty:
            self._no_data(); return
        pal = self._palette(df, hue)
        for key in sorted(df[hue].unique()):
            sub = df[df[hue] == key].sort_values("time")
            cumul = range(1, len(sub) + 1)
            self.ax.step(sub["time"].values, list(cumul), where="post",
                         label=key, color=pal[key], linewidth=2,
                         marker="o", markersize=4)
        self.ax.set_ylabel("Cumulative Messages")
        self.ax.set_xlabel("Time of Day")
        self.ax.legend(title=self.group_by, loc="upper left", fontsize=7)
        self._time_axis()

    # ── 2. Scatter (seaborn scatterplot) ─────────────────────────────────
    def _draw_scatter(self, df, hue):
        if df.empty:
            self._no_data(); return
        pal = self._palette(df, hue)
        sns.scatterplot(data=df, x="time", y=hue, hue=hue, size="content_len",
                        sizes=(40, 300), palette=pal, alpha=0.8,
                        edgecolor="white", ax=self.ax, legend="brief")
        self.ax.set_xlabel("Time of Day")
        self.ax.set_ylabel(self.group_by)
        self.ax.legend(title=self.group_by, loc="upper left", fontsize=7,
                       ncol=2)
        self._time_axis()

    # ── 3. Bar (seaborn countplot) ───────────────────────────────────────
    def _draw_bar(self, df, hue):
        if df.empty:
            self._no_data(); return
        pal = self._palette(df, hue)
        order = df[hue].value_counts().index.tolist()
        sns.countplot(data=df, x=hue, hue=hue, palette=pal, order=order,
                      ax=self.ax, legend=False)
        # Annotate counts
        for p in self.ax.patches:
            h = p.get_height()
            if h > 0:
                self.ax.text(p.get_x() + p.get_width() / 2, h + 0.3,
                             str(int(h)), ha="center", fontsize=9,
                             fontweight="bold")
        self.ax.set_ylabel("Message Count")
        self.ax.set_xlabel(self.group_by)
        self.ax.tick_params(axis="x", rotation=35)

    # ── 4. Stacked bar ──────────────────────────────────────────────────
    def _draw_stacked_bar(self, df, hue):
        if df.empty:
            self._no_data(); return
        if hue == "sender":
            primary, stack = "sender", "chat"
        elif hue == "chat":
            primary, stack = "chat", "sender"
        else:
            primary, stack = "type", "sender"

        ct = pd.crosstab(df[primary], df[stack])
        pal = sns.color_palette(PALETTE_NAME, ct.shape[1])
        ct.plot.bar(stacked=True, color=pal, ax=self.ax, width=0.7)
        self.ax.set_ylabel("Message Count")
        self.ax.set_xlabel(self.group_by)
        self.ax.legend(title=stack.title(), fontsize=7, loc="upper right")
        self.ax.tick_params(axis="x", rotation=35)

    # ── 5. Pie ───────────────────────────────────────────────────────────
    def _draw_pie(self, df, hue):
        if df.empty:
            self._no_data(); return
        counts = df[hue].value_counts().sort_index()
        pal = sns.color_palette(PALETTE_NAME, len(counts))
        wedges, texts, autotexts = self.ax.pie(
            counts.values, labels=counts.index, colors=pal,
            autopct="%1.1f%%", startangle=140, pctdistance=0.8)
        for t in autotexts:
            t.set_fontsize(9); t.set_fontweight("bold")
        self.ax.set_ylabel("")

    # ── 6. Histogram (seaborn histplot) ──────────────────────────────────
    def _draw_histogram(self, df, hue):
        if df.empty:
            self._no_data(); return
        pal = self._palette(df, hue)
        sns.histplot(data=df, x="hour", hue=hue, palette=pal,
                     bins=np.arange(0, 25, 1), multiple="stack",
                     edgecolor="white", ax=self.ax)
        self.ax.set_xlabel("Hour of Day")
        self.ax.set_ylabel("Message Count")
        self.ax.set_xticks(range(0, 25, 2))
        self.ax.set_xticklabels(
            [f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}"
             for h in range(0, 25, 2)], rotation=45, ha="right")
        self.ax.legend(title=self.group_by, fontsize=7, loc="upper right")

    # ── 7. Heatmap (seaborn heatmap) ─────────────────────────────────────
    def _draw_heatmap(self, df, hue):
        if df.empty:
            self._no_data(); return
        ct = pd.crosstab(df[hue], df["hour_int"])
        # Ensure all 24 hours present
        for h in range(24):
            if h not in ct.columns:
                ct[h] = 0
        ct = ct[sorted(ct.columns)]
        ct.columns = [f"{h % 12 or 12}{'a' if h < 12 else 'p'}"
                      for h in ct.columns]

        if self.y_scale == "Log (Y)":
            ct_display = np.log1p(ct)
            fmt = ".1f"
        elif self.y_scale == "SymLog (Y)":
            ct_display = np.sign(ct) * np.log1p(np.abs(ct))
            fmt = ".1f"
        else:
            ct_display = ct
            fmt = "d"

        sns.heatmap(ct_display, annot=ct, fmt=fmt, cmap="YlOrRd",
                    linewidths=0.5, ax=self.ax, cbar_kws={"shrink": 0.7})
        self.ax.set_xlabel("Hour of Day")
        self.ax.set_ylabel(self.group_by)

    # ── 8. Stacked area ─────────────────────────────────────────────────
    def _draw_area(self, df, hue):
        if df.empty:
            self._no_data(); return
        keys = sorted(df[hue].unique())
        pal = sns.color_palette(PALETTE_NAME, len(keys))
        all_times = sorted(df["time"].unique())

        stacks = []
        for key in keys:
            sub = df[df[hue] == key].sort_values("time")
            times_list = sub["time"].tolist()
            c, idx, series = 0, 0, []
            for t in all_times:
                while idx < len(times_list) and times_list[idx] <= t:
                    c += 1; idx += 1
                series.append(c)
            stacks.append(series)

        self.ax.stackplot(all_times, stacks, labels=keys, colors=pal,
                          alpha=0.85)
        self.ax.set_ylabel("Cumulative Messages (stacked)")
        self.ax.set_xlabel("Time of Day")
        self.ax.legend(title=self.group_by, loc="upper left", fontsize=7)
        self._time_axis()

    # ── 9. Violin (seaborn) — content length distribution ────────────────
    def _draw_violin(self, df, hue):
        if df.empty or df[hue].nunique() < 1:
            self._no_data(); return
        pal = self._palette(df, hue)
        sns.violinplot(data=df, x=hue, y="hour", hue=hue,
                       palette=pal, inner="quart", density_norm="width",
                       ax=self.ax, legend=False)
        self.ax.set_ylabel("Hour of Day")
        self.ax.set_xlabel(self.group_by)
        self.ax.tick_params(axis="x", rotation=35)

    # ── 10. Box plot (seaborn) — message timing distribution ─────────────
    def _draw_box(self, df, hue):
        if df.empty:
            self._no_data(); return
        pal = self._palette(df, hue)
        sns.boxplot(data=df, x=hue, y="hour", hue=hue, palette=pal,
                    ax=self.ax, legend=False, width=0.5)
        self.ax.set_ylabel("Hour of Day")
        self.ax.set_xlabel(self.group_by)
        self.ax.tick_params(axis="x", rotation=35)

    # ── 11. Swarm plot (seaborn) — every message as a dot ────────────────
    def _draw_swarm(self, df, hue):
        if df.empty:
            self._no_data(); return
        # Limit to 300 points for performance
        plot_df = df if len(df) <= 300 else df.sample(300, random_state=42)
        pal = self._palette(plot_df, hue)
        sns.swarmplot(data=plot_df, x=hue, y="hour", hue=hue, palette=pal,
                      size=5, ax=self.ax, legend=False)
        self.ax.set_ylabel("Hour of Day")
        self.ax.set_xlabel(self.group_by)
        self.ax.tick_params(axis="x", rotation=35)

    # ── 12. KDE (seaborn) — message density over time ────────────────────
    def _draw_kde(self, df, hue):
        if df.empty or len(df) < 2:
            self._no_data(); return
        pal = self._palette(df, hue)
        for key in sorted(df[hue].unique()):
            sub = df[df[hue] == key]
            if len(sub) >= 2:
                sns.kdeplot(data=sub, x="hour", color=pal[key], label=key,
                            fill=True, alpha=0.3, linewidth=2, ax=self.ax,
                            bw_adjust=0.8, clip=(0, 24))
        self.ax.set_xlabel("Hour of Day")
        self.ax.set_ylabel("Density")
        self.ax.set_xlim(0, 24)
        self.ax.set_xticks(range(0, 25, 2))
        self.ax.set_xticklabels(
            [f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}"
             for h in range(0, 25, 2)], rotation=45, ha="right")
        self.ax.legend(title=self.group_by, fontsize=7, loc="upper right")

    # ── Show ─────────────────────────────────────────────────────────────
    def show(self):
        plt.show()


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    chats = load_all_chats(DATA_DIR)
    if not chats:
        print("No JSON chat files found in", DATA_DIR)
        return

    df = build_dataframe(chats)
    print(f"Loaded {len(chats)} chat(s), {len(df)} messages total:")
    for name in sorted(chats.keys()):
        sub = df[df["chat"] == name]
        senders = ", ".join(sorted(sub["sender"].unique()))
        dates = sorted(sub["msg_date"].dropna().unique()) if "msg_date" in sub.columns else []
        date_range = f", dates: {dates[0]}..{dates[-1]}" if dates else ""
        print(f"  - {name}: {len(sub)} msgs, senders: {senders}{date_range}")

    print("\nOpening interactive dashboard (Seaborn)...")
    dashboard = Dashboard(df)
    dashboard.show()


if __name__ == "__main__":
    main()
