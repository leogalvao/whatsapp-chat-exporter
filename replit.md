# WhatsApp Chat Exporter

## Overview
This project consists of two components:
1. **Chrome Extension**: A WhatsApp Web chat exporter that exports chats to JSON/CSV/TXT formats with organized media folders
2. **Data Dashboard**: A Python Dash web application that analyzes exported WhatsApp chat data with interactive charts and filters for field crew productivity tracking

## Project Architecture

### Chrome Extension Files
- `manifest.json` - Chrome extension manifest (v3)
- `popup.html` / `popup.js` / `popup.css` - Extension popup UI
- `content.js` / `content-styles.css` - Content script injected into WhatsApp Web
- `background.js` - Service worker for handling downloads
- `icons/` - Extension icons

### Data Analysis (Python)
- `data/dashboard_web.py` - **Main runnable**: Plotly Dash web dashboard (runs on port 5000)
- `data/dashboard.py` - Matplotlib/Seaborn interactive dashboard (desktop only)
- `data/analyze_chat.py` - CLI chat analysis and timeline plotting

### Dependencies
- Python 3.11
- dash, dash-bootstrap-components
- plotly, pandas, numpy

## Dashboard Structure
The dashboard has 5 tabs:
1. **Overview** - Message volume, types, activity heatmap, sender participation, daily timeline
2. **Productivity** - Daily productivity scores, crew leaderboard with rankings, first report times, reporting windows, idle time detection, daily message trends
3. **Crew Analysis** - Per-sender metrics, message gaps, crew scorecards, sites/hour, transition times, route timelines, pace consistency, top locations
4. **Deployments** - Deployment summary, timeline, cross-deployment comparison, performance trends, sites heatmap, downloadable HTML reports
5. **Data Quality** - Noise filtering overview, raw message data table

### Key Dashboard Features
- **KPI Summary Bar**: Always-visible row of 6 metric cards (Total Messages, Active Crews, Avg Sites/Hour, Avg First Report, Total Sites, Avg Transition)
- **Collapsible Sidebar Filters**: Accordion-style with People & Chats, Time & Dates, Message Filters sections
- **Productivity Score**: Weighted composite (50% pace + 30% punctuality + 20% coverage) per crew per day
- **Crew Leaderboard**: Ranked table with trend indicators (improving/declining/stable)
- **Idle Time Detection**: Flags gaps > 45 minutes between consecutive messages
- **File Upload**: Drag-and-drop JSON files or upload entire folders
- **Deployment Reports**: Downloadable HTML reports with all deployment charts

## Running
The dashboard runs via `python data/dashboard_web.py` on port 5000. It reads WhatsApp JSON exports from the `data/` directory (files matching `whatsapp_*.json`).

## Data Files
JSON chat exports are gitignored. Place exported files in `data/archive/` or `data/` subdirectories.

## Recent Changes
- 2026-02-23: Major dashboard restructuring
  - Reorganized from 7 tabs to 5 (Overview, Productivity, Crew Analysis, Deployments, Data Quality)
  - Added KPI summary bar with 6 at-a-glance metric cards
  - Added productivity scoring system with weighted daily scores
  - Added crew leaderboard with trend indicators
  - Added idle time detection (gaps > 45 min)
  - Made sidebar collapsible with accordion sections
  - Added section headers with descriptions to each tab
  - Fixed chart-sender-chat ValueError with uniform data
- 2026-02-23: Imported from GitHub, configured for Replit environment
  - Changed dashboard port from 8050 to 5000
  - Added empty data handling so dashboard starts without JSON files
