# Snow Removal Deployment Tracker

## Overview
A Python Dash web application that analyzes exported WhatsApp chat data with interactive charts and filters for snow removal field crew productivity tracking.

## Project Architecture

### Data Analysis (Python)
- `data/dashboard_web.py` - **Main runnable**: Plotly Dash web dashboard (runs on port 5000)
- `data/dashboard.py` - Matplotlib/Seaborn interactive dashboard (desktop only)
- `data/analyze_chat.py` - CLI chat analysis and timeline plotting

### Dependencies
- Python 3.11
- dash, dash-bootstrap-components
- plotly, pandas, numpy

### Domain Model (Snow Removal)
- `data/domain_model.py` - Standalone domain model module (no Dash dependencies)
  - Job log builder, crew summary, route segments, deployment burndown
  - Location type stats, traffic analysis, delay report, recall detection
  - Trackable sender filtering
- `data/config/snow_removal.json` - Configuration for deployment types, location types, expected service times, non-trackable senders, standard travel times

## Dashboard Structure
The dashboard has 7 tabs:
1. **Overview** - Message volume, types, activity heatmap, sender participation, daily timeline
2. **Productivity** - Daily productivity scores, crew leaderboard with rankings, first report times, reporting windows, idle time detection, daily message trends
3. **Crew Analysis** - Per-sender metrics, message gaps, crew scorecards, sites/hour, transition times, route timelines, pace consistency, top locations
4. **Deployments** - Deployment summary, timeline, cross-deployment comparison, performance trends, sites heatmap, downloadable HTML reports
5. **Operations** - Routing Gantt chart, deployment burn-down (actual vs 12hr expected pace), location type performance, traffic analysis, delay report, recall summary
6. **Data Quality** - Noise filtering overview, raw message data table
7. **Settings** - Crew location type assignment (Sidewalk/Parking Lot), non-trackable sender management, expected deployment hours, service time configuration

### Key Dashboard Features
- **KPI Summary Bar**: Always-visible row of 6 metric cards (Total Messages, Active Crews, Avg Sites/Hour, Avg First Report, Total Sites, Avg Transition)
- **Collapsible Sidebar Filters**: Accordion-style with People & Chats, Time & Dates, Message Filters sections
- **Productivity Score**: Weighted composite (50% pace + 30% punctuality + 20% coverage) per crew per day
- **Crew Leaderboard**: Ranked table with trend indicators (improving/declining/stable)
- **Idle Time Detection**: Flags gaps > 45 minutes between consecutive messages
- **Recall Detection**: Flags when crews return to previously-visited locations, tracks added time
- **Location Type Auto-Detection**: Infers Sidewalk/Parking Lot from chat names
- **Trackable Sender Filtering**: Excludes non-trackable senders from Operations KPIs
- **Deployment Burn-Down**: Compares actual site completion pace vs expected (configurable, default 12hr)
- **File Upload**: Drag-and-drop JSON files or upload entire folders
- **Deployment Reports**: Downloadable HTML reports with all deployment charts

## Running
The dashboard runs via `python data/dashboard_web.py` on port 5000. It reads WhatsApp JSON exports from the `data/` directory (files matching `whatsapp_*.json`).

## Data Files
JSON chat exports are gitignored. Place exported files in `data/archive/` or `data/` subdirectories.

## Recent Changes
- 2026-02-23: Fixed CSV and JSON export endpoints
  - Fixed KeyError on 'coverage' column missing from productivity score DataFrame
  - Added coverage column to _build_daily_productivity_score output
  - Fixed unsafe date.date() calls on string values causing AttributeError
  - Added non-trackable sender filtering to export data (matches dashboard filtering)
  - Added missing CSV sections: Transitions, Recalls, Location Type Stats
  - Wrapped export endpoint in try/except for graceful error handling
- 2026-02-23: Non-trackable sender filtering applied globally
  - Non-trackable senders now excluded from ALL metrics (KPIs, Overview, Productivity, Crew Analysis, Deployments, Operations)
  - Filtering applied centrally in get_filtered_df so all callbacks automatically exclude non-trackable senders
  - Sender dropdowns in sidebar no longer show non-trackable senders
  - Removed redundant filter_trackable calls from Operations callbacks
  - Settings tab still shows all senders for non-trackable management
- 2026-02-23: Defensive error handling for all callbacks
  - All Operations callbacks wrapped with try/except (routing gantt, burndown, location type stats, traffic analysis, delay report, recall summary)
  - chart-sender-chat callback wrapped with error handling and limited to top 50 combinations
  - Domain model build_job_logs validates required columns before processing
  - All errors logged to console with descriptive prefixes for debugging
  - Callbacks return empty figures/placeholders on error instead of crashing
- 2026-02-23: Settings tab and cleanup
  - Added Settings tab with crew location type assignment (Sidewalk/Parking Lot/Auto-detect)
  - Added non-trackable sender management via checkboxes
  - Added expected deployment hours and service time configuration
  - Settings persist to data/config/snow_removal.json
  - Removed Chrome extension files (no longer used)
- 2026-02-23: Snow Removal Operations integration
  - Created domain model module (data/domain_model.py) with formalized entities
  - Added configuration system (data/config/snow_removal.json)
  - New Operations tab with 6 sections: Routing Gantt, Burn-Down, Location Type Performance, Traffic Analysis, Delay Report, Recall Summary
  - Recall detection system flagging crews returning to previously-visited locations
  - Trackable sender filtering applied to all Operations KPIs
  - Route segments include actual start/end timestamps for accurate Gantt charts
  - Burndown uses configurable expected_deployment_hours from config
  - Export API includes recalls and location type stats
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
