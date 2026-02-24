# Snow Removal Deployment Tracker

## Overview
This project is a Python Dash web application designed to analyze exported WhatsApp chat data. Its primary purpose is to provide interactive charts and filters for tracking the productivity of snow removal field crews. The application aims to offer insights into operational efficiency, financial performance, and crew activity, leveraging chat data to monitor deployments and identify areas for improvement.

## User Preferences
The user prefers clear and concise information. The agent should focus on delivering functional code and avoid overly verbose explanations. Iterative development with regular check-ins for major architectural decisions is preferred. The user values maintainable and well-structured code.

## System Architecture

### Data Analysis Core
The application is built on Python 3.11, utilizing Plotly Dash for the web interface. Core data analysis is handled by modules responsible for chat analysis, timeline plotting, and generating interactive dashboards. A standalone domain model module (`data/domain_model.py`) encapsulates snow removal specific logic, including job log building, crew summaries, and deployment burndown. Configuration for deployment types, service times, and non-trackable senders is managed via `data/config/snow_removal.json`, while contract pricing is stored in `data/config/pricing.json`.

### Dashboard Structure and Features
The web dashboard is organized into 9 main tabs:
- **Overview**: Message volume, activity heatmaps, daily timelines.
- **Productivity**: Daily scores, crew leaderboard, idle time detection.
- **Crew Analysis**: Per-sender metrics, message gaps, route timelines.
- **Deployments**: Summary, timeline, cross-deployment comparison, downloadable HTML reports, breakdown with crew-location reassignment, per-deployment overrides (crew sizes, route counts).
- **Operations**: Routing Gantt chart, deployment burn-down (supports route overrides), traffic analysis, recall detection.
- **Finances**: Revenue forecasting, cost analysis, profit metrics, invoice reconciliation.
- **Map**: DC Service Map with OpenStreetMap tiles, location color-coding by crew, deployment filtering, marker clustering.
- **Data Quality**: Noise filtering, raw message data.
- **Settings**: Crew assignment configuration, sender management, expected hours, service times, location registry upload, invoice upload.

Key dashboard features include:
- A persistent KPI summary bar.
- Collapsible sidebar filters for granular data selection.
- A weighted productivity scoring system.
- Billable route model: DeploymentID + LocationID + ServiceArea combinations (not just unique addresses).
- Detection of idle time and recall events.
- Automatic location type inference and trackable sender filtering.
- Deployment burn-down charts scaled by crew size, with elapsed-hours overlay for cross-deployment comparison.
- Deployment time windows: configurable start/end date+time per deployment (Settings tab), stored in `deployment_time_windows` config; controls which messages belong to each deployment.
- Per-deployment overrides for crew sizes (SW/PL workers) and total route count (single field matching invoice format), saved to config.
- Map marker clustering (per-crew, auto-clusters at lower zoom levels).
- Crew merging capabilities for consistent tracking.
- Drag-and-drop file upload for various data formats.
- Distance-based travel efficiency analysis.
- Location registry with fuzzy matching and map integration.
- Invoice reconciliation with cross-referencing.
- Full invoice management: batch import, editable fields (type, tier, date, deployment, sites, total), split across deployments, multi-select delete, delete all.
- Stable invoice IDs with auto-detected field preservation (auto_* fields alongside editable overrides).
- Invoice routes are unified (no SW/PL distinction on invoices; split only applies to costs/expenses).
- Invoice-based route creation: invoice line items become routes in the deployment breakdown (Source: Chat/Invoice/Both), with auto-matching between chat locations and invoice addresses.
- Auto-route count from invoices: when no manual override is set, invoice site_count is used as the route count for finance calculations, KPI bar, deployment summary, and burn-down charts. Priority: manual override > invoice site_count (max per deployment) > chat-detected routes. Multiple invoice types per deployment use max (not sum) for site count.

### Dispatch Routes
- Per-deployment crew-to-site assignments stored in `dispatch_routes` within `data/config/snow_removal.json`.
- Imported from structured CSV files with columns: crew_name, site_name, address, ward, lat, lng, stop_order.
- Feb 21-24 CSV: 131 sites, 15 SW crews, plus parking lot crew cross-references (9 sites with PL crew noted).
- Feb 10-11 CSV: 50 sites, 9 crews (8 named + UNASSIGNED group of 5 sites).
- Cross-referenced with location registry; 29 new locations added from Feb 10-11 data.
- "SW Crew 9" in Feb 21-24 data has no named crew leader (9 sites including Triangle Parks, Marie Reed, Belmont Park).
- "Anonio Rivera" is a typo for "Antonio Rivera" in Feb 21-24 data.

### Supported Data Formats
The application supports multiple JSON data formats for chat exports, including unified multi-chat exports, legacy per-chat exports, pre-computed metrics (`metrics_report.json`), and automatically skips OCR dispatch data (`_dispatches.json`). It also includes robust handling for a new `v2 CrewChatData` format, supporting message-level deduplication and detailed message attributes.

## External Dependencies

- **Python Libraries**: `dash`, `dash-bootstrap-components`, `plotly`, `pandas`, `numpy`.
- **Mapping Services**: OpenStreetMap for map tiles, ArcGIS API for DC boundary polygon data.
- **Data Storage**: Configuration and pricing data are stored in JSON files (`data/config/snow_removal.json`, `data/config/pricing.json`).

### Invoice Parser
- `data/invoice_parser.py` - Supports both Excel (.xlsx) and CSV (.csv) invoice files
- Auto-detects file format from extension
- Three invoice formats: simple billing, pre-treatment report, completion report
- Auto-detects deployment type, snow tier, and date from filename/headers
- Skips summary rows (TOTAL, SUBTOTAL, GRAND TOTAL, T&M)
- CSV parsing converts cells to float where possible for price column detection