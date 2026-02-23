# WhatsApp Chat Exporter

## Overview
This project consists of two components:
1. **Chrome Extension**: A WhatsApp Web chat exporter that exports chats to JSON/CSV/TXT formats with organized media folders
2. **Data Dashboard**: A Python Dash web application that analyzes exported WhatsApp chat data with interactive charts and filters

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

## Running
The dashboard runs via `python data/dashboard_web.py` on port 5000. It reads WhatsApp JSON exports from the `data/` directory (files matching `whatsapp_*.json`).

## Data Files
JSON chat exports are gitignored. Place exported files in `data/archive/` or `data/` subdirectories.

## Recent Changes
- 2026-02-23: Imported from GitHub, configured for Replit environment
  - Changed dashboard port from 8050 to 5000
  - Added empty data handling so dashboard starts without JSON files
