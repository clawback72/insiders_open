# Insider Trading Research Pipeline

A Python data pipeline that collects, normalizes, and analyzes publicly available U.S. corporate insider trading disclosures (SEC Form 4 filings) and related market pricing data.

The long-term goal of this project is to build a research dataset and web interface for studying insider transaction behavior and its relationship to stock performance.

---

## What This Project Does

The system performs three major tasks:

1. **Scrape Insider Transactions**
   - Pulls insider trading activity from OpenInsider
   - Captures the original SEC Form 4 filing URL
   - Normalizes transaction fields (dates, quantities, codes, etc.)
   - Stores transactions idempotently (no duplicates)

2. **Maintain Security Reference Data**
   - Tracks newly discovered tickers
   - Enriches securities with metadata (sector, exchange, etc.)
   - Maintains first_seen / last_seen timestamps

3. **Market Data Collection (in progress)**
   - Backfills historical price data
   - Tracks OHLCV + adjusted prices
   - Records corporate actions (splits/dividends)
   - Enables proper historical comparisons

---

## Tech Stack

- Python 3
- SQLite
- Pandas
- BeautifulSoup / Requests
- yfinance
- Flask (planned analytics UI)

---

## Database Design

The database is structured as an analytics-friendly relational model.

### Core Tables

| Table | Purpose |
|------|------|
| `insider_data` | Individual insider transactions |
| `ticker_data` | Security reference information |
| `pricing_data` | Daily OHLCV + adjusted pricing |
| `corporate_actions` | Splits and dividends |
| `runs` | Job execution logging |

The dataset is designed to support historical research — not just daily monitoring.

---

## Running the Pipeline

### 1. Initialize database

```
python init_db.py
```

### 2. Run insider scraper

```
python inside_scrape.py
```

### 3. (Planned) Run pricing job

```
python price.py
```

---

## Key Design Decisions

### Idempotent ingestion
Transactions are stored using a generated `transaction_id` derived from the SEC filing URL and transaction fields.  
This allows the scraper to run repeatedly without creating duplicates.

### Adjusted pricing
Historical analysis uses adjusted prices rather than recalculating split adjustments manually.  
Corporate actions are tracked separately for auditability.

### Separation of concerns
- Scraper discovers events
- Pricing job maintains market state

This allows rebuilding market data without re-scraping filings.

---

## Project Status

Current stage: **data acquisition and normalization**

Planned stages:

- Historical performance analytics
- Insider signal classification
- Web dashboard (Flask)
- Research blog posts based on dataset findings

---

## Data Source Disclaimer

All data originates from publicly available SEC filings and public market data providers.  
This project is for research and educational purposes only and does not provide investment advice.

