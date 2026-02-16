PRAGMA foreign_keys = ON;

-- =========================
-- Reference / Dimensions
-- =========================

CREATE TABLE IF NOT EXISTS ticker_data (
  ticker           TEXT PRIMARY KEY,
  company_name     TEXT,
  short_name       TEXT,
  sector           TEXT,
  industry         TEXT,
  exchange         TEXT,
  website          TEXT,
  cik              TEXT,            -- for later move to SEC as source
  currency         TEXT,
  country          TEXT,
  first_seen_at    TEXT NOT NULL DEFAULT (datetime('now')),
  last_seen_at     TEXT NOT NULL DEFAULT (datetime('now')),
  status           TEXT,
  status_reason    TEXT,
  last_price_date  TEXT,
  last_price_attempt_at TEXT,
  fail_count.      INTEGER DEFAULT 0    
);

CREATE TABLE IF NOT EXISTS index_data (
  ticker       TEXT PRIMARY KEY,
  index_name   TEXT,
  short_name   TEXT
);

-- =========================
-- Operational logging
-- =========================

CREATE TABLE IF NOT EXISTS runs (
  run_id            INTEGER PRIMARY KEY AUTOINCREMENT,
  job_name          TEXT NOT NULL,        -- e.g. 'inside_scrape' or 'price_pull'
  started_at        TEXT NOT NULL DEFAULT (datetime('now')),
  finished_at       TEXT,
  status            TEXT NOT NULL DEFAULT 'RUNNING', -- RUNNING / SUCCESS / FAILED
  rows_inserted     INTEGER NOT NULL DEFAULT 0,
  rows_updated      INTEGER NOT NULL DEFAULT 0,
  error_message     TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_job_started
ON runs (job_name, started_at);

-- =========================
-- Insider transactions
-- =========================

CREATE TABLE IF NOT EXISTS insider_data (
  transaction_id   TEXT PRIMARY KEY,
  source           TEXT NOT NULL DEFAULT 'openinsider',
  scraped_at       TEXT NOT NULL DEFAULT (datetime('now')),

  filing_date      TEXT NOT NULL CHECK (length(filing_date)=10),
  trade_date       TEXT CHECK (trade_date IS NULL OR length(trade_date)=10),
  ticker           TEXT NOT NULL,

  insider_name     TEXT,
  title            TEXT,

  -- raw label + normalized code
  trade_type       TEXT,      -- e.g. 'S - Sale+OE'
  sec_tx_code      TEXT,      -- e.g. 'S', 'P', 'F'
  x_flags          TEXT,      -- raw flags like 'DM', 'A', etc.

  is_open_market   INTEGER CHECK (is_open_market IN (0,1)),
  classification   TEXT CHECK (classification IN ('OPEN_MARKET','OPTION_EXERCISE','OTHER','UNKNOWN')),

  price            REAL CHECK (price IS NULL OR price >= 0),
  qty              INTEGER CHECK (qty IS NULL OR qty >= 0),
  value            REAL,

  -- amendment/versioning support
  is_active        INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0,1)),
  superseded_by    TEXT,
  superseded_at    TEXT,

  source_url       TEXT NOT NULL,

  FOREIGN KEY (ticker) REFERENCES ticker_data (ticker)
);

CREATE INDEX IF NOT EXISTS idx_insider_active
ON insider_data (is_active);

CREATE INDEX IF NOT EXISTS idx_insider_ticker_trade_date
ON insider_data (ticker, trade_date);

CREATE INDEX IF NOT EXISTS idx_insider_ticker_filing_date
ON insider_data (ticker, filing_date);

CREATE INDEX IF NOT EXISTS idx_insider_insider_name
ON insider_data (insider_name);

CREATE INDEX IF NOT EXISTS idx_insider_active_filing_date
ON insider_data (is_active, filing_date);

-- fast lookup for “what should this amendment supersede?”
CREATE INDEX IF NOT EXISTS idx_insider_active_natural_key
ON insider_data (is_active, ticker, insider_name, trade_date, sec_tx_code);


-- =========================
-- Daily prices (OHLCV + adj_close)
-- =========================

CREATE TABLE IF NOT EXISTS pricing_data (
  ticker        TEXT NOT NULL,
  date          TEXT NOT NULL CHECK (length(date)=10),            -- YYYY-MM-DD
  open          REAL,
  high          REAL,
  low           REAL,
  close         REAL,
  adj_close     REAL,
  volume        INTEGER CHECK (volume IS NULL OR volume >=0),
  source        TEXT NOT NULL DEFAULT 'yahoo',
  fetched_at    TEXT NOT NULL DEFAULT (datetime('now')),

  PRIMARY KEY (ticker, date),
  FOREIGN KEY (ticker) REFERENCES ticker_data (ticker)
);

CREATE INDEX IF NOT EXISTS idx_pricing_date ON pricing_data(date);

-- =========================
-- Corporate actions (splits/dividends)
-- =========================

CREATE TABLE IF NOT EXISTS corporate_actions (
  ticker            TEXT NOT NULL,
  action_date       TEXT NOT NULL CHECK (length(action_date)=10),        -- effective/ex-date for the action
  action_type       TEXT NOT NULL CHECK (action_type IN ('SPLIT','DIVIDEND')),        -- 'SPLIT' or 'DIVIDEND'

  -- SPLIT fields (use either ratio OR numerator/denominator)
  split_ratio       REAL CHECK (split_ratio IS NULL OR split_ratio >0),                 -- e.g. 2.0 for a 2-for-1
  split_numerator   REAL,
  split_denominator REAL,

  -- DIVIDEND fields
  dividend_cash     REAL CHECK (dividend_cash IS NULL OR dividend_cash >= 0),                 -- cash dividend per share

  source            TEXT NOT NULL DEFAULT 'yahoo',
  fetched_at        TEXT NOT NULL DEFAULT (datetime('now')),

  PRIMARY KEY (ticker, action_date, action_type),
  FOREIGN KEY (ticker) REFERENCES ticker_data (ticker)
);

CREATE INDEX IF NOT EXISTS idx_corp_actions_ticker_date
ON corporate_actions (ticker, action_date);

CREATE TABLE IF NOT EXISTS ticker_events (
  event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
  old_ticker      TEXT,
  new_ticker      TEXT,
  event_date      TEXT NOT NULL CHECK (length(event_date)=10),
  event_type      TEXT NOT NULL CHECK (event_type IN ('TICKER_CHANGE','NAME_CHANGE')),
  old_name        TEXT,
  new_name        TEXT,
  source          TEXT NOT NULL DEFAULT 'manual',
  noted_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_ticker_events_new_ticker_date
ON ticker_events (new_ticker, event_date);

CREATE INDEX IF NOT EXISTS idx_ticker_events_old_ticker_date
ON ticker_events (old_ticker, event_date);


