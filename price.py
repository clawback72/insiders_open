from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

import sqlite3
import logging
from logging.handlers import RotatingFileHandler

# expects you have logger configured at module level
# logger = setup_logging("price")  # for example

ET = ZoneInfo("America/New_York")

def setup_logging(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
    
    handler = RotatingFileHandler(
        f"{name}.log",
        maxBytes=5_000_000,  # 5 MB
        backupCount=5  # Keep up to 5 log files
    )
    
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

logger = setup_logging("price")

def main(db: sqlite3.Connection) -> None:
    cursor = db.cursor()

    df_equities = pd.read_sql("SELECT DISTINCT ticker FROM ticker_data;", db)
    if df_equities.empty:
        logger.info("No equities found in ticker_data.")
        rows_eq = 0
    else:
        rows_eq = get_historical_prices_v2(
            cursor, df_equities, db, term_days=5, ticker_col="ticker", include_actions=True
        )

    df_indexes = pd.read_sql("SELECT DISTINCT ticker FROM index_data;", db)
    if df_indexes.empty:
        logger.info("No indexes found in index_data.")
        rows_ix = 0
    else:
        rows_ix = get_historical_prices_v2(
            cursor, df_indexes, db, term_days=5, ticker_col="ticker", include_actions=False
        )

    logger.info("Price job complete: equities_rows=%d index_rows=%d", rows_eq, rows_ix)

def _compute_date_range(term_days: int) -> tuple[str, str]:
    """
    Returns (start_date, end_date) in YYYY-MM-DD.
    end_date is exclusive for yfinance download().
    Excludes today's partial candle if before 4:10pm ET.
    """
    now_et = datetime.now(ET)
    cutoff = now_et.replace(hour=16, minute=10, second=0, microsecond=0)

    # If before cutoff, don't include today (end_date = today)
    # If after cutoff, allow including today by making end_date tomorrow (exclusive)
    if now_et < cutoff:
        end_dt = now_et.date()  # today, exclusive -> excludes today candle
    else:
        end_dt = (now_et.date() + timedelta(days=1))  # tomorrow exclusive

    start_dt = (now_et.date() - timedelta(days=term_days))

    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


def _extract_tickers(tickers_df: pd.DataFrame | list[str], ticker_col: str = "ticker") -> list[str]:
    if isinstance(tickers_df, list):
        tickers = tickers_df
    else:
        if ticker_col not in tickers_df.columns:
            raise ValueError(f"Missing ticker column '{ticker_col}' in tickers_df")
        tickers = tickers_df[ticker_col].dropna().astype(str).unique().tolist()

    # yfinance prefers space-separated list; we'll keep list for filtering
    tickers = [t.strip() for t in tickers if t and t.strip()]
    return tickers


def get_historical_prices_v2(
    cursor,
    tickers_df: pd.DataFrame | list[str],
    db,
    term_days: int,
    *,
    ticker_col: str = "ticker",
    include_actions: bool = True,
    source: str = "yahoo",
) -> int:
    """
    Fetch OHLCV + Adj Close for tickers over the last `term_days` and upsert into pricing_data.
    Optionally record splits/dividends into corporate_actions.

    Returns number of price rows upserted (attempted).
    """
    tickers = _extract_tickers(tickers_df, ticker_col=ticker_col)
    if not tickers:
        return 0

    start_date, end_date = _compute_date_range(term_days)

    # Batch download (much faster than per-ticker history())
    # auto_adjust=False so we get both Close and Adj Close
    data = yf.download(
        tickers=" ".join(tickers),
        start=start_date,
        end=end_date,
        group_by="ticker",
        auto_adjust=False,
        actions=False,
        threads=True,
        progress=False,
    )

    if data is None or data.empty:
        logger.warning("No pricing data returned for %d tickers (%s to %s)", len(tickers), start_date, end_date)
        return 0

    # Normalize into rows: (ticker, date, open, high, low, close, adj_close, volume, source)
    rows = []

    # yfinance returns:
    # - Single ticker: columns like ["Open","High",...]
    # - Multi ticker: column MultiIndex: (ticker, field)
    multi = isinstance(data.columns, pd.MultiIndex)

    if not multi:
        # Single ticker case
        t = tickers[0]
        df_t = data.copy()
        for dt, r in df_t.iterrows():
            date_str = dt.strftime("%Y-%m-%d")
            rows.append((
                t,
                date_str,
                float(r.get("Open")) if pd.notna(r.get("Open")) else None,
                float(r.get("High")) if pd.notna(r.get("High")) else None,
                float(r.get("Low")) if pd.notna(r.get("Low")) else None,
                float(r.get("Close")) if pd.notna(r.get("Close")) else None,
                float(r.get("Adj Close")) if pd.notna(r.get("Adj Close")) else None,
                int(r.get("Volume")) if pd.notna(r.get("Volume")) else None,
                source,
            ))
    else:
        # Multi ticker case
        # data columns: (ticker, field)
        for t in tickers:
            if t not in data.columns.get_level_values(0):
                continue
            df_t = data[t].dropna(how="all")
            for dt, r in df_t.iterrows():
                date_str = dt.strftime("%Y-%m-%d")
                rows.append((
                    t,
                    date_str,
                    float(r.get("Open")) if pd.notna(r.get("Open")) else None,
                    float(r.get("High")) if pd.notna(r.get("High")) else None,
                    float(r.get("Low")) if pd.notna(r.get("Low")) else None,
                    float(r.get("Close")) if pd.notna(r.get("Close")) else None,
                    float(r.get("Adj Close")) if pd.notna(r.get("Adj Close")) else None,
                    int(r.get("Volume")) if pd.notna(r.get("Volume")) else None,
                    source,
                ))

    if not rows:
        logger.warning("Pricing download returned no rows after normalization (%s to %s)", start_date, end_date)
        return 0

    cursor.executemany(
        """
        INSERT INTO pricing_data (
            ticker, date, open, high, low, close, adj_close, volume, source
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, date) DO UPDATE SET
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            adj_close = excluded.adj_close,
            volume = excluded.volume,
            source = excluded.source,
            fetched_at = datetime('now')
        """,
        rows,
    )

    # corporate actions (splits/dividends) — optional
    if include_actions:
        _upsert_corporate_actions(cursor, tickers, source=source)

    db.commit()
    logger.info("Upserted %d pricing rows for %d tickers (%s to %s)", len(rows), len(tickers), start_date, end_date)
    return len(rows)


def _upsert_corporate_actions(cursor, tickers: list[str], *, source: str = "yahoo") -> None:
    """
    Pull splits/dividends from yfinance per ticker (no batch API exposed).
    This is usually light data and can run daily.
    """
    split_rows = []
    div_rows = []

    for t in tickers:
        try:
            yt = yf.Ticker(t)

            # Splits: Series indexed by date with ratio (e.g., 2.0)
            splits = getattr(yt, "splits", None)
            if splits is not None and not splits.empty:
                for dt, ratio in splits.items():
                    action_date = pd.to_datetime(dt).strftime("%Y-%m-%d")
                    ratio_val = float(ratio) if pd.notna(ratio) else None
                    if ratio_val:
                        split_rows.append((
                            t, action_date, "SPLIT",
                            ratio_val, None, None,
                            None,
                            source
                        ))

            # Dividends: Series indexed by date with cash amount
            divs = getattr(yt, "dividends", None)
            if divs is not None and not divs.empty:
                for dt, cash in divs.items():
                    action_date = pd.to_datetime(dt).strftime("%Y-%m-%d")
                    cash_val = float(cash) if pd.notna(cash) else None
                    if cash_val is not None:
                        div_rows.append((
                            t, action_date, "DIVIDEND",
                            None, None, None,
                            cash_val,
                            source
                        ))
        except Exception as e:
            logger.warning("Corporate actions fetch failed for %s: %s", t, e, exc_info=True)

    if split_rows:
        cursor.executemany(
            """
            INSERT INTO corporate_actions (
                ticker, action_date, action_type,
                split_ratio, split_numerator, split_denominator,
                dividend_cash,
                source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, action_date, action_type) DO UPDATE SET
                split_ratio = excluded.split_ratio,
                split_numerator = excluded.split_numerator,
                split_denominator = excluded.split_denominator,
                dividend_cash = excluded.dividend_cash,
                source = excluded.source,
                fetched_at = datetime('now')
            """,
            split_rows,
        )

    if div_rows:
        cursor.executemany(
            """
            INSERT INTO corporate_actions (
                ticker, action_date, action_type,
                split_ratio, split_numerator, split_denominator,
                dividend_cash,
                source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, action_date, action_type) DO UPDATE SET
                split_ratio = excluded.split_ratio,
                split_numerator = excluded.split_numerator,
                split_denominator = excluded.split_denominator,
                dividend_cash = excluded.dividend_cash,
                source = excluded.source,
                fetched_at = datetime('now')
            """,
            div_rows,
        )

def connect_db(db_name: str) -> sqlite3.Connection:
    db = sqlite3.connect(db_name)
    db.execute("PRAGMA foreign_keys = ON;")
    return db

if __name__ == "__main__":
    # connect to database
    db = connect_db("insider_database_v2.db")
    try:
        main(db)
    finally:
        db.close()