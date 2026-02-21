from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

import sqlite3
import logging
from logging.handlers import RotatingFileHandler

from typing import Iterable, Optional

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

    # Pull indexes with names so we can store a friendly placeholder in ticker_data
    df_indexes = pd.read_sql("SELECT ticker, index_name FROM index_data;", db)
    if df_indexes.empty:
        logger.info("No indexes found in index_data.")
        rows_ix = 0
    else:
        ensured = ensure_tickers_exist(cursor, df_indexes, ticker_col="ticker", name_col="index_name")
        db.commit()
        logger.info("Ensured %d index tickers exist in ticker_data (FK safety).", ensured)

        rows_ix = get_historical_prices_v2(
            cursor, df_indexes[["ticker"]], db, term_days=5, ticker_col="ticker", include_actions=False
        )
        
    evaluate_ticker_health(cursor, job_name="price")
    db.commit()

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

def _upsert_ticker_failure(
    cursor,
    *,
    ticker: str,
    job_name: str,
    reason: str,
    error_class: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    cursor.execute(
        """
        INSERT INTO ticker_failures (
            ticker, job_name, failure_date, reason, error_class, error_message,
            first_seen_at, last_seen_at, fail_count
        )
        VALUES (?, ?, date('now'), ?, ?, ?, datetime('now'), datetime('now'), 1)
        ON CONFLICT(ticker, job_name) DO UPDATE SET
            failure_date  = date('now'),
            reason        = excluded.reason,
            error_class   = COALESCE(excluded.error_class, ticker_failures.error_class),
            error_message = COALESCE(excluded.error_message, ticker_failures.error_message),
            last_seen_at  = datetime('now'),
            fail_count    = ticker_failures.fail_count + 1
        """,
        (ticker, job_name, reason, error_class, error_message),
    )
    
    # roll-up status into ticker_data for quick filtering
    cursor.execute(
        """
        UPDATE ticker_data
        SET status = CASE
                WHEN status IN ('EXCLUDED','DELISTED') THEN status
                WHEN status IS NULL OR status = '' THEN 'WATCH'
                ELSE status
            END,
            status_reason = COALESCE(?, status_reason),
            last_price_attempt_at = datetime('now')
        WHERE ticker = ?
        """,
        (reason, ticker),
    )

    if cursor.rowcount == 0:
        # ticker not present in ticker_data yet → create it
        cursor.execute(
            """
            INSERT OR IGNORE INTO ticker_data (
                ticker, first_seen_at, last_seen_at, status, status_reason, last_price_attempt_at
            )
            VALUES (?, datetime('now'), datetime('now'), 'WATCH', ?, datetime('now'))
            """,
            (ticker, reason),
        )

    
    # check fail_count from ticker and exclude if failure is greater than 3
    row = cursor.execute(
        "SELECT fail_count FROM ticker_failures WHERE ticker=? AND job_name=?",(ticker, job_name),
    ).fetchone()

def evaluate_ticker_health(cursor, *, job_name: str, threshold: int = 3) -> None:
    """
    After a pricing run completes, decide which tickers should be excluded.
    Only excludes if the ticker has not succeeded recently.
    """

    rows = cursor.execute(
        """
        SELECT f.ticker, f.fail_count, t.last_price_date
        FROM ticker_failures f
        JOIN ticker_data t ON t.ticker = f.ticker
        WHERE f.job_name = ?
          AND f.fail_count >= ?
        """,
        (job_name, threshold),
    ).fetchall()

    for ticker, fail_count, last_price_date in rows:
        # If ticker has ever produced a price, do NOT auto-exclude yet
        if last_price_date is not None:
            continue

        cursor.execute(
            """
            INSERT INTO ticker_exclusions (ticker, status, reason, source, is_active)
            VALUES (?, 'NO_DATA', 'AUTO: >=3 failures with no price history', 'system', 1)
            ON CONFLICT(ticker) DO UPDATE SET
                status='NO_DATA',
                excluded_at=datetime('now'),
                is_active=1
            """,
            (ticker,),
        )

        cursor.execute(
            """
            UPDATE ticker_data
            SET status='EXCLUDED',
                status_reason='AUTO: NO_DATA (>=3 failures)'
            WHERE ticker=?
            """,
            (ticker,),
        )

def _clear_ticker_failure(cursor, *, ticker: str, job_name: str) -> None:
    cursor.execute(
        "DELETE FROM ticker_failures WHERE ticker = ? AND job_name = ?",
        (ticker, job_name),
    )

def _touch_ticker_attempt(cursor, *, ticker: str) -> None:
    cursor.execute(
        """
        UPDATE ticker_data
        SET last_price_attempt_at = datetime('now')
        WHERE ticker = ?
        """,
        (ticker,),
    )

def _touch_ticker_success(cursor, *, ticker: str) -> None:
    cursor.execute(
        """
        UPDATE ticker_data
        SET last_seen_at = datetime('now'),
            last_price_attempt_at = datetime('now'),
            status = COALESCE(status, 'ACTIVE'),
            status_reason = NULL
        WHERE ticker = ?
        """,
        (ticker,),
    )
    
def record_price_success(cursor: sqlite3.Cursor, ticker: str, last_date: str | None) -> None:
    cursor.execute(
        """
        UPDATE ticker_data
           SET last_price_attempt_at = datetime('now'),
               last_price_date = COALESCE(?, last_price_date),
               status = COALESCE(status, 'ACTIVE'),
               status_reason = NULL,
               last_seen_at = datetime('now')
         WHERE ticker = ?
        """,
        (last_date, ticker),
    )

    # Clear any existing failure row for this job if you want:
    cursor.execute(
        """
        DELETE FROM ticker_failures
         WHERE ticker = ? AND job_name = 'price'
        """,
        (ticker,),
    )


def record_price_failure(
    cursor: sqlite3.Cursor,
    ticker: str,
    *,
    reason: str,
    exc: Exception | None = None,
) -> None:
    err_class = type(exc).__name__ if exc else None
    err_msg = str(exc)[:500] if exc else None

    # ticker_failures upsert
    cursor.execute(
        """
        INSERT INTO ticker_failures (
            ticker, job_name, failure_date, reason, error_class, error_message,
            first_seen_at, last_seen_at, fail_count
        )
        VALUES (?, 'price', date('now'), ?, ?, ?, datetime('now'), datetime('now'), 1)
        ON CONFLICT(ticker, job_name) DO UPDATE SET
            failure_date = date('now'),
            reason = excluded.reason,
            error_class = excluded.error_class,
            error_message = excluded.error_message,
            last_seen_at = datetime('now'),
            fail_count = ticker_failures.fail_count + 1
        """,
        (ticker, reason, err_class, err_msg),
    )

    # ticker_data health fields
    cursor.execute(
        """
        UPDATE ticker_data
           SET last_price_attempt_at = datetime('now'),
               status = COALESCE(status, 'WATCH'),
               status_reason = ?
         WHERE ticker = ?
        """,
        (reason, ticker),
    )

def _chunk(seq: list[str], size: int) -> list[list[str]]:
    return [seq[i:i+size] for i in range(0, len(seq), size)]

def get_historical_prices_v2(
    cursor,
    tickers_df: pd.DataFrame | list[str],
    db,
    term_days: int,
    *,
    ticker_col: str = "ticker",
    include_actions: bool = True,
    source: str = "yahoo",
    chunk_size: int = 50,
    threads: bool = False,   # <- important: reduces getaddrinfo thread failures
) -> int:
    
    logger.info("get_historical_prices_v2 started with term_days=%d, include_actions=%s, source=%s, chunk_size=%d, threads=%s",
                term_days, include_actions, source, chunk_size, threads)
    
    tickers = _extract_tickers(tickers_df, ticker_col=ticker_col)
    if not tickers:
        return 0

    start_date, end_date = _compute_date_range(term_days)

    # Skip active exclusions (TEMP exclusions not expired)
    excluded = set(r[0] for r in cursor.execute(
        """
        SELECT ticker
        FROM ticker_exclusions
        WHERE is_active = 1
          AND (expires_at IS NULL OR expires_at > datetime('now'))
        """
    ).fetchall())

    tickers = [t for t in tickers if t not in excluded]
    if not tickers:
        logger.info("All tickers excluded for this run.")
        return 0

    # Touch attempt for all tickers upfront so you always record the attempt
    for t in tickers:
        _touch_ticker_attempt(cursor, ticker=t)

    total_rows = 0

    for batch in _chunk(tickers, chunk_size):
        total_rows += _process_price_batch(
            cursor,
            batch,
            db,
            start_date=start_date,
            end_date=end_date,
            include_actions=include_actions,
            source=source,
            threads=threads,
        )

    db.commit()
    logger.info(
        "Upserted %d pricing rows for %d tickers (%s to %s)",
        total_rows, len(tickers), start_date, end_date
    )
    return total_rows


def _process_price_batch(
    cursor,
    tickers: list[str],
    db,
    *,
    start_date: str,
    end_date: str,
    include_actions: bool,
    source: str,
    threads: bool,
) -> int:
    # Batch download
    try:
        data = yf.download(
            tickers=" ".join(tickers),
            start=start_date,
            end=end_date,
            group_by="ticker",
            auto_adjust=False,
            actions=False,
            threads=threads,
            progress=False,
        )
    except Exception as e:
        # Whole-batch failure: mark each ticker failed
        for t in tickers:
            _upsert_ticker_failure(
                cursor,
                ticker=t,
                job_name="price",
                reason="yfinance download exception (batch)",
                error_class=type(e).__name__,
                error_message=str(e),
            )
        return 0

    if data is None or data.empty:
        for t in tickers:
            _upsert_ticker_failure(
                cursor,
                ticker=t,
                job_name="price",
                reason="yfinance download returned empty dataframe (batch)",
                error_class="YFEmptyBatch",
                error_message=None,
            )
        return 0

    # Identify returned tickers
    if isinstance(data.columns, pd.MultiIndex):
        returned = set(data.columns.get_level_values(0))
    else:
        returned = {tickers[0]}  # single-ticker case

    missing = [t for t in tickers if t not in returned]
    for t in missing:
        _upsert_ticker_failure(
            cursor,
            ticker=t,
            job_name="price",
            reason="yfinance download did not return ticker (missing column group)",
            error_class="YFMissingTicker",
            error_message=None,
        )

    # Present but all NaN -> failure
    if isinstance(data.columns, pd.MultiIndex):
        for t in (returned & set(tickers)):
            try:
                df_t = data[t]
            except Exception:
                continue
            if df_t.dropna(how="all").empty:
                _upsert_ticker_failure(
                    cursor,
                    ticker=t,
                    job_name="price",
                    reason="yfinance returned ticker group but no OHLCV rows (all-NaN)",
                    error_class="YFNoRows",
                    error_message=None,
                )

    # Normalize rows
    rows: list[tuple] = []
    multi = isinstance(data.columns, pd.MultiIndex)

    if not multi:
        t = tickers[0]
        df_t = data.dropna(how="all")
        for dt, r in df_t.iterrows():
            rows.append((
                t,
                dt.strftime("%Y-%m-%d"),
                float(r.get("Open")) if pd.notna(r.get("Open")) else None,
                float(r.get("High")) if pd.notna(r.get("High")) else None,
                float(r.get("Low")) if pd.notna(r.get("Low")) else None,
                float(r.get("Close")) if pd.notna(r.get("Close")) else None,
                float(r.get("Adj Close")) if pd.notna(r.get("Adj Close")) else None,
                int(r.get("Volume")) if pd.notna(r.get("Volume")) else None,
                source,
            ))
    else:
        lvl0 = set(data.columns.get_level_values(0))
        for t in tickers:
            if t not in lvl0:
                continue
            df_t = data[t].dropna(how="all")
            if df_t.empty:
                continue
            for dt, r in df_t.iterrows():
                rows.append((
                    t,
                    dt.strftime("%Y-%m-%d"),
                    float(r.get("Open")) if pd.notna(r.get("Open")) else None,
                    float(r.get("High")) if pd.notna(r.get("High")) else None,
                    float(r.get("Low")) if pd.notna(r.get("Low")) else None,
                    float(r.get("Close")) if pd.notna(r.get("Close")) else None,
                    float(r.get("Adj Close")) if pd.notna(r.get("Adj Close")) else None,
                    int(r.get("Volume")) if pd.notna(r.get("Volume")) else None,
                    source,
                ))

    if not rows:
        # If nothing normalized, mark tickers that "returned" but gave no rows as failures
        for t in (returned & set(tickers)):
            _upsert_ticker_failure(
                cursor,
                ticker=t,
                job_name="price",
                reason="Pricing download returned no rows after normalization",
                error_class="YFNoNormalizedRows",
                error_message=None,
            )
        return 0

    # Success set
    tickers_with_rows = {r[0] for r in rows}

    for t in tickers_with_rows:
        _clear_ticker_failure(cursor, ticker=t, job_name="price")
        _touch_ticker_success(cursor, ticker=t)

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

    # Update last_price_date for tickers we successfully wrote pricing for
    if tickers_with_rows:
        cursor.execute(
            f"""
            UPDATE ticker_data
               SET last_price_date = (
                   SELECT MAX(date) FROM pricing_data p
                    WHERE p.ticker = ticker_data.ticker
               )
             WHERE ticker IN ({",".join("?" for _ in tickers_with_rows)})
            """,
            tuple(tickers_with_rows),
        )

    if include_actions:
        _upsert_corporate_actions(cursor, tickers, source=source)

    # commit per batch (reduces “one bad batch nukes all work”)
    db.commit()
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
        
def ensure_tickers_exist(cursor, df_symbols: pd.DataFrame, *, ticker_col="ticker", name_col=None) -> None:
    """
    Insert placeholder rows into ticker_data so FK constraints for pricing_data pass.
    Optionally uses a name_col (e.g. index_name) for company_name/short_name.
    """
    rows = []
    for _, r in df_symbols.iterrows():
        t = str(r[ticker_col]).strip()
        if not t:
            continue
        name = None
        if name_col and name_col in df_symbols.columns:
            name = r[name_col]
        rows.append((t, name, name))

    cursor.executemany(
        """
        INSERT OR IGNORE INTO ticker_data (ticker, company_name, short_name)
        VALUES (?, ?, ?)
        """,
        rows,
    )
    return len(rows)

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