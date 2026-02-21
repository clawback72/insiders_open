import requests
import pandas as pd
import yfinance as yf
import sqlite3
import logging
from logging.handlers import RotatingFileHandler
import pytz
import sys
import hashlib
from bs4 import BeautifulSoup as bs
from datetime import datetime, timedelta
from urllib.parse import urljoin

OPENINSIDER_BASE = "https://openinsider.com/"
SEC_HTTP = "http://www.sec.gov/"
SEC_HTTPS = "https://www.sec.gov/"

def main(db):
    cursor = db.cursor()

    insider_page_url = (
        "http://openinsider.com/screener?s=&o=&pl=&ph=&ll=&lh=&fd=730&fdr=&td=0&tdr=&"
        "fdlyl=&fdlyh=&daysago=&xp=1&xs=1&vl=&vh=&ocl=&och=&sic1=-1&sicl=100&sich=9999&"
        "grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h=&sortcol=0&cnt=500&page=1"
    )

    run_id = start_run(cursor, "inside_scrape")
    db.commit()  # important: persist RUNNING row immediately

    # Optional “before” metrics (handy for sanity checks, but don’t rely on for rows_inserted)
    before_ins = _scalar(db, "SELECT COUNT(*) FROM insider_data", default=0)
    before_tkr = _scalar(db, "SELECT COUNT(*) FROM ticker_data", default=0)

    tickers_inserted = 0
    tx_inserted = 0
    tx_skipped = 0
    tx_deactivated = 0
    idx_inserted = 0

    try:
        df_insiders = scrape(insider_page_url)
        logger.info("Scraped %d insider rows", len(df_insiders))
        logger.debug("Head:\n%s", df_insiders.head().to_string())

        tickers_inserted = insert_tickers(cursor, df_insiders, db)
        logger.info("Tickers inserted=%d", tickers_inserted)

        tx_inserted, tx_skipped, tx_deactivated = insert_insiders(cursor, df_insiders, db)
        logger.info(
            "Transactions: inserted=%d, skipped=%d, deactivated=%d",
            tx_inserted, tx_skipped, tx_deactivated,
        )

        index_data = {
            "ticker": ["^GSPC", "^DJI", "^IXIC", "^VIX", "^RUT", "^FTW5000"],
            "index_name": [
                "S&P 500",
                "Dow Jones Industrial Average",
                "NASDAQ",
                "CBOE Volatility Index",
                "Russell 2000",
                "Wilshire 5000",
            ],
        }
        df_index = pd.DataFrame(index_data)

        idx_inserted = insert_index(cursor, df_index, db)
        logger.info("Indexes inserted=%d", idx_inserted)

        # Prefer your “known inserted counts” over table deltas
        rows_inserted = int(tickers_inserted) + int(tx_inserted) + int(idx_inserted)

        # Optional after-metrics for debugging
        after_ins = _scalar(db, "SELECT COUNT(*) FROM insider_data", default=0)
        after_tkr = _scalar(db, "SELECT COUNT(*) FROM ticker_data", default=0)
        logger.info(
            "Counts: insider_data %d→%d, ticker_data %d→%d",
            before_ins, after_ins, before_tkr, after_tkr,
        )

        finish_run(cursor, run_id, status="SUCCESS", rows_inserted=rows_inserted)
        db.commit()

    except Exception as e:
        # Always record the failure in runs
        finish_run(cursor, run_id, status="FAILED", error_message=str(e))
        db.commit()
        logger.exception("inside_scrape failed run_id=%s", run_id)
        raise

def scrape(page_url: str) -> pd.DataFrame:
    r = requests.get(page_url, timeout=30)
    r.raise_for_status()

    soup = bs(r.content, features="lxml")
    table = soup.find("table", {"class": "tinytable"})
    if table is None:
        raise RuntimeError("Could not find table.tinytable on the page")

    # headers (visible text)
    headers = [th.get_text(strip=True).replace("\xa0", " ") for th in table.find_all("th")]

    rows = []
    
    filing_idx = headers.index("Filing Date") if "Filing Date" in headers else None
    
    for tr in table.find_all("tr")[1:]:
        tds = tr.find_all("td")
        if not tds:
            continue

        # build a dict keyed by header
        row = {}
        for i, cell in enumerate(tds):
            if i >= len(headers):
                break
            row[headers[i]] = cell.get_text(strip=True)

        # Extract the Form 4 link from the Filing Date cell (anchor tag)
        # OpenInsider typically has <td><a href="...">Filing Date</a></td>
        source_url = None
        if filing_idx is not None and filing_idx < len(tds):
            a = tds[filing_idx].find("a", href=True)
            if a:
                source_url = urljoin(OPENINSIDER_BASE, a["href"])
                
                #normalize SEC http -> https
                if source_url.startswith("http://www.sec.gov/"):
                    source_url = source_url.replace("http://www.sec.gov/", "https://www.sec.gov/")
                    
        row["Source URL"] = source_url
        
        rows.append(row)

    df = pd.DataFrame(rows)

    # normalize column names (just in case)
    df.columns = df.columns.str.replace("\xa0", " ")

    # Keep only the columns you care about (adjust as needed)
    keep = [
        "X", "Filing Date", "Trade Date", "Ticker", "Company Name",
        "Insider Name", "Title", "Trade Type", "Price", "Qty",
        "Owned", "ΔOwn", "Value", "Source URL"
    ]
    df = df[[c for c in keep if c in df.columns]]

    # Cast dates
    df["Filing Date"] = pd.to_datetime(df["Filing Date"], errors="coerce")
    df["Trade Date"] = pd.to_datetime(df["Trade Date"], errors="coerce")

    # IMPORTANT: your new schema expects YYYY-MM-DD (length=10)
    df["Filing Date"] = df["Filing Date"].dt.strftime("%Y-%m-%d")
    df["Trade Date"] = df["Trade Date"].dt.strftime("%Y-%m-%d")

    # Qty int, Price float
    df["Qty"] = df["Qty"].replace(r"[^\d]", "", regex=True)
    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce").astype("Int64")

    df["Price"] = df["Price"].replace(r"[^\d.]", "", regex=True)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    
    df.columns = (
        df.columns
            .str.strip()
            .str.replace(" ", "_")
            .str.replace("Δ", "delta", regex=False)
            .str.lower()
    )
    
    # fix "Trade Type" field to just letter code
    df["sec_tx_code"] = df["trade_type"].astype("string").str.split(" - ", n=1).str[0]
    
    return df

def insert_tickers(cursor, df, db) -> int:
    """
    Insert new tickers into ticker_data and update last_seen_at for all tickers present in df.
    Returns count of newly inserted tickers.
    """

    # unique tickers seen in this scrape
    df_tickers = (
        df[["ticker", "company_name"]]
        .dropna(subset=["ticker"])
        .drop_duplicates()
        .copy()
    )

    # existing tickers set
    existing = pd.read_sql("SELECT ticker FROM ticker_data", db)["ticker"].tolist()
    existing_set = set(existing)

    # split into new vs seen
    new_df = df_tickers[~df_tickers["ticker"].isin(existing_set)].copy()

    # Always bump last_seen_at for tickers present in scrape
    # (do this once, not row-by-row if you can)
    for t in df_tickers["ticker"].tolist():
        cursor.execute(
            "UPDATE ticker_data SET last_seen_at=datetime('now') WHERE ticker=?",
            (t,)
        )

    # If no new tickers, commit last_seen_at updates and return
    if new_df.empty:
        db.commit()
        return 0

    # Enrich new tickers with yfinance info
    # Initialize schema columns
    new_df["short_name"] = None
    new_df["sector"] = None
    new_df["industry"] = None
    new_df["exchange"] = None
    new_df["website"] = None
    new_df["cik"] = None
    new_df["currency"] = None
    new_df["country"] = None

    for idx, row in new_df.iterrows():
        ticker_symbol = row["ticker"]
        try:
            info = yf.Ticker(ticker_symbol).info or {}
            new_df.at[idx, "short_name"] = info.get("shortName")
            new_df.at[idx, "sector"] = info.get("sector")
            new_df.at[idx, "industry"] = info.get("industry")
            new_df.at[idx, "exchange"] = info.get("exchange")
            new_df.at[idx, "website"] = info.get("website")
            new_df.at[idx, "cik"] = info.get("cik")
            new_df.at[idx, "currency"] = info.get("currency")
            new_df.at[idx, "country"] = info.get("country")
        except Exception as e:
            logger.warning("yfinance info failed for %s: %s", ticker_symbol, e)

    # Insert new tickers (first_seen_at/last_seen_at handled by defaults)
    for _, row in new_df.iterrows():
        cursor.execute(
            """
            INSERT OR IGNORE INTO ticker_data
            (ticker, company_name, short_name, sector, industry, exchange, website, cik, currency, country)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["ticker"],
                row["company_name"],
                row["short_name"],
                row["sector"],
                row["industry"],
                row["exchange"],
                row["website"],
                row["cik"],
                row["currency"],
                row["country"],
            ),
        )

    db.commit()

    return len(new_df)

def insert_index(cursor, df, db):
    existing = pd.read_sql("SELECT ticker FROM index_data", db)
    existing_set = set(existing["ticker"].tolist())

    df_idx = df[["ticker", "index_name"]].drop_duplicates().copy()
    new_idx = df_idx[~df_idx["ticker"].isin(existing_set)].copy()

    if new_idx.empty:
        return 0

    new_idx["short_name"] = None

    for i, row in new_idx.iterrows():
        t = row["ticker"]
        try:
            info = yf.Ticker(t).info or {}
            new_idx.at[i, "short_name"] = info.get("shortName")
        except Exception as e:
            logger.warning("Index info failed for %s: %s", t, e, exc_info=True)

    for _, row in new_idx.iterrows():
        cursor.execute(
            """
            INSERT INTO index_data (ticker, index_name, short_name)
            VALUES (?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
              index_name = excluded.index_name,
              short_name = excluded.short_name
            """,
            (row["ticker"], row["index_name"], row["short_name"]),
        )

    db.commit()
    return len(new_idx)

def _none_if_na(x):
    # Handles pandas <NA>, NaN, None
    if x is None:
        return None
    try:
        # NaN check: NaN != NaN
        if x != x:
            return None
    except Exception:
        pass
    return x

def _to_float(x):
    x = _none_if_na(x)
    return None if x is None else float(x)

def _to_int(x):
    x = _none_if_na(x)
    return None if x is None else int(x)

def _tx_id(
    *,
    source_url: str | None,
    filing_date: str | None,
    trade_date: str | None,
    ticker: str | None,
    insider_name: str | None,
    title: str | None,
    trade_type: str | None,
    price: float | None,
    qty: int | None,
    x_flags: str | None,
) -> str:
    """
    Stable transaction_id.
    Includes source_url (strong anchor) + key transaction fields to avoid collisions
    across multiple rows tied to the same filing.
    """
    parts = [
        source_url or "",
        filing_date or "",
        trade_date or "",
        ticker or "",
        insider_name or "",
        title or "",
        trade_type or "",
        f"{price}" if price is not None else "",
        f"{qty}" if qty is not None else "",
        x_flags or "",
    ]
    canonical = "|".join(parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def insert_insiders(cursor, df, db):
    """
    Insert insider transactions into insider_data (v2 schema).
    Returns (inserted_count, skipped_count, deactivated_count).
    """

    inserted = 0
    skipped = 0
    deactivated_total = 0

    # basic guardrails
    required = ["ticker", "filing_date", "source_url"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"insert_insiders: missing required column '{col}'")

    # iterate records
    for rec in df.to_dict(orient="records"):
        ticker = _none_if_na(rec.get("ticker"))
        filing_date = _none_if_na(rec.get("filing_date"))
        trade_date = _none_if_na(rec.get("trade_date"))
        insider_name = _none_if_na(rec.get("insider_name"))
        title = _none_if_na(rec.get("title"))
        trade_type = _none_if_na(rec.get("trade_type"))
        sec_tx_code = _none_if_na(rec.get("sec_tx_code"))
        source_url = _none_if_na(rec.get("source_url"))

        # your scraper column is 'x' (formerly X)
        x_flags = _none_if_na(rec.get("x"))

        price = _to_float(rec.get("price"))
        qty = _to_int(rec.get("qty"))

        # compute value when possible
        value = None
        if price is not None and qty is not None:
            value = price * qty

        # simple classification rules (refine later if you parse SEC XML)
        is_open_market = None
        classification = "UNKNOWN"
        if trade_type == "P - Purchase":
            is_open_market = 1
            classification = "OPEN_MARKET"
        elif trade_type == "S - Sale":
            is_open_market = 1
            classification = "OPEN_MARKET"
        elif trade_type == "S - Sale+OE":
            is_open_market = 0
            classification = "OPTION_EXERCISE"
        elif trade_type == "F - Tax":
            is_open_market = 0
            classification = "OTHER"
        elif trade_type is None:
            is_open_market = None
            classification = "UNKNOWN"
        else:
            # unknown label, keep but mark OTHER/UNKNOWN
            is_open_market = None
            classification = "OTHER"

        transaction_id = _tx_id(
            source_url=source_url,
            filing_date=filing_date,
            trade_date=trade_date,
            ticker=ticker,
            insider_name=insider_name,
            title=title,
            trade_type=trade_type,
            price=price,
            qty=qty,
            x_flags=x_flags,
        )

        cursor.execute(
            """
            INSERT INTO insider_data (
                transaction_id,
                filing_date,
                trade_date,
                ticker,
                insider_name,
                title,
                trade_type,
                sec_tx_code,
                x_flags,
                is_open_market,
                classification,
                price,
                qty,
                value,
                source_url,
                is_active
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(transaction_id) DO NOTHING
            """,
            (
                transaction_id,
                filing_date,
                trade_date,
                ticker,
                insider_name,
                title,
                trade_type,
                sec_tx_code,
                x_flags,
                is_open_market,
                classification,
                price,
                qty,
                value,
                source_url,
            ),
        )

        if cursor.rowcount == 1:
            inserted += 1

            # Amendment handling: if x_flags contains 'A', deactivate matching prior active rows
            is_amendment = isinstance(x_flags, str) and ("A" in x_flags)
            if is_amendment:
                # Natural key: ticker + insider_name + trade_date + sec_tx_code
                cursor.execute(
                    """
                    UPDATE insider_data
                    SET is_active = 0,
                        superseded_by = ?,
                        superseded_at = datetime('now')
                    WHERE is_active = 1
                      AND ticker = ?
                      AND (insider_name IS ?)
                      AND (trade_date IS ?)
                      AND (sec_tx_code IS ?)
                      AND transaction_id <> ?
                    """,
                    (transaction_id, ticker, insider_name, trade_date, sec_tx_code, transaction_id),
                )
                deactivated = cursor.rowcount
                deactivated_total += deactivated

                # if deactivated != 1, it's not fatal, but it’s worth logging
                if deactivated != 1:
                    logger.warning(
                        "Amendment inserted (%s) deactivated %d prior rows for key (%s, %s, %s, %s)",
                        transaction_id, deactivated, ticker, insider_name, trade_date, sec_tx_code
                    )
        else:
            skipped += 1

    db.commit()
    return inserted, skipped, deactivated_total

def _scalar(db, sql, params=(), default=None):
    row = db.execute(sql, params).fetchone()
    return row[0] if row and row[0] is not None else default

def start_run(cursor, job_name: str) -> int:
    cursor.execute(
        "INSERT INTO runs (job_name, status, started_at) VALUES (?, 'RUNNING', datetime('now'))",
        (job_name,),
    )
    return cursor.lastrowid

def finish_run(cursor, run_id: int, *, status: str, rows_inserted: int = 0, rows_updated: int = 0, error_message: str | None = None) -> None:
    cursor.execute(
        """
        UPDATE runs
        SET finished_at = datetime('now'),
            status = ?,
            rows_inserted = ?,
            rows_updated = ?,
            error_message = ?
        WHERE run_id = ?
        """,
        (status, rows_inserted, rows_updated, error_message, run_id),
    )

def connect_db(db_name):
    return sqlite3.connect(db_name)

def setup_logging(name):
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

logger = setup_logging("inside_scrape")

if __name__ == "__main__":
    # connect to database
    db = connect_db('insider_database_v2.db')
    db.execute('PRAGMA foreign_keys = ON;')
    try:
        main(db)
    finally:
        db.close()