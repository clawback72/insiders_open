import requests
import pandas as pd
import yfinance as yf
import sqlite3
import logging
from logging.handlers import RotatingFileHandler
import pytz
import sys
from bs4 import BeautifulSoup as bs
from datetime import datetime, timedelta
from urllib.parse import urljoin

OPENINSIDER_BASE = "https://openinsider.com/"
SEC_HTTP = "http://www.sec.gov/"
SEC_HTTPS = "https://www.sec.gov/"

def main(db):

    # set cursor
    cursor = db.cursor()

    # define webpage
    
    insider_page_url = 'http://openinsider.com/screener?s=&o=&pl=&ph=&ll=&lh=&fd=730&fdr=&td=0&tdr=&fdlyl=&fdlyh=&daysago=&xp=1&xs=1&vl=&vh=&ocl=&och=&sic1=-1&sicl=100&sich=9999&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h=&sortcol=0&cnt=500&page=1'

    # create dataframe for insider transactions
    df_insiders = scrape(insider_page_url)
    logger.info('Scraped %d insider rows', len(df_insiders))
    logger.debug('Head:\n%s', df_insiders.head().to_string())

    # send dataframe to insert_tickers to update database with new securities
    tickers_inserted = insert_tickers(cursor, df_insiders, db)

    # send dataframe to insert_insiders to update database with new transactions
    tx_inserted, tx_skipped = insert_insiders(cursor, df_insiders, db)
    
    logger.info(
        'DB updates: Inserted tickers=%d, transactions inserted=%d,  skipped=%d',
        tickers_inserted, tx_inserted, tx_skipped
    )

    # create dataframe for insertion of indexes
    index_data = {
        'Ticker': ['^GSPC', '^DJI', '^IXIC', '^VIX', '^RUT', '^FTW5000'],
        'Name': ['S&P 500', 'Dow Jones Industrial Average', 'NASDAQ',
                 'CBOE Volatility Index', 'Russell 2000', 'Wilshire 5000']
    }

    df_index = pd.DataFrame(index_data)

    # insert indexes and prices to database
    insert_index(cursor, df_index, db)

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

def insert_tickers(cursor, df, db):
    # update database with new tickers

    # get existing tickers from database
    query = "SELECT Ticker, Company_Name FROM ticker_data"
    existing_tickers = pd.read_sql(query, db)

    # copy passed dataframe to get tickers and company name
    df_ticker_data = df[['Ticker', 'Company Name']].copy().drop_duplicates()

    # create dataframe to isolate new tickers
    merged = pd.merge(df_ticker_data, existing_tickers, on=['Ticker'], how='left', indicator=True)

    new_tickers = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

    # get additional information columns added
    new_tickers['Short Name'] = None
    new_tickers['Sector'] = None
    new_tickers['Industry'] = None
    new_tickers['Exchange'] = None
    new_tickers['Website'] = None

    for index, row in new_tickers.iterrows():
        ticker_symbol = row['Ticker']
        try:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info

            # update fields in dataframe
            new_tickers.at[index, 'Short Name'] = info.get('shortName', None)
            new_tickers.at[index, 'Sector'] = info.get('sector', None)
            new_tickers.at[index, 'Industry'] = info.get('industry', None)
            new_tickers.at[index, 'Exchange'] = info.get('exchange', None)
            new_tickers.at[index, 'Website'] = info.get('website', None)
        
        except Exception as e:
            logging.error(f'Error retrieving information data for {ticker_symbol}: {e}', exc_info=True)

    # insert into database
    for index, row in new_tickers.iterrows():
        cursor.execute('''
            INSERT OR IGNORE INTO ticker_data (Ticker, Company_Name, Short_Name, Sector,
            Industry, Exchange, Website) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (row['Ticker'], row['Company Name'], row['Short Name'], row['Sector']
                  , row['Industry'], row['Exchange'], row['Website']))
        
    db.commit()

    # get historical pricing for new tickers
    get_historical_prices(cursor, new_tickers, db, 30)

def insert_index(cursor, df, db):
    # update database with new tickers

    # get existing tickers from database
    query = "SELECT Ticker, Index_Name FROM index_data"
    existing_tickers = pd.read_sql(query, db)

    # copy passed dataframe to get tickers and Index name
    df_ticker_data = df[['Ticker', 'Name']].copy().drop_duplicates()

    # create dataframe to isolate new tickers
    merged = pd.merge(df_ticker_data, existing_tickers, on=['Ticker'], how='left', indicator=True)

    new_tickers = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

    # get additional information columns added
    new_tickers['Short Name'] = None

    for index, row in new_tickers.iterrows():
        ticker_symbol = row['Ticker']
        try:
            stock = yf.Ticker(ticker_symbol)
            info = stock.info

            # update fields in dataframe
            new_tickers.at[index, 'Short Name'] = info.get('shortName', None)
        
        except Exception as e:
            logging.error(f'Error retrieving information data for {ticker_symbol}: {e}', exc_info=True)

    # insert into database
    for index, row in new_tickers.iterrows():
        cursor.execute('''
            INSERT OR IGNORE INTO index_data (Ticker, Index_Name, Short_Name
            ) VALUES (?, ?, ?)
            ''', (row['Ticker'], row['Name'], row['Short Name']))
        
    db.commit()

    # get historical pricing for new tickers
    get_historical_prices(cursor, new_tickers, db, 30)

def get_historical_prices(cursor, new_tickers, db, term):
    # define cutoff time so we don't import today's close prices early (prior to 4:00 PM EST)
    # otherwise, early pre-close prices may be inserted to database and will be painful to correct
    est = pytz.timezone('US/Eastern')
    # get current time in EST
    current_time_est = datetime.now(est)
    # define cutoff time for getting today's prices after 4:00 PM EST
    cutoff_time_est = current_time_est.replace(hour=16, minute=10, second=0, microsecond=0)
   
    if current_time_est < cutoff_time_est:
        end_date = datetime.today().strftime('%Y-%m-%d')
    else:
        end_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
        
    start_date = (datetime.today() - timedelta(days=term)).strftime('%Y-%m-%d')

    # iterate through each new ticker
    for index, row in new_tickers.iterrows():
        ticker_symbol = row['Ticker']

        try:
            # get historical pricing data
            ticker = yf.Ticker(ticker_symbol)
            history = ticker.history(start=start_date, end=end_date)
            
            # print(history)

            # check if data is returned
            if not history.empty:
                for date, data in history.iterrows():
                    trade_date = date.strftime('%Y-%m-%d')
                    open_price = round(data['Open'], 3)
                    close_price = round(data['Close'], 3)
                    
                    # print each price as returned - comment out for production
                    print(ticker_symbol, trade_date, open_price, close_price)

                    # insert ticker data into database
                    cursor.execute('''
                        INSERT OR IGNORE INTO pricing_data (Ticker, Trade_Date, Open, Close)
                        VALUES (?, ?, ?, ?)
                    ''', (ticker_symbol, trade_date, open_price, close_price))

            # commit ticker data to database
            db.commit()

        except Exception as e:
            # log errors for failed pulls
            with open("scraper.log", "a") as log_file:
                log_file.write(f'Failed to get historical data for {ticker_symbol}: {e}\n')
            print(f'Failed to get historical data for {ticker_symbol}: {e}')
            
def insert_insiders(cursor, df, db):
    query = "SELECT Filing_Date FROM insider_data"
    existing_transactions = pd.read_sql(query, db)

    # copy passed data frame to get desired info for database
    df_insider_data = df[['X', 'Filing Date', 'Trade Date', 'Ticker', 'Insider Name', 'Title', 'Price', 'Qty']].copy()
    print(df_insider_data)

    # create dataframe to isolate new tickers
    merged = pd.merge(df_insider_data, existing_transactions, left_on='Filing Date', right_on='Filing_Date', how='left', indicator=True)
    
    new_insider_data = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

    print(new_insider_data)

    # add additional columns to df
    new_insider_data['Trade Close'] = None
    new_insider_data['Trade Low'] = None
    new_insider_data['Mkt Buy'] = None
    new_insider_data['52Wk Low'] = None
    new_insider_data['52Wk High'] = None
    new_insider_data['50Day Avg'] = None
    new_insider_data['200Day Avg'] = None

    for index, row in new_insider_data.iterrows():
        ticker_symbol = row['Ticker']
        trade_date = row['Trade Date']
        price = row['Price']

        try:
            # get close and low price for transaction
            trade_close, trade_low = get_closing_price(ticker_symbol, trade_date)
            print(f'individual pricing for {ticker_symbol} on {trade_date}:', trade_close, trade_low)
            
            low_52, high_52, avg_50, avg_200 = get_price_info(ticker_symbol)
            print(f'52 Day Low:{low_52}  52 Day High:{high_52}  50 Day Avg:{avg_50} 200 Day Avg:{avg_200}')
            
            # check that close and low are > 0, if not mkt_buy = none
            if trade_low is not None and trade_low > 0:
                # if price is above 99% of low, mkt_buy = 1, else 0
                if price > (trade_low * 0.99):
                    mkt_buy = 1
                else:
                    mkt_buy = 0
            else:
                mkt_buy = None

            # update fields in dataframe
            new_insider_data.at[index, 'Trade Close'] = trade_close
            new_insider_data.at[index, 'Trade Low'] = trade_low
            new_insider_data.at[index, 'Mkt Buy'] = mkt_buy
            new_insider_data.at[index, '52Wk Low'] = low_52
            new_insider_data.at[index, '52Wk High'] = high_52
            new_insider_data.at[index, '50Day Avg'] = avg_50
            new_insider_data.at[index, '200Day Avg'] = avg_200

        except Exception as e:
            logging.error(f'Error retrieving individual data for {ticker_symbol}: {e}', exc_info=True)

    # insert new transactions into database
    for index, row in new_insider_data.iterrows():
        cursor.execute('''
            INSERT OR IGNORE INTO insider_data (X, Filing_Date, Trade_Date,
            Ticker, Insider_Name, Title, Price, Qty, Trade_Close, Trade_Low,
            Mkt_Buy, Low_52, High_52, Avg_50, Avg_200) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (row['X'], row['Filing Date'], row['Trade Date'], row['Ticker'], 
            row['Insider Name'], row['Title'], row['Price'], row['Qty'], 
            row['Trade Close'], row['Trade Low'], row['Mkt Buy'],
            row['52Wk Low'], row['52Wk High'], row['50Day Avg'], 
            row['200Day Avg']))
        
    # commit to database
    db.commit()

    print(new_insider_data)                        

def get_closing_price(ticker, trade_date):
    # format trade date
    trade_date = pd.to_datetime(trade_date).strftime('%Y-%m-%d')

    # yFinance wont work with just one date or using the same dates
    # so create different start and end dates for our query then pull the desired date's price 
    start_date = (pd.to_datetime(trade_date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    end_date = (pd.to_datetime(trade_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    stock = yf.Ticker(ticker)
    history = stock.history(start=start_date, end=end_date)
    # print(trade_date, ticker, history.loc[trade_date]['Close'], history.loc[trade_date]['Low')
    
    # Check if the date is in the DataFrame
    if not history.empty:
        close_price = history.loc[trade_date]['Close']
        low_price = history.loc[trade_date]['Low']
        return round(close_price, 3), round(low_price, 3)
    else:
        return None, None  # case where the date or ticker data is not available

def get_price_info(ticker):
    # get and return 52 week low and high prices and moving 50 and 200 day average prices
    
    # get stock info
    stock = yf.Ticker(ticker)
    stock_info = stock.info

    # get desired info and return
    low_52 = stock_info.get('fiftyTwoWeekLow')
    high_52 = stock_info.get('fiftyTwoWeekHigh')
    avg_50 = stock_info.get('fiftyDayAverage')
    avg_200 = stock_info.get('twoHundredDayAverage')

    return round(low_52, 3), round(high_52, 3), round(avg_50, 3), round(avg_200, 3)

def connect_db(db_name):
    return sqlite3.connect(db_name)

def setup_logging(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
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
    
    cursor = db.cursor()
    
    # create run record
    cursor.execute('''
                   INSERT INTO runs (job_name, status)
                   VALUES (?, 'RUNNING')
                   ''', ('inside_scrape',))
    run_id = cursor.lastrowid
    db.commit()

    try:
        logger.info('Starting inside_scrape run_id=%s', run_id)
        
        main(db)
        
        cursor.execute('''
                       UPDATE runs
                       SET status = 'SUCCESS', finished_at=datetime('now')
                       WHERE run_id=?
                       ''', (run_id,))
        db.commit()
        
        logger.info('Completed inside_scrape run_id=%s', run_id)
        
    except Exception as e:
        logger.exception('Run failed run_id=%s', run_id)
        
        cursor.execute('''
                       UPDATE runs
                       SET status='FAILED',
                       finished_at=datetime('now'),
                       error_message=?
                       WHERE run_id=?
                       ''', (str(e), run_id))
        db.commit()
        
        raise
    
    finally:
        db.close()