import requests
import pandas as pd
import yfinance as yf
import sqlite3
import logging
import pytz
import sys
from bs4 import BeautifulSoup as bs
from datetime import datetime, timedelta

# set up logging
logging.basicConfig(filename='scraper.log',
                    level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def main(db):

    # set up database
    # set cursor
    cursor = db.cursor()

    # define webpages
    clusterPage = "http://openinsider.com/latest-cluster-buys"
    insidersPage = 'http://openinsider.com/insider-purchases-25k'

    # create dataframe for cluster buys
    df_cluster = scrape(clusterPage, 'cluster')
    print(df_cluster.head())

    # send dataframe to insert_tickers to update database with new securities
    insert_tickers(cursor, df_cluster, db)

    # send cluster dataframe to insert_cluster to update database with new transactions
    insert_cluster(cursor, df_cluster, db)

    # delete cluster dataframe - we're done with it
    del df_cluster


    # create dataframe for insider buys
    df_insiders = scrape(insidersPage, 'insider')
    print(df_insiders.head())

    # send dataframe to insert_tickers to update database with new securities
    insert_tickers(cursor, df_insiders, db)

    # send dataframe to insert_insiders to update database with new transactions
    insert_insiders(cursor, df_insiders, db)

    # delete insider dataframe - we're done with it
    del df_insiders

    # create dataframe for insertion of indexes
    index_data = {
        'Ticker': ['^GSPC', '^DJI', '^IXIC', '^VIX', '^RUT', '^FTW5000'],
        'Name': ['S&P 500', 'Dow Jones Industrial Average', 'NASDAQ',
                 'CBOE Volatility Index', 'Russel 2000', 'Wilshire 5000']
    }

    df_index = pd.DataFrame(index_data)

    # insert indexes and prices to database
    insert_index(cursor, df_index, db)

def scrape(page, type):
    # load webpage
    r=requests.get(page)

    # convert to beautiful soup object
    soup = bs(r.content, features="lxml")

    # extract table
    table = soup.find('table', {'class': 'tinytable'})

    # extract headers
    headers = []
    for th in table.find_all('th'):
        headers.append(th.text.strip())

    # extract table rows
    rows = []
    for tr in table.find_all('tr')[1:]:  # Skip the header row
        cells = tr.find_all('td')
        row = [cell.text.strip() for cell in cells]
        rows.append(row)

    # create dataframes
    df = pd.DataFrame(rows, columns=headers)

    # get rid of unicode characters between column names
    df.columns = df.columns.str.replace('\xa0', ' ')

    # clean up columns depending on type
    if type == 'cluster':
        df = df[['X', 'Filing Date', 'Trade Date', 'Ticker', 'Company Name', 'Industry', 'Ins', 'Trade Type', 'Price', 'Qty', 'Owned', 'ΔOwn', 'Value']]
    elif type == 'insider':
        df = df[['X', 'Filing Date', 'Trade Date', 'Ticker', 'Company Name', 'Insider Name', 'Title', 'Trade Type', 'Price', 'Qty', 'Owned', 'ΔOwn', 'Value']]
    else:
        logging.error(f"No type specified for scraper function")
        sys.exit(1)

    # cast filing and trade dates to datetime
    df['Filing Date'] = pd.to_datetime(df['Filing Date'], errors='coerce')
    df['Trade Date'] = pd.to_datetime(df['Trade Date'], errors='coerce')

    # cast into string for database insertion
    # will have to be cast back to string for any yFinance queries
    df['Filing Date'] = df['Filing Date'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    df['Trade Date'] = df['Trade Date'].dt.strftime('%Y-%m-%d')

    #cast quantity to integer and price to float
    df['Qty'] = df['Qty'].replace(r'[^\d]', '', regex=True).astype(int)
    df['Price'] = df['Price'].replace(r'[^\d.]', '', regex=True).astype(float)

    # cast Ins column as integer
    if type == 'cluster':
        df['Ins'] = pd.to_numeric(df['Ins'], errors='coerce').fillna(0).astype(int)

    # return dataframe
    return(df)

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

def insert_cluster(cursor, df, db):
    # update database with new cluster transactions

    # get existing transactions from database
    query = "SELECT Filing_Date FROM cluster_data"
    existing_transactions = pd.read_sql(query, db)

    print(existing_transactions)

    # copy passed dataframe to get desired info for database
    df_cluster_data = df[['Filing Date', 'Trade Date', 'Ticker', 'Ins', 'Price', 'Qty']].copy()
    print(df_cluster_data)

    # create dataframe to isolate new tickers
    merged = pd.merge(df_cluster_data, existing_transactions, left_on='Filing Date', right_on='Filing_Date', how='left', indicator=True)

    new_cluster_data = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

    print(new_cluster_data)

    # get additional information columns added
    new_cluster_data['Trade Close'] = None
    new_cluster_data['Trade Low'] = None
    new_cluster_data['Mkt Buy'] = None

    for index, row in new_cluster_data.iterrows():
        ticker_symbol = row['Ticker']
        trade_date = row['Trade Date']
        price = row['Price']

        try:
            # get close and low price for transaction
            trade_close, trade_low = get_closing_price(ticker_symbol, trade_date)
            print(f'cluster pricing for {ticker_symbol} on {trade_date}:', trade_close, trade_low)

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
            new_cluster_data.at[index, 'Trade Close'] = trade_close
            new_cluster_data.at[index, 'Trade Low'] = trade_low
            new_cluster_data.at[index, 'Mkt Buy'] = mkt_buy

        except Exception as e:
            logging.error(f'Error retrieving or inserting cluster buy for {ticker_symbol}: {e}', exc_info=True)

    # insert new transactions into database
    for index, row in new_cluster_data.iterrows():
        cursor.execute('''
            INSERT OR IGNORE INTO cluster_data (Filing_Date, Trade_Date,
            Ticker, Ins, Price, Qty, Trade_Close, Trade_Low, Mkt_Buy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (row['Filing Date'], row['Trade Date'], row['Ticker'], 
            row['Ins'],row['Price'], row['Qty'], row['Trade Close'],
            row['Trade Low'], row['Mkt Buy']))
            
    # comit to database
    db.commit()
        
    print(new_cluster_data)

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

if __name__ == "__main__":
    # connect to database
    db = connect_db('insider_database.db')

    try:
        main(db)
    finally:
        db.close()
