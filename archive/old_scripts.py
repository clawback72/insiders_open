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