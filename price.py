import logging
import sqlite3
import sys
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from inside_scrape import get_historical_prices

# set up logging
logging.basicConfig(
    filename="price.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main(db):

    # set cursor
    cursor = db.cursor()

    # set query for all tickers and create dataframe with it
    equity_query = "SELECT DISTINCT Ticker FROM ticker_data;"
    df_all_tickers = pd.read_sql(equity_query, db)

    # set number of days back to pull prices
    equity_pricing_term = 5

    # send dataframe to get tickers priced for term
    get_historical_prices(cursor, df_all_tickers, db, equity_pricing_term)

    # delete ticker dataframe - we're done with it
    del df_all_tickers

    # set query for all Indexes and create dataframe with it
    index_query = "SELECT DISTINCT TICKER FROM index_data;"
    df_all_indexes = pd.read_sql(index_query, db)

    # set number of days back to pull index prices
    index_pricing_term = 5

    # send dataframe to get indexes priced for term
    get_historical_prices(cursor, df_all_indexes, db, index_pricing_term)

    # delete index dataframe - we're done with it
    del df_all_indexes


def connect_db(db_name):
    return sqlite3.connect(db_name)


if __name__ == "__main__":
    # connect to database
    db = connect_db("insider_database.db")

    try:
        main(db)
    finally:
        db.close()
