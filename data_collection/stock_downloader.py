import yfinance as yf
import pandas as pd
import os
from datetime import datetime
import sys

# Fix: __file__ is not defined in Colab notebooks.
# Using os.getcwd() to get the current directory and navigating up one level
# to typically reach the project root where 'config' might reside.
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# --- FIX: ModuleNotFoundError for 'config' --- #
# Define configuration variables directly as 'config' module was not found.
# You can uncomment the sys.path.append line and ensure your project structure
# has a 'config' directory with a 'config.py' file if you prefer external config.
STOCK_DATA_DIR = 'stock_data'
DEFAULT_TICKERS = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA']
DEFAULT_START_DATE = '2020-01-01'
# --- END FIX --- #


class StockDownloader:
    """Downloads historical stock data from Yahoo Finance"""

    def __init__(self, tickers=None, start_date=None, end_date=None):
        self.tickers = tickers or DEFAULT_TICKERS
        self.start_date = start_date or DEFAULT_START_DATE
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        os.makedirs(STOCK_DATA_DIR, exist_ok=True)

    def download_stock_data(self, ticker):
        print(f"Downloading {ticker}...")

        try:
            df = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False
            )

            if df.empty:
                print(f"[WARNING] No data found for {ticker}")
                return None

            df.reset_index(inplace=True)
            df["Ticker"] = ticker
            df = df[["Ticker", "Date", "Open", "High", "Low", "Close", "Volume"]]

            return df

        except Exception as e:
            print(f"[ERROR] Failed to download {ticker}: {e}")
            return None

    def download_all(self):
        stock_data = {}

        for ticker in self.tickers:
            df = self.download_stock_data(ticker)
            if df is not None:
                stock_data[ticker] = df
            else:
                print(f"[SKIPPED] {ticker} due to errors.")

        return stock_data

    def save_to_csv(self, stock_data):
        for ticker, df in stock_data.items():
            filename = f"{ticker}_stock_data.csv"
            path = os.path.join(STOCK_DATA_DIR, filename)
            df.to_csv(path, index=False)
            print(f"Saved: {path}")

    def validate_data(self, stock_data):
        for ticker, df in stock_data.items():

            # Minimum rows
            if len(df) < 200:
                print(f"[INVALID] {ticker}: Less than 200 rows")
                return False

            # Missing values
            if df[["Open", "High", "Low", "Close", "Volume"]].isnull().any().any():
                print(f"[INVALID] {ticker}: Missing critical values")
                return False

            # Duplicate dates
            if df["Date"].duplicated().any():
                print(f"[INVALID] {ticker}: Duplicate dates found")
                return False

            # Positive price checks
            if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
                print(f"[INVALID] {ticker}: Non-positive prices detected")
                return False

            # Outliers >50% daily change
            df["DailyChange"] = df["Close"].pct_change().abs()
            if df["DailyChange"].max() > 0.5:
                print(f"[INVALID] {ticker}: Extreme price swing detected")
                return False

        return True


def run_data_collection():
    print("=" * 60)
    print("STOCK DATA COLLECTION")
    print("=" * 60)

    downloader = StockDownloader()
    all_data = downloader.download_all()

    print("\nValidating data...")
    if not downloader.validate_data(all_data):
        print(" Data validation failed. Fix issues before saving.")
        return

    print("\nSaving files...")
    downloader.save_to_csv(all_data)

    print("\n Done! Stock dataset successfully downloaded & saved.")
    print(f"Tickers processed: {list(all_data.keys())}")


if __name__ == "__main__":
    run_data_collection()