import os
import sqlite3
import pandas as pd


class DatabaseManager:
    """
    Handles SQLite storage for processed stock data.
    Creates database, loads Parquet data, runs queries.
    """

    def __init__(self, db_path="data/financial_data.db"):
        self.db_path = db_path

        # Ensure parent folder exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Connect to SQLite
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # Create table if needed
        self.create_tables()

    def create_tables(self):
        """Create database tables"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS stock_data (
            ticker TEXT NOT NULL,
            date DATETIME NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            ma_7 REAL,
            ma_30 REAL,
            ma_90 REAL,
            rsi REAL,
            volatility REAL,
            daily_return REAL,
            sharpe_ratio REAL,
            PRIMARY KEY (ticker, date)
        );
        """

        self.cursor.execute(create_table_sql)

        # Index for faster queries
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON stock_data (ticker)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON stock_data (date)")

        self.conn.commit()

    def load_parquet(self, parquet_path):
        """
        Load processed parquet data into SQLite database
        """
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        print(f"\nLoading parquet into SQLite: {parquet_path}")

        # Use pandas to read Parquet
        df = pd.read_parquet(parquet_path)

        # Insert into SQLite table
        df.to_sql("stock_data", self.conn, if_exists="replace", index=False)

        print("✅ Loaded data into SQLite successfully.")
        print(f"📦 Rows inserted: {len(df)}")

    def get_stock_data(self, ticker):
        """
        Query all historical data for a specific stock
        """
        query = """
        SELECT *
        FROM stock_data
        WHERE ticker = ?
        ORDER BY date;
        """

        df = pd.read_sql_query(query, self.conn, params=[ticker])
        return df

    def get_latest_prices(self):
        """
        Fetch the latest available closing price for each ticker
        """
        query = """
        SELECT s1.*
        FROM stock_data s1
        JOIN (
            SELECT ticker, MAX(date) AS max_date
            FROM stock_data
            GROUP BY ticker
        ) s2
        ON s1.ticker = s2.ticker AND s1.date = s2.max_date
        ORDER BY s1.ticker;
        """

        df = pd.read_sql_query(query, self.conn)
        return df

    def close(self):
        self.conn.close()