import os
import sys
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DateType

# Add project root to path
# Fix for Colab: __file__ is not defined. Assume current working directory is the project root.
sys.path.append(os.getcwd())

# Try import config values; provide sane defaults if missing
try:
    from config.config import (
        STOCK_DATA_DIR, PROCESSED_DATA_DIR,
        MA_WINDOWS, RSI_WINDOW, VOLATILITY_WINDOW
    )
except Exception:
    # FIX: Align STOCK_DATA_DIR with where the StockDownloader saved the data.
    # The StockDownloader saves to 'stock_data', not 'data/stock_data/'.
    STOCK_DATA_DIR = "stock_data"
    PROCESSED_DATA_DIR = "data/"
    MA_WINDOWS = [7, 30, 90]
    RSI_WINDOW = 14
    VOLATILITY_WINDOW = 30


class SparkPreprocessor:
    """PySpark-based data preprocessing and feature engineering"""

    def __init__(self, spark=None):
        """
        Initialize preprocessor with Spark session and load CSV files into a DataFrame
        """
        # create or use provided spark session
        if spark is None:
            self.spark = SparkSession.builder \
                .appName("StockPreprocessing") \
                .config("spark.driver.memory", "4g") \
                .getOrCreate()
            self._created_spark = True
        else:
            self.spark = spark
            self._created_spark = False

        # ensure paths exist
        os.makedirs(STOCK_DATA_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

        # load CSVs into a single DataFrame
        self.df = self.load_csv_files(STOCK_DATA_DIR)
        if self.df is None:
            # create empty schema if nothing found to avoid crashes later
            self.df = self.spark.createDataFrame([], schema="Ticker string, Date string, Open double, High double, Low double, Close double, Volume double")

    def load_csv_files(self, input_dir):
        """
        Load all *_stock_data.csv files from input_dir into a single Spark DataFrame.
        Expects columns: Ticker, Date, Open, High, Low, Close, Volume
        """
        csv_files = []
        for fname in os.listdir(input_dir):
            if fname.lower().endswith("_stock_data.csv"):
                csv_files.append(os.path.join(input_dir, fname))

        if not csv_files:
            print(f"[WARN] No CSV files found in {input_dir}")
            return None

        # Read all CSVs (Spark can read a list)
        df = self.spark.read.option("header", True).option("inferSchema", True).csv(csv_files)

        # Normalize column names and types
        expected_cols = ["Ticker", "Date", "Open", "High", "Low", "Close", "Volume"]
        for col in expected_cols:
            if col not in df.columns:
                # try lowercase or alternative names mapping
                for alt in df.columns:
                    if alt.lower() == col.lower():
                        df = df.withColumnRenamed(alt, col)
                        break

        # cast types
        df = df.withColumn("Date", F.to_date(F.col("Date"))) \
               .withColumn("Open", F.col("Open").cast("double")) \
               .withColumn("High", F.col("High").cast("double")) \
               .withColumn("Low", F.col("Low").cast("double")) \
               .withColumn("Close", F.col("Close").cast("double")) \
               .withColumn("Volume", F.col("Volume").cast("double"))

        # drop rows with null date or ticker because those break ordering/partitioning
        df = df.dropna(subset=["Ticker", "Date"])

        # Ensure sorted order per ticker
        df = df.select(*expected_cols).orderBy("Ticker", "Date")
        return df

    def calculate_moving_averages(self, df, windows=None):
        """
        Calculate moving averages using PySpark Window functions
        """
        windows = windows or MA_WINDOWS
        print(f"\nCalculating moving averages: {windows}")

        # base window specification (rowsBetween uses integer offsets)
        for n in windows:
            w = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-n + 1, 0)
            col_name = f"MA_{n}"
            df = df.withColumn(col_name, F.avg(F.col("Close")).over(w))

        return df

    def calculate_rsi(self, df, window=None):
        """
        Calculate RSI (Relative Strength Index)
        """
        window = window or RSI_WINDOW
        print(f"\nCalculating RSI (window={window})...")

        # Window specs
        w_order = Window.partitionBy("Ticker").orderBy("Date")
        w_roll = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-(window - 1), 0)

        # Step 1: Change
        df = df.withColumn("prev_close", F.lag("Close", 1).over(w_order))
        df = df.withColumn("Change", F.when(F.col("prev_close").isNull(), F.lit(0.0)).otherwise(F.col("Close") - F.col("prev_close")))

        # Step 2: Gains and Losses
        df = df.withColumn("Gain", F.when(F.col("Change") > 0, F.col("Change")).otherwise(0.0))
        df = df.withColumn("Loss", F.when(F.col("Change") < 0, -F.col("Change")).otherwise(0.0))

        # Step 3: Average gain and loss over rolling window
        df = df.withColumn("AvgGain", F.avg("Gain").over(w_roll))
        df = df.withColumn("AvgLoss", F.avg("Loss").over(w_roll))

        # Step 4: RS
        df = df.withColumn("RS", F.when(F.col("AvgLoss") == 0, None).otherwise(F.col("AvgGain") / F.col("AvgLoss")))

        # Step 5: RSI
        # If AvgLoss == 0 and AvgGain == 0 => RSI = 50 (neutral); if AvgLoss == 0 and AvgGain > 0 => RSI = 100
        df = df.withColumn(
            "RSI",
            F.when((F.col("AvgGain") == 0) & (F.col("AvgLoss") == 0), F.lit(50.0))
             .when(F.col("AvgLoss") == 0, F.lit(100.0))
             .otherwise(100.0 - (100.0 / (1.0 + F.col("RS"))))
        )

        # Drop intermediates
        df = df.drop("prev_close", "Change", "Gain", "Loss", "AvgGain", "AvgLoss", "RS")
        return df

    def calculate_volatility(self, df, window=None):
        """
        Calculate rolling volatility (standard deviation of close prices)
        """
        window = window or VOLATILITY_WINDOW
        print(f"\nCalculating volatility (window={window})...")

        w_roll = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-(window - 1), 0)
        df = df.withColumn("Volatility", F.stddev(F.col("Close")).over(w_roll))
        return df

    def calculate_returns(self, df):
        """
        Calculate daily returns and Sharpe ratio
        """
        print(f"\nCalculating returns and Sharpe ratio...")

        w_order = Window.partitionBy("Ticker").orderBy("Date")
        # daily return
        df = df.withColumn("prev_close", F.lag("Close", 1).over(w_order))
        df = df.withColumn("Daily_Return", F.when(F.col("prev_close").isNull(), None).otherwise((F.col("Close") - F.col("prev_close")) / F.col("prev_close")))

        # Sharpe ratio per ticker (mean / stddev of daily returns). Use partition-level aggregates.
        w_part = Window.partitionBy("Ticker")
        mean_ret = F.mean("Daily_Return").over(w_part)
        std_ret = F.stddev("Daily_Return").over(w_part)

        df = df.withColumn("Sharpe_Ratio", F.when(std_ret.isNull() | (std_ret == 0), None).otherwise(mean_ret / std_ret))

        # drop temp
        df = df.drop("prev_close")
        print(f"  ✅ Created Daily_Return and Sharpe_Ratio columns")
        return df

    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset
        Strategy used:
        - Drop rows missing critical OHLCV columns
        - Forward-fill indicator columns per ticker by last non-null value
        """
        print(f"\nHandling missing values...")

        # 1) Drop rows with missing critical values
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

        # 2) Forward fill indicator-like columns (MA_*, RSI, Volatility, Daily_Return, Sharpe_Ratio)
        indicator_cols = [c for c in df.columns if c.startswith("MA_")] + ["RSI", "Volatility", "Daily_Return", "Sharpe_Ratio"]

        # window to get last non-null value up to current row
        w_ffill = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(Window.unboundedPreceding, 0)

        for col in indicator_cols:
            if col in df.columns:
                df = df.withColumn(f"{col}_ffill", F.last(F.col(col), ignorenulls=True).over(w_ffill))
                df = df.drop(col).withColumnRenamed(f"{col}_ffill", col)

        # Optionally drop any remaining rows with nulls if indicators still missing (e.g., very first rows)
        df = df.dropna()

        return df

    def save_to_parquet(self, df, output_path):
        """
        Save processed data to Parquet format
        """
        print(f"\nSaving to Parquet: {output_path}")
        df.write.mode("overwrite").parquet(output_path)
        print(f"✅ Saved processed data to {output_path}")

    def run_pipeline(self):
        """
        Run the full preprocessing pipeline and return processed DataFrame
        """
        df = self.df

        # Ensure Date is date typed and sort
        if "Date" in df.columns and dict(df.dtypes)["Date"] != "date":
            df = df.withColumn("Date", F.to_date(F.col("Date")))

        df = df.orderBy("Ticker", "Date")

        # Feature engineering
        df = self.calculate_moving_averages(df, windows=MA_WINDOWS)
        df = self.calculate_rsi(df, window=RSI_WINDOW)
        df = self.calculate_volatility(df, window=VOLATILITY_WINDOW)
        df = self.calculate_returns(df)

        # Handle missing values
        df = self.handle_missing_values(df)

        # select & order output columns required by assignment
        out_cols = ["Ticker", "Date", "Open", "High", "Low", "Close", "Volume"]
        # add MA columns if present
        for n in MA_WINDOWS:
            col = f"MA_{n}"
            if col in df.columns:
                out_cols.append(col)
        if "RSI" in df.columns:
            out_cols.append("RSI")
        if "Volatility" in df.columns:
            out_cols.append("Volatility")
        if "Daily_Return" in df.columns:
            out_cols.append("Daily_Return")
        if "Sharpe_Ratio" in df.columns:
            out_cols.append("Sharpe_Ratio")

        processed = df.select(*out_cols).orderBy("Ticker", "Date")
        return processed


def run_preprocessing(spark=None):
    """
    Main function to run preprocessing pipeline
    """
    print("=" * 60)
    print("PYSPARK DATA PREPROCESSING")
    print("=" * 60)

    pre = SparkPreprocessor(spark=spark)
    processed_df = pre.run_pipeline()

    output_path = os.path.join(PROCESSED_DATA_DIR, "processed_stocks.parquet")
    pre.save_to_parquet(processed_df, output_path)

    print("\nSample of processed data:")
    processed_df.show(5, truncate=False)

    return processed_df


if __name__ == "__main__":
    """
    Run this file to test your preprocessing

    Usage: python preprocessing/spark_preprocessor.py
    """
    spark = SparkSession.builder \
        .appName("StockPreprocessing") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    run_preprocessing(spark)

    spark.stop()
