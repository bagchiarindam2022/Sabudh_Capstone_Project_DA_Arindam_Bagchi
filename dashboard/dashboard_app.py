import os
import sys
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession

# Ensure project root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml_models.spark_gbt_forecaster import SparkGBTForecaster, PROCESSED_PARQUET, MODEL_OUTDIR

EXPORT_DIR = "powerbi_exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

# --------------------------
# 1. START SPARK
# --------------------------
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("PowerBI_Exports") \
    .getOrCreate()

print("[INFO] Spark started.")

# --------------------------
# 2. LOAD HISTORICAL DATA
# --------------------------
if not os.path.exists(PROCESSED_PARQUET):
    print(f"[ERROR] {PROCESSED_PARQUET} not found!")
    print("[INFO] To generate this file, run: python preprocessing/spark_preprocessor.py")
    print("[INFO] Or run inside Docker: docker run --rm ... stock-analysis-app:latest python preprocessing/spark_preprocessor.py")
    spark.stop()
    exit(1)

try:
    df_spark = spark.read.parquet(PROCESSED_PARQUET)
    df = df_spark.toPandas()
    df["Date"] = pd.to_datetime(df["Date"]).dt.date

    # rename for Power BI clarity
    df.rename(columns={"Ticker": "ticker", "Date": "date"}, inplace=True)

    print("[INFO] Historical data loaded:", df.shape)

    df.to_csv(f"{EXPORT_DIR}/historical_data.csv", index=False)
    print("[OK] historical_data.csv exported.")
except Exception as e:
    print(f"[ERROR] Failed to load parquet: {e}")
    print("[INFO] Run this script inside Docker (which has Hadoop support) instead:")
    print("[CMD] docker run --rm -v <path>:/workspace -w /workspace stock-analysis-app:latest python dashboard/dashboard_app.py")
    spark.stop()
    exit(1)

# --------------------------
# 3. TECHNICAL INDICATORS
# --------------------------
def compute_indicators(group):
    group = group.sort_values("date")
    group["MA7"] = group["Close"].rolling(7).mean()
    group["MA30"] = group["Close"].rolling(30).mean()
    group["MA90"] = group["Close"].rolling(90).mean()

    # RSI
    delta = group["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    group["RSI"] = 100 - (100 / (1 + rs))

    # Volatility = Std dev of returns (20-day)
    group["Return"] = group["Close"].pct_change()
    group["Volatility20"] = group["Return"].rolling(20).std()

    return group

indicators = df.groupby("ticker").apply(compute_indicators).reset_index(drop=True)

indicators.to_csv(f"{EXPORT_DIR}/technical_indicators.csv", index=False)
print("[OK] technical_indicators.csv exported.")

# --------------------------
# 4. ML PREDICTIONS
# --------------------------
print("[INFO] Loading GBT model...")

try:
    forecaster = SparkGBTForecaster(spark)
    forecaster.load_data(PROCESSED_PARQUET)
    forecaster.load_model(MODEL_OUTDIR)
except Exception as e:
    print("[ERROR] Model loading failed:", e)
    print("Skipping ML predictions export.")
    forecaster = None

predictions_all = []

if forecaster:
    tickers = df["ticker"].unique()

    for t in tickers:
        preds = forecaster.predict_future(t, num_days=7)
        for i, p in enumerate(preds, start=1):
            predictions_all.append({
                "ticker": t,
                "day": i,
                "predicted_close": p
            })

    pred_df = pd.DataFrame(predictions_all)
    pred_df.to_csv(f"{EXPORT_DIR}/ml_predictions.csv", index=False)
    print("[OK] ml_predictions.csv exported.")

# --------------------------
# 5. SIMPLE INVESTMENT CLASSIFICATION
# --------------------------
print("[INFO] Generating investment classifications...")

classification = []

for t in df["ticker"].unique():
    recent = df[df["ticker"] == t].tail(30)
    mean_close = recent["Close"].mean()

    if mean_close >= 300:
        cls = "High Value"
    elif mean_close >= 100:
        cls = "Medium Value"
    else:
        cls = "Low Value"

    classification.append({
        "ticker": t,
        "avg_close": float(mean_close),
        "class": cls
    })

class_df = pd.DataFrame(classification)
class_df.to_csv(f"{EXPORT_DIR}/investment_classification.csv", index=False)

print("[OK] investment_classification.csv exported.")
print("\n🎉 All Power BI files exported successfully!")
