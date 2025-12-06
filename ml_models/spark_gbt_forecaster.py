# ml_models/spark_gbt_forecaster.py
import os
import math
import random
from typing import List

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.pipeline import PipelineModel

import numpy as np

# Config / paths
PROCESSED_PARQUET = "data/processed_stocks.parquet"
MODEL_OUTDIR = "models/gbt_forecaster"
LAG_MAX = 30
FUTURE_HORIZON = 7  # predict 7 days ahead as required
SEED = 42


class SparkGBTForecaster:
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.model: PipelineModel = None
        self.feature_cols: List[str] = []
        # underlying DataFrame loaded from parquet
        self.df: DataFrame = None

    def load_data(self, parquet_path=PROCESSED_PARQUET):
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Processed parquet not found: {parquet_path}")

        print(f"Loading processed parquet: {parquet_path}")
        df = self.spark.read.parquet(parquet_path)

        # ensure Date is date and sorting exists
        df = df.withColumn("Date", F.to_date(F.col("Date")))

        # keep required columns (Open, High, Low, Close, Volume, Ticker, Date)
        cols_expected = ["Ticker", "Date", "Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in cols_expected if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in parquet: {missing}")

        self.df = df.select(*cols_expected).orderBy("Ticker", "Date")
        print("Loaded and sorted DataFrame.")
        return self.df

    def create_lag_features(self, df: DataFrame, lag_max=LAG_MAX):
        """
        Create lag features in batches to avoid deep execution plans and stack overflow.
        We create features for Close, Open, High, Low, Volume.
        """
        print(f"Creating lag features up to {lag_max} days...")

        feature_bases = ["Close", "Open", "High", "Low", "Volume"]
        self.feature_cols = []

        # Use a partition window per ticker ordered by date
        win = Window.partitionBy("Ticker").orderBy("Date")

        # Create lags in smaller batches to reduce plan explosion; batch size ~10
        batch_size = 10
        for batch_start in range(1, lag_max + 1, batch_size):
            batch_end = min(batch_start + batch_size - 1, lag_max)
            print(f"  Creating lags {batch_start}..{batch_end} (batch)")
            for base in feature_bases:
                for lag in range(batch_start, batch_end + 1):
                    col_name = f"{base}_lag_{lag}"
                    df = df.withColumn(col_name, F.lag(F.col(base), lag).over(win))
                    self.feature_cols.append(col_name)

            # cache and count to materialize the batch and reduce plan size
            df = df.cache()
            df.count()

        # drop any rows with nulls in the lag features (these are early rows without full history)
        df = df.dropna(subset=self.feature_cols + ["Close"])
        print(f"  Created {len(self.feature_cols)} lag features.")
        self.df = df
        return df

    def create_label(self, df: DataFrame, horizon=FUTURE_HORIZON):
        """
        Create label column = Close price 'horizon' days in the future.
        We use lead(Close, horizon) over the per-ticker window.
        """
        print(f"Creating {horizon}-day ahead label column...")
        win = Window.partitionBy("Ticker").orderBy("Date")
        df = df.withColumn("label", F.lead("Close", horizon).over(win))
        # drop rows where label is null (near the end of each ticker history)
        df = df.dropna(subset=["label"])
        self.df = df
        return df

    def train_val_test_split(self, df: DataFrame):
        """
        Chronological split per ticker into 80% train, 10% val, 10% test.
        We compute a row number per ticker and use fraction thresholds to split.
        """
        print("Splitting data into train/validation/test chronologically (per ticker)...")
        w = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(Window.unboundedPreceding, 0)
        # Row number per ticker
        df = df.withColumn("row_num", F.row_number().over(Window.partitionBy("Ticker").orderBy("Date")))
        # total counts per ticker
        counts = df.groupBy("Ticker").agg(F.max("row_num").alias("max_row"))
        df = df.join(counts, on="Ticker", how="left")
        # fractional position
        df = df.withColumn("frac", F.col("row_num") / F.col("max_row"))

        train_df = df.filter(F.col("frac") <= 0.8).drop("row_num", "max_row", "frac")
        val_df = df.filter((F.col("frac") > 0.8) & (F.col("frac") <= 0.9)).drop("row_num", "max_row", "frac")
        test_df = df.filter(F.col("frac") > 0.9).drop("row_num", "max_row", "frac")

        # force materialization & cache
        train_df = train_df.cache(); train_df.count()
        val_df = val_df.cache(); val_df.count()
        test_df = test_df.cache(); test_df.count()

        print(f"  Train rows: {train_df.count()}, Val rows: {val_df.count()}, Test rows: {test_df.count()}")
        return train_df, val_df, test_df

    def build_and_train(self, train_df: DataFrame, val_df: DataFrame, test_df: DataFrame):
        """
        Assemble feature vector, train GBTRegressor, evaluate on validation/test, save model.
        """
        print("Assembling features and training GBTRegressor...")

        assembler = VectorAssembler(inputCols=self.feature_cols, outputCol="features", handleInvalid="skip")

        # GBT hyperparameters as requested
        gbt = GBTRegressor(
            featuresCol="features",
            labelCol="label",
            predictionCol="prediction",
            maxIter=100,
            maxDepth=6,
            stepSize=0.1,
            subsamplingRate=0.8,
            seed=SEED
        )

        pipeline = Pipeline(stages=[assembler, gbt])

        # Fit model on train; we might use validation to pick best model in a real pipeline,
        # but here we just train and evaluate on val/test.
        model = pipeline.fit(train_df)

        # Save model
        os.makedirs(MODEL_OUTDIR, exist_ok=True)
        model.write().overwrite().save(MODEL_OUTDIR)
        print(f"Saved model to: {MODEL_OUTDIR}")

        # evaluate
        self.model = model
        self.evaluate(test_df)
        return model

    def evaluate(self, df_test: DataFrame):
        print("Evaluating model on test set...")

        # Ensure model is loaded
        if self.model is None:
            raise RuntimeError("Model not trained/loaded.")

        preds = self.model.transform(df_test).select("Ticker", "Date", "label", "prediction")

        # Evaluate metrics using Spark's evaluator and manual computation for mean % error
        evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
        evaluator_mae = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
        evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

        rmse = evaluator_rmse.evaluate(preds)
        mae = evaluator_mae.evaluate(preds)
        r2 = evaluator_r2.evaluate(preds)

        # Mean % Error computed manually (avoid division by zero)
        preds_nonzero = preds.filter(F.col("label") != 0)
        mpe_df = preds_nonzero.withColumn("pct_err", F.abs((F.col("prediction") - F.col("label")) / F.col("label")))
        mean_pct_err = mpe_df.agg(F.mean("pct_err")).first()[0] if mpe_df.count() > 0 else None
        mean_pct_err_percent = float(mean_pct_err) * 100.0 if mean_pct_err is not None else None

        print("Model Performance:")
        print(f"  Test RMSE: {rmse:.4f}")
        print(f"  Test MAE: {mae:.4f}")
        print(f"  Test R^2: {r2:.4f}")
        if mean_pct_err_percent is not None:
            print(f"  Mean % Error: {mean_pct_err_percent:.2f}%")
        else:
            print("  Mean % Error: N/A (possible division by zero)")

        return {"rmse": rmse, "mae": mae, "r2": r2, "mean_pct_error": mean_pct_err_percent}

    def load_model(self, model_dir=MODEL_OUTDIR):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        self.model = PipelineModel.load(model_dir)
        print(f"Loaded model from {model_dir}")
        return self.model

    def predict_future(self, ticker: str, num_days: int = 7) -> List[float]:
        """
        Predict the next num_days for the given ticker using the trained model.
        Approach:
          - collect last LAG_MAX rows for the ticker into memory
          - construct a feature vector for the most recent day (with lag1 = last Close, etc.)
          - iteratively predict next day, then shift feature history to include predicted value
          - add a small random noise per day (std = 1% of prediction) to mimic realistic variation
        """
        if self.model is None:
            raise RuntimeError("Model is not trained or loaded.")

        # load ticker history
        hist = (
            self.df
            .filter(F.col("Ticker") == ticker)
            .orderBy(F.col("Date").desc())
            .limit(LAG_MAX)
            .select("Date", "Open", "High", "Low", "Close", "Volume")
        ).toPandas()

        if hist.shape[0] < LAG_MAX:
            raise ValueError(f"Not enough history for ticker {ticker}: need {LAG_MAX}, got {hist.shape[0]}")

        # hist is in descending order; convert to ascending for lags (oldest -> latest)
        hist = hist.iloc[::-1].reset_index(drop=True)

        # helper to build feature vector from current arrays
        def build_features_from_arrays(close_arr, open_arr, high_arr, low_arr, vol_arr):
            feat = {}
            # close_lag_1 is most recent previous -> in our arrays last element is the most recent
            for lag in range(1, LAG_MAX + 1):
                feat[f"Close_lag_{lag}"] = close_arr[-lag]
                feat[f"Open_lag_{lag}"] = open_arr[-lag]
                feat[f"High_lag_{lag}"] = high_arr[-lag]
                feat[f"Low_lag_{lag}"] = low_arr[-lag]
                feat[f"Volume_lag_{lag}"] = vol_arr[-lag]
            return feat

        # prepare arrays
        close_arr = hist["Close"].tolist()
        open_arr = hist["Open"].tolist()
        high_arr = hist["High"].tolist()
        low_arr = hist["Low"].tolist()
        vol_arr = hist["Volume"].tolist()

        preds = []
        for day in range(num_days):
            feat_dict = build_features_from_arrays(close_arr, open_arr, high_arr, low_arr, vol_arr)

            # create a single-row spark df for prediction
            row = self.spark.createDataFrame([feat_dict])
            # assemble features using the pipeline's assembler stage name; our pipeline expects inputCols = self.feature_cols
            # The pipeline model will include the assembler and the model, so transform will handle vectorization.
            pred = self.model.transform(row).select("prediction").collect()[0]["prediction"]

            # add small realistic variation: gaussian noise with std = 1% of predicted value
            noise = random.gauss(0, 0.01 * pred)
            pred_noisy = float(pred + noise)

            preds.append(pred_noisy)

            # shift arrays: drop oldest, append predicted values for next iteration
            close_arr.pop(0); close_arr.append(pred_noisy)
            # for other series (open/high/low/volume), we approximate by using last known values (or small perturbations)
            # For simplicity, assume open/high/low scale with close; keep volume same as last with small noise.
            last_open = open_arr[-1]
            last_high = high_arr[-1]
            last_low = low_arr[-1]
            last_vol = vol_arr[-1]

            # approximate relation for next day's OHLC (naive approach)
            open_next = last_open + (pred_noisy - close_arr[-2]) * 0.25
            high_next = max(last_high, pred_noisy) * (1 + random.uniform(0, 0.005))
            low_next = min(last_low, pred_noisy) * (1 - random.uniform(0, 0.005))
            vol_next = max(1.0, last_vol * (1 + random.uniform(-0.02, 0.02)))

            open_arr.pop(0); open_arr.append(open_next)
            high_arr.pop(0); high_arr.append(high_next)
            low_arr.pop(0); low_arr.append(low_next)
            vol_arr.pop(0); vol_arr.append(vol_next)

        return preds


def run_training_pipeline(spark: SparkSession):
    forecaster = SparkGBTForecaster(spark)
    df = forecaster.load_data(PROCESSED_PARQUET)
    df = forecaster.create_lag_features(df, lag_max=LAG_MAX)
    df = forecaster.create_label(df, horizon=FUTURE_HORIZON)
    train_df, val_df, test_df = forecaster.train_val_test_split(df)
    model = forecaster.build_and_train(train_df, val_df, test_df)
    return forecaster


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("SparkGBTForecaster") \
        .config("spark.driver.memory", "6g") \
        .getOrCreate()

    forecaster = run_training_pipeline(spark)

    # example: predict next 7 days for AAPL
    try:
        preds = forecaster.predict_future("AAPL", num_days=7)
        print("7-day predictions for AAPL:", preds)
    except Exception as e:
        print("Prediction error:", e)

    spark.stop()
