import os
import pytest

from chatbot.ai_prediction_chatbot import detect_intent


def test_detect_intent_basic():
    assert detect_intent("Predict AAPL next 7 days") == "predict"
    assert detect_intent("Show TSLA data") == "show"
    assert detect_intent("What is RSI?") == "explain"


def test_processed_parquet_exists():
    # The preprocessing step should have produced this file
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed_stocks.parquet")
    assert os.path.exists(path), f"Processed parquet not found at {path}"


def test_model_directory_exists():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(repo_root, "models", "gbt_forecaster")
    assert os.path.isdir(model_dir), f"Model directory not found: {model_dir}"
    # Expect metadata directory to exist inside the model dir
    metadata = os.path.join(model_dir, "metadata")
    assert os.path.exists(metadata), "Model metadata folder missing"


def _pyspark_available():
    try:
        import pyspark  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _pyspark_available(), reason="PySpark not available in this environment")
def test_forecaster_can_load_model_and_predict(tmp_path):
    # This test will attempt to create a SparkSession, load the model and run a tiny prediction.
    from pyspark.sql import SparkSession
    from ml_models.spark_gbt_forecaster import SparkGBTForecaster, PROCESSED_PARQUET

    spark = SparkSession.builder.master("local[1]").appName("test_forecaster").getOrCreate()
    forecaster = SparkGBTForecaster(spark)

    # Try to load processed parquet - if not present this will raise and the test should fail
    assert os.path.exists(PROCESSED_PARQUET), f"Processed parquet not found: {PROCESSED_PARQUET}"
    forecaster.load_data(PROCESSED_PARQUET)

    # Load model (will raise if model missing)
    forecaster.load_model("models/gbt_forecaster")

    # Pick a ticker from the processed file (read small sample via pandas)
    import pandas as pd
    pdf = pd.read_parquet(PROCESSED_PARQUET)
    tickers = pdf["Ticker"].unique().tolist()
    assert len(tickers) > 0, "No tickers found in processed data"

    # Use first ticker to predict; wrapped to ensure we don't crash the test process
    preds = forecaster.predict_future(tickers[0], num_days=1)
    assert isinstance(preds, list)
    assert len(preds) == 1

    spark.stop()
