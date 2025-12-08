# feature_importance_export.py
import os
import sys
import pandas as pd
from pyspark.sql import SparkSession

# Ensure project root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml_models.spark_gbt_forecaster import SparkGBTForecaster, MODEL_OUTDIR

spark = SparkSession.builder.master("local[*]") \
    .appName("ExportFeatureImportance") \
    .config("spark.driver.memory","4g") \
    .getOrCreate()

try:
    # Load model
    forecaster = SparkGBTForecaster(spark)
    forecaster.load_model(MODEL_OUTDIR)

    # Extract feature importances from the GBT model
    # The model is a PipelineModel, so we need to get the GBTRegressor stage (last stage)
    gbt_model = forecaster.model.stages[-1]  # Get GBTRegressionModel
    imp_vector = gbt_model.featureImportances  # gives a Vector

    # Feature names are lag features; reconstruct them
    # Based on ml_models/spark_gbt_forecaster.py: 5 bases * 30 lags = 150 features
    feature_bases = ["Close", "Open", "High", "Low", "Volume"]
    feature_names = []
    for base in feature_bases:
        for lag in range(1, 31):
            feature_names.append(f"{base}_lag_{lag}")

    importances = imp_vector.toArray().tolist()

    # Build DataFrame
    df = pd.DataFrame({
        "feature": feature_names[:len(importances)],  # in case counts don't match
        "importance": importances
    }).sort_values("importance", ascending=False)

    # Save
    os.makedirs("powerbi_exports", exist_ok=True)
    df.to_csv("powerbi_exports/feature_importance.csv", index=False)

    print("✅ feature_importance.csv generated")
    print(f"Top 10 features:\n{df.head(10).to_string(index=False)}")

except Exception as e:
    print(f"[ERROR] {e}")
    print("[INFO] Run this inside Docker instead:")
    print("[CMD] docker run --rm -v <path>:/workspace -w /workspace stock-analysis-app:latest python dashboard/feature_importance_export.py")
    spark.stop()
    exit(1)
