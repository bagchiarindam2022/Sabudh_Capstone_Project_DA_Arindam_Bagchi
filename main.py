"""
Main Pipeline Orchestrator - AI Financial Analysis Platform
Usage:
    python main.py
"""

import os
import sys
from pyspark.sql import SparkSession

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from config.config import PROCESSED_PARQUET
from sql_interface.database_manager import DatabaseManager
from ml_models.spark_gbt_forecaster import SparkGBTForecaster


spark = None


def initialize_spark():
    global spark
    if spark is not None:
        print("✅ Spark already initialized")
        return spark

    print("\n⚡ Initializing Spark session...")
    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("AI_Financial_Pipeline")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    print("✅ Spark initialized")
    return spark


def setup_database():
    print("\n🗄 Setting up SQLite database...")
    db = DatabaseManager()
    db.load_parquet(PROCESSED_PARQUET)
    db.close()
    print("✅ Database setup completed")


def train_ml_models():
    print("\n🤖 Training ML model (Spark GBT)...")
    spark = initialize_spark()

    forecaster = SparkGBTForecaster(spark)
    forecaster.load_data(PROCESSED_PARQUET)
    forecaster.train()
    forecaster.save_model()

    print("✅ ML model trained and saved")


def run_chatbot():
    print("\n💬 Running AI Chatbot (Streamlit)...")
    print("👉 Command to run manually:")
    print("streamlit run chatbot/ai_prediction_chatbot.py --server.port 8502")


def run_dashboard():
    print("\n📊 Power BI Dashboard")
    print("👉 Open Power BI Desktop")
    print("👉 Load CSV outputs from /data folder")
    print("👉 Use the provided .pbix file (if created)")
    print("⚠ Power BI is NOT launched via Python (by design)")


def run_complete_pipeline():
    initialize_spark()
    setup_database()
    train_ml_models()
    run_chatbot()
    run_dashboard()
    print("\n🎉 COMPLETE PIPELINE EXECUTED SUCCESSFULLY")


def main():
    while True:
        print("""
========= MAIN MENU =========
1. Initialize Spark
2. Setup Database
3. Train ML Models
4. Run Chatbot
5. Dashboard Instructions
6. Run Complete Pipeline
7. Exit
=============================
""")
        choice = input("Select option: ").strip()

        if choice == "1":
            initialize_spark()
        elif choice == "2":
            setup_database()
        elif choice == "3":
            train_ml_models()
        elif choice == "4":
            run_chatbot()
        elif choice == "5":
            run_dashboard()
        elif choice == "6":
            run_complete_pipeline()
        elif choice == "7":
            print("👋 Exiting pipeline")
            break
        else:
            print("❌ Invalid choice. Try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
