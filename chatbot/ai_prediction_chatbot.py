import os
import io
import sys
import base64
import sqlite3
import traceback
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import requests

from pyspark.sql import SparkSession

# Ensure project root is on sys.path so package imports work under Streamlit
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
cwd = os.getcwd()
for p in (ROOT, cwd):
    if p and p not in sys.path:
        sys.path.insert(0, p)

# Import the forecaster class and paths from the existing module
from ml_models.spark_gbt_forecaster import SparkGBTForecaster, PROCESSED_PARQUET, MODEL_OUTDIR


APP_STATE: Dict[str, Any] = {}


def create_spark() -> SparkSession:
    if "spark" in APP_STATE:
        return APP_STATE["spark"]
    spark = SparkSession.builder.master("local[*]") \
        .appName("AI_Prediction_Chatbot") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    APP_STATE["spark"] = spark
    return spark


def ollama_available() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def query_database(ticker: str, limit: int = 60) -> pd.DataFrame:
    """Return recent historical rows for `ticker` as a pandas DataFrame.

    If a SQLite DB `data/stocks.db` exists and has a `stock_data` (or `stocks`) table, query it.
    Otherwise fallback to reading `data/processed_stocks.parquet` with Spark.
    """
    ticker = ticker.upper()
    db_path = os.path.join("data", "stocks.db")

    # Try SQLite first
    try:
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            # Try common table names
            for table in ("stock_data", "stocks", "stock_prices"):
                try:
                    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
                    if cursor.fetchone():
                        q = f"SELECT ticker, date, open, high, low, close, volume FROM {table} WHERE ticker = ? ORDER BY date DESC LIMIT ?"
                        df = pd.read_sql_query(q, conn, params=(ticker, limit))
                        conn.close()
                        if not df.empty:
                            df["date"] = pd.to_datetime(df["date"]).dt.date
                            return df.sort_values("date")
                except Exception:
                    continue
            conn.close()
    except Exception:
        # Fall through to parquet
        pass

    # Fallback to parquet via Spark
    try:
        spark = create_spark()
        if not os.path.exists(PROCESSED_PARQUET):
            return pd.DataFrame()
        sdf = spark.read.parquet(PROCESSED_PARQUET).filter("Ticker = '{}'".format(ticker)).orderBy("Date", ascending=False).limit(limit)
        pdf = sdf.toPandas()
        if "Date" in pdf.columns:
            pdf["Date"] = pd.to_datetime(pdf["Date"])
            pdf = pdf.rename(columns={"Date": "date"})
        return pdf.sort_values("date")
    except Exception:
        traceback.print_exc()
        return pd.DataFrame()


def get_prediction(ticker: str, days: int = 7) -> List[float]:
    """Load (or reuse) the saved forecaster and return next `days` predictions for `ticker`.

    This uses the existing `SparkGBTForecaster` class so behavior matches training code.
    """
    try:
        spark = create_spark()

        # cached forecaster
        if "forecaster" not in APP_STATE:
            forecaster = SparkGBTForecaster(spark)
            # load processed data (required to set forecaster.df used by predict_future)
            forecaster.load_data(PROCESSED_PARQUET)
            # load model if exists
            try:
                forecaster.load_model(MODEL_OUTDIR)
            except Exception:
                # model not found
                APP_STATE["forecaster"] = forecaster
                return []
            APP_STATE["forecaster"] = forecaster

        forecaster: SparkGBTForecaster = APP_STATE["forecaster"]
        preds = forecaster.predict_future(ticker, num_days=days)
        return preds
    except Exception:
        traceback.print_exc()
        return []


def generate_prediction_graph(ticker: str, hist_df: pd.DataFrame, preds: List[float]) -> bytes:
    """Create a PNG image (bytes) of historical close prices + predicted future points."""
    try:
        plt.style.use("seaborn-darkgrid")
        fig, ax = plt.subplots(figsize=(9, 4))

        if not hist_df.empty:
            # identify close column (case-insensitive)
            close_col = None
            for c in hist_df.columns:
                if c.lower() == "close":
                    close_col = c
                    break
            if close_col is None:
                # try Close or close
                if "Close" in hist_df.columns:
                    close_col = "Close"
            dates = pd.to_datetime(hist_df[hist_df.columns[1]]) if "date" not in hist_df.columns and "Date" in hist_df.columns else pd.to_datetime(hist_df.get("date", hist_df.get("Date")))
            ax.plot(dates, hist_df[close_col], label="Historical Close", color="#2b8cbe")

            last_date = dates.max()
        else:
            last_date = pd.Timestamp.today()

        # build future dates
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(preds), freq="D")
        if preds:
            ax.plot(future_dates, preds, linestyle="--", marker="o", color="#f03b20", label="Predicted Close")

        ax.set_title(f"{ticker.upper()} - Historical Close and {len(preds)}-day Prediction")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        traceback.print_exc()
        return b""


def detect_intent(text: str) -> str:
    t = text.lower()
    # simple rule-based detection
    if any(k in t for k in ("predict", "forecast", "next", "7 day", "7-day", "next 7")):
        return "predict"
    if any(k in t for k in ("show", "display", "data", "history", "historical")):
        return "show"
    if any(k in t for k in ("what is", "explain", "rsi", "moving average", "ma", "indicator")):
        return "explain"
    # fallback: if contains a known ticker pattern (all letters, 1-5 chars)
    words = [w.strip(".,!?()\n\r") for w in t.split()]
    for w in words:
        if 1 <= len(w) <= 5 and w.isalpha():
            return "show"
    return "chat"


def call_ollama_explain(question: str) -> str:
    if not ollama_available():
        return "Ollama not running — cannot fetch LLM explanation."
    try:
        payload = {"model": "llama3.1", "messages": [{"role": "user", "content": question}], "stream": False}
        r = requests.post("http://localhost:11434/api/chat", json=payload, timeout=15)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "No response from LLM.")
    except Exception:
        traceback.print_exc()
        return "Ollama request failed."


def get_response(text: str) -> Dict[str, Any]:
    intent = detect_intent(text)
    resp: Dict[str, Any] = {"intent": intent}

    if intent == "predict":
        # extract ticker (naive): find first 1-5 letter uppercase-like word
        words = [w.strip(".,?!") for w in text.split()]
        ticker = None
        for w in words:
            if 1 <= len(w) <= 5 and w.isalpha():
                ticker = w.upper()
                break
        if not ticker:
            resp["error"] = "No ticker found in the request. Use e.g. 'Predict AAPL next 7 days'."
            return resp

        hist = query_database(ticker, limit=90)
        preds = get_prediction(ticker, days=7)
        img = None
        if preds:
            img = generate_prediction_graph(ticker, hist, preds)

        resp.update({"ticker": ticker, "history": hist, "predictions": preds, "image": img})
        return resp

    if intent == "show":
        # find ticker
        words = [w.strip(".,?!") for w in text.split()]
        ticker = None
        for w in words:
            if 1 <= len(w) <= 5 and w.isalpha():
                ticker = w.upper()
                break
        if not ticker:
            resp["error"] = "No ticker found. Try: 'Show AAPL data'."
            return resp
        hist = query_database(ticker, limit=120)
        resp.update({"ticker": ticker, "history": hist})
        return resp

    if intent == "explain":
        # attempt LLM explain of the question
        answer = call_ollama_explain(text)
        resp.update({"explanation": answer})
        return resp

    # default chat: try LLM then fallback
    if ollama_available():
        answer = call_ollama_explain(text)
        resp.update({"answer": answer})
        return resp

    resp.update({"answer": "Demo mode: I can explain indicators (type 'What is RSI?'), show data (e.g., 'Show AAPL data'), or predict (e.g., 'Predict AAPL next 7 days')."})
    return resp


def streamlit_app():
    st.set_page_config(page_title="AI Prediction Chatbot", layout="wide")
    st.title("🤖 AI Prediction Chatbot")
    st.write("Ask for predictions, show historical data, or request explanations. Works with Ollama if available.")

    query = st.text_input("Enter your question", value="Predict AAPL next 7 days")
    if st.button("Submit"):
        with st.spinner("Processing..."):
            result = get_response(query)

        if "error" in result:
            st.error(result["error"]) 
            return

        if result["intent"] == "predict":
            st.subheader(f"Predictions for {result.get('ticker')}")
            preds = result.get("predictions", [])
            if not preds:
                st.warning("No model available or prediction failed. Make sure model is trained and saved to `models/gbt_forecaster`.")
            else:
                st.write(pd.DataFrame({"day": list(range(1, len(preds)+1)), "predicted_close": preds}))
                img = result.get("image")
                if img:
                    st.image(img, use_column_width=True)
            hist = result.get("history")
            if isinstance(hist, pd.DataFrame) and not hist.empty:
                st.markdown("**Recent historical data**")
                st.dataframe(hist.tail(20))

        elif result["intent"] == "show":
            st.subheader(f"Historical data for {result.get('ticker')}")
            hist = result.get("history")
            if isinstance(hist, pd.DataFrame) and not hist.empty:
                st.dataframe(hist)
            else:
                st.warning("No historical data found for this ticker.")

        elif result["intent"] == "explain":
            st.subheader("Explanation")
            st.write(result.get("explanation"))

        else:
            st.subheader("Chatbot reply")
            st.write(result.get("answer"))


if __name__ == "__main__":
    streamlit_app()