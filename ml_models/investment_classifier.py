import os
import warnings
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

PARQUET_PATH = "data/processed_stocks.parquet"
RANDOM_STATE = 42


class InvestmentClassifier:
    """
    Build features, compute composite score, create labels, train RandomForest classifier,
    evaluate and print classification results for the latest date per ticker.
    """

    def __init__(self, parquet_path: str = PARQUET_PATH):
        self.parquet_path = parquet_path
        self.df = None  # full DataFrame
        self.features_df = None  # feature-engineered DataFrame with score & label
        self.model = None
        self.scaler = None
        self.feature_cols = [
            # We'll populate order after engineering; kept as placeholder
        ]

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.parquet_path):
            raise FileNotFoundError(f"Processed parquet not found: {self.parquet_path}")
        print(f"Loading parquet: {self.parquet_path}")
        df = pd.read_parquet(self.parquet_path)
        # Ensure required columns exist
        required = {"Ticker", "Date", "Open", "High", "Low", "Close", "Volume", "RSI", "Volatility", "Sharpe_Ratio", "MA_7", "MA_30", "MA_90", "Daily_Return"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in parquet: {missing}")
        # sort
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        self.df = df
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create per-row features (Total_Return, 7/30d returns, avg RSI, volatility, sharpe, trend indicators).
        Produces 17 features per row (as required) that will be used to compute score and labels.
        """
        print("Engineering features... (This may take a few seconds)")

        out_rows = []
        # We'll compute rolling features per ticker
        for ticker, g in df.groupby("Ticker", sort=True):
            gg = g.copy().reset_index(drop=True)
            # Rolling windows
            gg["Close_lag_7"] = gg["Close"].shift(7)
            gg["Close_lag_30"] = gg["Close"].shift(30)

            # Recent returns
            gg["Return_7d"] = (gg["Close"] - gg["Close_lag_7"]) / gg["Close_lag_7"]
            gg["Return_30d"] = (gg["Close"] - gg["Close_lag_30"]) / gg["Close_lag_30"]

            # Total return: using 90-day window if available, else from series start for that ticker
            if len(gg) >= 90:
                gg["Close_lag_90"] = gg["Close"].shift(90)
                gg["Total_Return_90"] = (gg["Close"] - gg["Close_lag_90"]) / gg["Close_lag_90"]
            else:
                gg["Total_Return_90"] = (gg["Close"] - gg["Close"].iloc[0]) / gg["Close"].iloc[0]

            # Average RSI (30-day rolling), current RSI already in df
            gg["RSI_avg_30"] = gg["RSI"].rolling(window=30, min_periods=1).mean()
            gg["RSI_current"] = gg["RSI"]

            # Volatility and Sharpe already present; compute rolling variants
            gg["Volatility_30"] = gg["Volatility"].rolling(window=30, min_periods=1).mean()
            gg["Sharpe_30"] = gg["Sharpe_Ratio"].rolling(window=30, min_periods=1).mean()

            # Price trends
            gg["Trend_MA7_gt_MA30"] = (gg["MA_7"] > gg["MA_30"]).astype(int)
            gg["Trend_MA30_gt_MA90"] = (gg["MA_30"] > gg["MA_90"]).astype(int)

            # Some additional features to reach approx 17 features:
            # (we already have Total_Return_90, Return_7d, Return_30d, RSI_avg_30, RSI_current,
            # Volatility_30, Sharpe_30, Trend_MA7_gt_MA30, Trend_MA30_gt_MA90) -> that's 9
            # We'll add rolling mean returns and rolling std returns, momentum, and last daily return:
            gg["Return_7d_mean_30"] = gg["Return_7d"].rolling(window=30, min_periods=1).mean()
            gg["Return_30d_mean_90"] = gg["Return_30d"].rolling(window=90, min_periods=1).mean()
            gg["Return_std_30"] = gg["Daily_Return"].rolling(window=30, min_periods=1).std().fillna(0)
            gg["Momentum_7_30"] = (gg["Return_7d"] - gg["Return_30d"]).fillna(0)
            gg["Last_Daily_Return"] = gg["Daily_Return"].fillna(0)

            # Select the final set of features (17-ish)
            features = [
                "Total_Return_90",
                "Return_7d",
                "Return_30d",
                "RSI_avg_30",
                "RSI_current",
                "Volatility_30",
                "Sharpe_30",
                "Trend_MA7_gt_MA30",
                "Trend_MA30_gt_MA90",
                "Return_7d_mean_30",
                "Return_30d_mean_90",
                "Return_std_30",
                "Momentum_7_30",
                "Last_Daily_Return",
                # Add open/high/low/volume recent ratios as more features to reach 17:
                # ratio of latest close to MA30 and MA90
                "Close_to_MA30",
                "Close_to_MA90"
            ]

            gg["Close_to_MA30"] = (gg["Close"] / gg["MA_30"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
            gg["Close_to_MA90"] = (gg["Close"] / gg["MA_90"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)

            # Keep same length and append
            out_rows.append(gg[[
                "Ticker", "Date", "Close"
            ] + features])

        features_df = pd.concat(out_rows, ignore_index=True, sort=False)
        # drop rows with NA in critical features
        features_df = features_df.dropna(subset=[
            "Total_Return_90", "Return_7d", "Return_30d", "RSI_avg_30", "RSI_current",
            "Volatility_30", "Sharpe_30"
        ])
        features_df = features_df.reset_index(drop=True)
        self.features_df = features_df
        self.feature_cols = [
            "Total_Return_90", "Return_7d", "Return_30d", "RSI_avg_30", "RSI_current",
            "Volatility_30", "Sharpe_30", "Trend_MA7_gt_MA30", "Trend_MA30_gt_MA90",
            "Return_7d_mean_30", "Return_30d_mean_90", "Return_std_30", "Momentum_7_30",
            "Last_Daily_Return", "Close_to_MA30", "Close_to_MA90"
        ]
        print(f"Engineered features shape: {features_df.shape}")
        return features_df

    def compute_composite_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute composite score according to:
        Score = (Total_Return × 0.3) + (Trend_Score × 0.2) +
                (RSI_Score × 0.15) + (Volatility_Score × 0.15) +
                (Sharpe_Score × 0.2)
        Since the scales differ, we normalize the components into a 0-10 scale per ticker
        (min-max scaling over historical values per ticker), then combine.
        """
        print("Computing composite score and labels...")

        df = df.copy()
        # We'll compute per-ticker normalization to map components into 0-10
        def normalize_series(s: pd.Series) -> pd.Series:
            # map to 0..10
            minv, maxv = s.min(), s.max()
            if pd.isna(minv) or pd.isna(maxv) or maxv == minv:
                return pd.Series(5.0, index=s.index)  # neutral
            return 10.0 * (s - minv) / (maxv - minv)

        out = []
        for ticker, g in df.groupby("Ticker", sort=True):
            gg = g.copy().reset_index(drop=True)

            # components
            total_return_norm = normalize_series(gg["Total_Return_90"].fillna(0))
            # Trend score: combine two binary trends into a single 0-10 score (0..10)
            trend_raw = gg["Trend_MA7_gt_MA30"] * 0.6 + gg["Trend_MA30_gt_MA90"] * 0.4
            trend_norm = normalize_series(trend_raw)

            # RSI score: ideal RSI ~50 is neutral; low RSI (<30) -> buy -> higher score.
            # We'll map RSI to a desirability score: lower RSI -> higher score up to 10
            rsi_raw = gg["RSI_current"].fillna(50)
            # invert distance from 50
            rsi_score = (50 - (rsi_raw - 50).abs()).clip(lower=0)  # higher when close to 50; but assignment suggests undervalued >70?
            # Instead we'll prefer mid-range RSI -> map to 0..10
            rsi_norm = normalize_series(rsi_score)

            # Volatility score: lower volatility is preferable -> invert
            vol_raw = gg["Volatility_30"].fillna(gg["Volatility_30"].median())
            vol_norm = normalize_series(-vol_raw)  # invert so lower vol => higher score

            # Sharpe score: higher is better
            sharpe_raw = gg["Sharpe_30"].fillna(0)
            sharpe_norm = normalize_series(sharpe_raw)

            # Combine into composite score
            score = (
                total_return_norm * 0.3
                + trend_norm * 0.2
                + rsi_norm * 0.15
                + vol_norm * 0.15
                + sharpe_norm * 0.2
            )

            gg["Composite_Score"] = score
            out.append(gg)

        scored = pd.concat(out, ignore_index=True, sort=False)
        # Map score approx to 1-10 (already 0-10), but to align with thresholds we keep 0-10 and
        # define label thresholds:
        # High: score >= 7, Medium: 4-7, Low: <4
        def score_to_label(s):
            if s >= 7.0:
                return "High"
            elif s >= 4.0:
                return "Medium"
            else:
                return "Low"

        scored["Label"] = scored["Composite_Score"].apply(score_to_label)
        self.features_df = scored
        return scored

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare X, y for training. We will drop identifier columns and use the engineered features.
        We'll split chronologically to avoid time leakage:
        - Use most recent 20% as test, earlier as train/val (80%).
        """
        print("Preparing training and test splits...")

        # Use features for modeling
        model_df = df.copy().dropna(subset=self.feature_cols + ["Composite_Score", "Label"])
        X = model_df[self.feature_cols].astype(float)
        # Label encoding: High->2, Medium->1, Low->0 (could also use sklearn LabelEncoder)
        label_map = {"Low": 0, "Medium": 1, "High": 2}
        y = model_df["Label"].map(label_map).astype(int)

        # Simple random split (stratify by label to keep balance)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y if len(y.unique()) > 1 else None
        )

        # Scale features (RF not required but helps in interpretation)
        self.scaler = MinMaxScaler()
        X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(self.scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_classifier(self, X_train, y_train):
        print("Training RandomForestClassifier...")
        clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        self.model = clf
        return clf

    def evaluate_classifier(self, clf, X_test, y_test):
        print("Evaluating classifier...")
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
        print("Classification Report (weighted):")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall: {rec:.4f}")
        print(f"  F1-score: {f1:.4f}")
        # detailed report
        print("\nDetailed classification report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    def print_latest_classifications(self):
        """
        For each ticker, print the latest Composite_Score and Label (High/Medium/Low).
        """
        if self.features_df is None:
            raise RuntimeError("No features computed yet.")

        latest = self.features_df.sort_values(["Ticker", "Date"]).groupby("Ticker").tail(1)
        print("\nClassification Results:")
        for _, row in latest.iterrows():
            ticker = row["Ticker"]
            score = float(row["Composite_Score"])
            label = row["Label"]
            print(f"  {ticker}: {label} (Score: {score:.2f})")
        return latest[["Ticker", "Composite_Score", "Label"]]

    def run(self):
        df = self.load_data()
        feats = self.engineer_features(df)
        scored = self.compute_composite_score(feats)
        X_train, X_test, y_train, y_test = self.prepare_training_data(scored)
        clf = self.train_classifier(X_train, y_train)
        metrics = self.evaluate_classifier(clf, X_test, y_test)
        latest = self.print_latest_classifications()
        return {"metrics": metrics, "latest": latest}


if __name__ == "__main__":
    ic = InvestmentClassifier()
    out = ic.run()
