"""
BiLSTM text-sentiment model comparison for TSLA next-day close forecasting.

A single, self-contained pipeline that trains and compares, under one identical
protocol, four forecasting variants plus a naive baseline:

    Persistence                 yhat_t = y_{t-1}                 (baseline)
    BiLSTM                      price only            -> (N, 60, 1)
    BiLSTM + BoW                price + BoW sentiment -> (N, 60, 2)
    BiLSTM + gemini-2.0-flash   price + Gemini scores -> (N, 60, 4)  [sentiment, risk, growth]
    BiLSTM + FinBERT            price + FinBERT scores-> (N, 60, 4)  [positive, neutral, negative]

All model arms share one architecture and one training budget, so differences in
the results table are attributable to the feature set, not the model or the
schedule. Every run produces the full comparison table, a comparison plot, and a
README, plus directional accuracy and a pipeline-integrity check.

Methodology (held identical across arms):
  - Train windows: target date <= TRAIN_END; test: TEST_START <= target <= TEST_END.
  - Per-column MinMax scalers are fit ONLY on rows with Date <= TRAIN_END.
  - Each MD&A document is aligned to trading days by its true SEC filing_date
    (from manifest.csv) via pd.merge_asof(direction="backward"), so a disclosure
    is only visible on the first trading day strictly after it was filed.

Text scores:
  - BoW     : recomputed locally from data/mda_extracted/*.txt.
  - FinBERT : local ProsusAI/finbert (no API key); auto-generated if missing.
  - Gemini  : Google Gemini (temperature=0); key from GEMINI_API_KEY env var.

Usage:
    python src/corrected_ablation.py                  # uses existing score CSVs
    python src/corrected_ablation.py --rescore        # regenerate FinBERT (local) + Gemini (needs GEMINI_API_KEY)
    python src/corrected_ablation.py --rebuild-manifest
"""

from __future__ import annotations

import argparse
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from gemini_api import build_client, get_api_key, score_financial_text, summarize_feature_health

# ---------------------------------------------------------------------------
# Constants (single source of truth for the protocol)
# ---------------------------------------------------------------------------
TRAIN_END = "2024-08-31"   # train windows: target date <= TRAIN_END (data starts 2015)
TEST_START = "2024-09-01"
TEST_END = None            # None -> last available date in Tesla.csv (~2025-09). Single switch.
LOOKBACK, HORIZON = 60, 1
SEED = 42

LEARNING_RATE = 1e-3
EPOCHS = 60
BATCH_SIZE = 32

ROOT = Path(__file__).resolve().parents[1]
PRICE_CSV = ROOT / "tesla_stock_price" / "Tesla.csv"
MDA_DIR = ROOT / "data" / "mda_extracted"
FINBERT_CSV = ROOT / "data" / "tesla_sentiment_features_finbert.csv"
FINBERT_MODEL = "ProsusAI/finbert"
GEMINI_CSV = ROOT / "data" / "tesla_sentiment_features_gemini.csv"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_THROTTLE_SECONDS = float(os.getenv("GEMINI_THROTTLE_SECONDS", "4"))
MANIFEST_CSV = ROOT / "tesla_financial_reports" / "manifest.csv"

RESULTS_DIR = ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "corrected_results.csv"
PLOT_PNG = RESULTS_DIR / "comparison_corrected.png"
README_MD = ROOT / "README.md"

# Arm names + their feature columns. All text arms keep the (N, 60, 4) input shape
# (Close + 3 text dimensions). The Gemini arm is named after the model version.
GEMINI_ARM = f"BiLSTM+{GEMINI_MODEL}"
FINBERT_ARM = "BiLSTM+FinBERT"
GEMINI_COLS = ["sentiment", "risk", "growth"]
FINBERT_COLS = ["positive", "neutral", "negative"]

# Display order for the comparison table / plot.
MODEL_ARMS = ["BiLSTM", "BiLSTM+BoW", GEMINI_ARM, FINBERT_ARM]
COL_ORDER = ["Persistence"] + MODEL_ARMS

TRAIN_END_TS = pd.Timestamp(TRAIN_END)
TEST_START_TS = pd.Timestamp(TEST_START)


def set_seeds() -> None:
    """Deterministic-as-possible seeding. NOTE: GPU runs are not bit-deterministic."""
    random.seed(SEED)
    np.random.seed(SEED)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)


# ---------------------------------------------------------------------------
# Bag-of-Words sentiment (simplified Loughran-McDonald style lexicon)
# ---------------------------------------------------------------------------
FIN_POSITIVE = ["growth", "expansion", "profit", "success", "increase", "strong", "achieved", "delivered"]
FIN_NEGATIVE = ["risk", "uncertainty", "loss", "decline", "challenge", "lawsuit", "shortfall", "pessimistic"]


def get_traditional_sentiment(text):
    """Word-count based sentiment score in [-1, 1]; 0 when no lexicon words appear."""
    text = text.lower()
    words = re.findall(r"\w+", text)
    pos_count = sum(1 for word in words if word in FIN_POSITIVE)
    neg_count = sum(1 for word in words if word in FIN_NEGATIVE)
    if (pos_count + neg_count) == 0:
        return 0
    return (pos_count - neg_count) / (pos_count + neg_count)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_prices() -> pd.DataFrame:
    """Robustly load Tesla.csv. Handles a possible multi-row header (extra
    'Ticker' line) by coercing Date/Close and dropping non-data rows."""
    df = pd.read_csv(PRICE_CSV)
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError(f"Unexpected columns in {PRICE_CSV}: {list(df.columns)}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    assert pd.api.types.is_datetime64_any_dtype(df["Date"]), "Date did not parse to datetime"
    assert pd.api.types.is_float_dtype(df["Close"]), "Close is not float"
    return df[["Date", "Close"]]


def _period_end_from_filename(fn: str) -> pd.Timestamp:
    m = re.search(r"(\d{8})", fn)
    if not m:
        raise ValueError(f"No 8-digit date in filename: {fn}")
    d = m.group(1)
    return pd.Timestamp(f"{d[:4]}-{d[4:6]}-{d[6:]}")


def list_mda_files() -> List[str]:
    return sorted(f for f in os.listdir(MDA_DIR) if f.endswith(".txt"))


def compute_bow_features() -> pd.DataFrame:
    """Recompute BoW sentiment from data/mda_extracted/*.txt (the available docs)."""
    rows = []
    for fn in list_mda_files():
        content = (MDA_DIR / fn).read_text(encoding="utf-8")
        rows.append(
            {
                "filename": fn,
                "period_end": _period_end_from_filename(fn),
                "trad_sentiment": get_traditional_sentiment(content),
            }
        )
    return pd.DataFrame(rows)


def load_score_csv(path: Path, cols: List[str], name: str) -> pd.DataFrame:
    """Load a text-score CSV and restrict to the documents that exist in
    data/mda_extracted/, so every arm uses the SAME filing set (comparability).
    Returns columns: filename, period_end, <cols>."""
    available = set(list_mda_files())
    df = pd.read_csv(path)
    df = df[df["filename"].isin(available)].copy()
    df["period_end"] = df["filename"].map(_period_end_from_filename)
    df = df[["filename", "period_end"] + cols].reset_index(drop=True)
    if float(df[cols].abs().to_numpy().sum()) == 0.0:
        print(f"[WARN] All {name} scores are 0.0 -> that arm is degenerate (≈ price-only).")
    return df


# ---------------------------------------------------------------------------
# Text scorers
# ---------------------------------------------------------------------------
def rescore_finbert() -> pd.DataFrame:
    """Local FinBERT (ProsusAI/finbert) scoring. No API key required.

    Long MD&A text is split into <=512-token chunks (BERT length limit); the
    per-chunk softmax probabilities are averaged to a document-level score.
    sentiment_score = positive_probability - negative_probability.
    Writes data/tesla_sentiment_features_finbert.csv.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Map output indices -> label names robustly (don't hardcode FinBERT's order).
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    max_length = 512
    chunk_size = max_length - 2  # leave room for [CLS] and [SEP]

    def score_text(text: str) -> Tuple[float, float, float]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            ids = [tokenizer.unk_token_id]
        chunks = [ids[i : i + chunk_size] for i in range(0, len(ids), chunk_size)]
        probs_accum = None
        with torch.no_grad():
            for ch in chunks:
                input_ids = torch.tensor(
                    [[tokenizer.cls_token_id] + ch + [tokenizer.sep_token_id]], device=device
                )
                attn = torch.ones_like(input_ids)
                logits = model(input_ids=input_ids, attention_mask=attn).logits
                probs = torch.softmax(logits, dim=-1)[0]
                probs_accum = probs if probs_accum is None else probs_accum + probs
        avg = (probs_accum / len(chunks)).detach().cpu().numpy()
        label_probs = {id2label[i]: float(avg[i]) for i in range(len(avg))}
        return (
            label_probs.get("positive", 0.0),
            label_probs.get("neutral", 0.0),
            label_probs.get("negative", 0.0),
        )

    rows = []
    for fn in list_mda_files():
        content = (MDA_DIR / fn).read_text(encoding="utf-8")
        pos, neu, neg = score_text(content)
        rows.append(
            {
                "positive": pos,
                "neutral": neu,
                "negative": neg,
                "sentiment_score": pos - neg,
                "date": _period_end_from_filename(fn).date().isoformat(),
                "filename": fn,
            }
        )
    df = pd.DataFrame(rows)
    FINBERT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FINBERT_CSV, index=False)
    print(f"[finbert] wrote {len(df)} document-level scores (model={FINBERT_MODEL}) -> {FINBERT_CSV}")
    return df


def rescore_gemini(api_key: str | None = None) -> pd.DataFrame:
    """Score MD&A text with Google Gemini using a financial-analyst prompt
    (sentiment / risk / growth), pinned to temperature=0 with JSON output for
    determinism. Reads the API key from GEMINI_API_KEY (never hardcoded).
    Writes data/tesla_sentiment_features_gemini.csv.
    """
    import time

    client = build_client(api_key=api_key)

    def _is_hard_zero_quota(msg: str) -> bool:
        # "limit: 0" means the project was never granted quota for this model
        # (billing/plan issue) -> retrying is pointless, fail fast with guidance.
        return "limit: 0" in msg

    def _score_with_retry(content: str, max_retries: int = 3) -> dict:
        last_err = ""
        for attempt in range(max_retries + 1):
            try:
                return score_financial_text(
                    client=client, text=content, model=GEMINI_MODEL, temperature=0.0
                )
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
                if _is_hard_zero_quota(last_err):
                    raise RuntimeError(
                        f"Gemini quota for model '{GEMINI_MODEL}' is 0 (free-tier not granted). "
                        "Enable billing on the API key's project, or set a different model via "
                        "the GEMINI_MODEL env var (e.g. gemini-2.5-flash). Original error: "
                        + last_err
                    ) from e
                if "429" in last_err and attempt < max_retries:
                    wait_s = min(60.0, GEMINI_THROTTLE_SECONDS * (2 ** attempt) + 5.0)
                    print(f"[gemini] 429 rate-limited; retry {attempt + 1}/{max_retries} in {wait_s:.0f}s ...")
                    time.sleep(wait_s)
                    continue
                raise
        raise RuntimeError(last_err)

    rows = []
    failed_calls = 0
    failure_examples: list[str] = []
    for fn in list_mda_files():
        content = (MDA_DIR / fn).read_text(encoding="utf-8")
        try:
            scores = _score_with_retry(content)
        except Exception as e:  # noqa: BLE001
            print(f"Error during Gemini API call: {e}")
            failed_calls += 1
            if len(failure_examples) < 3:
                failure_examples.append(str(e))
            scores = {"sentiment": 0.0, "risk": 0.0, "growth": 0.0}
        time.sleep(GEMINI_THROTTLE_SECONDS)  # spread requests to respect per-minute limits
        rows.append(
            {
                "sentiment": scores.get("sentiment", 0),
                "risk": scores.get("risk", 0),
                "growth": scores.get("growth", 0),
                "date": _period_end_from_filename(fn).date().isoformat(),
                "filename": fn,
            }
        )
    if rows and failed_calls == len(rows):
        example = failure_examples[0] if failure_examples else "unknown error"
        raise RuntimeError(
            "All Gemini API calls failed; refusing to write a degenerate all-zero Gemini feature file. "
            f"First error: {example}"
        )
    df = pd.DataFrame(rows)
    GEMINI_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(GEMINI_CSV, index=False)
    print(f"[gemini] wrote {len(df)} deterministic (temperature=0) scores (model={GEMINI_MODEL}) -> {GEMINI_CSV}")
    if failed_calls:
        print(f"[gemini] partial failures: {failed_calls}/{len(rows)} requests failed and were filled with 0.0")
    report_gemini_feature_health(df, source="rescore")
    return df


# ---------------------------------------------------------------------------
# Filing-date mapping + trading-day alignment
# ---------------------------------------------------------------------------
def load_manifest(rebuild: bool) -> pd.DataFrame:
    if rebuild or not MANIFEST_CSV.exists():
        if not MANIFEST_CSV.exists():
            print(f"[manifest] {MANIFEST_CSV} not found -> regenerating from SEC EDGAR ...")
        from build_manifest import build_manifest

        build_manifest(MANIFEST_CSV)
    man = pd.read_csv(MANIFEST_CSV)
    man = man[man["form"].isin(["10-K", "10-Q"])].copy()  # exclude /A amendments
    man["filing_date"] = pd.to_datetime(man["filing_date"], errors="coerce")
    man["report_date"] = pd.to_datetime(man["report_date"], errors="coerce")
    man = man.dropna(subset=["filing_date", "report_date"])
    # If multiple rows share a report_date, keep the earliest filing_date.
    man = (
        man.sort_values("filing_date")
        .groupby("report_date", as_index=False)
        .first()[["report_date", "filing_date", "form"]]
    )
    return man


def map_filing_dates(feat: pd.DataFrame, man: pd.DataFrame) -> pd.DataFrame:
    merged = feat.merge(
        man[["report_date", "filing_date"]],
        left_on="period_end",
        right_on="report_date",
        how="left",
    )
    unmatched = merged[merged["filing_date"].isna()]
    if len(unmatched) > 0:
        print("[ERROR] Unmatched documents (period_end not found in manifest.report_date):")
        print(unmatched[["filename", "period_end"]].to_string(index=False))
    assert len(unmatched) == 0, "Unmatched documents -> cannot map filing dates. STOP."
    return merged.drop(columns=["report_date"])


def align_to_trading_days(
    prices: pd.DataFrame, feat: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, Dict[str, int], List[pd.Timestamp]]:
    """merge_asof(direction='backward') forward-fills from the last filing that
    was already public (filing_date < trading day). Returns the aligned frame,
    a dict of {exact_join, asof} match counts, and the list of filings whose
    filing_date fell on a non-trading day (assigned to the next trading day)."""
    feat = feat.sort_values("filing_date").reset_index(drop=True)
    prices = prices.sort_values("Date").reset_index(drop=True)

    df = pd.merge_asof(
        prices,
        feat[["filing_date"] + feature_cols],
        left_on="Date",
        right_on="filing_date",
        direction="backward",
        allow_exact_matches=False,  # filings often drop after close -> usable next trading day
    )
    # Pre-first-filing NaNs -> 0.0 (neutral); documented in README.
    for c in feature_cols:
        df[c] = df[c].fillna(0.0)

    trading_days = set(prices["Date"])
    exact_join = int(feat["filing_date"].isin(trading_days).sum())
    matched_asof = pd.merge_asof(
        prices[["Date"]],
        feat[["filing_date"]].assign(_fd=feat["filing_date"]),
        left_on="Date",
        right_on="filing_date",
        direction="backward",
        allow_exact_matches=False,
    )["_fd"].dropna().unique()
    asof = int(len(matched_asof))
    recovered = sorted(set(feat["filing_date"]) - trading_days)
    return df, {"exact_join": exact_join, "asof": asof}, recovered


# ---------------------------------------------------------------------------
# Scaling (fit on train only) + window building
# ---------------------------------------------------------------------------
def fit_scalers(df: pd.DataFrame, feature_cols: List[str]):
    """Per-column MinMax scalers, fit ONLY on rows with Date <= TRAIN_END."""
    from sklearn.preprocessing import MinMaxScaler

    train_mask = df["Date"] <= TRAIN_END_TS
    scalers: Dict[str, MinMaxScaler] = {}
    scaled = df[["Date"]].copy()
    for col in ["Close"] + feature_cols:
        sc = MinMaxScaler()
        sc.fit(df.loc[train_mask, [col]])
        scaled[col] = sc.transform(df[[col]])
        scalers[col] = sc
    return scaled, scalers, int(train_mask.sum())


def build_windows(scaled: pd.DataFrame, arm_cols: List[str]):
    """Sliding windows over the full series. X = lookback of arm cols, y = scaled
    Close at target day. Returns X, y, and the target Timestamp per window."""
    cols = ["Close"] + [c for c in arm_cols if c != "Close"]
    values = scaled[cols].to_numpy(dtype=np.float32)
    close_idx = cols.index("Close")
    dates = scaled["Date"].to_numpy()

    X, y, tdates = [], [], []
    for i in range(len(values) - LOOKBACK):
        target = i + LOOKBACK  # HORIZON == 1
        X.append(values[i:target, :])
        y.append(values[target, close_idx])
        tdates.append(dates[target])
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    tdates = pd.to_datetime(np.asarray(tdates))
    return X, y, tdates


def split_by_target_date(tdates: pd.DatetimeIndex, test_end_ts: pd.Timestamp):
    train_idx = np.where(tdates <= TRAIN_END_TS)[0]
    test_idx = np.where((tdates >= TEST_START_TS) & (tdates <= test_end_ts))[0]
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Model + shared training config (identical for every arm)
# ---------------------------------------------------------------------------
def build_bilstm(input_shape, lr: float = LEARNING_RATE):
    from tensorflow.keras import layers, models, optimizers

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Dropout(0.2),
            layers.Bidirectional(layers.LSTM(32, return_sequences=False)),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return model


def train_arm(X_train, y_train):
    from tensorflow.keras import callbacks

    model = build_bilstm(input_shape=(X_train.shape[1], X_train.shape[2]))
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
    )
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )
    epochs_trained = len(history.history["loss"])
    return model, epochs_trained


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(y - yhat)))


def mape(y, yhat) -> float:
    return float(np.mean(np.abs((y - yhat) / y)) * 100.0)


def directional_accuracy(y_true, y_pred, y_prev) -> Tuple[float, int]:
    """DA = mean( sign(yhat - y_{t-1}) == sign(y - y_{t-1}) ), excluding flat days."""
    true_dir = np.sign(y_true - y_prev)
    pred_dir = np.sign(y_pred - y_prev)
    mask = true_dir != 0  # exclude days where y_t == y_{t-1}
    if mask.sum() == 0:
        return float("nan"), 0
    return float(np.mean(pred_dir[mask] == true_dir[mask])), int(mask.sum())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def ensure_scores(rescore: bool, gemini_api_key: str | None = None) -> None:
    """Make sure both text-score CSVs are available before the comparison.
    FinBERT is local/free (auto-generated). Gemini needs GEMINI_API_KEY."""
    if rescore or not FINBERT_CSV.exists():
        reason = "--rescore" if rescore else "file missing"
        print(f"[finbert] scoring ({reason}) ...")
        rescore_finbert()

    if rescore:
        # Resolve the key via gemini_api.get_api_key so all sources are honored:
        # --gemini-api-key > GEMINI_API_KEY env var > project .env file.
        effective_key = get_api_key(gemini_api_key)
        print("[gemini] rescoring enabled -> requesting fresh Gemini features ...")
        rescore_gemini(api_key=effective_key)
    if not GEMINI_CSV.exists():
        raise RuntimeError(
            f"{GEMINI_CSV} not found. Run with --rescore and GEMINI_API_KEY set "
            "to generate Gemini scores first."
        )


def report_gemini_feature_health(gemini_df: pd.DataFrame, source: str = "loaded CSV") -> None:
    info = summarize_feature_health(gemini_df, GEMINI_COLS)
    print(
        f"[gemini-check] source={source}, rows={info['rows']}, "
        f"all_scores_zero={info['all_scores_zero']}, all_zero_rows={info['all_zero_rows']}"
    )
    print(f"[gemini-check] zero_counts={info['zero_counts']}")
    print(f"[gemini-check] non_zero_counts={info['non_zero_counts']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BiLSTM text-sentiment model comparison for TSLA next-day close."
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Regenerate text scores: FinBERT (local) and Gemini (needs GEMINI_API_KEY).",
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Optional Gemini key override. Recommended: use GEMINI_API_KEY environment variable.",
    )
    parser.add_argument("--rebuild-manifest", action="store_true", help="Force EDGAR manifest regeneration.")
    args = parser.parse_args()

    set_seeds()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load + feature engineering -------------------------------------
    prices = load_prices()
    test_end_ts = pd.Timestamp(TEST_END) if TEST_END else prices["Date"].max()
    prices = prices[prices["Date"] <= test_end_ts].reset_index(drop=True)

    ensure_scores(args.rescore, gemini_api_key=args.gemini_api_key)

    bow = compute_bow_features()
    gemini_df = load_score_csv(GEMINI_CSV, GEMINI_COLS, "gemini")
    report_gemini_feature_health(gemini_df, source=GEMINI_CSV.name)
    finbert_df = load_score_csv(FINBERT_CSV, FINBERT_COLS, "finbert")

    # Merge BoW + Gemini + FinBERT on the shared filing set (inner join).
    feat = (
        bow.merge(gemini_df[["filename"] + GEMINI_COLS], on="filename", how="inner")
        .merge(finbert_df[["filename"] + FINBERT_COLS], on="filename", how="inner")
    )
    n_docs = len(feat)

    man = load_manifest(rebuild=args.rebuild_manifest)
    n_manifest = len(man)
    feat = map_filing_dates(feat, man)

    all_feature_cols = ["trad_sentiment"] + GEMINI_COLS + FINBERT_COLS
    aligned, match_counts, recovered = align_to_trading_days(prices, feat, all_feature_cols)

    # --- Scaling (fit on train only) ------------------------------------
    scaled, scalers, n_scaler_rows = fit_scalers(aligned, all_feature_cols)
    close_scaler = scalers["Close"]

    arms = {
        "BiLSTM": ["Close"],
        "BiLSTM+BoW": ["Close", "trad_sentiment"],
        GEMINI_ARM: ["Close"] + GEMINI_COLS,
        FINBERT_ARM: ["Close"] + FINBERT_COLS,
    }

    # Reference (price-scale) actuals + previous-close, computed once from the
    # full series so the first test day's reference is the last pre-test close.
    closes = aligned["Close"].to_numpy(dtype=np.float64)
    dates_all = aligned["Date"]

    results: Dict[str, Dict[str, float]] = {}
    epochs_log: Dict[str, int] = {}
    preds_for_plot: Dict[str, np.ndarray] = {}
    da_log: Dict[str, Tuple[float, int]] = {}
    test_dates_ref = None
    y_true_ref = None
    y_prev_ref = None
    train_range = test_range = None

    for arm_name, arm_cols in arms.items():
        X, y, tdates = build_windows(scaled, arm_cols)
        assert not np.isnan(X).any() and not np.isnan(y).any(), f"NaN in X/y for {arm_name}"
        train_idx, test_idx = split_by_target_date(tdates, test_end_ts)

        max_train = tdates[train_idx].max()
        min_test = tdates[test_idx].min()
        assert max_train <= TRAIN_END_TS < min_test, "Split ordering violated"

        if train_range is None:
            train_range = (tdates[train_idx].min(), max_train, len(train_idx))
            test_range = (min_test, tdates[test_idx].max(), len(test_idx))

        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[test_idx]

        print(f"\n===== Training arm: {arm_name}  (X_train={X_train.shape}, X_test={X_test.shape}) =====")
        model, ep = train_arm(X_train, y_train)
        epochs_log[arm_name] = ep

        yhat_scaled = model.predict(X_test, verbose=0)
        yhat_price = close_scaler.inverse_transform(yhat_scaled).ravel()

        # Map window target indices back to the full-series row index for prev-close.
        target_rows = test_idx + LOOKBACK  # row index in `aligned`
        y_true = closes[target_rows]
        y_prev = closes[target_rows - 1]  # previous trading day actual close
        test_dates = dates_all.to_numpy()[target_rows]

        if test_dates_ref is None:
            test_dates_ref = pd.to_datetime(test_dates)
            y_true_ref = y_true
            y_prev_ref = y_prev

        results[arm_name] = {
            "RMSE": rmse(y_true, yhat_price),
            "MAE": mae(y_true, yhat_price),
            "MAPE": mape(y_true, yhat_price),
        }
        da_log[arm_name] = directional_accuracy(y_true, yhat_price, y_prev)
        preds_for_plot[arm_name] = yhat_price

    # --- Persistence baseline (yhat_t = y_{t-1}) ------------------------
    persistence_pred = y_prev_ref
    results["Persistence"] = {
        "RMSE": rmse(y_true_ref, persistence_pred),
        "MAE": mae(y_true_ref, persistence_pred),
        "MAPE": mape(y_true_ref, persistence_pred),
    }

    # --- Results table --------------------------------------------------
    table = pd.DataFrame(index=["RMSE", "MAE", "MAPE", "DA"], columns=COL_ORDER, dtype=object)
    for col in COL_ORDER:
        table.loc["RMSE", col] = round(results[col]["RMSE"], 4)
        table.loc["MAE", col] = round(results[col]["MAE"], 4)
        table.loc["MAPE", col] = round(results[col]["MAPE"], 4)
    table.loc["DA", "Persistence"] = "-"  # persistence predicts zero change -> DA undefined
    for col in MODEL_ARMS:
        da_val, _ = da_log[col]
        table.loc["DA", col] = round(da_val, 4)

    table.to_csv(RESULTS_CSV)
    md_table = render_markdown_table(table)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON (price scale; DA in [0,1], coin-flip ref = 0.50)")
    print("=" * 70)
    print(md_table)
    print(
        "\nDA n (non-flat test days) per arm: "
        + ", ".join(f"{k}={da_log[k][1]}" for k in MODEL_ARMS)
    )

    # --- Plot -----------------------------------------------------------
    make_plot(test_dates_ref, y_true_ref, preds_for_plot)

    # --- Pipeline integrity check --------------------------------------
    print(render_integrity_check(
        train_range, test_range, n_scaler_rows, n_docs, n_manifest, match_counts, recovered, da_log
    ))

    # --- README ---------------------------------------------------------
    write_readme(table, epochs_log)
    print(f"\n[done] Wrote: {RESULTS_CSV}\n[done] Wrote: {PLOT_PNG}\n[done] Wrote: {README_MD}")
    print("[done] Per-arm epochs trained: " + ", ".join(f"{k}={v}" for k, v in epochs_log.items()))


def render_markdown_table(table: pd.DataFrame) -> str:
    cols = list(table.columns)
    header = "| Metric | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [header, sep]
    for metric in table.index:
        lines.append("| " + metric + " | " + " | ".join(str(table.loc[metric, c]) for c in cols) + " |")
    return "\n".join(lines)


def _arm_label(arm: str) -> str:
    return {
        "BiLSTM": "BiLSTM (price only)",
        "BiLSTM+BoW": "BiLSTM + BoW",
        GEMINI_ARM: f"BiLSTM + {GEMINI_MODEL}",
        FINBERT_ARM: "BiLSTM + FinBERT",
    }.get(arm, arm)


def make_plot(test_dates, y_true, preds: Dict[str, np.ndarray]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "BiLSTM": "#2980b9",
        "BiLSTM+BoW": "#27ae60",
        GEMINI_ARM: "#e67e22",
        FINBERT_ARM: "#8e44ad",
    }
    plt.figure(figsize=(14, 7))
    plt.plot(test_dates, y_true, label="Actual", color="black", linewidth=2)
    for arm in MODEL_ARMS:
        plt.plot(test_dates, preds[arm], label=_arm_label(arm), color=colors[arm], linewidth=1.6)
    plt.title("TSLA Close: Actual vs BiLSTM model variants (unified protocol)")
    plt.xlabel("Date")
    plt.ylabel("Close (USD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PNG, dpi=150)
    plt.close()


def render_integrity_check(
    train_range, test_range, n_scaler_rows, n_docs, n_manifest, match_counts, recovered, da_log
) -> str:
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("PIPELINE INTEGRITY CHECK")
    lines.append("=" * 70)
    lines.append(f"docs scored & aligned = {n_docs}; manifest 10-K/10-Q rows = {n_manifest}")
    lines.append(f"text sources: {GEMINI_CSV.name}, {FINBERT_CSV.name} (BoW recomputed from MD&A)")
    lines.append(
        f"Train target dates: {pd.Timestamp(train_range[0]).date()} -> "
        f"{pd.Timestamp(train_range[1]).date()}  (n={train_range[2]})"
    )
    lines.append(
        f"Test  target dates: {pd.Timestamp(test_range[0]).date()} -> "
        f"{pd.Timestamp(test_range[1]).date()}  (n={test_range[2]})"
    )
    lines.append(f"[PASS] split ordering: max(train) <= {TRAIN_END} < min(test)")
    lines.append(
        f"[PASS] scaler-fit rows subset of train period: {n_scaler_rows} rows, all Date <= {TRAIN_END}"
    )
    lines.append("[PASS] every matched filing has filing_date < trading day "
                 "(merge_asof backward, allow_exact_matches=False)")
    lines.append(
        f"Filing-date alignment: exact-date matches = {match_counts['exact_join']}; "
        f"merge_asof matches = {match_counts['asof']}; "
        f"{match_counts['asof'] - match_counts['exact_join']} filing(s) on non-trading days "
        "assigned to the next trading day."
    )
    if recovered:
        lines.append("  Filings whose filing_date fell on a non-trading day:")
        for d in recovered:
            lines.append(f"    - {pd.Timestamp(d).date()}")
    lines.append("DA notes: coin-flip reference = 0.50; persistence DA = '-' (predicts zero change).")
    for k in MODEL_ARMS:
        lines.append(f"  {_arm_label(k)}: DA computed over n={da_log[k][1]} non-flat test days.")
    return "\n".join(lines)


def write_readme(table: pd.DataFrame, epochs_log) -> None:
    md_table = render_markdown_table(table)
    epochs_str = ", ".join(f"{k}={v}" for k, v in epochs_log.items())
    content = f"""# BiLSTM Text-Sentiment Model Comparison - TSLA Next-Day Close

## Project overview
A direct, like-for-like comparison of next-day TSLA close forecasts from four
model variants plus a naive persistence baseline. The question: **does adding
disclosure-text sentiment (from Tesla SEC 10-K / 10-Q MD&A sections) to a
price-only BiLSTM improve the forecast, and does the choice of text model matter?**
We compare a price-only BiLSTM, BiLSTM + bag-of-words (BoW) sentiment,
BiLSTM + Google **{GEMINI_MODEL}** scores, and BiLSTM + local **FinBERT**
(`ProsusAI/finbert`) scores.

> This README is auto-generated by `src/corrected_ablation.py`; the results below
> are populated from the actual run (never hand-edited).

## Protocol (identical for every arm)
- **Windows / split.** One set of date constants drives everything:
  `TRAIN_END = 2024-08-31`, `TEST_START = 2024-09-01`, `TEST_END = None`
  (-> last available date in `Tesla.csv`, ~2025-09). Lookback 60, horizon 1.
  Windows are assigned to train/test **by target date**.
- **Scaling.** Per-column `MinMaxScaler`s are fit **only** on rows with
  `Date <= TRAIN_END`, then used to transform the full series. The `Close` scaler
  is reused to inverse-transform predictions back to price scale.
- **Filing-date alignment.** Each MD&A doc's filename date is the fiscal
  **period-end**, not the filing date. We map period-end -> true `filing_date`
  via `tesla_financial_reports/manifest.csv` (from SEC EDGAR) and align to trading
  days with `pd.merge_asof(direction="backward", allow_exact_matches=False)` -- so
  a disclosure is only visible on the first trading day strictly **after** it was
  filed. Pre-first-filing feature values are filled with `0.0` (neutral).
- **Unified training.** Every model arm shares one builder
  (`BiLSTM(64)->Dropout(0.2)->BiLSTM(32)->Dropout(0.2)->Dense(32,relu)->Dense(1)`,
  Adam 1e-3, MSE) and one training config (`epochs=60, batch_size=32,
  validation_split=0.1`, EarlyStopping patience 10 + restore-best,
  ReduceLROnPlateau factor 0.5 / patience 5 / min_lr 1e-5).
- **Same filing set for all text arms** (the docs present in `data/mda_extracted/`),
  for an apples-to-apples comparison.
- Epochs actually trained this run: {epochs_str}.

## Text-sentiment feature sources
- **BoW** -- simplified Loughran-McDonald lexicon, recomputed locally; 1 feature
  (`trad_sentiment`).
- **{GEMINI_MODEL}** -- Google Gemini via the `google-genai` SDK with a
  financial-analyst prompt at `temperature=0` and JSON output; key from the
  `GEMINI_API_KEY` env var (never hardcoded); 3 features `[sentiment, risk, growth]`.
- **FinBERT** -- local `ProsusAI/finbert`; MD&A split into <=512-token chunks,
  per-chunk softmax probabilities averaged to document level; 3 features
  `[positive, neutral, negative]`.

## Results (price scale)
DA = directional accuracy (coin-flip reference 0.50); persistence DA is undefined
(it predicts zero change).

{md_table}

## Limitations
- **Single asset, single split.** One ticker (TSLA), one chronological split; no
  walk-forward cross-validation, so results are high-variance.
- **Level-based metrics.** RMSE/MAE/MAPE on price levels are dominated by trend;
  near a strong uptrend a persistence baseline is very hard to beat. Directional
  accuracy is included as a more decision-relevant view.
- **LLM hindsight.** Gemini is a large general-purpose model whose pretraining may
  overlap the evaluation period, and the prompt explicitly names Tesla, so the
  Gemini arm may reflect pretraining hindsight rather than genuine disclosure
  reading. FinBERT is local/deterministic but its pretraining corpus likewise
  predates parts of the evaluation period.

## Reproduce
```bash
pip install -r requirements.txt
# regenerate text scores (FinBERT local + Gemini), then run the full comparison:
$env:GEMINI_API_KEY = "<your-key>"           # PowerShell; bash: export GEMINI_API_KEY=...
python src/corrected_ablation.py --rescore
# re-run the comparison reusing existing score CSVs (no API calls):
python src/corrected_ablation.py
python src/build_manifest.py                  # (re)build the EDGAR filing manifest
```
Outputs: `results/corrected_results.csv`, `results/comparison_corrected.png`, this README.
"""
    README_MD.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
