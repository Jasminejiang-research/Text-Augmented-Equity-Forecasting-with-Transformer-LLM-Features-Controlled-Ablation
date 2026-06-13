# Text-Augmented Equity Forecasting with Transformer & LLM Features

## Controlled Ablation on TSLA using SEC 10-K / 10-Q MD&A Disclosures

> **Goal:** Test whether SEC disclosure text features improve next-day TSLA close-price forecasting beyond a price-only BiLSTM and a naive persistence baseline, under a leakage-controlled evaluation protocol.

**Tech stack:** Python · TensorFlow/Keras · Hugging Face Transformers · FinBERT · Gemini API · scikit-learn · pandas · SEC EDGAR · Matplotlib

---

## 1. Executive Summary

This project builds an end-to-end **NLP-to-forecasting pipeline** that connects public company disclosures to equity price prediction. It ingests Tesla SEC 10-K / 10-Q filings, extracts MD&A text, scores the text with both traditional and modern NLP methods, and evaluates whether these features improve a deep-learning time-series forecaster.

The central finding is intentionally not framed as “LLMs always improve forecasting.” Under a controlled daily-horizon setting, **no text-augmented BiLSTM variant beat the naive persistence baseline**. Directional accuracy was also statistically indistinguishable from chance. This result is consistent with the idea that daily equity prices are difficult to improve upon using public disclosures once information timing and data leakage are handled correctly.

---

## 2. Research Question

**Can financial disclosure text features extracted from Tesla SEC filings improve next-day TSLA close-price forecasts compared with a price-only recurrent neural network and a naive persistence baseline?**

The project evaluates this question through a controlled ablation:


| Arm                  | Input Features                                                    | Purpose                                                                        |
| -------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Persistence baseline | Previous trading day's close                                      | Strong naive benchmark for daily price-level prediction                        |
| BiLSTM               | Close price only                                                  | Tests whether the neural sequence model adds predictive value over persistence |
| BiLSTM + BoW         | Close price + traditional sentiment score                         | Tests whether simple financial word-count sentiment helps                      |
| BiLSTM + FinBERT     | Close price + FinBERT positive / neutral / negative probabilities | Tests whether local financial-domain transformer features help                 |
| BiLSTM + Gemini      | Close price + Gemini / LLM sentiment, risk, and growth scores     | Tests whether structured LLM financial analysis features help                  |


---

## 3. Data Sources


| Data Source                     | Content                                  | Usage                                                    |
| ------------------------------- | ---------------------------------------- | -------------------------------------------------------- |
| Yahoo Finance / local price CSV | Daily TSLA close prices                  | Target variable and price-history input                  |
| SEC EDGAR                       | Tesla 10-K / 10-Q filings                | Source documents for textual signals                     |
| Extracted MD&A sections         | Management Discussion & Analysis text    | Input for BoW, FinBERT, and Gemini scoring               |
| Filing manifest                 | Filing metadata with SHA-256 audit trail | Maps report period-end dates to true public filing dates |


The project explicitly distinguishes between:

- **period-end date**: the fiscal reporting date in the filename or report;
- **filing date**: the date when the disclosure became public.

This distinction is essential because using the period-end date as the feature-availability date would leak information into the model before the market could have observed it.

---

## 4. Pipeline Overview

```text
SEC EDGAR filings
        │
        ▼
SHA-256 audited filing manifest
        │
        ▼
MD&A extraction
        │
        ├── Traditional BoW sentiment
        ├── Local FinBERT scoring
        └── Gemini structured JSON scoring
        │
        ▼
Filing-date alignment to trading days
        │
        ▼
Train-only scaling and sliding-window construction
        │
        ▼
Four-arm BiLSTM ablation
        │
        ▼
RMSE / MAE / MAPE / Directional Accuracy / Significance check
```

---

## 5. Methodology Summary


| Stage                 | Method                                | Implementation Detail                                                                          | Why It Matters                                                                  |
| --------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Filing ingestion      | SEC EDGAR pipeline                    | Builds a filing manifest with metadata and SHA-256 auditability                                | Makes the disclosure dataset reproducible and traceable                         |
| Text extraction       | MD&A extraction                       | Extracts Management Discussion & Analysis sections from 10-K / 10-Q filings                    | Focuses on forward-looking business discussion rather than full-document noise  |
| Traditional sentiment | Bag-of-words scoring                  | Counts finance-oriented positive and negative terms                                            | Provides a simple interpretable NLP baseline                                    |
| Transformer sentiment | FinBERT                               | Splits long MD&A text into ≤512-token chunks and averages document-level softmax probabilities | Handles long disclosures while respecting BERT sequence limits                  |
| LLM scoring           | Gemini API                            | Uses a financial-analyst prompt with temperature=0 and structured JSON output                  | Produces interpretable `sentiment`, `risk`, and `growth` features               |
| Feature timing        | Filing-date alignment                 | Uses true `filing_date`, not fiscal period-end date                                            | Prevents look-ahead bias                                                        |
| Trading-day alignment | `merge_asof`                          | Carries the latest already-public filing features forward to trading days                      | Recovers weekend / non-trading-day filings without leaking same-day information |
| Scaling               | Train-only MinMaxScaler               | Fits scalers only on rows with dates inside the training period                                | Prevents test-period price-range leakage                                        |
| Forecast model        | BiLSTM                                | Shared architecture and training budget across all model arms                                  | Ensures fair ablation comparison                                                |
| Baseline              | Persistence                           | Predicts tomorrow's close as today's close                                                     | Provides a hard-to-beat benchmark for daily equity price-level forecasting      |
| Evaluation            | RMSE, MAE, MAPE, directional accuracy | Tests both price-level error and direction prediction                                          | Separates numerical fit from trading-relevant signal quality                    |
| Statistical check     | Directional accuracy threshold        | Compares DA against chance-level performance                                                   | Avoids overclaiming weak directional results                                    |


---

## 6. Text Feature Engineering

### 6.1 Bag-of-Words Sentiment

The traditional sentiment score is calculated from domain-specific positive and negative financial keywords:

```text
sentiment = (positive_count - negative_count) / (positive_count + negative_count)
```

If no positive or negative keywords are found, the score is set to zero.

This arm is intentionally simple and interpretable. It serves as a baseline for testing whether even low-complexity disclosure sentiment improves forecasting.

---

### 6.2 FinBERT Features

FinBERT is used as a local financial-domain transformer model. Since MD&A disclosures often exceed BERT's maximum input length, each document is split into token chunks.

For each document:

1. tokenize MD&A text;
2. split into chunks of up to 512 tokens;
3. run FinBERT inference on each chunk;
4. apply softmax to obtain class probabilities;
5. average chunk-level probabilities to document level.

The final FinBERT features are:


| Feature    | Meaning                                        |
| ---------- | ---------------------------------------------- |
| `positive` | Average probability of positive financial tone |
| `neutral`  | Average probability of neutral tone            |
| `negative` | Average probability of negative financial tone |


This creates a fully local, no-API text scoring path.

---

### 6.3 Gemini LLM Features

Gemini is prompted as a financial analyst and asked to return structured JSON scores. In the reported run, the Gemini arm uses `gemini-2.5-flash` with deterministic generation.

The output features are:


| Feature     | Scale   | Interpretation                                  |
| ----------- | ------- | ----------------------------------------------- |
| `sentiment` | -1 to 1 | Overall tone, from pessimistic to optimistic    |
| `risk`      | -1 to 1 | Risk profile, from high-risk to low-risk / safe |
| `growth`    | -1 to 1 | Growth outlook, from declining to strong growth |


The Gemini path uses:

- deterministic scoring with `temperature=0`;
- structured JSON output;
- environment-variable API key handling;
- no hardcoded credentials.

---

## 7. Leakage-Controlled Experimental Design

Financial forecasting is especially vulnerable to subtle leakage. This project therefore enforces the following controls.


| Leakage Risk              | Potential Error                                                          | Control Implemented                                                              |
| ------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| Scaler leakage            | Fitting scalers on the full dataset exposes test-period price ranges     | Scalers are fitted only on training-period rows                                  |
| Look-ahead from filings   | Using fiscal period-end date makes filings visible too early             | Text features are aligned by true SEC filing date                                |
| Same-day filing ambiguity | A filing may be published after market close                             | Exact same-day matches are disallowed; features become usable after filing date  |
| Weekend filing loss       | Exact joins can drop filings published on non-trading days               | `merge_asof` aligns filings to the next eligible trading days                    |
| Text-arm comparability    | Different text arms may use different filing universes                   | All text arms use the same MD&A filing set where available                       |
| Protocol drift            | Different arms may accidentally use different splits or training budgets | All arms share one model builder, one split rule, and one training configuration |
| Silent leakage regression | Future code changes may reintroduce leakage                              | Automated runner self-checks validate scaler fitting, filing alignment, and DA   |
| Baseline omission         | Neural models may appear good without a strong naive benchmark           | Persistence baseline is evaluated alongside all learned models                   |


---

## 8. Model Architecture

All learned model arms use the same BiLSTM architecture:

```text
Input sequence: 60 trading days × feature dimension
        │
        ▼
Bidirectional LSTM, 64 units, return sequences
        │
        ▼
Dropout, 0.2
        │
        ▼
Bidirectional LSTM, 32 units
        │
        ▼
Dropout, 0.2
        │
        ▼
Dense, 32 units, ReLU
        │
        ▼
Dense, 1 unit
```

Training configuration:


| Parameter        | Value                                     |
| ---------------- | ----------------------------------------- |
| Lookback window  | 60 trading days                           |
| Forecast horizon | 1 trading day                             |
| Optimizer        | Adam                                      |
| Learning rate    | 1e-3                                      |
| Loss             | MSE                                       |
| Batch size       | 32                                        |
| Max epochs       | 60                                        |
| Validation split | 10% of training windows                   |
| Early stopping   | Patience 10, restore best weights         |
| LR scheduler     | ReduceLROnPlateau, factor 0.5, patience 5 |


---

## 9. Train-Test Setup


| Split            | Period / Size                             |
| ---------------- | ----------------------------------------- |
| Training windows | 2,372 windows, covering 2015–2024         |
| Test period      | 269 out-of-sample trading days, 2024–2025 |
| Target           | Next-day TSLA close price                 |
| Input sequence   | Previous 60 trading days                  |


The split is based on the **target date** of each sliding window, not on a hardcoded row index. This avoids accidental window leakage around the train-test boundary. The reported run uses `TRAIN_END = 2024-08-31`, `TEST_START = 2024-09-01`, and `TEST_END = None`, meaning the test period extends to the last available date in the local TSLA price file.

All learned arms use the same model architecture and training budget. In the reported run, the number of epochs actually trained was: BiLSTM = 24, BiLSTM + BoW = 46, BiLSTM + Gemini-2.5-flash = 26, and BiLSTM + FinBERT = 33.

---

## 10. Evaluation Metrics


| Metric                 | Definition                                        | Interpretation                                      |
| ---------------------- | ------------------------------------------------- | --------------------------------------------------- |
| RMSE                   | Root mean squared error                           | Penalizes large price-level errors                  |
| MAE                    | Mean absolute error                               | Average absolute dollar error                       |
| MAPE                   | Mean absolute percentage error                    | Scale-normalized price-level error                  |
| Directional Accuracy   | Share of correct up/down predictions              | Measures whether the model predicts price direction |
| Significance threshold | Minimum DA needed to reject chance-level behavior | Helps avoid overclaiming noisy directional results  |


For directional accuracy, the null benchmark is approximately 50%. In the reported test setting with `n = 269`, the directional significance threshold was approximately **56%**. The best learned model in the reported run reached **54.28%**, so directional performance remained statistically indistinguishable from chance.

---

## 11. Model Score Comparison

### Headline Result

The naive persistence baseline outperformed all neural variants on price-level error. No text-augmented variant beat persistence: persistence achieved **RMSE 13.2**, while the learned BiLSTM variants ranged from **RMSE 19.9 to 46.3**. Directional accuracy was also no better than chance, with the best text-augmented result at **54.3%** on `n = 269` out-of-sample test days.

### Results on Price Scale

DA = directional accuracy. Persistence DA is undefined because the persistence baseline predicts zero change.


| Model / Arm               | Feature Set                               | RMSE        | MAE     | MAPE    | Directional Accuracy | Interpretation                                                                    |
| ------------------------- | ----------------------------------------- | ----------- | ------- | ------- | -------------------- | --------------------------------------------------------------------------------- |
| Persistence               | Previous close only                       | **13.2156** | 9.8148  | 3.1279  | N/A                  | Best price-level benchmark; difficult to beat at daily horizon                    |
| BiLSTM                    | Close only                                | 29.2919     | 23.9963 | 7.1783  | 52.04%               | Neural sequence modeling did not outperform persistence                           |
| BiLSTM + BoW              | Close + traditional sentiment             | 19.9159     | 15.1331 | 4.6752  | 53.53%               | Simple sentiment improved over the price-only BiLSTM but did not beat persistence |
| BiLSTM + Gemini-2.5-flash | Close + `sentiment`, `risk`, `growth`     | 46.2762     | 40.7292 | 12.1311 | 49.81%               | LLM-derived structured features did not produce reliable daily-horizon signal     |
| BiLSTM + FinBERT          | Close + `positive`, `neutral`, `negative` | 23.6160     | 18.4853 | 5.5553  | **54.28%**           | Transformer sentiment did not beat the naive price baseline or the DA threshold   |


> Exact per-arm metrics are produced by the experiment runner in `results/corrected_results.csv`. The key finding is that all text-augmented variants remained worse than the persistence baseline on RMSE and below the directional significance threshold. This supports a cautious interpretation consistent with weak-form market efficiency at the daily horizon, and the negative result is replicated across independent FinBERT and Gemini text-scoring paths.

---

## 12. Interpretation

The result should not be read as “text is useless” or “LLMs cannot support financial analysis.” A more precise interpretation is:

1. **Daily close-price levels are a very hard forecasting target.**
  At a one-day horizon, a persistence baseline is often extremely competitive.
2. **Public disclosure text may already be priced in quickly.**
  Once filings are aligned by true public availability dates, their incremental predictive value at daily frequency becomes difficult to detect.
3. **LLM features are not automatically alpha.**
  Structured LLM scores may be useful for interpretability, screening, qualitative research, or longer-horizon analysis, but they do not guarantee short-horizon price predictability.
4. **Controlled ablation matters.**
  Without leakage controls, text-augmented models can appear artificially strong. This project shows how to test that claim under a fair protocol.
5. **A strong baseline is essential.**
  The persistence model exposes whether the deep model is learning useful signal or simply tracking price levels less efficiently.

---

## 13. Business and Analytics Takeaways

For a data science / business analytics audience, the key lesson is not simply model performance. The key lesson is **decision-grade evaluation**.


| Lesson                            | Practical Meaning                                                                        |
| --------------------------------- | ---------------------------------------------------------------------------------------- |
| Build the baseline first          | A complex model must beat a simple operational benchmark                                 |
| Control information timing        | In finance, feature availability is as important as feature engineering                  |
| Use ablations                     | Separate the contribution of price history, simple sentiment, and LLM features           |
| Interpret negative results        | A failed improvement can still be valuable if it prevents overinvestment in weak signals |
| Prefer reproducibility            | Automated self-checks and manifest-based ingestion improve auditability                  |
| Be cautious with LLM alpha claims | LLMs can enrich features, but market prediction requires rigorous testing                |


---

## 14. What This Project Demonstrates

This repository demonstrates practical capability across the full data science workflow:

- **Data engineering:** SEC filing ingestion, manifest construction, metadata validation, and reproducible document processing.
- **NLP:** MD&A extraction, bag-of-words sentiment, FinBERT inference, and LLM-based structured feature generation.
- **Time-series modeling:** Sliding-window construction, BiLSTM sequence forecasting, and chronological train-test splitting.
- **Experiment design:** Controlled ablation, shared training configuration, and strong baseline comparison.
- **Financial ML awareness:** Leakage control, information timing, and cautious interpretation of apparent alpha.
- **Business analytics judgment:** Translating model results into investment-relevant and operationally realistic conclusions.

---

## 15. Repository Structure

```text
.
├── src/
│   ├── corrected_ablation.py
│   └── build_manifest.py
├── data/
│   ├── mda_extracted/
│   ├── tesla_sentiment_features.csv
│   ├── tesla_sentiment_features_finbert.csv
│   └── tesla_sentiment_features_gemini.csv
├── tesla_stock_price/
│   └── Tesla.csv
├── tesla_financial_reports/
│   └── manifest.csv
├── results/
│   ├── corrected_results.csv
│   └── comparison_corrected.png
└── README.md
```

---

## 16. Reproducibility

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run default experiment

```bash
python src/corrected_ablation.py
```

The default run reuses existing text-score CSV files and makes no paid API calls if precomputed scores are available.

### Regenerate text scores and run the full comparison

PowerShell:

```powershell
$env:GEMINI_API_KEY = "<your-key>"
python src/corrected_ablation.py --rescore
```

Bash / macOS / Linux:

```bash
export GEMINI_API_KEY="your_key_here"
python src/corrected_ablation.py --rescore
```

This regenerates the local FinBERT scores and Gemini scores, then runs the full comparison.

### Rebuild SEC filing manifest

```bash
python src/build_manifest.py
```

Outputs:

```text
results/corrected_results.csv
results/comparison_corrected.png
README.md
```

---

## 17. Limitations and Next Steps


| Limitation                 | Potential Extension                                                                    |
| -------------------------- | -------------------------------------------------------------------------------------- |
| Single ticker              | Extend to a cross-sectional multi-equity panel                                         |
| Single chronological split | Add walk-forward validation                                                            |
| Daily horizon only         | Test weekly, monthly, and event-window horizons                                        |
| Price-level target         | Add return, volatility, abnormal return, or direction targets                          |
| Simple LLM scores          | Use richer embeddings, topic decomposition, or retrieval-augmented disclosure features |
| LLM pretraining hindsight  | Use strictly time-bounded models or historical-only evaluation infrastructure          |
| No trading simulation      | Add transaction costs, slippage, and risk-adjusted portfolio metrics                   |
| Disclosure-only text       | Add earnings calls, analyst revisions, news, or macro factors                          |


---

## 18. Final Conclusion

This project shows that adding Transformer and LLM-derived disclosure features to a BiLSTM forecasting model does not automatically improve daily equity price prediction. Under a leakage-controlled setup, the naive persistence baseline achieved lower RMSE than all neural variants, while directional accuracy remained statistically indistinguishable from chance, with the best learned model at 54.28% over 269 out-of-sample test days. The negative result was replicated across independent FinBERT and Gemini text-scoring paths.

The main contribution is therefore methodological: it demonstrates how to build a credible NLP-to-forecasting experiment, how to prevent common financial machine-learning leakage, and how to evaluate LLM-augmented models with discipline rather than hype.

For applied data science roles, this project highlights the ability to combine **NLP, deep learning, financial data engineering, statistical evaluation, and business interpretation** in a single reproducible workflow.