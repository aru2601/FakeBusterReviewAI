# Fake Review Detection — Streamlit Dashboard

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

## Files

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit dashboard |
| `results_summary.csv` | Pre-loaded results (your 10-fold CV output) |
| `requirements.txt` | Python dependencies |

## Tabs

| Tab | Content |
|-----|---------|
| **Overview** | KPI cards, grouped bar chart with error bars, F1 heatmap |
| **Ablation** | Config A vs B, ΔF1 bars, paired t-test per dataset |
| **Cross-Domain** | Ott vs Salminen generalisation, scatter plot |
| **Raw Data** | Filterable table + CSV download |

## Re-running the ML pipeline

To run the full training pipeline (requires `deceptive-opinion.csv` and `fake_reviews_dataset.csv`):

```bash
pip install textblob imbalanced-learn scikit-learn nltk
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
# Then run your original code.py (converted from the Colab notebook)
```

The new `results_summary.csv` it produces will be automatically picked up by the dashboard on next load (or use the sidebar uploader).
