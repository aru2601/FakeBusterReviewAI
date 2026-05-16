# FakeBusterReviewAI

An end-to-end NLP system for detecting deceptive hotel reviews, with a production Streamlit dashboard for real-time classification, ablation analysis, and cross-domain generalisation testing.

> **F1 = 0.891 · Precision = 0.888 · Recall = 0.892 · ROC-AUC = 0.961**  
> 10-fold stratified cross-validation on 1,600 labelled hotel reviews.

<!-- ![FakeBusterReviewAI dashboard](docs/demo.gif) -->

## The problem
Deceptive online reviews distort consumer trust and platform credibility. FakeBusterReviewAI classifies hotel reviews as **genuine** or **deceptive** using a hybrid of n-gram language patterns and hand-engineered linguistic signals, then tests how well that model generalises beyond its training domain.

## Approach

### Data
- **Ott et al. (2011)** deceptive-opinion corpus — 1,600 hotel reviews, balanced 50/50 genuine vs. deceptive.
- **Salminen et al.** dataset — used for cross-domain generalisation testing.

### Features
A **hybrid feature set** combining:
- **TF-IDF n-gram vectors** — unigrams + bigrams, 10,000 features
- **Six linguistic signals:** sentiment polarity, subjectivity, type-token ratio, exclamation frequency, superlative density, word count

### Models
Ablation across four classifiers (Logistic Regression, SVM, Random Forest, Gradient Boosting) and two feature configurations (TF-IDF only vs. TF-IDF + linguistic signals), evaluated under 10-fold stratified cross-validation with paired t-tests between configurations.

Final model: **Logistic Regression** with the hybrid feature set.

## Results

| Metric    | Score |
|-----------|-------|
| F1        | 0.891 |
| Precision | 0.888 |
| Recall    | 0.892 |
| ROC-AUC   | 0.961 |

## Streamlit dashboard

| Tab | Content |
|-----|---------|
| **Overview** | KPI cards, grouped bar chart with error bars, F1 heatmap |
| **Ablation** | Config A vs B, ΔF1 bars, paired t-test per dataset |
| **Cross-Domain** | Ott vs Salminen generalisation, scatter plot |
| **Raw Data** | Filterable table + CSV download |

Plus real-time inference on pasted review text, with a feature-interpretation panel showing which signals drove each prediction.

## Run locally

```bash
git clone https://github.com/aru2601/FakeBusterReviewAI.git
cd FakeBusterReviewAI
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at http://localhost:8501.

### Re-running the full training pipeline
Requires `deceptive-opinion.csv` and `fake_reviews_dataset.csv` (included).

```bash
pip install textblob imbalanced-learn scikit-learn nltk
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
python pipeline.py   # or your training script
```

The regenerated `results_summary.csv` is picked up automatically by the dashboard.

## Tech stack
Python · Scikit-learn · NLTK · TextBlob · Pandas · NumPy · Matplotlib · Seaborn · Streamlit

## Reference
Ott, M., Choi, Y., Cardie, C., & Hancock, J. T. (2011). *Finding deceptive opinion spam by any stretch of the imagination.* ACL.

---
*Built as part of my MSc Data Science dissertation, University of East London.*
