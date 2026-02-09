# Rhetorical Adaptation Following Party Switching: Evidence from Brazil's Chamber of Deputies

**Author:** Arthur Gomes Nery
**Date:** January 2026

## Overview

This repository contains the replication materials for "Rhetorical Adaptation Following Party Switching: Evidence from Brazil's Chamber of Deputies." The study analyzes 365,551 legislative speeches from Brazil's Chamber of Deputies (2003–2025) to measure linguistic adaptation following 1,642 party switches involving 933 deputies.

### Key Findings

- Deputies' speeches become **3.9 percentage points** less likely to be classified as their old party within one bimester of switching (p < 0.001), representing a **34% decline** from baseline
- Approximately **73%** of detected change reflects shifts in whom deputies reference (named entities), while **27%** reflects changes in policy vocabulary
- Linguistic and voting changes are **not correlated** at the individual level among switchers with sufficient data
- Rightward switches show **larger effects** than leftward switches (p = 0.025)

## Repository Structure

```
src
├── 0_scrape_proposicoes.py      # Script to scrape legislative proposals
├── 0_scrape_votos.py            # Script to scrape roll-call voting records
├── 0_scrape_discursos.py        # Script to scrape legislative speeches
├── 1_corpus_preprocessing.ipynb  # Text preprocessing pipeline
├── 2_panel_construction.ipynb    # Panel dataset construction
├── 3_modeling_.ipynb             # Main analysis and modeling
└── Paper_ICMC_USP_MBA_Arthur_Nery.pdf  # Full paper
```

## Data Sources

All data are collected from the Brazilian Chamber of Deputies Open Data API:

- **Legislative speeches**: 365,551 speeches from 1,634 deputies (2003-2025)
- **Party switching records**: 1,642 switch events from administrative data
- **Roll-call votes**: 388,911 individual vote records
- **Deputy affiliations**: Historical party membership data

API endpoint: `https://dadosabertos.camara.leg.br/api/v2/`

## Methodology

### Text Analysis Pipeline

1. **Data Collection** (`0_scrape_*.py`)

   - Asynchronous scraping with rate limiting and retry logic
   - Parquet output with Brotli compression
2. **Preprocessing** (`1_corpus_preprocessing.ipynb`)

   - Multi-level cleaning (structural → normalization → NER → stemming)
   - Named Entity Recognition using spaCy Portuguese model
   - Portuguese stopword removal and stemming (RSLP)
3. **Panel Construction** (`2_panel_construction.ipynb`)

   - Converting data into longform panel
4. **Modeling** (`3_modeling_.ipynb`)

   - Multinomial logistic regression classifier (25 parties, 28.9% accuracy)
   - Double Machine Learning with 3-fold cross-fitting
   - Event study specification with parallel trends tests

## Empirical Strategy

### Classifier

- **Algorithm**: Multinomial logistic regression with class balancing
- **Features**: TF-IDF vectors (5,000 terms, unigrams + bigrams)
- **Training**: 611 non-switching deputies (139,677 speeches)
- **Performance**: 28.9% accuracy (7.23× random baseline)

### Event Study Design

$$
Y_{it} = \alpha + \sum_{\tau \neq -1} \beta_\tau \cdot \mathbb{1}[\text{EventTime}_{it} = \tau] + X'_{it}\gamma + \varepsilon_{it}
$$

Where:

- $Y_{it}$ = P(old party | speech)
- $\tau$ ∈ {-6, -5, ..., -1, +1, ..., +6} (bimesters)
- Controls: career tenure, legislative activity, legislature FE, topic FE

### Double Machine Learning

Three-step procedure with cross-fitting:

1. Model outcome residuals using gradient boosting
2. Model treatment propensity using gradient boosting
3. Regress outcome residuals on treatment residuals

## Main Results Summary

| Analysis                      | Estimate | Std. Error | p-value | N      |
| ----------------------------- | -------- | ---------- | ------- | ------ |
| Linguistic adaptation (τ=+1) | -0.0387  | 0.0073     | <0.001  | 47,019 |
| Named entity component        | -0.0284  | —         | <0.001  | 47,019 |
| Policy vocabulary component   | -0.0103  | —         | <0.001  | 47,019 |
| Voting loyalty change         | +0.029   | 0.016      | 0.063   | 321    |
| Linguistic-voting correlation | -0.114   | 0.065      | 0.074   | 247    |

## Robustness Checks

All main results are robust to:

- Alternative estimators (OLS, TWFE, DML)
- Different window widths (±6, ±9, ±12 months)
- Different bin sizes (30, 60, 90 days)
- Alternative classifiers (Naive Bayes, SVM, Random Forest)
- Bloc-level classification (Left/Center/Right)
- Placebo tests on non-switchers

## Citation

```bibtex
@unpublished{nery2026rhetorical,
  title={Rhetorical Adaptation Following Party Switching: Evidence from Brazil's Chamber of Deputies},
  author={Nery, Arthur Gomes},
  year={2026},
  month={January}
}
```

---

**Keywords:** Party switching, legislative behavior, political rhetoric, text-as-data, Brazil, machine learning, Double Machine Learning, event study design
