
# Credit Risk Intelligence Platform

An end-to-end machine learning project that analyses credit-default risk using Python, scikit-learn, pandas and Streamlit.

## Project Overview

This project explores customer credit data, compares classification models, evaluates prediction thresholds, and presents findings through an interactive financial analytics dashboard.

## Objectives

- Explore and clean historical credit-default data.
- Compare Logistic Regression and Random Forest models.
- Evaluate models using precision, recall, F1-score, ROC-AUC and PR-AUC.
- Investigate feature importance and classification thresholds.
- Build an interactive dashboard for customer risk analysis.

## Tools and Technologies

- Python
- pandas and NumPy
- scikit-learn
- Plotly
- Streamlit
- Matplotlib
- Git and GitHub

## Model Results

The initial Random Forest model achieved the following results on a held-out test set:

| Metric | Result |
|---|---:|
| Accuracy | 79.76% |
| Precision | 54.61% |
| Recall | 50.45% |
| F1-score | 52.45% |
| ROC-AUC | 77.13% |
| PR-AUC | 55.11% |

These results are from the initial experiment. Model performance depends on the dataset, evaluation method and selected threshold.

## Key Findings

- Recent repayment status was the strongest feature in the initial Random Forest feature-importance analysis.
- Lower classification thresholds identified more actual defaults but also generated more false positives.
- The threshold with the highest F1-score among the five tested thresholds was 0.40.

These are associations and model findings, not evidence of causation.

## Dashboard Features

- Portfolio overview
- Customer risk categories
- Model comparison
- Feature-importance visualisation
- Threshold analysis
- Individual customer risk scoring

## Dataset

The project uses the UCI Default of Credit Card Clients dataset.

Dataset source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

Please consult the dataset source for its terms of use and citation requirements.

## Limitations

- The model is an educational portfolio project, not a production credit-scoring system.
- The dataset represents historical outcomes and may not reflect current lending conditions.
- Predictions can contain false positives and false negatives.
- Feature importance does not establish causation.
- The model should not be used to make real lending decisions without appropriate validation, fairness assessment, governance and monitoring.

## Author

Data Science and Economics portfolio project.
