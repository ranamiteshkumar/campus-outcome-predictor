# Campus Outcome Predictor

Predicting whether a student will **drop out, remain enrolled or graduate**, using five classification models, with an interactive Streamlit app for evaluation.

**Live App:** https://ranamiteshkumar-campus-outcome-predictor.streamlit.app

---

## a. Problem Statement

Student dropout is a major concern for educational institutions because it affects student success rates, institutional reputation, and resource planning. This project aims to predict whether a student will **drop out, remain enrolled, or graduate** based on demographic, socio-economic, and academic information. Accurate prediction of student outcomes can help universities identify at-risk students at an early stage. Such insights can support timely interventions such as academic counselling, mentoring programmes, and financial assistance to improve student retention and graduation rates.

---

## b. Dataset Description

| Item | Detail |
|---|---|
| Source | UCI Machine Learning Repository: *Predict Students' Dropout and Academic Success* |
| Citation / DOI | 10.24432/C5MC89 |
| Licence | CC BY 4.0 (Creative Commons Attribution 4.0 International License) |
| Instances | 4,424 students |
| Features | 36 (demographic, socio-economic, academic path at enrolment, 1st and 2nd semester performance, macro-economic indicators) |
| Target | `Target`: Dropout / Enrolled / Graduate (multi-class) |
| Class balance | Graduate 49.9% · Dropout 32.1% · Enrolled 17.9% (imbalanced) |
| Missing values / duplicates | None / none |

The dataset was collected from a higher education institution and is publicly available through the UCI Machine Learning Repository. It contains demographic characteristics, socio-economic background information, admission details, and academic performance indicators from the first and second semesters. These features provide a comprehensive view of a student's educational journey, allowing machine learning models to identify patterns associated with dropout, continued enrolment, or successful graduation.

Column names are kept exactly as published by UCI, including the original spelling `Nacionality`.

### Preprocessing

| Step | Choice | Reason |
|---|---|---|
| Nominal codes (9 columns, e.g. Course, Nacionality, parents' occupation) | One-hot encoding, categories with < 20 students grouped as "infrequent" | Codes are labels, not quantities; grouping tames rare categories and handles unseen codes |
| Binary flags (8 columns) | Kept as 0/1 | Already model-ready |
| Numeric (19 columns) | Standard scaling | Needed by Logistic Regression, KNN and SVM |
| Resulting feature count | 119 | |
| Train/test split | 80/20, stratified, fixed seed | Same class balance in both sets; reproducible |
| Class imbalance | `class_weight="balanced"` for LR, Decision Tree, SVM | Extra weight on minority classes, especially Enrolled |
| Metric averaging | Macro | Each class counts equally, so weak Enrolled performance is not hidden |

Preprocessing is built into each saved model pipeline, so the app applies identical transformations to uploaded data and never retrains.

---

## c. GitHub Repository Link

https://github.com/ranamiteshkumar/campus-outcome-predictor

```text
campus-outcome-predictor/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── data/
│   └── data.csv
├── model/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── support_vector_machine.pkl
│   └── metrics_summary.csv
└── notebooks/
    └── model_training.ipynb
```

---

## d. Models Used

Logistic Regression, Decision Tree (max depth 8), K-Nearest Neighbours (k = 15, distance-weighted), Gaussian Naive Bayes (`var_smoothing` tuned using 5-fold cross-validation on the training data only), and Support Vector Machine (RBF kernel, probabilities via calibration). All five models use the same preprocessing pipeline and are evaluated on the same held-out test set.

### Comparison Table (Test Set: 885 Students)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7503 | 0.8909 | 0.7086 | 0.7108 | 0.7049 | 0.6036 |
| Decision Tree | 0.7006 | 0.8311 | 0.6822 | 0.6824 | 0.6689 | 0.5407 |
| KNN | 0.7073 | 0.8361 | 0.6689 | 0.5883 | 0.5977 | 0.5144 |
| Naive Bayes | 0.7096 | 0.8244 | 0.6633 | 0.6412 | 0.6460 | 0.5246 |
| Support Vector Machine | 0.7650 | 0.8795 | 0.7156 | 0.7094 | 0.7103 | 0.6191 |

*AUC is one-vs-rest, macro-averaged; Precision, Recall and F1 are macro-averaged.*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Logistic Regression achieved 75.03% accuracy and an MCC of 0.6036, making it the second-best model overall. It also achieved the highest AUC (0.8909) and the highest recall (0.7108), indicating strong ability to distinguish between the three outcome classes. The model performs well because important academic indicators such as approved units and semester grades have a fairly direct relationship with student outcomes. |
| **Decision Tree** | The Decision Tree classifier achieved 70.06% accuracy and an MCC of 0.5407. Its AUC score (0.8311) was noticeably lower than Logistic Regression and SVM because decision trees assign the same probability to all samples within a leaf, resulting in coarse probability estimates. A depth limit of 8 was used to reduce overfitting, although this likely reduced predictive performance slightly. |
| **KNN** | KNN achieved 70.73% accuracy but recorded the lowest recall (0.5883) and F1 score (0.5977) among the five models. Unlike Logistic Regression, Decision Tree, and SVM, KNN does not use class weighting, which allows the majority Graduate class to dominate neighbourhood voting. In addition, distance calculations become less informative across 119 processed features, which likely contributed to the weaker recall. |
| **Naive Bayes** | Naive Bayes achieved 70.96% accuracy and an MCC of 0.5246 after hyperparameter tuning. During initial experiments, the untuned model achieved only around 30% accuracy and heavily over-predicted the Enrolled class because one-hot encoded 0/1 features do not follow the bell-shaped distribution that Gaussian Naive Bayes assumes. Tuning the `var_smoothing` parameter through 5-fold cross-validation on the training data only improved model stability and increased accuracy to 70.96%. |
| **Support Vector Machine** | Support Vector Machine delivered the strongest overall results, achieving the highest accuracy (76.50%), precision (0.7156), F1 score (0.7103), and MCC (0.6191). Although its AUC (0.8795) was slightly lower than Logistic Regression, it provided the best balance across all evaluation metrics. |
| **Overall Winner** | **Support Vector Machine (SVM)** was selected as the best-performing model because it achieved the highest Accuracy, Precision, F1 Score, and MCC values. However, Logistic Regression was a very close second and achieved the best AUC and Recall while offering greater interpretability, making it a practical alternative when explainability is important. In this dataset, the **Enrolled** class was generally more difficult to predict than the Dropout and Graduate classes because these students have not yet reached a final outcome and therefore share characteristics with both groups. |

---

## Streamlit App

**Link:** https://ranamiteshkumar-campus-outcome-predictor.streamlit.app

### Features

- Upload test dataset (CSV)
- Select any of the five trained models
- View Accuracy, AUC, Precision, Recall, F1 Score, and MCC
- Display confusion matrix
- Display classification report
- Compare all models side-by-side
- Download sample `test_data.csv`

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```