# Hotel Booking Cancellation Prediction

**Machine Learning, Assignment 2 (M.Tech DSE, BITS WILP)** · **Student ID: 2025NS04132** · **Group 44**

| | |
|---|---|
| **GitHub repository** | <https://github.com/sjainuni/2025ns04132-ml-assignment2> |
| **Live Streamlit app** | <https://2025ns04132.streamlit.app/> |

---

## a. Problem statement

Hotels lose revenue when a room stays empty after a late cancellation, and they lose goodwill when they overbook to compensate. This project builds and compares **five classification models** that predict whether a hotel booking will be cancelled (`is_canceled`: 0 = kept, 1 = cancelled), so that a revenue manager can focus on high-risk reservations (for example by requiring a deposit, sending a reminder, or planning a controlled overbooking).

It is a **binary classification** problem. The positive class is *Cancelled*, so Precision, Recall and F1 below are reported for cancellations. The models are served through an interactive Streamlit app.

## b. Dataset description

| Property | Value |
|---|---|
| Source | Kaggle, *Hotel Booking Demand* (`hotel_bookings.csv`) |
| Task | Binary classification, target `is_canceled` |
| Raw size | **119,390 instances × 32 features** (requirement: ≥ 500 instances and ≥ 12 features) |
| Contents | Booking records of a city hotel and a resort hotel (arrivals 2015–2017): lead time, stay length, guests, meal, country, market segment, deposit type, ADR (average daily rate), special requests, previous cancellations, etc. |
| Class balance | 62.7% not cancelled / 37.3% cancelled after cleaning (moderate imbalance) |

**Preprocessing** (all in `model/2025NS04132.ipynb`, each step explained there):

1. Removed the leakage columns `reservation_status` and `reservation_status_date`, and the sparse `company` column (94% missing).
2. Imputed `children` and `agent` with 0 and `country` with the most frequent value.
3. Removed impossible records (negative or extreme ADR, bookings with no guests) and outliers (lead time above 730 days, zero-night stays). **118,340 rows remain.**
4. Label-encoded the categorical columns and engineered new features (`total_stay_nights`, `weekend_ratio`, `total_guests`, `room_was_changed`, `guest_cancel_rate`, …).
5. Stratified 80/20 split (`random_state=42`): 94,672 training rows and **23,668 test rows**.
6. **SMOTE on the training set only** to balance the classes (118,798 rows after).
7. **RFE** (Recursive Feature Elimination) reduced the 27 features to the **16 features** used by every model.
8. `StandardScaler` (inside a pipeline, fitted on training data only) for Logistic Regression, kNN and SVM.

**Features used by the models (16):** `lead_time`, `country`, `booking_changes`, `deposit_type`, `agent`, `customer_type`, `adr`, `required_car_parking_spaces`, `total_of_special_requests`, `room_was_changed`, `total_stay_nights`, `arrival_date_month_encoded`, `arrival_year_month`, `weekend_ratio`, `total_guests`, `guest_cancel_rate`.

`test_data.csv` is the **23,668-row hold-out set** (16 features + `is_canceled`) that no model saw during training or tuning. It is already cleaned and encoded, so the app can score it directly.

## c. GitHub repository link

<https://github.com/sjainuni/2025ns04132-ml-assignment2>

```
2025ns04132-ml-assignment2/
│-- app.py                  Streamlit app (model demonstration)
│-- requirements.txt        Python dependencies
│-- README.md               This file
│-- test_data.csv           Held-out test data (23,668 rows) used by the app
│-- data/
│   └-- hotel_bookings.csv  Original dataset
└-- model/
    │-- 2025NS04132.ipynb   Notebook: EDA, preprocessing, 5 models, evaluation, comparison
    │-- logistic_regression.pkl
    │-- decision_tree.pkl
    │-- knn.pkl
    │-- naive_bayes.pkl
    │-- svm.pkl
    │-- feature_names.json  The 16 feature names, in model order
    └-- metrics.json        Notebook evaluation results (shown beside live results in the app)
```

**Run locally**

```bash
pip install -r requirements.txt
streamlit run app.py
```

To re-train, open `model/2025NS04132.ipynb` and run all cells (about 5 minutes; the SVM step is the slowest). It regenerates the `.pkl` files, `metrics.json` and `test_data.csv`. The notebook also needs `imbalanced-learn`, which it installs itself if missing.

## d. Models used

All five models were trained on the **same** SMOTE-balanced training set with the **same** 16 features, tuned with cross-validated grid search (ROC-AUC), and evaluated on the **same** untouched test set of **23,668 bookings**. Precision, Recall and F1 are for the positive class (*Cancelled*).

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7688 | 0.8494 | 0.6847 | 0.7034 | 0.6939 | 0.5083 |
| Decision Tree | **0.8495** | 0.9284 | **0.8081** | 0.7818 | 0.7947 | 0.6763 |
| kNN | 0.8483 | **0.9339** | 0.7757 | **0.8340** | **0.8038** | **0.6816** |
| Naive Bayes | 0.7533 | 0.8349 | 0.6507 | 0.7292 | 0.6877 | 0.4870 |
| Support Vector Machine | 0.8263 | 0.8981 | 0.7591 | 0.7820 | 0.7704 | 0.6310 |

*(Best value per metric in bold. Source: `model/metrics.json`, produced by the notebook.)*

Settings chosen by cross-validation: Logistic Regression `C=100`; Decision Tree `max_depth=12`, `min_samples_leaf=15`, `min_samples_split=60`, entropy criterion, balanced class weights; kNN `k=31` with distance weighting; Gaussian Naive Bayes `var_smoothing=1e-07`; SVM with RBF kernel, `C=5`, `gamma='scale'`, trained on a stratified 20,000-row sample of the training set.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy 0.769 and AUC 0.849, clearly below the three stronger models. Its cross-validated AUC was identical (≈ 0.862) for every value of `C`, so regularisation was not the issue: the linear boundary and the arbitrary numeric codes of label-encoded features (`deposit_type`, `country`, `agent`) limit it. Its coefficients are still informative: parking spaces, room changes and special requests lower the cancellation risk, while a history of cancellations raises it. |
| Decision Tree | The best on **Accuracy (0.850)** and **Precision (0.808)**: when it flags a cancellation it is right 81% of the time, and it makes the fewest false alarms (1,637). It misses more cancellations than kNN (recall 0.782). It is tiny (about 19 KB), instant and explainable as rules. `deposit_type`, `agent` and `country` are its most important features. |
| kNN | The best on **AUC (0.934), Recall (0.834), F1 (0.804) and MCC (0.682)**, and it misses the fewest cancellations (1,464 of 8,818). Bookings that resemble past ones tend to behave alike. Costs: it stores the whole training set and is slower to predict. About 27% of the dataset's rows are exact duplicates, which helps a memory-based model like kNN. |
| Naive Bayes | The weakest model (accuracy 0.753, MCC 0.487). Its two assumptions fail here: the features are correlated, and several are skewed or arbitrary codes rather than bell-shaped. About 1 in 3 of its cancellation alerts is a false alarm. It is very fast but not suitable for this dataset. |
| Support Vector Machine | Third overall (accuracy 0.826, AUC 0.898, MCC 0.631), clearly ahead of Logistic Regression and Naive Bayes thanks to the RBF kernel. It was trained on only 20,000 of the 118,798 training rows (training took 23 seconds in our run), which probably holds it back. It is the slowest model to train and to predict. |
| **Overall, winner for your dataset?** | **kNN.** It has the best average rank across the six metrics (1.33, against 1.83 for the Decision Tree) and wins on AUC, Recall, F1 and MCC, the metrics that matter when a missed cancellation costs revenue. The Decision Tree is a very close second (best accuracy and precision) and is the better choice when speed, size or explainability matter. |

### Limitations

- About 27% of rows are exact duplicates, so identical bookings can appear in both train and test, which may inflate every model's scores (most for kNN and the Decision Tree).
- `room_was_changed` is recorded at or after check-in, so it may leak information about the outcome. It would not be available when predicting at booking time.
- Results come from a single 80/20 split; differences of less than about 0.5 percentage points between models should not be treated as decisive.

## e. Streamlit app

Live app: <https://2025ns04132.streamlit.app/>

The app (`app.py`) loads the five saved models and lets you:

- **Upload a CSV** of test data (or use the bundled `test_data.csv`). The file must contain the 16 feature columns and, for metrics, `is_canceled`. Invalid files give a clear error message.
- **Select a model** from a dropdown.
- See the **six evaluation metrics**, a **confusion matrix**, a **classification report**, an ROC curve and the distribution of predicted probabilities, with an adjustable decision threshold.
- **Compare all five models** side by side (metric table, bar chart, combined ROC curves, confusion matrices).
- **Browse booking-level predictions** (filter by risk band, download as CSV) and inspect one booking to see what each model predicts for it.
