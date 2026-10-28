HOTEL BOOKING CANCELLATION PREDICTION
Machine Learning - Assignment 2 | Student ID: 2025NS04132
======================================================================

1. PROBLEM
----------
Predict whether a hotel booking will be cancelled (1) or kept (0) using
five classification models, compare them using six evaluation metrics,
and explore them in an interactive Streamlit web app.

Task   : supervised binary classification
Target : is_canceled (0 = Not Cancelled, 1 = Cancelled)

2. DATASET
----------
Hotel Booking Demand (Kaggle): 119,390 bookings, 32 columns, from a city
hotel and a resort hotel in Portugal.
https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
After cleaning about 118,560 bookings remain. 20% (23,712 bookings) form
the held-out test set that is never used for training or tuning.

3. FILES IN THIS REPOSITORY
---------------------------
2025NS04132.ipynb   Notebook: EDA, pre-processing, 5 models, comparison,
                    saving of models
app.py              Streamlit app (loads saved models, does not retrain)
requirements.txt    Python packages needed by the app
test_data.csv       Held-out test set (16 encoded features + is_canceled)
model/              Saved models (*.pkl), feature_names.json, metrics.json
data/hotel_bookings.csv   Raw dataset (needed only to re-run the notebook)
README.txt          This file

4. METHOD
---------
1) EDA: data types, statistics, distributions, missing values, outliers
   (IQR), impossible values, duplicates, class balance (~63% / 37%).
2) Pre-processing: remove leakage columns (reservation_status,
   reservation_status_date) and 'company'; impute children, agent,
   country; remove impossible records; cap lead_time at 730 days;
   label-encode categorical columns; bin lead_time.
3) Feature engineering: total_stay_nights, weekend_ratio,
   room_was_changed, total_guests, has_children, children_per_adult,
   is_summer, is_winter, guest_cancel_rate.
4) Stratified 80/20 train/test split (random_state = 42).
5) SMOTE on the training set ONLY (test set is left untouched).
6) RFE (Decision Tree) selects 16 features, used by every model.
7) Five models tuned with cross-validated grid search.
8) All models evaluated on the same untouched test set.

5. MODELS
---------
1. Logistic Regression  - StandardScaler + LogisticRegression, C tuned
2. Decision Tree        - GridSearchCV; final: entropy, max_depth=12,
                          min_samples_leaf=15, min_samples_split=60,
                          class_weight='balanced'
3. K-Nearest Neighbours - StandardScaler + kNN; n_neighbors and weights
                          tuned on a 20,000-row sample
4. Gaussian Naive Bayes - var_smoothing tuned
5. Support Vector Machine - StandardScaler + RBF SVC; C and gamma tuned
                          on 10,000 rows; final model trained on a
                          20,000-row sample (full data is too slow)

6. METRICS
---------
Accuracy, AUC (ROC-AUC), Precision, Recall, F1 Score and Matthews
Correlation Coefficient (MCC); also confusion matrices and ROC curves.
Positive class = Cancelled.

Final results: see Section 10 of the notebook and the "Compare all
models" tab of the app.

7. HOW TO RUN
-------------
Notebook:
    pip install -r requirements.txt
    pip install imbalanced-learn seaborn jupyterlab
    jupyter lab 2025NS04132.ipynb
  (put hotel_bookings.csv in data/ or next to the notebook, run all
  cells; the last section writes the files the app needs)

App (locally):
    pip install -r requirements.txt
    streamlit run app.py

Streamlit Community Cloud:
  a) Push ALL files (requirements.txt in the repository ROOT next to
     app.py, test_data.csv and the model/ folder) to GitHub.
  b) On share.streamlit.io: New app -> select repo and branch -> main
     file = app.py -> choose the same Python version used for training.
  c) After editing requirements.txt use: Manage app -> Reboot app.

IMPORTANT: scikit-learn must be the SAME version that was used to train
and save the .pkl models. Find it with
    python -c "import sklearn; print(sklearn.__version__)"
and pin it in requirements.txt, e.g. scikit-learn==1.x.x

8. APP FEATURES
---------------
- Choose one of five models; decision-threshold slider (0.50 = the
  notebook's own predictions).
- Metrics, confusion matrix, ROC curve, classification report and
  probability histogram.
- "Compare all models": bar charts, ROC curves, confusion matrices.
- Booking-level predictions with risk bands (Low/Medium/High), filters
  and CSV download; inspect one booking across all five models.
- Upload your own ENCODED test CSV (16 feature columns, plus
  is_canceled to get metrics).

9. LIMITATIONS
--------------
- Label encoding adds an artificial order to categories (affects
  distance/linear models more than trees).
- SMOTE interpolates encoded categorical codes, giving non-integer
  values in synthetic rows.
- kNN and SVM were tuned/trained on subsamples to limit run time.
- Data is from two hotels in Portugal; results may not generalise.

10. TECH STACK
--------------
Python, pandas, NumPy, scikit-learn, imbalanced-learn, matplotlib,
seaborn, joblib, Streamlit
