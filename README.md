# Hotel Booking Cancellation Prediction

## a. Problem statement

Hotel booking cancellations can affect occupancy planning, revenue management, and operational decisions. This project builds and compares five supervised classification models to predict whether a hotel booking will be cancelled.

The target column is `is_canceled`:
- `0` — booking not cancelled
- `1` — booking cancelled

The models are evaluated on the same held-out test set using Accuracy, ROC-AUC, Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC).

## b. Dataset description

- **Dataset:** Hotel Booking Demand (`hotel_bookings.csv`)
- **Task:** Binary classification
- **Original dataset size stated in the notebook:** 119,390 rows and 32 columns
- **Target:** `is_canceled`
- **Data preparation:** missing-value handling, data-quality filtering, categorical encoding, derived-feature creation, stratified train/test split, SMOTE on the training split only, and Recursive Feature Elimination (RFE).
- **Model input:** 16 RFE-selected features.
- **Evaluation:** a stratified 80/20 train/test split using `random_state=42`; the held-out test set is not oversampled.

**Dataset source:** Add the exact public dataset URL used to obtain `hotel_bookings.csv` before submitting.

**Important reproducibility note:** The notebook expects `hotel_bookings.csv` in the project root, or in a `data/` subfolder. The CSV is not included in this repository by this README; add it only if permitted by the dataset's licence and the assignment rules.

## c. GitHub repository link

**Repository:** `[ADD YOUR PUBLIC GITHUB REPOSITORY URL]`

The repository should contain at least:

```text
project-folder/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── svm.pkl
    ├── feature_names.json
    └── metrics.json
```

Also include the notebook (`2025NS04132.ipynb`) if required by your project layout. Confirm that `app.py` loads the saved models and does not retrain them when the app starts.

## d. Models used and evaluation

All five models are trained for the same binary classification task.

| ML model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | **Fill from notebook output** | **Fill** | **Fill** | **Fill** | **Fill** | **Fill** |
| Decision Tree | **Fill from notebook output** | **Fill** | **Fill** | **Fill** | **Fill** | **Fill** |
| k-Nearest Neighbours (kNN) | **Fill from notebook output** | **Fill** | **Fill** | **Fill** | **Fill** | **Fill** |
| Gaussian Naive Bayes | **Fill from notebook output** | **Fill** | **Fill** | **Fill** | **Fill** | **Fill** |
| Support Vector Machine (SVM) | **Fill from notebook output** | **Fill** | **Fill** | **Fill** | **Fill** | **Fill** |

**Do not submit the placeholder values above.** Run the notebook through the comparison section and copy the six values from the `comparison` table into this table. The notebook file supplied for review had no saved cell outputs, and the dataset was not attached, so verified metric values could not be recovered without running the experiment.

### Observations on model performance

Replace the prompts below with observations based on the actual metrics produced by your run. Compare models using the same test set; do not claim a model is best unless the reported metrics support it.

- **Logistic Regression:** Comment on its Accuracy, AUC, Precision/Recall balance, F1, and MCC. Mention that the model uses a `StandardScaler` pipeline.
- **Decision Tree:** Comment on its scores and the trade-off between interpretability and generalisation after tuning.
- **kNN:** Comment on its scores and prediction cost. kNN is distance-based, so feature scaling and consistent preprocessing are important.
- **Gaussian Naive Bayes:** Comment on its scores and whether its conditional-independence assumption appears limiting for this dataset.
- **Support Vector Machine:** Comment on its scores and the performance/cost trade-off of the RBF kernel.
- **Overall winner:** State the model that gives the best overall result for your objective. Use the full metric set rather than Accuracy alone; AUC, Recall, F1, and MCC can tell different parts of the story.

## How to run the notebook

1. Place `hotel_bookings.csv` in the project root or in `data/hotel_bookings.csv`.
2. Install the packages listed in `requirements.txt`.
3. Run `2025NS04132.ipynb` from top to bottom.
4. Confirm that the comparison table is generated and that the model reload check passes.
5. Confirm that the five model files, `feature_names.json`, `metrics.json`, and `test_data.csv` are generated.

## Streamlit app

- **Live app:** `[ADD YOUR STREAMLIT COMMUNITY CLOUD URL AFTER DEPLOYMENT]`
- **GitHub repository:** `[ADD YOUR REPOSITORY URL]`

The app should allow a user to upload test CSV data, select one of the five saved models, view the evaluation metrics, and inspect a confusion matrix or classification report. The uploaded CSV must contain the same 16 selected feature columns and the `is_canceled` target column if the app is calculating evaluation metrics.

## Dependencies

Keep `requirements.txt` aligned with the imports used by both the notebook and `app.py`. At minimum, the notebook uses pandas, NumPy, Matplotlib, Seaborn, scikit-learn, imbalanced-learn, joblib, and Graphviz-related tooling. If the deployed app does not use notebook-only plotting or graph-rendering libraries, avoid adding unnecessary dependencies to the app requirements.

## Limitations and reproducibility

- The reported test metrics estimate performance on the held-out portion of this dataset; they do not guarantee performance on future bookings or other hotels.
- The notebook applies label encoding to categorical variables. The integer codes may imply an artificial order for nominal categories, which can affect distance-based and linear models.
- For a fair evaluation, any preprocessing or feature selection learned from data should be fitted using training data only.
- Keep the same selected feature names and column order when training, saving, reloading, and evaluating models.
