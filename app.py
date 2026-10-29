"""Hotel Booking Cancellation - model demonstration app (Streamlit).

Loads the five classifiers trained in model/2025NS04132.ipynb, scores an uploaded
test CSV (or the bundled test_data.csv) and shows metrics, confusion matrix,
classification report, ROC curves and a per-booking inspector.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, roc_curve)

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"
TARGET = "is_canceled"
BUNDLED_ROWS = sum(1 for _ in open(BASE / "test_data.csv")) - 1   # rows in the bundled test set
LABELS = ["Not Cancelled", "Cancelled"]
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

MODEL_INFO = {
    "Logistic Regression": ("📈", "Linear model: weighted sum of features pushed through a sigmoid."),
    "Decision Tree":       ("🌳", "Series of if/else splits learned from the data (tuned, depth 12)."),
    "kNN":                 ("📍", "Votes among the most similar past bookings."),
    "Naive Bayes":         ("🎲", "Bayes' rule with independent Gaussian features."),
    "SVM":                 ("🧭", "RBF-kernel margin classifier trained on a 20k sample."),
}
PALETTE = {"Logistic Regression": "#4C72B0", "Decision Tree": "#55A868", "kNN": "#C44E52",
           "Naive Bayes": "#8172B2", "SVM": "#CCB974"}

st.set_page_config(page_title="Booking Cancellation Lab", page_icon="🏨", layout="wide")

st.markdown("""
<style>
.hero {background: linear-gradient(120deg,#0f2742 0%,#1d5c8a 55%,#2a9d8f 100%);
       padding: 1.6rem 2rem; border-radius: 14px; color: #fff; margin-bottom: 1.2rem;}
.hero h1 {margin: 0; font-size: 2rem; color: #fff;}
.hero p  {margin: .4rem 0 0; opacity: .88; font-size: 1.02rem;}
.kpi {border: 1px solid rgba(128,128,128,.28); border-radius: 12px; padding: .85rem 1rem;
      background: rgba(128,128,128,.07); text-align: center;}
.kpi .v {font-size: 1.65rem; font-weight: 700; line-height: 1.2;}
.kpi .l {font-size: .78rem; letter-spacing: .06em; text-transform: uppercase; opacity: .7;}
.kpi .d {font-size: .78rem; opacity: .75;}
.callout {border-left: 5px solid #2a9d8f; background: rgba(42,157,143,.12);
          padding: .8rem 1rem; border-radius: 6px; margin: .6rem 0;}
.pill {display:inline-block; padding: .1rem .6rem; border-radius: 99px; font-weight: 600; font-size: .85rem;}
.pill.low {background:#d8f3dc; color:#1b4332;} .pill.mid {background:#fff3cd; color:#664d03;}
.pill.high {background:#f8d7da; color:#58151c;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- loading
@st.cache_resource(show_spinner=False)
def load_assets():
    meta = json.loads((MODEL_DIR / "metrics.json").read_text())
    features = json.loads((MODEL_DIR / "feature_names.json").read_text())
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in meta["files"].items()}
    return models, features, meta


@st.cache_data(show_spinner=False)
def predict_proba(model_name: str, x: pd.DataFrame) -> np.ndarray:
    models, _, _ = load_assets()
    return models[model_name].predict_proba(x)[:, 1]


@st.cache_data(show_spinner=False)
def predict_native(model_name: str, x: pd.DataFrame) -> np.ndarray:
    """The model's own predict(): identical to what the notebook evaluated."""
    models, _, _ = load_assets()
    return models[model_name].predict(x)


def labels_for(model_name, x, proba, threshold):
    """Native predict() at the default 0.50; otherwise cut the probability at the chosen threshold."""
    if abs(threshold - 0.5) < 1e-9:
        return predict_native(model_name, x)
    return (proba >= threshold).astype(int)


def score(y_true, proba, pred):
    out = {"Accuracy": accuracy_score(y_true, pred),
           "Precision": precision_score(y_true, pred, zero_division=0),
           "Recall": recall_score(y_true, pred, zero_division=0),
           "F1": f1_score(y_true, pred, zero_division=0),
           "MCC": matthews_corrcoef(y_true, pred)}
    out["AUC"] = roc_auc_score(y_true, proba) if y_true.nunique() == 2 else float("nan")
    return {k: out[k] for k in METRIC_NAMES}


def kpi(label, value, note=""):
    return (f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
            f'<div class="d">{note}&nbsp;</div></div>')


try:
    models, FEATURES, META = load_assets()
except Exception as exc:  # missing/incompatible model files
    st.error(f"Could not load the saved models from `model/`: {exc}")
    st.stop()

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded = st.file_uploader("Upload test data (CSV)", type="csv",
                                help="Use only test data. The CSV needs the 16 feature columns "
                                     f"and, for metrics, the `{TARGET}` column.")
    st.caption(f"No file? The bundled `test_data.csv` ({BUNDLED_ROWS:,} held-out bookings) is used.")
    model_name = st.selectbox("Choose a model", list(models),
                              format_func=lambda m: f"{MODEL_INFO[m][0]}  {m}")
    st.caption(MODEL_INFO[model_name][1])
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01,
                          help="At 0.50 each model uses its own native decision rule (exactly what the notebook "
                               "reported). Move it to flag a booking as 'Cancelled' when its probability "
                               "reaches the chosen value. Lower = catch more cancellations.")
    st.divider()
    with st.expander("Required columns"):
        st.code("\n".join(FEATURES + [TARGET]), language=None)

# ----------------------------------------------------------------------------- data
using_bundled = uploaded is None
try:
    data = pd.read_csv(BASE / "test_data.csv" if using_bundled else uploaded, float_precision="round_trip")
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

missing = [c for c in FEATURES if c not in data.columns]
if missing:
    st.error("The uploaded file is missing required feature columns: " + ", ".join(missing))
    st.stop()
x = data[FEATURES].apply(pd.to_numeric, errors="coerce")
if x.isnull().any().any():
    st.error("Feature columns contain non-numeric or empty values. "
             "Please upload the encoded test data (same format as `test_data.csv`).")
    st.stop()
has_target = TARGET in data.columns
y = data[TARGET].astype(int) if has_target else None

st.markdown(f"""
<div class="hero"><h1>🏨 Booking Cancellation Lab</h1>
<p>Compare five classifiers on hotel bookings and see which reservations are likely to be cancelled.
Currently scoring <b>{len(data):,}</b> bookings from <b>{"the bundled test set" if using_bundled else uploaded.name}</b>
with <b>{model_name}</b>.</p></div>
""", unsafe_allow_html=True)
if not has_target:
    st.warning(f"`{TARGET}` is not in the file, so only predictions are shown (no metrics).")

with st.spinner(f"Scoring with {model_name}…"):
    proba = predict_proba(model_name, x)

tab_res, tab_cmp, tab_pred, tab_about = st.tabs(
    ["📊 Model results", "🏁 Compare all models", "🔍 Predictions", "ℹ️ About"])

# ----------------------------------------------------------------------------- tab 1
with tab_res:
    if not has_target:
        st.info("Upload data that contains the target column to see evaluation metrics.")
    else:
        pred = labels_for(model_name, x, proba, threshold)
        m = score(y, proba, pred)
        bench = META["metrics"].get(model_name, {}) if (using_bundled and abs(threshold - 0.5) < 1e-9) else {}
        cols = st.columns(6)
        for col, k in zip(cols, METRIC_NAMES):
            note = f"notebook: {bench[k]:.4f}" if k in bench else ""
            col.markdown(kpi(k, f"{m[k]:.4f}", note), unsafe_allow_html=True)
        st.write("")

        left, right = st.columns(2)
        with left:
            st.subheader("Confusion matrix")
            cm = confusion_matrix(y, pred, labels=[0, 1])
            fig, ax = plt.subplots(figsize=(4.8, 4))
            ax.imshow(cm, cmap="Blues")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, f"{v:,}\n({v / cm[i].sum():.0%} of row)", ha="center", va="center",
                        color="white" if v > cm.max() / 2 else "black", fontsize=10)
            ax.set_xticks([0, 1], LABELS); ax.set_yticks([0, 1], LABELS)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            st.pyplot(fig, width="stretch"); plt.close(fig)
        with right:
            st.subheader("ROC curve")
            fpr, tpr, _ = roc_curve(y, proba)
            fig, ax = plt.subplots(figsize=(4.8, 4))
            ax.plot(fpr, tpr, lw=2.4, color=PALETTE[model_name], label=f"AUC = {m['AUC']:.3f}")
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
            op_fpr = ((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)
            op_tpr = ((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
            ax.scatter([op_fpr], [op_tpr], s=70, color="black", zorder=5, label=f"threshold {threshold:.2f}")
            ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
            ax.legend(loc="lower right")
            st.pyplot(fig, width="stretch"); plt.close(fig)

        st.subheader("Classification report")
        rep = pd.DataFrame(classification_report(y, pred, target_names=LABELS, output_dict=True,
                                                 zero_division=0)).T
        rep["support"] = rep["support"].astype(int)
        st.dataframe(rep.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}"}),
                     width="stretch")

        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        st.markdown(f"""<div class="callout"><b>How to read this:</b> out of {tp + fn:,} real cancellations the model
        caught <b>{tp:,}</b> and missed <b>{fn:,}</b> (recall {m['Recall']:.1%}). It also raised <b>{fp:,}</b>
        false alarms among {tn + fp:,} bookings that were kept (precision {m['Precision']:.1%}).</div>""",
                    unsafe_allow_html=True)

        st.subheader("Predicted cancellation probability")
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.hist(proba[y == 0], bins=40, alpha=.65, color="#4C72B0", label="Actually kept")
        ax.hist(proba[y == 1], bins=40, alpha=.65, color="#C44E52", label="Actually cancelled")
        ax.axvline(threshold, color="k", ls="--", lw=1.2, label="threshold")
        ax.set_xlabel("Predicted probability of cancellation"); ax.set_ylabel("Bookings"); ax.legend()
        st.pyplot(fig, width="stretch"); plt.close(fig)

# ----------------------------------------------------------------------------- tab 2
with tab_cmp:
    if not has_target:
        st.info("Upload data that contains the target column to compare models.")
    else:
        with st.spinner("Scoring all five models (kNN and SVM are the slow ones, results are cached)…"):
            probas = {n: predict_proba(n, x) for n in models}
        table = pd.DataFrame({n: score(y, p, labels_for(n, x, p, threshold)) for n, p in probas.items()}).T
        table.index.name = "ML Model"

        best = table.idxmax()
        winner = table.rank(ascending=False).mean(axis=1).idxmin()
        st.markdown(f"""<div class="callout">🏆 <b>Best overall at threshold {threshold:.2f}: {winner}</b>
        (best average rank across the six metrics). Highest F1: <b>{best['F1']}</b> ·
        highest AUC: <b>{best['AUC']}</b> · highest MCC: <b>{best['MCC']}</b>.</div>""", unsafe_allow_html=True)

        st.dataframe(table.style.highlight_max(color="#c7e9c0").format("{:.4f}"), width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Metric by model")
            fig, ax = plt.subplots(figsize=(7, 4.2))
            w = 0.8 / len(table)
            for i, (n, row) in enumerate(table.iterrows()):
                ax.bar(np.arange(6) + i * w, row.values, w, label=n, color=PALETTE[n])
            ax.set_xticks(np.arange(6) + 0.4 - w / 2, METRIC_NAMES)
            ax.set_ylim(0, 1); ax.legend(fontsize=8, ncol=2, loc="lower center")
            st.pyplot(fig, width="stretch"); plt.close(fig)
        with c2:
            st.subheader("ROC curves")
            fig, ax = plt.subplots(figsize=(7, 4.2))
            for n, p in probas.items():
                f, t, _ = roc_curve(y, p)
                ax.plot(f, t, lw=2, color=PALETTE[n], label=f"{n} ({roc_auc_score(y, p):.3f})")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(loc="lower right")
            st.pyplot(fig, width="stretch"); plt.close(fig)

        st.subheader("Confusion matrices")
        cols = st.columns(5)
        for col, (n, p) in zip(cols, probas.items()):
            cm = confusion_matrix(y, labels_for(n, x, p, threshold), labels=[0, 1])
            fig, ax = plt.subplots(figsize=(3, 2.8))
            ax.imshow(cm, cmap="Blues")
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=9,
                        color="white" if v > cm.max() / 2 else "black")
            ax.set_xticks([0, 1], ["Kept", "Canc."], fontsize=8); ax.set_yticks([0, 1], ["Kept", "Canc."], fontsize=8)
            ax.set_title(n, fontsize=9)
            col.pyplot(fig, width="stretch"); plt.close(fig)

# ----------------------------------------------------------------------------- tab 3
with tab_pred:
    out = x.copy()
    out.insert(0, "cancel_probability", proba.round(4))
    out.insert(1, "prediction", np.where(labels_for(model_name, x, proba, threshold) == 1, "Cancelled", "Not Cancelled"))
    out.insert(2, "risk", pd.cut(proba, [-0.01, 0.3, 0.6, 1.01], labels=["Low", "Medium", "High"]).astype(str))
    if has_target:
        out.insert(3, "actual", np.where(y == 1, "Cancelled", "Not Cancelled"))
        out.insert(4, "correct", (out["prediction"] == out["actual"]))

    st.subheader(f"Booking-level predictions ({model_name})")
    f1c, f2c, f3c = st.columns([2, 2, 1])
    risk_f = f1c.multiselect("Risk band", ["Low", "Medium", "High"], default=["High", "Medium", "Low"])
    only_wrong = f2c.checkbox("Only show wrong predictions", disabled=not has_target)
    n_rows = f3c.number_input("Rows to show", 10, 1000, 100, step=10)
    view = out[out["risk"].isin(risk_f)]
    if only_wrong and has_target:
        view = view[~view["correct"]]
    st.caption(f"{len(view):,} bookings match the filters")
    st.dataframe(view.head(int(n_rows)), width="stretch", height=320)
    st.download_button("⬇️ Download all predictions (CSV)", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{model_name.replace(' ', '_').lower()}.csv", mime="text/csv")

    st.divider()
    st.subheader("Inspect one booking: what does each model say?")
    row_no = st.number_input("Row number", 0, len(x) - 1, 0)
    a, b = st.columns([3, 2])
    a.dataframe(x.iloc[[int(row_no)]].T.rename(columns={x.index[int(row_no)]: "value"}), width="stretch",
                height=360)
    with b:
        if has_target:
            st.markdown(f"**Actual outcome:** {LABELS[int(y.iloc[int(row_no)])]}")
        for n in models:
            p = float(predict_proba(n, x.iloc[[int(row_no)]])[0])
            band = "low" if p < .3 else "mid" if p < .6 else "high"
            st.markdown(f"**{MODEL_INFO[n][0]} {n}** <span class='pill {band}'>{p:.0%} cancel</span>",
                        unsafe_allow_html=True)
            st.progress(min(max(p, 0.0), 1.0))

# ----------------------------------------------------------------------------- tab 4
with tab_about:
    st.subheader("Problem")
    st.write("Hotels lose revenue when rooms stay empty after a late cancellation, and lose goodwill when they "
             "overbook. This app predicts the probability that a booking will be cancelled (`is_canceled`: "
             "0 = kept, 1 = cancelled) so that revenue managers can act on high-risk reservations.")
    st.subheader("Dataset")
    st.write("Hotel Booking Demand data (Kaggle): 119,390 bookings and 32 columns for a city hotel and a resort "
             "hotel (arrivals 2015-2017). After cleaning, 118,340 bookings and 16 RFE-selected features remain. The test "
             f"set holds {BUNDLED_ROWS:,} bookings that no model saw during training or tuning.")
    st.subheader("The 16 model features")
    st.write(", ".join(f"`{f}`" for f in FEATURES))
    st.subheader("Models")
    for n, (icon, desc) in MODEL_INFO.items():
        st.markdown(f"- {icon} **{n}**: {desc}")
    st.subheader("Using the app")
    st.markdown("1. Pick a model in the sidebar.  \n2. (Optional) upload your own **encoded** test CSV with the "
                "columns listed in the sidebar.  \n3. Move the threshold slider to trade precision against recall.  "
                "\n4. Open *Compare all models* for the side-by-side view.")
