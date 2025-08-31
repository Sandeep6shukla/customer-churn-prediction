# train_model.py
# Train churn models with CV + hyperparameter tuning + full evaluation & threshold optimization.

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# ---------- Config ----------
DATA = "customer_churn_cleaned.csv"
TARGET = "ChurnStatus"

OUTDIR = Path("model_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
N_JOBS = -1

# ---------- Load data ----------
if not os.path.exists(DATA):
    raise FileNotFoundError(f"{DATA} not found. Run clean_preprocess.py first.")

df = pd.read_csv(DATA)
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

# ---------- CV setup ----------
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
scoring = "roc_auc"  # primary metric for imbalanced churn

# ---------- Models & search spaces ----------
logit = LogisticRegression(
    class_weight="balanced",  # handle imbalance
    solver="liblinear",       # supports l1/l2
    max_iter=500,
    random_state=RANDOM_STATE
)
logit_search = {
    "penalty": ["l1", "l2"],
    "C": np.logspace(-3, 3, 15)
}

rf = RandomForestClassifier(
    class_weight="balanced",
    random_state=RANDOM_STATE
)
rf_search = {
    "n_estimators": [150, 250, 350, 500],
    "max_depth": [None, 5, 8, 12, 16],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

def tune(model, param_dist, name):
    search = RandomizedSearchCV(
        model, param_distributions=param_dist,
        n_iter=25, scoring=scoring, cv=cv, random_state=RANDOM_STATE,
        n_jobs=N_JOBS, verbose=1, refit=True
    )
    search.fit(X_train, y_train)
    print(f"\nBest {name} params:", search.best_params_)
    print(f"Best CV ROC-AUC ({name}): {search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_, search.best_score_

logit_best, logit_params, logit_cv = tune(logit, logit_search, "LogisticRegression")
rf_best, rf_params, rf_cv = tune(rf, rf_search, "RandomForest")

# ---------- Evaluate on test set ----------
def evaluate(model, name, threshold=0.5):
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)

    metrics = {
        "model": name,
        "threshold": float(threshold),
        "roc_auc": roc_auc_score(y_test, prob),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "support_pos": int((y_test == 1).sum()),
        "support_neg": int((y_test == 0).sum())
    }
    print(f"\n{name} @ {threshold:.2f}: "
          f"ROC-AUC={metrics['roc_auc']:.4f}, "
          f"Acc={metrics['accuracy']:.4f}, "
          f"Prec={metrics['precision']:.4f}, "
          f"Rec={metrics['recall']:.4f}, "
          f"F1={metrics['f1']:.4f}")
    return metrics, prob, pred

metrics_logit_05, prob_logit, pred_logit = evaluate(logit_best, "LogisticRegression")
metrics_rf_05, prob_rf, pred_rf = evaluate(rf_best, "RandomForest")

# ---------- Threshold optimization (maximize F1 or custom cost) ----------
def find_best_threshold(prob, y_true, objective="f1"):
    thresholds = np.linspace(0.1, 0.9, 81)
    best_t, best_val = 0.5, -1
    for t in thresholds:
        p = (prob >= t).astype(int)
        if objective == "f1":
            val = f1_score(y_true, p, zero_division=0)
        else:
            # Example custom profit/cost: weight recall more than precision
            # val = 0.7*recall_score(y_true,p) + 0.3*precision_score(y_true,p)
            val = f1_score(y_true, p, zero_division=0)
        if val > best_val:
            best_val, best_t = val, t
    return float(best_t), float(best_val)

t_logit, f1_logit = find_best_threshold(prob_logit, y_test, "f1")
t_rf, f1_rf = find_best_threshold(prob_rf, y_test, "f1")

metrics_logit_bestT, _, pred_logit_bestT = evaluate(logit_best, "LogisticRegression*bestT", threshold=t_logit)
metrics_rf_bestT, _, pred_rf_bestT = evaluate(rf_best, "RandomForest*bestT", threshold=t_rf)

# ---------- Confusion matrices ----------
def save_cm(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix: {name}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    out = OUTDIR / f"cm_{name.replace('*','')}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out.as_posix()

cm1 = save_cm(y_test, pred_logit, "LogisticRegression_thr0.50")
cm2 = save_cm(y_test, pred_rf, "RandomForest_thr0.50")
cm3 = save_cm(y_test, pred_logit_bestT, f"LogisticRegression_thr{t_logit:.2f}")
cm4 = save_cm(y_test, pred_rf_bestT, f"RandomForest_thr{t_rf:.2f}")

# ---------- ROC & PR curves ----------
def save_curves(model, name):
    fig1, ax1 = plt.subplots(figsize=(5,4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax1)
    ax1.set_title(f"ROC Curve: {name}")
    fig1.tight_layout()
    roc_path = OUTDIR / f"roc_{name}.png"
    fig1.savefig(roc_path, bbox_inches="tight"); plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(5,4))
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax2)
    ax2.set_title(f"Precision-Recall: {name}")
    fig2.tight_layout()
    pr_path = OUTDIR / f"pr_{name}.png"
    fig2.savefig(pr_path, bbox_inches="tight"); plt.close(fig2)

save_curves(logit_best, "LogisticRegression")
save_curves(rf_best, "RandomForest")

# ---------- Feature importance ----------
# Logistic: coefficient magnitudes
coef_path = None
if isinstance(logit_best, LogisticRegression):
    coefs = pd.Series(logit_best.coef_.ravel(), index=X_train.columns).sort_values(key=lambda s: s.abs(), ascending=False)
    top20 = coefs.head(20).to_frame("coefficient")
    coef_path = OUTDIR / "feature_importance_logit_top20.csv"
    top20.to_csv(coef_path, index=True)

# RandomForest: permutation importance (on test for fairness)
perm = permutation_importance(rf_best, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=N_JOBS)
imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
imp_path = OUTDIR / "feature_importance_rf_perm.csv"
imp.to_csv(imp_path, index=True)

# ---------- Persist models & metrics ----------
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
joblib.dump(logit_best, OUTDIR / f"logistic_regression_{timestamp}.pkl")
joblib.dump(rf_best, OUTDIR / f"random_forest_{timestamp}.pkl")

all_metrics = [
    metrics_logit_05, metrics_rf_05,
    metrics_logit_bestT, metrics_rf_bestT
]
with open(OUTDIR / "metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)

# Also save a neat CSV
pd.DataFrame(all_metrics).to_csv(OUTDIR / "metrics.csv", index=False)

print("\nSaved outputs in:", OUTDIR.resolve())
print("Best thresholds:",
      {"logistic": {"t": t_logit, "F1": f1_logit},
       "random_forest": {"t": t_rf, "F1": f1_rf}})