# =============================================================================
# FRAUD DRIFT DETECTION - MODEL TRAINING MODULE
# Author  : Janitha
# Part    : Model Training (Random Forest & XGBoost)
# Project : Explainability-Driven Drift Detection for Credit Card Fraud Models
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)
from sklearn.utils import resample
import xgboost as xgb
import joblib

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH   = "creditcard1.csv"
OUTPUT_DIR  = "janitha_outputs"
RANDOM_SEED = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  FRAUD DETECTION — MODEL TRAINING")
print("  Author: Janitha")
print("=" * 60)


# ─────────────────────────────────────────────
# STEP 1: LOAD & INSPECT DATA
# ─────────────────────────────────────────────
print("\n[1/6] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape        : {df.shape}")
print(f"  Fraud cases  : {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
print(f"  Normal cases : {(df['Class'] == 0).sum()}")
print(f"  Missing vals : {df.isnull().sum().sum()}")


# ─────────────────────────────────────────────
# STEP 2: PREPROCESSING
# ─────────────────────────────────────────────
print("\n[2/6] Preprocessing...")

# Scale Amount and Time (V1–V28 are already PCA-transformed)
scaler = StandardScaler()
df["Amount_Scaled"] = scaler.fit_transform(df[["Amount"]])
df["Time_Scaled"]   = scaler.fit_transform(df[["Time"]])

feature_cols = [c for c in df.columns if c.startswith("V")] + ["Amount_Scaled", "Time_Scaled"]
X = df[feature_cols]
y = df["Class"]

# Train / test split (stratified to preserve fraud ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)
print(f"  Train size : {X_train.shape[0]}  |  Test size : {X_test.shape[0]}")

# Handle class imbalance via oversampling on training set
df_train = pd.concat([X_train, y_train], axis=1)
majority = df_train[df_train["Class"] == 0]
minority = df_train[df_train["Class"] == 1]
minority_upsampled = resample(minority, replace=True,
                              n_samples=len(majority) // 2,
                              random_state=RANDOM_SEED)
df_balanced = pd.concat([majority, minority_upsampled])
X_train_bal = df_balanced[feature_cols]
y_train_bal  = df_balanced["Class"]
print(f"  Balanced train fraud ratio: {y_train_bal.mean()*100:.1f}%")


# ─────────────────────────────────────────────
# STEP 3: TRAIN RANDOM FOREST
# ─────────────────────────────────────────────
print("\n[3/6] Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_SEED,
    n_jobs=-1
)
rf_model.fit(X_train_bal, y_train_bal)

rf_preds      = rf_model.predict(X_test)
rf_proba      = rf_model.predict_proba(X_test)[:, 1]
rf_roc_auc    = roc_auc_score(y_test, rf_proba)
rf_f1         = f1_score(y_test, rf_preds)
rf_precision  = precision_score(y_test, rf_preds)
rf_recall     = recall_score(y_test, rf_preds)

print(f"  ROC-AUC   : {rf_roc_auc:.4f}")
print(f"  F1 Score  : {rf_f1:.4f}")
print(f"  Precision : {rf_precision:.4f}")
print(f"  Recall    : {rf_recall:.4f}")


# ─────────────────────────────────────────────
# STEP 4: TRAIN XGBOOST
# ─────────────────────────────────────────────
print("\n[4/6] Training XGBoost...")
scale_pos_weight = (y_train_bal == 0).sum() / (y_train_bal == 1).sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric="aucpr",
    random_state=RANDOM_SEED,
    n_jobs=-1
)
xgb_model.fit(X_train_bal, y_train_bal,
              eval_set=[(X_test, y_test)],
              verbose=False)

xgb_preds      = xgb_model.predict(X_test)
xgb_proba      = xgb_model.predict_proba(X_test)[:, 1]
xgb_roc_auc    = roc_auc_score(y_test, xgb_proba)
xgb_f1         = f1_score(y_test, xgb_preds)
xgb_precision  = precision_score(y_test, xgb_preds)
xgb_recall     = recall_score(y_test, xgb_preds)

print(f"  ROC-AUC   : {xgb_roc_auc:.4f}")
print(f"  F1 Score  : {xgb_f1:.4f}")
print(f"  Precision : {xgb_precision:.4f}")
print(f"  Recall    : {xgb_recall:.4f}")


# ─────────────────────────────────────────────
# STEP 5: VISUALISATIONS
# ─────────────────────────────────────────────
print("\n[5/6] Generating charts...")

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"rf": "#2196F3", "xgb": "#FF5722", "neutral": "#4CAF50"}

# --- 5a. Confusion Matrices (side by side) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, preds, title, color in zip(
    axes,
    [rf_preds, xgb_preds],
    ["Random Forest", "XGBoost"],
    [COLORS["rf"], COLORS["xgb"]]
):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Fraud"],
                yticklabels=["Normal", "Fraud"],
                linewidths=0.5)
    ax.set_title(f"{title}\nConfusion Matrix", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
fig.suptitle("Model Comparison — Confusion Matrices", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 01_confusion_matrices.png")

# --- 5b. ROC Curves ---
fig, ax = plt.subplots(figsize=(8, 6))
for proba, label, color in [
    (rf_proba,  f"Random Forest (AUC={rf_roc_auc:.3f})",  COLORS["rf"]),
    (xgb_proba, f"XGBoost       (AUC={xgb_roc_auc:.3f})", COLORS["xgb"]),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, label=label, lw=2)
ax.plot([0,1],[0,1], "k--", lw=1, label="Random Classifier")
ax.fill_between(*roc_curve(y_test, xgb_proba)[:2],
                alpha=0.08, color=COLORS["xgb"])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — Random Forest vs XGBoost", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 02_roc_curves.png")

# --- 5c. Precision-Recall Curves ---
fig, ax = plt.subplots(figsize=(8, 6))
for proba, label, color in [
    (rf_proba,  f"Random Forest (AP={average_precision_score(y_test, rf_proba):.3f})",  COLORS["rf"]),
    (xgb_proba, f"XGBoost       (AP={average_precision_score(y_test, xgb_proba):.3f})", COLORS["xgb"]),
]:
    prec, rec, _ = precision_recall_curve(y_test, proba)
    ax.plot(rec, prec, label=label, lw=2)
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_precision_recall.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 03_precision_recall.png")

# --- 5d. Feature Importance (XGBoost top 20) ---
importances = pd.Series(xgb_model.feature_importances_, index=feature_cols)
top20 = importances.nlargest(20)
fig, ax = plt.subplots(figsize=(9, 7))
colors_bar = [COLORS["xgb"] if i < 5 else COLORS["rf"] for i in range(len(top20))]
top20.sort_values().plot(kind="barh", ax=ax, color=colors_bar[::-1], edgecolor="white")
ax.set_title("XGBoost — Top 20 Feature Importances", fontsize=14, fontweight="bold")
ax.set_xlabel("Importance Score", fontsize=12)
for bar, val in zip(ax.patches, top20.sort_values().values):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 04_feature_importance.png")

# --- 5e. Model Metrics Comparison Bar Chart ---
metrics = ["ROC-AUC", "F1 Score", "Precision", "Recall"]
rf_vals  = [rf_roc_auc,  rf_f1,  rf_precision,  rf_recall]
xgb_vals = [xgb_roc_auc, xgb_f1, xgb_precision, xgb_recall]

x = np.arange(len(metrics))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 6))
bars1 = ax.bar(x - width/2, rf_vals,  width, label="Random Forest", color=COLORS["rf"],  alpha=0.85)
bars2 = ax.bar(x + width/2, xgb_vals, width, label="XGBoost",       color=COLORS["xgb"], alpha=0.85)
for bars in [bars1, bars2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Random Forest vs XGBoost — Performance Metrics", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_metrics_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: 05_metrics_comparison.png")


# ─────────────────────────────────────────────
# STEP 6: SAVE MODELS & SUMMARY
# ─────────────────────────────────────────────
print("\n[6/6] Saving models...")
joblib.dump(rf_model,  f"{OUTPUT_DIR}/random_forest_model.pkl")
joblib.dump(xgb_model, f"{OUTPUT_DIR}/xgboost_model.pkl")
joblib.dump(scaler,    f"{OUTPUT_DIR}/scaler.pkl")
print("  Saved: random_forest_model.pkl")
print("  Saved: xgboost_model.pkl")
print("  Saved: scaler.pkl")

print("\n" + "=" * 60)
print("  FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"\n  {'Metric':<15} {'Random Forest':>15} {'XGBoost':>15}")
print(f"  {'-'*45}")
for m, rv, xv in zip(metrics, rf_vals, xgb_vals):
    winner = "  ← best" if xv > rv else ""
    print(f"  {m:<15} {rv:>15.4f} {xv:>15.4f}{winner}")

best = "XGBoost" if xgb_roc_auc > rf_roc_auc else "Random Forest"
print(f"\n  ✅ Best model for deployment: {best}")
print(f"  ✅ Outputs saved to: ./{OUTPUT_DIR}/")
print("=" * 60)

print("\n  Classification Report — XGBoost:")
print(classification_report(y_test, xgb_preds, target_names=["Normal", "Fraud"]))
