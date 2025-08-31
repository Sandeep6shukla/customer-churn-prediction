# generate_eda_report.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({"figure.dpi": 120})

MASTER_CSV = "customer_churn_master.csv"
PLOTS_DIR = "eda_plots"
OUT_DIR   = "eda_summaries"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def savefig(path):
    plt.savefig(path, bbox_inches="tight"); plt.close()

def main():
    if not os.path.exists(MASTER_CSV):
        raise FileNotFoundError(f"{MASTER_CSV} not found. Run build_master.py first.")

    df = pd.read_csv(MASTER_CSV)

    # Summaries
    schema = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum().values,
        "pct_missing": (df.isna().mean().values * 100).round(2)
    })
    schema.to_csv(os.path.join(OUT_DIR, "schema_missingness.csv"), index=False)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    if num_cols: df[num_cols].describe().T.to_csv(os.path.join(OUT_DIR, "numeric_describe.csv"))
    if cat_cols:
        pd.Series({c: df[c].nunique() for c in cat_cols}).rename("unique_values")\
          .to_frame().sort_values("unique_values", ascending=False)\
          .to_csv(os.path.join(OUT_DIR, "categorical_cardinality.csv"))

    # Plots
    if "ChurnStatus" in df.columns:
        plt.figure(figsize=(5,4))
        sns.countplot(x="ChurnStatus", data=df)
        plt.title("Churn Distribution (0=Stay, 1=Churn)")
        savefig(os.path.join(PLOTS_DIR, "01_churn_distribution.png"))

    if "Age" in df.columns:
        plt.figure(figsize=(6,4))
        sns.histplot(df["Age"].dropna(), bins=20, kde=True)
        plt.title("Age Distribution")
        savefig(os.path.join(PLOTS_DIR, "02_age_distribution.png"))

    for col in ["Gender","MaritalStatus","IncomeLevel","ServiceUsage"]:
        if col in df.columns:
            plt.figure(figsize=(7,4))
            df[col].value_counts(dropna=False).plot(kind="bar")
            plt.title(f"{col} — Value Counts"); plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            savefig(os.path.join(PLOTS_DIR, f"03_{col}_value_counts.png"))

    for col in ["Total_Spend","LoginFrequency","Days_Since_LastLogin","Txn_Count","Distinct_Categories","Avg_Spend","Days_Since_LastInteraction"]:
        if col in df.columns and "ChurnStatus" in df.columns:
            plt.figure(figsize=(7,4))
            sns.boxplot(x="ChurnStatus", y=col, data=df)
            plt.title(f"{col} vs ChurnStatus")
            savefig(os.path.join(PLOTS_DIR, f"04_{col}_vs_churn_box.png"))

    # Correlations
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True)
        plt.figure(figsize=(10,8))
        sns.heatmap(corr, annot=False, cmap="coolwarm", linewidths=0.3)
        plt.title("Correlation Heatmap (Numeric Features)")
        savefig(os.path.join(PLOTS_DIR, "05_correlation_heatmap.png"))

    # Churn rate by categories
    def churn_rate_by(cat):
        tmp = df.groupby(cat)["ChurnStatus"].mean().sort_values(ascending=False)
        plt.figure(figsize=(7,4))
        tmp.plot(kind="bar")
        plt.ylabel("Churn Rate"); plt.title(f"Churn Rate by {cat}")
        plt.xticks(rotation=45, ha="right")
        savefig(os.path.join(PLOTS_DIR, f"06_churn_rate_by_{cat}.png"))

    if "ChurnStatus" in df.columns:
        for cat in ["Gender","MaritalStatus","IncomeLevel","ServiceUsage"]:
            if cat in df.columns and df[cat].nunique() <= 12:
                churn_rate_by(cat)

    print(f"✅ EDA complete. Plots → {PLOTS_DIR} | Summaries → {OUT_DIR}")

if __name__ == "__main__":
    main()
