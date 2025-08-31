# clean_preprocess.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

MASTER_CSV = "customer_churn_master.csv"
CLEAN_CSV  = "customer_churn_cleaned.csv"
TARGET     = "ChurnStatus"

def main():
    if not os.path.exists(MASTER_CSV):
        raise FileNotFoundError(f"{MASTER_CSV} not found. Run build_master.py first.")
    df = pd.read_csv(MASTER_CSV)
    assert TARGET in df.columns, f"{TARGET} is missing."

    # Identify columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET in num_cols: num_cols.remove(TARGET)
    cat_cols = [c for c in df.columns if c not in num_cols and c != TARGET]

    # Missing values
    for col in num_cols:
        miss_rate = df[col].isna().mean()
        if miss_rate > 0.05:
            df[f"{col}__was_missing"] = df[col].isna().astype(int)
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        if df[col].isna().any():
            mode_val = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")

    # Outliers (IQR winsorization)
    for col in num_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
            df[col] = df[col].clip(lo, hi)

    # Scale numerics
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Encode categoricals
    encoded_parts = []
    high_card = []
    for col in cat_cols:
        nunique = df[col].nunique(dropna=False)
        if nunique <= 50:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=np.uint8)
            encoded_parts.append(dummies)
        else:
            freqs = df[col].value_counts(normalize=True)
            df[f"{col}__freq"] = df[col].map(freqs).fillna(0.0)
            high_card.append(col)

    if encoded_parts:
        encoded_df = pd.concat(encoded_parts, axis=1)
        drop_oh = [c for c in cat_cols if c not in high_card]
        df = pd.concat([df.drop(columns=drop_oh), encoded_df], axis=1)
    if high_card:
        df = df.drop(columns=high_card)

    # Move target last & save
    cols = [c for c in df.columns if c != TARGET] + [TARGET]
    df = df[cols]
    df.to_csv(CLEAN_CSV, index=False)
    print(f"✅ Saved cleaned dataset: {CLEAN_CSV} — shape: {df.shape}")

if __name__ == "__main__":
    main()
