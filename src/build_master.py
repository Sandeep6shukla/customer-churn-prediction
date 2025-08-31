# build_master.py
import os
import pandas as pd
import numpy as np

DATA_FILE = "Customer_Churn_Data_Large.xlsx"
OUT_CSV   = "customer_churn_master.csv"

def std_cols(df):
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df

def build_master(xlsx_path: str) -> pd.DataFrame:
    xls = pd.ExcelFile(xlsx_path)

    # Load sheets
    demo_df    = std_cols(pd.read_excel(xls, "Customer_Demographics"))
    txn_df     = std_cols(pd.read_excel(xls, "Transaction_History"))
    service_df = std_cols(pd.read_excel(xls, "Customer_Service"))
    online_df  = std_cols(pd.read_excel(xls, "Online_Activity"))
    churn_df   = std_cols(pd.read_excel(xls, "Churn_Status"))

    # --- Transactions → per-customer summary
    # Expect: CustomerID, Transaction_ID, Transaction_Date, AmountSpent, ProductCategory
    if {"CustomerID","AmountSpent","ProductCategory"}.issubset(txn_df.columns):
        txn_summary = txn_df.groupby("CustomerID").agg({
            "AmountSpent": ["sum", "mean", "count"],
            "ProductCategory": "nunique"
        }).reset_index()
        txn_summary.columns = ["CustomerID","Total_Spend","Avg_Spend","Txn_Count","Distinct_Categories"]
    else:
        txn_summary = pd.DataFrame(columns=["CustomerID","Total_Spend","Avg_Spend","Txn_Count","Distinct_Categories"])

    # --- Service → per-customer summary
    # Expect: CustomerID, InteractionID, InteractionDate, InteractionType, ResolutionStatus
    service_summary = pd.DataFrame(columns=["CustomerID","Total_Interactions","Resolved_Count","Last_Interaction_Date","Unresolved_Count"])
    if {"CustomerID","InteractionID","InteractionDate","ResolutionStatus"}.issubset(service_df.columns):
        service_df["InteractionDate"] = pd.to_datetime(service_df["InteractionDate"], errors="coerce")
        service_summary = service_df.groupby("CustomerID").agg({
            "InteractionID": "count",
            "ResolutionStatus": lambda x: (x == "Resolved").sum(),
            "InteractionDate": "max"
        }).reset_index()
        service_summary.columns = ["CustomerID","Total_Interactions","Resolved_Count","Last_Interaction_Date"]
        service_summary["Unresolved_Count"] = service_summary["Total_Interactions"] - service_summary["Resolved_Count"]

    # --- Online → recency feature
    # Expect: CustomerID, LastLoginDate, LoginFrequency, ServiceUsage
    online_summary = online_df.copy()
    if "LastLoginDate" in online_summary.columns:
        online_summary["LastLoginDate"] = pd.to_datetime(online_summary["LastLoginDate"], errors="coerce")
        cutoff = online_summary["LastLoginDate"].max()
        online_summary["Days_Since_LastLogin"] = (cutoff - online_summary["LastLoginDate"]).dt.days
        online_summary = online_summary.drop(columns=["LastLoginDate"])

    # --- Merge all to one row per customer
    master = churn_df.merge(demo_df, on="CustomerID", how="left")
    master = master.merge(service_summary, on="CustomerID", how="left")
    master = master.merge(online_summary, on="CustomerID", how="left")
    master = master.merge(txn_summary, on="CustomerID", how="left")

    # Optional: service recency feature (derived after merge)
    if "Last_Interaction_Date" in master.columns:
        cutoff_interact = pd.to_datetime(master["Last_Interaction_Date"], errors="coerce").max()
        master["Last_Interaction_Date"] = pd.to_datetime(master["Last_Interaction_Date"], errors="coerce")
        master["Days_Since_LastInteraction"] = (cutoff_interact - master["Last_Interaction_Date"]).dt.days

    return master

if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Cannot find {DATA_FILE} in the current folder.")
    master_df = build_master(DATA_FILE)
    master_df.to_csv(OUT_CSV, index=False)
    print(f"✅ Saved master dataset: {OUT_CSV} — shape: {master_df.shape}")
