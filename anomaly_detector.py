import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from rapidfuzz import fuzz

warnings.filterwarnings("ignore", category=UserWarning)


DATE_OLD_THRESHOLD_DAYS = 90     
Z_SCORE_THRESHOLD       = 3.0     
ISOLATION_CONTAMINATION = 0.10    
DUPLICATE_FUZZY_THRESH  = 90      




def _flag_math_errors(df: pd.DataFrame) -> pd.Series:
    
    both_known = df["grand_total"].notna() & df["calculated_total"].notna()
    diff       = (df["grand_total"] - df["calculated_total"]).abs()
    return both_known & (diff > 0.02)


def _flag_date_issues(df: pd.DataFrame) -> pd.DataFrame:
    
    today = datetime.today().date()
    cutoff = today - timedelta(days=DATE_OLD_THRESHOLD_DAYS)

    future_mask = pd.Series(False, index=df.index)
    old_mask    = pd.Series(False, index=df.index)

    for idx, raw_date in df["invoice_date"].items():
        try:
            d = datetime.strptime(raw_date, "%Y-%m-%d").date()
            future_mask.at[idx] = d > today
            old_mask.at[idx]    = d < cutoff
        except (ValueError, TypeError):
            pass    

    return pd.DataFrame({"date_future": future_mask, "date_old": old_mask})




def _compute_z_scores(df: pd.DataFrame) -> pd.Series:
    
    z_scores = pd.Series(np.nan, index=df.index)

    for vendor, group in df.groupby("vendor_name"):
        amounts = group["grand_total"].dropna()
        if len(amounts) < 2:
            
            continue
        mean = amounts.mean()
        std  = amounts.std(ddof=0)
        if std == 0:
            continue
        z_scores.loc[amounts.index] = (amounts - mean).abs() / std

    return z_scores




def _build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
    
    rows  = []
    idxs  = []
    today = datetime.today().date()

    for idx, row in df.iterrows():
        try:
            total      = float(row["grand_total"])
            n_items    = int(row["num_line_items"])
            d          = datetime.strptime(row["invoice_date"], "%Y-%m-%d").date()
            dow        = d.weekday()
            days_since = (today - d).days
        except (ValueError, TypeError):
            continue    

        rows.append([total, n_items, dow, days_since])
        idxs.append(idx)

    return np.array(rows, dtype=float), idxs


def _run_isolation_forest(df: pd.DataFrame) -> pd.Series:
    
    X, valid_idxs = _build_feature_matrix(df)
    iso_flag = pd.Series(False, index=df.index)

    if len(X) < 5:
        print("  ⚠ Not enough valid rows for Isolation Forest (need ≥ 5).")
        return iso_flag

    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=ISOLATION_CONTAMINATION,
        random_state=42,
    )
    preds = model.fit_predict(X_scaled)     

    for idx, pred in zip(valid_idxs, preds):
        iso_flag.at[idx] = pred == -1

    return iso_flag




def _find_duplicates(df: pd.DataFrame) -> pd.Series:
    
    dup_flag = pd.Series(False, index=df.index)
    rows     = df[["invoice_number", "grand_total"]].copy()

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            idx_i = rows.index[i]
            idx_j = rows.index[j]

            num_i = str(rows.at[idx_i, "invoice_number"])
            num_j = str(rows.at[idx_j, "invoice_number"])
            amt_i = rows.at[idx_i, "grand_total"]
            amt_j = rows.at[idx_j, "grand_total"]

            
            if pd.isna(amt_i) or pd.isna(amt_j):
                continue
            if abs(amt_i - amt_j) > 0.01:
                continue

            similarity = fuzz.ratio(num_i, num_j)
            if similarity >= DUPLICATE_FUZZY_THRESH:
                dup_flag.at[idx_i] = True
                dup_flag.at[idx_j] = True

    return dup_flag


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    #
    df["is_anomaly"]     = False
    df["anomaly_reason"] = ""
    df["z_score"]        = np.nan

   
    print("  [1/4] Checking mathematical integrity …")
    math_err = _flag_math_errors(df)
    _annotate(df, math_err, "math_error")

    
    print("  [2/4] Checking date logic …")
    date_flags = _flag_date_issues(df)
    _annotate(df, date_flags["date_future"], "future_date")
    _annotate(df, date_flags["date_old"],    "old_date (>90 days)")

    0
    print("  [3/4] Computing per-vendor Z-scores …")
    z_scores = _compute_z_scores(df)
    df["z_score"] = z_scores.round(2)
    z_outlier = z_scores > Z_SCORE_THRESHOLD
    _annotate(df, z_outlier.fillna(False), f"z_score_outlier (>{Z_SCORE_THRESHOLD}σ)")

    
    print("  [4a/4] Training Isolation Forest …")
    iso_flag = _run_isolation_forest(df)
    _annotate(df, iso_flag, "isolation_forest")

    
    print("  [4b/4] Running duplicate detection …")
    dup_flag = _find_duplicates(df)
    _annotate(df, dup_flag, "possible_duplicate")
    df["is_anomaly"] = df["anomaly_reason"].str.len() > 0

    total = df["is_anomaly"].sum()
    print(f"\n  ✅  Anomaly detection complete — {total}/{len(df)} invoices flagged.")
    return df


def _annotate(df: pd.DataFrame, mask: pd.Series, reason: str) -> None:
    for idx in df[mask].index:
        existing = df.at[idx, "anomaly_reason"]
        df.at[idx, "anomaly_reason"] = (existing + ", " + reason) if existing else reason
