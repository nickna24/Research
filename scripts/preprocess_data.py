#!/usr/bin/env python3
import argparse, os, json
import pandas as pd
import numpy as np

def fillna_series(s, strategy="median"):
    if s.dtype.kind in "biufc":
        if strategy == "median":
            return s.fillna(s.median())
        elif strategy == "mean":
            return s.fillna(s.mean())
        else:
            return s.fillna(0)
    else:
        mode = s.mode()
        return s.fillna(mode.iloc[0] if not mode.empty else "Unknown")

def one_hot(df, col, categories):
    df[col] = df[col].fillna(categories[0])
    out = pd.get_dummies(df[col], prefix=col)
    for c in categories:
        key = f"{col}_{c}"
        if key not in out.columns:
            out[key] = 0
    return out[[f"{col}_{c}" for c in categories]]

def standardize(df, cols, means=None, stds=None):
    df = df.copy()
    if means is None or stds is None:
        means = {c: float(df[c].mean()) for c in cols}
        stds = {c: float(df[c].std() or 1.0) for c in cols}
    for c in cols:
        s = stds[c] if stds[c] != 0 else 1.0
        df[c] = (df[c] - means[c]) / s
    return df, means, stds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    test_df  = pd.read_csv(args.test_csv)

    cat_domains = {"Sex": ["male", "female"], "Embarked": ["C", "Q", "S"]}
    num_cols = ["Age", "SibSp", "Parch", "Fare"]

    for col in ["Age", "Fare"]:
        train_df[col] = fillna_series(train_df[col])
        test_df[col] = fillna_series(test_df[col])
    for col in ["Embarked"]:
        train_df[col] = fillna_series(train_df[col])
        test_df[col] = fillna_series(test_df[col])

    # Build features
    def build_features(df, meta=None):
        pclass_oh = one_hot(df, "Pclass", [1, 2, 3])
        sex_oh = one_hot(df, "Sex", cat_domains["Sex"])
        emb_oh = one_hot(df, "Embarked", cat_domains["Embarked"])
        num_df = df[num_cols].copy()
        num_df, means, stds = standardize(num_df, num_cols)
        X_df = pd.concat([pclass_oh, sex_oh, emb_oh, num_df], axis=1)
        return X_df, means, stds

    X_train_df, means, stds = build_features(train_df)
    X_test_df, _, _ = build_features(test_df)

    feature_cols = list(X_train_df.columns)
    y_train = train_df["Survived"].astype(int).to_numpy().reshape(-1, 1)

    # Save preprocessed data
    np.save(os.path.join(args.out_dir, "train_proc.npy"), X_train_df.to_numpy().astype(np.float32))
    np.save(os.path.join(args.out_dir, "test_proc.npy"), X_test_df.to_numpy().astype(np.float32))
    np.save(os.path.join(args.out_dir, "labels.npy"), y_train.astype(np.float32))

    meta = {
        "cat_domains": cat_domains,
        "num_cols": num_cols,
        "num_means": means,
        "num_stds": stds,
        "feature_cols": feature_cols,
        "train_size": len(train_df),
        "test_size": len(test_df)
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Preprocessing complete. Saved to {args.out_dir}")

if __name__ == "__main__":
    main()
