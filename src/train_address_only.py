from __future__ import annotations
import json
import os
from pathlib import Path
import joblib
import pandas as pd
import lightgbm as lgb

from config import AppConfig
from features import build_ward_keep_set, apply_ward_mapping, time_split_last_days
from metrics import mae, mape
from preprocess import load_processed_or_preprocess


def _prepare_features(df: pd.DataFrame, ward_keep: set[str]) -> pd.DataFrame:
    df = df.copy()
    df["ward"] = apply_ward_mapping(df["ward"], ward_keep)
    return df


def train_address_only(cfg: AppConfig) -> dict:
    df = load_processed_or_preprocess(cfg)
    required = ["lat", "lon", "price_yen_per_m2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    train_df, valid_df, cutoff = time_split_last_days(df, "date", cfg.valid_last_days)
    if train_df.empty or valid_df.empty:
        raise ValueError("time split resulted in empty train/valid")

    ward_keep = build_ward_keep_set(train_df, "ward", cfg.ward_min_count)
    train_df = _prepare_features(train_df, ward_keep)
    valid_df = _prepare_features(valid_df, ward_keep)

    age_median = float(train_df["age_years"].median())
    train_df["age_years"] = train_df["age_years"].fillna(age_median)
    valid_df["age_years"] = valid_df["age_years"].fillna(age_median)

    feature_cols = ["ward", "lat", "lon", "age_years"]
    target = "price_yen_per_m2"

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target]
    X_valid = valid_df[feature_cols].copy()
    y_valid = valid_df[target]
    X_train["ward"] = X_train["ward"].astype("category")
    X_valid["ward"] = X_valid["ward"].astype("category")

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "random_state": cfg.random_state,
        "verbosity": -1,
    }

    train_ds = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=["ward"],
        free_raw_data=False,
    )
    valid_ds = lgb.Dataset(
        X_valid,
        label=y_valid,
        categorical_feature=["ward"],
        reference=train_ds,
        free_raw_data=False,
    )

    model = lgb.train(
        params,
        train_ds,
        valid_sets=[train_ds, valid_ds],
        valid_names=["train", "valid"],
        num_boost_round=5000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=200),
            lgb.log_evaluation(period=100),
        ],
    )

    pred = model.predict(X_valid, num_iteration=model.best_iteration)
    metrics = {
        "mae": mae(y_valid, pred),
        "mape_percent": mape(y_valid, pred),
        "best_iteration": int(model.best_iteration),
    }

    bundle = {
        "model": model,
        "features": feature_cols,
        "categorical": ["ward"],
        "ward_keep": sorted(list(ward_keep)),
        "ward_min_count": cfg.ward_min_count,
        "age_median": age_median,
        "target": target,
        "split_cutoff_date": str(cutoff.date()),
        "mode": "address_only",
    }

    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(cfg.model_dir, "model_address_only.joblib")
    joblib.dump(bundle, model_path)

    report = {
        "data_path": cfg.processed_data_path,
        "split_cutoff_date": str(cutoff.date()),
        "counts": {"train": int(len(train_df)), "valid": int(len(valid_df))},
        "metrics": metrics,
        "model_path": model_path,
    }

    report_path = os.path.join(cfg.model_dir, "train_report_address_only.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def main() -> None:
    cfg = AppConfig()
    report = train_address_only(cfg)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
