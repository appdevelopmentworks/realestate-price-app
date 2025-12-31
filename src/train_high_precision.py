from __future__ import annotations
import json
import os
from pathlib import Path
import joblib
import pandas as pd
import lightgbm as lgb

from config import AppConfig
from features import (
    build_ward_keep_set,
    apply_ward_mapping,
    apply_ward_medians,
    time_split_last_days,
)
from metrics import mae, mape
from preprocess import load_processed_or_preprocess


def _prepare_features(df: pd.DataFrame, ward_keep: set[str]) -> pd.DataFrame:
    df = df.copy()
    df["ward"] = apply_ward_mapping(df["ward"], ward_keep)
    return df


def train_high_precision(cfg: AppConfig) -> dict:
    df = load_processed_or_preprocess(cfg)
    if "price_yen_per_m2" not in df.columns:
        raise ValueError("price_yen_per_m2 is required")

    train_df, valid_df, cutoff = time_split_last_days(df, "date", cfg.valid_last_days)
    if train_df.empty or valid_df.empty:
        raise ValueError("time split resulted in empty train/valid")

    ward_keep = build_ward_keep_set(train_df, "ward", cfg.ward_min_count)
    train_df = _prepare_features(train_df, ward_keep)
    valid_df = _prepare_features(valid_df, ward_keep)

    ward_medians = train_df.groupby("ward")[["far", "bcr"]].median().to_dict()

    num_fill_cols = ["station_walk_min", "age_years", "area_m2", "far", "bcr"]
    num_medians = {c: float(train_df[c].median()) for c in num_fill_cols}

    train_df = apply_ward_medians(
        train_df,
        "ward",
        ["far", "bcr"],
        ward_medians,
        num_medians,
    )
    valid_df = apply_ward_medians(
        valid_df,
        "ward",
        ["far", "bcr"],
        ward_medians,
        num_medians,
    )
    for c in ["station_walk_min", "age_years", "area_m2"]:
        train_df[c] = train_df[c].fillna(num_medians[c])
        valid_df[c] = valid_df[c].fillna(num_medians[c])

    feature_cols = [
        "ward",
        "age_years",
        "area_m2",
        "station_walk_min",
        "far",
        "bcr",
    ]
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
        "ward_medians": ward_medians,
        "fill_medians": num_medians,
        "target": target,
        "split_cutoff_date": str(cutoff.date()),
        "mode": "high_precision",
    }

    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(cfg.model_dir, "model_high_precision.joblib")
    joblib.dump(bundle, model_path)

    report = {
        "data_path": cfg.processed_data_path,
        "split_cutoff_date": str(cutoff.date()),
        "counts": {"train": int(len(train_df)), "valid": int(len(valid_df))},
        "metrics": metrics,
        "model_path": model_path,
    }

    report_path = os.path.join(cfg.model_dir, "train_report_high_precision.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def main() -> None:
    cfg = AppConfig()
    report = train_high_precision(cfg)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
