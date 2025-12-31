from __future__ import annotations
import argparse
import json
import os
import pandas as pd
import joblib

from config import AppConfig
from features import apply_ward_mapping, apply_ward_medians, time_split_last_days
from metrics import mae, mape
from preprocess import load_processed_or_preprocess


def evaluate_high_precision(cfg: AppConfig) -> dict:
    model_path = os.path.join(cfg.model_dir, "model_high_precision.joblib")
    bundle = joblib.load(model_path)
    df = load_processed_or_preprocess(cfg)

    train_df, valid_df, cutoff = time_split_last_days(df, "date", cfg.valid_last_days)

    ward_keep = set(bundle["ward_keep"])
    valid_df["ward"] = apply_ward_mapping(valid_df["ward"], ward_keep)

    valid_df = apply_ward_medians(
        valid_df,
        "ward",
        ["far", "bcr"],
        bundle.get("ward_medians", {}),
        bundle.get("fill_medians", {}),
    )
    for c in ["station_walk_min", "age_years", "area_m2"]:
        valid_df[c] = valid_df[c].fillna(bundle.get("fill_medians", {}).get(c, valid_df[c].median()))

    X_valid = valid_df[bundle["features"]].copy()
    y_valid = valid_df[bundle["target"]]
    X_valid["ward"] = X_valid["ward"].astype("category")

    model = bundle["model"]
    pred = model.predict(X_valid, num_iteration=model.best_iteration)

    return {
        "mode": "high_precision",
        "split_cutoff_date": str(cutoff.date()),
        "mae": mae(y_valid, pred),
        "mape_percent": mape(y_valid, pred),
    }


def evaluate_address_only(cfg: AppConfig) -> dict:
    model_path = os.path.join(cfg.model_dir, "model_address_only.joblib")
    bundle = joblib.load(model_path)
    df = load_processed_or_preprocess(cfg)

    train_df, valid_df, cutoff = time_split_last_days(df, "date", cfg.valid_last_days)

    ward_keep = set(bundle["ward_keep"])
    valid_df["ward"] = apply_ward_mapping(valid_df["ward"], ward_keep)
    valid_df["age_years"] = valid_df["age_years"].fillna(bundle.get("age_median", valid_df["age_years"].median()))

    X_valid = valid_df[bundle["features"]].copy()
    y_valid = valid_df[bundle["target"]]
    X_valid["ward"] = X_valid["ward"].astype("category")

    model = bundle["model"]
    pred = model.predict(X_valid, num_iteration=model.best_iteration)

    return {
        "mode": "address_only",
        "split_cutoff_date": str(cutoff.date()),
        "mae": mae(y_valid, pred),
        "mape_percent": mape(y_valid, pred),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["high_precision", "address_only", "all"], default="all")
    args = parser.parse_args()

    cfg = AppConfig()
    results = []
    if args.mode in ("high_precision", "all"):
        results.append(evaluate_high_precision(cfg))
    if args.mode in ("address_only", "all"):
        results.append(evaluate_address_only(cfg))

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
