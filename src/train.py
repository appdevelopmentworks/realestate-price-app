# src/train.py
from __future__ import annotations

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import lightgbm as lgb

from config import TrainConfig
from features import (
    winsorize_series,
    normalize_ward,
    add_missing_flags,
    fill_by_ward_median,
    time_split_last_days,
)
from metrics import mae, mape


def prepare_high_precision(df: pd.DataFrame, cfg: TrainConfig):
    # 目的変数 winsorize
    df = df.copy()
    df["price_yen_per_m2"] = winsorize_series(
        df["price_yen_per_m2"], cfg.winsor_low, cfg.winsor_high
    )

    # wardの希少カテゴリ統合
    df = normalize_ward(df, "ward", cfg.ward_min_count)

    # far/bcr欠損の補完（行政区中央値→全体中央値）
    df = add_missing_flags(df, ["far", "bcr", "station_walk_min", "age_years", "area_m2"])
    df = fill_by_ward_median(df, "ward", ["far", "bcr"])

    # 数値の補完（最低限）
    for c in ["station_walk_min", "age_years", "area_m2"]:
        df[c] = df[c].fillna(df[c].median())

    features = [
        "ward",
        "age_years",
        "area_m2",
        "station_walk_min",
        "far",
        "bcr",
        "far__missing",
        "bcr__missing",
        "station_walk_min__missing",
        "age_years__missing",
        "area_m2__missing",
    ]
    target = "price_yen_per_m2"
    return df, features, target


def prepare_address_only(df: pd.DataFrame, cfg: TrainConfig):
    df = df.copy()
    df["price_yen_per_m2"] = winsorize_series(
        df["price_yen_per_m2"], cfg.winsor_low, cfg.winsor_high
    )

    df = normalize_ward(df, "ward", cfg.ward_min_count)

    # 必須：lat/lon が無いと学習不可
    required = ["lat", "lon"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"address_only model requires columns: {missing}")

    df = add_missing_flags(df, ["age_years"])
    # 築年数は任意入力想定 → 未入力は中央値
    df["age_years"] = df["age_years"].fillna(df["age_years"].median())

    features = ["ward", "lat", "lon", "age_years", "age_years__missing"]
    target = "price_yen_per_m2"
    return df, features, target


def train_lgbm_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    categorical_features: list[str],
    seed: int,
):
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
        "random_state": seed,
        "verbosity": -1,
    }

    train_ds = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )
    valid_ds = lgb.Dataset(
        X_valid,
        label=y_valid,
        categorical_feature=categorical_features,
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
    return model


def evaluate_and_pack(model, X_valid, y_valid):
    pred = model.predict(X_valid, num_iteration=model.best_iteration)
    return {
        "mae": mae(y_valid, pred),
        "mape_percent": mape(y_valid, pred),
        "best_iteration": int(model.best_iteration),
    }


def main():
    cfg = TrainConfig()
    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg.data_path)
    if "date" not in df.columns:
        raise ValueError("CSV must include 'date' column for time-series split.")
    if "price_yen_per_m2" not in df.columns:
        raise ValueError("CSV must include 'price_yen_per_m2' as target.")

    # 時系列split（最新1年をvalid）
    train_df, valid_df, cutoff = time_split_last_days(df, "date", cfg.valid_last_days)
    if len(train_df) < 1000 or len(valid_df) < 300:
        print(f"[WARN] small split sizes: train={len(train_df)}, valid={len(valid_df)}")

    # =========================
    # 1) 高精度モデル
    # =========================
    train_hp, hp_feats, target = prepare_high_precision(train_df, cfg)
    valid_hp, _, _ = prepare_high_precision(valid_df, cfg)

    Xtr = train_hp[hp_feats]
    ytr = train_hp[target]
    Xva = valid_hp[hp_feats]
    yva = valid_hp[target]

    cat = ["ward"]
    model_hp = train_lgbm_regressor(Xtr, ytr, Xva, yva, categorical_features=cat, seed=cfg.random_state)
    metrics_hp = evaluate_and_pack(model_hp, Xva, yva)

    hp_bundle = {
        "model": model_hp,
        "features": hp_feats,
        "categorical": cat,
        "ward_min_count": cfg.ward_min_count,
        "target": target,
        "split_cutoff_date": str(cutoff.date()),
        "mode": "high_precision",
    }
    joblib.dump(hp_bundle, os.path.join(cfg.model_dir, "model_high_precision.joblib"))

    # =========================
    # 2) 住所のみ専用モデル
    # =========================
    train_ad, ad_feats, target = prepare_address_only(train_df, cfg)
    valid_ad, _, _ = prepare_address_only(valid_df, cfg)

    Xtr2 = train_ad[ad_feats]
    ytr2 = train_ad[target]
    Xva2 = valid_ad[ad_feats]
    yva2 = valid_ad[target]

    cat2 = ["ward"]
    model_ad = train_lgbm_regressor(Xtr2, ytr2, Xva2, yva2, categorical_features=cat2, seed=cfg.random_state)
    metrics_ad = evaluate_and_pack(model_ad, Xva2, yva2)

    ad_bundle = {
        "model": model_ad,
        "features": ad_feats,
        "categorical": cat2,
        "ward_min_count": cfg.ward_min_count,
        "target": target,
        "split_cutoff_date": str(cutoff.date()),
        "mode": "address_only",
    }
    joblib.dump(ad_bundle, os.path.join(cfg.model_dir, "model_address_only.joblib"))

    # =========================
    # レポート保存
    # =========================
    report = {
        "data_path": cfg.data_path,
        "split_cutoff_date": str(cutoff.date()),
        "counts": {"train": int(len(train_df)), "valid": int(len(valid_df))},
        "high_precision": metrics_hp,
        "address_only": metrics_ad,
    }
    with open(os.path.join(cfg.model_dir, "train_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== TRAIN DONE ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
