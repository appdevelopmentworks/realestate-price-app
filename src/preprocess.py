from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

from config import AppConfig
from features import winsorize_series


REQUIRED_COLUMNS = [
    "date",
    "ward",
    "age_years",
    "area_m2",
    "station_walk_min",
    "far",
    "bcr",
    "lat",
    "lon",
]


def generate_sample_csv(path: str, rows: int = 500) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    wards = [
        "千代田区",
        "港区",
        "新宿区",
        "渋谷区",
        "中央区",
        "横浜市西区",
        "川崎市中原区",
        "さいたま市大宮区",
        "千葉市中央区",
        "船橋市",
        "柏市",
    ]
    start = np.datetime64("2018-01-01")
    dates = start + rng.integers(0, 365 * 6, size=rows).astype("timedelta64[D]")
    area = rng.normal(55, 12, size=rows).clip(20, 120)
    price_per_m2 = rng.normal(950_000, 200_000, size=rows).clip(350_000, 1_800_000)
    total_price = (price_per_m2 * area).astype(int)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates).strftime("%Y-%m-%d"),
            "ward": rng.choice(wards, size=rows),
            "age_years": rng.integers(1, 45, size=rows),
            "area_m2": area,
            "station_walk_min": rng.integers(2, 20, size=rows),
            "far": rng.integers(200, 800, size=rows),
            "bcr": rng.integers(40, 80, size=rows),
            "lat": rng.normal(35.68, 0.12, size=rows),
            "lon": rng.normal(139.76, 0.15, size=rows),
            "price_total_yen": total_price,
        }
    )
    df.to_csv(path, index=False)


def load_raw_data(raw_path: str) -> pd.DataFrame:
    if not os.path.exists(raw_path):
        generate_sample_csv(raw_path)
    df = pd.read_csv(raw_path)
    return df


def ensure_price_per_m2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "price_yen_per_m2" not in df.columns:
        if "price_total_yen" not in df.columns or "area_m2" not in df.columns:
            raise ValueError("price_yen_per_m2 or price_total_yen and area_m2 are required")
        df["price_yen_per_m2"] = df["price_total_yen"] / df["area_m2"]
    return df


def preprocess_dataset(cfg: AppConfig) -> pd.DataFrame:
    df = load_raw_data(cfg.raw_data_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = ensure_price_per_m2(df)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["price_yen_per_m2"] = winsorize_series(
        df["price_yen_per_m2"], cfg.winsor_low, cfg.winsor_high
    )
    Path(cfg.processed_data_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.processed_data_path, index=False)
    return df


def load_processed_or_preprocess(cfg: AppConfig) -> pd.DataFrame:
    if os.path.exists(cfg.processed_data_path):
        df = pd.read_csv(cfg.processed_data_path, parse_dates=["date"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df.dropna(subset=["date"])
    return preprocess_dataset(cfg)


def main() -> None:
    cfg = AppConfig()
    df = preprocess_dataset(cfg)
    print(f"processed rows: {len(df)}")


if __name__ == "__main__":
    main()
