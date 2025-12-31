from __future__ import annotations
import os
from typing import Optional
import joblib
import pandas as pd

from config import AppConfig
from features import apply_ward_mapping, apply_ward_medians, extract_ward
from geocode import geocode_address


def load_bundle(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model not found: {model_path}")
    return joblib.load(model_path)


def _predict(model, X: pd.DataFrame) -> float:
    pred = model.predict(X, num_iteration=getattr(model, "best_iteration", None))
    return float(pred[0])


def predict_high_precision(
    ward: str,
    age_years: Optional[float],
    area_m2: Optional[float],
    station_walk_min: Optional[float],
    far: Optional[float],
    bcr: Optional[float],
    bundle: dict,
) -> float:
    ward_keep = set(bundle["ward_keep"])
    df = pd.DataFrame(
        {
            "ward": [ward],
            "age_years": [age_years],
            "area_m2": [area_m2],
            "station_walk_min": [station_walk_min],
            "far": [far],
            "bcr": [bcr],
        }
    )

    df["ward"] = apply_ward_mapping(df["ward"], ward_keep)
    df = apply_ward_medians(
        df,
        "ward",
        ["far", "bcr"],
        bundle.get("ward_medians", {}),
        bundle.get("fill_medians", {}),
    )

    for c in ["station_walk_min", "age_years", "area_m2"]:
        if df[c].isna().any():
            df[c] = df[c].fillna(bundle.get("fill_medians", {}).get(c, df[c].median()))

    model = bundle["model"]
    X = df[bundle["features"]].copy()
    X["ward"] = X["ward"].astype("category")
    return _predict(model, X)


def predict_address_only(
    address: str,
    age_years: Optional[float],
    lat: Optional[float],
    lon: Optional[float],
    bundle: dict,
    cfg: Optional[AppConfig] = None,
) -> tuple[float, Optional[tuple[float, float]]]:
    if cfg is None:
        cfg = AppConfig()

    ward = extract_ward(address)
    ward_keep = set(bundle["ward_keep"])
    ward = apply_ward_mapping(pd.Series([ward]), ward_keep).iloc[0]

    coords = None
    if lat is None or lon is None:
        result = geocode_address(address, cfg)
        if not result:
            raise ValueError("failed to geocode address")
        lat, lon, _provider = result
        coords = (lat, lon)

    if age_years is None:
        age_years = float(bundle.get("age_median", 0.0))

    df = pd.DataFrame(
        {
            "ward": [ward],
            "lat": [lat],
            "lon": [lon],
            "age_years": [age_years],
        }
    )

    model = bundle["model"]
    X = df[bundle["features"]].copy()
    X["ward"] = X["ward"].astype("category")
    pred = _predict(model, X)
    return pred, coords


def load_high_precision_bundle(cfg: AppConfig) -> dict:
    path = os.path.join(cfg.model_dir, "model_high_precision.joblib")
    return load_bundle(path)


def load_address_only_bundle(cfg: AppConfig) -> dict:
    path = os.path.join(cfg.model_dir, "model_address_only.joblib")
    return load_bundle(path)
