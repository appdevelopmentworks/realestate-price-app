# src/features.py
from __future__ import annotations
import re
import pandas as pd
import numpy as np

def winsorize_series(s: pd.Series, low_q: float, high_q: float) -> pd.Series:
    lo = s.quantile(low_q)
    hi = s.quantile(high_q)
    return s.clip(lower=lo, upper=hi)

def normalize_ward(df: pd.DataFrame, ward_col: str, min_count: int) -> pd.DataFrame:
    df = df.copy()
    counts = df[ward_col].value_counts(dropna=False)
    rare = counts[counts < min_count].index
    df[ward_col] = df[ward_col].where(~df[ward_col].isin(rare), "その他")
    df[ward_col] = df[ward_col].fillna("不明")
    return df


def build_ward_keep_set(df: pd.DataFrame, ward_col: str, min_count: int) -> set[str]:
    counts = df[ward_col].value_counts(dropna=False)
    keep = counts[counts >= min_count].index.astype(str).tolist()
    return set(keep)


def apply_ward_mapping(s: pd.Series, keep: set[str]) -> pd.Series:
    s = s.astype("object")
    s = s.where(s.isin(list(keep)), "その他")
    s = s.fillna("不明")
    return s


def extract_ward(address: str | None) -> str:
    if not address:
        return "不明"
    match = re.search(r"([一-龥0-9A-Za-z]{1,6}[区市])", address)
    if not match:
        return "不明"
    return match.group(1)

def add_missing_flags(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[f"{c}__missing"] = df[c].isna().astype("int8")
    return df

def fill_by_ward_median(
    df: pd.DataFrame,
    ward_col: str,
    target_cols: list[str],
) -> pd.DataFrame:
    """
    far/bcrなど、取れないケースは行政区中央値で補完（MVP方針）
    """
    df = df.copy()
    for c in target_cols:
        med = df.groupby(ward_col)[c].median()
        df[c] = df[c].fillna(df[ward_col].map(med))
        df[c] = df[c].fillna(df[c].median())
    return df


def apply_ward_medians(
    df: pd.DataFrame,
    ward_col: str,
    target_cols: list[str],
    ward_medians: dict[str, dict[str, float]],
    global_medians: dict[str, float],
) -> pd.DataFrame:
    df = df.copy()
    for c in target_cols:
        med_map = ward_medians.get(c, {})
        df[c] = df[c].fillna(df[ward_col].map(med_map))
        df[c] = df[c].fillna(global_medians.get(c, df[c].median()))
    return df

def time_split_last_days(df: pd.DataFrame, date_col: str, last_days: int):
    """
    date_col の最大日付から last_days 以内をvalid、それ以外をtrain
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    max_date = df[date_col].max()
    cutoff = max_date - pd.Timedelta(days=last_days)
    train_df = df[df[date_col] < cutoff].copy()
    valid_df = df[df[date_col] >= cutoff].copy()
    return train_df, valid_df, cutoff
