from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from config import AppConfig
from preprocess import preprocess_dataset


QUARTER_MONTH = {
    1: 2,
    2: 5,
    3: 8,
    4: 11,
}


def _detect_encoding(path: Path) -> str:
    for enc in ("cp932", "shift_jis", "utf-8-sig", "utf-8"):
        try:
            pd.read_csv(path, encoding=enc, nrows=1)
            return enc
        except Exception:
            continue
    raise ValueError("Unable to detect encoding. Try specifying --encoding.")


def _parse_quarter_date(value: str | float | int | None) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value)
    match = re.search(r"(\d{4})年.*?(\d)四半期", text)
    if not match:
        return None
    year = int(match.group(1))
    quarter = int(match.group(2))
    month = QUARTER_MONTH.get(quarter)
    if not month:
        return None
    return f"{year:04d}-{month:02d}-01"


def _parse_build_year(value: str | float | int | None) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value)
    match = re.search(r"(\d{4})", text)
    if not match:
        return None
    return int(match.group(1))


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def convert_raw_to_training(raw_path: Path, output_path: Path, encoding: str | None) -> pd.DataFrame:
    enc = encoding or _detect_encoding(raw_path)
    df = pd.read_csv(raw_path, encoding=enc)

    # Filter to used condo transactions only.
    if "種類" in df.columns:
        df = df[df["種類"].astype(str).str.contains("中古マンション")]
    if "価格情報区分" in df.columns:
        df = df[df["価格情報区分"].astype(str).str.contains("成約")]

    df = df.copy()
    df["date"] = df.get("取引時期").apply(_parse_quarter_date)
    df["ward"] = df.get("市区町村名")
    df["area_m2"] = _to_numeric(df.get("面積（㎡）"))
    df["station_walk_min"] = _to_numeric(df.get("最寄駅：距離（分）"))
    df["far"] = _to_numeric(df.get("容積率（％）"))
    df["bcr"] = _to_numeric(df.get("建ぺい率（％）"))
    df["price_total_yen"] = _to_numeric(df.get("取引価格（総額）"))

    build_year = df.get("建築年").apply(_parse_build_year)
    trade_year = df["date"].str.slice(0, 4).astype("float")
    df["age_years"] = trade_year - build_year.astype("float")

    # lat/lon are not available in raw data. Keep NaN columns for now.
    df["lat"] = pd.NA
    df["lon"] = pd.NA

    cols = [
        "date",
        "ward",
        "age_years",
        "area_m2",
        "station_walk_min",
        "far",
        "bcr",
        "lat",
        "lon",
        "price_total_yen",
    ]
    out = df[cols]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw MLIT CSV to training CSV.")
    parser.add_argument(
        "--input",
        default="data/raw/Tokyo_20243_20252.csv",
        help="Raw MLIT CSV path.",
    )
    parser.add_argument(
        "--output",
        default="data/raw/transactions_mansion.csv",
        help="Output training CSV path.",
    )
    parser.add_argument(
        "--encoding",
        default=None,
        help="Override input encoding (cp932/shift_jis/utf-8).",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip running preprocess.py after conversion.",
    )
    args = parser.parse_args()

    raw_path = Path(args.input)
    output_path = Path(args.output)
    convert_raw_to_training(raw_path, output_path, args.encoding)

    if not args.skip_preprocess:
        cfg = AppConfig()
        preprocess_dataset(cfg)


if __name__ == "__main__":
    main()
