# src/config.py
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    raw_data_path: str = "data/raw/transactions_mansion.csv"
    processed_data_path: str = "data/processed/mansion_train.csv"
    model_dir: str = "models"
    cache_db_path: str = "data/cache/geocode_cache.sqlite"

    # 時系列分割：最新1年をvalid（date列が必要）
    valid_last_days: int = 365

    # 外れ値処理（winsorize）
    winsor_low: float = 0.01
    winsor_high: float = 0.99

    # wardの出現数が少ないカテゴリは「その他」
    ward_min_count: int = 200

    random_state: int = 42

    gsi_endpoint: str = "https://msearch.gsi.go.jp/address-search/AddressSearch"
    nominatim_endpoint: str = "https://nominatim.openstreetmap.org/search"


# Backward compatibility for existing modules
TrainConfig = AppConfig
