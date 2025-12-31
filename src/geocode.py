from __future__ import annotations
import os
import time
from typing import Optional
import requests

from config import AppConfig
from cache import get_cached, set_cached


def _geocode_gsi(address: str, endpoint: str, timeout: float) -> Optional[tuple[float, float]]:
    params = {"q": address}
    resp = requests.get(endpoint, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return None
    coords = data[0].get("geometry", {}).get("coordinates")
    if not coords or len(coords) < 2:
        return None
    lon, lat = coords[0], coords[1]
    return float(lat), float(lon)


def _geocode_nominatim(address: str, endpoint: str, timeout: float) -> Optional[tuple[float, float]]:
    user_agent = os.getenv("NOMINATIM_USER_AGENT", "realestate-price-app")
    params = {"format": "json", "q": address}
    headers = {"User-Agent": user_agent}
    resp = requests.get(endpoint, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return None
    lat = data[0].get("lat")
    lon = data[0].get("lon")
    if lat is None or lon is None:
        return None
    return float(lat), float(lon)


def geocode_address(address: str, cfg: AppConfig, timeout: float = 5.0) -> Optional[tuple[float, float, str]]:
    cached = get_cached(cfg.cache_db_path, address)
    if cached:
        lat, lon, provider = cached
        return lat, lon, provider

    time.sleep(0.3)
    try:
        result = _geocode_gsi(address, cfg.gsi_endpoint, timeout)
        if result:
            lat, lon = result
            set_cached(cfg.cache_db_path, address, lat, lon, "gsi")
            return lat, lon, "gsi"
    except Exception:
        result = None

    time.sleep(0.3)
    try:
        result = _geocode_nominatim(address, cfg.nominatim_endpoint, timeout)
        if result:
            lat, lon = result
            set_cached(cfg.cache_db_path, address, lat, lon, "nominatim")
            return lat, lon, "nominatim"
    except Exception:
        return None

    return None
