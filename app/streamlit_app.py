from __future__ import annotations
import os
import sys
from pathlib import Path
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from config import AppConfig
from inference import (
    load_high_precision_bundle,
    load_address_only_bundle,
    predict_high_precision,
    predict_address_only,
)


st.set_page_config(page_title="不動産価格推定", page_icon="🏢", layout="centered")

cfg = AppConfig()

header_path = Path(__file__).resolve().parents[1] / "imgs" / "header.png"
if header_path.exists():
    st.image(str(header_path), use_container_width=True)

st.title("中古マンション価格推定（円/㎡）")

property_type = st.selectbox("物件種別", ["中古マンション", "土地", "戸建"], index=0)
if property_type != "中古マンション":
    st.info("選択された物件種別は準備中です。中古マンションのみ推定できます。")
    st.stop()

mode = st.radio("推定モード", ["高精度", "住所のみ"], horizontal=True)
if mode == "住所のみ":
    st.warning("住所のみモードは精度が下がります。")

model_missing = []
if not os.path.exists(os.path.join(cfg.model_dir, "model_high_precision.joblib")):
    model_missing.append("高精度モデル")
if not os.path.exists(os.path.join(cfg.model_dir, "model_address_only.joblib")):
    model_missing.append("住所のみモデル")
if model_missing:
    st.warning("モデルが見つかりません。先に学習を実行してください。")

with st.form("estimate_form"):
    if mode == "高精度":
        ward = st.text_input("行政区（例: 港区, 横浜市西区）")
        age_years = st.number_input("築年数", min_value=0, max_value=100, value=10)
        area_m2 = st.number_input("専有面積（㎡）", min_value=0.0, max_value=200.0, value=50.0)
        station_walk_min = st.number_input("駅徒歩分", min_value=0, max_value=60, value=10)
        far = st.number_input("容積率", min_value=0.0, max_value=2000.0, value=400.0)
        bcr = st.number_input("建蔽率", min_value=0.0, max_value=200.0, value=60.0)
    else:
        address = st.text_input("住所（例: 東京都港区芝公園...）")
        age_years = st.number_input("築年数（任意）", min_value=0, max_value=100, value=0)
        area_m2 = st.number_input("専有面積（㎡・任意）", min_value=0.0, max_value=200.0, value=0.0)

    submitted = st.form_submit_button("推定する")

if submitted:
    try:
        if mode == "高精度":
            if not ward:
                st.error("行政区を入力してください。")
                st.stop()
            if area_m2 <= 0:
                st.error("専有面積を入力してください。")
                st.stop()
            bundle = load_high_precision_bundle(cfg)
            pred = predict_high_precision(
                ward=ward,
                age_years=age_years,
                area_m2=area_m2,
                station_walk_min=station_walk_min,
                far=far,
                bcr=bcr,
                bundle=bundle,
            )
        else:
            if not address:
                st.error("住所を入力してください。")
                st.stop()
            bundle = load_address_only_bundle(cfg)
            pred, _coords = predict_address_only(
                address=address,
                age_years=age_years if age_years > 0 else None,
                lat=None,
                lon=None,
                bundle=bundle,
                cfg=cfg,
            )

        st.success(f"推定価格（円/㎡）: {pred:,.0f} 円/㎡")
        if area_m2 and area_m2 > 0:
            total = pred * area_m2
            st.write(f"推定総額: {total:,.0f} 円")
        else:
            st.info("面積未入力のため総額は表示できません。専有面積を入力してください。")
    except Exception as exc:
        st.error(f"推定に失敗しました: {exc}")
