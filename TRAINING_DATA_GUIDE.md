# 学習データ仕様と学習手順ガイド

このドキュメントは、学習に使うCSVのデータ構造と、学習の実行方法を分かりやすくまとめたものです。

## 1. 学習データの概要

- 対象: **中古マンション（区分所有）の成約データのみ**
- 目的変数: **成約価格ベースの円/㎡（price_yen_per_m2）**
- 対象地域: 首都圏（東京23区＋横浜/川崎/さいたま/千葉/船橋/柏）

## 2. CSVファイルの置き場所

学習に使うデータは以下に配置します。

```
realestate-price-app/data/raw/transactions_mansion.csv
```

このファイルが無い場合、`preprocess.py` が**サンプルCSVを自動生成**します。

## 3. 必須カラム一覧（data_spec.md準拠）

| カラム名 | 内容 | 型の目安 |
| --- | --- | --- |
| date | 成約日 | YYYY-MM-DD 形式の文字列または日付 | 
| ward | 行政区（区/市） | 文字列 | 
| age_years | 築年数 | 数値 | 
| area_m2 | 専有面積（㎡） | 数値 | 
| station_walk_min | 最寄駅徒歩分 | 数値 | 
| far | 容積率 | 数値 | 
| bcr | 建蔽率 | 数値 | 
| lat | 緯度 | 数値 | 
| lon | 経度 | 数値 | 
| price_yen_per_m2 | 成約価格（円/㎡） | 数値 | 

### 補足
- `price_yen_per_m2` が無い場合は、`price_total_yen / area_m2` から自動計算します。
- 行政区は出現数が少ない場合に「その他」へ統合されます。
- 外れ値は 1%〜99% のウィンズライズを行います。

## 4. 学習用CSVの例

```csv
date,ward,age_years,area_m2,station_walk_min,far,bcr,lat,lon,price_total_yen
2022-05-10,港区,12,55.3,7,400,60,35.6581,139.7516,70000000
```

※ `price_yen_per_m2` は前処理で計算されます。

## 5. 学習の実行手順

### ① 前処理（学習データ作成）

```bash
python src/preprocess.py
```

- `data/raw/transactions_mansion.csv` を読み込み
- 必須カラムの検証
- `price_yen_per_m2` の作成
- 外れ値処理
- `data/processed/mansion_train.csv` へ保存

### ② 高精度モデルの学習

```bash
python src/train_high_precision.py
```

### ③ 住所のみモデルの学習

```bash
python src/train_address_only.py
```

学習後、モデルは `models/` に保存されます。

## 6. よくある注意点

- 実データを使う場合は **必ず上記の必須カラムを揃えて配置**してください。
- サンプルデータはランダム生成のため精度評価には使えません。
- 学習の評価指標は MAE（主）と MAPE（補助）です。

