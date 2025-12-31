# Real Estate Price Estimator (JP)

## Overview
Estimate transaction price per square meter (JPY/m2) for used condominiums in the Tokyo metro area.

## Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training Commands
```bash
python src/preprocess.py
python src/train_high_precision.py
python src/train_address_only.py
```

## Evaluation Command
```bash
python src/evaluate.py --mode all
```

## Streamlit App
```bash
streamlit run app/streamlit_app.py
```

## Notes
- Address-only mode is less accurate by design.
- If `data/raw/transactions_mansion.csv` is missing, a sample CSV is generated.
