# Campaign Usage Prediction (User Behavior + Campaign Interaction)

## Overview
This project predicts whether a user is likely to benefit from / use a cashback campaign based on user behavior features and campaign parameters. A Logistic Regression model is trained on historical anonymized data, then used to simulate a new campaign scenario and estimate how many users would qualify.

## Tech Stack
- Python
- pandas
- scikit-learn

## Input Files (not included)
Due to confidentiality, the datasets are anonymized and not included in this repository.

Expected files in the project root:
- `Dataset1.xlsx` (user-level features)
  - Example columns: `UNIQUE_ID`, `CUS_AGE`, `CATEGORY_ID`, `TUTAR`, `CASHBACK_AMT`, `CASHBACK_STATUS`
- `Dataset2.xlsx` (campaign configuration)
  - Example columns: `CATEGORY_ID`, `CAMPAIGN_ID`, `MIN_AMOUNT`, `MAX_CASHBACK`, `CASHBACK_RATE`

## How It Works
1. Merge user and campaign tables by `CATEGORY_ID`
2. Fill missing campaign fields with 0 (no-campaign cases)
3. Train Logistic Regression with standardized features
4. Evaluate accuracy on a test split
5. Apply a new campaign configuration and estimate how many users qualify (grouped by `UNIQUE_ID`)

## How to Run
```bash
pip install -r requirements.txt
python main.py
