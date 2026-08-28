from tensorflow import keras
import sklearn as sk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from pathlib import Path
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# loading data
def load_data():
    csv_files = sorted(RAW_DATA_DIR.glob('*.csv'))
    combined_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    return combined_df

combined_df = load_data()
PROCESSED_DATA_DIR.mkdir(parents = True, exist_ok = True)
combined_df.to_csv(PROCESSED_DATA_DIR / 'combined_data.csv', index=False)

def more_features(df):
    model_df = combined_df[["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND"]]
    model_df["SETTLEMENT_DATE"] = pd.to_datetime(model_df["SETTLEMENT_DATE"], dayfirst = True)

    # creating a timestamp column
    model_df['timestamp'] = model_df['SETTLEMENT_DATE'] + pd.to_timedelta((model_df['SETTLEMENT_PERIOD'] - 1) * 30, unit='m')

    model_df["Month"] = (pd.to_datetime(model_df['SETTLEMENT_DATE']).dt.month)
    model_df["Month_Sin"] = np.sin(((pd.to_datetime(model_df['SETTLEMENT_DATE']).dt.month) - 1)/12)
    model_df["Month_Cos"] = np.cos(((pd.to_datetime(model_df['SETTLEMENT_DATE']).dt.month) - 1)/12)
    model_df["Year"] = pd.to_datetime(model_df['SETTLEMENT_DATE']).dt.year
    model_df["Day"] = pd.to_datetime(model_df['SETTLEMENT_DATE']).dt.day_name()
    model_df["Is_Weekend"] = [1 if day == "Saturday" or day == "Sunday" else 0 for day in model_df["Day"]]

    return model_df

model_df =  more_features(combined_df)
print(model_df.isna().sum())
print(model_df.duplicated(subset = ["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"]).sum())
print(model_df.dtypes)
print(model_df.sort_values(["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"]))
print(model_df.columns.tolist())
print(model_df["ND"].describe())
model_df = model_df.drop(columns = ["Day"])

model_df = model_df.sort_values(["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"]).reset_index(drop = True)

print(model_df.head())
model_df.to_csv(PROCESSED_DATA_DIR / "cleaned_data.csv", index = False)





    
                            
