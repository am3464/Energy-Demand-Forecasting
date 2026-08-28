import pandas as pd
from config import PROCESSED_DATA_DIR

def train_val_test_split(df):
    train_df = df[df["Year"] <= 2022].copy()
    validation_df = df[df["Year"] == 2023].copy()
    test_df = df[df["Year"] > 2023].copy()

    print(train_df.head())
    print(validation_df.head())
    print(test_df.head())

    train_df.to_csv(PROCESSED_DATA_DIR / "train_sb.csv", index = False)
    validation_df.to_csv(PROCESSED_DATA_DIR / "validation_sb.csv", index = False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test_sb.csv", index = False)



def add_previous_week(df):
    df["SETTLEMENT_DATE"] = pd.to_datetime(df["SETTLEMENT_DATE"])
    past = df[["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND"]].copy()
    past["SETTLEMENT_DATE"] += pd.Timedelta(days = 7)
    past = past.rename(columns = {"ND":"ND_7prior"})
    return df.merge(past, on = ["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"], how = "left")

def main():
    model_df = pd.read_csv(PROCESSED_DATA_DIR / "cleaned_data.csv")
    model_df = add_previous_week(model_df)
    train_val_test_split(model_df)

if __name__ == "__main__":
    main()


