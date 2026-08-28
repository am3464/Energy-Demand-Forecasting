import json
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import numpy as np
from config import PROCESSED_DATA_DIR, FIGURES_DIR, METRICS_DIR

def seasonal_predictions_previous_week(model_df):
    model_df["SETTLEMENT_DATE"] = pd.to_datetime(model_df["SETTLEMENT_DATE"])
    past = model_df[["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND"]].copy()
    past["SETTLEMENT_DATE"] += pd.Timedelta(days = 7)
    past = past.rename(columns = {"ND":"ND_7prior"})
    model_df = model_df.merge(
        past, on = ["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"], how = "left"
    )
    return model_df

def seasonal_pred_error(baseline_eval):
    baseline_mae = mean_absolute_error(baseline_eval["ND"], baseline_eval["ND_7prior"])
    baseline_rmse = np.sqrt(mean_squared_error(baseline_eval["ND"], baseline_eval["ND_7prior"]))
    baseline_mape = mean_absolute_percentage_error(baseline_eval["ND"], baseline_eval["ND_7prior"])
    return baseline_mae, baseline_rmse, baseline_mape

def plot_seasonal_baseline(number_of_points, test_set):
    actual = test_set["ND"].values[:number_of_points]
    predicted = test_set["ND_7prior"].values[:number_of_points]
    plt.figure(figsize = (12,5))
    plt.plot(actual, label = "Actual demand")
    plt.plot(predicted, label = "Seasonal baseline prediction", color = "deeppink")
    plt.xlabel("Half-hourly observation")
    plt.ylabel("Demand (MW)")
    plt.title("Seasonal baseline test forecast: two weeks")
    plt.legend()
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents = True, exist_ok = True)
    plt.savefig(FIGURES_DIR / "seasonal_baseline_forecast.png", dpi = 300)

def main():
    model_df = pd.read_csv(PROCESSED_DATA_DIR / "cleaned_data.csv")
    model_df = seasonal_predictions_previous_week(model_df)
    test_set = model_df[model_df["Year"] >= 2024].copy()
    baseline_eval = test_set.dropna(subset = ["ND_7prior"]).copy()
    baseline_mae_tes, baseline_rmse_tes, baseline_mape_tes = seasonal_pred_error(baseline_eval)
    results = {
        "model":"weekly seasonal baseline",
        "metric":"MAE",
        "MAE_MW_test":float(baseline_mae_tes),
        "RMSE_MW_test":float(baseline_rmse_tes),
        "MAPE_MW_test":float(baseline_mape_tes)
    }
    METRICS_DIR.mkdir(parents = True, exist_ok = True)
    with open(METRICS_DIR / "experiment_results_baseline.json", "w") as file:
        json.dump(results, file, indent = 4)
    plot_seasonal_baseline(7*48*2, baseline_eval)

if __name__ == "__main__":
    main()
