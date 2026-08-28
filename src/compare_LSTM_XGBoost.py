import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json
from config import PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR

def create_dataset(X, y, time_steps=7*48):
    Xs, ys = [], []
    for i in range(len(X) - time_steps - 48 + 1):
        v = X.iloc[i : (i + time_steps)].to_numpy()
        Xs.append(v)
        ys.append(y.iloc[i + time_steps : i + time_steps + 48])
    return np.array(Xs), np.array(ys)



def save_predict(X, y, scaler, path):
    model = keras.models.load_model(path)
    y_pred_scaled = model.predict(X)
    y_pred_unscaled = scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).reshape(y_pred_scaled.shape)
    y_unscaled = scaler.inverse_transform(
        y.reshape(-1, 1)
    ).reshape(y.shape)
    unscaled_mae = mean_absolute_error(y_unscaled, y_pred_unscaled)
    unscaled_rmse = np.sqrt(mean_squared_error(y_unscaled, y_pred_unscaled))
    unscaled_mape = mean_absolute_percentage_error(y_unscaled, y_pred_unscaled)
    unscaled_mae = mean_absolute_error(y_unscaled, y_pred_unscaled)
    return unscaled_mae, unscaled_rmse, unscaled_mape, y_unscaled, y_pred_unscaled

def plot_LSTM(number_of_points, y_test, test_predictions):
    actual = y_test[::48].reshape(-1)[:number_of_points]
    predicted = test_predictions[::48].reshape(-1)[:number_of_points]
    plt.figure(figsize = (12,5))
    plt.plot(actual, label = "Actual demand")
    plt.plot(predicted, label = "LSTM prediction", color = "red")
    plt.xlabel("Half-hourly observation")
    plt.ylabel("Demand (MW)")
    plt.title("LSTM test forecast: two weeks")
    plt.legend()
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents = True, exist_ok = True)
    plt.savefig(FIGURES_DIR / "lstm_test_forecast.png", dpi = 300)

def main():
    FEATURES = ['SETTLEMENT_PERIOD', 'Month', 'Month_Sin', 'Month_Cos', 'Year', 'Is_Weekend', 'ND_7prior']
    TARGET = 'ND'
    TIMESTEPS = 7 * 48

    val = pd.read_csv(PROCESSED_DATA_DIR / 'scaled_val.csv')
    test = pd.read_csv(PROCESSED_DATA_DIR / 'scaled_test.csv')

    X_val, y_val = val[FEATURES], val[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    Xs_val, ys_val = create_dataset(X_val, y_val, TIMESTEPS)
    Xs_test, ys_test = create_dataset(X_test, y_test, TIMESTEPS)

    train_orig = pd.read_csv(PROCESSED_DATA_DIR / "train_sb.csv")
    ND_transformer = StandardScaler().fit(train_orig[['ND']].to_numpy())

    unscaled_mae_val, unscaled_rmse_val, unscaled_mape_val, y_val, val_predictions = save_predict(Xs_val, ys_val, ND_transformer, MODELS_DIR / "best_lstm.keras")
    unscaled_mae_tes, unscaled_rmse_tes, unscaled_mape_tes, y_test, test_predictions = save_predict(Xs_test, ys_test, ND_transformer, MODELS_DIR / "best_lstm.keras")

    results = {
        "model":"LSTM",
        "metric":"MAE",
        "MAE_MW_val":float(unscaled_mae_val),
        "RMSE_MW_val":float(unscaled_rmse_val),
        "MAPE_MW_val":float(unscaled_mape_val),
        "MAE_MW_test":float(unscaled_mae_tes),
        "RMSE_MW_test":float(unscaled_rmse_tes),
        "MAPE_MW_test":float(unscaled_mape_tes)
    }
    METRICS_DIR.mkdir(parents = True, exist_ok = True)
    with open(METRICS_DIR / "experiment_results_LSTM.json", "w") as file:
        json.dump(results, file, indent = 4)

    plot_LSTM(14*48, y_test, test_predictions)

if __name__ == "__main__":
    main()
