import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from Transformer import PositionalEncoding, MultiHeadSelfAttention, TransformerEncoder, TransformerBlock
from LSTM import create_dataset
import matplotlib.pyplot as plt
import json
from config import PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR

def plot_Transformer(number_of_points, y_test, test_predictions):
    actual = y_test[::48].reshape(-1)[:number_of_points]
    predicted = test_predictions[::48].reshape(-1)[:number_of_points]
    plt.figure(figsize = (12,5))
    plt.plot(actual, label = "Actual demand")
    plt.plot(predicted, label = "Transformer prediction", color = "green")
    plt.xlabel("Half-hourly observation")
    plt.ylabel("Demand (MW)")
    plt.title("Transformer test forecast: two weeks")
    plt.legend()
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents = True, exist_ok = True)
    plt.savefig(FIGURES_DIR / "transformer_test_forecast.png", dpi = 300)

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

    train_orig = pd.read_csv(PROCESSED_DATA_DIR / 'train_sb.csv')
    ND_transformer = StandardScaler().fit(train_orig[['ND']].to_numpy())

    model = keras.models.load_model(
        MODELS_DIR / "best_transformer.keras",
        custom_objects={
            "PositionalEncoding": PositionalEncoding,
            "MultiHeadSelfAttention": MultiHeadSelfAttention,
            "TransformerEncoder": TransformerEncoder,
            "TransformerBlock": TransformerBlock
        }
    )

    pred_val = model.predict(Xs_val, verbose = 0)
    pred_test = model.predict(Xs_test, verbose = 0)

    pred_val = ND_transformer.inverse_transform(pred_val.reshape(-1, 1)).reshape(pred_val.shape)
    pred_test = ND_transformer.inverse_transform(pred_test.reshape(-1, 1)).reshape(pred_test.shape)

    ys_val = ND_transformer.inverse_transform(ys_val.reshape(-1, 1)).reshape(ys_val.shape)
    ys_test = ND_transformer.inverse_transform(ys_test.reshape(-1, 1)).reshape(ys_test.shape)

    unscaled_mae_val = mean_absolute_error(ys_val, pred_val)
    unscaled_mae_test = mean_absolute_error(ys_test, pred_test)
    unscaled_rmse_val = np.sqrt(mean_squared_error(ys_val, pred_val))
    unscaled_rmse_test = np.sqrt(mean_squared_error(ys_test, pred_test))
    unscaled_mape_val = mean_absolute_percentage_error(ys_val, pred_val)
    unscaled_mape_test = mean_absolute_percentage_error(ys_test, pred_test)

    results = {
        "model":"Transformer",
        "metric":"MAE",
        "MAE_MW_val":float(unscaled_mae_val),
        "RMSE_MW_val":float(unscaled_rmse_val),
        "MAPE_MW_val":float(unscaled_mape_val),
        "MAE_MW_test":float(unscaled_mae_test),
        "RMSE_MW_test":float(unscaled_rmse_test),
        "MAPE_MW_test":float(unscaled_mape_test)
    }

    METRICS_DIR.mkdir(parents = True, exist_ok = True)
    with open(METRICS_DIR / "experiment_results_Transformer.json", "w") as file:
        json.dump(results, file, indent = 4)

    plot_Transformer(14*48, ys_test, pred_test)

if __name__ == "__main__":
    main()
