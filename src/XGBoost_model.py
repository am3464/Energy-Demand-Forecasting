from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from pathlib import Path
import json
from config import PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR

def train_loop(metric):

    for i in range(3, 9):

        reg = xgb.XGBRegressor(n_estimators = 1000, early_stopping_rounds = 50, objective = "reg:absoluteerror", eval_metric="mae", learning_rate = 0.05, max_depth = i, random_state = 67)
        reg.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_val, y_val)], verbose = 50)

        val_predictions = reg.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_predictions)
        print(val_mae)

        MODELS_DIR.mkdir(parents = True, exist_ok = True)
        reg.save_model(MODELS_DIR / f"xgboost_mae_model{i}.json")

def plot_XGBoost(number_of_points, y_val, val_predictions):
    
    actual = np.asarray(y_val)[:number_of_points]
    predicted = val_predictions[:number_of_points]
    # plotting xgboost predictions
    plt.figure(figsize = (12,5))
    plt.plot(actual, label = "Actual demand")
    plt.plot(predicted, label = "XGBoost prediction")

    plt.xlabel("Half-hourly observation")
    plt.ylabel("Demand (MW)")
    plt.title("XGBoost forecast: two weeks")
    plt.legend()
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents = True, exist_ok = True)
    plt.savefig(FIGURES_DIR / "xgboost_forecast.png", dpi = 300)

def plot_feature_importance(best_model):
    # plotting feature importance
    xgb.plot_importance(best_model, importance_type = "gain")
    plt.title("XGBoost feature importance")
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents = True, exist_ok = True)
    plt.savefig(FIGURES_DIR / "xgboost_feature_importance.png", dpi = 300)

def main():
    train = pd.read_csv(PROCESSED_DATA_DIR / 'train_sb.csv')
    val = pd.read_csv(PROCESSED_DATA_DIR / 'validation_sb.csv')
    test = pd.read_csv(PROCESSED_DATA_DIR / 'test_sb.csv')

    FEATURES = ['SETTLEMENT_PERIOD', 'Month', 'Month_Sin', 'Month_Cos', 'Year', 'Is_Weekend', 'ND_7prior']
    TARGET = 'ND'

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_val = val[FEATURES]
    y_val = val[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    best_model = xgb.XGBRegressor()
    best_model.load_model(MODELS_DIR / "xgboost_mae_model5.json")
    val_predictions = best_model.predict(X_val)
    test_predictions = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, test_predictions)
    rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    mape = mean_absolute_percentage_error(y_test, test_predictions)

    import json
    
    results = {
        "model":"XGBoost",
        "metric":"MAE",
        "MAE_MW_test":float(mae),
        "RMSE_MW_test":float(rmse),
        "MAPE_MW_test":float(mape)
    }
    METRICS_DIR.mkdir(parents = True, exist_ok = True)
    with open(METRICS_DIR / "experiment_results_XGBoost.json", "w") as file:
        json.dump(results, file, indent = 4)

    plot_XGBoost(7*48*2, y_test, test_predictions)
    plot_feature_importance(best_model)

if __name__ == "__main__":
    main()
