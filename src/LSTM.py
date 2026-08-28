from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from config import PROCESSED_DATA_DIR, MODELS_DIR

RANDOM_SEED = 67
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def scale_data(train, val, test):
    # Ensure any NaN values from feature engineering (e.g. shift) are dropped
    train = train.dropna().reset_index(drop=True)
    val = val.dropna().reset_index(drop=True)
    test = test.dropna().reset_index(drop=True)

    f_columns = ['SETTLEMENT_PERIOD', 'Month', 'Month_Sin', 'Month_Cos', 'Year', 'Is_Weekend']

    # Convert columns to float
    for df in [train, val, test]:
        df[f_columns] = df[f_columns].astype(float)
        df['ND'] = df['ND'].astype(float)
        df['ND_7prior'] = df['ND_7prior'].astype(float)


    f_transformer = StandardScaler().fit(train[f_columns].to_numpy())
    ND_transformer = StandardScaler().fit(train[['ND']].to_numpy())

    train.loc[:, f_columns] = f_transformer.transform(train[f_columns].to_numpy())
    val.loc[:, f_columns] = f_transformer.transform(val[f_columns].to_numpy())
    test.loc[:, f_columns] = f_transformer.transform(test[f_columns].to_numpy())


    train['ND'] = ND_transformer.transform(train[['ND']].to_numpy()).ravel()
    val['ND'] = ND_transformer.transform(val[['ND']].to_numpy()).ravel()
    test['ND'] = ND_transformer.transform(test[['ND']].to_numpy()).ravel()


    train['ND_7prior'] = ND_transformer.transform(train[['ND_7prior']].to_numpy()).ravel()
    val['ND_7prior'] = ND_transformer.transform(val[['ND_7prior']].to_numpy()).ravel()
    test['ND_7prior'] = ND_transformer.transform(test[['ND_7prior']].to_numpy()).ravel()


    PROCESSED_DATA_DIR.mkdir(parents = True, exist_ok = True)
    train.to_csv(PROCESSED_DATA_DIR / "scaled_train.csv", index=False)
    val.to_csv(PROCESSED_DATA_DIR / "scaled_val.csv", index=False)
    test.to_csv(PROCESSED_DATA_DIR / "scaled_test.csv", index=False)



def create_dataset(X, y, time_steps=7*48):
    Xs, ys = [], []
    for i in range(len(X) - time_steps - 48 + 1):
        v = X.iloc[i : (i + time_steps)].to_numpy()
        Xs.append(v)
        ys.append(y.iloc[i + time_steps : i + time_steps + 48])
    return np.array(Xs), np.array(ys)

def sequential_dataset():
    FEATURES = ['SETTLEMENT_PERIOD', 'Month', 'Month_Sin', 'Month_Cos', 'Year', 'Is_Weekend', 'ND_7prior']
    TARGET = 'ND'
    TIMESTEPS = 7 * 48

    train = pd.read_csv(PROCESSED_DATA_DIR / 'scaled_train.csv')
    val = pd.read_csv(PROCESSED_DATA_DIR / 'scaled_val.csv')
    test = pd.read_csv(PROCESSED_DATA_DIR / 'scaled_test.csv')

    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]

    Xs_train, ys_train = create_dataset(X_train, y_train, TIMESTEPS)
    Xs_val, ys_val = create_dataset(X_val, y_val, TIMESTEPS)
    return Xs_train, ys_train, Xs_val, ys_val



def LSTM_model(number_of_features):
    model = keras.Sequential([
        keras.layers.Input(shape=(336, number_of_features)),
        keras.layers.LSTM(32, dropout = 0.2),
        keras.layers.Dense(48)
    ])
    model.compile(optimizer="adam", loss="mae", metrics=["mae"])
    return model

def main():
    # Run scaling
    scale_data(pd.read_csv(PROCESSED_DATA_DIR / "train_sb.csv"), pd.read_csv(PROCESSED_DATA_DIR / "validation_sb.csv"), pd.read_csv(PROCESSED_DATA_DIR / "test_sb.csv"))
    Xs_train, ys_train, Xs_val, ys_val = sequential_dataset()

    # Diagnostic checks
    print("Xs_train NaNs:", np.isnan(Xs_train).sum())
    print("ys_train NaNs:", np.isnan(ys_train).sum())   
    model_current = LSTM_model(7)

    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    MODELS_DIR.mkdir(parents = True, exist_ok = True)
    checkpoint = keras.callbacks.ModelCheckpoint(MODELS_DIR / "best_lstm.keras", monitor="val_loss", save_best_only=True)

    history = model_current.fit( Xs_train, ys_train,validation_data=(Xs_val, ys_val),epochs=100,batch_size=32,callbacks=[early_stopping, checkpoint],shuffle=True,verbose=1)


if __name__ == "__main__":
    main()
