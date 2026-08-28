# Energy-Demand-Forecasting
Machine learning architectures (XGBoost, LSTM and Transformer) are compared against a seasonal baseline

|Model|MAE /MW|RMSE /MW|MAPE|
|---|---|---|---|
|Seasonal Baseline (Weekly)|3346|5093|0.1289
|XGBoost|2750|4177|0.1079
|LSTM|2837|3996|0.1091
|Transformer| | | 

The following images show the forecasts of the models on unseen data

<img width="3600" height="1500" alt="transformer_test_forecast" src="https://github.com/user-attachments/assets/c41e79f3-de62-40b0-ab31-9f7c66cdb1a7" />
<img width="3600" height="1500" alt="lstm_test_forecast" src="https://github.com/user-attachments/assets/4a769c43-55a7-4090-9d24-0490c5a76a60" />
<img width="3600" height="1500" alt="xgboost_forecast" src="https://github.com/user-attachments/assets/b3709b07-a6a5-4772-9e07-09ddc50cd226" />
<img width="3600" height="1500" alt="seasonal_baseline_forecast" src="https://github.com/user-attachments/assets/fc8f1f94-bc82-4c17-92ce-3e547d724528" />




