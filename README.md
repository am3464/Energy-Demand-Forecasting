# Energy-Demand-Forecasting
Machine learning architectures (XGBoost, LSTM and Transformer) are compared against a seasonal baseline

|Model|MAE /MW|RMSE /MW|MAPE|
|---|---|---|---|
|Seasonal Baseline (Weekly)|3346|5093|0.1289
|XGBoost|2750|4177|0.1079
|LSTM|2837|3996|0.1091
|Transformer| | | 

The following images show the forecasts of the models on unseen data

<img width="3600" height="1500" alt="seasonal_baseline_forecast" src="https://github.com/user-attachments/assets/c9510d51-6f22-4d43-99a7-084b68f661bb" />

<img width="3600" height="1500" alt="lstm_test_forecast" src="https://github.com/user-attachments/assets/a695deec-7522-4117-bc90-fc0c0bc6ea0a" />

<img width="3600" height="1500" alt="xgboost_forecast" src="https://github.com/user-attachments/assets/740953ad-732b-4ec6-b62a-56ec5ea4bf2c" />



