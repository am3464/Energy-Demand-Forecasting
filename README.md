# Energy-Demand-Forecasting
Machine learning architectures: XGBoost, LSTM and a Transformer implemented from scratch are compared against a seasonal baseline

|Model|MAE /MW|RMSE /MW|MAPE|
|---|---|---|---|
|Seasonal Baseline (Weekly)|3346|5093|0.1289
|XGBoost|2750|4177|0.1079
|LSTM|2837|3996|0.1091
|Transformer| 2836|4079| 0.1112

The following images show the forecasts of the models on unseen data. Please note that these graphs only show the first fortnight of the test set and therefore these aren't representative of the overall model performance. 

<img width="3600" height="1500" alt="transformer_test_forecast" src="https://github.com/user-attachments/assets/c41e79f3-de62-40b0-ab31-9f7c66cdb1a7" />
<img width="3600" height="1500" alt="lstm_test_forecast" src="https://github.com/user-attachments/assets/65829076-262d-45b2-bd30-bf511c05ccd7" />
<img width="3600" height="1500" alt="xgboost_forecast" src="https://github.com/user-attachments/assets/b3709b07-a6a5-4772-9e07-09ddc50cd226" />
<img width="3600" height="1500" alt="seasonal_baseline_forecast" src="https://github.com/user-attachments/assets/fc8f1f94-bc82-4c17-92ce-3e547d724528" />




