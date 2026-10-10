
import pandas as pd

from forecasting import (
    prepare_forecasting_data,
    train_forecasting_model
)


# Test 1: Prepare next-day forecasting data

sample_weather = pd.DataFrame({
    "datetime": pd.to_datetime([
        "2024-01-01",
        "2024-01-02",
        "2024-01-04"
    ]),
    "temp_max": [30, 31, 29],
    "pressure_mean": [1010, 1011, 1012],
    "dew_point_mean": [24, 24, 23],
    "wind_speed_mean": [10, 12, 11],
    "wind_chill_mean": [29, 30, 28],
    "uv_index_mean": [5, 6, 4],
    "visibility_mean": [10, 10, 9],
    "humidity_mean": [80, 78, 82]
})


result = prepare_forecasting_data(sample_weather)

print("Test 1: Prepared forecasting data")
print(result[[
    "datetime",
    "target_date",
    "temp_max",
    "target_temp_max"
]])


# Test 2: Train a forecasting model
sample_training = pd.DataFrame({
    "temp_max": [28, 29, 30, 31, 32],
    "humidity_mean": [85, 82, 80, 77, 75]
})

X_train = sample_training[["temp_max", "humidity_mean"]]
y_train = pd.Series([29, 30, 31, 32, 33])

model = train_forecasting_model(X_train, y_train)

predictions = model.predict(X_train)

print("\nTest 2: Model predictions")
print("Actual temperatures:", y_train.tolist())
print("Predicted temperatures:", predictions.round(2).tolist())
