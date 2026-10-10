from sklearn.linear_model import LinearRegression
import pandas as pd 

FORECAST_FEATURES = [
    "temp_max",
    "pressure_mean",
    "dew_point_mean",
    "wind_speed_mean",
    "uv_index_mean",
    "visibility_mean",
    "humidity_mean"
]

def calculate_baseline_prediction(current_temperature):
    """
    Predict tommorrow's maximum temperature 
    using today's maximum temperature.
    """
    return current_temperature

def train_forecasting_model(X_train, y_train):
    """
    Train a Linear Regression model
    using historical weather data.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model

def prepare_forecasting_data(daily_predictors):
    """
    Prepare daily weather data for next_day forecasting

    """
    
    forecast_data = daily_predictors.copy()
    
    forecast_data = forecast_data.sort_values(
        "datetime"
    ).reset_index(drop=True)
    
    forecast_data["target_temp_max"] = (
        forecast_data["temp_max"].shift(-1)
    )
    
    forecast_data["target_date"] = (
        forecast_data["datetime"].shift(-1)
    )
    
    forecast_data["date_gap"] = (
        forecast_data["target_date"]
        - forecast_data["datetime"]
    ).dt.days
    
    forecast_data = forecast_data[
        forecast_data["date_gap"] == 1
    ].copy()
    
    forecast_data = forecast_data.dropna(
        subset=["target_temp_max"] + FORECAST_FEATURES
    ).reset_index(drop=True)

    return forecast_data
