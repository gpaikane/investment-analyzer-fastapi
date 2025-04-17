import yfinance as yf
from datetime import datetime
from prophet import Prophet
import pandas as pd



class ForeCast:
    @classmethod
    def get_history(cls, ticker, start_date = "2019-01-01"):
        end_date = str(datetime.now().date())

        # Download data
        data = yf.download(ticker, start = start_date, end = end_date)

        # Keep only the 'Close' column and reset index
        df = data[['Close']].reset_index()

        # Rename columns to fit Prophet's expected format
        df = df.rename(columns={"Date": "ds", "Close": "y"})

        data = pd.DataFrame()
        data["ds"] = df.iloc[:, 0]
        data["y"] = df.iloc[:, 1]
        return data


    @classmethod
    def forecast_by_prophet(cls,data ):
        # Initialize and fit the model
        model = Prophet(daily_seasonality=True)
        model.fit(data)

        # Make a future dataframe for prediction (e.g., 180 days into the future)
        future = model.make_future_dataframe(periods=180)

        # Forecast the future
        forecast = model.predict(future)

        return forecast


    @classmethod
    def get_forecasted_data(cls, ticker):
        history = cls.get_history(ticker)
        return cls.forecast_by_prophet(history)