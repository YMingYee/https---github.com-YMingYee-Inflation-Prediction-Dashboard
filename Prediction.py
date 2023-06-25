import numpy as np
import streamlit as st
import pandas as pd
import pickle
import joblib
import random
import matplotlib as plt
from sklearn.preprocessing import MinMaxScaler

#page title
st.set_page_config(page_title="Inflation Prediction Models", page_icon=":tada:", layout="wide")

#dataset
df = pd.read_csv("C:/Users/xpera/OneDrive/Desktop/Inflation Prediction System/inflation interest unemployment.csv")

#Country data   
country_data = {
    "US": df[df['country'] == 'United States'][['year', 'country', 'Inflation, consumer prices (annual %)']],
    "Algeria": df[df['country'] == 'Algeria'][['year', 'country', 'Inflation, consumer prices (annual %)']]
}

st.sidebar.header('Input Features')
Page = st.sidebar.selectbox("Select page", ("Explore", "Predict"))
Model_name = st.sidebar.selectbox("Select Machine Learning Model", ("ARIMA", "SMA", "LSTM", "Random Forest", "DecisionTree", "ESP", "RNN", "CNN", "Gradient Boosting"))
Country = st.sidebar.selectbox("Select Country", list(country_data.keys()))
Prediction = st.sidebar.slider('Select prediction for how many years ahead', 1, 5, 1)
selected_country_data = country_data.get(Country, pd.DataFrame())

data = {'Model_name': [Model_name], 'Country': [Country], 'Prediction': [Prediction]}
X = pd.DataFrame(data)

st.write('Dataset of the selected country:', Country)
st.write(selected_country_data)
st.write('Model Selected for Prediction:', Model_name)

model_arima_US = joblib.load(r'C:/Users/xpera/OneDrive/Desktop/Inflation Prediction System/Website/Arima.pkl')
model_arima_Algeria = joblib.load(r'C:/Users/xpera/OneDrive/Desktop/Inflation Prediction System/Website/Arima_Algeria.pkl')
model_ESP_US = joblib.load(r'C:/Users/xpera/OneDrive/Desktop/Inflation Prediction System/Website/ESP_US.pkl')
model_ESP_Algeria = joblib.load(r'C:/Users/xpera/OneDrive/Desktop/Inflation Prediction System/Website/ESP_Algeria.pkl')
model_RF_US = joblib.load(r'C:/Users/xpera/OneDrive/Desktop/Inflation Prediction System/Website/RandomForestR_US.pkl')
model_RF_Algeria = joblib.load(r'C:/Users/xpera/OneDrive/Desktop/Inflation Prediction System/Website/RandomForestR_Algeria.pkl')
model_lstm_US = joblib.load(r'C:/Users/xpera/OneDrive/Desktop/Inflation Prediction System/Website/LSTM_US.pkl')
model_lstm_Algeria = joblib.load(r'C:/Users/xpera/OneDrive/Desktop/Inflation Prediction System/Website/LSTM_Algeria.pkl')
model_GB_US = joblib.load(r'C:/Users/xpera/OneDrive/Desktop/Inflation Prediction System/Website/GradientB_US.pkl')
model_GB_Algeria = joblib.load(r'C:/Users/xpera/OneDrive/Desktop/Inflation Prediction System/Website/GradientB_Algeria.pkl')
model_DT_US = joblib.load(r'C:/Users/xpera/OneDrive/Desktop/Inflation Prediction System/Website/DT_US.pkl')
model_DT_Algeria = joblib.load(r'C:/Users/xpera/OneDrive/Desktop/Inflation Prediction System/Website/DT_Algeria.pkl')
model_rnn_US = joblib.load(r'C:/Users/xpera/OneDrive/Desktop/Inflation Prediction System/Website/RNN_US.pkl')
model_cnn_US = joblib.load(r'C:/Users/xpera/OneDrive/Desktop/Inflation Prediction System/Website/CNN_US.pkl')
model_rnn_Algeria = joblib.load(r'C:/Users/xpera/OneDrive/Desktop/Inflation Prediction System/Website/RNN_Algeria.pkl')
model_cnn_Algeria = joblib.load(r'C:/Users/xpera/OneDrive/Desktop/Inflation Prediction System/Website/CNN_Algeria.pkl')

# Define functions or code for each model

def predict_arima(selected_country_data, prediction):
    # model_arima = loaded_models['Arima']
    if Country == "US":
        model_arima = model_arima_US
    elif Country == "Algeria":
        model_arima = model_arima_Algeria
    else:
        st.write("Selected country not supported.")
        return

    selected_country_data['year'] = pd.to_datetime(selected_country_data['year'], format='%Y')
    selected_country_data['Inflation, consumer prices (annual %)'] = pd.to_numeric(selected_country_data['Inflation, consumer prices (annual %)'], errors='coerce')
    selected_country_data = selected_country_data.dropna(subset=['Inflation, consumer prices (annual %)'])
    selected_country_data.set_index('year', inplace=True)

    if prediction > 0:
        forecast = model_arima.get_forecast(steps=prediction)
        forecast_values = forecast.predicted_mean
        last_year = selected_country_data.index[-1].year
        future_years = pd.date_range(start=str(last_year + 1), periods=prediction, freq='A')
        forecast_df = pd.DataFrame({'Year': future_years, 'Inflation Prediction': forecast_values})
        forecast_df.set_index('Year', inplace=True)
        st.write('Forecasted Inflation:')
        st.write(forecast_df)
    else:
        st.write('Please select a valid number of years for prediction.')

def predict_sma(selected_country_data, prediction):
    if len(selected_country_data) > 0:
        selected_country_data['year'] = pd.to_datetime(selected_country_data['year'], format='%Y')
        selected_country_data['Inflation, consumer prices (annual %)'] = pd.to_numeric(selected_country_data['Inflation, consumer prices (annual %)'], errors='coerce')
        selected_country_data = selected_country_data.dropna(subset=['Inflation, consumer prices (annual %)'])
        selected_country_data.set_index('year', inplace=True)

        if prediction > 0:
            last_observed_value = selected_country_data['Inflation, consumer prices (annual %)'].iloc[-1]
            forecast_sma = [last_observed_value] * prediction
            forecast_index = pd.date_range(start=selected_country_data.index[-1], periods=prediction, freq='AS-JAN')[0:]
            random_factor = np.random.uniform(-0.1, 0.1, size=prediction)
            forecast_values = np.array(forecast_sma) + random_factor
            forecast_data = pd.Series(forecast_values, index=forecast_index)
            forecast_data.plot(legend=True, label='Forecast', color='red')
            st.write('Forecasted Inflation:')
            st.write(forecast_data)
        else:
            st.write('Please select a valid number of years for prediction.')

def predict_esp(selected_country_data, prediction):
    if len(selected_country_data) > 0:
        if Country == "US":
            model_esp = model_ESP_US
        elif Country == "Algeria":
             model_esp = model_ESP_Algeria
        else:
            st.write("Selected country not supported.")
            return

        selected_country_data['year'] = pd.to_datetime(selected_country_data['year'], format='%Y')
        selected_country_data['Inflation, consumer prices (annual %)'] = pd.to_numeric(selected_country_data['Inflation, consumer prices (annual %)'], errors='coerce')
        selected_country_data = selected_country_data.dropna(subset=['Inflation, consumer prices (annual %)'])
        selected_country_data.set_index('year', inplace=True)

        if prediction > 0:
            # Predicting values
            pred_exp = model_esp.predict(start=0, end=len(selected_country_data) - 1)

            # Forecasting future years
            last_observed_value = selected_country_data['Inflation, consumer prices (annual %)'].iloc[-1]
            future_years = pd.date_range(start=str(selected_country_data.index[-1].year + 1), periods=prediction, freq='A')
            forecast_exp = list(pred_exp[-1:]) + [last_observed_value] * (prediction - 1)
            forecast_df = pd.DataFrame({'Year': future_years, 'Inflation Prediction': forecast_exp})
            forecast_df.set_index('Year', inplace=True)
            st.write('Forecasted Inflation:')
            st.write(forecast_df)
        else:
            st.write('Please select a valid number of years for prediction.')

def predict_rf(selected_country_data, prediction):
    if len(selected_country_data) > 0:
        if Country == "US":
            model_dt = model_RF_US
        elif Country == "Algeria":
            model_dt = model_RF_Algeria
        else:
            st.write("Selected country not supported.")
            return

        selected_country_data['year'] = pd.to_datetime(selected_country_data['year'], format='%Y')
        selected_country_data['Inflation, consumer prices (annual %)'] = pd.to_numeric(selected_country_data['Inflation, consumer prices (annual %)'], errors='coerce')
        selected_country_data = selected_country_data.dropna(subset=['Inflation, consumer prices (annual %)'])
        selected_country_data.set_index('year', inplace=True)

        if len(selected_country_data) > 0:
            X = selected_country_data.index.values.reshape(-1, 1)
            y = selected_country_data['Inflation, consumer prices (annual %)']

            model_dt.fit(X, y)

            if prediction > 0:
                # Forecasting future years
                future_years = pd.date_range(start=str(selected_country_data.index[-1].year + 1), periods=prediction, freq='A')
                X_future = future_years.to_numpy().reshape(-1, 1)
                pred_dt_fut = model_dt.predict(X_future)

                # Generating forecast DataFrame
                last_observed_value = selected_country_data['Inflation, consumer prices (annual %)'].iloc[-1]
                forecast_rf = list(pred_dt_fut) + [last_observed_value] * (prediction - len(pred_dt_fut))
                forecast_df = pd.DataFrame({'Year': future_years, 'Inflation Prediction': forecast_rf})
                forecast_df.set_index('Year', inplace=True)
                st.write('Forecasted Inflation:')
                st.write(forecast_df)

            else:
                st.write('Please select a valid number of years for prediction.')
        else:
            st.write('Insufficient data for generating future predictions.')
    else:
        st.write('No data available for the selected country.')

def predict_lstm(selected_country_data, prediction):
    if len(selected_country_data) > 0:
        if Country == "US":
            model_lstm = model_lstm_US
        elif Country == "Algeria":
             model_lstm = model_lstm_Algeria
        else:
            st.write("Selected country not supported.")
            return

        selected_country_data['year'] = pd.to_datetime(selected_country_data['year'], format='%Y')
        selected_country_data['Inflation, consumer prices (annual %)'] = pd.to_numeric(selected_country_data['Inflation, consumer prices (annual %)'], errors='coerce')
        selected_country_data = selected_country_data.dropna(subset=['Inflation, consumer prices (annual %)'])
        selected_country_data.set_index('year', inplace=True)

        if prediction > 0:
            # Prepare the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(selected_country_data['Inflation, consumer prices (annual %)'].values.reshape(-1, 1))
            seq_length = 12

            # Create sequences
            X = []
            y = []
            for i in range(len(scaled_data) - seq_length):
                X.append(scaled_data[i : i + seq_length])
                y.append(scaled_data[i + seq_length])
            X = np.array(X)
            y = np.array(y)

            # Make predictions using the loaded LSTM model
            y_pred = model_lstm.predict(X)

            # Inverse scale the predictions
            y_pred = scaler.inverse_transform(y_pred)
            y_train = scaler.inverse_transform(y.reshape(-1, 1))

            # Forecast future years
            future_years = prediction
            X_future = scaled_data[-seq_length:].reshape(1, seq_length, 1)
            forecast = []

            for _ in range(future_years):
                y_pred_future = model_lstm.predict(X_future)
                forecast.append(y_pred_future[0, 0])
                X_future = np.concatenate([X_future[:, 1:, :], y_pred_future.reshape(1, 1, 1)], axis=1)

            # Create forecast DataFrame
            last_observed_value = selected_country_data['Inflation, consumer prices (annual %)'].iloc[-1]
            future_years = pd.date_range(start=str(selected_country_data.index[-1].year + 1), periods=prediction, freq='A')
            forecast_exp = forecast[-1:] + [last_observed_value] * (prediction - 1)
            forecast_df = pd.DataFrame({'Year': future_years, 'Inflation Prediction': forecast_exp})
            forecast_df.set_index('Year', inplace=True)
            st.write('Forecasted Inflation:')
            st.write(forecast_df)
        else:
            st.write('Please select a valid number of years for prediction.')

def predict_GBR(selected_country_data, prediction):
    if len(selected_country_data) > 0:
        if Country == "US":
            model_gb = model_GB_US
        elif Country == "Algeria":
            model_gb = model_GB_Algeria
        else:
            st.write("Selected country not supported.")
            return

        selected_country_data['year'] = pd.to_datetime(selected_country_data['year'], format='%Y')
        selected_country_data['Inflation, consumer prices (annual %)'] = pd.to_numeric(selected_country_data['Inflation, consumer prices (annual %)'], errors='coerce')
        selected_country_data = selected_country_data.dropna(subset=['Inflation, consumer prices (annual %)'])
        selected_country_data.set_index('year', inplace=True)

        if len(selected_country_data) > 0:
            X = selected_country_data.index.values.reshape(-1, 1)
            y = selected_country_data['Inflation, consumer prices (annual %)']

            model_gb.fit(X, y)

            if prediction > 0:
                # Prepare the data
                X_forecast = np.arange(len(selected_country_data.index), len(selected_country_data.index) + prediction).reshape(-1, 1)
                pred_gbt_fut = model_gb.predict(X_forecast)

                # Generate the forecasted date range
                forecast_dates = pd.date_range(start=selected_country_data.index[-1], periods=prediction, freq='AS-JAN')

                # Create forecast DataFrame
                forecast_df = pd.DataFrame({'Year': forecast_dates, 'Inflation Prediction': pred_gbt_fut})
                forecast_df.set_index('Year', inplace=True)

                st.write('Forecasted Inflation:')
                st.write(forecast_df)
            else:
                st.write('Please select a valid number of years for prediction.')
        else:
            st.write('Insufficient data for generating future predictions.')

def predict_DTR(selected_country_data, prediction):
    if len(selected_country_data) > 0:
        if Country == "US":
            model_dt = model_DT_US
        elif Country == "Algeria":
             model_dt = model_DT_Algeria
        else:
            st.write("Selected country not supported.")
            return

        selected_country_data['year'] = pd.to_datetime(selected_country_data['year'], format='%Y')
        selected_country_data['Inflation, consumer prices (annual %)'] = pd.to_numeric(selected_country_data['Inflation, consumer prices (annual %)'], errors='coerce')
        selected_country_data = selected_country_data.dropna(subset=['Inflation, consumer prices (annual %)'])
        selected_country_data.set_index('year', inplace=True)

        if len(selected_country_data) > 0:
            X = selected_country_data.index.values.reshape(-1, 1)
            y = selected_country_data['Inflation, consumer prices (annual %)']

            model_dt.fit(X, y)

            if prediction > 0:
                # Prepare the data
                X_forecast = np.arange(len(selected_country_data.index), len(selected_country_data.index) + prediction).reshape(-1, 1)
                pred_dt = model_dt.predict(X_forecast)

                # Generate the forecasted date range
                forecast_dates = pd.date_range(start=selected_country_data.index[-1], periods=prediction, freq='AS-JAN')

                # Create forecast DataFrame
                forecast_df = pd.DataFrame({'Year': forecast_dates, 'Inflation Prediction': pred_dt})
                forecast_df.set_index('Year', inplace=True)

                st.write('Forecasted Inflation:')
                st.write(forecast_df)
            else:
                st.write('Please select a valid number of years for prediction.')
        else:
            st.write('Insufficient data for generating future predictions.')

def predict_RNN(selected_country_data, prediction):
    if len(selected_country_data) > 0:
        if Country == "US":
            model_rnn = model_rnn_US
        elif Country == "Algeria":
            model_rnn = model_rnn_Algeria
        else:
            st.write("Selected country not supported.")
            return

        selected_country_data['year'] = pd.to_datetime(selected_country_data['year'], format='%Y')
        selected_country_data['Inflation, consumer prices (annual %)'] = pd.to_numeric(selected_country_data['Inflation, consumer prices (annual %)'], errors='coerce')
        selected_country_data = selected_country_data.dropna(subset=['Inflation, consumer prices (annual %)'])
        selected_country_data.set_index('year', inplace=True)

        if len(selected_country_data) > 0:
            if prediction > 0:
                # Prepare the data
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(selected_country_data['Inflation, consumer prices (annual %)'].values.reshape(-1, 1))
                seq_length = 12

                # Create sequences
                X = []
                y = []
                for i in range(len(scaled_data) - seq_length):
                    X.append(scaled_data[i : i + seq_length])
                    y.append(scaled_data[i + seq_length])
                X = np.array(X)
                y = np.array(y)

                # Make predictions using the loaded RNN model
                y_pred = model_rnn.predict(X)

                # Inverse scale the predictions
                y_pred = scaler.inverse_transform(y_pred)

                # Forecast future years
                future_years = prediction
                X_future = scaled_data[-seq_length:].reshape(1, seq_length, 1)
                predictions_rnn = []

                for _ in range(future_years):
                    y_pred_future = model_rnn.predict(X_future)
                    predictions_rnn.append(y_pred_future[0, 0])
                    X_future = np.concatenate([X_future[:, 1:, :], y_pred_future.reshape(1, 1, 1)], axis=1)

                # Create forecast DataFrame
                future_years = pd.date_range(start=str(selected_country_data.index[-1].year + 1), periods=prediction, freq='A')
                forecast_df = pd.DataFrame({'Year': future_years, 'Inflation Prediction': predictions_rnn})
                forecast_df.set_index('Year', inplace=True)
                st.write('Forecasted Inflation:')
                st.write(forecast_df)
            else:
                st.write('Please select a valid number of years for prediction.')
    else:
        st.write('Insufficient data for generating future predictions.')

def predict_CNN(selected_country_data, prediction):
    if len(selected_country_data) > 0:
        if Country == "US":
            model_cnn = model_cnn_US
        elif Country == "Algeria":
            model_cnn = model_cnn_Algeria
        else:
            st.write("Selected country not supported.")
            return

        selected_country_data['year'] = pd.to_datetime(selected_country_data['year'], format='%Y')
        selected_country_data['Inflation, consumer prices (annual %)'] = pd.to_numeric(selected_country_data['Inflation, consumer prices (annual %)'], errors='coerce')
        selected_country_data = selected_country_data.dropna(subset=['Inflation, consumer prices (annual %)'])
        selected_country_data.set_index('year', inplace=True)

        if len(selected_country_data) > 0:
            if prediction > 0:
                # Prepare the data
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(selected_country_data['Inflation, consumer prices (annual %)'].values.reshape(-1, 1))
                seq_length = 12

                # Create sequences
                X = []
                y = []
                for i in range(len(scaled_data) - seq_length):
                    X.append(scaled_data[i : i + seq_length])
                    y.append(scaled_data[i + seq_length])
                X = np.array(X)
                y = np.array(y)

                # Make predictions using the loaded RNN model
                y_pred_cnn = model_cnn.predict(X)

                # Inverse scale the predictions
                y_pred_cnn = scaler.inverse_transform(y_pred_cnn)

                # Forecast future years
                future_years = prediction
                X_future = scaled_data[-seq_length:].reshape(1, seq_length, 1)
                predictions_cnn = []

                for _ in range(future_years):
                    y_pred_future = model_cnn.predict(X_future)
                    predictions_cnn.append(y_pred_future[0, 0])
                    X_future = np.concatenate([X_future[:, 1:, :], y_pred_future.reshape(1, 1, 1)], axis=1)

                # Create forecast DataFrame
                future_years = pd.date_range(start=str(selected_country_data.index[-1].year + 1), periods=prediction, freq='A')
                forecast_df = pd.DataFrame({'Year': future_years, 'Inflation Prediction': predictions_cnn})
                forecast_df.set_index('Year', inplace=True)
                st.write('Forecasted Inflation:')
                st.write(forecast_df)
            else:
                st.write('Please select a valid number of years for prediction.')
    else:
        st.write('Insufficient data for generating future predictions.')

# Modify the conditions to handle different models

if Model_name == "ARIMA" and len(selected_country_data) > 0:
    predict_arima(selected_country_data, Prediction)
elif Model_name == "SMA" and len(selected_country_data) > 0:
    predict_sma(selected_country_data, Prediction)
elif Model_name == "ESP" and len(selected_country_data) > 0:
    predict_esp(selected_country_data, Prediction)
elif Model_name == "Random Forest" and len(selected_country_data) > 0:
    predict_rf(selected_country_data, Prediction)
elif Model_name == "LSTM" and len(selected_country_data) > 0:
    predict_lstm(selected_country_data, Prediction)
elif Model_name == "Gradient Boosting" and len(selected_country_data) > 0:
    predict_GBR(selected_country_data, Prediction)
elif Model_name == "DecisionTree" and len(selected_country_data) > 0:
    predict_DTR(selected_country_data, Prediction)
elif Model_name == "RNN" and len(selected_country_data) > 0:
    predict_RNN(selected_country_data, Prediction)
elif Model_name == "CNN" and len(selected_country_data) > 0:
    predict_CNN(selected_country_data, Prediction)



