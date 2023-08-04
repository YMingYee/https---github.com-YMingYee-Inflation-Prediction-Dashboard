import numpy as np
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

#page title
st.set_page_config(page_title="Inflation Prediction Models", page_icon=":tada:", layout="wide")

#dataset
df = pd.read_csv("C:/Users/xpera/OneDrive/Desktop/Inflation Prediction System/inflation interest unemployment.csv")

#Country data   
country_data = {
    "US": df[df['country'] == 'United States'][['year', 'country', 'Inflation, consumer prices (annual %)']],
    "Algeria": df[df['country'] == 'Algeria'][['year', 'country', 'Inflation, consumer prices (annual %)']]
}

#sidebar
st.sidebar.header('Input Features')
st.title('Inflation Prediction Dashboard')
Model_name = st.sidebar.selectbox("Select Machine Learning Model", ("ARIMA", "SMA", "LSTM", "Random Forest", "DecisionTree", "ESP", "RNN", "CNN", "Gradient Boosting"))
Country = st.sidebar.selectbox("Select Country", list(country_data.keys()))
Prediction = st.sidebar.slider('Select prediction for how many years ahead', 1, 5, 1)
selected_country_data = country_data.get(Country, pd.DataFrame())

data = {'Model_name': [Model_name], 'Country': [Country], 'Prediction': [Prediction]}
X = pd.DataFrame(data)

def text_box(text, content=None):
    # Define CSS styling for the text box
    css = """
    <style>
    .text-box {
        border: 1px solid black;
        padding: 10px;
        margin: 10px;
    }
    </style>
    """

    # Display the text box with the provided text and optional content
    st.markdown(css, unsafe_allow_html=True)
    if content is not None:
        st.markdown(f'<div class="text-box"><p><strong>{text}</strong></p>{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="text-box"><p><strong>{text}</strong></p></div>', unsafe_allow_html=True)

# Create two columns for the text boxes
col1, col2, col3 = st.columns(3)

# Display the text boxes in the columns
with col1:
    text_box("Selected country:", Country)

with col2:
    text_box("Model Selected for Prediction:", Model_name)

with col3:
    text_box("Page:<br>Prediction")

def text_box2(text2, content=None):
    # Define CSS styling for the country data box
    css = """
    <style>
    .text-box2 {
        border: 1px solid black;
        padding: 10px;
        margin: 10px;
        height: 300px;
        overflow-y: scroll;
    }
    </style>
    """

    # Display the text box with the provided text and optional content
    st.markdown(css, unsafe_allow_html=True)
    if content is not None:
        st.markdown(f'<div class="text-box2"><p><strong>{text2}</strong></p>{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="text-box2"><p><strong>{text2}</strong></p></div>', unsafe_allow_html=True)


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

#css for Prediction column
global_css = """
    <style>
    .text-box2 {
        border: 1px solid black;
        padding: 10px;
        margin: 10px;
        height: 300px;
        overflow-y: scroll;
    }
    .styled-table td:first-child {
        font-weight: bold;
    }
    </style>
"""

st.markdown(global_css, unsafe_allow_html=True)

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

        # Plotting the line graph
        fig, ax = plt.subplots(figsize=(16, 8)) 
        ax.plot(selected_country_data.index, selected_country_data['Inflation, consumer prices (annual %)'], label='Historical')
        ax.plot(forecast_df.index, forecast_df['Inflation Prediction'], label='Forecast')
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('Inflation, consumer prices (annual %)', fontsize=10)
        ax.set_title('Inflation Forecast', fontsize=10)
        ax.legend()

        selected_country_data['Inflation, consumer prices (annual %)'] = selected_country_data['Inflation, consumer prices (annual %)'].astype(float)

        # Align the lengths of selected_country_data and forecast_values
        selected_country_data = selected_country_data.iloc[:len(forecast_values)]

        # Declare the variables for evaluation metrics
        rmse = mean_squared_error(selected_country_data['Inflation, consumer prices (annual %)'], forecast_values, squared=False)
        mae = mean_absolute_error(selected_country_data['Inflation, consumer prices (annual %)'], forecast_values)
        corr = np.corrcoef(selected_country_data['Inflation, consumer prices (annual %)'], forecast_values)[0, 1]

        return forecast_df, fig, rmse, mae, corr
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
            forecast_df = pd.Series(forecast_values, index=forecast_index)
            forecast_df = pd.DataFrame({'Forecasted Inflation': forecast_values}, index=forecast_index)

            fig, ax = plt.subplots(figsize=(16, 8)) 
            ax.plot(selected_country_data.index, selected_country_data[selected_country_data.columns[1]], label='Historical')
            ax.plot(forecast_df.index, forecast_df['Forecasted Inflation'], label='Forecast')
            ax.set_xlabel('Year')
            ax.set_ylabel('Inflation, consumer prices (annual %)')
            ax.set_title('Inflation Forecast')
            ax.legend()

            selected_country_data['Inflation, consumer prices (annual %)'] = selected_country_data['Inflation, consumer prices (annual %)'].astype(float)

            # Align the lengths of selected_country_data and forecast_values
            selected_country_data = selected_country_data.iloc[:len(forecast_values)]

            # Declare the variables for evaluation metrics
            rmse = mean_squared_error(selected_country_data['Inflation, consumer prices (annual %)'], forecast_values, squared=False)
            mae = mean_absolute_error(selected_country_data['Inflation, consumer prices (annual %)'], forecast_values)
            corr = np.corrcoef(selected_country_data['Inflation, consumer prices (annual %)'], forecast_values)[0, 1]

            return forecast_df, fig, rmse, mae, corr

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

            fig, ax = plt.subplots(figsize=(16, 8)) 
            ax.plot(selected_country_data.index, selected_country_data['Inflation, consumer prices (annual %)'], label='Historical')
            ax.plot(forecast_df.index, forecast_df['Inflation Prediction'], label='Forecast')
            ax.set_xlabel('Year')
            ax.set_ylabel('Inflation, consumer prices (annual %)')
            ax.set_title('Inflation Forecast')
            ax.legend()

            selected_country_data['Inflation, consumer prices (annual %)'] = selected_country_data['Inflation, consumer prices (annual %)'].astype(float)

            # Align the lengths of selected_country_data and forecast_values
            selected_country_data = selected_country_data.iloc[:len(forecast_exp)]

            # Declare the variables for evaluation metrics
            rmse = mean_squared_error(selected_country_data['Inflation, consumer prices (annual %)'], forecast_exp, squared=False)
            mae = mean_absolute_error(selected_country_data['Inflation, consumer prices (annual %)'], forecast_exp)
            corr = np.corrcoef(selected_country_data['Inflation, consumer prices (annual %)'], forecast_exp)[0, 1]

            return forecast_df, fig, rmse, mae, corr

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

                fig, ax = plt.subplots(figsize=(16, 8)) 
                ax.plot(selected_country_data.index, selected_country_data['Inflation, consumer prices (annual %)'], label='Historical')
                ax.plot(forecast_df.index, forecast_df['Inflation Prediction'], label='Forecast')
                ax.set_xlabel('Year')
                ax.set_ylabel('Inflation, consumer prices (annual %)')
                ax.set_title('Inflation Forecast')
                ax.legend()

                selected_country_data['Inflation, consumer prices (annual %)'] = selected_country_data['Inflation, consumer prices (annual %)'].astype(float)

                # Align the lengths of selected_country_data and forecast_values
                selected_country_data = selected_country_data.iloc[:len(forecast_rf)]

                # Declare the variables for evaluation metrics
                rmse = mean_squared_error(selected_country_data['Inflation, consumer prices (annual %)'], forecast_rf, squared=False)
                mae = mean_absolute_error(selected_country_data['Inflation, consumer prices (annual %)'], forecast_rf)
                corr = np.corrcoef(selected_country_data['Inflation, consumer prices (annual %)'], forecast_rf)[0, 1]

                return forecast_df, fig, rmse, mae, corr
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

            fig, ax = plt.subplots(figsize=(16, 8)) 
            ax.plot(selected_country_data.index, selected_country_data['Inflation, consumer prices (annual %)'], label='Historical')
            ax.plot(forecast_df.index, forecast_df['Inflation Prediction'], label='Forecast')
            ax.set_xlabel('Year')
            ax.set_ylabel('Inflation, consumer prices (annual %)')
            ax.set_title('Inflation Forecast')
            ax.legend()

            selected_country_data['Inflation, consumer prices (annual %)'] = selected_country_data['Inflation, consumer prices (annual %)'].astype(float)

            # Align the lengths of selected_country_data and forecast_values
            selected_country_data = selected_country_data.iloc[:len(forecast_exp)]

            # Declare the variables for evaluation metrics
            rmse = mean_squared_error(selected_country_data['Inflation, consumer prices (annual %)'], forecast_exp, squared=False)
            mae = mean_absolute_error(selected_country_data['Inflation, consumer prices (annual %)'], forecast_exp)
            corr = np.corrcoef(selected_country_data['Inflation, consumer prices (annual %)'], forecast_exp)[0, 1]

            return forecast_df, fig, rmse, mae, corr
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

                fig, ax = plt.subplots(figsize=(16, 8)) 
                ax.plot(selected_country_data.index, selected_country_data['Inflation, consumer prices (annual %)'], label='Historical')
                ax.plot(forecast_df.index, forecast_df['Inflation Prediction'], label='Forecast')
                ax.set_xlabel('Year')
                ax.set_ylabel('Inflation, consumer prices (annual %)')
                ax.set_title('Inflation Forecast')
                ax.legend()

                selected_country_data['Inflation, consumer prices (annual %)'] = selected_country_data['Inflation, consumer prices (annual %)'].astype(float)

                # Align the lengths of selected_country_data and forecast_values
                selected_country_data = selected_country_data.iloc[:len(pred_gbt_fut)]

                # Declare the variables for evaluation metrics
                rmse = mean_squared_error(selected_country_data['Inflation, consumer prices (annual %)'], pred_gbt_fut, squared=False)
                mae = mean_absolute_error(selected_country_data['Inflation, consumer prices (annual %)'], pred_gbt_fut)
                corr = np.corrcoef(selected_country_data['Inflation, consumer prices (annual %)'], pred_gbt_fut)[0, 1]

                return forecast_df, fig, rmse, mae, corr
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

                fig, ax = plt.subplots(figsize=(16, 8)) 
                ax.plot(selected_country_data.index, selected_country_data['Inflation, consumer prices (annual %)'], label='Historical')
                ax.plot(forecast_df.index, forecast_df['Inflation Prediction'], label='Forecast')
                ax.set_xlabel('Year')
                ax.set_ylabel('Inflation, consumer prices (annual %)')
                ax.set_title('Inflation Forecast')
                ax.legend()

                selected_country_data['Inflation, consumer prices (annual %)'] = selected_country_data['Inflation, consumer prices (annual %)'].astype(float)

                # Align the lengths of selected_country_data and forecast_values
                selected_country_data = selected_country_data.iloc[:len(pred_dt)]

                # Declare the variables for evaluation metrics
                rmse = mean_squared_error(selected_country_data['Inflation, consumer prices (annual %)'], pred_dt, squared=False)
                mae = mean_absolute_error(selected_country_data['Inflation, consumer prices (annual %)'], pred_dt)
                corr = np.corrcoef(selected_country_data['Inflation, consumer prices (annual %)'], pred_dt)[0, 1]

                return forecast_df, fig, rmse, mae, corr
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
                scaled_data = scaler.fit_transform(selected_country_data[['Inflation, consumer prices (annual %)']].values)
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
                future_years = prediction  # Use the user-selected prediction value
                X_future = scaled_data[-seq_length:].reshape(1, seq_length, -1)
                predictions_rnn = []

                for _ in range(future_years):
                    y_pred_future = model_rnn.predict(X_future)
                    predictions_rnn.append(y_pred_future[0, 0])

                    # Reshape y_pred_future to match the shape of X_future[:, 1:, :]
                    y_pred_future_reshaped = y_pred_future.reshape(1, 1, 1)

                    X_future = np.concatenate([X_future[:, 1:, :], y_pred_future_reshaped], axis=1)

                # Inverse scale the predictions
                predictions_rnn = scaler.inverse_transform(np.array(predictions_rnn).reshape(-1, 1))

                # Generate the future years index
                last_year = selected_country_data.index[-1].year
                future_years_index = pd.date_range(start=str(last_year + 1), periods=future_years, freq='A')

                # Create forecast DataFrame
                forecast_df = pd.DataFrame({'Year': future_years_index, 'Inflation Prediction': predictions_rnn.flatten()})
                forecast_df.set_index('Year', inplace=True)

                # Plot the data if historical data has at least 2 points
                if len(selected_country_data) >= 2:
                    fig, ax = plt.subplots(figsize=(16, 8))
                    ax.plot(selected_country_data.index, selected_country_data['Inflation, consumer prices (annual %)'], label='Historical')
                    ax.plot(forecast_df.index, forecast_df['Inflation Prediction'], label='Forecast')
                    ax.set_xlabel('Year')
                    ax.set_ylabel('Inflation, consumer prices (annual %)')
                    ax.set_title('Inflation Forecast')
                    ax.legend()
                else:
                    fig = None
                    st.write('Insufficient data for generating future predictions.')

                selected_country_data['Inflation, consumer prices (annual %)'] = selected_country_data['Inflation, consumer prices (annual %)'].astype(float)

                # Align the lengths of selected_country_data and forecast_values
                selected_country_data = selected_country_data.iloc[:len(predictions_rnn)]

                selected_country_data_flat = selected_country_data['Inflation, consumer prices (annual %)'].to_numpy().flatten()
                predictions_rnn_flat = predictions_rnn.flatten()

                # Compute correlation if both arrays have length at least 2
                if len(selected_country_data_flat) >= 2 and len(predictions_rnn_flat) >= 2:
                    corr_coefficient, _ = pearsonr(selected_country_data_flat, predictions_rnn_flat)
                else:
                    corr_coefficient = None

                # Declare the variables for evaluation metrics
                rmse = mean_squared_error(selected_country_data['Inflation, consumer prices (annual %)'], predictions_rnn, squared=False)
                mae = mean_absolute_error(selected_country_data['Inflation, consumer prices (annual %)'], predictions_rnn)

                return forecast_df, fig, rmse, mae, corr_coefficient

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
                scaled_data = scaler.fit_transform(selected_country_data[['Inflation, consumer prices (annual %)']].values)
                seq_length = 12

                # Create sequences
                X = []
                y = []
                for i in range(len(scaled_data) - seq_length):
                    X.append(scaled_data[i : i + seq_length])
                    y.append(scaled_data[i + seq_length])
                X = np.array(X)
                y = np.array(y)

                # Make predictions using the loaded CNN model
                y_pred_cnn = model_cnn.predict(X)

                # Inverse scale the predictions
                y_pred_cnn = scaler.inverse_transform(y_pred_cnn)

                # Forecast future years
                future_years = prediction
                X_future = scaled_data[-seq_length:].reshape(1, seq_length, -1)
                predictions_cnn = []

                for _ in range(future_years):
                    y_pred_future = model_cnn.predict(X_future)
                    predictions_cnn.append(y_pred_future[0, 0])
                    # Reshape y_pred_future to match the shape of X_future[:, 1:, :]
                    y_pred_future_reshaped = y_pred_future.reshape(1, 1, 1)
                    X_future = np.concatenate([X_future[:, 1:, :], y_pred_future_reshaped], axis=1)

                # Inverse scale the predictions
                predictions_cnn = scaler.inverse_transform(np.array(predictions_cnn).reshape(-1, 1))

                # Generate the future years index
                last_year = selected_country_data.index[-1].year
                future_years_index = pd.date_range(start=str(last_year + 1), periods=future_years, freq='A')

                # Create forecast DataFrame
                forecast_df = pd.DataFrame({'Year': future_years_index, 'Inflation Prediction': predictions_cnn.flatten()})
                forecast_df.set_index('Year', inplace=True)

                # Plot the data if historical data has at least 2 points
                if len(selected_country_data) >= 2:
                    fig, ax = plt.subplots(figsize=(16, 8))
                    ax.plot(selected_country_data.index, selected_country_data['Inflation, consumer prices (annual %)'], label='Historical')
                    ax.plot(forecast_df.index, forecast_df['Inflation Prediction'], label='Forecast')
                    ax.set_xlabel('Year')
                    ax.set_ylabel('Inflation, consumer prices (annual %)')
                    ax.set_title('Inflation Forecast')
                    ax.legend()
                else:
                    fig = None
                    st.write('Insufficient data for generating future predictions.')

                selected_country_data['Inflation, consumer prices (annual %)'] = selected_country_data['Inflation, consumer prices (annual %)'].astype(float)

                # Align the lengths of selected_country_data and forecast_values
                selected_country_data = selected_country_data.iloc[:len(predictions_cnn)]

                selected_country_data_flat = selected_country_data['Inflation, consumer prices (annual %)'].to_numpy().flatten()
                predictions_cnn_flat = predictions_cnn.flatten()

                # Compute correlation if both arrays have length at least 2
                if len(selected_country_data_flat) >= 2 and len(predictions_cnn_flat) >= 2:
                    corr_coefficient, _ = pearsonr(selected_country_data_flat, predictions_cnn_flat)
                else:
                    corr_coefficient = None

                # Declare the variables for evaluation metrics
                rmse = mean_squared_error(selected_country_data['Inflation, consumer prices (annual %)'], predictions_cnn, squared=False)
                mae = mean_absolute_error(selected_country_data['Inflation, consumer prices (annual %)'], predictions_cnn)

                return forecast_df, fig, rmse, mae, corr_coefficient
            else:
                st.write('Please select a valid number of years for prediction.')
    else:
        st.write('Insufficient data for generating future predictions.')


# Modify the conditions to handle different models
if Model_name == "ARIMA" and len(selected_country_data) > 0:
    forecasted_data, fig, rmse, mae, corr = predict_arima(selected_country_data, Prediction)
    if forecasted_data is not None:
        # Display the plot above col1 and col2
        # Create an expander with a custom header
        with st.expander("Inflation Forecast", expanded=True):
            # Apply custom CSS to control the plot size
            st.markdown(
                """
                <style>
                    /* Adjust box size properties here */
                    Img {
                        height: 500px;
                    }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Display the plot using st.pyplot inside the st.expander block
            st.pyplot(fig)

        # Create two columns for the text boxes
        col1, col2, col3 = st.columns(3)

        # Display the text boxes in the columns
        with col1:
            text_box2("Dataset of the selected country:", content=selected_country_data.to_html(index=False))

        with col2:
            st.markdown(global_css, unsafe_allow_html=True)
            forecast_table = forecasted_data.reset_index().to_html(index=False, classes='styled-table')
            st.markdown(f'<div class="text-box2">{forecast_table}</div>', unsafe_allow_html=True)
        
        with col3:
            text_box2("Model Accuracy:\n\n" +
              "<b>Root Mean Squared Error (RMSE):</b> " + str(rmse) + "\n\n" +
              "<b>Mean Absolute Error (MAE):<b> " + str(mae) + "\n\n" +
              "<b>Correlation:</b> " + str(corr))

elif Model_name == "SMA" and len(selected_country_data) > 0:
    forecasted_data, fig, rmse, mae, corr = predict_sma(selected_country_data, Prediction)
    if forecasted_data is not None:
        # Display the plot above col1 and col2
        with st.expander("Inflation Forecast", expanded=True):
            # Apply custom CSS to control the plot size
            st.markdown(
                """
                <style>
                    /* Adjust box size properties here */
                    Img {
                        height: 500px;
                    }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Display the plot using st.pyplot inside the st.expander block
            st.pyplot(fig)

        # Create two columns for the text boxes
        col1, col2, col3 = st.columns(3)

        # Display the text boxes in the columns
        with col1:
            text_box2("Dataset of the selected country:", content=selected_country_data.to_html(index=False))

        with col2:
            st.markdown(global_css, unsafe_allow_html=True)
            forecast_table = forecasted_data.reset_index().to_html(index=False, classes='styled-table')
            st.markdown(f'<div class="text-box2">{forecast_table}</div>', unsafe_allow_html=True)
        
        with col3:
            text_box2("Model Accuracy:\n\n" +
              "<b>Root Mean Squared Error (RMSE):</b> " + str(rmse) + "\n\n" +
              "<b>Mean Absolute Error (MAE):<b> " + str(mae) + "\n\n" +
              "<b>Correlation:</b> " + str(corr))

elif Model_name == "ESP" and len(selected_country_data) > 0:
    forecasted_data, fig, rmse, mae, corr = predict_esp(selected_country_data, Prediction)
    if forecasted_data is not None:
        # Display the plot above col1 and col2
        with st.expander("Inflation Forecast", expanded=True):
            # Apply custom CSS to control the plot size
            st.markdown(
                """
                <style>
                    /* Adjust box size properties here */
                    Img {
                        height: 500px;
                    }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Display the plot using st.pyplot inside the st.expander block
            st.pyplot(fig)

        # Create two columns for the text boxes
        col1, col2, col3 = st.columns(3)

        # Display the text boxes in the columns
        with col1:
            text_box2("Dataset of the selected country:", content=selected_country_data.to_html(index=False))

        with col2:
            st.markdown(global_css, unsafe_allow_html=True)
            forecast_table = forecasted_data.reset_index().to_html(index=False, classes='styled-table')
            st.markdown(f'<div class="text-box2">{forecast_table}</div>', unsafe_allow_html=True)
        
        with col3:
            text_box2("Model Accuracy:\n\n" +
              "<b>Root Mean Squared Error (RMSE):</b> " + str(rmse) + "\n\n" +
              "<b>Mean Absolute Error (MAE):<b> " + str(mae) + "\n\n" +
              "<b>Correlation:</b> " + str(corr))

elif Model_name == "Random Forest" and len(selected_country_data) > 0:
    forecasted_data, fig, rmse, mae, corr = predict_rf(selected_country_data, Prediction)
    if forecasted_data is not None:
        # Display the plot above col1 and col2
        with st.expander("Inflation Forecast", expanded=True):
            # Apply custom CSS to control the plot size
            st.markdown(
                """
                <style>
                    /* Adjust box size properties here */
                    Img {
                        height: 500px;
                    }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Display the plot using st.pyplot inside the st.expander block
            st.pyplot(fig)

        # Create two columns for the text boxes
        col1, col2, col3 = st.columns(3)

        # Display the text boxes in the columns
        with col1:
            text_box2("Dataset of the selected country:", content=selected_country_data.to_html(index=False))

        with col2:
            st.markdown(global_css, unsafe_allow_html=True)
            forecast_table = forecasted_data.reset_index().to_html(index=False, classes='styled-table')
            st.markdown(f'<div class="text-box2">{forecast_table}</div>', unsafe_allow_html=True)
        
        with col3:
            text_box2("Model Accuracy:\n\n" +
              "<b>Root Mean Squared Error (RMSE):</b> " + str(rmse) + "\n\n" +
              "<b>Mean Absolute Error (MAE):<b> " + str(mae) + "\n\n" +
              "<b>Correlation:</b> " + str(corr))

elif Model_name == "LSTM" and len(selected_country_data) > 0:
    forecasted_data, fig, rmse, mae, corr = predict_lstm(selected_country_data, Prediction)
    if forecasted_data is not None:
        # Display the plot above col1 and col2
        with st.expander("Inflation Forecast", expanded=True):
            # Apply custom CSS to control the plot size
            st.markdown(
                """
                <style>
                    /* Adjust box size properties here */
                    Img {
                        height: 500px;
                    }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Display the plot using st.pyplot inside the st.expander block
            st.pyplot(fig)

        # Create two columns for the text boxes
        col1, col2, col3 = st.columns(3)

        # Display the text boxes in the columns
        with col1:
            text_box2("Dataset of the selected country:", content=selected_country_data.to_html(index=False))

        with col2:
            st.markdown(global_css, unsafe_allow_html=True)
            forecast_table = forecasted_data.reset_index().to_html(index=False, classes='styled-table')
            st.markdown(f'<div class="text-box2">{forecast_table}</div>', unsafe_allow_html=True)
        
        with col3:
            text_box2("Model Accuracy:\n\n" +
              "<b>Root Mean Squared Error (RMSE):</b> " + str(rmse) + "\n\n" +
              "<b>Mean Absolute Error (MAE):<b> " + str(mae) + "\n\n" +
              "<b>Correlation:</b> " + str(corr))

elif Model_name == "Gradient Boosting" and len(selected_country_data) > 0:
    forecasted_data, fig, rmse, mae, corr = predict_GBR(selected_country_data, Prediction)
    if forecasted_data is not None:
        # Display the plot above col1 and col2
        with st.expander("Inflation Forecast", expanded=True):
            # Apply custom CSS to control the plot size
            st.markdown(
                """
                <style>
                    /* Adjust box size properties here */
                    Img {
                        height: 500px;
                    }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Display the plot using st.pyplot inside the st.expander block
            st.pyplot(fig)

        # Create two columns for the text boxes
        col1, col2, col3 = st.columns(3)

        # Display the text boxes in the columns
        with col1:
            text_box2("Dataset of the selected country:", content=selected_country_data.to_html(index=False))

        with col2:
            st.markdown(global_css, unsafe_allow_html=True)
            forecast_table = forecasted_data.reset_index().to_html(index=False, classes='styled-table')
            st.markdown(f'<div class="text-box2">{forecast_table}</div>', unsafe_allow_html=True)
        
        with col3:
            text_box2("Model Accuracy:\n\n" +
              "<b>Root Mean Squared Error (RMSE):</b> " + str(rmse) + "\n\n" +
              "<b>Mean Absolute Error (MAE):<b> " + str(mae) + "\n\n" +
              "<b>Correlation:</b> " + str(corr))

elif Model_name == "DecisionTree" and len(selected_country_data) > 0:
    forecasted_data, fig, rmse, mae, corr = predict_DTR(selected_country_data, Prediction)
    if forecasted_data is not None:
        # Display the plot above col1 and col2
        with st.expander("Inflation Forecast", expanded=True):
            # Apply custom CSS to control the plot size
            st.markdown(
                """
                <style>
                    /* Adjust box size properties here */
                    Img {
                        height: 500px;
                    }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Display the plot using st.pyplot inside the st.expander block
            st.pyplot(fig)

        # Create two columns for the text boxes
        col1, col2, col3 = st.columns(3)

        # Display the text boxes in the columns
        with col1:
            text_box2("Dataset of the selected country:", content=selected_country_data.to_html(index=False))

        with col2:
            st.markdown(global_css, unsafe_allow_html=True)
            forecast_table = forecasted_data.reset_index().to_html(index=False, classes='styled-table')
            st.markdown(f'<div class="text-box2">{forecast_table}</div>', unsafe_allow_html=True)
        
        with col3:
            text_box2("Model Accuracy:\n\n" +
              "<b>Root Mean Squared Error (RMSE):</b> " + str(rmse) + "\n\n" +
              "<b>Mean Absolute Error (MAE):<b> " + str(mae) + "\n\n" +
              "<b>Correlation:</b> " + str(corr))

elif Model_name == "RNN" and len(selected_country_data) > 0:
    forecasted_data, fig, rmse, mae, corr = predict_RNN(selected_country_data, Prediction)
    if forecasted_data is not None:
        # Display the plot above col1 and col2
        with st.expander("Inflation Forecast", expanded=True):
            # Apply custom CSS to control the plot size
            st.markdown(
                """
                <style>
                    /* Adjust box size properties here */
                    Img {
                        height: 500px;
                    }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Display the plot using st.pyplot inside the st.expander block
            st.pyplot(fig)

        # Create two columns for the text boxes
        col1, col2, col3 = st.columns(3)

        # Display the text boxes in the columns
        with col1:
            text_box2("Dataset of the selected country:", content=selected_country_data.to_html(index=False))

        with col2:
            st.markdown(global_css, unsafe_allow_html=True)
            forecast_table = forecasted_data.reset_index().to_html(index=False, classes='styled-table')
            st.markdown(f'<div class="text-box2">{forecast_table}</div>', unsafe_allow_html=True)
        
        with col3:
            text_box2("Model Accuracy:\n\n" +
              "<b>Root Mean Squared Error (RMSE):</b> " + str(rmse) + "\n\n" +
              "<b>Mean Absolute Error (MAE):<b> " + str(mae) + "\n\n" +
              "<b>Correlation:</b> " + str(corr))

elif Model_name == "CNN" and len(selected_country_data) > 0:
    forecasted_data, fig, rmse, mae, corr = predict_CNN(selected_country_data, Prediction)
    if forecasted_data is not None:
        # Display the plot above col1 and col2
        with st.expander("Inflation Forecast", expanded=True):
            # Apply custom CSS to control the plot size
            st.markdown(
                """
                <style>
                    /* Adjust box size properties here */
                    Img {
                        height: 500px;
                    }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Display the plot using st.pyplot inside the st.expander block
            st.pyplot(fig)

        # Create two columns for the text boxes
        col1, col2, col3 = st.columns(3)

        # Display the text boxes in the columns
        with col1:
            text_box2("Dataset of the selected country:", content=selected_country_data.to_html(index=False))

        with col2:
            st.markdown(global_css, unsafe_allow_html=True)
            forecast_table = forecasted_data.reset_index().to_html(index=False, classes='styled-table')
            st.markdown(f'<div class="text-box2">{forecast_table}</div>', unsafe_allow_html=True)
        
        with col3:
            text_box2("Model Accuracy:\n\n" +
              "<b>Root Mean Squared Error (RMSE):</b> " + str(rmse) + "\n\n" +
              "<b>Mean Absolute Error (MAE):<b> " + str(mae) + "\n\n" +
              "<b>Correlation:</b> " + str(corr))
