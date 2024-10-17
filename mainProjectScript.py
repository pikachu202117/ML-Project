import os
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense # type: ignore
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import tensorflow as tf
import streamlit as st
from datetime import datetime, timedelta

# Step 1: Fetch historical stock data
def get_historical_data(stock_symbol, model_path, start_date='2010-01-01', end_date='2024-10-11'):
    if not os.path.exists(f"{model_path}.csv"):
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
        stock_data.to_csv(f"{model_path}.csv")
        return stock_data
    else:
        stock_data = pd.read_csv(f"{model_path}.csv", index_col=0, parse_dates=True)
        return stock_data

# Step 2: Preprocess data (scaling and dataset creation)
def preprocess_data(stock_data, time_step=100):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))
    
    def create_dataset(data, time_step):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)
    
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y, scaler

# Step 3: Build the LSTM model
def build_model(time_step):
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(units=64))
    model.add(Dense(units=64))
    model.add(Dense(units=1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 4: Train the model
def train_model(stock_symbol, sym_dir, model_path, start_date, end_date, epochs=10, batch_size=64, save_model=True, time_step=100):
    stock_data = get_historical_data(stock_symbol, model_path, start_date, end_date)
    X, y, scaler = preprocess_data(stock_data, time_step)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    model = build_model(time_step)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    test_loss = model.evaluate(X_test, y_test)
    st.write(f"Test Loss: {test_loss:.4f}")
    
    if save_model:
        model.save(f"{model_path}_lstm_model.keras")
        st.write(f"Model saved at {model_path}_lstm_model.keras")
    
    return model, scaler, stock_data, X_test, y_test

# Step 5: Make predictions and plot the results
def plot_predictions(model, X_test, y_test, scaler, stock_data):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    
    original_data = stock_data['Close'].values
    predicted_data = np.empty_like(original_data)
    predicted_data[:] = np.nan
    predicted_data[-len(predictions):] = predictions.reshape(-1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(original_data, label='Original Data')
    plt.plot(predicted_data, label='Predicted Data', color='orange')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    st.pyplot(plt)

def fetch_intraday_data(ticker):
    stock_data = yf.download(tickers=ticker, period='1d', interval='1m')
    return stock_data

def create_candlestick_chart(stock_data, ticker):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name=f'{ticker} Candlestick'
    ))
    fig.update_layout(
        title=f'{ticker} Intraday Stock Price (1 Day)',
        xaxis_title='Time',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=False,
        template='plotly_white'
    )
    st.plotly_chart(fig)

def create_line_chart(stock_data, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stock_data.index,
        y=stock_data['Close'],
        mode='lines+markers',
        name=f'{ticker} Closing Price'
    ))
    fig.update_layout(
        title=f'{ticker} Intraday Stock Price (1 Day)',
        xaxis_title='Time',
        yaxis_title='Price (USD)',
        template='plotly_white'
    )
    st.plotly_chart(fig)

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def preprocess_for_prediction(stock_data, scaler):
    close_prices = stock_data['Close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(close_prices)
    X_test = scaled_data[-60:].reshape(1, 60, 1)
    return X_test

def predict_next_price(model, X_test, scaler):
    scaled_prediction = model.predict(X_test)
    predicted_price = scaler.inverse_transform(scaled_prediction)
    return predicted_price[0][0]

def create_updated_candlestick(stock_data, predicted_price, ticker):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name=f'{ticker} Actual Prices'
    ))
    last_time = stock_data.index[-1]
    next_time = last_time + pd.Timedelta(minutes=1)
    fig.add_trace(go.Scatter(
        x=[next_time],
        y=[predicted_price],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Predicted Price'
    ))
    fig.update_layout(
        title=f'{ticker} Intraday Stock Price with Next Prediction',
        xaxis_title='Time',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=False,
        template='plotly_white'
    )
    st.plotly_chart(fig)

def decide_trade_action(current_price, predicted_price):
    if predicted_price > current_price:
        return "Buy"
    elif predicted_price < current_price:
        return "Sell"
    else:
        return "Hold"

# Streamlit application starts here
st.set_page_config(page_title="Stock Price Predictor Pro", layout="wide", initial_sidebar_state="expanded")

# Custom CSS to enhance the UI
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .main .block-container {
        padding: 2rem;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #1E3A8A;
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🚀 Stock Price Predictor Pro")
st.markdown("### Harness the power of AI for smarter investment decisions")

# Sidebar for user input
with st.sidebar:
    st.header("📊 Configure Your Analysis")
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA):", "AAPL")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    analyze_button = st.button("🔍 Analyze Stock", use_container_width=True)

# Main content area
if analyze_button and (start_date < end_date) and stock_symbol:
    with st.spinner(f"Analyzing {stock_symbol}... Please wait."):
        cwd = os.getcwd()
        directories = [
            os.path.join(cwd, "files", stock_symbol, "models"),
            os.path.join(cwd, "files", stock_symbol, "data")
        ]

        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)

        model_path = os.path.join(cwd, "files", stock_symbol, "models", f"{stock_symbol}_{start_date}_to_{end_date}")

        # Set default epochs and batch size
        default_epochs = 10
        default_batch_size = 64

        if not os.path.exists(f"{model_path}_lstm_model.keras"):
            model, scaler, stock_data, X_test, y_test = train_model(stock_symbol, directories[0], model_path, start_date, end_date, epochs=default_epochs, batch_size=default_batch_size)
        else:
            model, scaler, stock_data, X_test, y_test = train_model(stock_symbol, directories[0], model_path, start_date, end_date, epochs=default_epochs, batch_size=default_batch_size, save_model=False)

    # Display predictions and charts in main area
    st.success(f"Analysis complete for {stock_symbol}")
    
    st.header(f"📈 {stock_symbol} Stock Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Historical vs Predicted Prices")
        plot_predictions(model, X_test, y_test, scaler, stock_data)
    
    with col2:
        st.subheader("Intraday Trading Data")
        stock_data2 = fetch_intraday_data(stock_symbol)
        create_candlestick_chart(stock_data2, stock_symbol)

    st.header("🔮 Price Prediction and Trading Recommendation")
    
    X_test = preprocess_for_prediction(stock_data2, scaler)
    predicted_price = predict_next_price(model, X_test, scaler)
    current_price = stock_data2['Close'].values[-1]
    trade_action = decide_trade_action(current_price, predicted_price)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    with col2:
        st.metric("Predicted Next Price", f"${predicted_price:.2f}", f"{((predicted_price - current_price) / current_price) * 100:.2f}%")
    with col3:
        st.metric("Recommended Action", trade_action, delta_color="off")

    st.subheader("Updated Price Prediction Chart")
    create_updated_candlestick(stock_data2, predicted_price, stock_symbol)

# Footer
st.markdown("---")
st.markdown("### 📚 About Stock Price Predictor Pro")
st.info("""
This application leverages advanced LSTM neural networks to predict stock prices based on historical data. 
It provides real-time analysis and trading recommendations to assist in making informed investment decisions. 
Please note that all predictions should be used in conjunction with comprehensive market research and professional financial advice.
""")

st.markdown("© 2024 Stock Price Predictor Pro. All rights reserved.")
