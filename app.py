import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import yfinance as yf
from curl_cffi import requests
import datetime

# Set Streamlit page config
st.set_page_config(page_title="📈 Stock Price Prediction", layout="wide")

# Load DL model
@st.cache_resource
def load_stock_model():
    try:
        model = load_model('stock_dl_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please ensure 'stock_dl_model.h5' is in the same directory as this script.")
        return None

model = load_stock_model()

# Custom function to fetch stock data safely
def get_stock_data(symbol, start, end):
    try:
        session = requests.Session(impersonate="chrome")
        ticker = yf.Ticker(symbol, session=session)
        df = ticker.history(start=start, end=end, auto_adjust=True)

        # Handle META fallback (old FB ticker)
        if df.empty and symbol.upper() == "META":
            ticker = yf.Ticker("FB", session=session)
            df = ticker.history(start=start, end=end, auto_adjust=True)

        return df
    except Exception as e:
        st.error(f"❌ Failed to fetch {symbol}: {e}")
        return pd.DataFrame()

# Sidebar inputs
st.sidebar.header("Stock Prediction Settings")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, AMZN, META):", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.date(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# Fetch stock data
df = get_stock_data(stock_symbol, start_date, end_date)

if df.empty:
    st.error("No data found. Please check the stock symbol or date range.")
    st.stop()

# Display stock data
st.subheader(f"📊 Stock Data for {stock_symbol}")
st.write(df.tail())

# Plot closing price
st.subheader("Closing Price Trend")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df['Close'], label='Closing Price', color='blue')
ax.set_title(f"{stock_symbol} Closing Price")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Calculate moving averages
df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
df['EMA_100'] = df['Close'].ewm(span=100, adjust=False).mean()
df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

# Plot EMAs
st.subheader("Exponential Moving Averages (EMA)")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df['Close'], label='Closing Price', color='blue')
ax.plot(df.index, df['EMA_50'], label='EMA 50', color='red', alpha=0.7)
ax.plot(df.index, df['EMA_100'], label='EMA 100', color='green', alpha=0.7)
ax.plot(df.index, df['EMA_200'], label='EMA 200', color='orange', alpha=0.7)
ax.set_title(f"{stock_symbol} Closing Price with EMAs")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Prepare data for prediction
if model:
    st.subheader("📈 Stock Price Prediction")

    data = df[['Close']].values
    training_size = int(len(data) * 0.70)
    test_data = data[training_size:]

    # Scale data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Prepare test input
    x_test = []
    y_test = []
    for i in range(100, len(test_data)):
        x_test.append(scaled_data[i-100:i])
        y_test.append(scaled_data[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)

    # Predict
    try:
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

        # Plot prediction vs actual
        st.subheader("Prediction vs Actual Prices")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index[-len(y_test):], df['Close'].iloc[-len(y_test):], label="Actual Price", color="blue")
        ax.plot(df.index[-len(predictions):], predictions, label="Predicted Price", color="red")
        ax.set_title(f"{stock_symbol} Stock Price Prediction")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Download data
st.subheader("⬇️ Download Data")
csv = df.to_csv().encode('utf-8')
st.download_button("Download CSV", csv, f"{stock_symbol}_data.csv", "text/csv")

# Metrics
st.subheader("📌 Key Metrics")
st.metric("Latest Closing Price", f"${df['Close'].iloc[-1]:.2f}")
st.metric("50-Day EMA", f"${df['EMA_50'].iloc[-1]:.2f}")
st.metric("100-Day EMA", f"${df['EMA_100'].iloc[-1]:.2f}")
st.metric("200-Day EMA", f"${df['EMA_200'].iloc[-1]:.2f}")
