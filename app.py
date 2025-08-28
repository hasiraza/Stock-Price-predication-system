import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import base64
import os
#kk

# =========================================================
# Utility: Load stock model
# =========================================================
@st.cache_resource
def load_stock_model():
    try:
        model = load_model('stock_dl_model.h5')
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Please ensure 'stock_dl_model.h5' is in the same directory.")
        return None

# =========================================================
# Utility: Background image
# =========================================================
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.warning("⚠️ Background image not found.")
        return None

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    if bin_str:
        page_bg_img = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)

# =========================================================
# Robust Stock Data Fetch
# =========================================================
def get_stock_data(symbol, start, end):
    try:
        # First attempt
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)

        # META fallback (old FB ticker)
        if df.empty and symbol.upper() == "META":
            df = yf.download("FB", start=start, end=end, progress=False, auto_adjust=True)

        return df
    except Exception as e:
        st.error(f"❌ Failed to fetch {symbol}: {e}")
        return pd.DataFrame()

# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="📈 Stock Price Prediction", layout="wide")
set_background("background.png")  # comment this line if you don’t want background

st.title("📊 Stock Price Prediction System")
st.write("Enter a stock symbol and date range to view historical prices and model predictions.")

# Sidebar inputs
stock_symbol = st.sidebar.text_input("Stock Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Fetch data
df = get_stock_data(stock_symbol, start_date, end_date)

if df.empty:
    st.error("⚠️ No data found for the given stock and date range.")
    st.stop()

st.subheader(f"📅 Historical Data for {stock_symbol}")
st.dataframe(df.tail())

# Plot
st.subheader("📉 Closing Price Chart")
fig, ax = plt.subplots(figsize=(10, 4))
df['Close'].plot(ax=ax)
ax.set_ylabel("Price (USD)")
ax.set_xlabel("Date")
st.pyplot(fig)

# =========================================================
# Model Prediction
# =========================================================
model = load_stock_model()

if model:
    try:
        data = df.filter(['Close'])
        dataset = data.values
        training_data_len = int(np.ceil(len(dataset) * 0.95))

        # Scale data
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        # Prepare input
        test_data = scaled_data[training_data_len - 60:, :]
        X_test, y_test = [], dataset[training_data_len:, :]

        for i in range(60, len(test_data)):
            X_test.append(test_data[i-60:i, 0])

        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Predict
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)

        # Show results
        st.subheader("📈 Model Predictions vs Actual Prices")
        valid = data[training_data_len:].copy()
        valid['Predictions'] = predictions

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(valid['Close'], label="Actual Price")
        ax2.plot(valid['Predictions'], label="Predicted Price")
        ax2.legend()
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
else:
    st.warning("⚠️ Model not loaded. Skipping predictions.")
