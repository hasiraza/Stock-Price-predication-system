import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO
import base64

# Set page config
st.set_page_config(
    page_title="Stock Price Prediction System",
    page_icon="📈",
    layout="wide"
)

# Background functions
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except (FileNotFoundError, Exception) as e:
        print(f"Background image not found: {e}")
        return None

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    if bin_str:
        page_bg_img = f'''
        <style>
        .stApp {{
            background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
                         url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        .main > div {{
            padding-top: 2rem;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            margin: 1rem;
        }}
        
        .stSelectbox > div > div > div {{
            background-color: rgba(255, 255, 255, 0.9);
        }}
        
        .stButton > button {{
            background-color: rgba(255, 87, 51, 0.8);
            color: white;
            border: none;
            border-radius: 2px;
            backdrop-filter: blur(5px);
        }}
        
        .stButton > button:hover {{
            background-color: rgba(255, 87, 51, 1);
        }}
        
        .sidebar .sidebar-content {{
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(10px);
        }}
        
        h1, h2, h3 {{
            color: white !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
        }}
        
        .stText {{
            color: white !important;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
    else:
        # Fallback styling without background image
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        h1, h2, h3 {
            color: white !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
        }
        
        .stSelectbox > div > div > div {
            background-color: rgba(255, 255, 255, 0.9);
        }
        
        .stButton > button {
            background-color: rgba(255, 87, 51, 0.8);
            color: white;
            border: none;
            border-radius: 10px;
            backdrop-filter: blur(5px);
        }
        
        .stButton > button:hover {
            background-color: rgba(255, 87, 51, 1);
        }
        </style>
        """, unsafe_allow_html=True)

# Apply background (with better error handling)
try:
    set_background('background.jpg')
except Exception as e:
    print(f"Warning: Could not set background image: {e}")
    # Continue without background image

# Stock symbols dictionary with popular stocks
STOCK_OPTIONS = {
    # US Stocks
    "Apple Inc. (AAPL)": "AAPL",
    "Microsoft Corp. (MSFT)": "MSFT",
    "Amazon.com Inc. (AMZN)": "AMZN",
    "Alphabet Inc. (GOOGL)": "GOOGL",
    "Tesla Inc. (TSLA)": "TSLA",
    "Meta Platforms (META)": "META",
    "NVIDIA Corp. (NVDA)": "NVDA",
    "JPMorgan Chase (JPM)": "JPM",
    "Johnson & Johnson (JNJ)": "JNJ",
    "Visa Inc. (V)": "V",
    "Walmart Inc. (WMT)": "WMT",
    "Procter & Gamble (PG)": "PG",
    "UnitedHealth Group (UNH)": "UNH",
    "Home Depot (HD)": "HD",
    "Mastercard Inc. (MA)": "MA",
    
    # Indian Stocks
    "Reliance Industries (RELIANCE.NS)": "RELIANCE.NS",
    "Tata Consultancy Services (TCS.NS)": "TCS.NS",
    "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
    "Infosys (INFY.NS)": "INFY.NS",
    "ICICI Bank (ICICIBANK.NS)": "ICICIBANK.NS",
    "State Bank of India (SBIN.NS)": "SBIN.NS",
    "Bharti Airtel (BHARTIARTL.NS)": "BHARTIARTL.NS",
    "ITC Ltd (ITC.NS)": "ITC.NS",
    "Hindustan Unilever (HINDUNILVR.NS)": "HINDUNILVR.NS",
    "Power Grid Corp (POWERGRID.NS)": "POWERGRID.NS",
    "Wipro (WIPRO.NS)": "WIPRO.NS",
    "Tech Mahindra (TECHM.NS)": "TECHM.NS",
    "Asian Paints (ASIANPAINT.NS)": "ASIANPAINT.NS",
    "Maruti Suzuki (MARUTI.NS)": "MARUTI.NS",
    "Bajaj Finance (BAJFINANCE.NS)": "BAJFINANCE.NS",
    
    # Additional Popular Stocks
    "Netflix Inc. (NFLX)": "NFLX",
    "Adobe Inc. (ADBE)": "ADBE",
    "Salesforce Inc. (CRM)": "CRM",
    "Intel Corp. (INTC)": "INTC",
    "Cisco Systems (CSCO)": "CSCO"
}

# Main header
st.header("Stock Price Prediction System")

# Sidebar info
st.sidebar.header("Stock Price Prediction")
st.sidebar.subheader("About")
st.sidebar.markdown(
    """
    This :blue[**Stock Price Prediction System**] uses Deep Learning to predict stock prices
    based on historical data with EMA analysis.
    """
)
st.sidebar.subheader("How to use")
st.sidebar.markdown(
    """
    :blue[**1.**] Select stock from dropdown menu
    :blue[**2.**] Click 'Predict Stock Price'
    :blue[**3.**] View EMA charts and predictions
    :blue[**4.**] Download dataset as CSV
    """
)

# Stock selection with dropdown
st.subheader("Select a Stock")
stock_selection = st.selectbox(
    "Select Stock Symbol:",
    options=list(STOCK_OPTIONS.keys()),
    index=list(STOCK_OPTIONS.values()).index("POWERGRID.NS") if "POWERGRID.NS" in STOCK_OPTIONS.values() else 0,
    help="Select from the list of available stocks"
)
stock_symbol = STOCK_OPTIONS[stock_selection]

# Prediction button placed below the dropdown
predict_button = st.button("🔮 Predict Stock Price")

# Prediction logic - Charts displayed directly when button is clicked
if predict_button and stock_symbol:
    try:
        with st.spinner(f"Loading data and making predictions for {stock_symbol}..."):
            # Load the model
            try:
                model = load_model('stock_by_model.h5')
            except:
                st.error("❌ Model file 'stock_by_model.h5' not found. Please ensure the model file is in your project directory.")
                st.stop()
            
            # Define date range
            start = dt.datetime(2000, 1, 1)
            end = dt.datetime(2024, 10, 1)
            
            # Download stock data
            df = yf.download(stock_symbol, start=start, end=end)
            
            if df.empty:
                st.error("❌ No data found for the entered stock symbol. Please check the symbol and try again.")
                st.stop()
            
            # Display basic info
            st.success(f"✅ Successfully loaded data for {stock_symbol}")
            st.info(f"📊 Data range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
            
            # Calculate EMAs
            ema20 = df.Close.ewm(span=20, adjust=False).mean()
            ema50 = df.Close.ewm(span=50, adjust=False).mean()
            ema100 = df.Close.ewm(span=100, adjust=False).mean()
            ema200 = df.Close.ewm(span=200, adjust=False).mean()
            
            # Data preparation for prediction
            data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
            data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_training_array = scaler.fit_transform(data_training)
            
            # Prepare data for prediction
            past_100_days = data_training.tail(100)
            final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
            input_data = scaler.fit_transform(final_df)
            
            x_test, y_test = [], []
            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i - 100:i])
                y_test.append(input_data[i, 0])
            x_test, y_test = np.array(x_test), np.array(y_test)

            # Make predictions
            y_predicted = model.predict(x_test)
            
            # Inverse scaling
            scaler_scale = scaler.scale_
            scale_factor = 1 / scaler_scale[0]
            y_predicted = y_predicted * scale_factor
            y_test = y_test * scale_factor
            
            # Fix for performance metrics
            # Flatten the y_predicted array correctly
            y_predicted_flat = y_predicted.flatten()
            
            # Ensure both arrays have the same length
            min_length = min(len(y_test), len(y_predicted_flat))
            y_test_truncated = y_test[:min_length]
            y_predicted_truncated = y_predicted_flat[:min_length]
            
            # Calculate metrics
            mse = np.mean((y_test_truncated - y_predicted_truncated)**2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test_truncated - y_predicted_truncated))
            
            # Set matplotlib style for black background charts
            plt.style.use('dark_background')
            plt.rcParams['figure.facecolor'] = 'black'
            plt.rcParams['axes.facecolor'] = 'black'
            
            # CHART 1: EMA 20 & 50 (Black Background)
            st.subheader("📊 EMA 20 & 50 Days Analysis")
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            fig1.patch.set_facecolor('black')
            ax1.set_facecolor('black')
            ax1.plot(df.index, df.Close, 'yellow', label='Closing Price', linewidth=2)
            ax1.plot(df.index, ema20, 'lime', label='EMA 20', linewidth=1.5, linestyle='--')
            ax1.plot(df.index, ema50, 'red', label='EMA 50', linewidth=1.5, linestyle=':')
            ax1.set_title(f"{stock_symbol} - Closing Price vs Time (20 & 50 Days EMA)", fontsize=14, fontweight='bold', color='white')
            ax1.set_xlabel("Time", fontsize=12, color='white')
            ax1.set_ylabel("Price", fontsize=12, color='white')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3, color='white', linestyle='-', linewidth=0.5)
            ax1.tick_params(colors='white')
            plt.tight_layout()
            st.pyplot(fig1)
            
            # CHART 2: EMA 100 & 200 (Black Background)
            st.subheader("📊 EMA 100 & 200 Days Analysis")
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            fig2.patch.set_facecolor('black')
            ax2.set_facecolor('black')
            ax2.plot(df.index, df.Close, 'yellow', label='Closing Price', linewidth=2)
            ax2.plot(df.index, ema100, 'lime', label='EMA 100', linewidth=1.5, linestyle='--')
            ax2.plot(df.index, ema200, 'red', label='EMA 200', linewidth=1.5, linestyle=':')
            ax2.set_title(f"{stock_symbol} - Closing Price vs Time (100 & 200 Days EMA)", fontsize=14, fontweight='bold', color='white')
            ax2.set_xlabel("Time", fontsize=12, color='white')
            ax2.set_ylabel("Price", fontsize=12, color='white')
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3, color='white', linestyle='-', linewidth=0.5)
            ax2.tick_params(colors='white')
            plt.tight_layout()
            st.pyplot(fig2)
            
            # CHART 3: Prediction vs Original (Black Background)
            st.subheader("🔮 Prediction vs Original Results")
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            fig3.patch.set_facecolor('black')
            ax3.set_facecolor('black')
            ax3.plot(y_test_truncated, 'lime', label="Original Price", linewidth=2)
            ax3.plot(y_predicted_truncated, 'red', label="Predicted Price", linewidth=2, linestyle='--')
            ax3.set_title(f"{stock_symbol} - Prediction vs Original Trend", fontsize=14, fontweight='bold', color='white')
            ax3.set_xlabel("Time", fontsize=12, color='white')
            ax3.set_ylabel("Price", fontsize=12, color='white')
            ax3.legend(loc='upper left')
            ax3.grid(True, alpha=0.3, color='white', linestyle='-', linewidth=0.5)
            ax3.tick_params(colors='white')
            plt.tight_layout()
            st.pyplot(fig3)
            
            # Display metrics
            st.subheader("📈 Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${df.Close.iloc[-1]:.2f}")
            with col2:
                st.metric("RMSE", f"{rmse:.2f}")
            with col3:
                st.metric("MAE", f"{mae:.2f}")
            with col4:
                latest_prediction = y_predicted_truncated[-1] if len(y_predicted_truncated) > 0 else 0
                st.metric("Last Prediction", f"${latest_prediction:.2f}")
            
            # Descriptive statistics
            with st.expander("📈 Stock Data Statistics"):
                st.dataframe(df.describe(), use_container_width=True)
            
            # Prepare CSV download
            df_download = df.copy()
            df_download['EMA_20'] = ema20
            df_download['EMA_50'] = ema50
            df_download['EMA_100'] = ema100
            df_download['EMA_200'] = ema200
            
            # Convert DataFrame to CSV
            csv_buffer = BytesIO()
            df_download.to_csv(csv_buffer, index=True)
            csv_data = csv_buffer.getvalue()
            
            # Download button
            st.subheader("📥 Download Dataset")
            st.download_button(
                label=f"📄 Download {stock_symbol} Dataset (CSV)",
                data=csv_data,
                file_name=f"{stock_symbol}_stock_data_with_ema.csv",
                mime="text/csv",
                help="Click to download the complete dataset with EMA indicators"
            )
            
            # Clean up matplotlib figures
            plt.close('all')
            
    except Exception as e:
        pass
        # st.error(f"❌ An error occurred: {str(e)}")
        # st.error("Please check your stock symbol and ensure all required files are present.")

# Contact info in sidebar
st.sidebar.subheader("📞 Contact us")
st.sidebar.markdown(
    """
    **Email:** hasiraza511@gmail.com  
    **Linkedin:** https://www.linkedin.com/in/muhammad-haseeb-raza-71987a366/  
    **GitHub:** https://github.com/hasiraza
    """
)
