import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import datetime as dt
import requests
from sklearn.preprocessing import MinMaxScaler
from io import BytesIO
import base64
import warnings
import json
import time
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Stock Price Prediction System",
    page_icon="📈",
    layout="wide"
)

# SOLUTION 1: Alternative data fetching using direct API calls
class StockDataFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def get_yahoo_data(self, symbol, start_date, end_date):
        """Alternative method using Yahoo Finance direct API"""
        try:
            # Convert dates to Unix timestamps
            start_ts = int(start_date.timestamp())
            end_ts = int(end_date.timestamp())
            
            # Yahoo Finance API endpoint
            url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
            params = {
                'period1': start_ts,
                'period2': end_ts,
                'interval': '1d',
                'events': 'history',
                'includeAdjustedClose': 'true'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                # Parse CSV data
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                return df
            else:
                return None
                
        except Exception as e:
            st.error(f"Direct API error: {str(e)}")
            return None
    
    def get_alpha_vantage_data(self, symbol, api_key=None):
        """Alternative using Alpha Vantage (requires API key)"""
        if not api_key:
            return None
            
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': api_key,
                'outputsize': 'full'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'Time Series (Daily)' in data:
                df = pd.DataFrame(data['Time Series (Daily)']).T
                df.index = pd.to_datetime(df.index)
                df = df.astype(float)
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                df.sort_index(inplace=True)
                return df
            else:
                return None
                
        except Exception as e:
            st.error(f"Alpha Vantage error: {str(e)}")
            return None

# SOLUTION 2: Fixed yfinance implementation
@st.cache_data(ttl=3600, show_spinner=False)
def download_stock_data_fixed(symbol, start_date, end_date):
    """Improved yfinance implementation with multiple fallbacks"""
    
    # Method 1: Try yfinance with fixed configuration
    try:
        import yfinance as yf
        
        # Disable multithreading and use simpler approach
        yf.pdr_override()
        
        ticker = yf.Ticker(symbol)
        
        # Use history method with minimal parameters
        df = ticker.history(
            start=start_date,
            end=end_date,
            auto_adjust=False,  # Set to False to avoid warning
            prepost=False,
            threads=False,  # Disable threading
            proxy=None
        )
        
        if not df.empty and len(df) > 50:
            return df
            
    except Exception as e:
        st.warning(f"yfinance method failed: {str(e)}")
    
    # Method 2: Try direct Yahoo API
    try:
        fetcher = StockDataFetcher()
        df = fetcher.get_yahoo_data(symbol, start_date, end_date)
        if df is not None and not df.empty:
            st.info("✅ Using direct Yahoo Finance API")
            return df
    except Exception as e:
        st.warning(f"Direct API method failed: {str(e)}")
    
    # Method 3: Use sample data as fallback
    st.warning("⚠️ Using sample data due to API limitations")
    return generate_sample_data(symbol, start_date, end_date)

def generate_sample_data(symbol, start_date, end_date):
    """Generate realistic sample stock data for demonstration"""
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days only
    n_days = len(dates)
    
    # Generate realistic stock price movement
    np.random.seed(42)  # For reproducible results
    
    # Starting price based on symbol
    price_map = {
        'AAPL': 150, 'MSFT': 300, 'GOOGL': 2500, 'AMZN': 3200, 'TSLA': 800,
        'META': 200, 'NVDA': 400, 'NFLX': 450, 'ADBE': 500
    }
    
    start_price = price_map.get(symbol.split('.')[0], 100)
    
    # Generate price series using random walk with drift
    returns = np.random.normal(0.0005, 0.02, n_days)  # Small positive drift with volatility
    price_series = [start_price]
    
    for i in range(1, n_days):
        price_series.append(price_series[-1] * (1 + returns[i]))
    
    # Create OHLCV data
    df = pd.DataFrame(index=dates)
    df['Close'] = price_series
    
    # Generate Open, High, Low based on Close with realistic variations
    df['Open'] = df['Close'].shift(1) * (1 + np.random.normal(0, 0.001, n_days))
    df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    df['Volume'] = np.random.randint(1000000, 10000000, n_days)
    
    # Fill NaN values
    df = df.fillna(method='bfill')
    
    return df

# SOLUTION 3: Pre-loaded data option
@st.cache_data
def load_preloaded_data():
    """Load pre-saved stock data if available"""
    preloaded_stocks = {
        'AAPL': 'aapl_data.csv',
        'MSFT': 'msft_data.csv',
        'GOOGL': 'googl_data.csv',
        'TSLA': 'tsla_data.csv'
    }
    
    available_data = {}
    for symbol, filename in preloaded_stocks.items():
        try:
            df = pd.read_csv(filename, index_col=0, parse_dates=True)
            available_data[symbol] = df
        except FileNotFoundError:
            continue
    
    return available_data

# Background styling (simplified version)
def set_background():
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
    
    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    </style>
    """, unsafe_allow_html=True)

# Apply styling
set_background()

# Stock options - simplified list for better reliability
STOCK_OPTIONS = {
    "Apple Inc. (AAPL)": "AAPL",
    "Microsoft Corp. (MSFT)": "MSFT",
    "Alphabet Inc. (GOOGL)": "GOOGL",
    "Tesla Inc. (TSLA)": "TSLA",
    "Amazon.com Inc. (AMZN)": "AMZN",
    "Meta Platforms (META)": "META",
    "NVIDIA Corp. (NVDA)": "NVDA",
    "Netflix Inc. (NFLX)": "NFLX",
    "Adobe Inc. (ADBE)": "ADBE",
    "SPDR S&P 500 ETF (SPY)": "SPY",
    "Invesco QQQ Trust (QQQ)": "QQQ"
}

# Main UI
st.header("🚀 Advanced Stock Price Prediction System")

# Sidebar with enhanced info
st.sidebar.header("📊 Stock Analysis Hub")
st.sidebar.markdown("""
### 🔧 **System Status**
- ✅ Deep Learning Model: Ready
- ⚠️ Data Source: Auto-fallback enabled
- 🔄 Cache: 1-hour refresh
""")

st.sidebar.markdown("""
### 🎯 **Features**
- Real-time stock data (with fallbacks)
- EMA trend analysis (20, 50, 100, 200)
- LSTM neural network predictions
- Performance metrics & insights
- Downloadable datasets
""")

# API Key input (optional)
st.sidebar.markdown("### 🔑 Optional API Configuration")
alpha_vantage_key = st.sidebar.text_input(
    "Alpha Vantage API Key (Optional)", 
    type="password",
    help="Get free API key from https://www.alphavantage.co/"
)

# Data source selection
data_source = st.sidebar.selectbox(
    "Data Source Priority:",
    ["Auto (Recommended)", "yFinance Only", "Direct API", "Sample Data"],
    help="Auto tries multiple sources automatically"
)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Select Stock for Analysis")
    stock_selection = st.selectbox(
        "Choose Stock Symbol:",
        options=list(STOCK_OPTIONS.keys()),
        index=0,
        help="Select from major stocks and ETFs"
    )
    stock_symbol = STOCK_OPTIONS[stock_selection]

with col2:
    st.subheader("⚙️ Analysis Settings")
    date_range = st.selectbox(
        "Data Range:",
        ["5 Years", "3 Years", "2 Years", "1 Year"],
        index=0
    )
    
    # Convert to years
    years_map = {"5 Years": 5, "3 Years": 3, "2 Years": 2, "1 Year": 1}
    years_back = years_map[date_range]

# Status indicators
st.markdown("---")
status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    st.metric("🎯 Selected Symbol", stock_symbol)

with status_col2:
    st.metric("📅 Analysis Period", f"{years_back} Year{'s' if years_back > 1 else ''}")

with status_col3:
    st.metric("🔄 Data Source", data_source)

# Main prediction button
st.markdown("---")
predict_button = st.button("🚀 **Start Stock Analysis & Prediction**", type="primary")

# Main prediction logic
if predict_button:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Load model
        status_text.text("🤖 Loading AI model...")
        progress_bar.progress(10)
        
        try:
            model = load_model('stock_by_model.h5')
            st.success("✅ AI Model loaded successfully")
        except Exception as e:
            st.error(f"❌ Model loading failed: {str(e)}")
            st.info("💡 Please ensure 'stock_by_model.h5' is in your project directory")
            st.stop()
        
        # Step 2: Prepare dates
        progress_bar.progress(20)
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=years_back * 365)
        
        # Step 3: Download data
        status_text.text(f"📊 Fetching {stock_symbol} data...")
        progress_bar.progress(30)
        
        df = download_stock_data_fixed(stock_symbol, start_date, end_date)
        
        if df is None or df.empty:
            st.error("❌ Could not fetch stock data from any source")
            st.info("Please try a different symbol or check your internet connection")
            st.stop()
        
        progress_bar.progress(50)
        st.success(f"✅ Data loaded: {len(df)} trading days from {df.index[0].date()} to {df.index[-1].date()}")
        
        # Step 4: Calculate EMAs
        status_text.text("📈 Calculating technical indicators...")
        progress_bar.progress(60)
        
        ema20 = df.Close.ewm(span=20, adjust=False).mean()
        ema50 = df.Close.ewm(span=50, adjust=False).mean()
        ema100 = df.Close.ewm(span=100, adjust=False).mean()
        ema200 = df.Close.ewm(span=200, adjust=False).mean()
        
        # Step 5: Prepare prediction data
        status_text.text("🔮 Preparing prediction model...")
        progress_bar.progress(70)
        
        if len(df) < 200:
            st.warning(f"⚠️ Limited data ({len(df)} days). Predictions may be less accurate.")
        
        # Data preparation
        split_point = int(len(df) * 0.8)
        data_training = pd.DataFrame(df['Close'][:split_point])
        data_testing = pd.DataFrame(df['Close'][split_point:])
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)
        
        # Prepare test data
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.transform(final_df)
        
        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])
        
        x_test, y_test = np.array(x_test), np.array(y_test)
        
        # Step 6: Make predictions
        status_text.text("🎯 Generating predictions...")
        progress_bar.progress(85)
        
        y_predicted = model.predict(x_test)
        
        # Inverse scaling
        scale_factor = 1 / scaler.scale_[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # Display results
        st.markdown("---")
        st.header("📊 **Analysis Results**")
        
        # Performance metrics
        y_pred_flat = y_predicted.flatten()
        min_len = min(len(y_test), len(y_pred_flat))
        y_test_clean = y_test[:min_len]
        y_pred_clean = y_pred_flat[:min_len]
        
        mse = np.mean((y_test_clean - y_pred_clean)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test_clean - y_pred_clean))
        
        # Metrics display
        st.subheader("📈 Key Metrics")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("💰 Current Price", f"${df.Close.iloc[-1]:.2f}")
        with metric_col2:
            st.metric("🎯 RMSE", f"{rmse:.2f}")
        with metric_col3:
            st.metric("📊 MAE", f"{mae:.2f}")
        with metric_col4:
            latest_pred = y_pred_clean[-1] if len(y_pred_clean) > 0 else df.Close.iloc[-1]
            st.metric("🔮 Next Prediction", f"${latest_pred:.2f}")
        
        # Charts
        plt.style.use('dark_background')
        
        # EMA Chart 1
        st.subheader("📊 EMA Analysis (20 & 50 Days)")
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        fig1.patch.set_facecolor('black')
        ax1.set_facecolor('black')
        ax1.plot(df.index, df.Close, 'yellow', label='Close Price', linewidth=2)
        ax1.plot(df.index, ema20, 'lime', label='EMA 20', linewidth=1.5)
        ax1.plot(df.index, ema50, 'red', label='EMA 50', linewidth=1.5)
        ax1.set_title(f'{stock_symbol} - Price with EMA 20 & 50', color='white', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig1)
        
        # EMA Chart 2
        st.subheader("📊 EMA Analysis (100 & 200 Days)")
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        fig2.patch.set_facecolor('black')
        ax2.set_facecolor('black')
        ax2.plot(df.index, df.Close, 'yellow', label='Close Price', linewidth=2)
        ax2.plot(df.index, ema100, 'lime', label='EMA 100', linewidth=1.5)
        ax2.plot(df.index, ema200, 'red', label='EMA 200', linewidth=1.5)
        ax2.set_title(f'{stock_symbol} - Price with EMA 100 & 200', color='white', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        
        # Prediction Chart
        st.subheader("🔮 Prediction vs Actual")
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        fig3.patch.set_facecolor('black')
        ax3.set_facecolor('black')
        ax3.plot(y_test_clean, 'lime', label='Actual', linewidth=2)
        ax3.plot(y_pred_clean, 'red', label='Predicted', linewidth=2, linestyle='--')
        ax3.set_title(f'{stock_symbol} - Model Predictions vs Actual', color='white', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3)
        
        # Additional insights
        st.subheader("💡 Market Insights")
        insight_col1, insight_col2, insight_col3 = st.columns(3)
        
        with insight_col1:
            daily_change = df.Close.iloc[-1] - df.Close.iloc[-2] if len(df) > 1 else 0
            st.metric("📈 Daily Change", f"${daily_change:.2f}", f"{daily_change:.2f}")
        
        with insight_col2:
            volatility = df.Close.pct_change().std() * np.sqrt(252) * 100
            st.metric("⚡ Volatility (Annual)", f"{volatility:.1f}%")
        
        with insight_col3:
            avg_volume = df.Volume.tail(20).mean() if 'Volume' in df.columns else 0
            st.metric("📊 Avg Volume (20d)", f"{avg_volume:,.0f}")
        
        # Download section
        st.subheader("📥 Download Data")
        df_download = df.copy()
        df_download['EMA_20'] = ema20
        df_download['EMA_50'] = ema50
        df_download['EMA_100'] = ema100
        df_download['EMA_200'] = ema200
        
        csv_buffer = BytesIO()
        df_download.to_csv(csv_buffer, index=True)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label=f"📄 Download {stock_symbol} Dataset with EMAs",
            data=csv_data,
            file_name=f"{stock_symbol}_analysis_{dt.datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Cleanup
        plt.close('all')
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.info("💡 Try selecting a different stock or data range")

# Sidebar contact info
st.sidebar.markdown("---")
st.sidebar.subheader("📞 Contact & Support")
st.sidebar.markdown("""
**Developer:** Muhammad Haseeb Raza  
**Email:** hasiraza511@gmail.com  
**LinkedIn:** [Connect](https://www.linkedin.com/in/muhammad-haseeb-raza-71987a366/)  
**GitHub:** [View Code](https://github.com/hasiraza)

---
**🆘 Need Help?**  
If you encounter data loading issues:
1. Try ETF symbols (SPY, QQQ)
2. Use shorter date ranges
3. Check internet connection
4. Consider using sample data mode
""")

st.sidebar.info("💡 **Pro Tip:** ETFs like SPY and QQQ usually have the most reliable data feeds!")
