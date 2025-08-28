# Stock Price Prediction System 📈

A sophisticated web application built with Streamlit that predicts stock prices using Deep Learning and technical analysis with EMA indicators.

![Stock Prediction](https://img.shields.io/badge/Stock-Prediction-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red) ![Deep Learning](https://img.shields.io/badge/Deep-Learning-orange) ![Python](https://img.shields.io/badge/Python-3.x-green)

## 🌟 Features

- **Stock Selection**: Choose from 30+ popular US and Indian stocks
- **Technical Analysis**: EMA (Exponential Moving Average) indicators for 20, 50, 100, and 200 days
- **Price Prediction**: Deep Learning model for stock price forecasting
- **Performance Metrics**: RMSE, MAE, and current price analysis
- **Data Export**: Download complete dataset with EMA indicators as CSV
- **Beautiful UI**: Modern dark theme with transparent elements

## 🚀 How to Use

1. **Select a Stock**: Choose from the dropdown menu of available stocks
2. **Predict**: Click the "Predict Stock Price" button
3. **Analyze**: View EMA charts and prediction results
4. **Download**: Export the dataset for further analysis

## 📊 Technical Indicators

- **EMA 20**: 20-day Exponential Moving Average
- **EMA 50**: 50-day Exponential Moving Average  
- **EMA 100**: 100-day Exponential Moving Average
- **EMA 200**: 200-day Exponential Moving Average

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/hasiraza/stock-price-prediction.git
cd stock-price-prediction
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## 📋 Requirements

- Python 3.7+
- Streamlit
- TensorFlow/Keras
- yfinance
- scikit-learn
- pandas
- numpy
- matplotlib

## 🎯 Model Architecture

The application uses a pre-trained Deep Learning model (`stock_by_model.h5`) that has been trained on historical stock data to predict future price movements.

## 📈 Supported Stocks

### US Stocks
- Apple Inc. (AAPL)
- Microsoft Corp. (MSFT)
- Amazon.com Inc. (AMZN)
- Alphabet Inc. (GOOGL)
- Tesla Inc. (TSLA)
- And 10+ more...

### Indian Stocks
- Reliance Industries (RELIANCE.NS)
- Tata Consultancy Services (TCS.NS)
- HDFC Bank (HDFCBANK.NS)
- Infosys (INFY.NS)
- ICICI Bank (ICICIBANK.NS)
- And 10+ more...

## 📞 Contact

**Muhammad Haseeb Raza**  
- 📧 Email: [hasiraza511@gmail.com](mailto:hasiraza511@gmail.com)
- 💼 LinkedIn: [https://www.linkedin.com/in/muhammad-haseeb-raza-71987a366/](https://www.linkedin.com/in/muhammad-haseeb-raza-71987a366/)
- 🐙 GitHub: [https://github.com/hasiraza](https://github.com/hasiraza)

## ⚠️ Disclaimer

This application is for educational and research purposes only. Stock market investments are subject to market risks. The predictions provided by this system should not be considered as financial advice. Always consult with a qualified financial advisor before making investment decisions.

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/hasiraza/stock-price-prediction/issues).

## 🙏 Acknowledgments

- Yahoo Finance for providing stock data through yfinance API
- Streamlit team for the amazing web application framework
- TensorFlow/Keras for deep learning capabilities

---

**⭐ Star this repo if you found it helpful!**