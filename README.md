<div align="center">

# 📈 Stock Trend Prediction

### *Predicting Stock Market Trends with Deep Learning*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A powerful web application that leverages LSTM neural networks to predict stock price trends and analyze market data**

[🚀 Features](#-features) • [💻 Tech Stack](#-tech-stack) • [📦 Installation](#-installation) • [🎯 Usage](#-usage) • [📁 Project Structure](#-project-structure)

---

</div>

## 🌟 Overview

**Stock Trend Prediction** is an intelligent web application that combines the power of **Deep Learning** and **Time Series Analysis** to forecast stock prices. Built with a sophisticated LSTM (Long Short-Term Memory) neural network, this application provides:

- 📊 **Real-time stock data analysis** from Yahoo Finance
- 🔮 **AI-powered price predictions** using deep learning models
- 📈 **Technical indicators** including multiple Exponential Moving Averages (EMA)
- 📥 **Data export capabilities** for further analysis
- 🎨 **Beautiful, interactive visualizations**

---

## ✨ Features

### 📊 **Comprehensive Stock Analysis**
- Download and analyze stock data from **2000 to 2025**
- Support for any stock ticker available on Yahoo Finance
- Automatic data preprocessing and normalization

### 🤖 **Deep Learning Predictions**
- **LSTM-based** neural network for accurate price forecasting
- 70/30 train-test split for robust model evaluation
- Prediction vs. actual price comparison charts

### 📈 **Technical Indicators**
- **EMA 20 & 50** - Short-term trend analysis
- **EMA 100 & 200** - Long-term trend identification
- Visual charts showing price movements with moving averages

### 💾 **Data Export**
- Download complete datasets as CSV files
- Descriptive statistics and data summaries
- Easy integration with external analysis tools

### 🎨 **Modern Web Interface**
- Clean, responsive Bootstrap UI
- Interactive charts and visualizations
- User-friendly stock ticker input

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Backend** | Python 3.8+, Flask |
| **Machine Learning** | TensorFlow/Keras, LSTM Neural Networks |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Data Source** | yfinance (Yahoo Finance API) |
| **Visualization** | Matplotlib, Plotly |
| **Frontend** | HTML5, CSS3, Bootstrap 5 |
| **Data Storage** | CSV files |

---

## 📦 Installation

### Prerequisites

Make sure you have Python 3.8 or higher installed on your system.

```bash
python --version
```

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Stock-Trend-Prediction.git
cd Stock-Trend-Prediction
```

> ⚠️ **Note**: Replace `yourusername` with your actual GitHub username

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install manually:

```bash
pip install flask tensorflow keras pandas numpy matplotlib yfinance scikit-learn pandas-datareader
```

### Step 4: Download Pre-trained Model

Ensure the `stock_dl_model.h5` file is present in the project root directory. If not, you'll need to train the model first using the Jupyter notebook.

---

## 🚀 Usage

### Running the Application

1. **Start the Flask server:**

```bash
python app.py
```

2. **Open your browser** and navigate to:

```
http://localhost:5000
```

### Using the Web Interface

1. **Enter Stock Ticker**: Type any valid stock symbol (e.g., `AAPL`, `BTC-USD`, `TSLA`)
2. **Click Submit**: The application will:
   - Download historical data from Yahoo Finance
   - Generate EMA charts (20/50 and 100/200)
   - Display prediction vs. actual price comparison
   - Show descriptive statistics
3. **Download Data**: Click the download button to save the dataset as CSV


### Screenshots

<img width="1900" height="1024" alt="Image" src="https://github.com/user-attachments/assets/62c57950-ec00-47cb-905f-0dd913faccf5" />

<img width="1896" height="1014" alt="Image" src="https://github.com/user-attachments/assets/bc6b7598-b5d1-40c1-9690-c65a95fe7a53" />

<img width="1900" height="1011" alt="Image" src="https://github.com/user-attachments/assets/48c390d2-f9d8-473f-8159-3fae2d5d337c" />

<img width="1903" height="1015" alt="Image" src="https://github.com/user-attachments/assets/305b0260-69c8-4abb-8bfc-02fb7f4ac3f9" />

### Example Stock Tickers

- **Stocks**: `AAPL`, `GOOGL`, `MSFT`, `TSLA`, `AMZN`
- **Cryptocurrencies**: `BTC-USD`, `ETH-USD`, `SOL-USD`
- **ETFs**: `SPY`, `QQQ`, `VTI`

---

## 📁 Project Structure

```
Stock-Trend-Prediction/
│
├── app.py                          # Flask application main file
├── stock_dl_model.h5              # Pre-trained LSTM model
├── Stock Price Prediction.ipynb   # Jupyter notebook for model training
├── btc-usd.csv                    # Sample dataset
│
├── templates/
│   └── index.html                 # Main web interface template
│
├── static/
│   ├── *.png                      # Generated chart images
│   └── *_dataset.csv              # Downloaded stock datasets
│
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## 🔬 Model Details

### Architecture
- **Type**: LSTM (Long Short-Term Memory) Neural Network
- **Input**: 100 days of historical closing prices
- **Output**: Next day closing price prediction
- **Training**: 70% of historical data
- **Testing**: 30% of historical data

### Data Preprocessing
- **Normalization**: MinMaxScaler (0-1 range)
- **Sequence Length**: 100 days
- **Features**: Closing price time series

---

## 📊 Sample Output

The application generates three main visualizations:

1. **Closing Price with EMA 20 & 50** - Short-term trend analysis
2. **Closing Price with EMA 100 & 200** - Long-term trend analysis  
3. **Prediction vs. Original Trend** - Model performance visualization

---

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute to this project:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Ideas for Contributions
- 📱 Mobile-responsive improvements
- 🌐 Additional technical indicators (RSI, MACD, etc.)
- 📧 Email alerts for price predictions
- 🔄 Real-time data updates
- 📈 Additional chart types and visualizations

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Your Name**

- GitHub: [@widushan](https://github.com/yourusername)
- Email: widushanp@gmail.com


---

## 🙏 Acknowledgments

- **yfinance** - For providing free stock market data
- **TensorFlow/Keras** - For the powerful deep learning framework
- **Flask** - For the lightweight web framework
- **Bootstrap** - For the beautiful UI components

---

## ⚠️ Disclaimer

**This application is for educational and research purposes only.**

Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.

---

<div align="center">

### ⭐ If you find this project helpful, please give it a star! ⭐

[⬆ Back to Top](#-stock-trend-prediction)

Made with ❤️ using Python and Deep Learning

</div>
