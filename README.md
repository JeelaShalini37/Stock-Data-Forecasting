# 📊 Stock Price Forecasting with Deep Learning (LSTM)

## 🚀 Overview
This project implements a robust, intelligent forecasting system designed to predict both next-day closing prices and high prices of stock market indices using historical time-series data. Instead of relying on a single predictive model, it evaluates and dynamically selects the best-performing model among CNN, LSTM, and RNN architectures for each prediction point, thereby optimizing accuracy under changing market dynamics.

By combining the strengths of different deep learning architectures and applying rigorous evaluation metrics, the solution delivers highly reliable and scalable financial forecasts for stock market analysis and investment decision-making.


## 🎯 Objectives
- Build a multi-architecture deep learning system to predict future stock values.

- Leverage time-series forecasting techniques for real-world financial data.

- Design a dynamic selection mechanism that chooses the best model (CNN, LSTM, or RNN) based on performance metrics at every time step.

- Provide meaningful visualizations and performance metrics to support results.


---

## 🛠 Technologies Used
- **Python** (Pandas, NumPy, Matplotlib, Seaborn)
- **TensorFlow / Keras** for deep learning LSTM model
- **Yahoo Finance API** for data acquisition
- **Spyder IDE** for development and testing

---

## 📈 Data
Source: Publicly available datasets (e.g., Yahoo Finance or similar)

Index Used: Shanghai Composite Index (000001)

Time Range: 1997–2022

Data Frequency: Daily

Features Considered:

Open

High

Low

Close

Adj Close

Volume

Each feature plays a crucial role in capturing market trends and investor behavior.
---

## 📊 Methodology
The solution follows a modular, interpretable pipeline:

1. Data Preprocessing
Handled missing values and cleaned dataset.

Used MinMaxScaler for normalization of input features.

Created sliding windows (look-back sequences) for training time-series models.

2. Model Development
Developed three separate neural network models:

📈 CNN Model: Extracts short-term spatial patterns in data using convolutional layers.

🔁 LSTM Model: Captures long-term temporal dependencies using memory cells.

🔄 RNN Model: Basic recurrent network to detect patterns over sequences.

Each model was trained independently on the same dataset.

3. Dynamic Model Selection Strategy
For each test window, all three models generated forecasts.

Evaluation metrics (RMSE, MAE, MAPE) were calculated.

The model with the lowest error was selected to make the final prediction for that day.

This daily decision strategy allows adaptability to market volatility and regime changes.

4. Visualization & Evaluation
Plotted predicted vs actual values to compare performance.

Created error metric dashboards to analyze accuracy over time.

Observed that no single model was best consistently, supporting the need for dynamic selection.


---

## 📉 Results
Metric	Closing Price Forecast	High Price Forecast
RMSE	~0.025	~0.03
MAPE	< 5%	< 6%
Best Model Hit Rate	~92%	~88%

The dynamic selector consistently outperformed any individual model alone.

Predictions followed market trends closely, even in periods of high volatility.

Visualizations confirmed the alignment between actual vs predicted values.


---



