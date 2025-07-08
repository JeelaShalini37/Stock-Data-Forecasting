# 📊 Stock Price Forecasting with Deep Learning (LSTM)

## 🚀 Overview
This project aims to predict stock closing prices using **Long Short-Term Memory (LSTM)** networks — a type of recurrent neural network effective for time-series data. The system was trained on over a decade of historical data from four major companies (Apple, Google, Cisco, and Domino's), enabling insight into market trends and investment strategies.

---

## 🎯 Objectives
- Predict next-day stock **closing prices** using LSTM deep learning.
- Evaluate model effectiveness using **RMSE** and backtesting.
- Visualize patterns in volume, returns, correlations, and volatility.
- Enable data-driven strategy building via predictive modeling.

---

## 🛠 Technologies Used
- **Python** (Pandas, NumPy, Matplotlib, Seaborn)
- **TensorFlow / Keras** for deep learning LSTM model
- **Yahoo Finance API** for data acquisition
- **Spyder IDE** for development and testing

---

## 📈 Data
- Source: [Yahoo Finance API]
- Daily price and volume data from 2011–2021
- Tickers used: AAPL, GOOG, CSCO, DPZ

---

## 📊 Methodology
1. **Data Preprocessing**: Cleaning, normalization (min-max scaling), and formatting into sequences for LSTM input.
2. **Model Development**: Built LSTM model with 5 hidden layers, optimized using **Adam**, trained using **MSE loss**.
3. **Evaluation**: Used RMSE for model performance and implemented **backtesting** strategies (10, 30, 60-day moving averages).
4. **Visualization**: Generated trend plots, loss curves, return-vs-risk plots, and correlation heatmaps.

---

## 📉 Results
| Ticker | RMSE |
|--------|------|
| AAPL   | 5.03 |
| GOOG   | 5.26 |
| CSCO   | 1.29 |
| DPZ    | 5.26 |

*DPZ showed higher divergence and less reliable prediction due to volatility.*

---

## 📌 Folder Structure

