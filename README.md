# Stock Market Analysis and Forecasting using Deep Learning

## Project Overview

This project provides a comprehensive analysis of historical stock market data and implements a deep learning model for price forecasting. The goal is to identify trends, analyze market behavior, and build a predictive model using PyTorch.

The project is divided into two main components:
1.  **Exploratory Data Analysis (EDA):** In-depth statistical analysis and visualization of stock data.
2.  **Forecasting:** Implementation of a deep learning model for time series prediction.

---

## Technology Stack

* **Python**
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Matplotlib:** For static data visualization.
* **Plotly:** For interactive data visualization.
* **PyTorch:** For building the deep learning forecasting model.

---

## Dataset

The analysis is performed on historical stock market data from 2006 to 2018 for the following companies:

* Google (GOOGL)
* Microsoft (MSFT)
* IBM (IBM)
* Amazon (AMZN)

---

## Part 1: Exploratory Data Analysis (EDA)

The first part of the project focuses on a deep dive into the data. The key analysis steps include:

* **Descriptive Statistics:** A foundational summary of the dataset.
* **Price Distribution:** Analyzing the distribution of 'Open' and 'Close' prices.
* **Price Correlation:** Examining the correlation between 'Open' and 'Close' prices.
* **Attribute Visualization:** Visualizing the key metrics (`Open`, `High`, `Low`, `Close`, `Volume`) over time.
* **Comparative Analysis:** Comparing the 'High' and 'Close' values for each stock.
* **Time Series Decomposition:** Identifying trends and seasonality in the stock data.

---

## Part 2: Stock Price Forecasting

The second part of the project focuses on building, training, and evaluating the deep learning model using PyTorch.

* **Data Preprocessing:** Before feeding data to the model, we apply feature scaling (e.g., `MinMaxScaler`) to normalize the 'Close' prices. The time series data is then transformed into supervised learning sequences (e.g., using a 60-day window to predict the 61st day's price).

* **Model Architecture:** A **Long Short-Term Memory (LSTM)** network is implemented. LSTMs are ideal for time series forecasting as they can capture long-term dependencies. The architecture consists of multiple stacked LSTM layers followed by a Dense (Linear) layer to output the final predicted price.

* **Model Training:** The model is trained on the sequential training data. We use a regression-based loss function, such as **Mean Squared Error (MSE)**, and the **Adam optimizer** to iteratively update the model's weights.

* **Evaluation:** The model's predictive performance is measured on the unseen test data. The **Root Mean Squared Error (RMSE)** is calculated to quantify the accuracy of the predictions. Finally, the predicted prices are plotted against the actual prices to visually assess how well the model captured the market trends.
