# Stock Price Prediction using ARIMA and Gradient Boosting For Investo

## Project Overview

This notebook implements a data science project to predict stock prices using machine learning models. The project aims to compare the performance of two different approaches:

1.  **ARIMA (Autoregressive Integrated Moving Average) Model:** A traditional time series forecasting method.
2.  **Gradient Boosting Model:** A powerful machine learning ensemble method that can capture complex patterns and leverage engineered features.

## Dataset

*   **Source:** Yahoo Finance, accessed using the `yfinance` Python library.
*   **Tickers:**  Initially downloaded data for `AAPL`, `MSFT`, `GOOG`, `AMZN`, `TSLA`, but the primary analysis focuses on `AAPL`.
*   **Data Type:** Daily OHLC (Open, High, Low, Close) and Volume data.
*   **Time Period:** January 1, 2018, to December 31, 2023.

## Methodology

The notebook follows a standard data science pipeline:

1.  **Data Preparation:**
    *   Downloads historical stock data using `yfinance`.
    *   Handles missing values using forward fill (`fillna(method='ffill')`).
    *   Selects 'Close' prices for AAPL and ensures the index is in datetime format.

2.  **Exploratory Data Analysis (EDA):**
    *   Visualizes AAPL closing prices and trading volume over time.
    *   Calculates and plots 50-day and 200-day moving averages.
    *   Computes and visualizes daily returns.
    *   Generates ACF (Autocorrelation Function) and PACF (Partial Autocorrelation Function) plots to aid in ARIMA parameter selection.

3.  **Feature Engineering:**
    *   Creates lagged features for 'Close', 'Volume', and 'Daily\_Return' (lags of 1, 3, and 7 days).
    *   Calculates 7-day and 30-day rolling means of the 'Close' price (shifted to avoid look-ahead bias).
    *   Computes the percentage change in volume.

4.  **Modeling:**
    *   **ARIMA Model:**
        *   Performs stationarity checks using the Augmented Dickey-Fuller (ADF) test.
        *   Applies differencing to make the time series stationary.
        *   Uses ACF and PACF plots to guide the selection of ARIMA parameters (p, d, q).
        *   Trains an ARIMA(1, 1, 1) model (example parameters).
        *   Forecasts future prices and inverts differencing to obtain forecasts in the original scale.
    *   **Gradient Boosting Model:**
        *   Prepares data with engineered features for the Gradient Boosting model.
        *   Splits data into training and testing sets.
        *   Scales features using `MinMaxScaler`.
        *   Performs hyperparameter tuning using `GridSearchCV` to find optimal parameters for `GradientBoostingRegressor`.
        *   Trains the best Gradient Boosting model.
        *   Predicts stock prices on the test set.

5.  **Model Evaluation:**
    *   Evaluates both ARIMA and Gradient Boosting models using the following metrics:
        *   **RMSE (Root Mean Squared Error)**
        *   **MAE (Mean Absolute Error)**
        *   **MAPE (Mean Absolute Percentage Error)**
    *   Compares the performance of the two models based on these metrics.

6.  **Report and Presentation (Conceptual - Notebook includes code for analysis and results):**
    *   The notebook is structured to facilitate the creation of a comprehensive report and presentation (outline provided in the notebook comments and previous responses).

## Results

The model evaluation yielded the following results for predicting AAPL stock prices on the test set:

**Model Evaluation Metrics:**

| Metric | ARIMA Model | Gradient Boosting Model |
|---|---|---|
| **RMSE** | 19.76 | **8.93** |
| **MAE** | 17.23 | **6.17** |
| **MAPE** | 9.83% | **3.40%** |

**Key Findings:**

*   **Gradient Boosting significantly outperformed the ARIMA model** across all evaluation metrics (RMSE, MAE, MAPE).
*   The Gradient Boosting model achieved a much lower MAPE of **3.40%** compared to ARIMA's **9.83%**, indicating a considerably better relative accuracy in price prediction.
*   These results suggest that for AAPL stock price prediction in this context, a non-linear model like Gradient Boosting, leveraging engineered features, is more effective than a traditional linear time series model like ARIMA.

**Visualizations:**

The notebook includes various visualizations to support the analysis, including:

*   Time series plots of AAPL closing price and volume.
*   Plots of closing price with moving averages.
*   Daily returns plot.
*   ACF and PACF plots for ARIMA analysis.
*   Forecast plots for both ARIMA and Gradient Boosting models compared to actual prices.

## Technologies Used

*   **Python Libraries:**
    *   `yfinance`: For downloading stock market data.
    *   `pandas`: For data manipulation and analysis.
    *   `numpy`: For numerical operations.
    *   `statsmodels`: For ARIMA modeling and time series analysis.
    *   `scikit-learn`: For Gradient Boosting, model evaluation, data splitting, and feature scaling.
    *   `matplotlib` and `seaborn`: For data visualization.

*   **Environment:** Google Colab (cloud-based Python environment).

## Usage

To run this notebook in Google Colab:

1.  **Open in Colab:** Click on "Open in Colab" button (if available) or upload the notebook file to your Google Drive and open it in Google Colab.
2.  **Install Libraries (if necessary):** The notebook includes `!pip install` commands at the beginning to install required libraries if they are not already present in your Colab environment. Run these cells first if needed.
3.  **Execute Cells Sequentially:** Run the notebook cells in order from top to bottom. You can use `Shift+Enter` to run a cell and move to the next one.
4.  **Observe Outputs and Visualizations:** Check the printed outputs and generated plots after running each section of the code.
