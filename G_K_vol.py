import pandas as pd
import numpy as np

def garman_klass_volatility(file_path, tickers, date):
    """
    Compute the Garman-Klass volatility for a list of tickers over the last 30 days from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file containing columns ['Ticker', 'CUSIP', 'Date', 'Open', 'High', 'Low', 'Close']
    tickers (list): List of stock tickers.
    date (str): The base date in 'YYYY-MM-DD' format.
    
    Returns:
    dict: A dictionary with tickers as keys and their Garman-Klass volatility as values.
    """
    # Load data from CSV file
    df = pd.read_csv(file_path)

    # Ensure the Date column is a datetime type
    df['Date'] = pd.to_datetime(df['Date'])

    # Dictionary to store the results
    results = {}

    for ticker in tickers:
        # Check if the ticker exists in the DataFrame
        if ticker not in df['Ticker'].unique():
            results[ticker] = f"Ticker '{ticker}' not found in the data."
            continue

        # Filter the dataframe for the given ticker and sort by date
        ticker_df = df[df['Ticker'] == ticker].sort_values(by='Date')

        # Determine the last 30 days from the given date
        end_date = pd.to_datetime(date)
        start_date = end_date - pd.DateOffset(days=30)

        # Filter the data for the last 30 days
        last_30_days = ticker_df[(ticker_df['Date'] >= start_date) & (ticker_df['Date'] <= end_date)]

        # Check if there is enough data
        if len(last_30_days) < 30:
            results[ticker] = "Insufficient data for the last 30 days."
            continue

        # Apply Garman-Klass formula
        log_hl = np.log(last_30_days['High'] / last_30_days['Low'])
        log_co = np.log(last_30_days['Close'] / last_30_days['Open'])
        
        volatility_squared = (0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2).mean()
        garman_klass_vol = np.sqrt(volatility_squared)
        
        results[ticker] = garman_klass_vol

    return results
