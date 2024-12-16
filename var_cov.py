import pandas as pd
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
from typing import Annotated
import os
from openai import OpenAI

def sp500_fundaments(file_path: Annotated[str, "File with stock fundamentals data"]):

    client = OpenAI(
    # Defaults to os.environ.get("OPENAI_API_KEY")
    )

    # Step 1: Create a Vector Store
    vector_store = client.beta.vector_stores.create(name="Fundamental Stock Data")
    print("Vector Store created:", vector_store.id)  # This is your vector_store.id

    # Step 2: Prepare Files for Upload
    file_paths = [file_path]
    file_streams = [open(path, "rb") for path in file_paths]

    # Step 3: Upload Files and Add to Vector Store (with status polling)
    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id, files=file_streams
    )

    # Step 4: Verify Completion (Optional)
    print("File batch status:", file_batch.status)
    #print("Uploaded file count:", file_batch.file_counts.processed)

    return vector_store.id

def calc_cov_matrix(file_path: Annotated[str, "File with stock price data"], tickers: Annotated[list, "List of stock tickers"], date: Annotated[str, "The base date in 'YYYY-MM-DD' format"]):

    # Read the CSV file into a DataFrame
    data = pd.read_csv(file_path)

    # Convert 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # Ensure the given date is of datetime type
    date = pd.to_datetime(date)

    # Filter data for the selected stocks
    stock_data = data[data['Ticker'].isin(tickers)]

    # Filter data for the last 30 days prior to the given date
    start_date = date - pd.Timedelta(days=30)
    filtered_data = stock_data[(stock_data['Date'] > start_date) & (stock_data['Date'] <= date)]

    # Pivot the DataFrame to have Dates as rows and Tickers as columns
    pivot_data = filtered_data.pivot(index='Date', columns='Ticker', values='Close')

    # Calculate daily returns
    returns = pivot_data.pct_change().dropna()

    # Calculate the covariance matrix of the returns
    covariance_matrix = returns.cov()

    return covariance_matrix

def portfolio_volatility(file_path: Annotated[str, "File with stock price data"], tickers: Annotated[list, "List of stock tickers"], date: Annotated[str, "The base date in 'YYYY-MM-DD' format"], weights: Annotated[list, "Weights of stocks in a portfolio"]) -> Annotated[float, "portfolio risk or portfolio volatility"]:

    # Convert weights from percentages to decimals
    weights = np.array(weights) / 100

    covariance_matrix=calc_cov_matrix(file_path, tickers, date)
    
    # Calculate portfolio variance
    portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))

    # Calculate portfolio volatility (standard deviation)
    portfolio_volatility = np.sqrt(portfolio_variance)

    return portfolio_volatility
    #return covariance_matrix




def optimize_for_target_risk(file_path: Annotated[str, "File with stock price data"], tickers: Annotated[list, "List of stock tickers"], date: Annotated[str, "The base date in 'YYYY-MM-DD' format"], weights: Annotated[list, "Weights of stocks in a portfolio"], target_risk: Annotated[float, "Target for portfolio risk"]):

    # Convert weights from percentages to decimals
    weights = np.array(weights) / 100
    initial_weights = weights

    covariance_matrix=calc_cov_matrix(file_path, tickers, date)
    print(covariance_matrix)

    # Objective function to minimize
    def objective(weights,initial_weights,covariance_matrix):
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        print (portfolio_volatility)
        print(weights)
        #print(100*(portfolio_volatility - target_risk)**2 + np.sum((weights - initial_weights)**2))
        return 100*(portfolio_volatility - target_risk)**2 + np.sum((weights - initial_weights)**2)/120

    # Constraints: weights must sum to 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    # Bounds for weights: each weight between 0 and 1
    bounds = [(0, 1) for _ in tickers]
    #print(bounds)

    # Initial guess for weights
    #initial_weights = np.array([1.0 / len(tickers)] * len(tickers))
    #print(initial_weights)
    
    # Optimization
    result = minimize(objective, weights, args=(initial_weights,covariance_matrix), method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x*100  # Convert weights back to percentages

#print(optimize_for_target_risk('D:/Witold/Documents/Computing/LLMAgentsOfficial/Hackathon/sp500_stock_data.csv',['NVDA','SCHO','UAL'],'2024-08-05',[50, 0, 50],0.02))
#print(sp500_fundaments('D:/Witold/Documents/Computing/LLMAgentsOfficial/Hackathon/sp500_stock_data_fundaments.txt'))
