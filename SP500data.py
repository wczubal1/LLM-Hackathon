import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Dummy function which should represent functionality to scrape CUSIP
# You must replace this with an actual solution or API capable of retrieving CUSIPs
def get_cusip(ticker):
    # This is a placeholder function. Implement your CUSIP retrieval as needed.
    # For instance, using web scraping or a specific financial database.
    return "CUSIP"  # Dummy value; implement actual retrieval here

def get_sp500_tickers():
    # Scrape S&P 500 tickers from Wikipedia
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    tickers_df = table[0]
    tickers = tickers_df['Symbol'].tolist()
    tickers.append('SCHO')
    return tickers

def download_sp500_data():
    tickers = get_sp500_tickers()  # Get S&P 500 tickers
    data = {}
    
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=365)
    
    all_data = pd.DataFrame()

    for ticker in tickers:
        try:
            print(f"Downloading data for {ticker}...")
            # Fetch historical data
            stock_data = yf.Ticker(ticker).history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
            
            # Fetch CUSIP
            cusip = get_cusip(ticker)
            
            # Create a DataFrame with CUSIP and historical data
            stock_data['Ticker'] = ticker
            stock_data['CUSIP'] = cusip
            stock_data.reset_index(inplace=True)
        
            stock_data = stock_data[['Ticker', 'CUSIP', 'Date', 'Open', 'High', 'Low', 'Close']]
            all_data = pd.concat([all_data, stock_data], ignore_index=True)
            stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
            #stock_data['Date'] = stock_data['Date'].dt.strftime('%Y-%m-%d')
        
        except Exception as e:
            print(f"Failed to download data for {ticker}: {e}")

    return all_data

sp500_data = download_sp500_data()

def run_down():
    sp500_data = download_sp500_data()
    
    # Save the combined data to a CSV file
    filename = "D:\Witold\Documents\Computing\LLMAgentsOfficial\Hackathon\sp500_stock_data.csv"
    sp500_data.to_csv(filename, index=False, date_format='%Y-%m-%d')
    print(f"All data saved to {filename}")

run_down()
