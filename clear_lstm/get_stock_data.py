import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def get_stock_data(ticker_symbol="MSFT", num_points=5000, interval="1h"):
    """
    Fetch historical stock data for a given ticker
    
    Args:
        ticker_symbol (str): Stock ticker symbol (e.g., "MSFT")
        num_points (int): Number of data points to retrieve
        interval (str): Data interval (e.g., "1h" for hourly)
        
    Returns:
        pd.DataFrame: DataFrame containing the stock data
    """
    # Define the time period 
    # For hourly data, 5000 hours ≈ 208 days, so we add extra buffer
    end_date = datetime.now()
    
    # Yahoo's limit for hourly data is typically around 730 days (2 years)
    start_date = end_date - timedelta(days=730)  
    
    try:
        # Fetch the data
        data = yf.download(ticker_symbol, start=start_date, end=end_date, interval=interval)
        
        # Check if we got any data
        if data.empty:
            print("No data retrieved. Check your ticker symbol or try a different interval.")
            return None
            
        # Check if we have enough data points
        if len(data) < num_points:
            print(f"Warning: Only {len(data)} data points available. This is less than the requested {num_points}.")
        
        # Keep only the last num_points data points
        if len(data) > num_points:
            data = data.iloc[-num_points:]
            
        return data
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def save_data(data, ticker_symbol, interval, output_dir="."):
    """
    Save the data to a CSV file
    
    Args:
        data (pd.DataFrame): Data to save
        ticker_symbol (str): Stock ticker symbol
        interval (str): Data interval used
        output_dir (str): Directory to save the file
    
    Returns:
        str: Path to the saved file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename = f"{ticker_symbol}_{interval}_data.csv"
    filepath = os.path.join(output_dir, filename)
    
    data.to_csv(filepath)
    return filepath

def main():
    ticker = "MSFT"
    num_points = 5000
    interval = "1d"
    
    print(f"Fetching {num_points} {interval} data points for {ticker}...")
    
    # Get the data
    stock_data = get_stock_data(ticker, num_points, interval)
    
    if stock_data is not None:
        # Save the data
        output_file = save_data(stock_data, ticker, interval)
        
        print(f"Stock data saved to {output_file}")
        print(f"Number of data points: {len(stock_data)}")
    else:
        print("Failed to retrieve stock data.")

if __name__ == "__main__":
    main()