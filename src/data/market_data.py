import yfinance as yf 

def get_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data for a given ticker symbol and date range.

    Parameters:
    ticker (str): The stock ticker symbol (e.g., 'AAPL' for Apple Inc.).
    start_date (str): The start date for the data in 'YYYY-MM-DD' format.
    end_date (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
    pandas.DataFrame: A DataFrame containing the historical stock data.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def get_stock_info(ticker):
    """
    Fetch basic information about a stock given its ticker symbol.

    Parameters:
    ticker (str): The stock ticker symbol (e.g., 'AAPL' for Apple Inc.).

    Returns:
    dict: A dictionary containing basic information about the stock.
    """
    stock = yf.Ticker(ticker)
    return stock.info