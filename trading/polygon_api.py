import os
import logging
import pandas as pd
from datetime import datetime, timedelta
import requests
from flask import current_app, g

logger = logging.getLogger(__name__)

class PolygonAPI:
    """Class to interact with Polygon.io API for market data"""
    
    def __init__(self, api_key=None):
        """
        Initialize PolygonAPI with API key
        
        Args:
            api_key (str, optional): Polygon API key. If None, will try to get from environment
        """
        self.api_key = api_key or os.environ.get('POLYGON_API_KEY')
        self.base_url = 'https://api.polygon.io'
        
        if not self.api_key:
            logger.warning("Polygon API key not provided")
        else:
            logger.info(f"Polygon API initialized with key: {self.api_key[:5]}...")
    
    def initialize_from_app(self, app=None):
        """
        Initialize API with credentials from Flask app
        
        Args:
            app (Flask, optional): Flask app with config. If None, will use current_app
        """
        if app is None:
            app = current_app
            
        self.api_key = self.api_key or app.config.get('POLYGON_API_KEY') or os.environ.get('POLYGON_API_KEY')
    
    def get_historical_data(self, symbol, timespan='minute', multiplier=5, start_date=None, end_date=None):
        """
        Get historical bar data from Polygon
        
        Args:
            symbol (str): Stock symbol
            timespan (str): Time frame for bars (default: 'minute')
            multiplier (int): Multiplier for timespan (default: 5 for 5Min)
            start_date (datetime): Start date for data (default: 1 year ago)
            end_date (datetime): End date for data (default: today)
            
        Returns:
            pandas.DataFrame: OHLCV data
        """
        try:
            # Set default dates if not provided
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=365)
            
            # Format dates as YYYY-MM-DD
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Make sure symbol is uppercase
            symbol = symbol.upper()
            
            # Construct API URL
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_str}/{end_str}"
            
            # Make API request
            params = {
                'apiKey': self.api_key,
                'sort': 'asc',
                'limit': 50000  # Maximum allowed
            }
            
            logger.info(f"Fetching data for {symbol} from {start_str} to {end_str} with timespan {multiplier}/{timespan}")
            response = requests.get(url, params=params)
            
            # Print full response for debugging
            logger.info(f"Polygon API response status: {response.status_code}")
            
            # Check for errors
            response.raise_for_status()
            data = response.json()
            
            # Print response status for debugging
            logger.info(f"Polygon API response status: {data.get('status')}")
            
            if 'results' not in data or not data['results']:
                logger.warning(f"No data returned for {symbol} from {start_str} to {end_str}")
                return pd.DataFrame()
            
            # Log success
            logger.info(f"Successfully fetched {len(data['results'])} bars for {symbol}")
            
            # Convert to DataFrame
            df = pd.DataFrame(data['results'])
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                't': 'timestamp'
            })
            
            # Convert timestamp from milliseconds to datetime and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            if 'response' in locals():
                try:
                    logger.error(f"Response content: {response.text}")
                except:
                    pass
            return pd.DataFrame()
    
    def get_current_market_data(self, symbols):
        """
        Get current market data for list of symbols
        
        Args:
            symbols (list): List of stock symbols
            
        Returns:
            dict: Dictionary with symbol as key and market data as value
        """
        result = {}
        try:
            for symbol in symbols:
                # Use the last close data as current data
                url = f"{self.base_url}/v2/aggs/ticker/{symbol}/prev"
                params = {'apiKey': self.api_key}
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if 'results' in data and data['results']:
                    result[symbol] = {
                        'symbol': symbol,
                        'last_price': data['results'][0]['c'],
                        'last_trade_time': datetime.fromtimestamp(data['results'][0]['t'] / 1000),
                        'change_percent': (data['results'][0]['c'] - data['results'][0]['o']) / data['results'][0]['o'] * 100
                    }
                else:
                    logger.warning(f"No current data available for {symbol}")
            
            return result
        except Exception as e:
            logger.error(f"Error getting current market data: {e}")
            return result
    
    def get_market_hours(self):
        """Get market hours for today (always returns market is open for demo purposes)"""
        try:
            # For our demo, we'll assume the market is always open during standard hours
            now = datetime.now()
            today = now.strftime('%Y-%m-%d')
            
            # Define standard market hours (9:30 AM - 4:00 PM ET)
            market_open = datetime.strptime(f"{today} 09:30:00", "%Y-%m-%d %H:%M:%S")
            market_close = datetime.strptime(f"{today} 16:00:00", "%Y-%m-%d %H:%M:%S")
            
            # Always return that the market is open for demo purposes
            # In real-world, you would check:
            # - If today is a weekday (Monday to Friday)
            # - If current time is between 9:30 AM and 4:00 PM ET
            # - If today is not a market holiday
            
            logger.info("Market hours check - returning market is OPEN for demo purposes")
            return {
                'is_open': True,  # Always open for demo
                'open_time': market_open,
                'close_time': market_close,
                'next_open': None,
                'next_close': None
            }
        except Exception as e:
            logger.error(f"Error fetching market hours: {e}")
            return {
                'is_open': True,  # Still return open even on error
                'open_time': datetime.now().replace(hour=9, minute=30, second=0, microsecond=0),
                'close_time': datetime.now().replace(hour=16, minute=0, second=0, microsecond=0),
                'next_open': None,
                'next_close': None
            }
    
    def get_account_info(self):
        """
        Get account information (placeholder - Polygon doesn't provide trading)
        
        Returns:
            dict: Basic account information
        """
        logger.info("Polygon does not provide account information as it's a data provider only")
        return {
            'account_number': 'POLYGON_DATA_ONLY',
            'cash': 0,
            'portfolio_value': 0,
            'status': 'ACTIVE',
            'trading_blocked': True,
            'transfers_blocked': True,
            'account_blocked': False,
            'created_at': datetime.now(),
            'message': 'Polygon is a data provider only, not a broker. Trading functionality is not available.'
        }
    
    def get_positions(self):
        """
        Get current positions (placeholder - Polygon doesn't provide trading)
        
        Returns:
            list: Empty list as positions aren't available
        """
        logger.info("Polygon does not provide position information as it's a data provider only")
        return []
    
    def submit_order(self, symbol, qty, side, order_type='market', time_in_force='day'):
        """
        Submit a trading order (placeholder - Polygon doesn't provide trading)
        
        Args:
            symbol (str): Stock symbol
            qty (int): Order quantity
            side (str): 'buy' or 'sell'
            order_type (str): 'market', 'limit', 'stop', 'stop_limit'
            time_in_force (str): 'day', 'gtc', 'opg', 'cls', 'ioc', 'fok'
            
        Returns:
            dict: Order information (simulated)
        """
        logger.info("Polygon does not provide order submission as it's a data provider only")
        return {
            'id': 'polygon-simulated-order',
            'client_order_id': 'polygon-simulated',
            'symbol': symbol,
            'side': side,
            'qty': qty,
            'order_type': order_type,
            'time_in_force': time_in_force,
            'status': 'rejected',
            'price': 0,
            'message': 'Polygon is a data provider only, not a broker. Trading is not available.'
        }