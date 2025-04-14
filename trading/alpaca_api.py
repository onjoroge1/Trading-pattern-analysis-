import logging
import os
import pandas as pd
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
from flask import current_app

logger = logging.getLogger(__name__)

class AlpacaAPI:
    """Class to interact with Alpaca API for market data and trading"""
    
    def __init__(self, api_key=None, api_secret=None, base_url=None, data_url=None):
        """
        Initialize AlpacaAPI with credentials
        
        Args:
            api_key (str, optional): Alpaca API key. If None, will try to get from environment or Flask config
            api_secret (str, optional): Alpaca API secret. If None, will try to get from environment or Flask config
            base_url (str, optional): Alpaca base URL. If None, will try to get from environment or Flask config
            data_url (str, optional): Alpaca data URL. If None, will try to get from environment or Flask config
        """
        # Attempt to get credentials from params, environment, or Flask app config
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("ALPACA_API_SECRET", "")
        self.base_url = base_url or os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        self.data_url = data_url or os.environ.get("ALPACA_DATA_URL", "https://data.alpaca.markets")
        
        # Create API client if credentials provided
        self.api = None
        self._initialize_api()
    
    def _initialize_api(self):
        """Initialize the Alpaca API client with the current credentials"""
        if not self.api_key or not self.api_secret:
            logger.warning("Alpaca API credentials not provided")
            return
        
        try:
            self.api = tradeapi.REST(
                self.api_key,
                self.api_secret,
                self.base_url,
                api_version='v2'
            )
            logger.info("Alpaca API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca API client: {e}")
            self.api = None
    
    def initialize_from_app(self, app=None):
        """
        Initialize API with credentials from Flask app
        
        Args:
            app (Flask, optional): Flask app with config. If None, will use current_app
        """
        try:
            if app is None:
                # Get credentials from current_app (must be within app context)
                self.api_key = current_app.config.get('ALPACA_API_KEY', '')
                self.api_secret = current_app.config.get('ALPACA_API_SECRET', '')
                self.base_url = current_app.config.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
                self.data_url = current_app.config.get('ALPACA_DATA_URL', 'https://data.alpaca.markets')
            else:
                # Get credentials from provided app
                self.api_key = app.config.get('ALPACA_API_KEY', '')
                self.api_secret = app.config.get('ALPACA_API_SECRET', '')
                self.base_url = app.config.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
                self.data_url = app.config.get('ALPACA_DATA_URL', 'https://data.alpaca.markets')
            
            # (Re)initialize the API client
            self._initialize_api()
        except RuntimeError:
            logger.warning("Failed to initialize from app: No application context available")
        except Exception as e:
            logger.error(f"Failed to initialize from app: {e}")
    
    def get_account_info(self):
        """Get account information from Alpaca"""
        try:
            account = self.api.get_account()
            return {
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'equity': float(account.equity),
                'portfolio_value': float(account.portfolio_value),
                'status': account.status
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    def get_historical_data(self, symbol, timeframe='5Min', start_date=None, end_date=None):
        """
        Get historical bar data from Alpaca
        
        Args:
            symbol (str): Stock symbol
            timeframe (str): Time frame for bars (default: '5Min')
            start_date (datetime): Start date for data (default: 1 year ago)
            end_date (datetime): End date for data (default: today)
            
        Returns:
            pandas.DataFrame: OHLCV data
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()
        
        try:
            bars = self.api.get_bars(
                symbol,
                timeframe,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                adjustment='raw'
            ).df
            
            # Convert index timezone to UTC for consistency
            bars.index = bars.index.tz_convert('UTC')
            
            logger.info(f"Retrieved {len(bars)} {timeframe} bars for {symbol}")
            return bars
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_current_market_data(self, symbols):
        """
        Get current market data for list of symbols
        
        Args:
            symbols (list): List of stock symbols
            
        Returns:
            dict: Dictionary with symbol as key and market data as value
        """
        try:
            quotes = {}
            for symbol in symbols:
                quote = self.api.get_last_quote(symbol)
                trade = self.api.get_last_trade(symbol)
                quotes[symbol] = {
                    'bid': quote.bidprice,
                    'ask': quote.askprice,
                    'price': trade.price,
                    'timestamp': trade.timestamp
                }
            return quotes
        except Exception as e:
            logger.error(f"Error fetching current market data: {e}")
            return {}
    
    def get_market_hours(self):
        """Get market hours for today"""
        try:
            clock = self.api.get_clock()
            calendar = self.api.get_calendar(start=datetime.now().date().isoformat())
            
            return {
                'is_open': clock.is_open,
                'next_open': clock.next_open.isoformat(),
                'next_close': clock.next_close.isoformat(),
                'today_open': calendar[0].open.isoformat() if calendar else None,
                'today_close': calendar[0].close.isoformat() if calendar else None
            }
        except Exception as e:
            logger.error(f"Error fetching market hours: {e}")
            return None
    
    def submit_order(self, symbol, qty, side, order_type='market', time_in_force='day'):
        """
        Submit a trading order
        
        Args:
            symbol (str): Stock symbol
            qty (int): Order quantity
            side (str): 'buy' or 'sell'
            order_type (str): 'market', 'limit', 'stop', 'stop_limit'
            time_in_force (str): 'day', 'gtc', 'opg', 'cls', 'ioc', 'fok'
            
        Returns:
            dict: Order information
        """
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force
            )
            
            logger.info(f"Order submitted: {side} {qty} {symbol}")
            return {
                'id': order.id,
                'symbol': order.symbol,
                'qty': order.qty,
                'side': order.side,
                'status': order.status,
                'created_at': order.created_at.isoformat()
            }
        except Exception as e:
            logger.error(f"Error submitting order for {symbol}: {e}")
            return None
    
    def get_positions(self):
        """Get current positions"""
        try:
            positions = self.api.list_positions()
            return [{
                'symbol': p.symbol,
                'qty': int(p.qty),
                'side': 'long' if int(p.qty) > 0 else 'short',
                'avg_entry_price': float(p.avg_entry_price),
                'market_value': float(p.market_value),
                'unrealized_pl': float(p.unrealized_pl),
                'current_price': float(p.current_price)
            } for p in positions]
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []
