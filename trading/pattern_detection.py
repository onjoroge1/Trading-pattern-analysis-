import numpy as np
import pandas as pd
import logging
from datetime import datetime, time

logger = logging.getLogger(__name__)

class PatternDetector:
    """Class to detect reversal patterns in stock data"""
    
    def __init__(self, min_consecutive_candles=3, doji_threshold=0.1, hammer_threshold=0.3):
        """
        Initialize PatternDetector with detection parameters
        
        Args:
            min_consecutive_candles (int): Minimum number of consecutive candles for a trend
            doji_threshold (float): Maximum body/range ratio for a doji
            hammer_threshold (float): Maximum upper shadow to lower shadow ratio for a hammer
        """
        self.min_consecutive_candles = min_consecutive_candles
        self.doji_threshold = doji_threshold
        self.hammer_threshold = hammer_threshold
        logger.info("PatternDetector initialized with parameters: "
                    f"min_consecutive_candles={min_consecutive_candles}, "
                    f"doji_threshold={doji_threshold}, "
                    f"hammer_threshold={hammer_threshold}")
    
    def is_market_open_time(self, timestamp):
        """
        Check if the timestamp is within market opening time (9:30 - 10:00 AM)
        
        Args:
            timestamp (datetime): Timestamp to check
            
        Returns:
            bool: True if within opening time, False otherwise
        """
        market_open = time(9, 30)
        early_market_end = time(10, 0)
        ts_time = timestamp.time()
        return market_open <= ts_time < early_market_end
    
    def detect_consecutive_candles(self, df):
        """
        Detect consecutive bullish or bearish candles
        
        Args:
            df (pandas.DataFrame): OHLC dataframe
            
        Returns:
            pandas.DataFrame: DataFrame with 'consecutive_bullish' and 'consecutive_bearish' columns
        """
        # Create a copy of the dataframe to avoid modifying the original
        result_df = df.copy()
        
        # Determine bullish and bearish candles
        result_df['bullish'] = result_df['close'] > result_df['open']
        result_df['bearish'] = result_df['close'] < result_df['open']
        
        # Initialize consecutive candle columns
        result_df['consecutive_bullish'] = 0
        result_df['consecutive_bearish'] = 0
        
        # Calculate consecutive bullish candles
        bullish_count = 0
        for i in range(len(result_df)):
            if result_df['bullish'].iloc[i]:
                bullish_count += 1
            else:
                bullish_count = 0
            result_df['consecutive_bullish'].iloc[i] = bullish_count
        
        # Calculate consecutive bearish candles
        bearish_count = 0
        for i in range(len(result_df)):
            if result_df['bearish'].iloc[i]:
                bearish_count += 1
            else:
                bearish_count = 0
            result_df['consecutive_bearish'].iloc[i] = bearish_count
        
        # Identify potential reversal patterns based on consecutive candles
        result_df['potential_bullish_reversal'] = (
            result_df['consecutive_bearish'].shift(1) >= self.min_consecutive_candles) & (
            result_df['bullish'])
        
        result_df['potential_bearish_reversal'] = (
            result_df['consecutive_bullish'].shift(1) >= self.min_consecutive_candles) & (
            result_df['bearish'])
        
        return result_df
    
    def detect_doji(self, df):
        """
        Detect Doji candlestick patterns
        
        Args:
            df (pandas.DataFrame): OHLC dataframe
            
        Returns:
            pandas.DataFrame: DataFrame with 'is_doji' column
        """
        result_df = df.copy()
        
        # Calculate candle body and range
        result_df['body_size'] = abs(result_df['close'] - result_df['open'])
        result_df['candle_range'] = result_df['high'] - result_df['low']
        
        # Detect doji when body is very small compared to the range
        result_df['is_doji'] = (
            (result_df['body_size'] / result_df['candle_range']) < self.doji_threshold) & (
            result_df['candle_range'] > 0)  # Avoid division by zero
        
        return result_df
    
    def detect_hammer(self, df):
        """
        Detect Hammer candlestick patterns
        
        Args:
            df (pandas.DataFrame): OHLC dataframe
            
        Returns:
            pandas.DataFrame: DataFrame with 'is_hammer' column
        """
        result_df = df.copy()
        
        # Calculate upper and lower shadows
        result_df['upper_shadow'] = result_df.apply(
            lambda row: row['high'] - max(row['open'], row['close']), axis=1)
        
        result_df['lower_shadow'] = result_df.apply(
            lambda row: min(row['open'], row['close']) - row['low'], axis=1)
        
        # Detect hammer - small upper shadow, long lower shadow
        result_df['is_hammer'] = (
            (result_df['upper_shadow'] < (result_df['lower_shadow'] * self.hammer_threshold)) & 
            (result_df['lower_shadow'] > 0) &  # Ensure there is a lower shadow
            (result_df['body_size'] < result_df['lower_shadow']))  # Body smaller than lower shadow
        
        return result_df
    
    def calculate_orb(self, df):
        """
        Calculate Opening Range Breakout (ORB) levels
        
        Args:
            df (pandas.DataFrame): OHLC dataframe with datetime index
            
        Returns:
            pandas.DataFrame: DataFrame with ORB high and low columns
        """
        result_df = df.copy()
        
        # Initialize ORB columns
        result_df['orb_high'] = np.nan
        result_df['orb_low'] = np.nan
        result_df['above_orb_high'] = False
        result_df['below_orb_low'] = False
        
        # Group by date
        result_df['date'] = result_df.index.date
        
        dates = result_df['date'].unique()
        
        for date in dates:
            day_df = result_df[result_df['date'] == date]
            
            # Get the first 15-min of trading (9:30 - 9:45)
            market_open = datetime.combine(date, time(9, 30))
            orb_end = datetime.combine(date, time(9, 45))
            
            orb_candles = day_df[
                (day_df.index.tz_localize(None) >= market_open) & 
                (day_df.index.tz_localize(None) < orb_end)
            ]
            
            if len(orb_candles) > 0:
                orb_high = orb_candles['high'].max()
                orb_low = orb_candles['low'].min()
                
                # Set ORB levels for all candles on this day
                result_df.loc[result_df['date'] == date, 'orb_high'] = orb_high
                result_df.loc[result_df['date'] == date, 'orb_low'] = orb_low
                
                # Check if price is above ORB high or below ORB low
                result_df.loc[result_df['date'] == date, 'above_orb_high'] = result_df['close'] > orb_high
                result_df.loc[result_df['date'] == date, 'below_orb_low'] = result_df['close'] < orb_low
        
        # Calculate distance from ORB levels
        result_df['distance_from_orb_high'] = (result_df['close'] - result_df['orb_high']) / result_df['orb_high']
        result_df['distance_from_orb_low'] = (result_df['orb_low'] - result_df['close']) / result_df['orb_low']
        
        # Clean up
        result_df = result_df.drop('date', axis=1)
        
        return result_df
    
    def detect_all_patterns(self, df, rsi_df=None):
        """
        Detect all reversal patterns in the OHLC data
        
        Args:
            df (pandas.DataFrame): OHLC dataframe
            rsi_df (pandas.DataFrame, optional): DataFrame with RSI values
            
        Returns:
            pandas.DataFrame: DataFrame with all pattern detection columns
        """
        # Make a copy of the dataframe
        result_df = df.copy()
        
        # Apply all pattern detection methods
        result_df = self.detect_consecutive_candles(result_df)
        result_df = self.detect_doji(result_df)
        result_df = self.detect_hammer(result_df)
        result_df = self.calculate_orb(result_df)
        
        # Add RSI information if provided
        if rsi_df is not None and 'rsi' in rsi_df.columns:
            result_df['rsi'] = rsi_df['rsi']
            
            # Market timing flags for RSI interpretation
            result_df['in_opening_hour'] = result_df.index.map(self.is_market_open_time)
            
            # RSI conditions
            result_df['rsi_overbought'] = result_df['rsi'] > 70
            result_df['rsi_oversold'] = result_df['rsi'] < 30
            
            # RSI bias based on market timing
            result_df['rsi_long_bias'] = (
                # Standard time: oversold -> long bias
                (~result_df['in_opening_hour'] & result_df['rsi_oversold']) |
                # Opening time: overbought -> long bias (momentum)
                (result_df['in_opening_hour'] & result_df['rsi_overbought'])
            )
            
            result_df['rsi_short_bias'] = (
                # Standard time: overbought -> short bias
                (~result_df['in_opening_hour'] & result_df['rsi_overbought']) |
                # Opening time: oversold -> short bias (momentum)
                (result_df['in_opening_hour'] & result_df['rsi_oversold'])
            )
        
        # Flag to detect any reversal pattern
        result_df['has_reversal_pattern'] = (
            result_df['is_doji'] | 
            result_df['is_hammer'] | 
            result_df['potential_bullish_reversal'] | 
            result_df['potential_bearish_reversal']
        )
        
        return result_df
