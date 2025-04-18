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
        consecutive_bullish = []
        
        for i in range(len(result_df)):
            if result_df['bullish'].iloc[i]:
                bullish_count += 1
            else:
                bullish_count = 0
            consecutive_bullish.append(bullish_count)
        
        result_df.loc[:, 'consecutive_bullish'] = consecutive_bullish
        
        # Calculate consecutive bearish candles
        bearish_count = 0
        consecutive_bearish = []
        
        for i in range(len(result_df)):
            if result_df['bearish'].iloc[i]:
                bearish_count += 1
            else:
                bearish_count = 0
            consecutive_bearish.append(bearish_count)
        
        result_df.loc[:, 'consecutive_bearish'] = consecutive_bearish
        
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
    
    def detect_engulfing(self, df):
        """
        Detect Engulfing patterns (Bullish and Bearish)
        
        Args:
            df (pandas.DataFrame): OHLC dataframe
            
        Returns:
            pandas.DataFrame: DataFrame with engulfing pattern columns
        """
        result_df = df.copy()
        
        # Calculate if current candle's body engulfs the previous candle's body
        result_df['bullish_engulfing'] = False
        result_df['bearish_engulfing'] = False
        
        # Skip pattern detection if dataframe is empty or has only one row
        if len(result_df) <= 1:
            return result_df
            
        # Create arrays to hold pattern results
        bullish_engulfing = [False] * len(result_df)
        bearish_engulfing = [False] * len(result_df)
            
        for i in range(1, len(result_df)):
            try:
                curr_open = result_df['open'].iloc[i]
                curr_close = result_df['close'].iloc[i]
                prev_open = result_df['open'].iloc[i-1]
                prev_close = result_df['close'].iloc[i-1]
                
                # Bullish engulfing: Current candle is bullish and engulfs the previous bearish candle
                bullish_engulfing[i] = (
                    (curr_close > curr_open) and  # Current candle is bullish
                    (prev_close < prev_open) and  # Previous candle is bearish
                    (curr_open < prev_close) and  # Current open below previous close
                    (curr_close > prev_open)      # Current close above previous open
                )
                
                # Bearish engulfing: Current candle is bearish and engulfs the previous bullish candle
                bearish_engulfing[i] = (
                    (curr_close < curr_open) and  # Current candle is bearish
                    (prev_close > prev_open) and  # Previous candle is bullish
                    (curr_open > prev_close) and  # Current open above previous close
                    (curr_close < prev_open)      # Current close below previous open
                )
            except Exception as e:
                # Just skip this iteration if there's any error
                continue
        
        # Assign the pattern results to the dataframe
        result_df['bullish_engulfing'] = bullish_engulfing
        result_df['bearish_engulfing'] = bearish_engulfing
        
        return result_df
    
    def detect_stars(self, df):
        """
        Detect Morning Star and Evening Star patterns
        
        Args:
            df (pandas.DataFrame): OHLC dataframe
            
        Returns:
            pandas.DataFrame: DataFrame with star pattern columns
        """
        result_df = df.copy()
        
        result_df['morning_star'] = False
        result_df['evening_star'] = False
        
        # Skip if dataframe is too small for pattern detection
        if len(result_df) < 3:
            return result_df
        
        # Create arrays to hold pattern results
        morning_star = [False] * len(result_df)
        evening_star = [False] * len(result_df)
            
        # We need at least 3 candles for star patterns
        for i in range(2, len(result_df)):
            try:
                # Get the three candles for the pattern
                first_open = result_df['open'].iloc[i-2]
                first_close = result_df['close'].iloc[i-2]
                second_open = result_df['open'].iloc[i-1]
                second_close = result_df['close'].iloc[i-1]
                third_open = result_df['open'].iloc[i]
                third_close = result_df['close'].iloc[i]
                
                # Calculate body sizes
                first_body = abs(first_close - first_open)
                second_body = abs(second_close - second_open)
                third_body = abs(third_close - third_open)
                
                # Morning Star: First bearish, second small body (doji-like), third bullish
                if (first_close < first_open and  # First candle bearish
                    second_body < 0.3 * first_body and  # Second candle small body
                    third_close > third_open and  # Third candle bullish
                    third_close > (first_open + first_close) / 2):  # Third closes above midpoint of first
                    
                    morning_star[i] = True
                
                # Evening Star: First bullish, second small body (doji-like), third bearish
                if (first_close > first_open and  # First candle bullish
                    second_body < 0.3 * first_body and  # Second candle small body
                    third_close < third_open and  # Third candle bearish
                    third_close < (first_open + first_close) / 2):  # Third closes below midpoint of first
                    
                    evening_star[i] = True
            except Exception as e:
                # Just skip this iteration if there's any error
                continue
        
        # Assign the pattern results to the dataframe
        result_df['morning_star'] = morning_star
        result_df['evening_star'] = evening_star
        
        return result_df
    
    def detect_shooting_star(self, df):
        """
        Detect Shooting Star pattern
        
        Args:
            df (pandas.DataFrame): OHLC dataframe
            
        Returns:
            pandas.DataFrame: DataFrame with shooting star column
        """
        result_df = df.copy()
        
        # Calculate body and shadow sizes if not already calculated
        if 'body_size' not in result_df.columns:
            result_df['body_size'] = abs(result_df['close'] - result_df['open'])
        
        if 'upper_shadow' not in result_df.columns:
            result_df['upper_shadow'] = result_df.apply(
                lambda row: row['high'] - max(row['open'], row['close']), axis=1)
            
        if 'lower_shadow' not in result_df.columns:
            result_df['lower_shadow'] = result_df.apply(
                lambda row: min(row['open'], row['close']) - row['low'], axis=1)
        
        # Shooting Star: Small body, long upper shadow, little to no lower shadow
        result_df['shooting_star'] = (
            (result_df['upper_shadow'] > 2 * result_df['body_size']) &  # Long upper shadow
            (result_df['lower_shadow'] < 0.1 * result_df['upper_shadow']) &  # Little to no lower shadow
            (result_df['body_size'] > 0)  # Ensure there is a body
        )
        
        return result_df
    
    def detect_piercing_patterns(self, df):
        """
        Detect Piercing Line and Dark Cloud Cover patterns
        
        Args:
            df (pandas.DataFrame): OHLC dataframe
            
        Returns:
            pandas.DataFrame: DataFrame with piercing pattern columns
        """
        result_df = df.copy()
        
        result_df['piercing_line'] = False
        result_df['dark_cloud_cover'] = False
        
        # Skip if dataframe is too small for pattern detection
        if len(result_df) <= 1:
            return result_df
        
        # Create arrays to hold pattern results
        piercing_line = [False] * len(result_df)
        dark_cloud_cover = [False] * len(result_df)
            
        for i in range(1, len(result_df)):
            try:
                prev_open = result_df['open'].iloc[i-1]
                prev_close = result_df['close'].iloc[i-1]
                prev_body = abs(prev_close - prev_open)
                
                curr_open = result_df['open'].iloc[i]
                curr_close = result_df['close'].iloc[i]
                curr_body = abs(curr_close - curr_open)
                
                # Piercing Line: Previous bearish, current bullish open below previous low,
                # close above the midpoint of previous candle
                if (prev_close < prev_open and  # Previous bearish
                    curr_close > curr_open and  # Current bullish
                    curr_open < prev_close and  # Current opens below previous close
                    curr_close > (prev_open + prev_close) / 2 and  # Current closes above midpoint
                    curr_body > 0.6 * prev_body):  # Current body size significant
                    
                    piercing_line[i] = True
                
                # Dark Cloud Cover: Previous bullish, current bearish open above previous high,
                # close below the midpoint of previous candle
                if (prev_close > prev_open and  # Previous bullish
                    curr_close < curr_open and  # Current bearish
                    curr_open > prev_close and  # Current opens above previous close
                    curr_close < (prev_open + prev_close) / 2 and  # Current closes below midpoint
                    curr_body > 0.6 * prev_body):  # Current body size significant
                    
                    dark_cloud_cover[i] = True
            except Exception as e:
                # Just skip this iteration if there's any error
                continue
        
        # Assign the pattern results to the dataframe
        result_df['piercing_line'] = piercing_line
        result_df['dark_cloud_cover'] = dark_cloud_cover
        
        return result_df
    
    def detect_three_candle_patterns(self, df):
        """
        Detect Three White Soldiers and Three Black Crows patterns
        
        Args:
            df (pandas.DataFrame): OHLC dataframe
            
        Returns:
            pandas.DataFrame: DataFrame with three-candle pattern columns
        """
        result_df = df.copy()
        
        result_df['three_white_soldiers'] = False
        result_df['three_black_crows'] = False
        
        # Skip if dataframe is too small for pattern detection
        if len(result_df) < 3:
            return result_df
        
        # Create arrays to hold pattern results
        three_white_soldiers = [False] * len(result_df)
        three_black_crows = [False] * len(result_df)
            
        # Need at least 3 candles for these patterns
        for i in range(2, len(result_df)):
            try:
                # Check Three White Soldiers
                if (result_df['close'].iloc[i-2] > result_df['open'].iloc[i-2] and  # First bullish
                    result_df['close'].iloc[i-1] > result_df['open'].iloc[i-1] and  # Second bullish
                    result_df['close'].iloc[i] > result_df['open'].iloc[i] and      # Third bullish
                    result_df['close'].iloc[i] > result_df['close'].iloc[i-1] and   # Each close higher than previous
                    result_df['close'].iloc[i-1] > result_df['close'].iloc[i-2] and
                    result_df['open'].iloc[i] > result_df['open'].iloc[i-1] and     # Each open higher than previous
                    result_df['open'].iloc[i-1] > result_df['open'].iloc[i-2]):
                    
                    three_white_soldiers[i] = True
                    
                # Check Three Black Crows
                if (result_df['close'].iloc[i-2] < result_df['open'].iloc[i-2] and  # First bearish
                    result_df['close'].iloc[i-1] < result_df['open'].iloc[i-1] and  # Second bearish
                    result_df['close'].iloc[i] < result_df['open'].iloc[i] and      # Third bearish
                    result_df['close'].iloc[i] < result_df['close'].iloc[i-1] and   # Each close lower than previous
                    result_df['close'].iloc[i-1] < result_df['close'].iloc[i-2] and
                    result_df['open'].iloc[i] < result_df['open'].iloc[i-1] and     # Each open lower than previous
                    result_df['open'].iloc[i-1] < result_df['open'].iloc[i-2]):
                    
                    three_black_crows[i] = True
            except Exception as e:
                # Just skip this iteration if there's any error
                continue
        
        # Assign the pattern results to the dataframe
        result_df['three_white_soldiers'] = three_white_soldiers
        result_df['three_black_crows'] = three_black_crows
        
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
        
        # Apply all base pattern detection methods
        result_df = self.detect_consecutive_candles(result_df)
        result_df = self.detect_doji(result_df)
        result_df = self.detect_hammer(result_df)
        result_df = self.calculate_orb(result_df)
        
        # Apply additional advanced pattern detection methods
        result_df = self.detect_engulfing(result_df)
        result_df = self.detect_stars(result_df)
        result_df = self.detect_shooting_star(result_df)
        result_df = self.detect_piercing_patterns(result_df)
        result_df = self.detect_three_candle_patterns(result_df)
        
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
        
        # Flag to detect any reversal pattern (now including advanced patterns)
        result_df['has_reversal_pattern'] = (
            # Basic patterns
            result_df['is_doji'] | 
            result_df['is_hammer'] | 
            result_df['potential_bullish_reversal'] | 
            result_df['potential_bearish_reversal'] |
            # Advanced patterns
            result_df.get('bullish_engulfing', False) | 
            result_df.get('bearish_engulfing', False) |
            result_df.get('morning_star', False) | 
            result_df.get('evening_star', False) |
            result_df.get('shooting_star', False) |
            result_df.get('piercing_line', False) | 
            result_df.get('dark_cloud_cover', False) |
            result_df.get('three_white_soldiers', False) | 
            result_df.get('three_black_crows', False)
        )
        
        return result_df
