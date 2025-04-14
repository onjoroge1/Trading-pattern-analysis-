import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Class to calculate technical indicators for stock data"""
    
    @staticmethod
    def calculate_rsi(df, window=14, round_values=True):
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            df (pandas.DataFrame): Dataframe with 'close' prices
            window (int): RSI period (default: 14)
            round_values (bool): Round RSI values to 2 decimal places
            
        Returns:
            pandas.DataFrame: DataFrame with 'rsi' column
        """
        # Create a copy of the dataframe with only the close column
        close_df = pd.DataFrame(df['close'])
        
        # Calculate price changes
        delta = close_df['close'].diff()
        
        # Get positive and negative price changes
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss over the window
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        
        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        if round_values:
            rsi = rsi.round(2)
        
        # Handle potential NaN values
        rsi = rsi.fillna(50)  # Neutral RSI for NaN values
        
        # Return as DataFrame with the same index as the input
        result_df = pd.DataFrame(index=df.index)
        result_df['rsi'] = rsi
        
        logger.info(f"Calculated RSI with window={window}")
        return result_df
    
    @staticmethod
    def calculate_moving_averages(df, ma_periods=[20, 50, 200]):
        """
        Calculate Moving Averages
        
        Args:
            df (pandas.DataFrame): Dataframe with 'close' prices
            ma_periods (list): List of MA periods to calculate
            
        Returns:
            pandas.DataFrame: DataFrame with MA columns
        """
        # Create a copy of the dataframe with only the close column
        close_df = pd.DataFrame(df['close'])
        
        # Calculate MAs for each period
        for period in ma_periods:
            close_df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
        
        logger.info(f"Calculated Moving Averages for periods: {ma_periods}")
        return close_df
    
    @staticmethod
    def calculate_bollinger_bands(df, window=20, num_std=2):
        """
        Calculate Bollinger Bands
        
        Args:
            df (pandas.DataFrame): Dataframe with 'close' prices
            window (int): Window for the moving average
            num_std (int): Number of standard deviations for the bands
            
        Returns:
            pandas.DataFrame: DataFrame with Bollinger Bands columns
        """
        # Create a copy of the dataframe with only the close column
        result_df = pd.DataFrame(index=df.index)
        
        # Calculate the simple moving average
        result_df['bb_ma'] = df['close'].rolling(window=window).mean()
        
        # Calculate the standard deviation
        result_df['bb_std'] = df['close'].rolling(window=window).std()
        
        # Calculate the upper and lower bands
        result_df['bb_upper'] = result_df['bb_ma'] + (result_df['bb_std'] * num_std)
        result_df['bb_lower'] = result_df['bb_ma'] - (result_df['bb_std'] * num_std)
        
        # Calculate Bandwidth and %B
        result_df['bb_bandwidth'] = (result_df['bb_upper'] - result_df['bb_lower']) / result_df['bb_ma']
        result_df['bb_percent_b'] = (df['close'] - result_df['bb_lower']) / (result_df['bb_upper'] - result_df['bb_lower'])
        
        logger.info(f"Calculated Bollinger Bands with window={window}, num_std={num_std}")
        return result_df
    
    @staticmethod
    def calculate_all_indicators(df):
        """
        Calculate all technical indicators
        
        Args:
            df (pandas.DataFrame): OHLC dataframe
            
        Returns:
            pandas.DataFrame: DataFrame with all indicators
        """
        # Initialize an empty DataFrame with the same index
        result_df = pd.DataFrame(index=df.index)
        
        # Calculate RSI
        rsi_df = TechnicalIndicators.calculate_rsi(df)
        result_df['rsi'] = rsi_df['rsi']
        
        # Calculate Moving Averages
        ma_df = TechnicalIndicators.calculate_moving_averages(df)
        for col in ma_df.columns:
            if col != 'close':  # Skip the original close column
                result_df[col] = ma_df[col]
        
        # Calculate Bollinger Bands
        bb_df = TechnicalIndicators.calculate_bollinger_bands(df)
        for col in bb_df.columns:
            result_df[col] = bb_df[col]
        
        logger.info("Calculated all technical indicators")
        return result_df
