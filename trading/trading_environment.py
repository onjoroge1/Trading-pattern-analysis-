import gym
from gym import spaces
import numpy as np
import pandas as pd
import random
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env):
    """Custom Gym environment for stock trading with reversal patterns"""
    
    def __init__(self, stock_data_dict, start_date=None, end_date=None, initial_balance=10000):
        """
        Initialize the trading environment
        
        Args:
            stock_data_dict (dict): Dictionary with stock symbols as keys and DataFrames as values
            start_date (datetime, optional): Start date for training
            end_date (datetime, optional): End date for training
            initial_balance (float): Initial account balance
        """
        super(TradingEnvironment, self).__init__()
        
        self.stock_data_dict = stock_data_dict
        self.stocks = list(stock_data_dict.keys())
        
        # Set date range
        self.start_date = start_date
        self.end_date = end_date
        
        # Account variables
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0  # 0: flat, 1: long, -1: short
        self.position_price = 0
        self.current_stock = None
        self.current_idx = 0
        self.current_data = None
        self.done = False
        
        # Trading variables
        self.trade_fee_pct = 0.0005  # 0.05% trading fee
        self.slippage_pct = 0.0002   # 0.02% slippage
        
        # State features
        self.state_dim = 10
        
        # Action space: 0 = do nothing, 1 = buy/go long, 2 = sell/go short
        self.action_space = spaces.Discrete(3)
        
        # Observation space: normalized state values between -1 and 1
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.state_dim,), dtype=np.float32)
        
        logger.info("TradingEnvironment initialized with "
                    f"{len(self.stocks)} stocks, initial_balance={initial_balance}")
    
    def _get_current_price(self):
        """Get the current closing price"""
        return self.current_data.iloc[self.current_idx]['close']
    
    def _get_state(self):
        """
        Create the state vector for the agent
        
        Returns:
            numpy.array: State vector with 10 features
        """
        if self.current_idx >= len(self.current_data):
            return np.zeros(self.state_dim)
        
        # Current row of data
        curr_row = self.current_data.iloc[self.current_idx]
        
        # 1. Price (normalized by dividing by 100)
        current_price = curr_row['close'] / 100
        
        # 2. RSI (already between 0-100, normalize to -1 to 1)
        rsi = (curr_row['rsi'] - 50) / 50 if 'rsi' in curr_row else 0
        
        # 3. Is Doji (0 or 1)
        is_doji = 1 if 'is_doji' in curr_row and curr_row['is_doji'] else 0
        
        # 4. Is Hammer (0 or 1)
        is_hammer = 1 if 'is_hammer' in curr_row and curr_row['is_hammer'] else 0
        
        # 5. Position (-1, 0, 1)
        position = self.position
        
        # 6 & 7. Distance from ORB high and low (normalize between -1 and 1)
        dist_orb_high = curr_row['distance_from_orb_high'] if 'distance_from_orb_high' in curr_row else 0
        dist_orb_high = np.clip(dist_orb_high, -1, 1)
        
        dist_orb_low = curr_row['distance_from_orb_low'] if 'distance_from_orb_low' in curr_row else 0
        dist_orb_low = np.clip(dist_orb_low, -1, 1)
        
        # 8 & 9. Boolean: above ORB high, below ORB low
        above_orb_high = 1 if 'above_orb_high' in curr_row and curr_row['above_orb_high'] else 0
        below_orb_low = 1 if 'below_orb_low' in curr_row and curr_row['below_orb_low'] else 0
        
        # 10. Boolean: within market open time
        in_opening_hour = 1 if 'in_opening_hour' in curr_row and curr_row['in_opening_hour'] else 0
        
        state = np.array([
            current_price,
            rsi,
            is_doji,
            is_hammer,
            position,
            dist_orb_high,
            dist_orb_low,
            above_orb_high,
            below_orb_low,
            in_opening_hour
        ], dtype=np.float32)
        
        return state
    
    def _calculate_reward(self, action):
        """
        Calculate the reward for the action taken
        
        Args:
            action (int): The action taken (0: do nothing, 1: buy/long, 2: sell/short)
            
        Returns:
            float: The reward value
        """
        if self.current_idx >= len(self.current_data) - 1:
            return 0
        
        curr_row = self.current_data.iloc[self.current_idx]
        current_price = curr_row['close']
        next_price = self.current_data.iloc[self.current_idx + 1]['close']
        price_change = (next_price - current_price) / current_price
        
        # Initialize reward
        reward = 0
        
        # Reward for price movement based on position
        if self.position == 1:  # Long position
            reward += price_change * 100  # Scale up for better learning
        elif self.position == -1:  # Short position
            reward += -price_change * 100  # Negative price change is good for shorts
        
        # Additional reward for correct positioning based on reversal patterns
        if 'rsi_long_bias' in curr_row and 'rsi_short_bias' in curr_row:
            # If we have RSI bias information
            if self.position == 1 and curr_row['rsi_long_bias']:
                reward += 0.1  # Bonus for being in the right position
            elif self.position == -1 and curr_row['rsi_short_bias']:
                reward += 0.1  # Bonus for being in the right position
        
        # Penalty for frequent trading (to avoid excessive trading)
        if (action == 1 and self.position != 1) or (action == 2 and self.position != -1):
            reward -= 0.05  # Small penalty for changing positions
        
        # Contextual reward based on ORB
        if 'above_orb_high' in curr_row and 'below_orb_low' in curr_row:
            if self.position == 1 and curr_row['above_orb_high']:
                reward += 0.05  # Bonus for long above ORB high
            elif self.position == -1 and curr_row['below_orb_low']:
                reward += 0.05  # Bonus for short below ORB low
        
        return reward
    
    def _apply_trade_costs(self, price, is_buy):
        """Apply trading costs (fees and slippage)"""
        direction = 1 if is_buy else -1
        adjusted_price = price * (1 + direction * self.slippage_pct)
        fee = adjusted_price * self.trade_fee_pct
        return adjusted_price + fee if is_buy else adjusted_price - fee
    
    def _take_action(self, action):
        """Execute the trading action"""
        if self.current_idx >= len(self.current_data) - 1:
            return 0  # No reward at the end of data
        
        current_price = self._get_current_price()
        reward = 0
        
        # Action: 0 = do nothing, 1 = buy/go long, 2 = sell/go short
        if action == 1:  # Buy/Go Long
            if self.position == -1:  # If currently short, close position
                sell_price = self._apply_trade_costs(current_price, False)
                profit = self.position_price - sell_price
                self.balance += profit
                self.position = 0
                self.position_price = 0
            
            if self.position == 0:  # If flat, open long position
                buy_price = self._apply_trade_costs(current_price, True)
                self.position = 1
                self.position_price = buy_price
                
        elif action == 2:  # Sell/Go Short
            if self.position == 1:  # If currently long, close position
                sell_price = self._apply_trade_costs(current_price, False)
                profit = sell_price - self.position_price
                self.balance += profit
                self.position = 0
                self.position_price = 0
            
            if self.position == 0:  # If flat, open short position
                sell_price = self._apply_trade_costs(current_price, False)
                self.position = -1
                self.position_price = sell_price
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Update state
        self.current_idx += 1
        if self.current_idx >= len(self.current_data) - 1:
            self.done = True
            
            # Close any open position at the end
            if self.position == 1:
                sell_price = self._apply_trade_costs(current_price, False)
                profit = sell_price - self.position_price
                self.balance += profit
            elif self.position == -1:
                buy_price = self._apply_trade_costs(current_price, True)
                profit = self.position_price - buy_price
                self.balance += profit
            
            # Add final P&L to reward
            final_pnl_pct = (self.balance - self.initial_balance) / self.initial_balance
            reward += final_pnl_pct * 100  # Scale up for better learning
        
        return reward
    
    def reset(self):
        """
        Reset the environment for a new episode
        
        Returns:
            numpy.array: Initial state
        """
        # Randomly select a stock
        self.current_stock = random.choice(self.stocks)
        self.current_data = self.stock_data_dict[self.current_stock]
        
        # Reset position and balance
        self.position = 0
        self.position_price = 0
        self.balance = self.initial_balance
        self.done = False
        
        # Randomly select a starting point
        # Ensure we have at least 100 data points ahead to avoid short episodes
        max_start = max(0, len(self.current_data) - 100)
        self.current_idx = random.randint(0, max_start) if max_start > 0 else 0
        
        logger.debug(f"Reset environment with stock {self.current_stock}, "
                     f"start_idx={self.current_idx}")
        
        return self._get_state()
    
    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action (int): The action to take
            
        Returns:
            tuple: (state, reward, done, info)
        """
        reward = self._take_action(action)
        next_state = self._get_state()
        
        # Additional info
        info = {
            'balance': self.balance,
            'position': self.position,
            'stock': self.current_stock,
            'timestamp': self.current_data.index[self.current_idx - 1] 
                         if self.current_idx > 0 and self.current_idx < len(self.current_data) 
                         else None,
            'price': self._get_current_price() if self.current_idx < len(self.current_data) else None
        }
        
        return next_state, reward, self.done, info
    
    def render(self, mode='human'):
        """Render the environment (not implemented)"""
        pass
    
    def close(self):
        """Close the environment"""
        pass
