import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MultiStock(gym.Env):  
    def __init__(self, df, tech_indicator_list=None, render_mode=None, sharpe_risk=False, risk_free_rate=0):
        super(MultiStock, self).__init__()
        
        self.render_mode = render_mode
        self.sharpe_risk = sharpe_risk
        self.risk_free_rate = risk_free_rate

        # Determine the number of unique stocks using the 'tic' column
        self.tickers = df['tic'].unique()
        self.num_stocks = len(self.tickers)  # Number of unique stocks based on 'tic' column
        
        # Stock price data filtered by each ticker symbol (multi-stock)
        self.df = df
        self.data = {tic: df[df['tic'] == tic]['Close'].values for tic in self.tickers}  # Dict of stock prices per ticker
        self.current_step = 0

        # User provides the list of technical indicators
        self.tech_indicator_list = tech_indicator_list or []  # If none provided, default to an empty list

        # Action space: -1 (sell) to 1 (buy) for each stock
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_stocks,), dtype=np.float32)

        # Observation space: (price, shares, cash, technical indicators) for each stock
        self.observation_space = spaces.Box(low=0, high=np.inf, 
                                            shape=(self.num_stocks * (2 + len(self.tech_indicator_list)) + 1,), 
                                            dtype=np.float32)
        
        # Starting balance and shares for each stock
        self.cash = 100000
        self.num_shares = np.zeros(self.num_stocks, dtype=np.float32)
        self.stock_prices = np.array([self.data[tic][self.current_step] for tic in self.tickers])
        self.portfolio_history = []
        self.stock_history = {tic: [] for tic in self.tickers}  # Dictionary to store stock shares history

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = 100000
        self.num_shares = np.zeros(self.num_stocks, dtype=np.float32)
        self.stock_prices = np.array([self.data[tic][self.current_step] for tic in self.tickers])
        self.portfolio_history = []  # Track portfolio value over time
        self.stock_history = {tic: [] for tic in self.tickers}  # Reset stock shares history
        
        # Build observation, including prices, shares, cash, and technical indicators
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        self.stock_prices = np.array([self.data[tic][self.current_step] for tic in self.tickers])
        
        # Normalize actions to ensure the sum of all proportions is 1
        action_sum = np.sum(np.abs(action))
        if action_sum > 0:
            normalized_action = action / action_sum  # Normalize actions to sum to 1
        else:
            normalized_action = action  # In case all actions are zero, keep them as is
        
        # Loop through each stock and apply the action (buy/sell shares)
        for i, act in enumerate(normalized_action):
            # Buy shares
            if act > 0:  
                proportion_to_spend = min(act, 1)  # Limit to 100% of cash
                amount_to_spend = proportion_to_spend * self.cash
                num_to_buy = int(amount_to_spend // self.stock_prices[i])
                self.num_shares[i] += num_to_buy
                self.cash -= num_to_buy * self.stock_prices[i]
            
            # Sell shares
            elif act < 0:  
                proportion_to_sell = min(-act, 1)  # Limit to 100% of shares
                num_to_sell = int(proportion_to_sell * self.num_shares[i])
                self.cash += num_to_sell * self.stock_prices[i]
                self.num_shares[i] -= num_to_sell

        self.current_step += 1
        terminated = self.current_step >= len(self.data[self.tickers[0]])
        truncated = False
        
        # Portfolio value calculation across all stocks
        total_value = self.cash + np.sum(self.num_shares * self.stock_prices)
        self.portfolio_history.append(total_value)
        reward = total_value - 100000

        # Append the number of shares for each stock at the current step to stock_history
        for i, tic in enumerate(self.tickers):
            self.stock_history[tic].append(self.num_shares[i])
        
        # Calculate Sharpe ratio if enabled
        if self.sharpe_risk and len(self.portfolio_history) > 1:
            returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
            portfolio_std = np.std(returns)
            avg_return = np.mean(returns) - self.risk_free_rate
            if portfolio_std != 0:
                sharpe_ratio = avg_return / portfolio_std
            else:
                sharpe_ratio = 0
            reward = sharpe_ratio  # Use Sharpe ratio as reward

        # Build the observation
        obs = self._get_observation()
        return obs, reward, terminated, truncated, {}

    def _get_observation(self):
        """Helper function to build observation."""
        tech_indicators = []
        for tic in self.tickers:
            indicators = [self.df[self.df['tic'] == tic][ind].values[self.current_step] for ind in self.tech_indicator_list]
            tech_indicators.extend(indicators)
        
        # Concatenate stock prices, shares, cash, and technical indicators into the observation
        obs = np.concatenate([self.stock_prices, self.num_shares, [self.cash]] + tech_indicators, dtype=np.float32)
        return obs

    def render(self, action, reward):
        """Handle rendering logic."""
        total_value = self.cash + np.sum(self.num_shares * self.stock_prices)
        if self.render_mode == 'human':
            print(f'Step: {self.current_step}, Action: {action}, Prices: {self.stock_prices}, Shares: {self.num_shares}, Cash: {self.cash}, Portfolio: {total_value}, Reward: {reward}')
