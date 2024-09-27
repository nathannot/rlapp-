import streamlit as st
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_checker import check_env
import yfinance as yf
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import BlockRNNModel
from darts.dataprocessing.transformers import MissingValuesFiller, Scaler
import plotly.express as px
import datetime
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

st.header('Reinforcement Learning for Optimizing Stock Portfolio')
st.write('Select a portfolio of up to 10 of the top 10 US Stocks')

stocks = ['aapl','msft','nvda', 'amzn', 'googl','meta', 'avgo', 'lly','tsm','tsla']

select_stocks = st.multiselect(label = 'Pick which stocks you want in your portfolio',
               options = stocks)

port_picker = st.button(label='Calculate Portfolio')

if port_picker:

    with st.spinner("Calculation in progress. Please wait. The more stocks you pick the longer it will take."):
    
        today = pd.Timestamp(datetime.datetime.today().date())
        data = []

        for s in select_stocks:
            df = yf.download(s,start='2023-09-15',end = today)
            df['tic'] = s
            data.append(df)
        final = pd.concat(data)

        future = []
        for t in final.tic.unique():
            data = final[final['tic']==t]
            series = TimeSeries.from_dataframe(data.reset_index(), time_col = 'Date', value_cols = 'Close', freq='B')
            fill = MissingValuesFiller()
            series1 = fill.transform(series)
            scale = Scaler()
            series2 = scale.fit_transform(series1)
            gru = BlockRNNModel(
                input_chunk_length= 4,
                output_chunk_length = 4,
                model='GRU',
                hidden_dim = 64,
                n_rnn_layers = 3,
                n_epochs = 5
            )
            gru.fit(series2)
            pred = gru.predict(14)
            predi = scale.inverse_transform(pred)
            pred_df = predi.pd_dataframe()
            pred_df['tic'] = t
            future.append(pred_df)
        predictions = pd.concat(future)

        env = MultiStock(final, sharpe_risk=True,  risk_free_rate = 0.4)
        envt = MultiStock(predictions, sharpe_risk=True, risk_free_rate = 0.2)

        steps  = len(final[final.tic == final.tic.iloc[0]])
        model = PPO('MlpPolicy', env, seed=42)
        model.learn(10*steps, progress_bar=True)
            
        obs,_ = env.reset()
        for step in range(steps):
            action, _ = model.predict(obs, deterministic = True)
            obs, reward, _,_,_ = env.step(action)
        profit = np.array(env.portfolio_history)-100000
        p = pd.DataFrame(profit, columns=['profit'],index = final[final.tic == final.tic.iloc[0]].index)
        if np.sum(profit) == 0:
            raise Exception('0 profit due to no optimal policy found, longer training needed. Try another portfolio.')
        st.write('This chart below shows theoritical profit on $100,000 trading the selected stocks over the last year.')

        fig = px.line(p.profit, title='RL Policy Profit')
        st.plotly_chart(fig)

        obs,_ = envt.reset()
        for step in range(14):
            action, _ = model.predict(obs, deterministic = True)
            obs, reward, _,_,_ = envt.step(action)

        portfolio = pd.DataFrame(envt.stock_history)

        for col in portfolio.columns:
            portfolio[f'{col}_buy_sell'] = portfolio[col].diff(1)
            portfolio.loc[0,f'{col}_buy_sell'] = portfolio[col].iloc[0]

        st.write('This table shows the reinforcment learning policy optimal portfolio picks for the next two weeks.')
        st.write('The buy_sell columns tell me how many stocks to buy or sell each day.')
        st.write('A negative number means sell and 0 means hold.')

        portfolio.index = predictions[0:14].index
        st.write(portfolio)

        st.write('This app uses Deep Learning to forecast stock prices and reinforcement deep learning to showcase the portfolio to give most profit.')
        st.write('Very simple models have been used to speed up training time so the app loads faster.')
        st.write('This app is for educational purposes. Do NOT make investment decisions based on this app.')
        st.write('You can use fake money to test how well the proposed portfolio works.')
    st.success('')
