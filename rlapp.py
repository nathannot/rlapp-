import streamlit as st
from tradingmultiple import MultiStock
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