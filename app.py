# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import os
import pandas as pd
import numpy as np
from Historic_Crypto import HistoricalData
from dash import Dash, html, dcc
import plotly.express as px
from train_ai import run_ai

fname = 'ticker_daily.pqt'
period = 60*60*24 #seconds
asset_ls = ['BTC-USD','ETH-USD']
adjustment_period = 7 ###ratio redistribution period

if os.path.exists(f'./{fname}'):
    daily_history_df = pd.read_parquet(f'./{fname}')
else:
    daily_history_df = pd.DataFrame()
    for asset in asset_ls:
        daily = HistoricalData(asset, period,'2015-01-01-00-00').retrieve_data()
        daily['ticker'] = asset
        daily_history_df = pd.concat([daily_history_df,daily])

    daily_history_df.to_parquet(fname)

if not os.path.exists('./portfolio.pqt'):
    run_ai(daily_history_df, asset_ls, adjustment_period=adjustment_period)

fig1 = px.line(
    pd.read_parquet('./portfolio.pqt')
        .sum(axis=1)
        .to_frame()
        .rename(columns={0:'Total Porfolio Value'})
        .assign(Two_BTC=daily_history_df[daily_history_df['ticker'] == 'BTC-USD'][['open']]*2)
        .assign(Eighty_ETH=daily_history_df[daily_history_df['ticker'] == 'ETH-USD'][['open']]*80)
        , title='Portfolio Total vs Assets'
    )

fig2 = px.line(pd.read_parquet('./portfolio.pqt'), title='In porfolio asset values in $')
fig3 = px.line(pd.read_parquet('./portfolio_ratio.pqt'), title='In porfolio asset ratio')
fig4 = px.line(daily_history_df[['open','ticker']], color="ticker", symbol="ticker")

app = Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Crypto Portfolio Manager'),

    html.Div(children='''
        An AI portfolio management system.
    '''),
        dcc.Graph(
        id='ts-total-compare',
        figure=fig1
    ),

    dcc.Graph(
        id='ts-portfolio',
        figure=fig2
    ),

    dcc.Graph(
        id='ts-portfolio-ratio',
        figure=fig3
    ),

    dcc.Graph(
        id='ts-raw-asset-price',
        figure=fig4
    ),
])

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=3000)