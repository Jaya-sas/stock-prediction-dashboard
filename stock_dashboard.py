import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# Dash app
app = dash.Dash(__name__)
server = app.server

# Available stock symbols (you can add more)
stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']

app.layout = html.Div([
    html.H1("📈 Stock Price Forecast Dashboard", style={'textAlign': 'center'}),
    
    html.Label("Select Stock:"),
    dcc.Dropdown(
        id='stock-dropdown',
        options=[{'label': stock, 'value': stock} for stock in stocks],
        value='AAPL'
    ),

    html.Br(),

    html.Label("Select Forecast Days (1-30):"),
    dcc.Slider(
        id='day-slider',
        min=1,
        max=30,
        step=1,
        value=7,
        marks={i: str(i) for i in range(1, 31)}
    ),

    html.Br(),

    dcc.Graph(id='stock-graph')
])

@app.callback(
    Output('stock-graph', 'figure'),
    [Input('stock-dropdown', 'value'),
     Input('day-slider', 'value')]
)
def update_graph(selected_stock, forecast_days):
    # Fetch stock data
    data = yf.download(selected_stock, period='1y')
    data = data[['Close']]
    data['Target'] = data['Close'].shift(-forecast_days)
    data.dropna(inplace=True)

    # Prepare training data
    X = np.arange(len(data)).reshape(-1, 1)
    y = data['Target'].values

    if len(data) == 0:
        return go.Figure().update_layout(title="Not enough data.")

    model = LinearRegression()
    model.fit(X, y)

    # Forecast
    future_days = np.arange(len(data), len(data)+forecast_days).reshape(-1, 1)
    future_preds = model.predict(future_days)
    future_dates = pd.date_range(start=data.index[-1], periods=forecast_days+1, freq='B')[1:]

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Actual Price'))
    fig.add_trace(go.Scatter(x=future_dates, y=future_preds, name='Predicted Price'))
    fig.update_layout(
        title=f"{selected_stock} Stock Price Forecast",
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white'
    )

    return fig

# Run server
if __name__ == '__main__':
    app.run(debug=True)

