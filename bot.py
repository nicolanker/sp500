import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator, StochRSIIndicator, WilliamsRIndicator
from ta.trend import MACD, ADXIndicator, EMAIndicator
from alpaca_trade_api.rest import REST, TimeFrame
from sklearn.linear_model import LinearRegression
import time

# CONFIG
API_KEY = 'PK6XDD08Q7GTVN8YDOQB'
SECRET_KEY = 'yYerw3WnLAgfNsNSKG6zIGSwraNFIOobCzTZeZAj'
BASE_URL = 'https://paper-api.alpaca.markets'
TICKER = 'SPY'
TIMEFRAME = TimeFrame.Minute
LOOKBACK = 300
TRADE_AMOUNT = 2000  # Base amount for position sizing

# INIT
api = REST(API_KEY, SECRET_KEY, BASE_URL)
trade_tracker = {}

# Fetch historical data
def fetch_data():
    bars = api.get_bars(TICKER, TIMEFRAME, limit=LOOKBACK).df
    return bars

# Add technical indicators
def add_indicators(df):
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    stoch = StochasticOscillator(df['high'], df['low'], df['close'], window=9, smooth_window=6)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    df['stoch_rsi'] = StochRSIIndicator(df['close']).stochrsi()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['adx'] = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
    df['williams_r'] = WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=14).williams_r()
    df['ema'] = EMAIndicator(close=df['close'], window=15).ema_indicator()
    return df.dropna()

# Build training data
def build_training_data(df):
    rows = []
    for i in range(50, len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        price_change = (next_row['close'] - row['close']) / row['close']

        signal_data = {
            'rsi': 1 if row['rsi'] < 30 else -1 if row['rsi'] > 70 else 0,
            'stoch': 1 if row['stoch_k'] > row['stoch_d'] and row['stoch_k'] < 20 else -1 if row['stoch_k'] < row['stoch_d'] and row['stoch_k'] > 80 else 0,
            'stoch_rsi': 1 if row['stoch_rsi'] < 0.2 else -1 if row['stoch_rsi'] > 0.8 else 0,
            'macd': 1 if row['macd'] > row['macd_signal'] else -1 if row['macd'] < row['macd_signal'] else 0,
            'adx': 1 if row['adx'] > 25 else 0,
            'williams_r': 1 if row['williams_r'] < -80 else -1 if row['williams_r'] > -20 else 0,
            'ema': 1 if row['close'] > row['ema'] else -1,
            'price_change': price_change
        }
        rows.append(signal_data)
    return pd.DataFrame(rows)

# Train ML model
def train_model(train_df):
    if train_df.empty:
        raise ValueError("Training data is empty. Increase LOOKBACK or check for NaNs.")
    print("Training data preview:\n", train_df.head())
    X = train_df.drop(columns=['price_change'])
    y = train_df['price_change']
    model = LinearRegression()
    model.fit(X, y)
    return dict(zip(X.columns, model.coef_))

# Generate trading signals
def get_signal(latest):
    signals = []
    if latest['rsi'] < 35:
        signals.append(('rsi', 'bullish'))
    elif latest['rsi'] > 65:
        signals.append(('rsi', 'bearish'))
    if latest['stoch_k'] > latest['stoch_d'] and latest['stoch_k'] < 20:
        signals.append(('stoch', 'bullish'))
    elif latest['stoch_k'] < latest['stoch_d'] and latest['stoch_k'] > 80:
        signals.append(('stoch', 'bearish'))
    if latest['stoch_rsi'] < 0.2:
        signals.append(('stoch_rsi', 'bullish'))
    elif latest['stoch_rsi'] > 0.8:
        signals.append(('stoch_rsi', 'bearish'))
    if latest['macd'] > latest['macd_signal']:
        signals.append(('macd', 'bullish'))
    elif latest['macd'] < latest['macd_signal']:
        signals.append(('macd', 'bearish'))
    if latest['adx'] > 25:
        signals.append(('adx', 'bullish'))
    if latest['williams_r'] < -80:
        signals.append(('williams_r', 'bullish'))
    elif latest['williams_r'] > -20:
        signals.append(('williams_r', 'bearish'))
    if latest['close'] > latest['ema']:
        signals.append(('ema', 'bullish'))
    else:
        signals.append(('ema', 'bearish'))
    return signals

# Calculate signal strength using weights
def calculate_position(signals, weights):
    score = 0
    for indicator, direction in signals:
        weight = weights.get(indicator, 1)
        score += weight * (1 if direction == 'bullish' else -1)
    return score

# Trade execution with dynamic position sizing
def place_trade(signal_strength, signals):
    positions = {p.symbol: p for p in api.list_positions()}
    position = positions.get(TICKER, None)
    price = api.get_latest_trade(TICKER).price

    multiplier = min(5, max(1, abs(signal_strength)))  # caps aggressive scaling
    qty = abs(int((TRADE_AMOUNT * multiplier) / price))

    if signal_strength > 0:
        if position and int(position.qty) < 0:
            api.submit_order(symbol=TICKER, qty=abs(int(position.qty)), side='buy', type='market', time_in_force='gtc')
        api.submit_order(symbol=TICKER, qty=qty, side='buy', type='market', time_in_force='gtc')
        print(f"BUY {qty} of {TICKER}")
    elif signal_strength < 0:
        if position and int(position.qty) > 0:
            api.submit_order(symbol=TICKER, qty=abs(int(position.qty)), side='sell', type='market', time_in_force='gtc')
        api.submit_order(symbol=TICKER, qty=qty, side='sell', type='market', time_in_force='gtc')
        print(f"SELL {qty} of {TICKER}")
    else:
        print("HOLD")

# Trailing stop-loss check
def update_trailing_stop(position, latest_price):
    symbol = position.symbol
    qty = int(position.qty)
    current_price = float(latest_price)
    entry_price = float(position.avg_entry_price)

    if symbol not in trade_tracker:
        trade_tracker[symbol] = {"high": current_price}

    trade_tracker[symbol]['high'] = max(trade_tracker[symbol]['high'], current_price)
    peak = trade_tracker[symbol]['high']

    if (current_price / peak - 1) < -0.01 or float(position.unrealized_plpc) > 0.02:
        side = 'sell' if qty > 0 else 'buy'
        api.submit_order(symbol=symbol, qty=abs(qty), side=side, type='market', time_in_force='gtc')
        del trade_tracker[symbol]
        print(f"Exit {symbol} due to stop or profit")

# Run once per cycle
def run_bot(weights):
    df = fetch_data()
    df = add_indicators(df)
    latest = df.iloc[-1]
    signals = get_signal(latest)
    signal_strength = calculate_position(signals, weights)

    positions = {p.symbol: p for p in api.list_positions()}
    if TICKER in positions:
        latest_price = api.get_latest_trade(TICKER).price
        update_trailing_stop(positions[TICKER], latest_price)

    print(f"Signals: {signals} | Strength: {signal_strength}")
    place_trade(signal_strength, signals)

# --- STARTUP ---
df = fetch_data()
df = add_indicators(df)
train_df = build_training_data(df)
if train_df.empty:
    raise ValueError("Training data is empty. Check data quality or increase LOOKBACK.")
weights = train_model(train_df)
print("Learned Weights:", weights)

# Main loop
while True:
    try:
        run_bot(weights)
        time.sleep(240)
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(60)
