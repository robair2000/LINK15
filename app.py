from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests
import streamlit as st
import xgboost as xgb
import yfinance as yf
from sklearn.model_selection import train_test_split

# --- STREAMLIT PAGINA CONFIGURATIE ---
st.set_page_config(
    page_title='Chainlink Institutional 15m Quant Dashboard',
    page_icon='🔗',
    layout='wide',
)

st.title('🔗 Chainlink (LINK) Institutioneel 15m Scalping Quant Model')
st.markdown(
    'Dit professionele dashboard combineert **15m orderflow**, **Multi-Timeframe'
    ' Trend Filter**, **ATR Risk Management** en **Backtesting** — 100% feitelijke data.'
)


# --- DATA OPHAAL (15m + Daily Trend voor LINK) ---
@st.cache_data(ttl=300)
def load_market_data():
  try:
    df = yf.download(
        'LINK-USD', period='30d', interval='15m', progress=False
    )
    if df is None or df.empty:
      return None
    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)
    df = df.reset_index()

    date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
    df = df.rename(
        columns={
            date_col: 'Date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
        }
    )
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').dropna(subset=['close']).reset_index(drop=True)

    df_daily = yf.download(
        'LINK-USD', period='60d', interval='1d', progress=False
    )
    if isinstance(df_daily.columns, pd.MultiIndex):
      df_daily.columns = df_daily.columns.get_level_values(0)
    df_daily = df_daily.reset_index()
    d_col = 'Datetime' if 'Datetime' in df_daily.columns else 'Date'
    df_daily = df_daily.rename(columns={d_col: 'Date', 'Close': 'daily_close'})
    df_daily['Date'] = pd.to_datetime(df_daily['Date'])
    df_daily['daily_SMA50'] = df_daily['daily_close'].rolling(window=50).mean()

    df['Date_only'] = df['Date'].dt.date
    df_daily['Date_only'] = df_daily['Date'].dt.date
    df = pd.merge(
        df,
        df_daily[['Date_only', 'daily_close', 'daily_SMA50']],
        on='Date_only',
        how='left',
    )
    df = df.drop(columns=['Date_only']).ffill().dropna().reset_index(drop=True)

    return df
  except Exception as e:
    st.error(f'Fout bij ophalen data: {e}')
    return None


def fetch_realtime_binance_price():
  try:
    url = 'https://api.binance.com/api/v3/ticker/price?symbol=LINKUSDT'
    res = requests.get(url, timeout=3).json()
    return float(res['price'])
  except Exception:
    return None


def add_advanced_indicators(df):
  typical_price = (df['high'] + df['low'] + df['close']) / 3
  df['VWAP'] = (typical_price * df['volume']).rolling(window=96).sum() / (
      df['volume'].rolling(window=96).sum() + 1e-8
  )
  df['VWAP_distance'] = (df['close'] - df['VWAP']) / (df['VWAP'] + 1e-8)

  sma_20 = df['close'].rolling(window=20).mean()
  std_20 = df['close'].rolling(window=20).std()
  upper_band = sma_20 + (std_20 * 2)
  lower_band = sma_20 - (std_20 * 2)
  df['BB_pct_b'] = (df['close'] - lower_band) / (
      upper_band - lower_
