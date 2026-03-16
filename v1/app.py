import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time
import threading

st.set_page_config(page_title="BB Options Trader Pro", layout="wide")
st.title("🚀 Bollinger Bands Options Trader Pro")
st.markdown("**Educational tool only** — Not financial advice.")

# ====================== SESSION STATE ======================
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["AAPL", "TSLA", "NVDA", "MSFT", "BTC-USD", "SPY"]
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

# ====================== AUTO REFRESH ======================
def auto_refresh():
    while True:
        time.sleep(300)
        try:
            st.rerun()
        except:
            pass

if 'refresh_thread' not in st.session_state:
    st.session_state.refresh_thread = threading.Thread(target=auto_refresh, daemon=True)
    st.session_state.refresh_thread.start()

st.caption(f"🕒 Last refreshed: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')} | Auto-refresh every 5 min")

# ====================== SIDEBAR ======================
st.sidebar.header("Settings")

symbol_single = st.sidebar.text_input("Single Symbol Analysis", value="AAPL").upper().strip()

timeframe = st.sidebar.selectbox("Time Period", ["6mo", "1y", "2y"], index=0)
interval_name = st.sidebar.selectbox("Chart Interval", ["Daily", "4 Hour", "Hourly"])
interval_map = {"Daily": "1d", "4 Hour": "4h", "Hourly": "1h"}
interval = interval_map[interval_name]

bb_length = st.sidebar.slider("Bollinger Length", 10, 50, 20)
bb_std = st.sidebar.slider("Std Dev", 1.5, 3.0, 2.0, 0.1)
rsi_period = st.sidebar.slider("RSI Period", 7, 21, 14)

st.sidebar.subheader("📡 Watchlist")
new_symbol = st.sidebar.text_input("Add Symbol")
if st.sidebar.button("➕ Add to Watchlist") and new_symbol:
    sym = new_symbol.upper().strip()
    if sym not in st.session_state.watchlist:
        st.session_state.watchlist.append(sym)

if st.sidebar.button("🗑️ Clear Watchlist"):
    st.session_state.watchlist = []

if st.sidebar.button("🔄 Manual Refresh"):
    st.rerun()

# ====================== FIXED DATA FETCH ======================
@st.cache_data(ttl=60)
def get_data(sym: str, tf: str, iv: str):
    try:
        df = yf.download(sym, period=tf, interval=iv, progress=False, threads=False)
        
        if df.empty or len(df) < bb_length + 10:
            return None, f"Insufficient raw data for {sym} ({len(df)} rows)"

        # FIX: Handle MultiIndex columns (new yfinance behavior)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)   # Remove the ticker level

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

        # Calculate indicators
        df['MA']    = df['Close'].rolling(window=bb_length).mean()
        df['Std']   = df['Close'].rolling(window=bb_length).std()
        df['Upper'] = df['MA'] + bb_std * df['Std']
        df['Lower'] = df['MA'] - bb_std * df['Std']

        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        df = df.dropna(subset=['MA', 'Upper', 'Lower', 'RSI']).copy()

        if df.empty:
            return None, f"No valid rows left after cleaning for {sym}"

        df['Signal'] = 'Neutral'
        df.loc[(df['Close'] < df['Lower']) & (df['RSI'] < 35), 'Signal'] = '🟢 Strong Buy CALL'
        df.loc[(df['Close'] > df['Upper']) & (df['RSI'] > 65), 'Signal'] = '🔴 Strong Buy PUT'

        df['BB_Width'] = (df['Upper'] - df['Lower']) / df['Close']
        return df, None

    except Exception as e:
        return None, f"Error fetching {sym}: {str(e)}"


# ====================== TABS ======================
tab1, tab2 = st.tabs(["📈 Single Symbol", "📡 Live Dashboard"])

with tab1:
    if st.button("Analyze Single Symbol", type="primary"):
        df, error = get_data(symbol_single, timeframe, interval)
        if df is not None and not df.empty:
            current = df.iloc[-1]
            color = "green" if "CALL" in current['Signal'] else "red" if "PUT" in current['Signal'] else "orange"
            
            st.subheader(f"Current Signal: {symbol_single}")
            st.markdown(f"<h2 style='color:{color};'>{current['Signal']}</h2>", unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                         low=df['Low'], close=df['Close'], name="Price"))
            fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], line=dict(color='red'), name='Upper'))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA'], line=dict(color='orange', width=2), name=f'MA({bb_length})'))
            fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], line=dict(color='green'), name='Lower'))

            fig.update_layout(title=f"{symbol_single} — Bollinger Bands", height=650, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(error or "Failed to load data")

with tab2:
    st.subheader("Live Multi-Symbol Dashboard")
    data_list = []
    for sym in st.session_state.watchlist:
        df, _ = get_data(sym, timeframe, interval)
        if df is None or df.empty:
            continue
        current = df.iloc[-1]
        data_list.append({
            "Symbol": sym,
            "Price": f"${current['Close']:.2f}",
            "RSI": f"{current['RSI']:.1f}",
            "Signal": current['Signal'],
            "BB Width": f"{current['BB_Width']*100:.1f}%",
            "Last Updated": current.name.strftime("%Y-%m-%d %H:%M")
        })

    if data_list:
        df_watch = pd.DataFrame(data_list)
        def highlight(val):
            if "Strong Buy CALL" in str(val): return 'background-color: #d4edda; color: green'
            if "Strong Buy PUT" in str(val): return 'background-color: #f8d7da; color: red'
            return ''
        st.dataframe(df_watch.style.applymap(highlight, subset=['Signal']), use_container_width=True)
    else:
        st.warning("No data loaded. Try **6mo** or **1y** period and click Manual Refresh.")

st.session_state.last_refresh = datetime.now()
st.caption("✅ MultiIndex column fix applied • Default period is now 6mo")
