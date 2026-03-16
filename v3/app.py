import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time
import threading
import numpy as np

st.set_page_config(page_title="BB + Volume Profile + TPO + Order Flow Pro", layout="wide")
st.title("🚀 Bollinger Bands + Volume Profile + TPO + Refined Order Flow")
st.markdown("**Educational tool only** — Not financial advice.")

# ====================== SESSION STATE ======================
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["AAPL", "TSLA", "NVDA", "MSFT", "BTC-USD", "SPY"]
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

def auto_refresh():
    while True:
        time.sleep(300)
        try: st.rerun()
        except: pass

if 'refresh_thread' not in st.session_state:
    st.session_state.refresh_thread = threading.Thread(target=auto_refresh, daemon=True)
    st.session_state.refresh_thread.start()

st.caption(f"🕒 Last refreshed: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')} | Auto-refresh every 5 min")

# ====================== SIDEBAR ======================
st.sidebar.header("Settings")
symbol_single = st.sidebar.text_input("Single Symbol Analysis", value="AAPL").upper().strip()
timeframe = st.sidebar.selectbox("Time Period", ["3mo", "6mo", "1y", "2y"], index=1)
interval_name = st.sidebar.selectbox("Chart Interval", ["Daily", "4 Hour", "Hourly"])
interval_map = {"Daily": "1d", "4 Hour": "4h", "Hourly": "1h"}
interval = interval_map[interval_name]
bb_length = st.sidebar.slider("Bollinger Length", 10, 50, 20)
bb_std = st.sidebar.slider("Bollinger Std Dev", 1.5, 3.0, 2.0, 0.1)

st.sidebar.subheader("Watchlist")
new_symbol = st.sidebar.text_input("Add Symbol")
if st.sidebar.button("➕ Add") and new_symbol:
    sym = new_symbol.upper().strip()
    if sym not in st.session_state.watchlist:
        st.session_state.watchlist.append(sym)
if st.sidebar.button("🗑️ Clear"): st.session_state.watchlist = []
if st.sidebar.button("🔄 Manual Refresh"): st.rerun()

# ====================== DATA FETCH ======================
@st.cache_data(ttl=60)
def get_data(sym: str, tf: str, iv: str):
    try:
        df = yf.download(sym, period=tf, interval=iv, progress=False, threads=False)
        if df.empty or len(df) < 50: return None, "Insufficient data"

        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)
        df = df[['Open','High','Low','Close','Volume']].copy()

        # Core Indicators
        df['BB_MA'] = df['Close'].rolling(bb_length).mean()
        df['BB_Upper'] = df['BB_MA'] + bb_std * df['Close'].rolling(bb_length).std()
        df['BB_Lower'] = df['BB_MA'] - bb_std * df['Close'].rolling(bb_length).std()

        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        low14 = df['Low'].rolling(14).min()
        high14 = df['High'].rolling(14).max()
        df['Stoch_K'] = 100 * (df['Close'] - low14) / (high14 - low14)
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

        df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['TPV'] = df['TP'] * df['Volume']
        df['Cum_TPV'] = df['TPV'].cumsum()
        df['Cum_Vol'] = df['Volume'].cumsum()
        df['VWAP'] = df['Cum_TPV'] / df['Cum_Vol']

        # ====================== REFINED ORDER FLOW ======================
        # Volume Delta (positive = buying pressure, negative = selling pressure)
        df['Body'] = df['Close'] - df['Open']
        df['Range'] = df['High'] - df['Low']
        df['Delta'] = df['Volume'] * (df['Body'] / df['Range']).fillna(0).clip(-1, 1)
        
        df['Cum_Delta'] = df['Delta'].cumsum()
        df['Cum_Delta_Zero'] = df['Cum_Delta'] - df['Cum_Delta'].iloc[0]  # Normalized

        # Divergence Detection (Price makes higher high but Delta makes lower high)
        df['Price_HH'] = df['Close'] > df['Close'].shift(5)
        df['Delta_LL'] = df['Cum_Delta'] < df['Cum_Delta'].shift(5)
        df['Bearish_Divergence'] = df['Price_HH'] & df['Delta_LL']
        
        df['Price_LL'] = df['Close'] < df['Close'].shift(5)
        df['Delta_HH'] = df['Cum_Delta'] > df['Cum_Delta'].shift(5)
        df['Bullish_Divergence'] = df['Price_LL'] & df['Delta_HH']

        # Volume Profile + TPO (kept from previous version)
        prices = np.linspace(df['Low'].min(), df['High'].max(), 50)
        vol_profile = [df.loc[(df['Low'] <= p) & (df['High'] >= p), 'Volume'].sum() for p in prices]
        total_vol = sum(vol_profile)
        poc_idx = np.argmax(vol_profile)
        poc_price = prices[poc_idx]
        cum_vol = np.cumsum(vol_profile)
        target = total_vol * 0.70
        vah_idx = np.where(cum_vol >= target)[0][0]
        val_idx = np.where(cum_vol[::-1] >= target)[0][0]
        vah = prices[vah_idx]
        val = prices[-val_idx-1]

        tpo_prices = np.linspace(df['Low'].min(), df['High'].max(), 40)
        tpo_counts = [((df['Low'] <= p) & (df['High'] >= p)).sum() for p in tpo_prices]

        df = df.dropna().copy()

        # Strength Score with Order Flow
        current = df.iloc[-1]
        score = 50
        if current['Close'] < current['BB_Lower']: score += 18
        if current['RSI'] < 35: score += 15
        if current['MACD'] > current['MACD_Signal']: score += 15
        if current['Stoch_K'] < 25: score += 12
        if abs(current['Close'] - poc_price) / poc_price < 0.015: score += 10
        if current['Close'] > current['VWAP']: score += 8
        if current['Delta'] > 0: score += 12
        if current['Bearish_Divergence']: score -= 15
        if current['Bullish_Divergence']: score += 15

        score = max(0, min(100, int(score)))
        df['Strength_Score'] = score
        df['Signal'] = '🟢 Strong Buy CALL' if score >= 75 else '🔴 Strong Buy PUT' if score <= 25 else 'Neutral'

        extra = {
            'poc': poc_price, 'vah': vah, 'val': val,
            'prices': prices, 'vol_profile': vol_profile,
            'tpo_prices': tpo_prices, 'tpo_counts': tpo_counts
        }
        return df, extra

    except Exception as e:
        return None, f"Error: {str(e)}"

# ====================== MAIN TABS ======================
tab1, tab2 = st.tabs(["📈 Single Symbol Analysis", "📡 Live Dashboard"])

with tab1:
    if st.button("Analyze Single Symbol", type="primary"):
        df, extra = get_data(symbol_single, timeframe, interval)
        if df is not None:
            current = df.iloc[-1]
            color = "green" if "CALL" in current['Signal'] else "red" if "PUT" in current['Signal'] else "orange"

            st.subheader(f"Signal & Strength Score: {symbol_single}")
            st.markdown(f"<h2 style='color:{color};'>{current['Signal']} (Score: {current['Strength_Score']}/100)</h2>", unsafe_allow_html=True)

            # Price Chart with Volume Profile
            fig_price = go.Figure()
            fig_price.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"))
            fig_price.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='red'), name='Upper BB'))
            fig_price.add_trace(go.Scatter(x=df.index, y=df['BB_MA'], line=dict(color='orange', width=2), name='BB MA'))
            fig_price.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='green'), name='Lower BB'))
            fig_price.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='purple', dash='dash'), name='VWAP'))

            max_vol = max(extra['vol_profile'])
            fig_price.add_trace(go.Bar(
                y=extra['prices'],
                x=[v / max_vol * 0.18 * (df['High'].max() - df['Low'].min()) for v in extra['vol_profile']],
                orientation='h', name='Volume Profile', marker_color='rgba(100,149,237,0.7)', xaxis='x2'
            ))
            fig_price.add_hline(y=extra['poc'], line_dash="solid", line_color="blue", line_width=3, annotation_text="POC")
            fig_price.add_hline(y=extra['vah'], line_dash="dash", line_color="red", annotation_text="VAH")
            fig_price.add_hline(y=extra['val'], line_dash="dash", line_color="green", annotation_text="VAL")

            fig_price.update_layout(title="Price + Volume Profile", height=650, template="plotly_dark",
                                    xaxis=dict(domain=[0, 0.82]), xaxis2=dict(domain=[0.82, 1], showticklabels=False))
            st.plotly_chart(fig_price, use_container_width=True)

            # Refined Order Flow Visualization
            st.subheader("Refined Order Flow Analysis")
            fig_flow = go.Figure()

            # Volume Delta Bars
            colors = ['green' if d > 0 else 'red' for d in df['Delta']]
            fig_flow.add_trace(go.Bar(
                x=df.index, y=df['Delta'], name='Volume Delta',
                marker_color=colors, opacity=0.85
            ))

            # Cumulative Delta Line
            fig_flow.add_trace(go.Scatter(
                x=df.index, y=df['Cum_Delta_Zero'], name='Cumulative Delta',
                line=dict(color='yellow', width=2.5)
            ))

            # Divergence Highlights
            bear_div = df[df['Bearish_Divergence']]
            if not bear_div.empty:
                fig_flow.add_trace(go.Scatter(
                    x=bear_div.index, y=bear_div['Cum_Delta_Zero'],
                    mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'),
                    name='Bearish Divergence'
                ))

            bull_div = df[df['Bullish_Divergence']]
            if not bull_div.empty:
                fig_flow.add_trace(go.Scatter(
                    x=bull_div.index, y=bull_div['Cum_Delta_Zero'],
                    mode='markers', marker=dict(color='lime', size=10, symbol='triangle-up'),
                    name='Bullish Divergence'
                ))

            fig_flow.update_layout(
                title="Order Flow - Volume Delta vs Cumulative Delta",
                height=400,
                template="plotly_dark",
                barmode='overlay',
                yaxis_title="Delta / Cum Delta"
            )
            st.plotly_chart(fig_flow, use_container_width=True)

            if current['Bearish_Divergence']:
                st.error("⚠️ Bearish Delta Divergence Detected — Watch for reversal")
            elif current['Bullish_Divergence']:
                st.success("✅ Bullish Delta Divergence Detected — Potential upside")

        else:
            st.error(str(extra))

with tab2:
    st.subheader("Live Multi-Symbol Dashboard")
    data_list = []
    for sym in st.session_state.watchlist:
        df, _ = get_data(sym, timeframe, interval)
        if df is None or df.empty: continue
        current = df.iloc[-1]
        data_list.append({
            "Symbol": sym,
            "Price": f"${current['Close']:.2f}",
            "Signal": current['Signal'],
            "Strength": f"{current['Strength_Score']}/100",
            "RSI": f"{current['RSI']:.1f}",
            "Delta": f"{current['Delta']:,.0f}",
            "Last": current.name.strftime("%Y-%m-%d %H:%M")
        })
    if data_list:
        df_watch = pd.DataFrame(data_list)
        def highlight(val):
            if "Strong Buy CALL" in str(val): return 'background-color: #d4edda; color: green'
            if "Strong Buy PUT" in str(val): return 'background-color: #f8d7da; color: red'
            return ''
        st.dataframe(df_watch.style.applymap(highlight, subset=['Signal']), use_container_width=True)

st.session_state.last_refresh = datetime.now()
st.caption("Order Flow Visualization Refined • Color-coded Delta • Divergence markers • Cleaner layout")
