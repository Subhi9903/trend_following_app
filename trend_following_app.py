import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="CTA Trend Following Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.25rem; }
    .sub-header { color: #555; font-size: 0.95rem; margin-bottom: 1.5rem; }
    .metric-card { background: #f8f9fa; border-radius: 10px; padding: 1rem; text-align: center; }
    .signal-long { background: #d4edda; color: #155724; padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: 600; }
    .signal-short { background: #f8d7da; color: #721c24; padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: 600; }
    .signal-flat { background: #e2e3e5; color: #383d41; padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

TICKERS = {
    "Equities": {
        "Nifty 50": "^NSEI",
        "S&P 500": "^GSPC",
        "FTSE 100": "^FTSE",
        "Nikkei 225": "^N225",
        "Hang Seng": "^HSI",
    },
    "Currencies": {
        "USD/INR": "USDINR=X",
        "GBP/USD": "GBPUSD=X",
        "EUR/USD": "EURUSD=X",
        "USD/JPY": "USDJPY=X",
        "CHF/USD": "CHFUSD=X",
    },
    "Commodities": {
        "Crude Oil (WTI)": "CL=F",
        "Natural Gas": "NG=F",
        "Gold": "GC=F",
        "Silver": "SI=F",
        "Copper": "HG=F",
        "Soybeans": "ZS=F",
        "Corn": "ZC=F",
        "Sugar": "SB=F",
        "Coffee": "KC=F",
    },
    "Crypto": {
        "Bitcoin (BTC)": "BTC-USD",
        "Ethereum (ETH)": "ETH-USD",
    },
    "Fixed Income": {
        "US 10Y (TLT)": "TLT",
    }
}

PORTFOLIO_SIZE = 10_000_000  # 1 Crore in INR, treated as base currency units
RISK_FACTOR = 0.002          # 20 bps per ATR
DMA_FAST = 50
DMA_SLOW = 100
BREAKOUT_PERIOD = 50
ATR_PERIOD = 20
TRAIL_ATR_MULT = 3.0

@st.cache_data(ttl=3600)
def fetch_data(ticker_symbol: str, period: str = "2y") -> pd.DataFrame:
    try:
        df = yf.download(ticker_symbol, period=period, progress=False, auto_adjust=True)
        if df.empty or len(df) < 110:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df[['Open', 'High', 'Low', 'Close']].dropna()
        return df
    except Exception:
        return None

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['DMA_50'] = df['Close'].rolling(DMA_FAST).mean()
    df['DMA_100'] = df['Close'].rolling(DMA_SLOW).mean()
    df['High_50'] = df['High'].rolling(BREAKOUT_PERIOD).max().shift(1)
    df['Low_50'] = df['Low'].rolling(BREAKOUT_PERIOD).min().shift(1)
    # ATR
    df['H_L'] = df['High'] - df['Low']
    df['H_PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L_PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H_L', 'H_PC', 'L_PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(ATR_PERIOD).mean()
    df = df.drop(columns=['H_L', 'H_PC', 'L_PC', 'TR'])
    return df.dropna()

def run_backtest(df: pd.DataFrame) -> dict:
    df = compute_indicators(df).copy()
    capital = PORTFOLIO_SIZE
    equity_curve = []
    trades = []
    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    trail_stop = 0.0
    peak_since_entry = 0.0
    trough_since_entry = 0.0
    units = 0.0
    trade_entry_date = None
    daily_equity = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        date = df.index[i]

        # Mark-to-market
        if position == 1:
            unrealized = (row['Close'] - entry_price) * units
        elif position == -1:
            unrealized = (entry_price - row['Close']) * units
        else:
            unrealized = 0.0
        equity = capital + unrealized
        daily_equity.append({'Date': date, 'Equity': equity, 'Position': position})

        if position == 0:
            # Check long entry
            if row['DMA_50'] > row['DMA_100'] and row['Close'] > row['High_50']:
                position = 1
                entry_price = row['Close']
                units = max(1, int((PORTFOLIO_SIZE * RISK_FACTOR) / row['ATR']))
                peak_since_entry = row['Close']
                trail_stop = entry_price - TRAIL_ATR_MULT * row['ATR']
                trade_entry_date = date
            # Check short entry
            elif row['DMA_50'] < row['DMA_100'] and row['Close'] < row['Low_50']:
                position = -1
                entry_price = row['Close']
                units = max(1, int((PORTFOLIO_SIZE * RISK_FACTOR) / row['ATR']))
                trough_since_entry = row['Close']
                trail_stop = entry_price + TRAIL_ATR_MULT * row['ATR']
                trade_entry_date = date
        elif position == 1:
            peak_since_entry = max(peak_since_entry, row['High'])
            trail_stop = peak_since_entry - TRAIL_ATR_MULT * row['ATR']
            if row['Close'] < trail_stop:
                pnl = (row['Close'] - entry_price) * units
                capital += pnl
                trades.append({
                    'Entry Date': trade_entry_date,
                    'Exit Date': date,
                    'Direction': 'Long',
                    'Entry': round(entry_price, 4),
                    'Exit': round(row['Close'], 4),
                    'Units': int(units),
                    'PnL': round(pnl, 2),
                    'Win': pnl > 0
                })
                position = 0
                units = 0.0
        elif position == -1:
            trough_since_entry = min(trough_since_entry, row['Low'])
            trail_stop = trough_since_entry + TRAIL_ATR_MULT * row['ATR']
            if row['Close'] > trail_stop:
                pnl = (entry_price - row['Close']) * units
                capital += pnl
                trades.append({
                    'Entry Date': trade_entry_date,
                    'Exit Date': date,
                    'Direction': 'Short',
                    'Entry': round(entry_price, 4),
                    'Exit': round(row['Close'], 4),
                    'Units': int(units),
                    'PnL': round(pnl, 2),
                    'Win': pnl > 0
                })
                position = 0
                units = 0.0

    eq_df = pd.DataFrame(daily_equity).set_index('Date')
    eq_df['Returns'] = eq_df['Equity'].pct_change().fillna(0)
    eq_df['Peak'] = eq_df['Equity'].cummax()
    eq_df['Drawdown'] = (eq_df['Equity'] - eq_df['Peak']) / eq_df['Peak']

    total_days = len(eq_df)
    years = total_days / 252
    final_equity = eq_df['Equity'].iloc[-1]
    total_return = (final_equity / PORTFOLIO_SIZE) - 1
    cagr = (final_equity / PORTFOLIO_SIZE) ** (1 / max(years, 0.01)) - 1
    max_dd = eq_df['Drawdown'].min()
    daily_returns = eq_df['Returns']
    sharpe = (daily_returns.mean() * 252 - 0.065) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
    time_in_market = (eq_df['Position'] != 0).sum() / total_days

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    win_rate = trades_df['Win'].mean() if not trades_df.empty else 0
    gross_profit = trades_df[trades_df['PnL'] > 0]['PnL'].sum() if not trades_df.empty else 0
    gross_loss = abs(trades_df[trades_df['PnL'] < 0]['PnL'].sum()) if not trades_df.empty else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return {
        'equity_curve': eq_df,
        'trades': trades_df,
        'indicators': df,
        'metrics': {
            'Total Return': f"{total_return:.1%}",
            'CAGR': f"{cagr:.1%}",
            'Max Drawdown': f"{max_dd:.1%}",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Win Rate': f"{win_rate:.1%}",
            'Profit Factor': f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞",
            'Total Trades': str(len(trades_df)),
            'Time in Market': f"{time_in_market:.1%}",
        },
        'total_return_val': total_return,
        'max_dd_val': max_dd,
        'sharpe_val': sharpe,
    }

def get_current_signal(df: pd.DataFrame) -> dict:
    df = compute_indicators(df)
    if df.empty:
        return {'signal': 'NO DATA', 'details': {}}
    last = df.iloc[-1]
    signal = 'FLAT'
    if last['DMA_50'] > last['DMA_100']:
        if last['Close'] >= last['High_50']:
            signal = 'LONG ↑'
        else:
            signal = 'TREND UP (no breakout)'
    elif last['DMA_50'] < last['DMA_100']:
        if last['Close'] <= last['Low_50']:
            signal = 'SHORT ↓'
        else:
            signal = 'TREND DOWN (no breakout)'

    units_suggested = max(1, int((PORTFOLIO_SIZE * RISK_FACTOR) / last['ATR'])) if last['ATR'] > 0 else 0

    return {
        'signal': signal,
        'details': {
            'Close': f"{last['Close']:.4f}",
            '50-DMA': f"{last['DMA_50']:.4f}",
            '100-DMA': f"{last['DMA_100']:.4f}",
            '50-Day High': f"{last['High_50']:.4f}",
            '50-Day Low': f"{last['Low_50']:.4f}",
            'ATR (20)': f"{last['ATR']:.4f}",
            'Suggested Units': str(units_suggested),
            '3-ATR Stop (Long)': f"{last['Close'] - TRAIL_ATR_MULT * last['ATR']:.4f}",
            '3-ATR Stop (Short)': f"{last['Close'] + TRAIL_ATR_MULT * last['ATR']:.4f}",
        }
    }

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    flat_tickers = {}
    for asset_class, tickers in TICKERS.items():
        for name, sym in tickers.items():
            flat_tickers[f"{name} ({asset_class})"] = sym

    selected_display = st.selectbox(
        "Select Ticker for Backtest",
        options=list(flat_tickers.keys()),
        index=0
    )
    backtest_ticker = flat_tickers[selected_display]

    st.markdown("---")
    st.markdown("**Custom Ticker**")
    custom_ticker = st.text_input("Enter yfinance symbol (e.g. AAPL, BTC-USD)", value="")

    if custom_ticker.strip():
        backtest_ticker = custom_ticker.strip().upper()
        selected_display = backtest_ticker

    st.markdown("---")
    st.markdown("### Strategy Parameters")
    st.markdown(f"- Portfolio: ₹{PORTFOLIO_SIZE/1e7:.0f} Cr")
    st.markdown(f"- Risk per trade: {RISK_FACTOR*100:.1f}% of ATR")
    st.markdown(f"- Fast DMA: {DMA_FAST}-day")
    st.markdown(f"- Slow DMA: {DMA_SLOW}-day")
    st.markdown(f"- Breakout: {BREAKOUT_PERIOD}-day high/low")
    st.markdown(f"- ATR Period: {ATR_PERIOD}-day")
    st.markdown(f"- Trail Stop: {TRAIL_ATR_MULT}×ATR")

    st.markdown("---")
    st.markdown("### 📖 Strategy Logic")
    st.markdown("""
**Entry (Long):** 50-DMA > 100-DMA AND price breaks 50-day high  
**Entry (Short):** 50-DMA < 100-DMA AND price breaks 50-day low  
**Exit:** 3-ATR trailing stop from peak/trough  
**Size:** (0.2% × Portfolio) / ATR  
    """)

# ─── MAIN ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📈 CTA Trend Following Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Diversified Trend Following · Clenow-style · 2-Year Backtest</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎯 Backtest", "📡 Live Signals (All)", "📋 Trade Log"])

with tab1:
    st.markdown(f"### Backtest: {selected_display}")

    with st.spinner(f"Fetching data for {backtest_ticker}..."):
        df_raw = fetch_data(backtest_ticker)

    if df_raw is None:
        st.error(f"Could not fetch data for '{backtest_ticker}'. Try another symbol.")
    else:
        results = run_backtest(df_raw)
        eq = results['equity_curve']
        ind = results['indicators']
        metrics = results['metrics']

        # Metric cards
        cols = st.columns(8)
        metric_items = list(metrics.items())
        for i, (label, val) in enumerate(metric_items):
            with cols[i]:
                color = "#155724" if (label in ['Total Return','CAGR','Sharpe Ratio','Win Rate','Profit Factor'] and
                                       not val.startswith('-') and val not in ['0.00','0.0%']) else \
                        "#721c24" if label == 'Max Drawdown' else "#1a1a2e"
                st.metric(label, val)

        st.markdown("---")

        # Equity Curve
        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.5, 0.25, 0.25],
            shared_xaxes=True,
            subplot_titles=("Equity Curve vs Buy & Hold", "Drawdown (%)", "Price with DMA & Signals")
        )

        bh_equity = PORTFOLIO_SIZE * (df_raw['Close'] / df_raw['Close'].iloc[0])
        bh_equity = bh_equity.reindex(eq.index).fillna(method='ffill')

        fig.add_trace(go.Scatter(x=eq.index, y=eq['Equity'], name='Strategy', line=dict(color='#2563eb', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=bh_equity.index, y=bh_equity.values, name='Buy & Hold', line=dict(color='#9ca3af', width=1.5, dash='dash')), row=1, col=1)

        fig.add_trace(go.Scatter(x=eq.index, y=eq['Drawdown'] * 100, name='Drawdown', fill='tozeroy',
                                  line=dict(color='#ef4444'), fillcolor='rgba(239,68,68,0.15)'), row=2, col=1)

        fig.add_trace(go.Scatter(x=ind.index, y=ind['Close'], name='Price', line=dict(color='#1a1a2e', width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=ind.index, y=ind['DMA_50'], name='50-DMA', line=dict(color='#f59e0b', width=1.2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=ind.index, y=ind['DMA_100'], name='100-DMA', line=dict(color='#8b5cf6', width=1.2)), row=3, col=1)

        if not results['trades'].empty:
            longs = results['trades'][results['trades']['Direction'] == 'Long']
            shorts = results['trades'][results['trades']['Direction'] == 'Short']
            if not longs.empty:
                fig.add_trace(go.Scatter(x=longs['Entry Date'], y=ind.loc[ind.index.isin(longs['Entry Date']), 'Close'],
                                          mode='markers', name='Long Entry', marker=dict(color='#16a34a', size=8, symbol='triangle-up')), row=3, col=1)
            if not shorts.empty:
                fig.add_trace(go.Scatter(x=shorts['Entry Date'], y=ind.loc[ind.index.isin(shorts['Entry Date']), 'Close'],
                                          mode='markers', name='Short Entry', marker=dict(color='#dc2626', size=8, symbol='triangle-down')), row=3, col=1)

        fig.update_layout(
            height=750,
            showlegend=True,
            plot_bgcolor='#fafafa',
            paper_bgcolor='white',
            font=dict(family='sans-serif', size=12),
            legend=dict(orientation='h', y=1.02, x=0),
            margin=dict(l=40, r=20, t=60, b=20)
        )
        fig.update_yaxes(tickprefix='₹', row=1, col=1)
        fig.update_yaxes(ticksuffix='%', row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # Current Signal
        sig = get_current_signal(df_raw)
        st.markdown("### 📡 Current Signal")
        signal_text = sig['signal']
        if 'LONG' in signal_text:
            badge = f'<span class="signal-long">{signal_text}</span>'
        elif 'SHORT' in signal_text:
            badge = f'<span class="signal-short">{signal_text}</span>'
        else:
            badge = f'<span class="signal-flat">{signal_text}</span>'
        st.markdown(badge, unsafe_allow_html=True)

        det_cols = st.columns(3)
        items = list(sig['details'].items())
        for i, (k, v) in enumerate(items):
            with det_cols[i % 3]:
                st.metric(k, v)

with tab2:
    st.markdown("### 📡 Live Signals Across All Tickers")
    st.info("Fetching signals for all configured tickers. This may take ~30 seconds...")

    DEFAULT_SCAN = [
        ("Nifty 50", "^NSEI"), ("S&P 500", "^GSPC"), ("Gold", "GC=F"),
        ("Bitcoin", "BTC-USD"), ("Crude Oil", "CL=F"), ("EUR/USD", "EURUSD=X"),
        ("Silver", "SI=F"), ("Natural Gas", "NG=F"), ("USD/INR", "USDINR=X"),
        ("Copper", "HG=F"), ("TLT (US 10Y)", "TLT"), ("Ethereum", "ETH-USD"),
    ]

    signal_rows = []
    progress = st.progress(0)
    for idx, (name, sym) in enumerate(DEFAULT_SCAN):
        progress.progress((idx + 1) / len(DEFAULT_SCAN))
        d = fetch_data(sym)
        if d is not None:
            s = get_current_signal(d)
            signal_rows.append({
                'Ticker': name,
                'Symbol': sym,
                'Signal': s['signal'],
                'Close': s['details'].get('Close', 'N/A'),
                '50-DMA': s['details'].get('50-DMA', 'N/A'),
                '100-DMA': s['details'].get('100-DMA', 'N/A'),
                'ATR': s['details'].get('ATR (20)', 'N/A'),
                'Units': s['details'].get('Suggested Units', 'N/A'),
            })
        else:
            signal_rows.append({'Ticker': name, 'Symbol': sym, 'Signal': 'NO DATA',
                                  'Close': '-', '50-DMA': '-', '100-DMA': '-', 'ATR': '-', 'Units': '-'})
    progress.empty()

    sig_df = pd.DataFrame(signal_rows)

    def color_signal(val):
        if 'LONG' in str(val): return 'background-color: #d4edda; color: #155724; font-weight: bold'
        elif 'SHORT' in str(val): return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
        elif 'TREND UP' in str(val): return 'background-color: #fff3cd; color: #856404'
        elif 'TREND DOWN' in str(val): return 'background-color: #fde8e9; color: #a94442'
        return 'color: #6c757d'

    st.dataframe(
        sig_df.style.applymap(color_signal, subset=['Signal']),
        use_container_width=True,
        height=450
    )

with tab3:
    st.markdown("### 📋 Trade Log")
    if df_raw is not None and not results['trades'].empty:
        tdf = results['trades'].copy()
        tdf['PnL'] = tdf['PnL'].apply(lambda x: f"₹{x:,.0f}")
        tdf['Win'] = tdf['Win'].apply(lambda x: '✅' if x else '❌')
        st.dataframe(tdf, use_container_width=True, height=500)
        st.markdown(f"**Total trades:** {len(results['trades'])} | "
                    f"**Winners:** {results['trades']['Win'].sum()} | "
                    f"**Losers:** {(~results['trades']['Win']).sum()}")
    else:
        st.info("Run a backtest in Tab 1 first.")

st.markdown("---")
st.markdown(
    "<center style='color:#9ca3af; font-size:0.8rem;'>CTA Trend Following · Based on Clenow (2013) · "
    "Data: Yahoo Finance · Zero slippage assumption · Last 2 years only</center>",
    unsafe_allow_html=True
)
