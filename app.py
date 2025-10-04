# deep3_auto_trading_longshort.py
# -*- coding: utf-8 -*-
"""
Wyckoff Shark - Sistema Profesional de Trading (LONG/SHORT, ejecución automática multi-timeframe)
- Soporta AUTO o forzar LONG/SHORT
- Monto por orden en USDT o % riesgo (ponga 0 para usar % riesgo)
- Coloca SL + TPs solo después de confirmar que la orden de entrada está abierta/llenada
- Evita errores ReduceOnly colocando SL/TP tras apertura de posición
- Muestra todos los precios con 8 decimales
- fetch_ohlcv usa _exchange para evitar UnhashableParamError de Streamlit
"""
import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import time
import decimal
import plotly.graph_objects as go
from scipy.signal import argrelextrema
from datetime import datetime, timezone

# -------------------------
# CONFIGURACIÓN / DEFAULTS
# -------------------------
st.set_page_config(page_title="Wyckoff Shark - LONG/SHORT (Auto)", layout="wide", initial_sidebar_state="expanded")
st.title("🦈 Wyckoff Shark - Sistema Profesional de Trading (LONG/SHORT Auto)")

# ---------- Panel lateral de configuración ----------
st.sidebar.header("🎯 Configuración")
MARKET_TYPE = st.sidebar.selectbox("Tipo de Mercado", ["Futuros"], index=0)
SYMBOL = st.sidebar.text_input("Símbolo (ej: TRUTH/USDT)", "TRUTH/USDT")
TF = st.sidebar.selectbox("Timeframe", ["1m", "3m", "5m", "15m", "1h"], index=0)

col1, col2 = st.sidebar.columns(2)
with col1:
    LIMIT_SLIDER = st.slider("Velas a Cargar", 100, 2000, 500, key="limit_slider")
with col2:
    LIMIT_INPUT = st.number_input("Velas (escribir)", min_value=100, max_value=2000, value=500, step=10, key="limit_input")
LIMIT = LIMIT_INPUT

POLL_SECONDS = st.sidebar.slider("Tasa de Actualización (segundos)", 2, 60, 5)
VOL_MULT = st.sidebar.slider("Umbral de Volumen (x promedio)", 1.5, 5.0, 2.0)
RISK_PER_TRADE = st.sidebar.slider("Riesgo por Operación (%)", 0.1, 5.0, 1.0) / 100.0
CAPITAL = st.sidebar.number_input("Capital Disponible (USDT)", min_value=1.0, value=60.0, step=1.0)
LEVERAGE = st.sidebar.selectbox("Apalancamiento", [1, 3, 5, 10, 20, 25, 50, 75, 100], index=2)

# Monto por orden (si >0 prioriza sobre % riesgo)
st.sidebar.markdown("---")
MONTO_USDT = st.sidebar.number_input("Monto por Orden (USDT) - poner 0 para usar % riesgo", min_value=0.0, value=5.0, step=1.0)

# Forzar tipo de operación o AUTO
st.sidebar.markdown("---")
tipo_operacion_forzada = st.sidebar.selectbox("Forzar tipo de operación (override)", ["AUTO", "LONG", "SHORT"])

st.sidebar.markdown("---")
st.sidebar.header("🔐 Trading / API")

# Modo LIVE activado por defecto
LIVE_MODE = st.sidebar.checkbox("Habilitar Trading en VIVO", value=True)
USE_TESTNET = st.sidebar.checkbox("Usar TESTNET (Recomendado)", value=True)

API_KEY = st.sidebar.text_input("Binance API Key", type="password")
API_SECRET = st.sidebar.text_input("Binance API Secret", type="password")

# Ejecución automática (sin confirmación manual) - opcional
st.sidebar.markdown("---")
AUTO_TRADE = st.sidebar.checkbox("Habilitar Ejecución Automática (AUTO_TRADE)", value=False)
AUTO_CONFIDENCE_THRESH = st.sidebar.slider("Umbral Confianza para AUTO_TRADE", 0.5, 0.95, 0.65, step=0.05)

# -------------------------
# UTILIDADES
# -------------------------
def fmt(x, d=8):
    try:
        return f"{decimal.Decimal(str(x)):.{d}f}"
    except Exception:
        try:
            return f"{float(x):.{d}f}"
        except Exception:
            return str(x)

def now_ts():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

def to_epoch_ms(dt: datetime):
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

# -------------------------
# INICIALIZACIÓN EXCHANGE
# -------------------------
@st.cache_resource
def init_exchange(market_type, api_key=None, api_secret=None, testnet=True):
    opts = {'enableRateLimit': True}
    if market_type == "Futuros":
        opts['options'] = {'defaultType': 'future'}
    exchange = ccxt.binance(opts)
    if api_key and api_secret:
        exchange.apiKey = api_key
        exchange.secret = api_secret
    try:
        if testnet:
            exchange.set_sandbox_mode(True)
    except Exception:
        pass
    try:
        exchange.load_markets()
    except Exception:
        # ignore load error here, will surface later
        pass
    return exchange

exchange = init_exchange(MARKET_TYPE, API_KEY if API_KEY else None, API_SECRET if API_SECRET else None, testnet=USE_TESTNET)

# -------------------------
# HELPERS API (apalancamiento, precision, order utils)
# -------------------------
def set_leverage_for_symbol(exchange, symbol, leverage):
    try:
        if hasattr(exchange, 'set_leverage'):
            exchange.set_leverage(leverage, symbol)
            return True, 'set_leverage'
    except Exception:
        pass
    try:
        params = {'symbol': symbol.replace('/', ''), 'leverage': int(leverage)}
        if hasattr(exchange, 'fapiPrivate_post_leverage'):
            exchange.fapiPrivate_post_leverage(params)
            return True, 'fapiPrivate_post_leverage'
        if hasattr(exchange, 'private_post_leverage'):
            exchange.private_post_leverage(params)
            return True, 'private_post_leverage'
    except Exception:
        pass
    return False, 'not_supported'

def get_amount_precision(exchange, symbol):
    try:
        m = exchange.markets.get(symbol) or exchange.markets.get(symbol.replace('/', ''))
        if m and 'precision' in m and 'amount' in m['precision']:
            return int(m['precision']['amount'])
    except Exception:
        pass
    return 6

def sleep_poll(interval_seconds):
    # helper to allow interruptible sleep in Streamlit
    time.sleep(interval_seconds)

# -------------------------
# OBTENER DATOS OHLCV (evitar UnhashableParamError)
# -------------------------
@st.cache_data(ttl=3)
def fetch_ohlcv(_exchange, symbol, timeframe="1m", limit=500):
    try:
        # algunos exchanges requieren símbolo sin '/'; ccxt maneja ambos
        params = {'type': 'future'}
        ohlcv = _exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, params=params)
        df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
        return df
    except Exception as e:
        st.error(f"Error obteniendo datos OHLCV: {e}")
        return None

# -------------------------
# INDICADORES
# -------------------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi_series(prices, period=14):
    prices = np.array(prices, dtype=float)
    delta = np.diff(prices)
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    up_ewm = pd.Series(up).ewm(alpha=1/period, adjust=False).mean()
    down_ewm = pd.Series(down).ewm(alpha=1/period, adjust=False).mean()
    rs = up_ewm / down_ewm
    rsi = 100 - (100 / (1 + rs))
    rsi_full = np.concatenate((np.full(period, np.nan), rsi.values))
    return pd.Series(rsi_full)

# -------------------------
# S/R detection (unchanged)
# -------------------------
def detect_sr(df, order=5, cluster_tol=0.02):
    highs = df['high'].values
    lows = df['low'].values
    max_idx = argrelextrema(highs, np.greater_equal, order=order)[0]
    min_idx = argrelextrema(lows, np.less_equal, order=order)[0]
    resistances = [highs[i] for i in max_idx]
    supports = [lows[i] for i in min_idx]
    def cluster(vals):
        vals = sorted(vals)
        clusters = []
        for v in vals:
            if not clusters:
                clusters.append([v]); continue
            m = np.mean(clusters[-1])
            if abs(v - m) / m <= cluster_tol:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        return [np.mean(c) for c in clusters]
    return cluster(supports), cluster(resistances)

# -------------------------
# Wyckoff signals (unchanged)
# -------------------------
def detect_spring(df, supports, vol_mult=1.7):
    if not supports:
        return False, None, 0
    sup = min(supports)
    current_low = df['low'].iloc[-1]
    current_close = df['close'].iloc[-1]
    current_volume = df['volume'].iloc[-1]
    vol_avg = df['volume'].rolling(20).mean().iloc[-1]
    spring_conditions = (
        current_low < sup * 0.995 and
        current_close > sup and
        current_volume > vol_avg * vol_mult
    )
    if spring_conditions:
        confidence = 0.7
        if current_volume > vol_avg * 3:
            confidence += 0.15
        if df['rsi14'].iloc[-1] < 35:
            confidence += 0.10
        confidence = min(confidence, 0.95)
        return True, sup, confidence
    return False, None, 0

def detect_upthrust(df, resistances, vol_mult=1.7):
    if not resistances:
        return False, None, 0
    res = max(resistances)
    current_high = df['high'].iloc[-1]
    current_close = df['close'].iloc[-1]
    current_volume = df['volume'].iloc[-1]
    vol_avg = df['volume'].rolling(20).mean().iloc[-1]
    upthrust_conditions = (
        current_high > res * 1.005 and
        current_close < res and
        current_volume > vol_avg * vol_mult
    )
    if upthrust_conditions:
        confidence = 0.7
        if current_volume > vol_avg * 3:
            confidence += 0.15
        if df['rsi14'].iloc[-1] > 65:
            confidence += 0.10
        confidence = min(confidence, 0.95)
        return True, res, confidence
    return False, None, 0

# -------------------------
# Determine phase & recommend (unchanged)
# -------------------------
def determine_phase_and_recommendation(df, supports, resistances, vol_mult):
    current_price = df['close'].iloc[-1]
    ema20 = df['ema20'].iloc[-1]
    ema50 = df['ema50'].iloc[-1]
    current_volume = df['volume'].iloc[-1]
    vol_avg = df['volume'].rolling(20).mean().iloc[-1]
    spring_flag, spring_price, spring_confidence = detect_spring(df, supports, vol_mult)
    ut_flag, ut_price, ut_confidence = detect_upthrust(df, resistances, vol_mult)
    if spring_flag:
        phase = "FASE A: ACUMULACIÓN (Spring Detectado)"
        desc = f"🟢 SEÑAL FUERTE DE COMPRA - Spring confirmado en {fmt(spring_price)}"
        action = "COMPRAR"
        confidence = spring_confidence
        return phase, desc, action, spring_price, ut_price, confidence
    if ut_flag:
        phase = "FASE A: DISTRIBUCIÓN (Upthrust Detectado)"
        desc = f"🔴 SEÑAL FUERTE DE VENTA - Upthrust confirmado en {fmt(ut_price)}"
        action = "VENDER"
        confidence = ut_confidence
        return phase, desc, action, spring_price, ut_price, confidence
    if current_price > ema20 > ema50:
        if current_volume > vol_avg:
            phase = "FASE B: MARKUP (Tendencia Alcista)"
            desc = "📈 TENDENCIA ALCISTA CONFIRMADA - Comprar en retrocesos"
            action = "COMPRAR"
            confidence = 0.65
        else:
            phase = "FASE B: MARKUP (Alcista Débil)"
            desc = "⚠️ ALCISTA SIN VOLUMEN - Esperar confirmación"
            action = "MANTENER"
            confidence = 0.4
        return phase, desc, action, spring_price, ut_price, confidence
    if current_price < ema20 < ema50:
        if current_volume > vol_avg:
            phase = "FASE B: MARKDOWN (Tendencia Bajista)"
            desc = "📉 TENDENCIA BAJISTA CONFIRMADA - Vender en rebotes"
            action = "VENDER"
            confidence = 0.65
        else:
            phase = "FASE B: MARKDOWN (Bajista Débil)"
            desc = "⚠️ BAJISTA SIN VOLUMEN - Esperar confirmación"
            action = "MANTENER"
            confidence = 0.4
        return phase, desc, action, spring_price, ut_price, confidence
    phase = "FASE C: CONSOLIDACIÓN"
    desc = "↔️ MERCADO EN RANGO - Esperar señal clara"
    action = "MANTENER"
    confidence = 0.3
    return phase, desc, action, spring_price, ut_price, confidence

# -------------------------
# Build trade plan (supports LONG/SHORT and optional monto_usdt)
# -------------------------
def build_trade_plan(df, supports, resistances, capital, leverage, risk_per_trade, action, monto_usdt=0.0):
    price = float(df['close'].iloc[-1])
    ema20 = float(df['ema20'].iloc[-1])

    # default placeholders
    entry_price = price
    sl = price
    tp1 = price
    tp2 = price
    tp3 = price
    entry_zone_low = ema20 * 0.998
    entry_zone_high = ema20 * 1.002

    if action == "COMPRAR":
        if supports:
            entry_zone_low = min(supports) * 0.999
            entry_zone_high = min(supports) * 1.01
        else:
            entry_zone_low = ema20 * 0.995
            entry_zone_high = ema20 * 1.005
        entry_price = (entry_zone_low + entry_zone_high) / 2
        sl = entry_zone_low * 0.985
        tp1 = entry_price * 1.008
        tp2 = entry_price * 1.015
        tp3 = entry_price * 1.025
        # ensure monotonic
        if not (sl < entry_price < tp1 < tp2 < tp3):
            tp1 = entry_price * 1.01
            tp2 = entry_price * 1.02
            tp3 = entry_price * 1.035
        direction = "LONG"
        side = "BUY"
    elif action == "VENDER":
        if resistances:
            entry_zone_high = max(resistances) * 1.001
            entry_zone_low = max(resistances) * 0.99
        else:
            entry_zone_high = ema20 * 1.005
            entry_zone_low = ema20 * 0.995
        entry_price = (entry_zone_low + entry_zone_high) / 2
        sl = entry_zone_high * 1.015
        tp1 = entry_price * 0.992
        tp2 = entry_price * 0.985
        tp3 = entry_price * 0.975
        if not (tp3 < tp2 < tp1 < entry_price < sl):
            tp1 = entry_price * 0.99
            tp2 = entry_price * 0.98
            tp3 = entry_price * 0.965
        direction = "SHORT"
        side = "SELL"
    else:
        # MANTENER
        entry_price = ema20
        sl = entry_price * 0.99
        tp1 = entry_price * 1.01
        tp2 = entry_price * 1.02
        tp3 = entry_price * 1.035
        direction = "NONE"
        side = "NONE"

    # calculate notional and amount
    if monto_usdt and monto_usdt > 0:
        position_notional = monto_usdt * leverage
    else:
        stop_dist = abs((entry_price - sl) / entry_price) if entry_price > 0 else 0.01
        if stop_dist <= 0.0005:
            stop_dist = 0.01
            sl = entry_price * (1 - stop_dist) if direction == "LONG" else entry_price * (1 + stop_dist)
        risk_amount = capital * risk_per_trade
        position_notional = risk_amount / stop_dist if stop_dist > 0 else capital * leverage

    max_notional = capital * leverage
    if position_notional > max_notional:
        position_notional = max_notional

    amount_base = position_notional / entry_price if entry_price > 0 else 0.0
    amount_prec = get_amount_precision(exchange, SYMBOL)
    # round amount to precision
    try:
        amount_base = round(amount_base, amount_prec)
    except Exception:
        amount_base = float(amount_base)

    stop_dist_pct = abs((entry_price - sl) / entry_price) if entry_price > 0 else 0.0
    potential_loss = position_notional * stop_dist_pct

    if direction == "LONG":
        potential_profit_tp1 = position_notional * (tp1 - entry_price) / entry_price
        potential_profit_tp2 = position_notional * (tp2 - entry_price) / entry_price
        potential_profit_tp3 = position_notional * (tp3 - entry_price) / entry_price
    elif direction == "SHORT":
        potential_profit_tp1 = position_notional * (entry_price - tp1) / entry_price
        potential_profit_tp2 = position_notional * (entry_price - tp2) / entry_price
        potential_profit_tp3 = position_notional * (entry_price - tp3) / entry_price
    else:
        potential_profit_tp1 = potential_profit_tp2 = potential_profit_tp3 = 0.0

    plan = {
        "action": action,
        "direction": direction,        # LONG / SHORT / NONE
        "side": side,                  # BUY / SELL / NONE
        "entry_price": float(entry_price),
        "entry_zone": (float(entry_zone_low), float(entry_zone_high)),
        "sl": float(sl),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "tp3": float(tp3),
        "position_notional": float(position_notional),
        "amount_base": float(amount_base),
        "stop_dist_pct": float(stop_dist_pct),
        "potential_loss": float(potential_loss),
        "potential_profit_tp1": float(potential_profit_tp1),
        "potential_profit_tp2": float(potential_profit_tp2),
        "potential_profit_tp3": float(potential_profit_tp3),
        "risk_reward_tp1": (potential_profit_tp1 / potential_loss) if potential_loss>0 else 0,
        "risk_reward_tp2": (potential_profit_tp2 / potential_loss) if potential_loss>0 else 0,
        "risk_reward_tp3": (potential_profit_tp3 / potential_loss) if potential_loss>0 else 0,
        "margin_used": float(position_notional / leverage),
        "monto_usdt": float(monto_usdt)
    }
    return plan

# -------------------------
# UTIL: check order status y esperar fill (timeout)
# -------------------------
def wait_for_order_fill(exchange, symbol, order_id, timeout=15, poll_interval=1.0):
    """Espera hasta que la orden pase a filled o hasta timeout (segundos). Devuelve order dict o None."""
    t0 = time.time()
    try:
        while time.time() - t0 < timeout:
            try:
                o = exchange.fetch_order(order_id, symbol)
                status = (o.get('status') or '').lower()
                if status in ['closed', 'filled', 'partially_filled'] or float(o.get('filled', 0)) > 0:
                    return o
            except Exception:
                # algunos endpoints devuelven error; intentar fetchOrder desde info id
                try:
                    o = exchange.fetch_order(order_id, symbol)
                    status = (o.get('status') or '').lower()
                    if status in ['closed', 'filled']:
                        return o
                except Exception:
                    pass
            time.sleep(poll_interval)
    except Exception:
        pass
    return None

# -------------------------
# PLACE ORDERS - coloca entrada, espera fill, luego SL+TPs (reduceOnly)
# -------------------------
def place_orders_binance_futures(exchange, symbol, side, amount_base, entry_price, sl_price, tp_prices, max_wait_fill=12):
    """
    Lógica:
    1) Intentar LIMIT entry (GTC). Si falla -> intentar MARKET.
    2) Si la entrada queda 'open', esperar hasta max_wait_fill segundos a que llene; si no llena -> intentar MARKET cancelando LIMIT.
    3) Tras confirmación (open/filled) de entrada, colocar STOP_MARKET (reduceOnly) para SL y LIMIT reduceOnly para cada TP.
    """
    results = {"open": None, "open_info": None, "sl_order": None, "tp_orders": [], "errors": []}
    side_uc = side.upper()
    amount_prec = get_amount_precision(exchange, symbol)
    try:
        amount_base = round(float(amount_base), amount_prec)
    except Exception:
        amount_base = float(amount_base)
    # 1) Crear LIMIT entry
    created_open = None
    try:
        created_open = exchange.create_order(symbol, 'LIMIT', side_uc, amount_base, float(entry_price), {'timeInForce': 'GTC'})
        results['open'] = created_open
    except Exception as e:
        results['errors'].append(f"Error Open LIMIT: {e}")
        # fallback market
        try:
            created_open = exchange.create_order(symbol, 'MARKET', side_uc, amount_base)
            results['open'] = created_open
        except Exception as e2:
            results['errors'].append(f"Error Open MARKET fallback: {e2}")
            return results

    # Try to determine order id & check fill
    order_id = None
    try:
        # ccxt sometimes returns nested structure
        if isinstance(created_open, dict) and 'id' in created_open:
            order_id = created_open['id']
        elif isinstance(created_open, dict) and 'info' in created_open and 'orderId' in created_open['info']:
            order_id = str(created_open['info']['orderId'])
    except Exception:
        order_id = None

    # If limit created and not filled, wait a bit then fallback to market if not filled
    filled_order = None
    if order_id:
        filled_order = wait_for_order_fill(exchange, symbol, order_id, timeout=max_wait_fill, poll_interval=1.0)
    else:
        # If no order_id, try to inspect created_open fields
        try:
            if created_open and (created_open.get('status') in ['closed','filled'] or float(created_open.get('filled', 0))>0):
                filled_order = created_open
        except Exception:
            filled_order = None

    # If still not filled -> attempt MARKET (to ensure position open)
    if not filled_order:
        # If the original order was LIMIT and still open, try to cancel it then MARKET
        try:
            if order_id:
                try:
                    exchange.cancel_order(order_id, symbol)
                except Exception:
                    pass
            market_resp = exchange.create_order(symbol, 'MARKET', side_uc, amount_base)
            results['open'] = market_resp
            filled_order = market_resp
        except Exception as e:
            results['errors'].append(f"Error filling via MARKET fallback: {e}")
            return results

    results['open_info'] = filled_order

    # Now place SL and TPs using reduceOnly true and opposite side
    try:
        # Stop Loss (STOP_MARKET) - opposite side to close
        stop_side = 'SELL' if side_uc == 'BUY' else 'BUY'
        sl_params = {'stopPrice': float(sl_price), 'reduceOnly': True}
        try:
            sl_order = exchange.create_order(symbol, 'STOP_MARKET', stop_side, amount_base, None, sl_params)
            results['sl_order'] = sl_order
        except Exception as e:
            # Some ccxt versions use different endpoints: try raw api
            results['errors'].append(f"Error Stop Loss: {e}")
    except Exception as e:
        results['errors'].append(f"Error StopLoss general: {e}")

    # Place TP LIMIT orders (reduceOnly)
    tp_results = []
    try:
        # decide distribution (equal)
        tp_amount = amount_base / len(tp_prices) if len(tp_prices) > 0 else 0
        tp_amount = round(tp_amount, amount_prec)
    except Exception:
        tp_amount = amount_base

    for i, tp in enumerate(tp_prices):
        try:
            tp_side = 'SELL' if side_uc == 'BUY' else 'BUY'
            tp_order = exchange.create_order(symbol, 'LIMIT', tp_side, tp_amount, float(tp), {'reduceOnly': True, 'timeInForce': 'GTC'})
            tp_results.append(tp_order)
        except Exception as e:
            tp_results.append({"error": f"TP{i+1}: {str(e)}"})
            results['errors'].append(f"Error TP{i+1}: {str(e)}")
    results['tp_orders'] = tp_results

    return results

# -------------------------
# TABLA DE ORDENES
# -------------------------
def register_order_in_table(symbol, plan, api_response=None, estado="OPEN"):
    if 'orders_table' not in st.session_state:
        st.session_state['orders_table'] = pd.DataFrame(columns=[
            "Hora", "Símbolo", "Tipo", "Acción", "Entrada", "SL", "TP1", "TP2", "TP3",
            "Cantidad", "Monto_USDT", "Apalancamiento", "Estado", "API_Response"
        ])
    new_row = {
        "Hora": now_ts(),
        "Símbolo": symbol,
        "Tipo": plan.get('direction', 'NONE'),   # LONG/SHORT/NONE
        "Acción": plan.get('side', 'NONE'),      # BUY/SELL/NONE
        "Entrada": fmt(plan.get('entry_price', 0.0), 8),
        "SL": fmt(plan.get('sl', 0.0), 8),
        "TP1": fmt(plan.get('tp1', 0.0), 8),
        "TP2": fmt(plan.get('tp2', 0.0), 8),
        "TP3": fmt(plan.get('tp3', 0.0), 8),
        "Cantidad": plan.get('amount_base', 0.0),
        "Monto_USDT": fmt(plan.get('monto_usdt', MONTO_USDT), 8),
        "Apalancamiento": f"{LEVERAGE}x",
        "Estado": estado,
        "API_Response": str(api_response)[:4000] if api_response is not None else ""
    }
    st.session_state['orders_table'] = pd.concat([pd.DataFrame([new_row]), st.session_state['orders_table']], ignore_index=True)

# -------------------------
# FUNCIÓN DISPLAY_TRADING_SIGNAL (FALTANTE)
# -------------------------
def display_trading_signal(phase, desc, action, confidence, plan):
    st.markdown("---")
    st.subheader("📊 SEÑAL DE TRADING WYCKOFF")
    
    # Mostrar fase y descripción
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info(f"**Fase Detectada:** {phase}")
        st.write(f"**Descripción:** {desc}")
        
        # Mostrar acción recomendada con color según el tipo
        if action == "COMPRAR":
            st.success(f"**🎯 ACCIÓN RECOMENDADA:** {action}")
        elif action == "VENDER":
            st.error(f"**🎯 ACCIÓN RECOMENDADA:** {action}")
        else:
            st.warning(f"**🎯 ACCIÓN RECOMENDADA:** {action}")
    
    with col2:
        # Mostrar confianza con barra de progreso
        st.metric("Confianza de Señal", f"{confidence:.1%}")
        st.progress(float(confidence))
        
        # Mostrar tipo de operación del plan
        if plan['direction'] == "LONG":
            st.success("**Tipo de Operación:** LONG")
        elif plan['direction'] == "SHORT":
            st.error("**Tipo de Operación:** SHORT")
        else:
            st.info("**Tipo de Operación:** NINGUNA")
    
    # Información adicional sobre la señal
    with st.expander("🔍 Detalles de la Señal"):
        st.write(f"**Confianza Numérica:** {confidence:.3f}")
        st.write(f"**Acción del Plan:** {plan.get('action', 'N/A')}")
        st.write(f"**Dirección del Plan:** {plan.get('direction', 'N/A')}")
        
        # Interpretación de la confianza
        if confidence >= 0.8:
            st.success("**🔔 ALTA CONFIABILIDAD** - Señal muy fuerte")
        elif confidence >= 0.6:
            st.warning("**📊 CONFIABILIDAD MEDIA** - Señal moderada")
        else:
            st.info("**⚡ CONFIABILIDAD BAJA** - Señal débil, esperar confirmación")

# -------------------------
# VISUALIZACIÓN Y PANEL (mantener diseño original, ajustar textos/decimales)
# -------------------------
def display_trade_plan(plan, capital, leverage, symbol, current_price):
    st.markdown("---")
    st.subheader("💰 PLAN DE TRADING - ESTRATEGIA WYCKOFF")
    # show Type LONG/SHORT
    tipo = plan.get('direction', 'NONE')
    if tipo == "LONG":
        st.info(f"📈 Tipo de Operación: LONG (Compra)")
    elif tipo == "SHORT":
        st.info(f"📉 Tipo de Operación: SHORT (Venta)")
    else:
        st.info("⚪ Tipo de Operación: NONE / Mantener")

    price_valid = True
    if plan['direction'] == "LONG":
        if not (plan['sl'] < plan['entry_price'] < plan['tp1'] < plan['tp2'] < plan['tp3']):
            price_valid = False
            st.error("❌ ERROR CRÍTICO: Niveles de precio ilógicos para LONG")
    elif plan['direction'] == "SHORT":
        if not (plan['tp3'] < plan['tp2'] < plan['tp1'] < plan['entry_price'] < plan['sl']):
            price_valid = False
            st.error("❌ ERROR CRÍTICO: Niveles de precio ilógicos para SHORT")
    if price_valid:
        st.success("✅ Estructura de precios lógica y consistente")

    tab1, tab2, tab3 = st.tabs(["🎯 NIVELES DE PRECIO", "📊 DETALLES FINANCIEROS", "⚡ GESTIÓN DE RIESGO"])
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**💰 Precio Actual**\n# {fmt(current_price,8)}")
            diff_pct = ((plan['entry_price'] - current_price) / current_price) * 100 if current_price>0 else 0
            if plan['direction'] == "LONG":
                if diff_pct < 0:
                    st.success(f"**🎯 Entrada Optimizada:** {abs(diff_pct):.2f}% bajo el mercado")
                else:
                    st.warning(f"**⚠️ Sobreprecio:** {diff_pct:.2f}% sobre el mercado")
            elif plan['direction'] == "SHORT":
                if diff_pct > 0:
                    st.success(f"**🎯 Entrada Optimizada:** {diff_pct:.2f}% sobre el mercado")
                else:
                    st.warning(f"**⚠️ Infraprecio:** {abs(diff_pct):.2f}% bajo el mercado")
            st.info(f"**📈 Precio de Entrada**\n# {fmt(plan['entry_price'],8)}")
            st.info(f"**🎯 Zona de Entrada:** {fmt(plan['entry_zone'][0],8)} - {fmt(plan['entry_zone'][1],8)}")
        with col2:
            if plan['direction'] == "LONG":
                st.error(f"**🛡️ Stop Loss (Vender para cerrar LONG)**\n# {fmt(plan['sl'],8)}")
                st.success(f"**🎯 Take Profit 1 (Vender)**\n# {fmt(plan['tp1'],8)}")
                st.success(f"**🎯 Take Profit 2 (Vender)**\n# {fmt(plan['tp2'],8)}")
                st.success(f"**🎯 Take Profit 3 (Vender)**\n# {fmt(plan['tp3'],8)}")
            elif plan['direction'] == "SHORT":
                st.error(f"**🛡️ Stop Loss (Comprar para cerrar SHORT)**\n# {fmt(plan['sl'],8)}")
                st.success(f"**🎯 Take Profit 1 (Comprar)**\n# {fmt(plan['tp1'],8)}")
                st.success(f"**🎯 Take Profit 2 (Comprar)**\n# {fmt(plan['tp2'],8)}")
                st.success(f"**🎯 Take Profit 3 (Comprar)**\n# {fmt(plan['tp3'],8)}")
            sl_distance = abs(plan['entry_price'] - plan['sl']) / plan['entry_price'] * 100 if plan['entry_price']>0 else 0
            tp1_distance = abs(plan['tp1'] - plan['entry_price']) / plan['entry_price'] * 100 if plan['entry_price']>0 else 0
            st.info(f"**📏 Distancia SL:** {sl_distance:.2f}%")
            st.info(f"**📐 Distancia TP1:** {tp1_distance:.2f}%")
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("💳 Capital", f"{capital:.2f} USDT")
            st.metric("⚡ Apalancamiento", f"{leverage}x")
            st.metric("💰 Margen Usado", f"{plan['margin_used']:.2f} USDT")
            st.metric("💼 Tamaño Posición", f"{plan['position_notional']:.2f} USDT")
        with col2:
            st.metric("📦 Cantidad", f"{plan['amount_base']:.8f}")
            st.metric("🏦 Balance Restante", f"{capital - plan['margin_used']:.2f} USDT")
            st.metric("🎯 Riesgo/Operación", f"{RISK_PER_TRADE*100:.1f}%")
            st.metric("📈 Exposición Total", f"{plan['position_notional']:.2f} USDT")
    with tab3:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.error("**🔴 Máxima Pérdida**")
            st.metric("Pérdida", f"{plan['potential_loss']:.2f} USDT")
            st.write(f"{(plan['potential_loss']/capital)*100:.2f}% del capital")
        with col2:
            st.success("**🟢 Ratios R/B**")
            st.metric("TP1", f"1:{plan['risk_reward_tp1']:.2f}")
            st.metric("TP2", f"1:{plan['risk_reward_tp2']:.2f}")
            st.metric("TP3", f"1:{plan['risk_reward_tp3']:.2f}")
        with col3:
            st.success("**📈 Beneficios**")
            st.metric("TP1", f"{plan['potential_profit_tp1']:.2f} USDT")
            st.metric("TP2", f"{plan['potential_profit_tp2']:.2f} USDT")
            st.metric("TP3", f"{plan['potential_profit_tp3']:.2f} USDT")

# -------------------------
# PANEL EJECUCIÓN
# -------------------------
def display_execution_panel(plan, symbol, current_price, phase, confidence, df):
    st.markdown("---")
    st.subheader("🚀 EJECUCIÓN DE ÓRDENES")
    if LIVE_MODE:
        st.warning("**🔴 TRADING EN VIVO ACTIVADO**")
        if not API_KEY or not API_SECRET:
            st.error("⚠️ API Key/Secret NO cargadas. No se puede ejecutar operaciones reales.")
        if plan['direction'] in ['LONG', 'SHORT']:
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"🎯 EJECUTAR {plan['direction']}", type="primary", use_container_width=True):
                    with st.expander("🔔 CONFIRMACIÓN FINAL", expanded=True):
                        st.error("⚠️ REVISE DETALLES ANTES DE CONFIRMAR")
                        st.write("**Especificaciones:**")
                        st.write(f"• Símbolo: {symbol}")
                        st.write(f"• Tipo: {plan['direction']} - Acción API: {plan['side']}")
                        st.write(f"• Entrada: {fmt(plan['entry_price'],8)}")
                        st.write(f"• Cantidad: {plan['amount_base']:.8f}")
                        st.write(f"• Monto USDT usado: {fmt(plan.get('monto_usdt', MONTO_USDT),8)}")
                        st.write("**Niveles:**")
                        st.write(f"• SL: {fmt(plan['sl'],8)}")
                        st.write(f"• TP1: {fmt(plan['tp1'],8)}")
                        st.write(f"• TP2: {fmt(plan['tp2'],8)}")
                        st.write(f"• TP3: {fmt(plan['tp3'],8)}")
                        if st.button("✅ CONFIRMAR EJECUCIÓN", type="primary"):
                            try:
                                with st.spinner("Ejecutando órdenes..."):
                                    api_action = plan['side']  # BUY or SELL
                                    # set leverage
                                    ok, method = set_leverage_for_symbol(exchange, symbol, LEVERAGE)
                                    if not ok:
                                        st.warning(f"No se pudo establecer apalancamiento automáticamente (método: {method}).")
                                    order_result = place_orders_binance_futures(
                                        exchange, symbol, api_action,
                                        plan['amount_base'], plan['entry_price'], plan['sl'],
                                        [plan['tp1'], plan['tp2'], plan['tp3']]
                                    )
                                st.json(order_result)
                                # registrar (estado OPEN o errors)
                                estado = "OPEN" if (order_result.get('open') or order_result.get('open_info')) else "ERROR"
                                register_order_in_table(symbol, plan, api_response=order_result, estado=estado)
                                if order_result.get('errors'):
                                    st.error("Algunos errores ocurrieron al colocar TP/SL (revisar API response).")
                                else:
                                    st.success("Órdenes colocadas (entrada y SL/TP donde fue posible).")
                            except Exception as e:
                                st.error(f"Error: {e}")
                                register_order_in_table(symbol, plan, api_response=str(e), estado="ERROR")
            with col2:
                if st.button("🔄 ACTUALIZAR SEÑAL", type="secondary", use_container_width=True):
                    st.rerun()
    else:
        st.info("**🔒 MODO SIMULACIÓN**")
        if st.button("📊 SIMULAR EJECUCIÓN", use_container_width=True):
            st.success("Simulación completada - Órdenes (simuladas) registradas")
            register_order_in_table(symbol, plan, api_response="SIMULACIÓN", estado="SIMULATED")

    # Tabla de órdenes
    st.markdown("---")
    st.subheader("📋 Órdenes ejecutadas / intentadas")
    if 'orders_table' not in st.session_state or st.session_state['orders_table'].empty:
        st.info("Aún no hay órdenes registradas")
    else:
        st.dataframe(st.session_state['orders_table'])

# -------------------------
# BUCLE PRINCIPAL
# -------------------------
status_box = st.empty()
if 'live_running' not in st.session_state:
    st.session_state.live_running = True
if 'update_count' not in st.session_state:
    st.session_state.update_count = 0

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("▶️ INICIAR MONITOREO", type="primary"):
        st.session_state.live_running = True
with col2:
    if st.button("⏸️ PAUSAR"):
        st.session_state.live_running = False
with col3:
    if st.button("🔄 ACTUALIZAR", key="manual_refresh"):
        st.session_state.update_count += 1
        st.rerun()

def run_iteration():
    df = fetch_ohlcv(exchange, SYMBOL, TF, LIMIT)
    if df is None or df.empty:
        status_box.error("No hay datos disponibles")
        return
    df['ema20'] = ema(df['close'], 20)
    df['ema50'] = ema(df['close'], 50)
    df['ema200'] = ema(df['close'], 200)
    df['rsi14'] = rsi_series(df['close'].values, 14)
    supports, resistances = detect_sr(df, order=5)
    phase, desc, action, spring_price, ut_price, confidence = determine_phase_and_recommendation(df, supports, resistances, VOL_MULT)
    current_price = float(df['close'].iloc[-1])

    # override action if user forced LONG/SHORT
    if tipo_operacion_forzada == "LONG":
        action = "COMPRAR"
    elif tipo_operacion_forzada == "SHORT":
        action = "VENDER"

    plan = build_trade_plan(df, supports, resistances, CAPITAL, LEVERAGE, RISK_PER_TRADE, action, monto_usdt=MONTO_USDT)
    current_volume = df['volume'].iloc[-1]
    vol_avg = df['volume'].rolling(20).mean().iloc[-1]
    volume_status = "ALTO" if current_volume > vol_avg * VOL_MULT else "NORMAL"

    # Gráfico principal
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'],
                                low=df['low'], close=df['close'], name="Precio"))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema20'], name='EMA20', line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema50'], name='EMA50', line=dict(width=1)))
    if plan['direction'] in ['LONG','SHORT']:
        fig.add_hline(y=plan['entry_price'], line=dict(width=3), annotation_text="ENTRADA")
        fig.add_hline(y=plan['sl'], line=dict(width=2), annotation_text="SL")
        fig.add_hline(y=plan['tp1'], line=dict(width=1), annotation_text="TP1")
        fig.add_hline(y=plan['tp2'], line=dict(width=1), annotation_text="TP2")
        fig.add_hline(y=plan['tp3'], line=dict(width=1), annotation_text="TP3")
    fig.update_layout(height=600, showlegend=True, xaxis_rangeslider_visible=False, title=f"Análisis - {SYMBOL} - {TF}")
    st.plotly_chart(fig, use_container_width=True)

    display_trading_signal(phase, desc, action, confidence, plan)
    display_trade_plan(plan, CAPITAL, LEVERAGE, SYMBOL, current_price)
    display_execution_panel(plan, SYMBOL, current_price, phase, confidence, df)

    # Auto trade logic: multi-timeframe allowed
    if AUTO_TRADE and LIVE_MODE and API_KEY and API_SECRET and plan['direction'] in ['LONG','SHORT']:
        if confidence >= AUTO_CONFIDENCE_THRESH:
            status_box.info(f"🔔 AUTO_TRADE: Señal detectada con confianza {confidence:.2f} - intentando ejecutar...")
            try:
                ok, method = set_leverage_for_symbol(exchange, SYMBOL, LEVERAGE)
                if not ok:
                    status_box.warning(f"No se pudo establecer apalancamiento automáticamente (método: {method}).")
                api_action = plan['side']  # BUY or SELL
                tp_list = [plan['tp1'], plan['tp2'], plan['tp3']]
                order_result = place_orders_binance_futures(
                    exchange, SYMBOL, api_action,
                    plan['amount_base'], plan['entry_price'], plan['sl'], tp_list
                )
                st.json(order_result)
                estado = "OPEN" if (order_result.get('open') or order_result.get('open_info')) else "ERROR"
                register_order_in_table(SYMBOL, plan, api_response=order_result, estado=estado)
                if order_result.get('errors'):
                    status_box.error("AUTO_TRADE: Hubo errores al crear órdenes (revisar API response).")
            except Exception as e:
                status_box.error(f"AUTO_TRADE - Error: {e}")
                register_order_in_table(SYMBOL, plan, api_response=str(e), estado="ERROR")
        else:
            status_box.info(f"AUTO_TRADE: confianza {confidence:.2f} < umbral {AUTO_CONFIDENCE_THRESH}")

    status_box.info(f"🔄 Actualización #{st.session_state.update_count} | {now_ts()} | Próxima en {POLL_SECONDS}s")


# Ejecución inicial
run_iteration()

# Bucle automático
if st.session_state.live_running:
    time.sleep(POLL_SECONDS)
    st.session_state.update_count += 1
    st.rerun()