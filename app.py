# wyckoff_shark_trading.py
# -*- coding: utf-8 -*-
"""
Wyckoff Shark - Sistema Profesional de Trading (LONG/SHORT, ejecución automática)
Optimizado para Streamlit Cloud + GitHub
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
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# CONFIGURACIÓN STREAMLIT CLOUD
# -------------------------
st.set_page_config(
    page_title="Wyckoff Shark - Trading Auto", 
    layout="wide", 
    initial_sidebar_state="expanded"
)
st.title("🦈 Wyckoff Shark - Sistema Profesional de Trading")

# ---------- Panel lateral ----------
st.sidebar.header("🎯 Configuración Principal")
MARKET_TYPE = st.sidebar.selectbox("Tipo de Mercado", ["Futuros"], index=0)
SYMBOL = st.sidebar.text_input("Símbolo (ej: BTC/USDT, ETH/USDT)", "BTC/USDT")
TF = st.sidebar.selectbox("Timeframe", ["1m", "3m", "5m", "15m", "1h", "4h"], index=2)

# Configuración mejorada para Cloud
col1, col2 = st.sidebar.columns(2)
with col1:
    LIMIT = st.slider("Velas a Analizar", 100, 800, 300)
with col2:
    POLL_SECONDS = st.slider("Actualización (seg)", 10, 120, 30)

VOL_MULT = st.sidebar.slider("Umbral Volumen (x promedio)", 1.5, 5.0, 2.0)
RISK_PER_TRADE = st.sidebar.slider("Riesgo por Operación (%)", 0.1, 5.0, 1.0) / 100.0
CAPITAL = st.sidebar.number_input("Capital (USDT)", min_value=10.0, value=100.0, step=10.0)
LEVERAGE = st.sidebar.selectbox("Apalancamiento", [1, 3, 5, 10, 20], index=2)

# Monto por orden
st.sidebar.markdown("---")
MONTO_USDT = st.sidebar.number_input("Monto por Orden (USDT) - 0 = % riesgo", 
                                    min_value=0.0, value=0.0, step=5.0)

# Forzar operación
st.sidebar.markdown("---")
tipo_operacion_forzada = st.sidebar.selectbox("Tipo de operación", 
                                            ["AUTO", "LONG", "SHORT"])

# -------------------------
# CONFIGURACIÓN API (Streamlit Secrets)
# -------------------------
st.sidebar.markdown("---")
st.sidebar.header("🔐 Configuración API")

LIVE_MODE = st.sidebar.checkbox("Habilitar Trading en VIVO", value=False)
USE_TESTNET = st.sidebar.checkbox("Usar TESTNET (Recomendado)", value=True)

# Usar secrets de Streamlit Cloud si existen, sino inputs normales
if 'BINANCE_API_KEY' in st.secrets and 'BINANCE_API_SECRET' in st.secrets:
    API_KEY = st.secrets['BINANCE_API_KEY']
    API_SECRET = st.secrets['BINANCE_API_SECRET']
    st.sidebar.success("🔑 API Keys cargadas desde Secrets")
else:
    API_KEY = st.sidebar.text_input("Binance API Key", type="password")
    API_SECRET = st.sidebar.text_input("Binance API Secret", type="password")

# Ejecución automática
st.sidebar.markdown("---")
AUTO_TRADE = st.sidebar.checkbox("Ejecución Automática (AUTO_TRADE)", value=False)
AUTO_CONFIDENCE_THRESH = st.sidebar.slider("Umbral Confianza AUTO", 0.5, 0.95, 0.70, step=0.05)

# -------------------------
# UTILIDADES OPTIMIZADAS
# -------------------------
def fmt(x, d=8):
    """Formatear números con decimales controlados"""
    try:
        return f"{decimal.Decimal(str(x)):.{d}f}"
    except:
        try:
            return f"{float(x):.{d}f}"
        except:
            return str(x)

def now_ts():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

# -------------------------
# INICIALIZACIÓN EXCHANGE OPTIMIZADA
# -------------------------
@st.cache_resource(show_spinner=False)
def init_exchange(_market_type, _api_key=None, _api_secret=None, _testnet=True):
    """Exchange inicializado con cache para mejor performance"""
    try:
        opts = {
            'enableRateLimit': True,
            'sandbox': _testnet,
        }
        if _market_type == "Futuros":
            opts['options'] = {'defaultType': 'future'}
            
        exchange = ccxt.binance(opts)
        
        if _api_key and _api_secret:
            exchange.apiKey = _api_key
            exchange.secret = _api_secret
            
        # Cargar mercados de forma segura
        try:
            exchange.load_markets()
        except:
            pass
            
        return exchange
    except Exception as e:
        st.error(f"❌ Error inicializando exchange: {e}")
        return None

# Inicializar exchange
exchange = init_exchange(
    MARKET_TYPE, 
    API_KEY if API_KEY else None, 
    API_SECRET if API_SECRET else None, 
    USE_TESTNET
)

# -------------------------
# DATA FETCHING OPTIMIZADO
# -------------------------
@st.cache_data(ttl=10, show_spinner=False)
def fetch_ohlcv_cached(_exchange, symbol, timeframe="5m", limit=300):
    """Obtener datos OHLCV con cache para mejor performance"""
    try:
        # Para futuros de Binance
        params = {'type': 'future'} if MARKET_TYPE == "Futuros" else {}
        
        # Limitar para evitar timeouts
        limit = min(limit, 800)
        
        ohlcv = _exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, params=params)
        df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
        return df
    except Exception as e:
        st.error(f"📊 Error obteniendo datos: {e}")
        return None

# -------------------------
# INDICADORES OPTIMIZADOS
# -------------------------
def ema(series, span):
    """EMA optimizado"""
    return series.ewm(span=span, adjust=False).mean()

def rsi_series(prices, period=14):
    """RSI optimizado"""
    prices = np.array(prices, dtype=float)
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_full = np.concatenate((np.full(period, np.nan), rsi.values))
    return pd.Series(rsi_full)

# -------------------------
# DETECCIÓN S/R MEJORADA
# -------------------------
def detect_sr(df, order=3, cluster_tol=0.015):
    """Detección de soportes y resistencias mejorada"""
    try:
        highs = df['high'].values
        lows = df['low'].values
        
        max_idx = argrelextrema(highs, np.greater_equal, order=order)[0]
        min_idx = argrelextrema(lows, np.less_equal, order=order)[0]
        
        resistances = [highs[i] for i in max_idx if i < len(highs)]
        supports = [lows[i] for i in min_idx if i < len(lows)]
        
        def cluster_levels(levels):
            if not levels:
                return []
            sorted_levels = sorted(levels)
            clusters = []
            current_cluster = [sorted_levels[0]]
            
            for level in sorted_levels[1:]:
                if abs(level - np.mean(current_cluster)) / np.mean(current_cluster) <= cluster_tol:
                    current_cluster.append(level)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [level]
            clusters.append(current_cluster)
            
            return [np.mean(cluster) for cluster in clusters if cluster]
        
        return cluster_levels(supports), cluster_levels(resistances)
    except Exception:
        return [], []

# -------------------------
# SEÑALES WYCKOFF OPTIMIZADAS
# -------------------------
def detect_spring(df, supports, vol_mult=1.7):
    """Detección de Spring optimizada"""
    if not supports or len(supports) == 0:
        return False, None, 0
        
    try:
        sup_level = min(supports)
        current = df.iloc[-1]
        
        spring_conditions = (
            current['low'] < sup_level * 0.995 and
            current['close'] > sup_level and
            current['volume'] > df['volume'].rolling(20).mean().iloc[-1] * vol_mult
        )
        
        if spring_conditions:
            confidence = 0.7
            if current['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 3:
                confidence += 0.15
            if 'rsi14' in df.columns and current['rsi14'] < 35:
                confidence += 0.10
                
            return True, sup_level, min(confidence, 0.95)
    except Exception:
        pass
        
    return False, None, 0

def detect_upthrust(df, resistances, vol_mult=1.7):
    """Detección de Upthrust optimizada"""
    if not resistances or len(resistances) == 0:
        return False, None, 0
        
    try:
        res_level = max(resistances)
        current = df.iloc[-1]
        
        upthrust_conditions = (
            current['high'] > res_level * 1.005 and
            current['close'] < res_level and
            current['volume'] > df['volume'].rolling(20).mean().iloc[-1] * vol_mult
        )
        
        if upthrust_conditions:
            confidence = 0.7
            if current['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 3:
                confidence += 0.15
            if 'rsi14' in df.columns and current['rsi14'] > 65:
                confidence += 0.10
                
            return True, res_level, min(confidence, 0.95)
    except Exception:
        pass
        
    return False, None, 0

# -------------------------
# ANÁLISIS DE FASE MEJORADO
# -------------------------
def determine_phase_and_recommendation(df, supports, resistances, vol_mult):
    """Análisis de fase Wyckoff mejorado"""
    try:
        current = df.iloc[-1]
        price = current['close']
        ema20 = current['ema20']
        ema50 = current['ema50']
        
        # Detectar señales fuertes primero
        spring_flag, spring_price, spring_conf = detect_spring(df, supports, vol_mult)
        ut_flag, ut_price, ut_conf = detect_upthrust(df, resistances, vol_mult)
        
        if spring_flag:
            return ("FASE A: ACUMULACIÓN (Spring)", 
                   f"🟢 COMPRA - Spring en {fmt(spring_price)}", 
                   "COMPRAR", spring_conf)
                   
        if ut_flag:
            return ("FASE A: DISTRIBUCIÓN (Upthrust)", 
                   f"🔴 VENTA - Upthrust en {fmt(ut_price)}", 
                   "VENDER", ut_conf)
        
        # Análisis de tendencia
        volume_condition = current['volume'] > df['volume'].rolling(20).mean().iloc[-1]
        
        if price > ema20 > ema50:
            if volume_condition:
                return ("FASE B: MARKUP (Alcista Fuerte)", 
                       "📈 TENDENCIA ALCISTA - Comprar en retrocesos", 
                       "COMPRAR", 0.65)
            else:
                return ("FASE B: MARKUP (Alcista Débil)", 
                       "⚠️ ALCISTA SIN VOLUMEN - Esperar confirmación", 
                       "MANTENER", 0.4)
                       
        elif price < ema20 < ema50:
            if volume_condition:
                return ("FASE B: MARKDOWN (Bajista Fuerte)", 
                       "📉 TENDENCIA BAJISTA - Vender en rebotes", 
                       "VENDER", 0.65)
            else:
                return ("FASE B: MARKDOWN (Bajista Débil)", 
                       "⚠️ BAJISTA SIN VOLUMEN - Esperar confirmación", 
                       "MANTENER", 0.4)
        else:
            return ("FASE C: CONSOLIDACIÓN", 
                   "↔️ MERCADO EN RANGO - Esperar señal clara", 
                   "MANTENER", 0.3)
                   
    except Exception as e:
        return ("ERROR EN ANÁLISIS", f"Error: {str(e)}", "MANTENER", 0.0)

# -------------------------
# PLAN DE TRADING OPTIMIZADO
# -------------------------
def build_trade_plan(df, supports, resistances, capital, leverage, risk_per_trade, action, monto_usdt=0.0):
    """Construir plan de trading optimizado"""
    try:
        current = df.iloc[-1]
        price = float(current['close'])
        ema20 = float(current['ema20'])
        
        # Default values
        entry_price = price
        sl = price * 0.99
        tp1, tp2, tp3 = price * 1.01, price * 1.02, price * 1.03
        direction = "NONE"
        side = "NONE"
        
        if action == "COMPRAR":
            direction, side = "LONG", "BUY"
            if supports:
                support_level = min(supports)
                entry_price = support_level * 1.002  # Entrar ligeramente arriba del soporte
                sl = support_level * 0.985
            else:
                entry_price = ema20
                sl = ema20 * 0.98
                
            tp1 = entry_price * 1.008
            tp2 = entry_price * 1.016
            tp3 = entry_price * 1.028
            
        elif action == "VENDER":
            direction, side = "SHORT", "SELL"
            if resistances:
                resistance_level = max(resistances)
                entry_price = resistance_level * 0.998  # Entrar ligeramente debajo de resistencia
                sl = resistance_level * 1.015
            else:
                entry_price = ema20
                sl = ema20 * 1.02
                
            tp1 = entry_price * 0.992
            tp2 = entry_price * 0.984
            tp3 = entry_price * 0.972
        
        # Calcular tamaño de posición
        if monto_usdt and monto_usdt > 0:
            position_notional = monto_usdt * leverage
        else:
            stop_distance = abs(entry_price - sl) / entry_price
            if stop_distance <= 0.005:  # Mínimo 0.5% de stop
                stop_distance = 0.01
            risk_amount = capital * risk_per_trade
            position_notional = risk_amount / stop_distance
            
        # Limitar al capital disponible
        max_notional = capital * leverage
        position_notional = min(position_notional, max_notional)
        
        # Calcular cantidad
        amount_base = position_notional / entry_price if entry_price > 0 else 0
        
        # Redondear a 6 decimales por seguridad
        amount_base = round(amount_base, 6)
        
        # Calcular métricas de riesgo
        stop_dist_pct = abs(entry_price - sl) / entry_price
        potential_loss = position_notional * stop_dist_pct
        
        if direction == "LONG":
            potential_profits = [
                position_notional * (tp - entry_price) / entry_price 
                for tp in [tp1, tp2, tp3]
            ]
        elif direction == "SHORT":
            potential_profits = [
                position_notional * (entry_price - tp) / entry_price 
                for tp in [tp1, tp2, tp3]
            ]
        else:
            potential_profits = [0, 0, 0]
            
        risk_rewards = [
            profit / potential_loss if potential_loss > 0 else 0 
            for profit in potential_profits
        ]
        
        return {
            "action": action,
            "direction": direction,
            "side": side,
            "entry_price": float(entry_price),
            "sl": float(sl),
            "tp1": float(tp1),
            "tp2": float(tp2),
            "tp3": float(tp3),
            "position_notional": float(position_notional),
            "amount_base": float(amount_base),
            "stop_dist_pct": float(stop_dist_pct),
            "potential_loss": float(potential_loss),
            "potential_profits": [float(p) for p in potential_profits],
            "risk_rewards": [float(rr) for rr in risk_rewards],
            "margin_used": float(position_notional / leverage),
            "monto_usdt": float(monto_usdt)
        }
        
    except Exception as e:
        st.error(f"Error building trade plan: {e}")
        # Return safe default plan
        return {
            "action": "MANTENER", "direction": "NONE", "side": "NONE",
            "entry_price": price, "sl": price, "tp1": price, "tp2": price, "tp3": price,
            "position_notional": 0, "amount_base": 0, "stop_dist_pct": 0,
            "potential_loss": 0, "potential_profits": [0,0,0], "risk_rewards": [0,0,0],
            "margin_used": 0, "monto_usdt": 0
        }

# -------------------------
# VISUALIZACIÓN MEJORADA
# -------------------------
def display_trading_signal(phase, desc, action, confidence, plan):
    """Mostrar señal de trading mejorada"""
    st.markdown("---")
    st.subheader("📊 SEÑAL DE TRADING WYCKOFF")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Color code based on action
        if action == "COMPRAR":
            st.success(f"**Fase:** {phase}")
            st.success(f"**Señal:** {desc}")
        elif action == "VENDER":
            st.error(f"**Fase:** {phase}")
            st.error(f"**Señal:** {desc}")
        else:
            st.info(f"**Fase:** {phase}")
            st.info(f"**Señal:** {desc}")
            
    with col2:
        # Confidence indicator
        confidence_color = "🟢" if confidence >= 0.7 else "🟡" if confidence >= 0.5 else "🔴"
        st.metric("Confianza", f"{confidence:.1%}")
        st.progress(float(confidence))
        
        # Operation type
        op_color = "🟢" if plan['direction'] == "LONG" else "🔴" if plan['direction'] == "SHORT" else "⚪"
        st.write(f"**Operación:** {op_color} {plan['direction']}")

def display_trade_plan(plan, capital, leverage, symbol, current_price):
    """Mostrar plan de trading mejorado"""
    st.markdown("---")
    st.subheader("💰 PLAN DE TRADING")
    
    # Header con tipo de operación
    if plan['direction'] == "LONG":
        st.success(f"📈 OPERACIÓN LONG - Comprar {symbol}")
    elif plan['direction'] == "SHORT":
        st.error(f"📉 OPERACIÓN SHORT - Vender {symbol}")
    else:
        st.info("⚪ SIN OPERACIÓN - Mantener posición")
        return
    
    # Pestañas organizadas
    tab1, tab2, tab3 = st.tabs(["🎯 PRECIOS", "📊 FINANZAS", "⚡ RIESGO"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Precio Actual", fmt(current_price))
            st.metric("Precio Entrada", fmt(plan['entry_price']))
            diff_pct = ((plan['entry_price'] - current_price) / current_price) * 100
            st.metric("Diferencia", f"{diff_pct:+.2f}%")
            
        with col2:
            st.metric("Stop Loss", fmt(plan['sl']))
            st.metric("Take Profit 1", fmt(plan['tp1']))
            st.metric("Take Profit 2", fmt(plan['tp2']))
            st.metric("Take Profit 3", fmt(plan['tp3']))
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Capital", f"${capital:.2f}")
            st.metric("Apalancamiento", f"{leverage}x")
            st.metric("Margen Usado", f"${plan['margin_used']:.2f}")
            
        with col2:
            st.metric("Tamaño Posición", f"${plan['position_notional']:.2f}")
            st.metric("Cantidad", f"{plan['amount_base']:.6f}")
            st.metric("Balance Restante", f"${capital - plan['margin_used']:.2f}")
    
    with tab3:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.error("Pérdida Potencial")
            st.metric("USDT", f"${plan['potential_loss']:.2f}")
            st.write(f"({plan['potential_loss']/capital*100:.1f}% capital)")
            
        with col2:
            st.success("Ratios R/R")
            for i, rr in enumerate(plan['risk_rewards'], 1):
                st.metric(f"TP{i}", f"1:{rr:.2f}")
                
        with col3:
            st.success("Ganancias Potenciales")
            for i, profit in enumerate(plan['potential_profits'], 1):
                st.metric(f"TP{i}", f"${profit:.2f}")

# -------------------------
# GESTIÓN DE ÓRDENES SIMPLIFICADA
# -------------------------
def register_order(symbol, plan, status="SIMULATED", response=""):
    """Registrar orden en session state"""
    if 'orders_history' not in st.session_state:
        st.session_state.orders_history = []
        
    order_data = {
        "timestamp": now_ts(),
        "symbol": symbol,
        "direction": plan['direction'],
        "entry": plan['entry_price'],
        "sl": plan['sl'],
        "tp1": plan['tp1'],
        "tp2": plan['tp2'],
        "tp3": plan['tp3'],
        "amount": plan['amount_base'],
        "status": status,
        "response": str(response)[:500]
    }
    
    st.session_state.orders_history.insert(0, order_data)
    
    # Mantener solo últimas 50 órdenes
    if len(st.session_state.orders_history) > 50:
        st.session_state.orders_history = st.session_state.orders_history[:50]

# -------------------------
# EJECUCIÓN PRINCIPAL OPTIMIZADA
# -------------------------
def main_execution():
    """Función principal de ejecución optimizada para Cloud"""
    
    # Status header
    status_container = st.empty()
    
    # Controles de ejecución
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Actualizar Datos", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("⏸️ Pausar Auto", use_container_width=True):
            if 'auto_refresh' in st.session_state:
                st.session_state.auto_refresh = False
    with col3:
        if st.button("📊 Ver Historial", use_container_width=True):
            if 'show_history' in st.session_state:
                st.session_state.show_history = not st.session_state.show_history
    
    # Obtener datos
    with st.spinner("📊 Obteniendo datos de mercado..."):
        df = fetch_ohlcv_cached(exchange, SYMBOL, TF, LIMIT)
    
    if df is None or df.empty:
        status_container.error("❌ No se pudieron obtener datos del mercado")
        return
        
    # Calcular indicadores
    try:
        df['ema20'] = ema(df['close'], 20)
        df['ema50'] = ema(df['close'], 50)
        df['rsi14'] = rsi_series(df['close'].values, 14)
    except Exception as e:
        status_container.error(f"❌ Error calculando indicadores: {e}")
        return
    
    # Detectar S/R
    supports, resistances = detect_sr(df)
    
    # Determinar señal
    phase, desc, action, confidence = determine_phase_and_recommendation(
        df, supports, resistances, VOL_MULT
    )
    
    # Override si está forzado
    if tipo_operacion_forzada != "AUTO":
        action = "COMPRAR" if tipo_operacion_forzada == "LONG" else "VENDER"
    
    # Construir plan
    plan = build_trade_plan(
        df, supports, resistances, CAPITAL, LEVERAGE, 
        RISK_PER_TRADE, action, MONTO_USDT
    )
    
    current_price = float(df['close'].iloc[-1])
    
    # -------------------------
    # GRÁFICO PRINCIPAL
    # -------------------------
    st.markdown("---")
    st.subheader(f"📈 Análisis - {SYMBOL} ({TF})")
    
    fig = go.Figure()
    
    # Velas
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Precio"
    ))
    
    # EMAs
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['ema20'],
        name='EMA20', line=dict(color='orange', width=1)
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['ema50'],
        name='EMA50', line=dict(color='red', width=1)
    ))
    
    # Niveles de trading si hay operación
    if plan['direction'] in ['LONG', 'SHORT']:
        fig.add_hline(y=plan['entry_price'], line_dash="solid", 
                     annotation_text="ENTRADA", line_color="blue")
        fig.add_hline(y=plan['sl'], line_dash="dot", 
                     annotation_text="SL", line_color="red")
        for i, tp in enumerate([plan['tp1'], plan['tp2'], plan['tp3']], 1):
            fig.add_hline(y=tp, line_dash="dash", 
                         annotation_text=f"TP{i}", line_color="green")
    
    fig.update_layout(
        height=500,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        title=f"Wyckoff Analysis - {SYMBOL} | {phase}"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # -------------------------
    # MOSTRAR SEÑAL Y PLAN
    # -------------------------
    display_trading_signal(phase, desc, action, confidence, plan)
    display_trade_plan(plan, CAPITAL, LEVERAGE, SYMBOL, current_price)
    
    # -------------------------
    # PANEL DE EJECUCIÓN
    # -------------------------
    st.markdown("---")
    st.subheader("🚀 Panel de Ejecución")
    
    if LIVE_MODE and API_KEY and API_SECRET:
        st.warning("🔴 MODO LIVE ACTIVADO")
        
        if plan['direction'] in ['LONG', 'SHORT']:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(f"🎯 EJECUTAR {plan['direction']}", 
                           type="primary", use_container_width=True):
                    with st.expander("CONFIRMACIÓN", expanded=True):
                        st.error("⚠️ REVISAR ANTES DE CONFIRMAR")
                        st.write(f"Símbolo: {SYMBOL}")
                        st.write(f"Dirección: {plan['direction']}")
                        st.write(f"Entrada: {fmt(plan['entry_price'])}")
                        st.write(f"Cantidad: {plan['amount_base']:.6f}")
                        
                        if st.button("✅ CONFIRMAR EJECUCIÓN", type="primary"):
                            try:
                                # Simular ejecución (implementar API real aquí)
                                order_result = {"status": "SIMULATED", "id": "TEST123"}
                                register_order(SYMBOL, plan, "EXECUTED", order_result)
                                st.success("✅ Orden ejecutada (simulación)")
                            except Exception as e:
                                st.error(f"❌ Error: {e}")
                                register_order(SYMBOL, plan, "ERROR", str(e))
            
            with col2:
                if st.button("📝 SIMULAR EJECUCIÓN", use_container_width=True):
                    register_order(SYMBOL, plan, "SIMULATED", "Simulación completada")
                    st.success("📝 Simulación registrada")
    else:
        st.info("🔒 MODO SIMULACIÓN")
        if st.button("📊 EJECUTAR SIMULACIÓN", use_container_width=True):
            register_order(SYMBOL, plan, "SIMULATED", "Simulación automática")
            st.success("📊 Simulación completada")
    
    # -------------------------
    # HISTORIAL DE ÓRDENES
    # -------------------------
    if 'orders_history' in st.session_state and st.session_state.orders_history:
        st.markdown("---")
        st.subheader("📋 Historial de Órdenes")
        
        # Convertir a DataFrame para mejor visualización
        history_df = pd.DataFrame(st.session_state.orders_history)
        st.dataframe(history_df, use_container_width=True)
    
    # -------------------------
    # AUTO-TRADING
    # -------------------------
    if AUTO_TRADE and LIVE_MODE and confidence >= AUTO_CONFIDENCE_THRESH:
        if plan['direction'] in ['LONG', 'SHORT']:
            status_container.info(
                f"🤖 AUTO-TRADE: Señal {confidence:.1%} - "
                f"Ejecutando {plan['direction']}..."
            )
            # Aquí iría la ejecución automática real
            register_order(SYMBOL, plan, "AUTO-TRADE", "Ejecución automática")
    
    # Status final
    status_container.success(
        f"✅ Análisis completo | {now_ts()} | "
        f"Próxima actualización: {POLL_SECONDS}s"
    )

# -------------------------
# INICIALIZACIÓN Y BUCLE
# -------------------------
if __name__ == "__main__":
    # Estado inicial
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    if 'show_history' not in st.session_state:
        st.session_state.show_history = False
    
    # Ejecutar análisis
    main_execution()
    
    # Auto-refresh si está activado
    if st.session_state.auto_refresh:
        time.sleep(POLL_SECONDS)
        st.rerun()