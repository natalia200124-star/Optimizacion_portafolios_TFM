import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime

# =========================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS
# =========================
st.set_page_config(
    page_title="Portfolio Optimizer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS PERSONALIZADO PROFESIONAL
st.markdown("""
<style>
    /* Importar fuente moderna */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Variables de color */
    :root {
        --primary-color: #1E88E5;
        --secondary-color: #00ACC1;
        --accent-color: #7C4DFF;
        --success-color: #00C853;
        --warning-color: #FFB300;
        --danger-color: #FF5252;
        --dark-bg: #0E1117;
        --card-bg: #1E2128;
        --text-primary: #FFFFFF;
        --text-secondary: #B4B4B4;
    }
    
    /* Fondo general */
    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #1A1D29 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Título principal */
    h1 {
        background: linear-gradient(120deg, #1E88E5 0%, #00ACC1 50%, #7C4DFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        font-size: 3.5rem !important;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem 0;
        letter-spacing: -0.02em;
    }
    
    /* Subtítulos */
    h2 {
        color: #1E88E5;
        font-weight: 600;
        font-size: 1.8rem !important;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #1E88E5;
    }
    
    h3 {
        color: #00ACC1;
        font-weight: 600;
        font-size: 1.4rem !important;
        margin-top: 1.5rem;
    }
    
    /* Tarjetas de información */
    .info-card {
        background: linear-gradient(135deg, #1E2128 0%, #2A2D3A 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1E88E5;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Cajas de métricas */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #1E88E5;
    }
    
    [data-testid="stMetricLabel"] {
        color: #B4B4B4;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    /* Botones */
    .stButton > button {
        background: linear-gradient(120deg, #1E88E5 0%, #00ACC1 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(30, 136, 229, 0.4);
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(120deg, #1976D2 0%, #0097A7 100%);
        box-shadow: 0 6px 20px rgba(30, 136, 229, 0.6);
        transform: translateY(-2px);
    }
    
    /* Inputs */
    .stTextInput > div > div > input {
        background-color: #1E2128;
        color: white;
        border: 2px solid #2A2D3A;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #1E88E5;
        box-shadow: 0 0 0 2px rgba(30, 136, 229, 0.2);
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background-color: #1E88E5;
    }
    
    /* DataFrames */
    [data-testid="stDataFrame"] {
        background-color: #1E2128;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Tablas */
    .dataframe {
        background-color: #1E2128 !important;
        color: white !important;
    }
    
    .dataframe thead tr th {
        background: linear-gradient(120deg, #1E88E5 0%, #00ACC1 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 12px !important;
    }
    
    .dataframe tbody tr {
        background-color: #1E2128 !important;
        border-bottom: 1px solid #2A2D3A !important;
    }
    
    .dataframe tbody tr:hover {
        background-color: #2A2D3A !important;
    }
    
    /* Gráficos */
    .stPlotlyChart, .stPyplot {
        background-color: #1E2128;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Mensajes de éxito/error */
    .stSuccess {
        background-color: rgba(0, 200, 83, 0.1);
        border-left: 4px solid #00C853;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stError {
        background-color: rgba(255, 82, 82, 0.1);
        border-left: 4px solid #FF5252;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stInfo {
        background-color: rgba(30, 136, 229, 0.1);
        border-left: 4px solid #1E88E5;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #1E88E5 50%, transparent 100%);
        margin: 2rem 0;
    }
    
    /* Chat */
    [data-testid="stChatMessageContent"] {
        background-color: #1E2128;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1E2128;
        border-radius: 8px;
        color: #1E88E5;
        font-weight: 600;
    }
    
    /* Markdown en info cards */
    .info-card p {
        color: #E0E0E0;
        line-height: 1.6;
    }
    
    /* Animación de carga */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .stSpinner > div {
        border-color: #1E88E5 !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE - INICIALIZACIÓN
# =========================
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False

# =========================
# HEADER CON DISEÑO PROFESIONAL
# =========================
st.markdown("<h1>📊 Optimización de Portafolios – Modelo de Markowitz</h1>", unsafe_allow_html=True)

# Descripción con diseño mejorado
st.markdown("""
<div class="info-card">
    <h3 style="margin-top: 0;">🎯 ¿Qué es un ticker?</h3>
    <p>Un <strong>ticker</strong> es el código con el que se identifica una acción en la bolsa de valores.
    Cada empresa cotizada tiene un ticker único que permite acceder a su información de mercado.</p>
    
    <p><strong>Ejemplos comunes:</strong></p>
    <ul>
        <li><strong>AAPL</strong> → Apple Inc.</li>
        <li><strong>MSFT</strong> → Microsoft Corporation</li>
        <li><strong>GOOGL</strong> → Alphabet (Google)</li>
    </ul>
    
    <p>Estos códigos se utilizan para descargar automáticamente los precios históricos
    y realizar el análisis financiero del portafolio.</p>
</div>
""", unsafe_allow_html=True)

# =========================
# INPUTS CON LAYOUT MEJORADO
# =========================
col1, col2 = st.columns([2, 1])

with col1:
    tickers_input = st.text_input(
        "🔍 Ingrese los tickers separados por comas",
        placeholder="Ejemplo: AAPL, MSFT, GOOGL",
        help="Use los códigos bursátiles oficiales. Separe cada ticker con una coma."
    )

with col2:
    years = st.slider(
        "📅 Horizonte temporal (años)",
        min_value=3,
        max_value=10,
        value=6
    )

# Botón de ejecución centrado
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    if st.button("🚀 Ejecutar optimización", use_container_width=True):
        st.session_state.run_analysis = True
        st.session_state.analysis_done = False

# =========================
# PROCESAMIENTO Y ANÁLISIS
# =========================
if st.session_state.run_analysis and not st.session_state.analysis_done:
    
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    if len(tickers) < 2:
        st.error("⚠️ Ingrese al menos 2 tickers.")
        st.stop()
    
    try:
        with st.spinner('🔄 Descargando datos y optimizando portafolio...'):
            
            # =====================================================================
            # 1.5) DESCARGA Y DEPURACIÓN DE DATOS (SIN LOOK-AHEAD BIAS)
            # =====================================================================
            end_date = datetime.today()
            start_date = end_date.replace(year=end_date.year - years)

            raw_data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False
            )

            # Usar precios ajustados (corrige splits y dividendos)
            data = raw_data["Adj Close"]

            # En caso de MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                data = data.droplevel(0, axis=1)

            data = data[tickers]

            # Ordenar por fecha (seguridad)
            data = data.sort_index()

            # Rellenar valores faltantes SOLO hacia adelante
            data = data.ffill()

            # Eliminar filas que sigan incompletas (inicio de la serie)
            data = data.dropna()

            st.markdown("### 📈 Precios ajustados depurados (primeras filas)")
            st.dataframe(data.head(), use_container_width=True)

            # =====================================================================
            # 2) RETORNOS Y MATRICES
            # =====================================================================
            returns = data.pct_change().dropna()
            mean_returns_daily = returns.mean()
            cov_daily = returns.cov()

            trading_days = 252
            mean_returns_annual = mean_returns_daily * trading_days
            cov_annual = cov_daily * trading_days

            # =====================================================================
            # 3) FUNCIONES DE OPTIMIZACIÓN
            # =====================================================================
            def performance(weights, mean_ret, cov):
                ret = np.dot(weights, mean_ret)
                vol = np.sqrt(weights.T @ cov @ weights)
                sharpe = ret / vol if vol > 0 else 0
                return ret, vol, sharpe

            def neg_sharpe(weights):
                r, v, _ = performance(weights, mean_returns_annual, cov_annual)
                return -(r / v) if v > 0 else 1e6

            def vol(weights):
                return np.sqrt(weights.T @ cov_annual @ weights)

            def max_drawdown(series):
                cumulative_max = series.cummax()
                drawdown = (series / cumulative_max) - 1
                return drawdown.min()

            n = len(tickers)
            x0 = np.repeat(1 / n, n)
            bounds = tuple((0, 1) for _ in range(n))
            constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

            # =====================================================================
            # 4) OPTIMIZACIONES
            # =====================================================================
            res_sharpe = minimize(neg_sharpe, x0, method="SLSQP",
                                  bounds=bounds, constraints=constraints)
            weights_sharpe = res_sharpe.x
            ret_sharpe, vol_sharpe, sharpe_sharpe = performance(
                weights_sharpe, mean_returns_annual, cov_annual
            )

            res_minvol = minimize(vol, x0, method="SLSQP",
                                  bounds=bounds, constraints=constraints)
            weights_minvol = res_minvol.x
            ret_minvol, vol_minvol, sharpe_minvol = performance(
                weights_minvol, mean_returns_annual, cov_annual
            )

            weights_equal = np.repeat(1 / n, n)
            ret_equal, vol_equal, sharpe_equal = performance(
                weights_equal, mean_returns_annual, cov_annual
            )

            # =====================================================================
            # 5) RENDIMIENTOS DE CADA ESTRATEGIA
            # =====================================================================
            cumulative_assets = (1 + returns).cumprod()

            daily_sharpe = returns.dot(weights_sharpe)
            daily_minvol = returns.dot(weights_minvol)
            daily_equal = returns.dot(weights_equal)

            cum_sharpe = (1 + daily_sharpe).cumprod()
            cum_minvol = (1 + daily_minvol).cumprod()
            cum_equal = (1 + daily_equal).cumprod()

            dd_sharpe = max_drawdown(cum_sharpe)
            dd_minvol = max_drawdown(cum_minvol)
            dd_equal = max_drawdown(cum_equal)

            # =====================================================================
            # 5.1) DESCARGA DE BENCHMARKS DE MERCADO
            # =====================================================================

            benchmarks = {
                "S&P 500 (SPY)": "SPY",
                "Nasdaq 100 (QQQ)": "QQQ",
                "MSCI World (URTH)": "URTH"
            }

            benchmark_data = yf.download(
                list(benchmarks.values()),
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False
            )["Adj Close"]

            # Asegurar formato correcto
            if isinstance(benchmark_data.columns, pd.MultiIndex):
                benchmark_data = benchmark_data.droplevel(0, axis=1)

            benchmark_data = benchmark_data.ffill().dropna()

            benchmark_returns = benchmark_data.pct_change().dropna()
            benchmark_cum = (1 + benchmark_returns).cumprod()


            # =====================================================================
            # 6) FRONTERA EFICIENTE
            # =====================================================================
            target_returns = np.linspace(
                mean_returns_annual.min(),
                mean_returns_annual.max(),
                50
            )

            efficient_vols, efficient_rets = [], []

            for targ in target_returns:
                cons = (
                    {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                    {"type": "eq",
                     "fun": lambda w, targ=targ: np.dot(w, mean_returns_annual) - targ}
                )
                res = minimize(vol, x0, method="SLSQP",
                               bounds=bounds, constraints=cons)
                if res.success:
                    r, v, _ = performance(res.x, mean_returns_annual, cov_annual)
                    efficient_rets.append(r)
                    efficient_vols.append(v)

        # =====================================================================
        # 7) PRECIOS 2025 Y TENDENCIA
        # =====================================================================
        st.markdown("---")
        st.markdown("### 📊 Precios relevantes del año 2025 (últimas 10 filas)")
        precios_2025 = data[data.index.year == 2025].tail(10)
        st.dataframe(precios_2025 if not precios_2025.empty else "No hay datos de 2025.", use_container_width=True)

        st.markdown(f"### 📈 Tendencia de precios (últimos {years} años)")
        st.line_chart(data, use_container_width=True)

        st.markdown("""
        <div class="info-card">
            <p><strong>Interpretación:</strong></p>
            <p>Este gráfico muestra la evolución histórica de los precios ajustados de cada activo
            durante el horizonte temporal seleccionado.</p>
            
            <ul>
                <li>Tendencias crecientes indican periodos de apreciación del activo.</li>
                <li>Periodos de alta pendiente reflejan fases de crecimiento acelerado.</li>
                <li>Movimientos bruscos o caídas pronunciadas suelen asociarse a eventos de mercado
                  o episodios de alta volatilidad.</li>
            </ul>
            
            <p>Este análisis permite identificar activos con comportamientos más estables
            frente a otros con mayor variabilidad en el tiempo.</p>
        </div>
        """, unsafe_allow_html=True)

        # =====================================================================
        # 8) COMPARACIÓN SISTEMÁTICA DE ESTRATEGIAS
        # =====================================================================
        st.markdown("---")
        st.markdown("### ⚖️ Comparación sistemática de estrategias")

        df_compare = pd.DataFrame({
            "Estrategia": ["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"],
            "Retorno Anual": [ret_sharpe, ret_minvol, ret_equal],
            "Volatilidad": [vol_sharpe, vol_minvol, vol_equal],
            "Sharpe": [sharpe_sharpe, sharpe_minvol, sharpe_equal],
            "Retorno Acumulado": [
                cum_sharpe.iloc[-1] - 1,
                cum_minvol.iloc[-1] - 1,
                cum_equal.iloc[-1] - 1
            ],
            "Máx Drawdown": [dd_sharpe, dd_minvol, dd_equal]
        })

        st.dataframe(df_compare, use_container_width=True)

        st.markdown("""
        <div class="info-card">
            <p><strong>Cómo interpretar esta tabla:</strong></p>
            <ul>
                <li><strong>Retorno acumulado:</strong> cuánto creció el capital total en el periodo.</li>
                <li><strong>Volatilidad:</strong> magnitud de las fluctuaciones (riesgo).</li>
                <li><strong>Sharpe:</strong> eficiencia riesgo–retorno.</li>
                <li><strong>Máx Drawdown:</strong> peor caída histórica desde un máximo.</li>
            </ul>
            
            <p><strong>Interpretación analítica de la comparación de estrategias:</strong></p>
            
            <p>Esta tabla sintetiza el desempeño de las distintas estrategias
            de construcción de portafolios bajo un enfoque riesgo–retorno,
            permitiendo una evaluación integral y comparativa.</p>
            
            <ul>
                <li>La estrategia de <strong>Sharpe Máximo</strong> tiende a ofrecer el mayor
                  retorno ajustado por riesgo, aunque suele presentar niveles
                  más elevados de volatilidad y drawdowns en periodos adversos.</li>
                <li>La estrategia de <strong>Mínima Volatilidad</strong> prioriza la estabilidad
                  del capital, reduciendo la exposición a caídas pronunciadas,
                  a costa de un menor potencial de crecimiento.</li>
                <li>La estrategia de <strong>Pesos Iguales</strong> actúa como referencia neutral,
                  proporcionando una diversificación básica sin optimización explícita.</li>
            </ul>
            
            <p>La combinación de métricas como retorno anual, volatilidad,
            Ratio de Sharpe y máximo drawdown permite identificar no solo
            la estrategia más rentable, sino también la más resiliente
            frente a escenarios de estrés de mercado.</p>
        </div>
        """, unsafe_allow_html=True)

        # =====================================================================
        # 8.1) VOLATILIDAD HISTÓRICA ROLLING (RIESGO DINÁMICO)
        # =====================================================================
        st.markdown("---")
        st.markdown("### 📉 Volatilidad histórica móvil")

        rolling_vol = pd.DataFrame({
            "Sharpe Máximo": daily_sharpe.rolling(252).std() * np.sqrt(252),
            "Mínima Volatilidad": daily_minvol.rolling(252).std() * np.sqrt(252),
            "Pesos Iguales": daily_equal.rolling(252).std() * np.sqrt(252)
        })

        st.line_chart(rolling_vol, use_container_width=True)

        st.markdown("""
        <div class="info-card">
            <p><strong>Interpretación:</strong></p>
            <p>Esta gráfica muestra cómo el riesgo <strong>cambia en el tiempo</strong>.</p>
            <ul>
                <li>Picos altos suelen coincidir con periodos de crisis.</li>
                <li>Estrategias más estables presentan curvas más suaves.</li>
            </ul>
            
            <p>La volatilidad histórica móvil permite analizar cómo
            evoluciona el riesgo del portafolio a lo largo del tiempo,
            capturando cambios estructurales en el comportamiento del mercado.</p>
            
            <p>En el análisis comparativo:</p>
            <ul>
                <li>El portafolio de <strong>Sharpe Máximo</strong> presenta picos de
                  volatilidad más elevados, reflejando una mayor exposición
                  al riesgo en escenarios adversos.</li>
                <li>La estrategia de <strong>Mínima Volatilidad</strong> mantiene un perfil
                  de riesgo más controlado a lo largo del tiempo.</li>
                <li>La asignación de <strong>Pesos Iguales</strong> muestra un comportamiento
                  intermedio, replicando parcialmente la dinámica del mercado.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # =====================================================================
        # 8.2) RATIO CALMAR
        # =====================================================================
        calmar_sharpe = ret_sharpe / abs(dd_sharpe)
        calmar_minvol = ret_minvol / abs(dd_minvol)
        calmar_equal = ret_equal / abs(dd_equal)

        st.markdown("---")
        st.markdown("### 📊 Ratio Calmar (retorno vs drawdown)")

        df_calmar = pd.DataFrame({
            "Estrategia": ["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"],
            "Calmar": [calmar_sharpe, calmar_minvol, calmar_equal]
        })

        # Mostrar en columnas con métricas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sharpe Máximo", f"{calmar_sharpe:.2f}")
        with col2:
            st.metric("Mínima Volatilidad", f"{calmar_minvol:.2f}")
        with col3:
            st.metric("Pesos Iguales", f"{calmar_equal:.2f}")

        st.dataframe(df_calmar, use_container_width=True)

        st.markdown("""
        <div class="info-card">
            <p><strong>Interpretación analítica del Ratio Calmar:</strong></p>
            
            <p>El Ratio Calmar relaciona el <strong>retorno anual esperado</strong> con el
            <strong>máximo drawdown histórico</strong>, ofreciendo una medida directa
            de la capacidad del portafolio para generar rentabilidad
            sin incurrir en pérdidas extremas prolongadas.</p>
            
            <ul>
                <li>Un <strong>Ratio Calmar elevado</strong> indica que la estrategia logra
                  retornos atractivos manteniendo caídas relativamente controladas.</li>
                <li>Valores bajos sugieren que el retorno obtenido no compensa
                  adecuadamente las pérdidas máximas sufridas.</li>
                <li>Esta métrica resulta especialmente relevante para
                  inversionistas con enfoque conservador o con restricciones
                  estrictas de preservación de capital.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # =====================================================================
        # 8.3) SORTINO RATIO
        # =====================================================================
        downside = returns.copy()
        downside[downside > 0] = 0
        downside_std = downside.std() * np.sqrt(252)

        sortino_sharpe = ret_sharpe / downside_std.dot(weights_sharpe)
        sortino_minvol = ret_minvol / downside_std.dot(weights_minvol)
        sortino_equal = ret_equal / downside_std.dot(weights_equal)

        st.markdown("---")
        st.markdown("### 📊 Ratio Sortino")

        df_sortino = pd.DataFrame({
            "Estrategia": ["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"],
            "Sortino": [sortino_sharpe, sortino_minvol, sortino_equal]
        })

        # Mostrar en columnas con métricas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sharpe Máximo", f"{sortino_sharpe:.2f}")
        with col2:
            st.metric("Mínima Volatilidad", f"{sortino_minvol:.2f}")
        with col3:
            st.metric("Pesos Iguales", f"{sortino_equal:.2f}")

        st.dataframe(df_sortino, use_container_width=True)

        st.markdown("""
        <div class="info-card">
            <p><strong>Interpretación analítica del Ratio Sortino:</strong></p>
            
            <p>El Ratio Sortino evalúa el desempeño del portafolio considerando
            exclusivamente la <strong>volatilidad negativa</strong>, es decir, aquellas
            fluctuaciones que representan pérdidas para el inversor.</p>
            
            <ul>
                <li>Un <strong>valor más alto de Sortino</strong> indica que la estrategia genera
                  mayor retorno por cada unidad de riesgo a la baja asumida.</li>
                <li>A diferencia del Ratio de Sharpe, este indicador <strong>no penaliza
                  la volatilidad positiva</strong>, lo que lo convierte en una métrica
                  más alineada con la percepción real del riesgo por parte del inversor.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # =====================================================================
        # 8.4) PERIODOS DE CRISIS (COVID 2020)
        # =====================================================================
        st.markdown("---")
        st.markdown("### 🦠 Comportamiento en periodo de crisis (COVID 2020)")

        crisis = (cum_sharpe.index.year == 2020)

        st.line_chart(pd.DataFrame({
            "Sharpe Máximo": cum_sharpe[crisis],
            "Mínima Volatilidad": cum_minvol[crisis],
            "Pesos Iguales": cum_equal[crisis]
        }), use_container_width=True)

        st.markdown("""
        <div class="info-card">
            <p><strong>Interpretación del comportamiento en periodo de crisis:</strong></p>
            
            <p>Esta visualización muestra el desempeño de las distintas
            estrategias durante un periodo de estrés sistémico,
            caracterizado por alta volatilidad y caídas abruptas del mercado.</p>
            
            <p>El análisis permite evaluar:</p>
            <ul>
                <li>La <strong>profundidad de la caída</strong> inicial (drawdown).</li>
                <li>La <strong>velocidad de recuperación</strong> tras el shock.</li>
                <li>La <strong>resiliencia relativa</strong> de cada estrategia ante eventos extremos.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # =====================================================================
        # 8.5) COMPARACIÓN CON BENCHMARKS DE MERCADO
        # =====================================================================
        st.markdown("---")
        st.markdown("### 📊 Comparación con benchmarks de mercado")

        def annualized_return(series):
            return (series.iloc[-1]) ** (252 / len(series)) - 1

        def annualized_vol(series):
            return series.std() * np.sqrt(252)

        benchmark_summary = []

        for name, ticker in benchmarks.items():
            ret = annualized_return(benchmark_cum[ticker])
            vol = annualized_vol(benchmark_returns[ticker])
            dd = max_drawdown(benchmark_cum[ticker])

            benchmark_summary.append({
                "Benchmark": name,
                "Retorno Anual": ret,
                "Volatilidad": vol,
                "Retorno Acumulado": benchmark_cum[ticker].iloc[-1] - 1,
                "Máx Drawdown": dd
            })

        df_benchmarks = pd.DataFrame(benchmark_summary)
        st.dataframe(df_benchmarks, use_container_width=True)

        with st.expander("📚 ¿Qué son los benchmarks y por qué son importantes?"):
            st.markdown("""
            ### ¿Qué es un benchmark?

            Un **benchmark** es un **punto de referencia** que se utiliza para evaluar si una estrategia de inversión es buena o mala.
            Funciona de forma similar a una *regla de medición*: permite comparar los resultados obtenidos con una alternativa estándar y ampliamente utilizada en los mercados financieros.

            ### ¿Qué representa el S&P 500?

            El **S&P 500** es uno de los índices bursátiles más conocidos del mundo. Agrupa a aproximadamente **500 de las empresas más grandes de Estados Unidos**, como Apple, Microsoft o Google.

            ### ¿Qué es el MSCI?

            **MSCI** (Morgan Stanley Capital International) es una empresa internacional que elabora **índices bursátiles** utilizados como referencia en todo el mundo.

            ### ¿Qué es el NASDAQ?

            El **NASDAQ** es una bolsa de valores estadounidense caracterizada por una **alta concentración de empresas tecnológicas y de innovación**.
            """)

        # =====================================================================
        # 8.6) RENDIMIENTO ACUMULADO: ESTRATEGIAS VS BENCHMARKS
        # =====================================================================
        st.markdown("---")
        st.markdown("### 📈 Rendimiento acumulado: estrategias vs benchmarks")

        comparison_cum = pd.DataFrame({
            "Sharpe Máximo": cum_sharpe,
            "Mínima Volatilidad": cum_minvol,
            "Pesos Iguales": cum_equal,
            "S&P 500 (SPY)": benchmark_cum["SPY"],
            "Nasdaq 100 (QQQ)": benchmark_cum["QQQ"],
            "MSCI World (URTH)": benchmark_cum["URTH"]
        })

        st.line_chart(comparison_cum, use_container_width=True)

        st.markdown("""
        <div class="info-card">
            <p><strong>Cómo interpretar la gráfica de rendimiento acumulado:</strong></p>
            
            <ul>
                <li>La línea que termina <strong>más arriba</strong> representa la estrategia con <strong>mayor crecimiento acumulado</strong>.</li>
                <li>Las curvas más <strong>suaves y estables</strong> indican menor volatilidad y menor exposición a crisis.</li>
                <li>Caídas pronunciadas reflejan periodos de estrés de mercado; una recuperación rápida indica mayor resiliencia.</li>
                <li>Si una estrategia optimizada supera de forma consistente a los benchmarks, se confirma que el modelo aporta valor frente a una inversión pasiva.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # =====================================================================
        # 9) SÍNTESIS ANALÍTICA PARA EL ASISTENTE (PERSISTENTE)
        # =====================================================================

        asset_summary = {}

        for ticker in tickers:
            asset_summary[ticker] = {
                "retorno_anual": mean_returns_annual[ticker],
                "volatilidad": np.sqrt(cov_annual.loc[ticker, ticker]),
                "contribucion_riesgo": cov_annual.loc[ticker].dot(weights_sharpe)
            }

        strategy_summary = {
            "Sharpe Máximo": {
                "retorno": ret_sharpe,
                "volatilidad": vol_sharpe,
                "sharpe": sharpe_sharpe,
                "drawdown": dd_sharpe
            },
            "Mínima Volatilidad": {
                "retorno": ret_minvol,
                "volatilidad": vol_minvol,
                "sharpe": sharpe_minvol,
                "drawdown": dd_minvol
            },
            "Pesos Iguales": {
                "retorno": ret_equal,
                "volatilidad": vol_equal,
                "sharpe": sharpe_equal,
                "drawdown": dd_equal
            }
        }

        # =====================================================================
        # 10) RENDIMIENTOS ACUMULADOS
        # =====================================================================
        st.markdown("---")
        st.markdown("### 📊 Rendimiento acumulado por acción")
        st.line_chart(cumulative_assets, use_container_width=True)

        st.markdown("### 📈 Comparación de rendimientos de estrategias")
        st.line_chart(
            pd.DataFrame({
                "Sharpe Máximo": cum_sharpe,
                "Mínima Volatilidad": cum_minvol,
                "Pesos Iguales": cum_equal
            }),
            use_container_width=True
        )

        st.markdown("""
        <div class="info-card">
            <p><strong>Interpretación:</strong></p>
            
            <p>El rendimiento acumulado refleja cómo habría evolucionado una inversión inicial
            en cada activo si se hubiera mantenido durante todo el periodo de análisis.</p>
            
            <ul>
                <li>Curvas más empinadas indican mayor crecimiento del capital.</li>
                <li>Activos con mayor volatilidad suelen mostrar trayectorias más irregulares.</li>
                <li>Diferencias significativas entre curvas evidencian distintos perfiles
                  de riesgo y rentabilidad.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # =====================================================================
        # GRÁFICO DE RETORNOS DIARIOS ACUMULADOS
        # =====================================================================
        st.markdown("---")
        st.markdown("### 📉 Retornos diarios de los activos")
        st.line_chart(returns, use_container_width=True)

        st.markdown("""
        <div class="info-card">
            <p><strong>Interpretación:</strong></p>
            
            <p>Este gráfico muestra los retornos porcentuales diarios de cada activo,
            evidenciando la volatilidad de corto plazo.</p>
            
            <ul>
                <li>Picos positivos o negativos representan movimientos abruptos del mercado.</li>
                <li>Mayor dispersión implica mayor riesgo.</li>
                <li>Periodos de alta concentración de picos suelen coincidir con crisis financieras
                  o eventos macroeconómicos relevantes.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # =====================================================================
        # GRÁFICO DE RETORNOS DIARIOS POR ACTIVO
        # =====================================================================
        st.markdown("---")
        st.markdown("### 📊 Retornos diarios por activo")

        # Mostrar en dos columnas para mejor organización
        num_cols = 2
        cols = st.columns(num_cols)
        
        for idx, ticker in enumerate(returns.columns):
            with cols[idx % num_cols]:
                st.markdown(f"#### {ticker}")
                st.line_chart(returns[[ticker]], use_container_width=True)

        st.markdown("""
        <div class="info-card">
            <p><strong>Interpretación:</strong></p>
            
            <p>Este gráfico muestra el comportamiento diario del retorno del activo,
            permitiendo identificar:</p>
            
            <ul>
                <li>Frecuencia e intensidad de pérdidas y ganancias.</li>
                <li>Presencia de volatilidad asimétrica (más caídas que subidas).</li>
                <li>Episodios de estrés específicos para el activo.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # =====================================================================
        # 11) FRONTERA EFICIENTE (MEJORADA CON ETIQUETAS)
        # =====================================================================
        st.markdown("---")
        st.markdown("### 🎯 Frontera eficiente (Retorno vs Volatilidad)")

        fig2, ax2 = plt.subplots(figsize=(10, 7))
        
        # Configurar el estilo oscuro para matplotlib
        fig2.patch.set_facecolor('#1E2128')
        ax2.set_facecolor('#1E2128')
        
        # Frontera eficiente
        ax2.plot(
            efficient_vols,
            efficient_rets,
            linestyle="-",
            linewidth=3,
            label="Frontera eficiente",
            color='#1E88E5'
        )
        
        # Portafolios destacados con colores distintivos
        ax2.scatter(
            vol_sharpe,
            ret_sharpe,
            s=150,
            marker="o",
            label="Sharpe Máximo",
            color='#00C853',
            edgecolors='white',
            linewidths=2,
            zorder=5
        )

        ax2.scatter(
            vol_minvol,
            ret_minvol,
            s=150,
            marker="^",
            label="Mínima Volatilidad",
            color='#7C4DFF',
            edgecolors='white',
            linewidths=2,
            zorder=5
        )
        
        ax2.scatter(
            vol_equal,
            ret_equal,
            s=150,
            marker="s",
            label="Pesos Iguales",
            color='#FFB300',
            edgecolors='white',
            linewidths=2,
            zorder=5
        )
        
        # Etiquetas de los puntos
        ax2.annotate(
            "Sharpe Máximo",
            (vol_sharpe, ret_sharpe),
            xytext=(8, 8),
            textcoords="offset points",
            fontweight="bold",
            color='white',
            fontsize=10
        )
        ax2.annotate(
            "Mínima Volatilidad",
            (vol_minvol, ret_minvol),
            xytext=(8, -12),
            textcoords="offset points",
            fontweight="bold",
            color='white',
            fontsize=10
        )
        ax2.annotate(
            "Pesos Iguales",
            (vol_equal, ret_equal),
            xytext=(8, 8),
            textcoords="offset points",
            fontweight="bold",
            color='white',
            fontsize=10
        )
        
        # Ejes y título con colores personalizados
        ax2.set_xlabel("Volatilidad anual (riesgo)", color='white', fontsize=12, fontweight='bold')
        ax2.set_ylabel("Retorno anual esperado", color='white', fontsize=12, fontweight='bold')
        ax2.set_title("Frontera eficiente y estrategias comparadas", color='white', fontsize=14, fontweight='bold', pad=20)
        
        # Personalizar la leyenda
        legend = ax2.legend(facecolor='#2A2D3A', edgecolor='#1E88E5', framealpha=0.9)
        for text in legend.get_texts():
            text.set_color('white')
        
        # Grid personalizado
        ax2.grid(True, alpha=0.2, color='white', linestyle='--')
        
        # Cambiar color de los ticks
        ax2.tick_params(colors='white')
        
        # Cambiar color de los spines (bordes)
        for spine in ax2.spines.values():
            spine.set_color('#2A2D3A')
        
        st.pyplot(fig2)

        st.markdown("""
        <div class="info-card">
            <p><strong>Interpretación analítica de la frontera eficiente:</strong></p>
            
            <p>La frontera eficiente representa el conjunto de portafolios
            óptimos que maximizan el retorno esperado para cada nivel
            de riesgo asumido, de acuerdo con la teoría media–varianza
            de Markowitz.</p>
            
            <ul>
                <li>Cada punto de la curva corresponde a una combinación
                  distinta de activos que no puede ser mejorada simultáneamente
                  en términos de mayor retorno y menor riesgo.</li>
                <li>Los portafolios situados por debajo de la frontera son
                  ineficientes, ya que existe al menos una alternativa
                  con mejor desempeño riesgo–retorno.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # =====================================================================
        # INTERPRETACIÓN FINAL – COMPORTAMIENTO REAL PONDERADO EN EL TIEMPO
        # =====================================================================
        st.markdown("---")
        st.markdown("### 🏆 Interpretación automática del mejor portafolio")

        df_strategies = pd.DataFrame({
            "Sharpe Máximo": daily_sharpe,
            "Mínima Volatilidad": daily_minvol,
            "Pesos Iguales": daily_equal
        })

        # Ponderación temporal (años recientes pesan más)
        years_index = df_strategies.index.year
        unique_years = np.sort(years_index.unique())

        year_weights = {
            year: (i + 1) / len(unique_years)
            for i, year in enumerate(unique_years)
        }

        weights_series = years_index.map(year_weights)

        # Retorno real ponderado
        weighted_performance = (
            (1 + df_strategies).cumprod()
            .mul(weights_series, axis=0)
            .iloc[-1]
        )

        best = weighted_performance.idxmax()

        st.dataframe(weighted_performance.rename("Desempeño_Ponderado"), use_container_width=True)

        # Interpretación con diseño mejorado
        if best == "Pesos Iguales":
            st.markdown("""
            <div class="info-card" style="border-left-color: #FFB300;">
                <h3 style="color: #FFB300; margin-top: 0;">🏆 Mejor portafolio: Pesos Iguales</h3>
                
                <p>El análisis del <strong>comportamiento real del portafolio en el tiempo</strong>, 
                ponderando más los años recientes, muestra que esta estrategia ha sido 
                la <strong>más robusta y consistente</strong>.</p>
                
                <ul>
                    <li>Menor dependencia de supuestos estadísticos.</li>
                    <li>Mejor desempeño agregado a lo largo del tiempo.</li>
                    <li>Alta estabilidad frente a cambios de mercado.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        elif best == "Sharpe Máximo":
            st.markdown("""
            <div class="info-card" style="border-left-color: #00C853;">
                <h3 style="color: #00C853; margin-top: 0;">🏆 Mejor portafolio: Sharpe Máximo</h3>
                
                <p>La evaluación temporal indica que esta estrategia ofrece el mejor 
                equilibrio riesgo–retorno en el comportamiento histórico reciente.</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="info-card" style="border-left-color: #7C4DFF;">
                <h3 style="color: #7C4DFF; margin-top: 0;">🏆 Mejor portafolio: Mínima Volatilidad</h3>
                
                <p>Esta estrategia destaca por su estabilidad, aunque sacrifica retorno 
                frente a las demás.</p>
            </div>
            """, unsafe_allow_html=True)

        st.success(f"✅ Portafolio recomendado según comportamiento real ponderado: **{best}**")

        # =====================================================================
        # 9) PESOS ÓPTIMOS SEGÚN PORTAFOLIO RECOMENDADO
        # =====================================================================
        st.markdown("---")
        st.markdown("### ⚖️ Pesos óptimos del portafolio recomendado")

        n_assets = len(tickers)

        if best == "Sharpe Máximo":
            final_weights = weights_sharpe
            metodo = "Optimización por Ratio de Sharpe"
            color_metodo = "#00C853"

        elif best == "Mínima Volatilidad":
            final_weights = weights_minvol
            metodo = "Optimización por Mínima Volatilidad"
            color_metodo = "#7C4DFF"

        else:  # Pesos Iguales
            final_weights = np.array([1 / n_assets] * n_assets)
            metodo = "Asignación Equitativa (Pesos Iguales)"
            color_metodo = "#FFB300"

        df_weights = pd.DataFrame({
            "Ticker": tickers,
            "Peso": final_weights,
            "Peso (%)": final_weights * 100
        })

        st.dataframe(df_weights, use_container_width=True)

        # --- Gráfico mejorado ---
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#1E2128')
        ax.set_facecolor('#1E2128')
        
        bars = ax.barh(df_weights["Ticker"], df_weights["Peso"], color=color_metodo, edgecolor='white', linewidth=1.5)
        
        # Añadir valores en las barras
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.1%}',
                   ha='left', va='center', color='white', fontweight='bold', fontsize=10, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#2A2D3A', edgecolor='none', alpha=0.8))
        
        ax.set_title(f"Composición del portafolio recomendado\n({metodo})", 
                    color='white', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("Peso del activo", color='white', fontsize=11, fontweight='bold')
        ax.tick_params(colors='white')
        
        for spine in ax.spines.values():
            spine.set_color('#2A2D3A')
        
        ax.grid(True, alpha=0.2, color='white', linestyle='--', axis='x')
        
        st.pyplot(fig)

        st.markdown(f"""
        <div class="info-card">
            <h3 style="margin-top: 0;">💡 Interpretación de los pesos</h3>
            
            <p>Los pesos mostrados corresponden <strong>exclusivamente</strong> al portafolio
            recomendado por el modelo (<strong>{best}</strong>).</p>
            
            <ul>
                <li>Cada peso indica qué proporción del capital debe asignarse a cada activo.</li>
                <li>La suma total de los pesos es del <strong>100%</strong>.</li>
                <li>Esta asignación refleja el comportamiento histórico del portafolio
                  bajo el criterio seleccionado.</li>
            </ul>
            
            <h3>📝 Explicación extendida de los pesos óptimos</h3>
            
            <p>Los <strong>pesos óptimos</strong> indican cómo distribuir el capital para obtener
            el mejor balance entre <strong>riesgo y retorno</strong>, según el modelo de Markowitz.</p>
            
            <ul>
                <li>Un <strong>peso del 40%</strong> significa que <strong>40 de cada 100 unidades monetarias</strong>
                  se asignan a ese activo.</li>
                <li><strong>Pesos altos</strong> reflejan activos que aportan mayor eficiencia al portafolio.</li>
                <li><strong>Pesos bajos</strong> indican activos que añaden más riesgo que beneficio relativo.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.session_state.analysis_done = True

        st.success("✅ Análisis del portafolio ejecutado correctamente")

        # ======================================================
        # GUARDAR RESULTADOS PARA EL CHAT
        # ======================================================
        st.session_state["analysis_results"] = {
            "tickers": tickers,
            "best": best,

            # Comparación general
            "comparison": df_compare,

            # Pesos del portafolio recomendado (tabla)
            "weights_recommended": df_weights,

            # Pesos óptimos por estrategia (clave para el chat)
            "weights": {
                "Sharpe Máximo": dict(zip(tickers, weights_sharpe)),
                "Mínima Volatilidad": dict(zip(tickers, weights_minvol)),
                "Pesos Iguales": dict(zip(tickers, [1 / len(tickers)] * len(tickers)))
            },

            # Retornos esperados
            "retornos": {
                "Sharpe Máximo": ret_sharpe,
                "Mínima Volatilidad": ret_minvol,
                "Pesos Iguales": ret_equal
            },

            # Volatilidades
            "volatilidades": {
                "Sharpe Máximo": vol_sharpe,
                "Mínima Volatilidad": vol_minvol,
                "Pesos Iguales": vol_equal
            },
            # 🔹 NUEVO — NO BORRES NADA DE ARRIBA
            "asset_summary": asset_summary,
            "strategy_summary": strategy_summary
        }

    except Exception as e:
        st.error(f"❌ Error: {e}")

# ======================================================
# MOSTRAR RESULTADOS (FUERA DEL BOTÓN)
# ======================================================

if st.session_state.analysis_done:
    results = st.session_state.analysis_results

    st.markdown("---")
    st.markdown("### 📊 Resumen de resultados")

    st.markdown("#### Comparación de estrategias")
    st.dataframe(results["comparison"], use_container_width=True)

    st.markdown("#### Pesos del portafolio recomendado")
    st.dataframe(results["weights_recommended"], use_container_width=True)

    df_retornos = pd.DataFrame(
        {
            "Retorno anual esperado": [
                results["retornos"]["Sharpe Máximo"],
                results["retornos"]["Mínima Volatilidad"],
                results["retornos"]["Pesos Iguales"]
            ]
        },
        index=["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"]
    )

    st.markdown("#### Ratio / retorno esperado por estrategia")
    st.dataframe(df_retornos, use_container_width=True)

# ======================================================
# ASISTENTE INTELIGENTE DEL PORTAFOLIO (GEMINI)
# ======================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h2 style="color: #1E88E5; font-size: 2.5rem; margin-bottom: 0.5rem;">🤖 Asistente inteligente del portafolio</h2>
    <p style="color: #B4B4B4; font-size: 1.1rem;">Pregunta lo que necesites sobre tu análisis</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.analysis_done:
    st.info("ℹ️ Ejecuta primero la optimización para habilitar el asistente.")
else:
    import requests
    import os

    # =========================
    # CONFIGURACIÓN GEMINI
    # =========================
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        st.warning("⚠️ El asistente requiere una API Key válida de Gemini.")
        st.stop()

    MODEL = "gemini-2.5-flash-lite"
    GEMINI_URL = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{MODEL}:generateContent?key={GEMINI_API_KEY}"
    )

    # =========================
    # HISTORIAL DE CHAT
    # =========================
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input(
        "💬 Pregunta sobre los tickers, riesgos o el portafolio recomendado..."
    )

    if user_question:
        st.session_state.chat_messages.append(
            {"role": "user", "content": user_question}
        )

        results = st.session_state.analysis_results

        # =========================
        # CONTEXTO FINANCIERO
        # =========================
        best_strategy = results["best"]
        weights_dict = results["weights"][best_strategy]

        weights_text = "\n".join(
            f"- {k}: {v:.2%}" for k, v in weights_dict.items()
        )

        asset_text = "\n".join(
            f"- {k}: retorno anual={v['retorno_anual']:.2%}, "
            f"volatilidad={v['volatilidad']:.2%}"
            for k, v in results["asset_summary"].items()
        )

        strategy_text = "\n".join(
            f"- {k}: retorno={v['retorno']:.2%}, "
            f"volatilidad={v['volatilidad']:.2%}, "
            f"Sharpe={v['sharpe']:.2f}, "
            f"drawdown={v['drawdown']:.2%}"
            for k, v in results["strategy_summary"].items()
        )

        # =========================
        # PROMPT OPTIMIZADO
        # =========================
        system_prompt = f"""
Actúa como un analista financiero profesional.

CONTEXTO (úsalo solo si es necesario):
Activos analizados: {', '.join(results['tickers'])}

Resumen de activos:
{asset_text}

Resumen de estrategias:
{strategy_text}

Estrategia recomendada: {best_strategy}
Pesos del portafolio recomendado:
{weights_text}

INSTRUCCIONES ESTRICTAS:
- Responde ÚNICAMENTE la pregunta del usuario.
- Usa lenguaje claro para personas no técnicas.
- La respuesta DEBE tener al menos 2 párrafos cortos.
- Máximo 4 párrafos en total.
- Cada párrafo debe aportar información distinta (no repetir ideas).
- No expliques teoría financiera innecesaria.
- Si aplica, menciona brevemente riesgo y retorno.
- Si preguntan por cifras, usa números concretos.
- No inventes datos.
- Termina siempre la respuesta.
"""

        # =========================
        # LLAMADA A GEMINI
        # =========================
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": system_prompt
                            + "\n\nPregunta del usuario:\n"
                            + user_question
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 900
            }
        }

        with st.spinner('Pensando...'):
            response = requests.post(GEMINI_URL, json=payload)

        if response.status_code != 200:
            answer = "⚠️ Error al generar la respuesta con Gemini."
        else:
            data = response.json()
            answer = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "No se obtuvo respuesta.")
            )

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.markdown(answer)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem 0; color: #B4B4B4;">
    <p>Portfolio Optimizer Pro | Powered by Modern Portfolio Theory</p>
    <p style="font-size: 0.9rem;">© 2025 - Análisis financiero avanzado</p>
</div>
""", unsafe_allow_html=True)






















