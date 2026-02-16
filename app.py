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

# CSS PERSONALIZADO PROFESIONAL COMPLETO - CORREGIDO
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
        font-size: 3rem !important;
        text-align: center;
        margin-bottom: 1.5rem;
        padding: 1rem 0;
        letter-spacing: -0.02em;
    }
    
    /* Subtítulos */
    h2 {
        color: #1E88E5 !important;
        font-weight: 600 !important;
        font-size: 1.8rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 2px solid #1E88E5 !important;
    }
    
    h3 {
        color: #00ACC1 !important;
        font-weight: 600 !important;
        font-size: 1.4rem !important;
        margin-top: 1.5rem !important;
    }
    
    /* Tarjetas de información - CORREGIDO CON SELECTORES MÁS ESPECÍFICOS */
    .info-card {
        background: linear-gradient(135deg, #1E2128 0%, #2A2D3A 100%) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
        border-left: 4px solid #1E88E5 !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Todos los párrafos dentro de info-card */
    .info-card p,
    .info-card > p,
    div.info-card p {
        color: #E0E0E0 !important;
        line-height: 1.8 !important;
        margin-bottom: 1rem !important;
        font-size: 1rem !important;
    }
    
    /* Todas las listas dentro de info-card */
    .info-card ul,
    .info-card > ul,
    div.info-card ul {
        color: #E0E0E0 !important;
        line-height: 1.8 !important;
        margin-left: 1.5rem !important;
        margin-bottom: 1rem !important;
        list-style-type: disc !important;
    }
    
    /* Todos los items de lista dentro de info-card */
    .info-card li,
    .info-card > ul > li,
    div.info-card ul li,
    div.info-card li {
        color: #E0E0E0 !important;
        margin-bottom: 0.5rem !important;
        font-size: 1rem !important;
    }
    
    /* Todos los elementos strong dentro de info-card */
    .info-card strong,
    .info-card p strong,
    .info-card li strong,
    div.info-card strong {
        color: #1E88E5 !important;
        font-weight: 600 !important;
    }
    
    /* Listas ordenadas dentro de info-card */
    .info-card ol,
    div.info-card ol {
        color: #E0E0E0 !important;
        line-height: 1.8 !important;
        margin-left: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* H3 dentro de info-card */
    .info-card h3,
    div.info-card h3 {
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
        color: #00ACC1 !important;
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

st.title("Optimización de Portafolios – Modelo de Markowitz")

st.markdown("""
<div class="info-card">
    <h3 style="margin-top: 1.5rem !important; margin-bottom: 1rem !important; color: #00ACC1 !important;">🎯 ¿Qué es un ticker?</h3>
    <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">Un <strong style="color: #1E88E5 !important; font-weight: 600 !important;">ticker</strong> es el código con el que se identifica una acción en la bolsa de valores.
    Cada empresa cotizada tiene un ticker único que permite acceder a su información de mercado.</p>
    
    <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;"><strong style="color: #1E88E5 !important; font-weight: 600 !important;">Ejemplos comunes:</strong></p>
    <ul style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-left: 1.5rem !important; list-style-type: disc !important;">
        <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;"><strong style="color: #1E88E5 !important; font-weight: 600 !important;">AAPL</strong> → Apple Inc.</li>
        <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;"><strong style="color: #1E88E5 !important; font-weight: 600 !important;">MSFT</strong> → Microsoft Corporation</li>
        <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;"><strong style="color: #1E88E5 !important; font-weight: 600 !important;">GOOGL</strong> → Alphabet (Google)</li>
    </ul>
    
    <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">Estos códigos se utilizan para descargar automáticamente los precios históricos
    y realizar el análisis financiero del portafolio.</p>
</div>
""", unsafe_allow_html=True)

tickers_input = st.text_input(
    "Ingrese los tickers separados por comas (ejemplo: AAPL, MSFT, GOOGL)",
    help="Use los códigos bursátiles oficiales. Separe cada ticker con una coma."
)

years = st.slider(
    "Seleccione el horizonte temporal (años)",
    min_value=3,
    max_value=10,
    value=6
)

if st.button("Ejecutar optimización"):
    st.session_state.run_analysis = True
    st.session_state.analysis_done = False

if st.session_state.run_analysis and not st.session_state.analysis_done:

        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

        if len(tickers) < 2:
            st.error("Ingrese al menos 2 tickers.")
            st.stop()

        try:

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

            st.subheader("Precios ajustados depurados (primeras filas)")
            st.dataframe(data.head())

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
            st.subheader("Precios relevantes del año 2025 (últimas 10 filas)")
            precios_2025 = data[data.index.year == 2025].tail(10)
            st.dataframe(precios_2025 if not precios_2025.empty else "No hay datos de 2025.")

            st.subheader(f"Tendencia de precios (últimos {years} años)")
            st.line_chart(data)

            st.markdown("""
            <div class="info-card">
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;"><strong style="color: #1E88E5 !important; font-weight: 600 !important;">Interpretación:</strong></p>
                
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">Este gráfico muestra la evolución histórica de los precios ajustados de cada activo
                durante el horizonte temporal seleccionado.</p>
                
                <ul style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-left: 1.5rem !important; list-style-type: disc !important;">
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Tendencias crecientes indican periodos de apreciación del activo.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Periodos de alta pendiente reflejan fases de crecimiento acelerado.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Movimientos bruscos o caídas pronunciadas suelen asociarse a eventos de mercado
                      o episodios de alta volatilidad.</li>
                </ul>
                
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">Este análisis permite identificar activos con comportamientos más estables
                frente a otros con mayor variabilidad en el tiempo.</p>
            </div>
            """, unsafe_allow_html=True)

            # =====================================================================
            # 8) COMPARACIÓN SISTEMÁTICA DE ESTRATEGIAS
            # =====================================================================
            st.subheader("Comparación sistemática de estrategias")

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

            st.dataframe(df_compare)

            st.markdown("""
            <div class="info-card">
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;"><strong style="color: #1E88E5 !important; font-weight: 600 !important;">Cómo interpretar esta tabla:</strong></p>
                <ul style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-left: 1.5rem !important; list-style-type: disc !important;">
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;"><strong style="color: #1E88E5 !important; font-weight: 600 !important;">Retorno acumulado:</strong> cuánto creció el capital total en el periodo.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;"><strong style="color: #1E88E5 !important; font-weight: 600 !important;">Volatilidad:</strong> magnitud de las fluctuaciones (riesgo).</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;"><strong style="color: #1E88E5 !important; font-weight: 600 !important;">Sharpe:</strong> eficiencia riesgo–retorno.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;"><strong style="color: #1E88E5 !important; font-weight: 600 !important;">Máx Drawdown:</strong> peor caída histórica desde un máximo.</li>
                </ul>
                
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;"><strong style="color: #1E88E5 !important; font-weight: 600 !important;">Interpretación analítica de la comparación de estrategias:</strong></p>
                
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">Esta tabla sintetiza el desempeño de las distintas estrategias
                de construcción de portafolios bajo un enfoque riesgo–retorno,
                permitiendo una evaluación integral y comparativa.</p>
                
                <ul style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-left: 1.5rem !important; list-style-type: disc !important;">
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">La estrategia de <strong style="color: #1E88E5 !important; font-weight: 600 !important;">Sharpe Máximo</strong> tiende a ofrecer el mayor
                      retorno ajustado por riesgo, aunque suele presentar niveles
                      más elevados de volatilidad y drawdowns en periodos adversos.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">La estrategia de <strong style="color: #1E88E5 !important; font-weight: 600 !important;">Mínima Volatilidad</strong> prioriza la estabilidad
                      del capital, reduciendo la exposición a caídas pronunciadas,
                      a costa de un menor potencial de crecimiento.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">La estrategia de <strong style="color: #1E88E5 !important; font-weight: 600 !important;">Pesos Iguales</strong> actúa como referencia neutral,
                      proporcionando una diversificación básica sin optimización explícita.</li>
                </ul>
                
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">La combinación de métricas como retorno anual, volatilidad,
                Ratio de Sharpe y máximo drawdown permite identificar no solo
                la estrategia más rentable, sino también la más resiliente
                frente a escenarios de estrés de mercado.</p>
                
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">Este análisis respalda decisiones de asignación de activos
                alineadas con el horizonte temporal y el perfil de riesgo del inversor.</p>
            </div>
            """, unsafe_allow_html=True)

            # =====================================================================
            # 8.1) VOLATILIDAD HISTÓRICA ROLLING (RIESGO DINÁMICO)
            # =====================================================================
            st.subheader("Volatilidad histórica móvil")

            rolling_vol = pd.DataFrame({
                "Sharpe Máximo": daily_sharpe.rolling(252).std() * np.sqrt(252),
                "Mínima Volatilidad": daily_minvol.rolling(252).std() * np.sqrt(252),
                "Pesos Iguales": daily_equal.rolling(252).std() * np.sqrt(252)
            })

            st.line_chart(rolling_vol)

            st.markdown("""
            <div class="info-card">
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;"><strong style="color: #1E88E5 !important; font-weight: 600 !important;">Interpretación:</strong></p>
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">Esta gráfica muestra cómo el riesgo <strong style="color: #1E88E5 !important; font-weight: 600 !important;">cambia en el tiempo</strong>.</p>
                <ul style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-left: 1.5rem !important; list-style-type: disc !important;">
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Picos altos suelen coincidir con periodos de crisis.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Estrategias más estables presentan curvas más suaves.</li>
                </ul>
                
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">La volatilidad histórica móvil permite analizar cómo
                evoluciona el riesgo del portafolio a lo largo del tiempo,
                capturando cambios estructurales en el comportamiento del mercado.</p>
                
                <ul style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-left: 1.5rem !important; list-style-type: disc !important;">
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Incrementos abruptos de la volatilidad suelen coincidir
                      con periodos de crisis financiera o incertidumbre macroeconómica.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Curvas más suaves indican estrategias con mayor estabilidad
                      y menor sensibilidad a shocks de mercado.</li>
                </ul>
                
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">En el análisis comparativo:</p>
                <ul style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-left: 1.5rem !important; list-style-type: disc !important;">
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">El portafolio de <strong style="color: #1E88E5 !important; font-weight: 600 !important;">Sharpe Máximo</strong> presenta picos de
                      volatilidad más elevados, reflejando una mayor exposición
                      al riesgo en escenarios adversos.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">La estrategia de <strong style="color: #1E88E5 !important; font-weight: 600 !important;">Mínima Volatilidad</strong> mantiene un perfil
                      de riesgo más controlado a lo largo del tiempo.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">La asignación de <strong style="color: #1E88E5 !important; font-weight: 600 !important;">Pesos Iguales</strong> muestra un comportamiento
                      intermedio, replicando parcialmente la dinámica del mercado.</li>
                </ul>
                
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">Este enfoque dinámico del riesgo complementa las métricas
                estáticas tradicionales y aporta una visión más realista
                del comportamiento del portafolio.</p>
            </div>
            """, unsafe_allow_html=True)

            # =====================================================================
            # 8.2) RATIO CALMAR
            # =====================================================================
            calmar_sharpe = ret_sharpe / abs(dd_sharpe)
            calmar_minvol = ret_minvol / abs(dd_minvol)
            calmar_equal = ret_equal / abs(dd_equal)

            st.subheader("Ratio Calmar (retorno vs drawdown)")

            df_calmar = pd.DataFrame({
                "Estrategia": ["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"],
                "Calmar": [calmar_sharpe, calmar_minvol, calmar_equal]
            })

            st.dataframe(df_calmar)

            st.markdown("""
            <div class="info-card">
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;"><strong style="color: #1E88E5 !important; font-weight: 600 !important;">Interpretación analítica del Ratio Calmar:</strong></p>
                
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">El Ratio Calmar relaciona el <strong style="color: #1E88E5 !important; font-weight: 600 !important;">retorno anual esperado</strong> con el
                <strong style="color: #1E88E5 !important; font-weight: 600 !important;">máximo drawdown histórico</strong>, ofreciendo una medida directa
                de la capacidad del portafolio para generar rentabilidad
                sin incurrir en pérdidas extremas prolongadas.</p>
                
                <ul style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-left: 1.5rem !important; list-style-type: disc !important;">
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Un <strong style="color: #1E88E5 !important; font-weight: 600 !important;">Ratio Calmar elevado</strong> indica que la estrategia logra
                      retornos atractivos manteniendo caídas relativamente controladas.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Valores bajos sugieren que el retorno obtenido no compensa
                      adecuadamente las pérdidas máximas sufridas.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Esta métrica resulta especialmente relevante para
                      inversionistas con enfoque conservador o con restricciones
                      estrictas de preservación de capital.</li>
                </ul>
                
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">A diferencia del Ratio de Sharpe, el Calmar se centra en el
                <strong style="color: #1E88E5 !important; font-weight: 600 !important;">riesgo extremo observado</strong>, lo que lo convierte en un
                indicador complementario para evaluar la resiliencia del
                portafolio en periodos de crisis o alta volatilidad.</p>
                
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">En el contexto del presente análisis, el Ratio Calmar permite
                identificar qué estrategia ofrece un <strong style="color: #1E88E5 !important; font-weight: 600 !important;">mejor equilibrio entre
                crecimiento del capital y control de pérdidas severas</strong>,
                reforzando la robustez del proceso de selección de portafolios.</p>
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

            st.subheader("Ratio Sortino")

            df_sortino = pd.DataFrame({
                "Estrategia": ["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"],
                "Sortino": [sortino_sharpe, sortino_minvol, sortino_equal]
            })

            st.dataframe(df_sortino)

            st.markdown("""
            <div class="info-card">
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;"><strong style="color: #1E88E5 !important; font-weight: 600 !important;">Interpretación analítica del Ratio Sortino:</strong></p>
                
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">El Ratio Sortino evalúa el desempeño del portafolio considerando
                exclusivamente la <strong style="color: #1E88E5 !important; font-weight: 600 !important;">volatilidad negativa</strong>, es decir, aquellas
                fluctuaciones que representan pérdidas para el inversor.</p>
                
                <ul style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-left: 1.5rem !important; list-style-type: disc !important;">
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Un <strong style="color: #1E88E5 !important; font-weight: 600 !important;">valor más alto de Sortino</strong> indica que la estrategia genera
                      mayor retorno por cada unidad de riesgo a la baja asumida.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">A diferencia del Ratio de Sharpe, este indicador <strong style="color: #1E88E5 !important; font-weight: 600 !important;">no penaliza
                      la volatilidad positiva</strong>, lo que lo convierte en una métrica
                      más alineada con la percepción real del riesgo por parte del inversor.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Estrategias con Sortino elevado suelen ser más adecuadas para
                      escenarios de mercado inciertos o para perfiles que priorizan
                      la protección frente a caídas.</li>
                </ul>
                
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">En el contexto del análisis comparativo, el Ratio Sortino permite
                identificar qué estrategia ofrece una <strong style="color: #1E88E5 !important; font-weight: 600 !important;">mejor compensación entre
                retorno y riesgo negativo</strong>, aportando una visión complementaria
                y más conservadora al proceso de toma de decisiones.</p>
            </div>
            """, unsafe_allow_html=True)

            # =====================================================================
            # 8.4) PERIODOS DE CRISIS (COVID 2020)
            # =====================================================================
            st.subheader("Comportamiento en periodo de crisis (COVID 2020)")

            crisis = (cum_sharpe.index.year == 2020)

            st.line_chart(pd.DataFrame({
                "Sharpe Máximo": cum_sharpe[crisis],
                "Mínima Volatilidad": cum_minvol[crisis],
                "Pesos Iguales": cum_equal[crisis]
            }))

            st.markdown("""
            <div class="info-card">
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;"><strong style="color: #1E88E5 !important; font-weight: 600 !important;">Interpretación del comportamiento en periodo de crisis:</strong></p>
                
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">Esta visualización muestra el desempeño de las distintas
                estrategias durante un periodo de estrés sistémico,
                caracterizado por alta volatilidad y caídas abruptas del mercado.</p>
                
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">El análisis permite evaluar:</p>
                <ul style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-left: 1.5rem !important; list-style-type: disc !important;">
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">La <strong style="color: #1E88E5 !important; font-weight: 600 !important;">profundidad de la caída</strong> inicial (drawdown).</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">La <strong style="color: #1E88E5 !important; font-weight: 600 !important;">velocidad de recuperación</strong> tras el shock.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">La <strong style="color: #1E88E5 !important; font-weight: 600 !important;">resiliencia relativa</strong> de cada estrategia ante eventos extremos.</li>
                </ul>
                
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">Los resultados evidencian que:</p>
                <ul style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-left: 1.5rem !important; list-style-type: disc !important;">
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Las estrategias optimizadas para maximizar el retorno
                      (como Sharpe Máximo) tienden a experimentar caídas más
                      pronunciadas en el corto plazo.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Las estrategias orientadas a la reducción de riesgo
                      (Mínima Volatilidad) presentan una mayor capacidad de
                      contención de pérdidas.</li>
                </ul>
                
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">Este análisis refuerza la idea de que la eficiencia
                riesgo–retorno debe evaluarse no solo en condiciones normales,
                sino también bajo escenarios adversos.</p>
            </div>
            """, unsafe_allow_html=True)

            # =====================================================================
            # 8.5) COMPARACIÓN CON BENCHMARKS DE MERCADO
            # =====================================================================

            st.subheader("Comparación con benchmarks de mercado")

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
            st.dataframe(df_benchmarks)

            st.markdown("""
            <div class="info-card">
                <h3 style="margin-top: 1.5rem !important; margin-bottom: 1rem !important; color: #00ACC1 !important;">¿Qué es un benchmark?</h3>

                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">Un <strong style="color: #1E88E5 !important; font-weight: 600 !important;">benchmark</strong> es un <strong style="color: #1E88E5 !important; font-weight: 600 !important;">punto de referencia</strong> que se utiliza para evaluar si una estrategia de inversión es buena o mala.
                Funciona de forma similar a una <em>regla de medición</em>: permite comparar los resultados obtenidos con una alternativa estándar y ampliamente utilizada en los mercados financieros.</p>

                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">En este trabajo, los benchmarks representan <strong style="color: #1E88E5 !important; font-weight: 600 !important;">formas simples y comunes de invertir</strong>, frente a las cuales se comparan las estrategias optimizadas desarrolladas en la aplicación.</p>

                <h3 style="margin-top: 1.5rem !important; margin-bottom: 1rem !important; color: #00ACC1 !important;">¿Qué representa el S&P 500?</h3>

                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">El <strong style="color: #1E88E5 !important; font-weight: 600 !important;">S&P 500</strong> es uno de los índices bursátiles más conocidos del mundo. Agrupa a aproximadamente <strong style="color: #1E88E5 !important; font-weight: 600 !important;">500 de las empresas más grandes de Estados Unidos</strong>, como Apple, Microsoft o Google.
                Invertir en el S&P 500 se considera una aproximación al comportamiento general del mercado y suele utilizarse como referencia básica para evaluar el desempeño de cualquier portafolio.</p>

                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">Si una estrategia no logra superar al S&P 500 en el largo plazo, resulta difícil justificar su complejidad frente a una inversión pasiva en el mercado.</p>

                <h3 style="margin-top: 1.5rem !important; margin-bottom: 1rem !important; color: #00ACC1 !important;">¿Qué es el MSCI?</h3>

                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;"><strong style="color: #1E88E5 !important; font-weight: 600 !important;">MSCI</strong> (Morgan Stanley Capital International) es una empresa internacional que elabora <strong style="color: #1E88E5 !important; font-weight: 600 !important;">índices bursátiles</strong> utilizados como referencia en todo el mundo.
                Un índice MSCI representa el comportamiento de un conjunto amplio de empresas de una región o del mercado global.</p>

                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">Por ejemplo:</p>
                <ul style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-left: 1.5rem !important; list-style-type: disc !important;">
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;"><strong style="color: #1E88E5 !important; font-weight: 600 !important;">MSCI World</strong> agrupa empresas grandes y medianas de países desarrollados.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;"><strong style="color: #1E88E5 !important; font-weight: 600 !important;">MSCI Emerging Markets</strong> representa mercados emergentes.</li>
                </ul>

                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">Estos índices se utilizan como benchmark porque reflejan el desempeño promedio de mercados completos y permiten evaluar si una estrategia supera o no una inversión diversificada a nivel internacional.</p>

                <h3 style="margin-top: 1.5rem !important; margin-bottom: 1rem !important; color: #00ACC1 !important;">¿Qué es el NASDAQ?</h3>

                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">El <strong style="color: #1E88E5 !important; font-weight: 600 !important;">NASDAQ</strong> es una bolsa de valores estadounidense caracterizada por una <strong style="color: #1E88E5 !important; font-weight: 600 !important;">alta concentración de empresas tecnológicas y de innovación</strong>, como Apple, Microsoft, Amazon o Google.
                El índice NASDAQ suele mostrar mayores crecimientos en periodos de expansión económica, pero también presenta <strong style="color: #1E88E5 !important; font-weight: 600 !important;">mayor volatilidad</strong> en momentos de crisis.</p>

                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">Por esta razón, el NASDAQ se utiliza como benchmark para comparar estrategias con un perfil más dinámico y orientado al crecimiento, especialmente en sectores tecnológicos.</p>

                <h3 style="margin-top: 1.5rem !important; margin-bottom: 1rem !important; color: #00ACC1 !important;">¿Por qué se incluyen estos índices como benchmarks?</h3>

                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">La inclusión del <strong style="color: #1E88E5 !important; font-weight: 600 !important;">S&P 500, MSCI y NASDAQ</strong> permite comparar los portafolios optimizados con:</p>
                <ul style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-left: 1.5rem !important; list-style-type: disc !important;">
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">El comportamiento general del mercado estadounidense (S&P 500),</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Una referencia de diversificación global (MSCI),</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Un mercado de alto crecimiento y mayor riesgo (NASDAQ).</li>
                </ul>

                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">De esta forma, se obtiene una evaluación más completa del desempeño relativo de las estrategias desarrolladas en la aplicación.</p>

                <h3 style="margin-top: 1.5rem !important; margin-bottom: 1rem !important; color: #00ACC1 !important;">¿Por qué se comparan varias estrategias?</h3>

                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">Además del S&P 500, se incluyen otras estrategias como:</p>
                <ul style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-left: 1.5rem !important; list-style-type: disc !important;">
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;"><strong style="color: #1E88E5 !important; font-weight: 600 !important;">Pesos iguales</strong>, donde todos los activos reciben la misma proporción.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;"><strong style="color: #1E88E5 !important; font-weight: 600 !important;">Portafolio de mínima volatilidad</strong>, orientado a reducir el riesgo.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;"><strong style="color: #1E88E5 !important; font-weight: 600 !important;">Portafolio de Sharpe máximo</strong>, que busca el mejor retorno ajustado por riesgo.</li>
                </ul>

                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">La comparación con estos benchmarks permite responder una pregunta clave:
                <strong style="color: #1E88E5 !important; font-weight: 600 !important;">¿La optimización realmente mejora los resultados frente a alternativas simples y ampliamente utilizadas?</strong></p>
            </div>
            """, unsafe_allow_html=True)

            # =====================================================================
            # 8.6) RENDIMIENTO ACUMULADO: ESTRATEGIAS VS BENCHMARKS
            # =====================================================================

            st.subheader("Rendimiento acumulado: estrategias vs benchmarks")

            comparison_cum = pd.DataFrame({
                "Sharpe Máximo": cum_sharpe,
                "Mínima Volatilidad": cum_minvol,
                "Pesos Iguales": cum_equal,
                "S&P 500 (SPY)": benchmark_cum["SPY"],
                "Nasdaq 100 (QQQ)": benchmark_cum["QQQ"],
                "MSCI World (URTH)": benchmark_cum["URTH"]
            })

            st.line_chart(comparison_cum)

            st.markdown("""
            <div class="info-card">
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;"><strong style="color: #1E88E5 !important; font-weight: 600 !important;">Cómo interpretar la gráfica de rendimiento acumulado</strong></p>

                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">Esta gráfica muestra cómo habría evolucionado una inversión inicial a lo largo del tiempo bajo cada estrategia.</p>

                <ul style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-left: 1.5rem !important; list-style-type: disc !important;">
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">La línea que termina <strong style="color: #1E88E5 !important; font-weight: 600 !important;">más arriba</strong> representa la estrategia con <strong style="color: #1E88E5 !important; font-weight: 600 !important;">mayor crecimiento acumulado</strong>.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Las curvas más <strong style="color: #1E88E5 !important; font-weight: 600 !important;">suaves y estables</strong> indican menor volatilidad y menor exposición a crisis.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Caídas pronunciadas reflejan periodos de estrés de mercado; una recuperación rápida indica mayor resiliencia.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Si una estrategia optimizada supera de forma consistente a los benchmarks, se confirma que el modelo aporta valor frente a una inversión pasiva.</li>
                </ul>

                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">La interpretación conjunta del gráfico permite evaluar no solo cuánto se gana, sino <strong style="color: #1E88E5 !important; font-weight: 600 !important;">cómo se gana</strong>, identificando estrategias más robustas frente a escenarios adversos.</p>
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
            st.subheader("Rendimiento acumulado por acción")
            st.line_chart(cumulative_assets)

            st.subheader("Comparación de rendimientos de estrategias")
            st.line_chart(
                pd.DataFrame({
                    "Sharpe Máximo": cum_sharpe,
                    "Mínima Volatilidad": cum_minvol,
                    "Pesos Iguales": cum_equal
                })

            )

            st.markdown("""
            <div class="info-card">
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;"><strong style="color: #1E88E5 !important; font-weight: 600 !important;">Interpretación:</strong></p>

                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">El rendimiento acumulado refleja cómo habría evolucionado una inversión inicial
                en cada activo si se hubiera mantenido durante todo el periodo de análisis.</p>

                <ul style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-left: 1.5rem !important; list-style-type: disc !important;">
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Curvas más empinadas indican mayor crecimiento del capital.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Activos con mayor volatilidad suelen mostrar trayectorias más irregulares.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Diferencias significativas entre curvas evidencian distintos perfiles
                      de riesgo y rentabilidad.</li>
                </ul>

                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">Este gráfico facilita la comparación directa del desempeño histórico
                entre los activos analizados.</p>
            </div>
            """, unsafe_allow_html=True)

            # =====================================================================
            # GRÁFICO DE RETORNOS DIARIOS ACUMULADOS
            # =====================================================================
            st.subheader("Retornos diarios de los activos")
            st.line_chart(returns)

            st.markdown("""
            <div class="info-card">
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;"><strong style="color: #1E88E5 !important; font-weight: 600 !important;">Interpretación:</strong></p>

                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">Este gráfico muestra los retornos porcentuales diarios de cada activo,
                evidenciando la volatilidad de corto plazo.</p>

                <ul style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-left: 1.5rem !important; list-style-type: disc !important;">
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Picos positivos o negativos representan movimientos abruptos del mercado.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Mayor dispersión implica mayor riesgo.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Periodos de alta concentración de picos suelen coincidir con crisis financieras
                      o eventos macroeconómicos relevantes.</li>
                </ul>

                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">Este análisis es clave para evaluar el riesgo diario asumido por el inversor.</p>
            </div>
            """, unsafe_allow_html=True)

            # =====================================================================
            # GRÁFICO DE RETORNOS DIARIOS POR ACTIVO
            # =====================================================================

            st.subheader("Retornos diarios por activo")

            for ticker in returns.columns:
                  st.markdown(f"### {ticker}")
                  st.line_chart(returns[[ticker]])

            st.markdown("""
            <div class="info-card">
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;"><strong style="color: #1E88E5 !important; font-weight: 600 !important;">Interpretación:</strong></p>

                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">Este gráfico muestra el comportamiento diario del retorno del activo,
                permitiendo identificar:</p>

                <ul style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-left: 1.5rem !important; list-style-type: disc !important;">
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Frecuencia e intensidad de pérdidas y ganancias.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Presencia de volatilidad asimétrica (más caídas que subidas).</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Episodios de estrés específicos para el activo.</li>
                </ul>

                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">Resulta útil para evaluar el riesgo individual antes de integrarlo
                dentro de un portafolio diversificado.</p>
            </div>
            """, unsafe_allow_html=True)

            # =====================================================================
            # 11) FRONTERA EFICIENTE (GRÁFICO MÁS PEQUEÑO) - REDUCIDO A 5x3.5
            # =====================================================================
            st.subheader("Frontera eficiente (Retorno vs Volatilidad)")

            # GRÁFICO REDUCIDO - 4.5x2.5
            fig2, ax2 = plt.subplots(figsize=(4.5, 2.5))

            # Frontera eficiente
            ax2.plot(
                    efficient_vols,
                    efficient_rets,
                    linestyle="-",
                    linewidth=2,
                    label="Frontera eficiente"
            )
            # Portafolios destacados
            ax2.scatter(
                    vol_sharpe,
                    ret_sharpe,
                    s=70,
                    marker="o",
                    label="Sharpe Máximo"
            )

            ax2.scatter(
                    vol_minvol,
                    ret_minvol,
                    s=70,
                    marker="^",
                    label="Mínima Volatilidad"
            )
            ax2.scatter(
                    vol_equal,
                    ret_equal,
                    s=70,
                    marker="s",
                    label="Pesos Iguales"
            )
            # Etiquetas de los puntos con tamaño de fuente reducido
            ax2.annotate(
                    "Sharpe Máx",
                    (vol_sharpe, ret_sharpe),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontweight="bold",
                    fontsize=8
            )
            ax2.annotate(
                    "Mín Vol",
                    (vol_minvol, ret_minvol),
                    xytext=(5, -10),
                    textcoords="offset points",
                    fontweight="bold",
                    fontsize=8
            )
            ax2.annotate(
                    "Pesos Iguales",
                    (vol_equal, ret_equal),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontweight="bold",
                    fontsize=8
            )
            # Ejes y título con tamaño reducido
            ax2.set_xlabel("Volatilidad anual (riesgo)", fontsize=9)
            ax2.set_ylabel("Retorno anual esperado", fontsize=9)
            ax2.set_title("Frontera eficiente y estrategias", fontsize=10)
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(labelsize=8)
            plt.tight_layout()
            st.pyplot(fig2)

            st.markdown("""
            <div class="info-card">
                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;"><strong style="color: #1E88E5 !important; font-weight: 600 !important;">Interpretación analítica de la frontera eficiente:</strong></p>

                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">La frontera eficiente representa el conjunto de portafolios
                óptimos que maximizan el retorno esperado para cada nivel
                de riesgo asumido, de acuerdo con la teoría media–varianza
                de Markowitz.</p>

                <ul style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-left: 1.5rem !important; list-style-type: disc !important;">
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Cada punto de la curva corresponde a una combinación
                      distinta de activos que no puede ser mejorada simultáneamente
                      en términos de mayor retorno y menor riesgo.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">Los portafolios situados por debajo de la frontera son
                      ineficientes, ya que existe al menos una alternativa
                      con mejor desempeño riesgo–retorno.</li>
                </ul>

                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">La ubicación de las estrategias analizadas sobre la frontera
                permite identificar su perfil:</p>
                <ul style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-left: 1.5rem !important; list-style-type: disc !important;">
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">El portafolio de <strong style="color: #1E88E5 !important; font-weight: 600 !important;">Sharpe Máximo</strong> se sitúa en una zona de
                      mayor eficiencia, priorizando la rentabilidad ajustada
                      por riesgo.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">La estrategia de <strong style="color: #1E88E5 !important; font-weight: 600 !important;">Mínima Volatilidad</strong> se posiciona en el
                      extremo de menor riesgo, sacrificando retorno esperado.</li>
                    <li style="color: #E0E0E0 !important; margin-bottom: 0.5rem !important;">La asignación de <strong style="color: #1E88E5 !important; font-weight: 600 !important;">Pesos Iguales</strong> actúa como referencia
                      neutral, sin optimización explícita.</li>
                </ul>

                <p style="color: #E0E0E0 !important; line-height: 1.8 !important; margin-bottom: 1rem !important;">Esta visualización facilita la comprensión del trade-off
                riesgo–retorno y constituye una herramienta central para
                la toma de decisiones de inversión.</p>
            </div>
            """, unsafe_allow_html=True)

            # =====================================================================
            # INTERPRETACIÓN FINAL – COMPORTAMIENTO REAL PONDERADO EN EL TIEMPO
            # =====================================================================
            st.subheader("Interpretación automática del mejor portafolio")

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

            st.dataframe(weighted_performance.rename("Desempeño_Ponderado"))

            # Interpretación
            if best == "Pesos Iguales":
                st.markdown(
                    "### Mejor portafolio: Pesos Iguales\n\n"
                    "El análisis del **comportamiento real del portafolio en el tiempo**, "
                    "ponderando más los años recientes, muestra que esta estrategia ha sido "
                    "la **más robusta y consistente**.\n\n"
                    "- Menor dependencia de supuestos estadísticos.\n"
                    "- Mejor desempeño agregado a lo largo del tiempo.\n"
                    "- Alta estabilidad frente a cambios de mercado."
                )

            elif best == "Sharpe Máximo":
                st.markdown(
                    "### Mejor portafolio: Sharpe Máximo\n\n"
                    "La evaluación temporal indica que esta estrategia ofrece el mejor "
                    "equilibrio riesgo–retorno en el comportamiento histórico reciente."
                )

            else:
                st.markdown(
                    "### Mejor portafolio: Mínima Volatilidad\n\n"
                    "Esta estrategia destaca por su estabilidad, aunque sacrifica retorno "
                    "frente a las demás."
                )

            st.success(f"Portafolio recomendado según comportamiento real ponderado: {best}")

            # =====================================================================
            # 9) PESOS ÓPTIMOS (GRÁFICO MÁS PEQUEÑO) - REDUCIDO A 5x3
            # =====================================================================
            st.subheader("Pesos óptimos del portafolio recomendado")

            n_assets = len(tickers)

            if best == "Sharpe Máximo":
                final_weights = weights_sharpe
                metodo = "Optimización por Ratio de Sharpe"

            elif best == "Mínima Volatilidad":
                final_weights = weights_minvol
                metodo = "Optimización por Mínima Volatilidad"

            else:  # Pesos Iguales
                final_weights = np.array([1 / n_assets] * n_assets)
                metodo = "Asignación Equitativa (Pesos Iguales)"

            df_weights = pd.DataFrame({
                "Ticker": tickers,
                "Peso": final_weights,
                "Peso (%)": final_weights * 100
            })

            st.dataframe(df_weights)

            # --- Gráfico REDUCIDO - 4.5x2.5 ---
            fig, ax = plt.subplots(figsize=(4.5, 2.5))
            ax.barh(df_weights["Ticker"], df_weights["Peso"])
            ax.set_title(f"Composición del portafolio\n({metodo})", fontsize=10)
            ax.set_xlabel("Peso", fontsize=9)
            ax.tick_params(labelsize=8)
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown(f"""
            <div class="info-card">
                <h3>Interpretación de los pesos</h3>

                <p>Los pesos mostrados corresponden <strong>exclusivamente</strong> al portafolio
                recomendado por el modelo (<strong>{best}</strong>).</p>

                <ul>
                    <li>Cada peso indica qué proporción del capital debe asignarse a cada activo.</li>
                    <li>La suma total de los pesos es del <strong>100%</strong>.</li>
                    <li>Esta asignación refleja el comportamiento histórico del portafolio
                      bajo el criterio seleccionado.</li>
                </ul>

                <h3>Explicación extendida de los pesos óptimos</h3>

                <p>Los <strong>pesos óptimos</strong> indican cómo distribuir el capital para obtener
                el mejor balance entre <strong>riesgo y retorno</strong>, según el modelo de Markowitz.</p>

                <ul>
                    <li>Un <strong>peso del 40%</strong> significa que <strong>40 de cada 100 unidades monetarias</strong>
                      se asignan a ese activo.</li>
                    <li><strong>Pesos altos</strong> reflejan activos que aportan mayor eficiencia al portafolio.</li>
                    <li><strong>Pesos bajos</strong> indican activos que añaden más riesgo que beneficio relativo.</li>
                </ul>

                <p>Para personas sin experiencia previa,
                esta tabla funciona como una <strong>guía práctica de asignación de capital</strong>,
                evitando decisiones intuitivas o emocionales.</p>
            </div>
            """, unsafe_allow_html=True)

            st.session_state.analysis_done = True

            st.success("Análisis del portafolio ejecutado correctamente")

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
            st.error(f"Error: {e}")
# ======================================================
# MOSTRAR RESULTADOS (FUERA DEL BOTÓN)
# ======================================================

if st.session_state.analysis_done:
    results = st.session_state.analysis_results

    st.subheader("Comparación de estrategias")
    st.dataframe(results["comparison"])

    st.subheader("Pesos del portafolio recomendado")
    st.dataframe(results["weights_recommended"])

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

    st.subheader("Ratio / retorno esperado por estrategia")
    st.dataframe(df_retornos)

# ======================================================
# ASISTENTE INTELIGENTE DEL PORTAFOLIO (GEMINI)
# ======================================================

st.divider()
st.subheader("🤖 Asistente inteligente del portafolio")

if not st.session_state.analysis_done:
    st.info("Ejecuta primero la optimización para habilitar el asistente.")
else:
    import requests
    import os

    # =========================
    # CONFIGURACIÓN GEMINI
    # =========================
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        st.warning("El asistente requiere una API Key válida de Gemini.")
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
        "Pregunta sobre los tickers, riesgos o el portafolio recomendado"
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
""".format(
    tickers=", ".join(results["tickers"]),
    asset_text=asset_text,
    strategy_text=strategy_text,
    best_strategy=best_strategy,
    weights_text=weights_text
)
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


