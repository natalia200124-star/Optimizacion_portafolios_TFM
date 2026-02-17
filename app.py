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
    page_title="Optimización de Portafolios",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS PERSONALIZADO - DISEÑO PROFESIONAL Y TECNOLÓGICO
st.markdown("""
<style>
    /* Importar fuente moderna */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Estilos generales */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Fondo principal con gradiente oscuro */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 50%, #0f1419 100%);
    }
    
    /* Contenedor principal */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Títulos principales */
    h1 {
        color: #00d9ff !important;
        font-weight: 700 !important;
        font-size: 3rem !important;
        text-align: center;
        margin-bottom: 1rem !important;
        text-shadow: 0 0 20px rgba(0, 217, 255, 0.5);
        letter-spacing: -1px;
    }
    
    h2 {
        color: #00d9ff !important;
        font-weight: 600 !important;
        font-size: 1.8rem !important;
        margin-top: 2rem !important;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(0, 217, 255, 0.3);
    }
    
    h3 {
        color: #66d9ff !important;
        font-weight: 500 !important;
        font-size: 1.3rem !important;
    }
    
    /* Cards de información */
    .info-card {
        background: linear-gradient(145deg, #1e2433 0%, #252d3f 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(0, 217, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 217, 255, 0.5);
        box-shadow: 0 12px 40px rgba(0, 217, 255, 0.2);
    }
    
    /* Inputs personalizados */
    .stTextInput > div > div > input {
        background-color: #1e2433 !important;
        border: 2px solid rgba(0, 217, 255, 0.3) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        font-size: 1rem !important;
        padding: 0.75rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #00d9ff !important;
        box-shadow: 0 0 15px rgba(0, 217, 255, 0.3) !important;
    }
    
    /* Slider personalizado */
    .stSlider > div > div > div > div {
        background-color: #00d9ff !important;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(to right, rgba(0, 217, 255, 0.1), rgba(0, 217, 255, 0.3));
    }
    
    /* Botón principal */
    .stButton > button {
        background: linear-gradient(135deg, #00d9ff 0%, #0099cc 100%) !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 3rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 5px 20px rgba(0, 217, 255, 0.3) !important;
        width: 100%;
        margin-top: 1rem !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(0, 217, 255, 0.5) !important;
        background: linear-gradient(135deg, #00f0ff 0%, #00b8e6 100%) !important;
    }
    
    /* DataFrames */
    .dataframe {
        background-color: #1e2433 !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }
    
    .stDataFrame {
        background: linear-gradient(145deg, #1e2433 0%, #252d3f 100%);
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid rgba(0, 217, 255, 0.2);
    }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        color: #00d9ff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #b0b8c1 !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
    }
    
    /* Chat */
    .stChatMessage {
        background: linear-gradient(145deg, #1e2433 0%, #252d3f 100%) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(0, 217, 255, 0.2) !important;
        margin: 0.5rem 0 !important;
    }
    
    .stChatInputContainer {
        background-color: #1e2433 !important;
        border-radius: 15px !important;
        border: 2px solid rgba(0, 217, 255, 0.3) !important;
    }
    
    /* Expander personalizado */
    .streamlit-expanderHeader {
        background: linear-gradient(145deg, #1e2433 0%, #252d3f 100%) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(0, 217, 255, 0.2) !important;
        color: #00d9ff !important;
        font-weight: 500 !important;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: rgba(0, 217, 255, 0.5) !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(0, 217, 255, 0.3) !important;
        margin: 2rem 0 !important;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: rgba(0, 217, 255, 0.1) !important;
        border-left: 4px solid #00d9ff !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }
    
    /* Texto general */
    p, li, span, label {
        color: #e1e7ed !important;
    }
    
    /* Animación de entrada */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .main .block-container > div {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Scrollbar personalizada */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1f2e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00d9ff 0%, #0099cc 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #00f0ff;
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
# HEADER PRINCIPAL
# =========================
st.markdown("<h1>📊 Optimización de Portafolios – Modelo de Markowitz</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #b0b8c1; font-size: 1.2rem; margin-bottom: 2rem;'>Análisis cuantitativo avanzado para maximizar rendimientos y minimizar riesgos</p>", unsafe_allow_html=True)

# =========================
# SECCIÓN: ¿QUÉ ES UN TICKER?
# =========================
st.markdown("""
<div class="info-card">
    <h3>💡 ¿Qué es un ticker?</h3>
    <p style='font-size: 1.05rem; line-height: 1.6;'>
    Un <strong style='color: #00d9ff;'>ticker</strong> es el código con el que se identifica una acción en la bolsa de valores.
    Cada empresa cotizada tiene un ticker único que permite acceder a su información de mercado.
    </p>
    <div style='margin-top: 1rem; padding: 1rem; background-color: rgba(0, 217, 255, 0.05); border-radius: 8px; border-left: 3px solid #00d9ff;'>
        <p style='margin: 0; font-weight: 500;'>📌 Ejemplos comunes:</p>
        <ul style='margin-top: 0.5rem;'>
            <li><strong style='color: #00d9ff;'>AAPL</strong> → Apple Inc.</li>
            <li><strong style='color: #00d9ff;'>MSFT</strong> → Microsoft Corporation</li>
            <li><strong style='color: #00d9ff;'>GOOGL</strong> → Alphabet (Google)</li>
        </ul>
    </div>
    <p style='margin-top: 1rem; font-size: 0.95rem; color: #b0b8c1;'>
    Estos códigos se utilizan para descargar automáticamente los precios históricos
    y realizar el análisis financiero del portafolio.
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# INPUTS DE USUARIO
# =========================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<h3 style='margin-top: 1.5rem;'>🎯 Selección de Activos</h3>", unsafe_allow_html=True)
    tickers_input = st.text_input(
        "Ingrese los tickers separados por comas (ejemplo: AAPL, MSFT, GOOGL)",
        help="Use los códigos bursátiles oficiales. Separe cada ticker con una coma.",
        placeholder="AAPL, MSFT, GOOGL, TSLA"
    )

with col2:
    st.markdown("<h3 style='margin-top: 1.5rem;'>📅 Horizonte Temporal</h3>", unsafe_allow_html=True)
    years = st.slider(
        "Seleccione el horizonte temporal (años)",
        min_value=3,
        max_value=10,
        value=6,
        help="Cantidad de años de datos históricos para el análisis"
    )

# Botón de ejecución centrado
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    if st.button("🚀 Ejecutar Optimización"):
        st.session_state.run_analysis = True
        st.session_state.analysis_done = False

# =========================
# PROCESAMIENTO Y ANÁLISIS
# =========================
if st.session_state.run_analysis and not st.session_state.analysis_done:
    
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    if len(tickers) < 2:
        st.error("❌ Ingrese al menos 2 tickers para realizar el análisis.")
        st.stop()
    
    try:
        with st.spinner("⚙️ Procesando datos del mercado..."):
            # =====================================================================
            # 1) DESCARGA Y DEPURACIÓN DE DATOS
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

            data = raw_data["Adj Close"]

            if isinstance(data.columns, pd.MultiIndex):
                data = data.droplevel(0, axis=1)

            data = data[tickers]
            data = data.sort_index()
            data = data.ffill()
            data = data.dropna()

            st.markdown("<h2>📈 Precios Ajustados Depurados</h2>", unsafe_allow_html=True)
            st.dataframe(data.head(), use_container_width=True)

            # =====================================================================
            # 2) PRECIOS 2025 Y TENDENCIA
            # =====================================================================
            st.subheader("Precios relevantes del año 2025 (últimas 10 filas)")
            precios_2025 = data[data.index.year == 2025].tail(10)
            st.dataframe(precios_2025 if not precios_2025.empty else "No hay datos de 2025.", use_container_width=True)

            st.subheader(f"Tendencia de precios (últimos {years} años)")
            st.line_chart(data)

            with st.expander("📖 Ver interpretación de tendencia de precios"):
                st.markdown("""
                **Interpretación:**

                Este gráfico muestra la evolución histórica de los precios ajustados de cada activo
                durante el horizonte temporal seleccionado.

                - Tendencias crecientes indican periodos de apreciación del activo.
                - Periodos de alta pendiente reflejan fases de crecimiento acelerado.
                - Movimientos bruscos o caídas pronunciadas suelen asociarse a eventos de mercado
                  o episodios de alta volatilidad.

                Este análisis permite identificar activos con comportamientos más estables
                frente a otros con mayor variabilidad en el tiempo.
                """)

            # =====================================================================
            # 3) RETORNOS Y MATRICES
            # =====================================================================
            returns = data.pct_change().dropna()
            mean_returns_daily = returns.mean()
            cov_daily = returns.cov()

            trading_days = 252
            mean_returns_annual = mean_returns_daily * trading_days
            cov_annual = cov_daily * trading_days

            # =====================================================================
            # 4) FUNCIONES DE OPTIMIZACIÓN
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
            # 5) OPTIMIZACIONES
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
            # 6) RENDIMIENTOS DE CADA ESTRATEGIA
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
            # 7) DESCARGA DE BENCHMARKS DE MERCADO
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

            if isinstance(benchmark_data.columns, pd.MultiIndex):
                benchmark_data = benchmark_data.droplevel(0, axis=1)

            benchmark_data = benchmark_data.ffill().dropna()
            benchmark_returns = benchmark_data.pct_change().dropna()
            benchmark_cum = (1 + benchmark_returns).cumprod()

            # =====================================================================
            # 8) FRONTERA EFICIENTE
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
            # 9) PRECIOS PROYECTADOS 2025
            # =====================================================================
            last_prices = data.iloc[-1]
            last_date = data.index[-1]
            first_prices = data.iloc[0]
            first_date = data.index[0]

            total_ret = (last_prices / first_prices) - 1
            years_elapsed = (last_date - first_date).days / 365.25
            cagr = (1 + total_ret) ** (1 / years_elapsed) - 1

            days_to_end_2025 = (datetime(2025, 12, 31) - last_date).days
            projected_2025 = last_prices * ((1 + cagr) ** (days_to_end_2025 / 365.25))

            # =====================================================================
            # 10) BENCHMARK vs PORTAFOLIOS
            # =====================================================================
            dates_common = cum_sharpe.index.intersection(benchmark_cum.index)
            benchmark_subset = benchmark_cum.loc[dates_common]
            sharpe_subset = cum_sharpe.loc[dates_common]
            minvol_subset = cum_minvol.loc[dates_common]
            equal_subset = cum_equal.loc[dates_common]

            final_benchmarks = benchmark_subset.iloc[-1]
            final_strategies = pd.Series({
                "Sharpe Máximo": sharpe_subset.iloc[-1],
                "Mínima Volatilidad": minvol_subset.iloc[-1],
                "Pesos Iguales": equal_subset.iloc[-1]
            })

            # =====================================================================
            # 11) COMPARACIÓN SISTEMÁTICA DE ESTRATEGIAS
            # =====================================================================
            st.markdown("<h2>📊 Comparación Sistemática de Estrategias</h2>", unsafe_allow_html=True)

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

            with st.expander("📖 Ver interpretación de comparación de estrategias"):
                st.markdown("""
                **Cómo interpretar esta tabla:**
                - **Retorno acumulado:** cuánto creció el capital total en el periodo.
                - **Volatilidad:** magnitud de las fluctuaciones (riesgo).
                - **Sharpe:** eficiencia riesgo–retorno.
                - **Máx Drawdown:** peor caída histórica desde un máximo.

                **Interpretación analítica de la comparación de estrategias:**

                Esta tabla sintetiza el desempeño de las distintas estrategias
                de construcción de portafolios bajo un enfoque riesgo–retorno,
                permitiendo una evaluación integral y comparativa.

                - La estrategia de **Sharpe Máximo** tiende a ofrecer el mayor
                  retorno ajustado por riesgo, aunque suele presentar niveles
                  más elevados de volatilidad y drawdowns en periodos adversos.
                - La estrategia de **Mínima Volatilidad** prioriza la estabilidad
                  del capital, reduciendo la exposición a caídas pronunciadas,
                  a costa de un menor potencial de crecimiento.
                - La estrategia de **Pesos Iguales** actúa como referencia neutral,
                  proporcionando una diversificación básica sin optimización explícita.

                La combinación de métricas como retorno anual, volatilidad,
                Ratio de Sharpe y máximo drawdown permite identificar no solo
                la estrategia más rentable, sino también la más resiliente
                frente a escenarios de estrés de mercado.
                """)

            # =====================================================================
            # 12) VOLATILIDAD HISTÓRICA ROLLING
            # =====================================================================
            st.subheader("Volatilidad histórica móvil")

            rolling_vol = pd.DataFrame({
                "Sharpe Máximo": daily_sharpe.rolling(252).std() * np.sqrt(252),
                "Mínima Volatilidad": daily_minvol.rolling(252).std() * np.sqrt(252),
                "Pesos Iguales": daily_equal.rolling(252).std() * np.sqrt(252)
            })

            st.line_chart(rolling_vol)

            with st.expander("📖 Ver interpretación de volatilidad móvil"):
                st.markdown("""
                **Interpretación:**
                
                La volatilidad histórica móvil permite analizar cómo
                evoluciona el riesgo del portafolio a lo largo del tiempo,
                capturando cambios estructurales en el comportamiento del mercado.

                - Incrementos abruptos de la volatilidad suelen coincidir
                  con periodos de crisis financiera o incertidumbre macroeconómica.
                - Curvas más suaves indican estrategias con mayor estabilidad
                  y menor sensibilidad a shocks de mercado.

                En el análisis comparativo:
                - El portafolio de **Sharpe Máximo** presenta picos de
                  volatilidad más elevados, reflejando una mayor exposición
                  al riesgo en escenarios adversos.
                - La estrategia de **Mínima Volatilidad** mantiene un perfil
                  de riesgo más controlado a lo largo del tiempo.
                - La asignación de **Pesos Iguales** muestra un comportamiento
                  intermedio, replicando parcialmente la dinámica del mercado.
                """)

            # =====================================================================
            # 13) RATIO CALMAR
            # =====================================================================
            calmar_sharpe = ret_sharpe / abs(dd_sharpe)
            calmar_minvol = ret_minvol / abs(dd_minvol)
            calmar_equal = ret_equal / abs(dd_equal)

            st.subheader("Ratio Calmar (retorno vs drawdown)")

            df_calmar = pd.DataFrame({
                "Estrategia": ["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"],
                "Calmar": [calmar_sharpe, calmar_minvol, calmar_equal]
            })

            st.dataframe(df_calmar, use_container_width=True)

            with st.expander("📖 Ver interpretación del Ratio Calmar"):
                st.markdown("""
                **Interpretación analítica del Ratio Calmar:**

                El Ratio Calmar relaciona el **retorno anual esperado** con el
                **máximo drawdown histórico**, ofreciendo una medida directa
                de la capacidad del portafolio para generar rentabilidad
                sin incurrir en pérdidas extremas prolongadas.

                - Un **Ratio Calmar elevado** indica que la estrategia logra
                  retornos atractivos manteniendo caídas relativamente
                  controladas.
                - Valores bajos sugieren que el retorno obtenido no compensa
                  adecuadamente las pérdidas máximas sufridas.
                - Esta métrica resulta especialmente relevante para
                  inversionistas con enfoque conservador o con restricciones
                  estrictas de preservación de capital.

                A diferencia del Ratio de Sharpe, el Calmar se centra en el
                **riesgo extremo observado**, lo que lo convierte en un
                indicador complementario para evaluar la resiliencia del
                portafolio en periodos de crisis o alta volatilidad.
                """)

            # =====================================================================
            # 14) SORTINO RATIO
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

            st.dataframe(df_sortino, use_container_width=True)

            with st.expander("📖 Ver interpretación del Ratio Sortino"):
                st.markdown("""
                **Interpretación analítica del Ratio Sortino:**

                El Ratio Sortino evalúa el desempeño del portafolio considerando
                exclusivamente la **volatilidad negativa**, es decir, aquellas
                fluctuaciones que representan pérdidas para el inversor.

                - Un **valor más alto de Sortino** indica que la estrategia genera
                  mayor retorno por cada unidad de riesgo a la baja asumida.
                - A diferencia del Ratio de Sharpe, este indicador **no penaliza
                  la volatilidad positiva**, lo que lo convierte en una métrica
                  más alineada con la percepción real del riesgo por parte del inversor.
                - Estrategias con Sortino elevado suelen ser más adecuadas para
                  escenarios de mercado inciertos o para perfiles que priorizan
                  la protección frente a caídas.
                """)

            # =====================================================================
            # 15) PERIODOS DE CRISIS (COVID 2020)
            # =====================================================================
            st.subheader("Comportamiento en periodo de crisis (COVID 2020)")

            crisis = (cum_sharpe.index.year == 2020)

            st.line_chart(pd.DataFrame({
                "Sharpe Máximo": cum_sharpe[crisis],
                "Mínima Volatilidad": cum_minvol[crisis],
                "Pesos Iguales": cum_equal[crisis]
            }))

            with st.expander("📖 Ver interpretación del comportamiento en crisis"):
                st.markdown("""
                **Interpretación del comportamiento en periodo de crisis:**

                Esta visualización muestra el desempeño de las distintas
                estrategias durante un periodo de estrés sistémico,
                caracterizado por alta volatilidad y caídas abruptas del mercado.

                El análisis permite evaluar:
                - La **profundidad de la caída** inicial (drawdown).
                - La **velocidad de recuperación** tras el shock.
                - La **resiliencia relativa** de cada estrategia ante eventos extremos.

                Los resultados evidencian que:
                - Las estrategias optimizadas para maximizar el retorno
                  (como Sharpe Máximo) tienden a experimentar caídas más
                  pronunciadas en el corto plazo.
                - Las estrategias orientadas a la reducción de riesgo
                  (Mínima Volatilidad) presentan una mayor capacidad de
                  contención de pérdidas.
                """)

            # =====================================================================
            # 16) COMPARACIÓN CON BENCHMARKS DE MERCADO
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
            st.dataframe(df_benchmarks, use_container_width=True)

            with st.expander("📖 Ver información sobre benchmarks"):
                st.markdown("""
                ### ¿Qué es un benchmark?

                Un **benchmark** es un **punto de referencia** que se utiliza para evaluar si una estrategia de inversión es buena o mala.
                Funciona de forma similar a una *regla de medición*: permite comparar los resultados obtenidos con una alternativa estándar y ampliamente utilizada en los mercados financieros.

                ### ¿Qué representa el S&P 500?

                El **S&P 500** es uno de los índices bursátiles más conocidos del mundo. Agrupa a aproximadamente **500 de las empresas más grandes de Estados Unidos**, como Apple, Microsoft o Google.
                Invertir en el S&P 500 se considera una aproximación al comportamiento general del mercado y suele utilizarse como referencia básica para evaluar el desempeño de cualquier portafolio.

                ### ¿Qué es el MSCI?

                **MSCI** (Morgan Stanley Capital International) es una empresa internacional que elabora **índices bursátiles** utilizados como referencia en todo el mundo.
                Un índice MSCI representa el comportamiento de un conjunto amplio de empresas de una región o del mercado global.

                ### ¿Qué es el NASDAQ?

                El **NASDAQ** es una bolsa de valores estadounidense caracterizada por una **alta concentración de empresas tecnológicas y de innovación**, como Apple, Microsoft, Amazon o Google.
                El índice NASDAQ suele mostrar mayores crecimientos en periodos de expansión económica, pero también presenta **mayor volatilidad** en momentos de crisis.
                """)

            # =====================================================================
            # 17) RENDIMIENTO ACUMULADO: ESTRATEGIAS VS BENCHMARKS
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

            with st.expander("📖 Ver interpretación de rendimiento acumulado"):
                st.markdown("""
                **Cómo interpretar la gráfica de rendimiento acumulado**

                Esta gráfica muestra cómo habría evolucionado una inversión inicial a lo largo del tiempo bajo cada estrategia.

                - La línea que termina **más arriba** representa la estrategia con **mayor crecimiento acumulado**.
                - Las curvas más **suaves y estables** indican menor volatilidad y menor exposición a crisis.
                - Caídas pronunciadas reflejan periodos de estrés de mercado; una recuperación rápida indica mayor resiliencia.
                - Si una estrategia optimizada supera de forma consistente a los benchmarks, se confirma que el modelo aporta valor frente a una inversión pasiva.
                """)

            # =====================================================================
            # 18) RENDIMIENTOS ACUMULADOS
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

            with st.expander("📖 Ver interpretación de rendimientos acumulados"):
                st.markdown("""
                **Interpretación:**

                El rendimiento acumulado refleja cómo habría evolucionado una inversión inicial
                en cada activo si se hubiera mantenido durante todo el periodo de análisis.

                - Curvas más empinadas indican mayor crecimiento del capital.
                - Activos con mayor volatilidad suelen mostrar trayectorias más irregulares.
                - Diferencias significativas entre curvas evidencian distintos perfiles
                  de riesgo y rentabilidad.
                """)

            # =====================================================================
            # 19) RETORNOS DIARIOS
            # =====================================================================
            st.subheader("Retornos diarios de los activos")
            st.line_chart(returns)

            with st.expander("📖 Ver interpretación de retornos diarios"):
                st.markdown("""
                **Interpretación:**

                Este gráfico muestra los retornos porcentuales diarios de cada activo,
                evidenciando la volatilidad de corto plazo.

                - Picos positivos o negativos representan movimientos abruptos del mercado.
                - Mayor dispersión implica mayor riesgo.
                - Periodos de alta concentración de picos suelen coincidir con crisis financieras
                  o eventos macroeconómicos relevantes.
                """)

            # =====================================================================
            # 20) RETORNOS DIARIOS POR ACTIVO
            # =====================================================================
            st.subheader("Retornos diarios por activo")

            for ticker in returns.columns:
                st.markdown(f"### {ticker}")
                st.line_chart(returns[[ticker]])

            with st.expander("📖 Ver interpretación de retornos individuales"):
                st.markdown("""
                **Interpretación:**

                Este gráfico muestra el comportamiento diario del retorno del activo,
                permitiendo identificar:

                - Frecuencia e intensidad de pérdidas y ganancias.
                - Presencia de volatilidad asimétrica (más caídas que subidas).
                - Episodios de estrés específicos para el activo.

                Resulta útil para evaluar el riesgo individual antes de integrarlo
                dentro de un portafolio diversificado.
                """)

            # =====================================================================
            # 21) FRONTERA EFICIENTE (MEJORADA CON ETIQUETAS)
            # =====================================================================
            st.subheader("Frontera eficiente (Retorno vs Volatilidad)")

            fig2, ax2 = plt.subplots(figsize=(10, 6), facecolor='#1e2433')
            ax2.set_facecolor('#1e2433')

            ax2.plot(efficient_vols, efficient_rets, "c-", linewidth=2.5, label="Frontera Eficiente")
            ax2.scatter(vol_sharpe, ret_sharpe, marker="*", color="#00d9ff", s=500,
                       edgecolor="white", linewidth=2, label="Máx Sharpe", zorder=5)
            ax2.scatter(vol_minvol, ret_minvol, marker="^", color="#00ff88", s=300,
                       edgecolor="white", linewidth=2, label="Mín Volatilidad", zorder=5)
            ax2.scatter(vol_equal, ret_equal, marker="s", color="#ff6b6b", s=250,
                       edgecolor="white", linewidth=2, label="Pesos Iguales", zorder=5)

            ax2.annotate("Sharpe Máximo", (vol_sharpe, ret_sharpe), xytext=(8, 8),
                        textcoords="offset points", fontweight="bold", color='white')
            ax2.annotate("Mínima Volatilidad", (vol_minvol, ret_minvol), xytext=(8, -12),
                        textcoords="offset points", fontweight="bold", color='white')
            ax2.annotate("Pesos Iguales", (vol_equal, ret_equal), xytext=(8, 8),
                        textcoords="offset points", fontweight="bold", color='white')

            ax2.set_xlabel("Volatilidad anual (riesgo)", fontsize=12, color='white', fontweight='bold')
            ax2.set_ylabel("Retorno anual esperado", fontsize=12, color='white', fontweight='bold')
            ax2.set_title("Frontera Eficiente de Markowitz", fontsize=14, color='#00d9ff', fontweight='bold')
            ax2.legend(facecolor='#252d3f', edgecolor='#00d9ff', framealpha=0.9, labelcolor='white')
            ax2.grid(True, alpha=0.2, color='#00d9ff')
            ax2.tick_params(colors='white')
            for spine in ax2.spines.values():
                spine.set_edgecolor('#00d9ff')
                spine.set_linewidth(2)

            st.pyplot(fig2)

            with st.expander("📖 Ver interpretación de la frontera eficiente"):
                st.markdown("""
                **Interpretación analítica de la frontera eficiente:**

                La frontera eficiente representa el conjunto de portafolios
                óptimos que maximizan el retorno esperado para cada nivel
                de riesgo asumido, de acuerdo con la teoría media–varianza
                de Markowitz.

                - Cada punto de la curva corresponde a una combinación
                  distinta de activos que no puede ser mejorada simultáneamente
                  en términos de mayor retorno y menor riesgo.
                - Los portafolios situados por debajo de la frontera son
                  ineficientes, ya que existe al menos una alternativa
                  con mejor desempeño riesgo–retorno.

                La ubicación de las estrategias analizadas sobre la frontera
                permite identificar su perfil:
                - El portafolio de **Sharpe Máximo** se sitúa en una zona de
                  mayor eficiencia, priorizando la rentabilidad ajustada
                  por riesgo.
                - La estrategia de **Mínima Volatilidad** se posiciona en el
                  extremo de menor riesgo, sacrificando retorno esperado.
                - La asignación de **Pesos Iguales** actúa como referencia
                  neutral, sin optimización explícita.
                """)

            # =====================================================================
            # 22) INTERPRETACIÓN FINAL – MEJOR PORTAFOLIO
            # =====================================================================
            st.markdown("<h2>🎯 Interpretación Automática del Mejor Portafolio</h2>", unsafe_allow_html=True)

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

            st.success(f"✅ Portafolio recomendado según comportamiento real ponderado: **{best}**")

            # =====================================================================
            # 23) PESOS ÓPTIMOS SEGÚN PORTAFOLIO RECOMENDADO
            # =====================================================================
            st.markdown("<h2>💼 Pesos Óptimos del Portafolio Recomendado</h2>", unsafe_allow_html=True)

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

            st.dataframe(df_weights, use_container_width=True)

            # Gráfico de barras
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1e2433')
            ax.set_facecolor('#1e2433')
            ax.barh(df_weights["Ticker"], df_weights["Peso"], color='#00d9ff', edgecolor='white', linewidth=2)
            ax.set_title(f"Composición del portafolio recomendado\n({metodo})", 
                        fontsize=14, color='#00d9ff', fontweight='bold')
            ax.set_xlabel("Peso", fontsize=12, color='white', fontweight='bold')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.2, color='#00d9ff', axis='x')
            for spine in ax.spines.values():
                spine.set_edgecolor('#00d9ff')
                spine.set_linewidth(2)
            st.pyplot(fig)

            with st.expander("📖 Ver interpretación de los pesos"):
                st.markdown(f"""
                ### Interpretación de los pesos

                Los pesos mostrados corresponden **exclusivamente** al portafolio
                recomendado por el modelo (**{best}**).

                - Cada peso indica qué proporción del capital debe asignarse a cada activo.
                - La suma total de los pesos es del **100%**.
                - Esta asignación refleja el comportamiento histórico del portafolio
                  bajo el criterio seleccionado.

                ### Explicación extendida de los pesos óptimos

                Los **pesos óptimos** indican cómo distribuir el capital para obtener
                el mejor balance entre **riesgo y retorno**, según el modelo de Markowitz.

                - Un **peso del 40%** significa que **40 de cada 100 unidades monetarias**
                  se asignan a ese activo.
                - **Pesos altos** reflejan activos que aportan mayor eficiencia al portafolio.
                - **Pesos bajos** indican activos que añaden más riesgo que beneficio relativo.

                Para personas sin experiencia previa,
                esta tabla funciona como una **guía práctica de asignación de capital**,
                evitando decisiones intuitivas o emocionales.
                """)

            # =====================================================================
            # 24) TABLA DE PROYECCIÓN DE PRECIOS A 2025
            # =====================================================================
            st.markdown("<h2>🔮 Proyección de Precios 2025</h2>", unsafe_allow_html=True)
            
            df_forecast = pd.DataFrame({
                "Ticker": tickers,
                "Precio Actual": last_prices.values,
                "CAGR histórico": [f"{c * 100:.2f}%" for c in cagr.values],
                "Precio proyectado 2025": projected_2025.values,
                "Ganancia potencial": [
                    f"{((proj / curr) - 1) * 100:.2f}%"
                    for proj, curr in zip(projected_2025.values, last_prices.values)
                ]
            })

            st.dataframe(df_forecast, use_container_width=True)

            # =====================================================================
            # 25) COMPARACIÓN FINAL vs BENCHMARKS
            # =====================================================================
            st.markdown("<h2>🏆 Comparación vs Benchmarks de Mercado</h2>", unsafe_allow_html=True)
            
            df_vs_bench = pd.DataFrame({
                "Estrategia / Benchmark": list(final_strategies.index) + list(final_benchmarks.index),
                "Valor final ($1)": list(final_strategies.values) + list(final_benchmarks.values)
            }).sort_values("Valor final ($1)", ascending=False).reset_index(drop=True)

            st.dataframe(df_vs_bench, use_container_width=True)

            # =====================================================================
            # 26) RESÚMENES PARA EL CHAT
            # =====================================================================
            asset_summary = {}
            for ticker in tickers:
                asset_summary[ticker] = {
                    "retorno_anual": mean_returns_annual[ticker],
                    "volatilidad": np.sqrt(cov_annual.loc[ticker, ticker])
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
            # 27) GUARDAR RESULTADOS EN SESSION STATE
            # =====================================================================
            st.session_state.analysis_done = True
            st.session_state.analysis_results = {
                "tickers": tickers,
                "best": best,
                "comparison": df_compare,
                "weights_recommended": df_weights,
                "forecast": df_forecast,
                "vs_benchmarks": df_vs_bench,
                "weights": {
                    "Sharpe Máximo": dict(zip(tickers, weights_sharpe)),
                    "Mínima Volatilidad": dict(zip(tickers, weights_minvol)),
                    "Pesos Iguales": dict(zip(tickers, [1 / len(tickers)] * len(tickers)))
                },
                "retornos": {
                    "Sharpe Máximo": ret_sharpe,
                    "Mínima Volatilidad": ret_minvol,
                    "Pesos Iguales": ret_equal
                },
                "volatilidades": {
                    "Sharpe Máximo": vol_sharpe,
                    "Mínima Volatilidad": vol_minvol,
                    "Pesos Iguales": vol_equal
                },
                "asset_summary": asset_summary,
                "strategy_summary": strategy_summary
            }

            st.success("✅ Análisis del portafolio ejecutado correctamente")

    except Exception as e:
        st.error(f"❌ Error en el análisis: {e}")

# =========================
# ASISTENTE INTELIGENTE
# =========================
if st.session_state.analysis_done:
    st.divider()
    st.markdown("<h2>🤖 Asistente Inteligente del Portafolio</h2>", unsafe_allow_html=True)

    import requests
    import os

    # CONFIGURACIÓN GEMINI
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        st.warning("⚠️ El asistente requiere una API Key válida de Gemini.")
        st.stop()

    MODEL = "gemini-2.5-flash-lite"
    GEMINI_URL = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{MODEL}:generateContent?key={GEMINI_API_KEY}"
    )

    # HISTORIAL DE CHAT
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input(
        "💬 Pregunta sobre los tickers, riesgos o el portafolio recomendado"
    )

    if user_question:
        st.session_state.chat_messages.append(
            {"role": "user", "content": user_question}
        )

        results = st.session_state.analysis_results

        # CONTEXTO FINANCIERO
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

        # PROMPT OPTIMIZADO
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

        # LLAMADA A GEMINI
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

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: #b0b8c1; font-size: 0.9rem;'>
    Desarrollado con 💙 usando Streamlit | Modelo de Markowitz para optimización de portafolios
</p>
""", unsafe_allow_html=True)


