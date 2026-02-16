import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime

# =========================
# CONFIGURACIÓN DE PÁGINA
# =========================
st.set_page_config(
    page_title="Portfolio Optimizer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CSS PROFESIONAL - SOLO DISEÑO
# =========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Fondo oscuro tecnológico */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1d3a 100%);
        font-family: 'Inter', sans-serif;
        color: #e2e8f0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1330 0%, #1a1f3a 100%);
        border-right: 1px solid #2d3748;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }
    
    /* Título principal con gradiente */
    h1 {
        background: linear-gradient(90deg, #00d4ff 0%, #7b2ff7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Subtítulos */
    h2, h3 {
        color: #00d4ff !important;
        font-weight: 600 !important;
    }
    
    /* Botones con gradiente */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Inputs modernos */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div {
        background-color: #1a1f3a !important;
        border: 2px solid #2d3748 !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }
    
    /* Slider */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #00d4ff 0%, #7b2ff7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a0aec0 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    
    /* DataFrames */
    [data-testid="stDataFrame"] {
        background-color: #1a1f3a;
        border-radius: 10px;
        border: 1px solid #2d3748;
    }
    
    /* Tablas */
    .dataframe {
        background-color: #1a1f3a !important;
        color: #e2e8f0 !important;
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 12px !important;
        border: none !important;
    }
    
    .dataframe tbody tr {
        background-color: #1a1f3a !important;
        border-bottom: 1px solid #2d3748 !important;
    }
    
    .dataframe tbody tr:hover {
        background-color: #252a40 !important;
    }
    
    /* Mensajes */
    .stSuccess {
        background-color: rgba(72, 187, 120, 0.1) !important;
        border-left: 4px solid #48bb78 !important;
        color: #9ae6b4 !important;
    }
    
    .stError {
        background-color: rgba(245, 101, 101, 0.1) !important;
        border-left: 4px solid #f56565 !important;
        color: #fc8181 !important;
    }
    
    .stInfo {
        background-color: rgba(66, 153, 225, 0.1) !important;
        border-left: 4px solid #4299e1 !important;
        color: #90cdf4 !important;
    }
    
    .stWarning {
        background-color: rgba(237, 137, 54, 0.1) !important;
        border-left: 4px solid #ed8936 !important;
        color: #fbd38d !important;
    }
    
    /* Gráficos matplotlib con fondo oscuro */
    .stPyplot {
        background-color: #1a1f3a;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #2d3748;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, #667eea 50%, transparent 100%);
        margin: 2rem 0;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1a1f3a !important;
        border-radius: 8px !important;
        border: 1px solid #2d3748 !important;
        color: #00d4ff !important;
    }
    
    /* Chat */
    [data-testid="stChatMessageContent"] {
        background-color: #1a1f3a !important;
        border-radius: 10px !important;
        border: 1px solid #2d3748 !important;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #e2e8f0;
    }
    
    /* Links */
    a {
        color: #00d4ff !important;
    }
    
    a:hover {
        color: #7b2ff7 !important;
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
# SIDEBAR - CONTROLES
# =========================
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    st.markdown("---")
    
    st.markdown("### 📊 Tickers")
    tickers_input = st.text_input(
        "Ingrese los tickers separados por comas",
        value="AAPL, MSFT, GOOGL",
        help="Use los códigos bursátiles oficiales. Separe cada ticker con una coma."
    )
    
    st.markdown("### 📅 Horizonte Temporal")
    years = st.slider(
        "Años de análisis",
        min_value=3,
        max_value=10,
        value=6
    )
    
    st.markdown("---")
    
    with st.expander("ℹ️ ¿Qué es un ticker?"):
        st.markdown("""
        Un **ticker** es el código con el que se identifica una acción en la bolsa de valores.
        
        **Ejemplos:**
        - **AAPL** → Apple Inc.
        - **MSFT** → Microsoft
        - **GOOGL** → Alphabet (Google)
        """)

# =========================
# HEADER
# =========================
col1, col2 = st.columns([4, 1])

with col1:
    st.title("📊 Optimización de Portafolios")
    st.markdown("**Modelo de Markowitz con IA**")

with col2:
    st.markdown(f"**📅 {datetime.now().strftime('%d/%m/%Y')}**")
    st.markdown(f"**🕐 {datetime.now().strftime('%H:%M')}**")

st.markdown("---")

# =========================
# BOTÓN DE ANÁLISIS
# =========================
if st.button("🚀 Ejecutar Optimización"):
    st.session_state.run_analysis = True
    st.session_state.analysis_done = False

# =========================
# AQUÍ VA TODO TU CÓDIGO ORIGINAL
# =========================
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

            st.subheader("📊 Precios ajustados depurados")
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
            st.subheader("📈 Precios relevantes del año 2025")
            precios_2025 = data[data.index.year == 2025].tail(10)
            if not precios_2025.empty:
                st.dataframe(precios_2025, use_container_width=True)
            else:
                st.info("No hay datos disponibles de 2025.")

            st.subheader(f"📊 Tendencia de precios (últimos {years} años)")
            st.line_chart(data)

            st.info("""
            **Interpretación:** Este gráfico muestra la evolución histórica de los precios ajustados.
            Las tendencias crecientes indican apreciación del activo.
            """)

            # =====================================================================
            # 8) COMPARACIÓN SISTEMÁTICA DE ESTRATEGIAS
            # =====================================================================
            st.subheader("🎯 Comparación sistemática de estrategias")

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
            **Cómo interpretar:**
            - **Retorno acumulado:** cuánto creció el capital total
            - **Volatilidad:** magnitud de las fluctuaciones (riesgo)
            - **Sharpe:** eficiencia riesgo–retorno
            - **Máx Drawdown:** peor caída histórica
            """)

            # =====================================================================
            # 8.1) VOLATILIDAD HISTÓRICA ROLLING
            # =====================================================================
            st.subheader("📉 Volatilidad histórica móvil")

            rolling_vol = pd.DataFrame({
                "Sharpe Máximo": daily_sharpe.rolling(252).std() * np.sqrt(252),
                "Mínima Volatilidad": daily_minvol.rolling(252).std() * np.sqrt(252),
                "Pesos Iguales": daily_equal.rolling(252).std() * np.sqrt(252)
            })

            st.line_chart(rolling_vol)

            # =====================================================================
            # 8.2) RATIO CALMAR
            # =====================================================================
            calmar_sharpe = ret_sharpe / abs(dd_sharpe)
            calmar_minvol = ret_minvol / abs(dd_minvol)
            calmar_equal = ret_equal / abs(dd_equal)

            st.subheader("💹 Ratio Calmar (retorno vs drawdown)")

            df_calmar = pd.DataFrame({
                "Estrategia": ["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"],
                "Calmar": [calmar_sharpe, calmar_minvol, calmar_equal]
            })

            st.dataframe(df_calmar, use_container_width=True)

            # =====================================================================
            # 8.3) SORTINO RATIO
            # =====================================================================
            downside = returns.copy()
            downside[downside > 0] = 0
            downside_std = downside.std() * np.sqrt(252)

            sortino_sharpe = ret_sharpe / downside_std.dot(weights_sharpe)
            sortino_minvol = ret_minvol / downside_std.dot(weights_minvol)
            sortino_equal = ret_equal / downside_std.dot(weights_equal)

            st.subheader("📊 Ratio Sortino")

            df_sortino = pd.DataFrame({
                "Estrategia": ["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"],
                "Sortino": [sortino_sharpe, sortino_minvol, sortino_equal]
            })

            st.dataframe(df_sortino, use_container_width=True)

            # =====================================================================
            # 8.4) PERIODOS DE CRISIS (COVID 2020)
            # =====================================================================
            st.subheader("⚠️ Comportamiento en periodo de crisis (COVID 2020)")

            crisis = (cum_sharpe.index.year == 2020)

            st.line_chart(pd.DataFrame({
                "Sharpe Máximo": cum_sharpe[crisis],
                "Mínima Volatilidad": cum_minvol[crisis],
                "Pesos Iguales": cum_equal[crisis]
            }))

            # =====================================================================
            # 8.5) COMPARACIÓN CON BENCHMARKS
            # =====================================================================

            st.subheader("📊 Comparación con benchmarks de mercado")

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

            with st.expander("ℹ️ ¿Qué es un benchmark?"):
                st.markdown("""
                Un **benchmark** es un punto de referencia para evaluar el desempeño de una estrategia.
                
                - **S&P 500:** Índice de las 500 empresas más grandes de EE.UU.
                - **NASDAQ:** Bolsa con alta concentración de empresas tecnológicas
                - **MSCI World:** Índice global de mercados desarrollados
                """)

            # =====================================================================
            # 8.6) RENDIMIENTO ACUMULADO VS BENCHMARKS
            # =====================================================================

            st.subheader("📈 Rendimiento acumulado: estrategias vs benchmarks")

            comparison_cum = pd.DataFrame({
                "Sharpe Máximo": cum_sharpe,
                "Mínima Volatilidad": cum_minvol,
                "Pesos Iguales": cum_equal,
                "S&P 500": benchmark_cum["SPY"],
                "Nasdaq 100": benchmark_cum["QQQ"],
                "MSCI World": benchmark_cum["URTH"]
            })

            st.line_chart(comparison_cum)

            # =====================================================================
            # 9) SÍNTESIS ANALÍTICA
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
            st.subheader("📊 Rendimiento acumulado por acción")
            st.line_chart(cumulative_assets)

            st.subheader("📈 Comparación de rendimientos de estrategias")
            st.line_chart(
                pd.DataFrame({
                    "Sharpe Máximo": cum_sharpe,
                    "Mínima Volatilidad": cum_minvol,
                    "Pesos Iguales": cum_equal
                })
            )

            # =====================================================================
            # GRÁFICO DE RETORNOS DIARIOS
            # =====================================================================
            st.subheader("📊 Retornos diarios de los activos")
            st.line_chart(returns)

            # =====================================================================
            # GRÁFICO DE RETORNOS DIARIOS POR ACTIVO
            # =====================================================================

            st.subheader("📉 Retornos diarios por activo")

            for ticker in returns.columns:
                  st.markdown(f"### {ticker}")
                  st.line_chart(returns[[ticker]])

            # =====================================================================
            # 11) FRONTERA EFICIENTE
            # =====================================================================
            st.subheader("🎯 Frontera eficiente (Retorno vs Volatilidad)")

            # Configurar matplotlib para fondo oscuro
            plt.style.use('dark_background')
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            fig2.patch.set_facecolor('#1a1f3a')
            ax2.set_facecolor('#1a1f3a')

            # Frontera eficiente
            ax2.plot(
                    efficient_vols,
                    efficient_rets,
                    linestyle="-",
                    linewidth=3,
                    label="Frontera eficiente",
                    color='#00d4ff'
            )
            
            # Portafolios destacados
            ax2.scatter(
                    vol_sharpe,
                    ret_sharpe,
                    s=150,
                    marker="o",
                    label="Sharpe Máximo",
                    color='#667eea',
                    edgecolors='white',
                    linewidths=2
            )

            ax2.scatter(
                    vol_minvol,
                    ret_minvol,
                    s=150,
                    marker="^",
                    label="Mínima Volatilidad",
                    color='#48bb78',
                    edgecolors='white',
                    linewidths=2
            )
            
            ax2.scatter(
                    vol_equal,
                    ret_equal,
                    s=150,
                    marker="s",
                    label="Pesos Iguales",
                    color='#ed8936',
                    edgecolors='white',
                    linewidths=2
            )
            
            # Etiquetas
            ax2.annotate(
                    "Sharpe Máx",
                    (vol_sharpe, ret_sharpe),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontweight="bold",
                    fontsize=10,
                    color='white'
            )
            ax2.annotate(
                    "Mín Vol",
                    (vol_minvol, ret_minvol),
                    xytext=(10, -15),
                    textcoords="offset points",
                    fontweight="bold",
                    fontsize=10,
                    color='white'
            )
            ax2.annotate(
                    "Pesos Iguales",
                    (vol_equal, ret_equal),
                    xytext=(10, 10),
                    textcoords="offset points",
                    fontweight="bold",
                    fontsize=10,
                    color='white'
            )
            
            ax2.set_xlabel("Volatilidad anual (riesgo)", fontsize=12, color='white')
            ax2.set_ylabel("Retorno anual esperado", fontsize=12, color='white')
            ax2.set_title("Frontera eficiente y estrategias", fontsize=14, color='white', pad=20)
            ax2.legend(fontsize=10, loc='best')
            ax2.grid(True, alpha=0.2, color='white')
            ax2.tick_params(colors='white')
            plt.tight_layout()
            st.pyplot(fig2)

            # =====================================================================
            # INTERPRETACIÓN FINAL
            # =====================================================================
            st.subheader("🎯 Interpretación automática del mejor portafolio")

            df_strategies = pd.DataFrame({
                "Sharpe Máximo": daily_sharpe,
                "Mínima Volatilidad": daily_minvol,
                "Pesos Iguales": daily_equal
            })

            # Ponderación temporal
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

            # Métricas del mejor
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🏆 Mejor Estrategia", best)
            with col2:
                st.metric("📈 Retorno Anual", f"{strategy_summary[best]['retorno']*100:.2f}%")
            with col3:
                st.metric("📊 Sharpe Ratio", f"{strategy_summary[best]['sharpe']:.2f}")

            # =====================================================================
            # 9) PESOS ÓPTIMOS
            # =====================================================================
            st.subheader("⚖️ Pesos óptimos del portafolio recomendado")

            n_assets = len(tickers)

            if best == "Sharpe Máximo":
                final_weights = weights_sharpe
                metodo = "Optimización por Ratio de Sharpe"

            elif best == "Mínima Volatilidad":
                final_weights = weights_minvol
                metodo = "Optimización por Mínima Volatilidad"

            else:
                final_weights = np.array([1 / n_assets] * n_assets)
                metodo = "Asignación Equitativa (Pesos Iguales)"

            df_weights = pd.DataFrame({
                "Ticker": tickers,
                "Peso": final_weights,
                "Peso (%)": final_weights * 100
            })

            st.dataframe(df_weights, use_container_width=True)

            # Gráfico de barras horizontal
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#1a1f3a')
            ax.set_facecolor('#1a1f3a')
            
            bars = ax.barh(df_weights["Ticker"], df_weights["Peso"], color='#667eea', edgecolor='white', linewidth=1.5)
            ax.set_title(f"Composición del portafolio\n({metodo})", fontsize=14, color='white', pad=20)
            ax.set_xlabel("Peso", fontsize=12, color='white')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.2, axis='x', color='white')
            plt.tight_layout()
            st.pyplot(fig)

            st.success(f"✅ Análisis del portafolio completado - Portafolio recomendado: **{best}**")

            st.session_state.analysis_done = True

            # ======================================================
            # GUARDAR RESULTADOS PARA EL CHAT
            # ======================================================
            st.session_state["analysis_results"] = {
                "tickers": tickers,
                "best": best,
                "comparison": df_compare,
                "weights_recommended": df_weights,
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

        except Exception as e:
            st.error(f"❌ Error: {e}")

# ======================================================
# MOSTRAR RESULTADOS (FUERA DEL BOTÓN)
# ======================================================

if st.session_state.analysis_done:
    results = st.session_state.analysis_results

    st.markdown("---")
    st.subheader("📊 Resumen de Estrategias")
    st.dataframe(results["comparison"], use_container_width=True)

    st.subheader("⚖️ Pesos del Portafolio Recomendado")
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

    st.subheader("📈 Ratio / Retorno Esperado por Estrategia")
    st.dataframe(df_retornos, use_container_width=True)

# ======================================================
# ASISTENTE INTELIGENTE DEL PORTAFOLIO (GEMINI)
# ======================================================

st.divider()
st.subheader("🤖 Asistente Inteligente del Portafolio")

if not st.session_state.analysis_done:
    st.info("💡 Ejecuta primero la optimización para habilitar el asistente.")
else:
    import requests
    import os

    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        st.warning("⚙️ El asistente requiere una API Key válida de Gemini.")
    else:
        MODEL = "gemini-2.0-flash-exp"
        GEMINI_URL = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{MODEL}:generateContent?key={GEMINI_API_KEY}"
        )

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

            system_prompt = f"""
Actúa como un analista financiero profesional.

CONTEXTO:
Activos analizados: {', '.join(results['tickers'])}

Resumen de activos:
{asset_text}

Resumen de estrategias:
{strategy_text}

Estrategia recomendada: {best_strategy}
Pesos del portafolio recomendado:
{weights_text}

INSTRUCCIONES:
- Responde de forma clara y concisa
- Usa lenguaje accesible
- Menciona riesgo y retorno cuando sea relevante
- No inventes datos
"""

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

            with st.spinner("Pensando..."):
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



