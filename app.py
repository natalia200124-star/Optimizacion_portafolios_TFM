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
    
    /* Efecto de brillo en hover */
    .glow-on-hover {
        position: relative;
        overflow: hidden;
    }
    
    .glow-on-hover::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 217, 255, 0.3), transparent);
        transition: left 0.5s;
    }
    
    .glow-on-hover:hover::before {
        left: 100%;
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
        "Años de análisis histórico",
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

            st.markdown("<h2>📈 Precios Ajustados Depurados</h2>", unsafe_allow_html=True)
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
            last_prices = data.iloc[-1]
            last_date = data.index[-1]
            first_prices = data.iloc[0]
            first_date = data.index[0]

            total_ret = (last_prices / first_prices) - 1
            years_elapsed = (last_date - first_date).days / 365.25
            cagr = (1 + total_ret) ** (1 / years_elapsed) - 1

            days_to_end_2025 = (
                datetime(2025, 12, 31) - last_date
            ).days

            projected_2025 = last_prices * ((1 + cagr) ** (days_to_end_2025 / 365.25))

            # =====================================================================
            # 8) BENCHMARK vs PORTAFOLIOS
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
            # 9) FIGURAS
            # =====================================================================
            # 9.1) Frontera eficiente
            fig1, ax1 = plt.subplots(figsize=(10, 6), facecolor='#1e2433')
            ax1.set_facecolor('#1e2433')
            ax1.plot(efficient_vols, efficient_rets, "c-", linewidth=2.5, label="Frontera Eficiente")
            ax1.scatter(vol_sharpe, ret_sharpe, marker="*", color="#00d9ff", s=500,
                       edgecolor="white", linewidth=2, label="Máx Sharpe", zorder=5)
            ax1.scatter(vol_minvol, ret_minvol, marker="^", color="#00ff88", s=300,
                       edgecolor="white", linewidth=2, label="Mín Volatilidad", zorder=5)
            ax1.scatter(vol_equal, ret_equal, marker="s", color="#ff6b6b", s=250,
                       edgecolor="white", linewidth=2, label="Pesos Iguales", zorder=5)
            ax1.set_xlabel("Volatilidad (σ)", fontsize=12, color='white', fontweight='bold')
            ax1.set_ylabel("Retorno esperado (μ)", fontsize=12, color='white', fontweight='bold')
            ax1.set_title("Frontera Eficiente de Markowitz", fontsize=14, color='#00d9ff', fontweight='bold')
            ax1.legend(facecolor='#252d3f', edgecolor='#00d9ff', framealpha=0.9, labelcolor='white')
            ax1.grid(True, alpha=0.2, color='#00d9ff')
            ax1.tick_params(colors='white')
            for spine in ax1.spines.values():
                spine.set_edgecolor('#00d9ff')
                spine.set_linewidth(2)

            # 9.2) Evolución de portafolios
            fig2, ax2 = plt.subplots(figsize=(12, 7), facecolor='#1e2433')
            ax2.set_facecolor('#1e2433')
            ax2.plot(cum_sharpe.index, cum_sharpe.values, label="Sharpe Máximo",
                    linewidth=2.5, color="#00d9ff")
            ax2.plot(cum_minvol.index, cum_minvol.values, label="Mín Volatilidad",
                    linewidth=2.5, color="#00ff88")
            ax2.plot(cum_equal.index, cum_equal.values, label="Pesos Iguales",
                    linewidth=2.5, color="#ff6b6b")
            ax2.set_xlabel("Fecha", fontsize=12, color='white', fontweight='bold')
            ax2.set_ylabel("Valor acumulado ($1 inicial)", fontsize=12, color='white', fontweight='bold')
            ax2.set_title("Evolución de Portafolios (Backtest)", fontsize=14, color='#00d9ff', fontweight='bold')
            ax2.legend(facecolor='#252d3f', edgecolor='#00d9ff', framealpha=0.9, labelcolor='white')
            ax2.grid(True, alpha=0.2, color='#00d9ff')
            ax2.tick_params(colors='white')
            for spine in ax2.spines.values():
                spine.set_edgecolor('#00d9ff')
                spine.set_linewidth(2)

            # 9.3) Evolución de activos individuales
            fig3, ax3 = plt.subplots(figsize=(12, 7), facecolor='#1e2433')
            ax3.set_facecolor('#1e2433')
            colors = ['#00d9ff', '#00ff88', '#ff6b6b', '#ffd700', '#ff00ff', '#00ffff']
            for i, ticker in enumerate(tickers):
                ax3.plot(cumulative_assets.index, cumulative_assets[ticker],
                        label=ticker, linewidth=2, color=colors[i % len(colors)])
            ax3.set_xlabel("Fecha", fontsize=12, color='white', fontweight='bold')
            ax3.set_ylabel("Valor acumulado ($1 inicial)", fontsize=12, color='white', fontweight='bold')
            ax3.set_title("Evolución de Activos Individuales", fontsize=14, color='#00d9ff', fontweight='bold')
            ax3.legend(facecolor='#252d3f', edgecolor='#00d9ff', framealpha=0.9, labelcolor='white')
            ax3.grid(True, alpha=0.2, color='#00d9ff')
            ax3.tick_params(colors='white')
            for spine in ax3.spines.values():
                spine.set_edgecolor('#00d9ff')
                spine.set_linewidth(2)

            # 9.4) Comparación con benchmarks
            fig4, ax4 = plt.subplots(figsize=(12, 7), facecolor='#1e2433')
            ax4.set_facecolor('#1e2433')
            ax4.plot(sharpe_subset.index, sharpe_subset, label="Sharpe Máximo",
                    linewidth=3, color="#00d9ff")
            ax4.plot(minvol_subset.index, minvol_subset, label="Mín Volatilidad",
                    linewidth=3, color="#00ff88")
            ax4.plot(equal_subset.index, equal_subset, label="Pesos Iguales",
                    linewidth=3, color="#ff6b6b")

            bench_colors = {"SPY": "#ffd700", "QQQ": "#ff00ff", "URTH": "#00ffff"}
            for col in benchmark_subset.columns:
                ax4.plot(benchmark_subset.index, benchmark_subset[col],
                        label=f"Benchmark {col}", linewidth=2, linestyle="--",
                        color=bench_colors.get(col, "#ffffff"), alpha=0.8)

            ax4.set_xlabel("Fecha", fontsize=12, color='white', fontweight='bold')
            ax4.set_ylabel("Valor acumulado ($1 inicial)", fontsize=12, color='white', fontweight='bold')
            ax4.set_title("Comparación: Portafolios vs Benchmarks de Mercado", fontsize=14, color='#00d9ff', fontweight='bold')
            ax4.legend(facecolor='#252d3f', edgecolor='#00d9ff', framealpha=0.9, labelcolor='white', fontsize=9)
            ax4.grid(True, alpha=0.2, color='#00d9ff')
            ax4.tick_params(colors='white')
            for spine in ax4.spines.values():
                spine.set_edgecolor('#00d9ff')
                spine.set_linewidth(2)

            # 9.5) Matriz de correlación
            fig5, ax5 = plt.subplots(figsize=(10, 8), facecolor='#1e2433')
            corr_matrix = returns.corr()
            im = ax5.imshow(corr_matrix, cmap="coolwarm", interpolation="nearest", vmin=-1, vmax=1)
            ax5.set_xticks(range(len(tickers)))
            ax5.set_yticks(range(len(tickers)))
            ax5.set_xticklabels(tickers, fontsize=10, color='white', fontweight='bold')
            ax5.set_yticklabels(tickers, fontsize=10, color='white', fontweight='bold')
            ax5.set_title("Matriz de Correlación entre Activos", fontsize=14, color='#00d9ff', fontweight='bold', pad=20)

            for i in range(len(tickers)):
                for j in range(len(tickers)):
                    val = corr_matrix.iloc[i, j]
                    color = 'white' if abs(val) < 0.5 else 'black'
                    ax5.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color=color, fontsize=10, fontweight='bold')

            cbar = plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(colors='white')
            cbar.set_label('Correlación', color='white', fontweight='bold')

            # =====================================================================
            # 10) TABLAS COMPARATIVAS
            # =====================================================================
            df_compare = pd.DataFrame({
                "Estrategia": ["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"],
                "Retorno anual": [ret_sharpe, ret_minvol, ret_equal],
                "Volatilidad anual": [vol_sharpe, vol_minvol, vol_equal],
                "Ratio de Sharpe": [sharpe_sharpe, sharpe_minvol, sharpe_equal],
                "Max Drawdown": [dd_sharpe, dd_minvol, dd_equal],
                "Valor final ($1)": [
                    cum_sharpe.iloc[-1],
                    cum_minvol.iloc[-1],
                    cum_equal.iloc[-1]
                ]
            })

            # =====================================================================
            # 11) RECOMENDACIÓN
            # =====================================================================
            best_idx = df_compare["Ratio de Sharpe"].idxmax()
            best_strategy = df_compare.loc[best_idx, "Estrategia"]

            if best_strategy == "Sharpe Máximo":
                recommended_weights = weights_sharpe
            elif best_strategy == "Mínima Volatilidad":
                recommended_weights = weights_minvol
            else:
                recommended_weights = weights_equal

            df_weights = pd.DataFrame({
                "Activo": tickers,
                "Peso (%)": [f"{w * 100:.2f}%" for w in recommended_weights]
            })

            # =====================================================================
            # 12) TABLA DE PROYECCIÓN DE PRECIOS A 2025
            # =====================================================================
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

            # =====================================================================
            # 13) COMPARACIÓN FINAL vs BENCHMARKS
            # =====================================================================
            df_vs_bench = pd.DataFrame({
                "Estrategia / Benchmark": list(final_strategies.index) + list(final_benchmarks.index),
                "Valor final ($1)": list(final_strategies.values) + list(final_benchmarks.values)
            }).sort_values("Valor final ($1)", ascending=False).reset_index(drop=True)

            # =====================================================================
            # 14) RESÚMENES PARA EL CHAT
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
            # 15) GUARDAR RESULTADOS EN SESSION STATE
            # =====================================================================
            st.session_state.analysis_done = True
            st.session_state.analysis_results = {
                # Información básica
                "tickers": tickers,
                "best": best_strategy,

                # Figuras
                "fig_efficient_frontier": fig1,
                "fig_portfolio_evolution": fig2,
                "fig_individual_assets": fig3,
                "fig_vs_benchmarks": fig4,
                "fig_correlation": fig5,

                # Tablas de pronóstico
                "forecast": df_forecast,

                # Comparación vs benchmarks
                "vs_benchmarks": df_vs_bench,

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
        st.error(f"❌ Error en el análisis: {e}")

# =========================
# MOSTRAR RESULTADOS
# =========================
if st.session_state.analysis_done:
    results = st.session_state.analysis_results
    
    st.success(f"✅ Análisis completado exitosamente. Estrategia recomendada: **{results['best']}**")
    
    # Sección de comparación de estrategias
    st.markdown("<h2>📊 Comparación de Estrategias</h2>", unsafe_allow_html=True)
    st.dataframe(results["comparison"], use_container_width=True)
    
    # Mostrar figuras en columnas
    st.markdown("<h2>📈 Visualizaciones de Análisis</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(results["fig_efficient_frontier"])
    with col2:
        st.pyplot(results["fig_correlation"])
    
    st.pyplot(results["fig_portfolio_evolution"])
    st.pyplot(results["fig_individual_assets"])
    st.pyplot(results["fig_vs_benchmarks"])
    
    # Pesos recomendados
    st.markdown("<h2>🎯 Portafolio Recomendado</h2>", unsafe_allow_html=True)
    st.dataframe(results["weights_recommended"], use_container_width=True)
    
    # Proyección de precios
    st.markdown("<h2>🔮 Proyección de Precios 2025</h2>", unsafe_allow_html=True)
    st.dataframe(results["forecast"], use_container_width=True)
    
    # Comparación con benchmarks
    st.markdown("<h2>🏆 Comparación vs Benchmarks de Mercado</h2>", unsafe_allow_html=True)
    st.dataframe(results["vs_benchmarks"], use_container_width=True)
    
    # Retornos esperados
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
    
    st.markdown("<h2>💰 Retorno Esperado por Estrategia</h2>", unsafe_allow_html=True)
    st.dataframe(df_retornos, use_container_width=True)

# =========================
# ASISTENTE INTELIGENTE
# =========================
st.divider()
st.markdown("<h2>🤖 Asistente Inteligente del Portafolio</h2>", unsafe_allow_html=True)

if not st.session_state.analysis_done:
    st.info("💡 Ejecuta primero la optimización para habilitar el asistente inteligente.")
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
        "💬 Pregunta sobre los tickers, riesgos o el portafolio recomendado"
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

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: #b0b8c1; font-size: 0.9rem;'>
    Desarrollado con 💙 usando Streamlit | Modelo de Markowitz para optimización de portafolios
</p>
""", unsafe_allow_html=True)


