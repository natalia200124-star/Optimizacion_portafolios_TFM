import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import requests
import os

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
# CSS MODERNO Y PROFESIONAL
# =========================
st.markdown("""
<style>
    /* Importar fuente */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Fondo general */
    .stApp {
        background: #0a0e27;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1330 0%, #1a1f3a 100%);
    }
    
    /* Título principal con efecto neón */
    h1 {
        background: linear-gradient(90deg, #00d4ff 0%, #7b2ff7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.5rem !important;
        letter-spacing: -0.02em;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
    }
    
    /* Botones */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff 0%, #7b2ff7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1a1f3a;
        border-radius: 10px;
        padding: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: #a0aec0;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Inputs */
    .stTextInput > div > div > input {
        background-color: #1a1f3a;
        border: 2px solid #2d3748;
        border-radius: 8px;
        color: white;
    }
    
    /* Dataframes */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR - CONTROLES
# =========================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/stocks-growth.png", width=80)
    st.title("⚙️ Configuración")
    
    st.markdown("---")
    
    # Input de tickers
    st.markdown("### 📈 Selección de Activos")
    tickers_input = st.text_input(
        "Tickers",
        value="AAPL, MSFT, GOOGL, AMZN",
        help="Ingrese los códigos de acciones separados por comas"
    )
    
    # Horizonte temporal
    st.markdown("### ⏱️ Horizonte Temporal")
    years = st.slider(
        "Años de análisis",
        min_value=1,
        max_value=10,
        value=5,
        help="Período de datos históricos"
    )
    
    # Opciones avanzadas
    with st.expander("🔧 Opciones Avanzadas"):
        risk_free_rate = st.number_input(
            "Tasa libre de riesgo (%)",
            value=3.0,
            min_value=0.0,
            max_value=10.0,
            step=0.1
        ) / 100
        
        rebalance_freq = st.selectbox(
            "Frecuencia de rebalanceo",
            ["Anual", "Trimestral", "Mensual"]
        )
    
    st.markdown("---")
    
    # Botón de análisis
    analyze_button = st.button("🚀 Ejecutar Análisis", use_container_width=True)

# =========================
# SESSION STATE
# =========================
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# =========================
# HEADER
# =========================
col1, col2 = st.columns([3, 1])
with col1:
    st.title("📊 Portfolio Optimizer Pro")
    st.markdown("**Optimización avanzada de portafolios con IA**")

with col2:
    st.markdown(f"**📅 {datetime.now().strftime('%d %b %Y')}**")
    st.markdown(f"**🕐 {datetime.now().strftime('%H:%M')}**")

st.markdown("---")

# =========================
# EJECUTAR ANÁLISIS
# =========================
if analyze_button:
    st.session_state.run_analysis = True
    st.session_state.analysis_done = False

if st.session_state.get("run_analysis", False) and not st.session_state.analysis_done:
    
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    if len(tickers) < 2:
        st.error("⚠️ Ingrese al menos 2 tickers")
        st.stop()
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Descargar datos
        status_text.text("⬇️ Descargando datos de mercado...")
        progress_bar.progress(20)
        
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
        
        data = data[tickers].ffill().dropna()
        
        progress_bar.progress(40)
        status_text.text("📊 Calculando retornos y correlaciones...")
        
        # Calcular retornos
        returns = data.pct_change().dropna()
        mean_returns_daily = returns.mean()
        cov_daily = returns.cov()
        
        trading_days = 252
        mean_returns_annual = mean_returns_daily * trading_days
        cov_annual = cov_daily * trading_days
        
        progress_bar.progress(60)
        status_text.text("🎯 Optimizando portafolios...")
        
        # Funciones de optimización
        def performance(weights, mean_ret, cov):
            ret = np.dot(weights, mean_ret)
            vol = np.sqrt(weights.T @ cov @ weights)
            sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
            return ret, vol, sharpe
        
        def neg_sharpe(weights):
            r, v, _ = performance(weights, mean_returns_annual, cov_annual)
            return -((r - risk_free_rate) / v) if v > 0 else 1e6
        
        def vol(weights):
            return np.sqrt(weights.T @ cov_annual @ weights)
        
        n = len(tickers)
        x0 = np.repeat(1 / n, n)
        bounds = tuple((0, 1) for _ in range(n))
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        
        # Optimizar
        res_sharpe = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        weights_sharpe = res_sharpe.x
        ret_sharpe, vol_sharpe, sharpe_sharpe = performance(weights_sharpe, mean_returns_annual, cov_annual)
        
        res_minvol = minimize(vol, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        weights_minvol = res_minvol.x
        ret_minvol, vol_minvol, sharpe_minvol = performance(weights_minvol, mean_returns_annual, cov_annual)
        
        weights_equal = np.repeat(1 / n, n)
        ret_equal, vol_equal, sharpe_equal = performance(weights_equal, mean_returns_annual, cov_annual)
        
        progress_bar.progress(80)
        status_text.text("📈 Generando visualizaciones...")
        
        # Calcular rendimientos
        daily_sharpe = returns.dot(weights_sharpe)
        daily_minvol = returns.dot(weights_minvol)
        daily_equal = returns.dot(weights_equal)
        
        cum_sharpe = (1 + daily_sharpe).cumprod()
        cum_minvol = (1 + daily_minvol).cumprod()
        cum_equal = (1 + daily_equal).cumprod()
        
        progress_bar.progress(100)
        status_text.text("✅ Análisis completado")
        
        # Guardar resultados
        st.session_state.analysis_results = {
            "tickers": tickers,
            "data": data,
            "returns": returns,
            "weights": {
                "Sharpe Máximo": weights_sharpe,
                "Mínima Volatilidad": weights_minvol,
                "Pesos Iguales": weights_equal
            },
            "performance": {
                "Sharpe Máximo": {"ret": ret_sharpe, "vol": vol_sharpe, "sharpe": sharpe_sharpe},
                "Mínima Volatilidad": {"ret": ret_minvol, "vol": vol_minvol, "sharpe": sharpe_minvol},
                "Pesos Iguales": {"ret": ret_equal, "vol": vol_equal, "sharpe": sharpe_equal}
            },
            "cumulative": {
                "Sharpe Máximo": cum_sharpe,
                "Mínima Volatilidad": cum_minvol,
                "Pesos Iguales": cum_equal
            }
        }
        
        st.session_state.analysis_done = True
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        progress_bar.empty()
        status_text.empty()

# =========================
# MOSTRAR RESULTADOS
# =========================
if st.session_state.analysis_done:
    results = st.session_state.analysis_results
    
    # KPIs principales
    st.markdown("### 📊 Métricas Clave")
    
    col1, col2, col3, col4 = st.columns(4)
    
    perf = results["performance"]["Sharpe Máximo"]
    
    with col1:
        st.metric(
            "Retorno Anual",
            f"{perf['ret']*100:.2f}%",
            delta=f"{(perf['ret'] - results['performance']['Pesos Iguales']['ret'])*100:.2f}%"
        )
    
    with col2:
        st.metric(
            "Volatilidad",
            f"{perf['vol']*100:.2f}%",
            delta=f"{(perf['vol'] - results['performance']['Pesos Iguales']['vol'])*100:.2f}%",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Ratio Sharpe",
            f"{perf['sharpe']:.2f}",
            delta=f"{(perf['sharpe'] - results['performance']['Pesos Iguales']['sharpe']):.2f}"
        )
    
    with col4:
        final_value = results["cumulative"]["Sharpe Máximo"].iloc[-1]
        st.metric(
            "Valor Final",
            f"${final_value:.2f}",
            delta=f"{(final_value - 1)*100:.1f}%"
        )
    
    st.markdown("---")
    
    # Tabs para organizar contenido
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Rendimiento", "🎯 Optimización", "📊 Análisis", "🤖 Asistente IA"])
    
    with tab1:
        st.markdown("### Evolución del Portafolio")
        
        # Gráfico interactivo de rendimiento
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=results["cumulative"]["Sharpe Máximo"].index,
            y=results["cumulative"]["Sharpe Máximo"].values,
            name="Sharpe Máximo",
            line=dict(color="#667eea", width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=results["cumulative"]["Mínima Volatilidad"].index,
            y=results["cumulative"]["Mínima Volatilidad"].values,
            name="Mínima Volatilidad",
            line=dict(color="#764ba2", width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=results["cumulative"]["Pesos Iguales"].index,
            y=results["cumulative"]["Pesos Iguales"].values,
            name="Pesos Iguales",
            line=dict(color="#48bb78", width=3)
        ))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=500,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparación de estrategias
        st.markdown("### Comparación de Estrategias")
        
        comparison_df = pd.DataFrame({
            "Estrategia": ["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"],
            "Retorno Anual": [
                results["performance"]["Sharpe Máximo"]["ret"],
                results["performance"]["Mínima Volatilidad"]["ret"],
                results["performance"]["Pesos Iguales"]["ret"]
            ],
            "Volatilidad": [
                results["performance"]["Sharpe Máximo"]["vol"],
                results["performance"]["Mínima Volatilidad"]["vol"],
                results["performance"]["Pesos Iguales"]["vol"]
            ],
            "Sharpe": [
                results["performance"]["Sharpe Máximo"]["sharpe"],
                results["performance"]["Mínima Volatilidad"]["sharpe"],
                results["performance"]["Pesos Iguales"]["sharpe"]
            ]
        })
        
        comparison_df["Retorno Anual"] = comparison_df["Retorno Anual"].apply(lambda x: f"{x*100:.2f}%")
        comparison_df["Volatilidad"] = comparison_df["Volatilidad"].apply(lambda x: f"{x*100:.2f}%")
        comparison_df["Sharpe"] = comparison_df["Sharpe"].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("### Composición del Portafolio Óptimo")
        
        # Gráfico de dona interactivo
        weights_df = pd.DataFrame({
            "Ticker": results["tickers"],
            "Peso": results["weights"]["Sharpe Máximo"] * 100
        })
        
        fig_pie = px.pie(
            weights_df,
            values="Peso",
            names="Ticker",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=500
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Tabla de pesos
        weights_df["Peso"] = weights_df["Peso"].apply(lambda x: f"{x:.2f}%")
        st.dataframe(weights_df, use_container_width=True, hide_index=True)
    
    with tab3:
        st.markdown("### Matriz de Correlación")
        
        # Heatmap de correlación
        corr_matrix = results["returns"].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        
        fig_corr.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=500
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Retornos individuales
        st.markdown("### Retornos Anualizados por Activo")
        
        returns_df = pd.DataFrame({
            "Ticker": results["tickers"],
            "Retorno Anual": (results["returns"].mean() * 252).values * 100
        }).sort_values("Retorno Anual", ascending=False)
        
        fig_bar = px.bar(
            returns_df,
            x="Ticker",
            y="Retorno Anual",
            color="Retorno Anual",
            color_continuous_scale="Viridis"
        )
        
        fig_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab4:
        st.markdown("### 🤖 Asistente Inteligente")
        
        st.info("💡 Pregúntame sobre el análisis del portafolio")
        
        # Chat con Gemini (si está configurado)
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        if GEMINI_API_KEY:
            user_question = st.text_input("Tu pregunta:", key="chat_input")
            
            if user_question:
                with st.spinner("Pensando..."):
                    # Preparar contexto
                    context = f"""
Activos: {', '.join(results['tickers'])}
Portafolio recomendado: Sharpe Máximo
Retorno anual: {results['performance']['Sharpe Máximo']['ret']*100:.2f}%
Volatilidad: {results['performance']['Sharpe Máximo']['vol']*100:.2f}%
Ratio Sharpe: {results['performance']['Sharpe Máximo']['sharpe']:.2f}
"""
                    
                    MODEL = "gemini-2.0-flash-exp"
                    GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={GEMINI_API_KEY}"
                    
                    payload = {
                        "contents": [{
                            "role": "user",
                            "parts": [{"text": f"{context}\n\nPregunta: {user_question}"}]
                        }]
                    }
                    
                    response = requests.post(GEMINI_URL, json=payload)
                    
                    if response.status_code == 200:
                        answer = response.json()["candidates"][0]["content"]["parts"][0]["text"]
                        st.markdown(f"**🤖 Respuesta:**\n\n{answer}")
                    else:
                        st.error("Error al generar respuesta")
        else:
            st.warning("⚙️ Configura GEMINI_API_KEY en secrets para usar el asistente")

else:
    # Pantalla inicial
    st.markdown("""
    <div style="text-align: center; padding: 100px 0;">
        <h2 style="color: #a0aec0;">👈 Configura los parámetros en el sidebar</h2>
        <p style="color: #718096; font-size: 1.2rem;">y presiona "Ejecutar Análisis" para comenzar</p>
    </div>
    """, unsafe_allow_html=True)



