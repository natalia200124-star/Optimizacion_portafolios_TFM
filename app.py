import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import os
from sklearn.covariance import LedoitWolf  # MEJORA 1: covarianza robusta
from scipy.stats import gaussian_kde
from matplotlib.ticker import FuncFormatter

# ── Paleta de diseño oscuro (coherente con el tema de la app) ──────────────
_BG_DARK  = "#0f1419"
_BG_PANEL = "#1a1f2e"
_BG_CARD  = "#1e2433"
_CYAN     = "#00d9ff"
_GREEN    = "#66ff99"
_PURPLE   = "#b388ff"
_ORANGE   = "#ff9966"
_GRID     = "#2a3347"
_TEXT     = "#e1e7ed"
_DIM      = "#7a8499"
_PCT      = FuncFormatter(lambda x, _: f"{x:.0%}")

def _dark_ax(ax):
    """Aplica el tema oscuro de la app a un eje de matplotlib."""
    ax.set_facecolor(_BG_PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(_GRID)
    ax.tick_params(colors=_DIM, labelsize=9)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.grid(True, color=_GRID, linewidth=0.6, alpha=0.7)
    return ax

# =========================
# RISK_FREE_RATE fuera de la función cacheada
# =========================
RISK_FREE_RATE = 0.045  # T-Bill 3 meses ~4.5%

# =========================
# DISEÑO PROFESIONAL
# =========================
st.set_page_config(
    page_title="Optimización de Portafolios",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 50%, #0f1419 100%);
    }

    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }

    h1 {
        color: #00d9ff !important;
        font-weight: 700 !important;
        font-size: 2.8rem !important;
        text-align: center;
        text-shadow: 0 0 20px rgba(0, 217, 255, 0.5);
        letter-spacing: -1px;
    }

    h2, h3 {
        color: #66d9ff !important;
        font-weight: 600 !important;
    }

    .stTextInput > div > div > input {
        background-color: #1e2433 !important;
        border: 2px solid rgba(0, 217, 255, 0.3) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        font-size: 1rem !important;
        padding: 0.75rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #00d9ff !important;
        box-shadow: 0 0 15px rgba(0, 217, 255, 0.3) !important;
    }

    .stSlider > div > div > div > div {
        background-color: #00d9ff !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #00d9ff 0%, #0099cc 100%) !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 3rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 5px 20px rgba(0, 217, 255, 0.3) !important;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(0, 217, 255, 0.5) !important;
    }

    .stDataFrame {
        background: linear-gradient(145deg, #1e2433 0%, #252d3f 100%);
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid rgba(0, 217, 255, 0.2);
    }

    .streamlit-expanderHeader {
        background: linear-gradient(145deg, #1e2433 0%, #252d3f 100%) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(0, 217, 255, 0.25) !important;
        color: #00d9ff !important;
        font-weight: 600 !important;
    }
    .streamlit-expanderContent {
        background-color: #151c28 !important;
        border: 1px solid rgba(0, 217, 255, 0.15) !important;
        border-radius: 0 0 10px 10px !important;
    }

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

    .stAlert {
        background-color: rgba(0, 217, 255, 0.08) !important;
        border-left: 4px solid #00d9ff !important;
        border-radius: 10px !important;
    }

    hr {
        border-color: rgba(0, 217, 255, 0.3) !important;
        margin: 2rem 0 !important;
    }

    p, li, span, label { color: #e1e7ed !important; }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #1a1f2e; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00d9ff, #0099cc);
        border-radius: 5px;
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


@st.cache_data(show_spinner="Descargando datos y optimizando portafolio…")
def cargar_y_optimizar(tickers_tuple: tuple, years: int):

    tickers = list(tickers_tuple)

    # Parámetros de las mejoras
    LAMBDA_REG    = 0.01    # MEJORA 2: fuerza de regularización L2
    N_SIMULATIONS = 15000   # MEJORA 6: 15,000 trayectorias para reducir ruido estadístico
    N_BOOTSTRAP   = 15000   # MEJORA 5: muestras Bootstrap histórico
    MAX_WEIGHT    = 0.5     # MEJORA 8: límite máximo por activo (evita concentraciones irreales)

    # =====================================================================
    # 1.5) DESCARGA Y DEPURACIÓN DE DATOS (SIN LOOK-AHEAD BIAS)
    # =====================================================================
    end_date   = datetime.today()
    start_date = end_date.replace(year=end_date.year - years)

    benchmark_tickers = ["SPY", "QQQ", "URTH"]
    all_tickers = tickers + benchmark_tickers

    raw_data = yf.download(
        all_tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False
    )

    raw_data = raw_data["Adj Close"]

    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data = raw_data.droplevel(0, axis=1)

    raw_data = raw_data.sort_index()
    raw_data = raw_data.ffill()

    data           = raw_data[tickers].copy()
    benchmark_data = raw_data[benchmark_tickers].copy()

    tickers_invalidos = [t for t in tickers if data[t].isnull().mean() > 0.2]
    if tickers_invalidos:
        raise ValueError(
            f"Los siguientes tickers no tienen datos suficientes para el "
            f"periodo seleccionado: {', '.join(tickers_invalidos)}. "
            f"Elimínalos e intente de nuevo."
        )

    data           = data.dropna()
    benchmark_data = benchmark_data.ffill().dropna()

    if data.empty:
        raise ValueError("No hay datos suficientes para el periodo seleccionado.")

    # =====================================================================
    # 2) RETORNOS LOGARÍTMICOS Y MATRICES
    # MEJORA 4: retornos logarítmicos — aditivos en el tiempo, simétricos.
    # La acumulación correcta es np.exp(cumsum()), NO (1+r).cumprod().
    # =====================================================================
    returns            = np.log(data / data.shift(1)).dropna()
    mean_returns_daily = returns.mean()

    trading_days        = 252
    mean_returns_annual = mean_returns_daily * trading_days

    # MEJORA 1: Ledoit-Wolf Shrinkage (MANTENER — pilar de estabilidad)
    lw = LedoitWolf()
    lw.fit(returns)
    cov_daily  = pd.DataFrame(
        lw.covariance_,
        index=returns.columns,
        columns=returns.columns
    )
    cov_annual = cov_daily * trading_days

    # =====================================================================
    # 3) FUNCIONES DE OPTIMIZACIÓN
    # =====================================================================
    def performance(weights, mean_ret, cov):
        ret    = np.dot(weights, mean_ret)
        vol    = np.sqrt(weights.T @ cov @ weights)
        sharpe = (ret - RISK_FREE_RATE) / vol if vol > 0 else 0
        return ret, vol, sharpe

    # CORRECCIÓN 1: neg_sharpe con regularización L2 — simétrico con MinVol
    # Antes la penalización L2 solo se aplicaba en volatilidad, lo que
    # generaba una asimetría matemática entre las dos optimizaciones.
    def neg_sharpe(weights):
        ret_val, vol_val, sharpe = performance(weights, mean_returns_annual, cov_annual)
        penalty = LAMBDA_REG * np.sum(weights ** 2)
        return -(sharpe - penalty) if vol_val > 0 else 1e6

    # MEJORA 2: Regularización L2 en volatilidad
    def vol_obj(weights):
        variance = weights.T @ cov_annual @ weights
        penalty  = LAMBDA_REG * np.sum(weights ** 2)
        return np.sqrt(variance) + penalty

    def max_drawdown(series):
        cumulative_max = series.cummax()
        drawdown       = (series / cumulative_max) - 1
        return drawdown.min()

    n      = len(tickers)
    x0     = np.repeat(1 / n, n)

    # MEJORA 8: límite máximo por activo = 50% — evita concentraciones irreales
    bounds      = tuple((0, MAX_WEIGHT) for _ in range(n))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    # =====================================================================
    # 4) OPTIMIZACIONES
    # =====================================================================
    res_sharpe     = minimize(neg_sharpe, x0, method="SLSQP",
                              bounds=bounds, constraints=constraints)
    weights_sharpe = res_sharpe.x
    ret_sharpe, vol_sharpe, sharpe_sharpe = performance(
        weights_sharpe, mean_returns_annual, cov_annual
    )

    res_minvol     = minimize(vol_obj, x0, method="SLSQP",
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
    # 5) RENDIMIENTOS ACUMULADOS
    # CORRECCIÓN 2: Con log returns, acumulación = np.exp(cumsum())
    # =====================================================================
    cumulative_assets = np.exp(returns.cumsum())

    daily_sharpe = returns.dot(weights_sharpe)
    daily_minvol = returns.dot(weights_minvol)
    daily_equal  = returns.dot(weights_equal)

    cum_sharpe = np.exp(daily_sharpe.cumsum())
    cum_minvol = np.exp(daily_minvol.cumsum())
    cum_equal  = np.exp(daily_equal.cumsum())

    dd_sharpe = max_drawdown(cum_sharpe)
    dd_minvol = max_drawdown(cum_minvol)
    dd_equal  = max_drawdown(cum_equal)

    # =====================================================================
    # 5.1) BENCHMARKS DE MERCADO
    # =====================================================================
    benchmark_log_returns = np.log(
        benchmark_data / benchmark_data.shift(1)
    ).dropna()
    benchmark_cum = np.exp(benchmark_log_returns.cumsum())

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
        res = minimize(vol_obj, x0, method="SLSQP",
                       bounds=bounds, constraints=cons)
        if res.success:
            r_val, v_val, _ = performance(res.x, mean_returns_annual, cov_annual)
            efficient_rets.append(r_val)
            efficient_vols.append(v_val)

    # =====================================================================
    # 8) COMPARACIÓN SISTEMÁTICA DE ESTRATEGIAS
    # =====================================================================
    df_compare = pd.DataFrame({
        "Estrategia":       ["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"],
        "Retorno Anual":    [ret_sharpe, ret_minvol, ret_equal],
        "Volatilidad":      [vol_sharpe, vol_minvol, vol_equal],
        "Sharpe":           [sharpe_sharpe, sharpe_minvol, sharpe_equal],
        "Retorno Acumulado":[
            cum_sharpe.iloc[-1] - 1,
            cum_minvol.iloc[-1] - 1,
            cum_equal.iloc[-1]  - 1
        ],
        "Máx Drawdown": [dd_sharpe, dd_minvol, dd_equal]
    })

    # =====================================================================
    # 8.1) VOLATILIDAD HISTÓRICA ROLLING
    # =====================================================================
    rolling_vol = pd.DataFrame({
        "Sharpe Máximo":      daily_sharpe.rolling(252).std() * np.sqrt(252),
        "Mínima Volatilidad": daily_minvol.rolling(252).std() * np.sqrt(252),
        "Pesos Iguales":      daily_equal.rolling(252).std()  * np.sqrt(252)
    })

    # =====================================================================
    # 8.2) RATIO CALMAR
    # =====================================================================
    df_calmar = pd.DataFrame({
        "Estrategia": ["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"],
        "Calmar": [
            ret_sharpe / abs(dd_sharpe),
            ret_minvol / abs(dd_minvol),
            ret_equal  / abs(dd_equal)
        ]
    })

    # =====================================================================
    # 8.3) SORTINO RATIO
    # =====================================================================
    def sortino_ratio(ret_anual, daily_portfolio_returns):
        downside     = np.minimum(daily_portfolio_returns, 0)
        downside_dev = np.sqrt((downside ** 2).mean()) * np.sqrt(252)
        return (ret_anual - RISK_FREE_RATE) / downside_dev if downside_dev > 0 else np.nan

    df_sortino = pd.DataFrame({
        "Estrategia": ["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"],
        "Sortino": [
            sortino_ratio(ret_sharpe, daily_sharpe),
            sortino_ratio(ret_minvol, daily_minvol),
            sortino_ratio(ret_equal,  daily_equal)
        ]
    })

    # =====================================================================
    # 8.3.5a) SIMULACIÓN MONTE CARLO MULTIVARIADA — MEJORA 3 + 6
    # MEJORA 4 confirmada: simulación multivariada correcta con
    # np.random.multivariate_normal — NO activos por separado.
    # 15,000 trayectorias para reducir ruido estadístico (MEJORA 6).
    # =====================================================================
    np.random.seed(42)
    sim_assets = np.random.multivariate_normal(
        mean_returns_annual.values,
        cov_annual.values,
        N_SIMULATIONS
    )

    sim_sharpe = sim_assets @ weights_sharpe
    sim_minvol = sim_assets @ weights_minvol
    sim_equal  = sim_assets @ weights_equal

    def var_cvar(simulated, alpha=0.05):
        var          = np.percentile(simulated, alpha * 100)
        cvar         = simulated[simulated <= var].mean()
        prob_perdida = (simulated < 0).mean()
        return var, cvar, prob_perdida

    var_s, cvar_s, prob_s = var_cvar(sim_sharpe)
    var_m, cvar_m, prob_m = var_cvar(sim_minvol)
    var_e, cvar_e, prob_e = var_cvar(sim_equal)

    df_mc_stats = pd.DataFrame({
        "Estrategia":    ["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"],
        "VaR 95%":       [var_s, var_m, var_e],
        "CVaR 95%":      [cvar_s, cvar_m, cvar_e],
        "Prob. Pérdida": [prob_s, prob_m, prob_e]
    })

    # =====================================================================
    # 8.3.5b) BOOTSTRAP HISTÓRICO — MEJORA 5
    # Remuestreo con reemplazo de retornos reales diarios.
    # No asume distribución normal: captura asimetría y colas pesadas.
    # Complementa Monte Carlo para mayor credibilidad técnica.
    # =====================================================================
    np.random.seed(42)
    returns_array = returns.values  # shape: (T, n_assets)

    # Remuestrear filas (días) con reemplazo
    boot_indices = np.random.randint(0, len(returns_array), size=(N_BOOTSTRAP, trading_days))
    boot_samples = returns_array[boot_indices]  # shape: (N_BOOTSTRAP, 252, n_assets)

    # Retorno anual de cada trayectoria = suma de log returns diarios del año
    boot_annual = boot_samples.sum(axis=1)  # shape: (N_BOOTSTRAP, n_assets)

    boot_sharpe = boot_annual @ weights_sharpe
    boot_minvol = boot_annual @ weights_minvol
    boot_equal  = boot_annual @ weights_equal

    var_bs, cvar_bs, prob_bs = var_cvar(boot_sharpe)
    var_bm, cvar_bm, prob_bm = var_cvar(boot_minvol)
    var_be, cvar_be, prob_be = var_cvar(boot_equal)

    df_bootstrap_stats = pd.DataFrame({
        "Estrategia":    ["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"],
        "VaR 95% (Boot)":  [var_bs, var_bm, var_be],
        "CVaR 95% (Boot)": [cvar_bs, cvar_bm, cvar_be],
        "Prob. Pérdida (Boot)": [prob_bs, prob_bm, prob_be]
    })

    # =====================================================================
    # 8.5) TABLA BENCHMARKS
    # =====================================================================
    benchmarks = {
        "S&P 500 (SPY)":     "SPY",
        "Nasdaq 100 (QQQ)":  "QQQ",
        "MSCI World (URTH)": "URTH"
    }

    def annualized_return(daily_returns_series):
        return daily_returns_series.mean() * 252

    def annualized_vol_bm(series):
        return series.std() * np.sqrt(252)

    benchmark_summary = []
    for name, ticker in benchmarks.items():
        ret_bm = annualized_return(benchmark_log_returns[ticker])
        v_bm   = annualized_vol_bm(benchmark_log_returns[ticker])
        dd_bm  = max_drawdown(benchmark_cum[ticker])
        benchmark_summary.append({
            "Benchmark":         name,
            "Retorno Anual":     ret_bm,
            "Volatilidad":       v_bm,
            "Retorno Acumulado": benchmark_cum[ticker].iloc[-1] - 1,
            "Máx Drawdown":      dd_bm
        })
    df_benchmarks = pd.DataFrame(benchmark_summary)

    # =====================================================================
    # 8.6) RENDIMIENTO ACUMULADO COMPARADO CON BENCHMARKS
    # =====================================================================
    comparison_cum = pd.DataFrame({
        "Sharpe Máximo":      cum_sharpe,
        "Mínima Volatilidad": cum_minvol,
        "Pesos Iguales":      cum_equal,
        "S&P 500 (SPY)":      benchmark_cum["SPY"],
        "Nasdaq 100 (QQQ)":   benchmark_cum["QQQ"],
        "MSCI World (URTH)":  benchmark_cum["URTH"]
    })

    # =====================================================================
    # 9) MEJORA 9: ANÁLISIS DE ROBUSTEZ TEMPORAL
    # Optimización en sub-periodos (3, 5 y N años completos).
    # Muestra cuánto cambian los pesos — señal de estabilidad del modelo.
    # =====================================================================
    robustez_rows = []
    periodos_disponibles = []
    for p_years in [3, 5, years]:
        p_start = end_date.replace(year=end_date.year - p_years)
        if p_start >= data.index[0].to_pydatetime().replace(tzinfo=None):
            p_ret = returns[returns.index >= pd.Timestamp(p_start)]
            if len(p_ret) < 60:
                continue
            p_mean = p_ret.mean() * trading_days
            lw_p   = LedoitWolf()
            lw_p.fit(p_ret)
            p_cov  = pd.DataFrame(lw_p.covariance_, index=p_ret.columns, columns=p_ret.columns) * trading_days

            def neg_sharpe_p(w):
                r_p = np.dot(w, p_mean)
                v_p = np.sqrt(w.T @ p_cov.values @ w)
                sh  = (r_p - RISK_FREE_RATE) / v_p if v_p > 0 else 0
                pen = LAMBDA_REG * np.sum(w ** 2)
                return -(sh - pen)

            res_p = minimize(neg_sharpe_p, x0, method="SLSQP",
                             bounds=bounds, constraints=constraints)
            if res_p.success:
                row = {"Periodo": f"{p_years} años"}
                for t, w in zip(tickers, res_p.x):
                    row[t] = round(w, 4)
                robustez_rows.append(row)
                periodos_disponibles.append(p_years)

    df_robustez = pd.DataFrame(robustez_rows) if robustez_rows else pd.DataFrame()

    # =====================================================================
    # 9) SÍNTESIS ANALÍTICA PARA EL ASISTENTE
    # =====================================================================
    asset_summary = {}
    for ticker in tickers:
        asset_summary[ticker] = {
            "retorno_anual":       mean_returns_annual[ticker],
            "volatilidad":         np.sqrt(cov_annual.loc[ticker, ticker]),
            "contribucion_riesgo": cov_annual.loc[ticker].dot(weights_sharpe)
        }

    strategy_summary = {
        "Sharpe Máximo": {
            "retorno": ret_sharpe, "volatilidad": vol_sharpe,
            "sharpe": sharpe_sharpe, "drawdown": dd_sharpe
        },
        "Mínima Volatilidad": {
            "retorno": ret_minvol, "volatilidad": vol_minvol,
            "sharpe": sharpe_minvol, "drawdown": dd_minvol
        },
        "Pesos Iguales": {
            "retorno": ret_equal, "volatilidad": vol_equal,
            "sharpe": sharpe_equal, "drawdown": dd_equal
        }
    }

    df_strategies = pd.DataFrame({
        "Sharpe Máximo":      daily_sharpe,
        "Mínima Volatilidad": daily_minvol,
        "Pesos Iguales":      daily_equal
    })

    years_index  = df_strategies.index.year
    unique_years = np.sort(years_index.unique())
    year_weights = {
        year: (i + 1) / len(unique_years)
        for i, year in enumerate(unique_years)
    }
    weights_series = years_index.map(year_weights)

    weighted_performance = (
        np.exp(df_strategies.cumsum())
        .mul(weights_series, axis=0)
        .iloc[-1]
    )
    best = weighted_performance.idxmax()

    n_assets = len(tickers)
    if best == "Sharpe Máximo":
        final_weights = weights_sharpe
        metodo        = "Optimización por Ratio de Sharpe"
    elif best == "Mínima Volatilidad":
        final_weights = weights_minvol
        metodo        = "Optimización por Mínima Volatilidad"
    else:
        final_weights = np.array([1 / n_assets] * n_assets)
        metodo        = "Asignación Equitativa (Pesos Iguales)"

    df_weights = pd.DataFrame({
        "Ticker":   tickers,
        "Peso":     final_weights.round(2),
        "Peso (%)": (final_weights * 100).round(2)
    })

    return {
        "tickers":            tickers,
        "data":               data,
        "returns":            returns,
        "cumulative_assets":  cumulative_assets,
        "daily_sharpe":       daily_sharpe,
        "daily_minvol":       daily_minvol,
        "daily_equal":        daily_equal,
        "cum_sharpe":         cum_sharpe,
        "cum_minvol":         cum_minvol,
        "cum_equal":          cum_equal,
        "df_compare":         df_compare,
        "rolling_vol":        rolling_vol,
        "df_calmar":          df_calmar,
        "df_sortino":         df_sortino,
        "df_mc_stats":        df_mc_stats,
        "df_bootstrap_stats": df_bootstrap_stats,
        "df_robustez":        df_robustez,
        "mc_simulations": {
            "Sharpe Máximo":      sim_sharpe,
            "Mínima Volatilidad": sim_minvol,
            "Pesos Iguales":      sim_equal
        },
        "bootstrap_simulations": {
            "Sharpe Máximo":      boot_sharpe,
            "Mínima Volatilidad": boot_minvol,
            "Pesos Iguales":      boot_equal
        },
        "df_benchmarks":      df_benchmarks,
        "comparison_cum":     comparison_cum,
        "weighted_performance": weighted_performance,
        "best":               best,
        "metodo":             metodo,
        "df_weights":         df_weights,
        "efficient_vols":     efficient_vols,
        "efficient_rets":     efficient_rets,
        "vol_sharpe": vol_sharpe, "ret_sharpe": ret_sharpe,
        "vol_minvol": vol_minvol, "ret_minvol": ret_minvol,
        "vol_equal":  vol_equal,  "ret_equal":  ret_equal,
        "asset_summary":    asset_summary,
        "strategy_summary": strategy_summary,
        "weights": {
            "Sharpe Máximo":      dict(zip(tickers, weights_sharpe)),
            "Mínima Volatilidad": dict(zip(tickers, weights_minvol)),
            "Pesos Iguales":      dict(zip(tickers, [1 / len(tickers)] * len(tickers)))
        },
        "retornos": {
            "Sharpe Máximo":      ret_sharpe,
            "Mínima Volatilidad": ret_minvol,
            "Pesos Iguales":      ret_equal
        },
        "volatilidades": {
            "Sharpe Máximo":      vol_sharpe,
            "Mínima Volatilidad": vol_minvol,
            "Pesos Iguales":      vol_equal
        }
    }


# =========================
# TÍTULO E INSTRUCCIONES
# =========================
st.title("Optimización de Portafolios – Modelo de Markowitz")

st.markdown("""
### ¿Qué es un ticker?

Un **ticker** es el código con el que se identifica una acción en la bolsa de valores.
Cada empresa cotizada tiene un ticker único que permite acceder a su información de mercado.

**Ejemplos comunes:**
- **AAPL** → Apple Inc.
- **MSFT** → Microsoft Corporation
- **GOOGL** → Alphabet (Google)

Estos códigos se utilizan para descargar automáticamente los precios históricos
y realizar el análisis financiero del portafolio.
""")

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
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if len(tickers) < 2:
        st.error("Ingrese al menos 2 tickers.")
    else:
        try:
            resultado = cargar_y_optimizar(tuple(tickers), years)
            st.session_state.analysis_results = resultado
            st.session_state.analysis_done    = True
            st.session_state.chat_messages    = []
            st.rerun()
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Error: {e}")

if st.session_state.analysis_done:

    r = st.session_state.analysis_results

    data         = r["data"]
    returns      = r["returns"]
    tickers      = r["tickers"]
    cum_sharpe   = r["cum_sharpe"]
    cum_minvol   = r["cum_minvol"]
    cum_equal    = r["cum_equal"]
    daily_sharpe = r["daily_sharpe"]
    daily_minvol = r["daily_minvol"]
    daily_equal  = r["daily_equal"]
    best         = r["best"]
    metodo       = r["metodo"]

    st.subheader("Precios ajustados depurados (primeras filas)")
    st.dataframe(data.head())

    # =====================================================================
    # 7) PRECIOS 2025 Y TENDENCIA
    # =====================================================================
    idx = data.index.tz_localize(None) if getattr(data.index, "tz", None) else data.index
    precios_2025 = data[idx.year == 2025].tail(10)

    if precios_2025.empty:
        st.info("No hay datos disponibles para 2025.")
    else:
        st.dataframe(precios_2025, use_container_width=True)

    st.subheader(f"Tendencia de precios (últimos {years} años)")
    st.line_chart(data)

    with st.expander("📖 Interpretación – Tendencia de precios"):
        st.markdown(
            """
            **Interpretación:**

            Este gráfico muestra la evolución histórica de los precios ajustados de cada activo
            durante el horizonte temporal seleccionado.

            - Tendencias crecientes indican periodos de apreciación del activo.
            - Periodos de alta pendiente reflejan fases de crecimiento acelerado.
            - Movimientos bruscos o caídas pronunciadas suelen asociarse a eventos de mercado
              o episodios de alta volatilidad.

            Este análisis permite identificar activos con comportamientos más estables
            frente a otros con mayor variabilidad en el tiempo.
            """
        )

    # =====================================================================
    # 8) COMPARACIÓN SISTEMÁTICA DE ESTRATEGIAS
    # =====================================================================
    st.subheader("Comparación sistemática de estrategias")
    st.dataframe(r["df_compare"])

    with st.expander("📖 Interpretación – Comparación de estrategias"):
        st.markdown(
            """
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

            Este análisis respalda decisiones de asignación de activos
            alineadas con el horizonte temporal y el perfil de riesgo del inversor.
            """
        )

    # =====================================================================
    # 8.1) VOLATILIDAD HISTÓRICA ROLLING
    # =====================================================================
    st.subheader("Volatilidad histórica móvil")
    st.line_chart(r["rolling_vol"])

    with st.expander("📖 Interpretación – Volatilidad histórica móvil"):
        st.markdown(
            """
            **Interpretación:**
            Esta gráfica muestra cómo el riesgo **cambia en el tiempo**.
            - Picos altos suelen coincidir con periodos de crisis.
            - Estrategias más estables presentan curvas más suaves.

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

            Este enfoque dinámico del riesgo complementa las métricas
            estáticas tradicionales y aporta una visión más realista
            del comportamiento del portafolio.
            """
        )

    # =====================================================================
    # 8.2) RATIO CALMAR
    # =====================================================================
    st.subheader("Ratio Calmar (retorno vs drawdown)")
    st.dataframe(r["df_calmar"])

    with st.expander("📖 Interpretación – Ratio Calmar"):
        st.markdown(
            """
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

            En el contexto del presente análisis, el Ratio Calmar permite
            identificar qué estrategia ofrece un **mejor equilibrio entre
            crecimiento del capital y control de pérdidas severas**,
            reforzando la robustez del proceso de selección de portafolios.
            """
        )

    # =====================================================================
    # 8.3) SORTINO RATIO
    # =====================================================================
    st.subheader("Ratio Sortino")
    st.dataframe(r["df_sortino"])

    with st.expander("📖 Interpretación – Ratio Sortino"):
        st.markdown(
            """
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

            En el contexto del análisis comparativo, el Ratio Sortino permite
            identificar qué estrategia ofrece una **mejor compensación entre
            retorno y riesgo negativo**, aportando una visión complementaria
            y más conservadora al proceso de toma de decisiones.
            """
        )

    # =====================================================================
    # 8.3.5) MONTE CARLO + BOOTSTRAP — panel lado a lado
    # =====================================================================
    st.subheader("Análisis de Riesgo Forward-Looking: Monte Carlo vs Bootstrap Histórico")

    # Tablas de métricas
    _t1, _t2 = st.columns(2)
    with _t1:
        st.caption("📊 Monte Carlo (distribución normal multivariada)")
        st.dataframe(r["df_mc_stats"], use_container_width=True)
    with _t2:
        st.caption("📊 Bootstrap Histórico (retornos reales sin supuesto de normalidad)")
        st.dataframe(r["df_bootstrap_stats"], use_container_width=True)

    # ── Gráfica combinada lado a lado ────────────────────────────────────
    def _kde_panel(ax, simulations_dict, title, subtitle):
        palette = [_CYAN, _GREEN, _PURPLE]
        for (name, sims), col in zip(simulations_dict.items(), palette):
            kde = gaussian_kde(sims, bw_method=0.18)
            xs  = np.linspace(
                min(sims.min(), -0.5),
                max(sims.max(),  0.7),
                500
            )
            ys = kde(xs)
            ax.fill_between(xs, ys, alpha=0.18, color=col)
            ax.plot(xs, ys, color=col, linewidth=2.2, label=name)
            var95   = np.percentile(sims, 5)
            xs_tail = xs[xs <= var95]
            ax.fill_between(xs_tail, kde(xs_tail), alpha=0.50, color=col)
            ax.axvline(var95, color=col, linewidth=1.0, linestyle=":")

        ax.axvline(0, color="white", linewidth=1.8,
                   linestyle="--", alpha=0.55, label="Retorno = 0%")

        ylim = ax.get_ylim()
        ax.text(ax.get_xlim()[0] + 0.02, ylim[1] * 0.88,
                "◀ Área sombreada\n   = Cola VaR 95%",
                color=_DIM, fontsize=7.5, style="italic")

        ax.xaxis.set_major_formatter(_PCT)
        ax.set_xlabel("Retorno anual simulado", fontsize=9)
        ax.set_ylabel("Densidad de probabilidad", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold", color=_CYAN, pad=10)
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7.5, color=_DIM)
        ax.legend(facecolor=_BG_CARD, edgecolor=_GRID,
                  labelcolor=_TEXT, fontsize=8)

    fig_dual, (ax_mc, ax_bt) = plt.subplots(
        1, 2, figsize=(13, 4.5), facecolor=_BG_DARK
    )
    _dark_ax(ax_mc)
    _dark_ax(ax_bt)

    _kde_panel(ax_mc, r["mc_simulations"],
               "Monte Carlo",
               "15,000 simulaciones · Asume distribución normal multivariada")
    _kde_panel(ax_bt, r["bootstrap_simulations"],
               "Bootstrap Histórico",
               "15,000 muestras · Usa retornos reales · Sin supuesto de normalidad")

    fig_dual.suptitle(
        "Distribución de Retornos Anuales Simulados por Estrategia",
        fontsize=12, fontweight="bold", color=_TEXT, y=1.02
    )
    fig_dual.tight_layout()
    st.pyplot(fig_dual)
    plt.close(fig_dual)

    with st.expander("📖 Interpretación – Monte Carlo vs Bootstrap Histórico"):
        st.markdown(
        """
        **¿Qué muestra esta gráfica?**

        Cada curva representa la distribución de posibles retornos anuales de una estrategia,
        estimada con dos métodos distintos que se muestran lado a lado para facilitar la comparación.

        **Panel izquierdo – Monte Carlo:**
        Genera 15,000 escenarios asumiendo que los retornos siguen una distribución normal
        multivariada. Es el método estándar en finanzas cuantitativas.

        **Panel derecho – Bootstrap Histórico:**
        Remuestrea directamente los retornos diarios reales, sin imponer ningún supuesto
        de distribución. Captura mejor los eventos extremos (crisis, crashes) y las colas
        pesadas que la distribución normal subestima.

        **Cómo leer las curvas:**
        - Curvas desplazadas a la **derecha** → mayor retorno esperado.
        - Curvas más **estrechas** → menor volatilidad, resultados más predecibles.
        - El **área sombreada** en cada curva representa la cola del VaR 95%: los peores escenarios
          (5% de probabilidad). Cuanto mayor sea esta zona a la izquierda del cero, mayor el riesgo extremo.
        - La **línea punteada vertical** marca el retorno = 0%. Todo lo que queda a su izquierda
          representa pérdida para el inversor.

        **Lectura comparativa entre paneles:**
        - Si ambos métodos muestran distribuciones similares → la distribución normal
          es una buena aproximación para ese portafolio.
        - Si el Bootstrap muestra colas más pesadas o mayor asimetría → los retornos históricos
          tienen comportamientos extremos que Monte Carlo no captura bien.
        """
        )

    # =====================================================================
    # 8.4) PERIODOS DE CRISIS (COVID 2020)
    # =====================================================================
    st.subheader("Comportamiento en periodo de crisis (COVID 2020)")

    crisis = (cum_sharpe.index.year == 2020)
    st.line_chart(pd.DataFrame({
        "Sharpe Máximo":      cum_sharpe[crisis],
        "Mínima Volatilidad": cum_minvol[crisis],
        "Pesos Iguales":      cum_equal[crisis]
    }))

    with st.expander("📖 Interpretación – Comportamiento en crisis (COVID 2020)"):
        st.markdown(
            """
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

            Este análisis refuerza la idea de que la eficiencia
            riesgo–retorno debe evaluarse no solo en condiciones normales,
            sino también bajo escenarios adversos.
            """
        )

    # =====================================================================
    # 8.5) COMPARACIÓN CON BENCHMARKS DE MERCADO
    # =====================================================================
    st.subheader("Comparación con benchmarks de mercado")
    st.dataframe(r["df_benchmarks"])

    with st.expander("📖 ¿Qué es un benchmark? – S&P 500, MSCI y NASDAQ explicados"):
        st.markdown("""
        ### ¿Qué es un benchmark?

        Un **benchmark** es un **punto de referencia** que se utiliza para evaluar si una estrategia de inversión es buena o mala.
        Funciona de forma similar a una *regla de medición*: permite comparar los resultados obtenidos con una alternativa estándar y ampliamente utilizada en los mercados financieros.

        En este trabajo, los benchmarks representan **formas simples y comunes de invertir**, frente a las cuales se comparan las estrategias optimizadas desarrolladas en la aplicación.

        ### ¿Qué representa el S&P 500?

        El **S&P 500** es uno de los índices bursátiles más conocidos del mundo. Agrupa a aproximadamente **500 de las empresas más grandes de Estados Unidos**, como Apple, Microsoft o Google.
        Invertir en el S&P 500 se considera una aproximación al comportamiento general del mercado y suele utilizarse como referencia básica para evaluar el desempeño de cualquier portafolio.

        Si una estrategia no logra superar al S&P 500 en el largo plazo, resulta difícil justificar su complejidad frente a una inversión pasiva en el mercado.

        ### ¿Qué es el MSCI?

        **MSCI** (Morgan Stanley Capital International) es una empresa internacional que elabora **índices bursátiles** utilizados como referencia en todo el mundo.
        Un índice MSCI representa el comportamiento de un conjunto amplio de empresas de una región o del mercado global.

        Por ejemplo:
        - **MSCI World** agrupa empresas grandes y medianas de países desarrollados.
        - **MSCI Emerging Markets** representa mercados emergentes.

        Estos índices se utilizan como benchmark porque reflejan el desempeño promedio de mercados completos y permiten evaluar si una estrategia supera o no una inversión diversificada a nivel internacional.

        ### ¿Qué es el NASDAQ?

        El **NASDAQ** es una bolsa de valores estadounidense caracterizada por una **alta concentración de empresas tecnológicas y de innovación**, como Apple, Microsoft, Amazon o Google.
        El índice NASDAQ suele mostrar mayores crecimientos en periodos de expansión económica, pero también presenta **mayor volatilidad** en momentos de crisis.

        Por esta razón, el NASDAQ se utiliza como benchmark para comparar estrategias con un perfil más dinámico y orientado al crecimiento, especialmente en sectores tecnológicos.

        ### ¿Por qué se incluyen estos índices como benchmarks?

        La inclusión del **S&P 500, MSCI y NASDAQ** permite comparar los portafolios optimizados con:
        - El comportamiento general del mercado estadounidense (S&P 500),
        - Una referencia de diversificación global (MSCI),
        - Un mercado de alto crecimiento y mayor riesgo (NASDAQ).

        De esta forma, se obtiene una evaluación más completa del desempeño relativo de las estrategias desarrolladas en la aplicación.

        ### ¿Por qué se comparan varias estrategias?

        Además del S&P 500, se incluyen otras estrategias como:
        - **Pesos iguales**, donde todos los activos reciben la misma proporción.
        - **Portafolio de mínima volatilidad**, orientado a reducir el riesgo.
        - **Portafolio de Sharpe máximo**, que busca el mejor retorno ajustado por riesgo.

        La comparación con estos benchmarks permite responder una pregunta clave:
        **¿La optimización realmente mejora los resultados frente a alternativas simples y ampliamente utilizadas?**
        """)

    # =====================================================================
    # 8.6) RENDIMIENTO ACUMULADO: ESTRATEGIAS VS BENCHMARKS
    # =====================================================================
    st.subheader("Rendimiento acumulado: estrategias vs benchmarks")
    st.line_chart(r["comparison_cum"])

    with st.expander("📖 Interpretación – Rendimiento acumulado vs benchmarks"):
        st.markdown("""
        **Cómo interpretar la gráfica de rendimiento acumulado**

        Esta gráfica muestra cómo habría evolucionado una inversión inicial a lo largo del tiempo bajo cada estrategia.

        - La línea que termina **más arriba** representa la estrategia con **mayor crecimiento acumulado**.
        - Las curvas más **suaves y estables** indican menor volatilidad y menor exposición a crisis.
        - Caídas pronunciadas reflejan periodos de estrés de mercado; una recuperación rápida indica mayor resiliencia.
        - Si una estrategia optimizada supera de forma consistente a los benchmarks, se confirma que el modelo aporta valor frente a una inversión pasiva.

        La interpretación conjunta del gráfico permite evaluar no solo cuánto se gana, sino **cómo se gana**, identificando estrategias más robustas frente a escenarios adversos.
        """)

    # =====================================================================
    # 8.7) ANÁLISIS DE ROBUSTEZ TEMPORAL — MEJORA 9
    # =====================================================================
    if not r["df_robustez"].empty:
        st.subheader("Robustez temporal – Estabilidad de pesos por sub-periodo")
        st.dataframe(r["df_robustez"])

        with st.expander("📖 Interpretación – Análisis de robustez temporal"):
            st.markdown("""
            **Interpretación del análisis de robustez temporal:**

            Esta tabla muestra cómo cambian los pesos óptimos del portafolio de Sharpe Máximo
            cuando se optimiza sobre diferentes sub-periodos históricos (3, 5 y N años).

            **¿Por qué es importante?**

            - Si los pesos son **similares entre periodos**, el modelo es **estable y robusto**:
              la señal de optimización es consistente independientemente del horizonte temporal.
            - Si los pesos **cambian drásticamente**, indica que el modelo es sensible
              a qué datos se incluyen, lo cual es una limitación importante que el inversor
              debe considerar.

            **Cómo leer la tabla:**
            - Cada fila representa una optimización independiente sobre ese sub-periodo.
            - Columnas son los tickers analizados.
            - Compare los valores fila a fila: variaciones menores a 10–15 puntos porcentuales
              sugieren estabilidad razonable; variaciones mayores indican fragilidad del modelo.

            Este análisis es uno de los más valorados en evaluaciones técnicas avanzadas,
            ya que demuestra si la optimización tiene validez fuera del periodo de entrenamiento.
            """)

    # =====================================================================
    # 9) SÍNTESIS — INTERPRETACIÓN FINAL PONDERADA EN EL TIEMPO
    # =====================================================================
    st.subheader("Interpretación automática del mejor portafolio")
    st.dataframe(r["weighted_performance"].rename("Desempeño_Ponderado"))

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
    # PESOS ÓPTIMOS SEGÚN PORTAFOLIO RECOMENDADO
    # =====================================================================
    st.subheader("Pesos óptimos del portafolio recomendado")

    df_weights = r["df_weights"]
    st.dataframe(df_weights)

    _cw1, _cw2, _cw3 = st.columns([0.5, 2, 0.5])
    with _cw2:
        _n = len(df_weights)
        _bar_colors = [_CYAN, _GREEN, _PURPLE, _ORANGE,
                       "#ff6b9d", "#ffd166"][:_n]
        _eq_w = 1 / _n

        fig_w, ax_w = plt.subplots(figsize=(7, max(3, _n * 0.9)),
                                    facecolor=_BG_DARK)
        _dark_ax(ax_w)

        bars = ax_w.barh(
            df_weights["Ticker"], df_weights["Peso"],
            color=_bar_colors, height=0.55,
            edgecolor=_BG_DARK, linewidth=0.8
        )

        # Etiquetas dentro/fuera de barra según ancho
        for bar, w in zip(bars, df_weights["Peso"]):
            if w > 0.10:
                ax_w.text(
                    bar.get_width() - 0.015,
                    bar.get_y() + bar.get_height() / 2,
                    f"{w:.1%}", va="center", ha="right",
                    color="white", fontsize=11, fontweight="bold"
                )
            else:
                ax_w.text(
                    bar.get_width() + 0.008,
                    bar.get_y() + bar.get_height() / 2,
                    f"{w:.1%}", va="center", ha="left",
                    color=_TEXT, fontsize=11, fontweight="bold"
                )

        # Línea de referencia: peso equitativo
        ax_w.axvline(_eq_w, color=_ORANGE, linewidth=1.4,
                     linestyle="--", alpha=0.85)
        ax_w.text(_eq_w + 0.005, _n - 0.65,
                  f"Peso igual ({_eq_w:.0%})",
                  color=_ORANGE, fontsize=8)

        ax_w.xaxis.set_major_formatter(_PCT)
        ax_w.set_xlabel("Proporción del capital asignada", fontsize=10)
        ax_w.set_title(
            f"Composición del Portafolio Recomendado",
            fontsize=11, fontweight="bold", color=_CYAN, pad=10
        )
        ax_w.text(0.5, 1.01, f"{metodo} · Límite máx. 50% por activo",
                  transform=ax_w.transAxes, ha="center",
                  va="bottom", fontsize=8, color=_DIM)
        ax_w.set_xlim(0, min(1.0, df_weights["Peso"].max() + 0.18))
        ax_w.invert_yaxis()

        fig_w.tight_layout()
        st.pyplot(fig_w)
        plt.close(fig_w)

    with st.expander("📖 Interpretación – Pesos óptimos del portafolio recomendado"):
        st.markdown(
            f"""
            ### Interpretación de los pesos

            Los pesos mostrados corresponden **exclusivamente** al portafolio
            recomendado por el modelo (**{best}**).

            - Cada peso indica qué proporción del capital debe asignarse a cada activo.
            - La suma total de los pesos es del **100%**.
            - Esta asignación refleja el comportamiento histórico del portafolio
              bajo el criterio seleccionado.
            - Ningún activo puede superar el **50% del portafolio** (límite técnico
              para evitar concentraciones excesivas poco realistas).

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
            """
        )

    st.success("Análisis del portafolio ejecutado correctamente")

    # =====================================================================
    # 10) RENDIMIENTOS ACUMULADOS
    # =====================================================================
    st.subheader("Rendimiento acumulado por acción")
    st.line_chart(r["cumulative_assets"])

    st.subheader("Comparación de rendimientos de estrategias")
    st.line_chart(pd.DataFrame({
        "Sharpe Máximo":      cum_sharpe,
        "Mínima Volatilidad": cum_minvol,
        "Pesos Iguales":      cum_equal
    }))

    with st.expander("📖 Interpretación – Rendimiento acumulado por acción"):
        st.markdown(
            """
            **Interpretación:**

            El rendimiento acumulado refleja cómo habría evolucionado una inversión inicial
            en cada activo si se hubiera mantenido durante todo el periodo de análisis.

            - Curvas más empinadas indican mayor crecimiento del capital.
            - Activos con mayor volatilidad suelen mostrar trayectorias más irregulares.
            - Diferencias significativas entre curvas evidencian distintos perfiles
              de riesgo y rentabilidad.

            Este gráfico facilita la comparación directa del desempeño histórico
            entre los activos analizados.
            """
        )

    # =====================================================================
    # RETORNOS DIARIOS
    # =====================================================================
    st.subheader("Retornos diarios de los activos")
    st.line_chart(returns)

    with st.expander("📖 Interpretación – Retornos diarios de los activos"):
        st.markdown(
            """
            **Interpretación:**

            Este gráfico muestra los retornos logarítmicos diarios de cada activo,
            evidenciando la volatilidad de corto plazo.

            - Picos positivos o negativos representan movimientos abruptos del mercado.
            - Mayor dispersión implica mayor riesgo.
            - Periodos de alta concentración de picos suelen coincidir con crisis financieras
              o eventos macroeconómicos relevantes.

            Este análisis es clave para evaluar el riesgo diario asumido por el inversor.
            """
        )

    st.subheader("Retornos diarios por activo")

    for ticker in returns.columns:
        st.markdown(f"### {ticker}")
        st.line_chart(returns[[ticker]])

    with st.expander("📖 Interpretación – Retornos diarios por activo individual"):
        st.markdown(
            """
            **Interpretación:**

            Este gráfico muestra el comportamiento diario del retorno del activo,
            permitiendo identificar:

            - Frecuencia e intensidad de pérdidas y ganancias.
            - Presencia de volatilidad asimétrica (más caídas que subidas).
            - Episodios de estrés específicos para el activo.

            Resulta útil para evaluar el riesgo individual antes de integrarlo
            dentro de un portafolio diversificado.
            """
        )

    # =====================================================================
    # 11) FRONTERA EFICIENTE
    # =====================================================================
    st.subheader("Frontera eficiente (Retorno vs Volatilidad)")

    _col1, _col2, _col3 = st.columns([0.5, 2, 0.5])
    with _col2:
        fig2, ax2 = plt.subplots(figsize=(8, 5), facecolor=_BG_DARK)
        _dark_ax(ax2)

        ev = np.array(r["efficient_vols"])
        er = np.array(r["efficient_rets"])

        # Zona sombreada = región factible (ineficiente)
        ax2.fill_between(ev, er, alpha=0.09, color=_CYAN)
        ax2.fill_between(ev, er.min() * 0.5, er, alpha=0.05, color=_CYAN)

        # Curva de la frontera
        ax2.plot(ev, er, color=_CYAN, linewidth=3,
                 zorder=4, label="Frontera eficiente")

        # Línea vertical en mínima varianza
        ax2.axvline(r["vol_minvol"], color=_DIM, linewidth=0.8,
                    linestyle=":", alpha=0.6)

        # Etiqueta de zona ineficiente
        mid_vol = (ev.min() + ev.max()) / 2
        ax2.text(mid_vol, er.min() * 0.7,
                 "Zona ineficiente\n(mismo riesgo, menos retorno)",
                 color=_DIM, fontsize=8, ha="center", style="italic",
                 bbox=dict(boxstyle="round,pad=0.4",
                           fc=_BG_CARD, ec=_GRID, alpha=0.75))

        # Puntos y anotaciones numeradas
        _pts = [
            (r["vol_sharpe"], r["ret_sharpe"], "① Sharpe Máximo",    _CYAN,   "o", (14,  6)),
            (r["vol_minvol"], r["ret_minvol"], "② Mín. Volatilidad", _GREEN,  "^", (10, -26)),
            (r["vol_equal"],  r["ret_equal"],  "③ Pesos Iguales",    _PURPLE, "s", (14,  6)),
        ]
        for vx, ry, label, col, mk, off in _pts:
            ax2.scatter(vx, ry, color=col, s=160, marker=mk,
                        edgecolors="white", linewidths=1.2, zorder=6)
            ax2.annotate(
                f"{label}\nRetorno: {ry:.1%}  ·  Riesgo: {vx:.1%}",
                xy=(vx, ry), xytext=off, textcoords="offset points",
                fontsize=8.5, color=col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4",
                          fc=_BG_CARD, ec=col, alpha=0.92),
                arrowprops=dict(arrowstyle="-", color=col, lw=0.8)
            )

        ax2.xaxis.set_major_formatter(_PCT)
        ax2.yaxis.set_major_formatter(_PCT)
        ax2.set_xlabel("Volatilidad anual  (↑ más riesgo)", fontsize=10)
        ax2.set_ylabel("Retorno anual esperado  (↑ mejor)", fontsize=10)
        ax2.set_title("Frontera Eficiente de Markowitz",
                      fontsize=13, fontweight="bold", color=_CYAN, pad=14)

        _handles = [
            plt.Line2D([0], [0], color=_CYAN,   linewidth=2.5,
                       label="Frontera eficiente"),
            plt.Line2D([0], [0], color=_CYAN,   marker="o",
                       linestyle="None", markersize=8, label="① Sharpe Máximo"),
            plt.Line2D([0], [0], color=_GREEN,  marker="^",
                       linestyle="None", markersize=8, label="② Mín. Volatilidad"),
            plt.Line2D([0], [0], color=_PURPLE, marker="s",
                       linestyle="None", markersize=8, label="③ Pesos Iguales"),
        ]
        ax2.legend(handles=_handles, facecolor=_BG_CARD,
                   edgecolor=_GRID, labelcolor=_TEXT,
                   fontsize=8.5, loc="lower right")

        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    with st.expander("📖 Interpretación – Frontera eficiente de Markowitz"):
        st.markdown(
            """
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

            Esta visualización facilita la comprensión del trade-off
            riesgo–retorno y constituye una herramienta central para
            la toma de decisiones de inversión.
            """
        )

    # =====================================================================
    # RESUMEN FINAL DE TABLAS
    # =====================================================================
    st.subheader("Comparación de estrategias")
    st.dataframe(r["df_compare"])

    st.subheader("Pesos del portafolio recomendado")
    st.dataframe(r["df_weights"])

    df_retornos = pd.DataFrame(
        {
            "Retorno anual esperado": [
                r["retornos"]["Sharpe Máximo"],
                r["retornos"]["Mínima Volatilidad"],
                r["retornos"]["Pesos Iguales"]
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
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        st.warning("El asistente requiere una API Key válida de Gemini.")
        st.stop()

    MODEL = "gemini-2.5-flash-lite"
    GEMINI_URL = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{MODEL}:generateContent?key={GEMINI_API_KEY}"
    )

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
        with st.chat_message("user"):
            st.markdown(user_question)

        results = st.session_state.analysis_results

        best_strategy = results["best"]
        weights_dict  = results["weights"][best_strategy]

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























































