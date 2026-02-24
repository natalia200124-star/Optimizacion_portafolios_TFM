import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
from datetime import datetime
import requests
import os
from sklearn.covariance import LedoitWolf

# =========================
# RISK_FREE_RATE fuera de la función cacheada
# =========================
RISK_FREE_RATE = 0.045  # T-Bill 3 meses ~4.5%

# =========================
# PALETA DE COLORES COMPARTIDA
# =========================
COLORS = {
    "sharpe":  "#00d9ff",   # cian
    "minvol":  "#66ffb2",   # verde menta
    "equal":   "#ff9966",   # naranja suave
    "bg":      "#0f1419",
    "panel":   "#1a1f2e",
    "border":  "#00d9ff30",
    "text":    "#e1e7ed",
    "grid":    "#ffffff18",
}

def apply_dark_style(fig, axes_list):
    """Aplica tema oscuro coherente a cualquier figura matplotlib."""
    fig.patch.set_facecolor(COLORS["bg"])
    for ax in (axes_list if hasattr(axes_list, '__iter__') else [axes_list]):
        ax.set_facecolor(COLORS["panel"])
        ax.tick_params(colors=COLORS["text"], labelsize=8)
        ax.xaxis.label.set_color(COLORS["text"])
        ax.yaxis.label.set_color(COLORS["text"])
        ax.title.set_color(COLORS["sharpe"])
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS["border"])
        ax.grid(True, color=COLORS["grid"], linewidth=0.6)


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
# SESSION STATE
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
    n = len(tickers)

    # ── Parámetros ────────────────────────────────────────────────────────
    LAMBDA_REG    = 0.01
    N_SIMULATIONS = 5000   # 5,000 es el estándar académico para VaR/CVaR 95%
    # MAX_WEIGHT dinámico: permite hasta 2× el peso igual, mín 40%, máx 80%
    # Esto evita el bug de forzar 50/50 con solo 2 activos
    MAX_WEIGHT    = min(0.80, max(2.0 / n, 0.40))

    # =====================================================================
    # 1) DESCARGA DE DATOS
    # =====================================================================
    end_date   = datetime.today()
    start_date = end_date.replace(year=end_date.year - years)

    benchmark_tickers = ["SPY", "QQQ", "URTH"]
    all_tickers = tickers + benchmark_tickers

    raw_data = yf.download(
        all_tickers, start=start_date, end=end_date,
        auto_adjust=False, progress=False
    )
    raw_data = raw_data["Adj Close"]
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data = raw_data.droplevel(0, axis=1)
    raw_data = raw_data.sort_index().ffill()

    data           = raw_data[tickers].copy()
    benchmark_data = raw_data[benchmark_tickers].copy()

    tickers_invalidos = [t for t in tickers if data[t].isnull().mean() > 0.2]
    if tickers_invalidos:
        raise ValueError(
            f"Tickers sin datos suficientes: {', '.join(tickers_invalidos)}."
        )

    data           = data.dropna()
    benchmark_data = benchmark_data.ffill().dropna()
    if data.empty:
        raise ValueError("No hay datos suficientes para el periodo seleccionado.")

    # =====================================================================
    # 2) RETORNOS LOGARÍTMICOS + LEDOIT-WOLF
    # =====================================================================
    returns            = np.log(data / data.shift(1)).dropna()
    mean_returns_daily = returns.mean()
    trading_days       = 252
    mean_returns_annual = mean_returns_daily * trading_days

    lw = LedoitWolf()
    lw.fit(returns)
    cov_daily  = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
    cov_annual = cov_daily * trading_days

    # =====================================================================
    # 3) FUNCIONES DE OPTIMIZACIÓN (regularización L2 simétrica)
    # =====================================================================
    def performance(weights, mean_ret, cov):
        ret    = np.dot(weights, mean_ret)
        vol    = np.sqrt(weights.T @ cov @ weights)
        sharpe = (ret - RISK_FREE_RATE) / vol if vol > 0 else 0
        return ret, vol, sharpe

    def neg_sharpe(weights):
        ret, vol_val, sharpe = performance(weights, mean_returns_annual, cov_annual)
        penalty = LAMBDA_REG * np.sum(weights ** 2)
        return -(sharpe - penalty) if vol_val > 0 else 1e6

    def vol_obj(weights):
        return np.sqrt(weights.T @ cov_annual @ weights) + LAMBDA_REG * np.sum(weights ** 2)

    def max_drawdown(series):
        return ((series / series.cummax()) - 1).min()

    x0          = np.repeat(1 / n, n)
    bounds      = tuple((0, MAX_WEIGHT) for _ in range(n))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    # =====================================================================
    # 4) OPTIMIZACIONES
    # =====================================================================
    res_sharpe     = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights_sharpe = res_sharpe.x
    ret_sharpe, vol_sharpe, sharpe_sharpe = performance(weights_sharpe, mean_returns_annual, cov_annual)

    res_minvol     = minimize(vol_obj, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights_minvol = res_minvol.x
    ret_minvol, vol_minvol, sharpe_minvol = performance(weights_minvol, mean_returns_annual, cov_annual)

    weights_equal = np.repeat(1 / n, n)
    ret_equal, vol_equal, sharpe_equal = performance(weights_equal, mean_returns_annual, cov_annual)

    # =====================================================================
    # 5) RENDIMIENTOS ACUMULADOS (exp cumsum — log returns)
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
    # 5.1) BENCHMARKS
    # =====================================================================
    benchmark_log_returns = np.log(benchmark_data / benchmark_data.shift(1)).dropna()
    benchmark_cum         = np.exp(benchmark_log_returns.cumsum())

    # =====================================================================
    # 6) FRONTERA EFICIENTE + NUBE DE PORTAFOLIOS ALEATORIOS
    # =====================================================================
    target_returns = np.linspace(mean_returns_annual.min(), mean_returns_annual.max(), 50)
    efficient_vols, efficient_rets = [], []
    for targ in target_returns:
        cons = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, targ=targ: np.dot(w, mean_returns_annual) - targ}
        )
        res = minimize(vol_obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
        if res.success:
            r, v, _ = performance(res.x, mean_returns_annual, cov_annual)
            efficient_rets.append(r)
            efficient_vols.append(v)

    # Nube de portafolios aleatorios para visualización de la frontera
    np.random.seed(0)
    n_random       = 2500
    rand_w         = np.random.dirichlet(np.ones(n), size=n_random)
    rand_rets      = rand_w @ mean_returns_annual.values
    rand_vols      = np.array([np.sqrt(w @ cov_annual.values @ w) for w in rand_w])
    rand_sharpes   = (rand_rets - RISK_FREE_RATE) / rand_vols

    # =====================================================================
    # 8) TABLAS DE MÉTRICAS
    # =====================================================================
    df_compare = pd.DataFrame({
        "Estrategia":       ["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"],
        "Retorno Anual":    [ret_sharpe, ret_minvol, ret_equal],
        "Volatilidad":      [vol_sharpe, vol_minvol, vol_equal],
        "Sharpe":           [sharpe_sharpe, sharpe_minvol, sharpe_equal],
        "Retorno Acumulado":[cum_sharpe.iloc[-1]-1, cum_minvol.iloc[-1]-1, cum_equal.iloc[-1]-1],
        "Máx Drawdown":     [dd_sharpe, dd_minvol, dd_equal]
    })

    rolling_vol = pd.DataFrame({
        "Sharpe Máximo":      daily_sharpe.rolling(252).std() * np.sqrt(252),
        "Mínima Volatilidad": daily_minvol.rolling(252).std() * np.sqrt(252),
        "Pesos Iguales":      daily_equal.rolling(252).std()  * np.sqrt(252)
    })

    df_calmar = pd.DataFrame({
        "Estrategia": ["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"],
        "Calmar": [ret_sharpe/abs(dd_sharpe), ret_minvol/abs(dd_minvol), ret_equal/abs(dd_equal)]
    })

    def sortino_ratio(ret_anual, daily_ret):
        downside     = np.minimum(daily_ret, 0)
        downside_dev = np.sqrt((downside**2).mean()) * np.sqrt(252)
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
    # MONTE CARLO + BOOTSTRAP
    # =====================================================================
    np.random.seed(42)

    sim_assets_mc = np.random.multivariate_normal(
        mean_returns_annual.values, cov_annual.values, N_SIMULATIONS
    )
    sim_sharpe_mc = sim_assets_mc @ weights_sharpe
    sim_minvol_mc = sim_assets_mc @ weights_minvol
    sim_equal_mc  = sim_assets_mc @ weights_equal

    n_obs      = len(returns)
    block_size = 20
    n_blocks   = (N_SIMULATIONS * trading_days) // block_size + 1
    starts     = np.random.randint(0, n_obs - block_size, size=n_blocks)
    boot_rows  = [returns.iloc[s:s+block_size].values for s in starts]
    boot_ret   = np.vstack(boot_rows)[:N_SIMULATIONS * trading_days]
    boot_ret   = boot_ret.reshape(N_SIMULATIONS, trading_days, n)

    sim_sharpe_boot = (boot_ret @ weights_sharpe).sum(axis=1)
    sim_minvol_boot = (boot_ret @ weights_minvol).sum(axis=1)
    sim_equal_boot  = (boot_ret @ weights_equal).sum(axis=1)

    def var_cvar(s, alpha=0.05):
        v = np.percentile(s, alpha*100)
        c = s[s <= v].mean()
        p = (s < 0).mean()
        return v, c, p

    vs_mc, cs_mc, ps_mc = var_cvar(sim_sharpe_mc)
    vm_mc, cm_mc, pm_mc = var_cvar(sim_minvol_mc)
    ve_mc, ce_mc, pe_mc = var_cvar(sim_equal_mc)
    vs_bt, cs_bt, ps_bt = var_cvar(sim_sharpe_boot)
    vm_bt, cm_bt, pm_bt = var_cvar(sim_minvol_boot)
    ve_bt, ce_bt, pe_bt = var_cvar(sim_equal_boot)

    df_mc_stats = pd.DataFrame({
        "Estrategia":         ["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"],
        "VaR MC 95%":         [vs_mc, vm_mc, ve_mc],
        "CVaR MC 95%":        [cs_mc, cm_mc, ce_mc],
        "Prob. Pérdida MC":   [ps_mc, pm_mc, pe_mc],
        "VaR Boot 95%":       [vs_bt, vm_bt, ve_bt],
        "CVaR Boot 95%":      [cs_bt, cm_bt, ce_bt],
        "Prob. Pérdida Boot": [ps_bt, pm_bt, pe_bt],
    })

    # =====================================================================
    # BENCHMARKS
    # =====================================================================
    benchmarks = {"S&P 500 (SPY)": "SPY", "Nasdaq 100 (QQQ)": "QQQ", "MSCI World (URTH)": "URTH"}
    benchmark_summary = []
    for name, ticker in benchmarks.items():
        ret = benchmark_log_returns[ticker].mean() * 252
        v   = benchmark_log_returns[ticker].std() * np.sqrt(252)
        dd  = max_drawdown(benchmark_cum[ticker])
        benchmark_summary.append({
            "Benchmark": name, "Retorno Anual": ret, "Volatilidad": v,
            "Retorno Acumulado": benchmark_cum[ticker].iloc[-1]-1, "Máx Drawdown": dd
        })
    df_benchmarks = pd.DataFrame(benchmark_summary)

    comparison_cum = pd.DataFrame({
        "Sharpe Máximo": cum_sharpe, "Mínima Volatilidad": cum_minvol, "Pesos Iguales": cum_equal,
        "S&P 500 (SPY)": benchmark_cum["SPY"], "Nasdaq 100 (QQQ)": benchmark_cum["QQQ"],
        "MSCI World (URTH)": benchmark_cum["URTH"]
    })

    # =====================================================================
    # ESTABILIDAD DE PESOS
    # =====================================================================
    def optimizar_en_ventana(ret_v):
        mr  = ret_v.mean() * trading_days
        lw_ = LedoitWolf(); lw_.fit(ret_v)
        cov_ = pd.DataFrame(lw_.covariance_ * trading_days, index=ret_v.columns, columns=ret_v.columns)
        n_  = len(ret_v.columns)
        x0_ = np.repeat(1/n_, n_)
        bds = tuple((0, min(0.80, max(2.0/n_, 0.40))) for _ in range(n_))
        con = {"type": "eq", "fun": lambda w: np.sum(w)-1}
        def ns_(w):
            r_ = np.dot(w, mr); v_ = np.sqrt(w.T @ cov_ @ w)
            sh = (r_ - RISK_FREE_RATE)/v_ if v_ > 0 else 0
            return -(sh - LAMBDA_REG*np.sum(w**2)) if v_ > 0 else 1e6
        def vo_(w):
            return np.sqrt(w.T @ cov_ @ w) + LAMBDA_REG*np.sum(w**2)
        ws = minimize(ns_, x0_, method="SLSQP", bounds=bds, constraints=con).x
        wm = minimize(vo_, x0_, method="SLSQP", bounds=bds, constraints=con).x
        return ws, wm

    stability_rows = []
    for horizon in [3, 5, years]:
        cutoff  = returns.index[-1] - pd.DateOffset(years=horizon)
        ret_sub = returns[returns.index >= cutoff]
        if len(ret_sub) < 252:
            continue
        ws_h, wm_h = optimizar_en_ventana(ret_sub)
        for t, ws, wm in zip(tickers, ws_h, wm_h):
            stability_rows.append({
                "Horizonte": f"{horizon} años", "Ticker": t,
                "Peso Sharpe Máx (%)": round(ws*100, 1),
                "Peso Mín Vol (%)":    round(wm*100, 1),
            })
    df_stability = pd.DataFrame(stability_rows) if stability_rows else pd.DataFrame()

    # =====================================================================
    # SÍNTESIS — MEJOR PORTAFOLIO
    # =====================================================================
    asset_summary = {}
    for ticker in tickers:
        asset_summary[ticker] = {
            "retorno_anual":       mean_returns_annual[ticker],
            "volatilidad":         np.sqrt(cov_annual.loc[ticker, ticker]),
            "contribucion_riesgo": cov_annual.loc[ticker].dot(weights_sharpe)
        }

    strategy_summary = {
        "Sharpe Máximo":    {"retorno": ret_sharpe, "volatilidad": vol_sharpe, "sharpe": sharpe_sharpe, "drawdown": dd_sharpe},
        "Mínima Volatilidad": {"retorno": ret_minvol, "volatilidad": vol_minvol, "sharpe": sharpe_minvol, "drawdown": dd_minvol},
        "Pesos Iguales":    {"retorno": ret_equal,  "volatilidad": vol_equal,  "sharpe": sharpe_equal,  "drawdown": dd_equal}
    }

    df_strategies    = pd.DataFrame({"Sharpe Máximo": daily_sharpe, "Mínima Volatilidad": daily_minvol, "Pesos Iguales": daily_equal})
    years_index      = df_strategies.index.year
    unique_years     = np.sort(years_index.unique())
    year_weights     = {y: (i+1)/len(unique_years) for i, y in enumerate(unique_years)}
    weights_series   = years_index.map(year_weights)
    weighted_performance = (
        np.exp(df_strategies.cumsum()).mul(weights_series, axis=0).iloc[-1]
    )
    best = weighted_performance.idxmax()

    if best == "Sharpe Máximo":
        final_weights = weights_sharpe; metodo = "Optimización por Ratio de Sharpe"
    elif best == "Mínima Volatilidad":
        final_weights = weights_minvol; metodo = "Optimización por Mínima Volatilidad"
    else:
        final_weights = weights_equal; metodo = "Asignación Equitativa (Pesos Iguales)"

    df_weights = pd.DataFrame({
        "Ticker": tickers, "Peso": final_weights.round(4), "Peso (%)": (final_weights*100).round(2)
    })

    return {
        "tickers": tickers, "data": data, "returns": returns,
        "cumulative_assets": cumulative_assets,
        "daily_sharpe": daily_sharpe, "daily_minvol": daily_minvol, "daily_equal": daily_equal,
        "cum_sharpe": cum_sharpe, "cum_minvol": cum_minvol, "cum_equal": cum_equal,
        "df_compare": df_compare, "rolling_vol": rolling_vol, "df_calmar": df_calmar, "df_sortino": df_sortino,
        "df_mc_stats": df_mc_stats,
        "mc_simulations_mc":   {"Sharpe Máximo": sim_sharpe_mc, "Mínima Volatilidad": sim_minvol_mc, "Pesos Iguales": sim_equal_mc},
        "mc_simulations_boot": {"Sharpe Máximo": sim_sharpe_boot, "Mínima Volatilidad": sim_minvol_boot, "Pesos Iguales": sim_equal_boot},
        "mc_var_mc":   {"Sharpe Máximo": vs_mc, "Mínima Volatilidad": vm_mc, "Pesos Iguales": ve_mc},
        "mc_cvar_mc":  {"Sharpe Máximo": cs_mc, "Mínima Volatilidad": cm_mc, "Pesos Iguales": ce_mc},
        "mc_var_bt":   {"Sharpe Máximo": vs_bt, "Mínima Volatilidad": vm_bt, "Pesos Iguales": ve_bt},
        "mc_cvar_bt":  {"Sharpe Máximo": cs_bt, "Mínima Volatilidad": cm_bt, "Pesos Iguales": ce_bt},
        "df_stability": df_stability, "df_benchmarks": df_benchmarks, "comparison_cum": comparison_cum,
        "weighted_performance": weighted_performance, "best": best, "metodo": metodo, "df_weights": df_weights,
        "efficient_vols": efficient_vols, "efficient_rets": efficient_rets,
        "rand_vols": rand_vols, "rand_rets": rand_rets, "rand_sharpes": rand_sharpes,
        "vol_sharpe": vol_sharpe, "ret_sharpe": ret_sharpe,
        "vol_minvol": vol_minvol, "ret_minvol": ret_minvol,
        "vol_equal":  vol_equal,  "ret_equal":  ret_equal,
        "asset_summary": asset_summary, "strategy_summary": strategy_summary,
        "weights": {
            "Sharpe Máximo":    dict(zip(tickers, weights_sharpe)),
            "Mínima Volatilidad": dict(zip(tickers, weights_minvol)),
            "Pesos Iguales":    dict(zip(tickers, weights_equal))
        },
        "retornos":     {"Sharpe Máximo": ret_sharpe, "Mínima Volatilidad": ret_minvol, "Pesos Iguales": ret_equal},
        "volatilidades":{"Sharpe Máximo": vol_sharpe, "Mínima Volatilidad": vol_minvol, "Pesos Iguales": vol_equal}
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

years = st.slider("Seleccione el horizonte temporal (años)", min_value=3, max_value=10, value=6)

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

    idx = data.index.tz_localize(None) if getattr(data.index, "tz", None) else data.index
    precios_2025 = data[idx.year == 2025].tail(10)
    if precios_2025.empty:
        st.info("No hay datos disponibles para 2025.")
    else:
        st.dataframe(precios_2025, use_container_width=True)

    st.subheader(f"Tendencia de precios (últimos {years} años)")
    st.line_chart(data)

    with st.expander("📖 Interpretación – Tendencia de precios"):
        st.markdown("""
            **Interpretación:**

            Este gráfico muestra la evolución histórica de los precios ajustados de cada activo
            durante el horizonte temporal seleccionado.

            - Tendencias crecientes indican periodos de apreciación del activo.
            - Periodos de alta pendiente reflejan fases de crecimiento acelerado.
            - Movimientos bruscos o caídas pronunciadas suelen asociarse a eventos de mercado
              o episodios de alta volatilidad.
        """)

    # =====================================================================
    # COMPARACIÓN DE ESTRATEGIAS
    # =====================================================================
    st.subheader("Comparación sistemática de estrategias")
    st.dataframe(r["df_compare"])

    with st.expander("📖 Interpretación – Comparación de estrategias"):
        st.markdown("""
            **Cómo interpretar esta tabla:**
            - **Retorno acumulado:** cuánto creció el capital total en el periodo.
            - **Volatilidad:** magnitud de las fluctuaciones (riesgo).
            - **Sharpe:** eficiencia riesgo–retorno.
            - **Máx Drawdown:** peor caída histórica desde un máximo.

            La estrategia de **Sharpe Máximo** tiende a ofrecer el mayor retorno ajustado por riesgo.
            La estrategia de **Mínima Volatilidad** prioriza la estabilidad del capital.
            La estrategia de **Pesos Iguales** actúa como referencia neutral sin optimización explícita.
        """)

    st.subheader("Volatilidad histórica móvil")
    st.line_chart(r["rolling_vol"])

    with st.expander("📖 Interpretación – Volatilidad histórica móvil"):
        st.markdown("""
            **Interpretación:**
            Esta gráfica muestra cómo el riesgo **cambia en el tiempo**.
            - Picos altos suelen coincidir con periodos de crisis.
            - Estrategias más estables presentan curvas más suaves.
        """)

    st.subheader("Ratio Calmar (retorno vs drawdown)")
    st.dataframe(r["df_calmar"])

    with st.expander("📖 Interpretación – Ratio Calmar"):
        st.markdown("""
            **Interpretación analítica del Ratio Calmar:**

            El Ratio Calmar relaciona el **retorno anual esperado** con el **máximo drawdown histórico**.
            Un valor elevado indica que la estrategia logra retornos atractivos manteniendo caídas
            controladas. Resulta especialmente relevante para inversionistas conservadores.
        """)

    st.subheader("Ratio Sortino")
    st.dataframe(r["df_sortino"])

    with st.expander("📖 Interpretación – Ratio Sortino"):
        st.markdown("""
            **Interpretación analítica del Ratio Sortino:**

            Evalúa el desempeño considerando exclusivamente la **volatilidad negativa**.
            Un valor más alto indica mayor retorno por unidad de riesgo a la baja.
            A diferencia del Sharpe, no penaliza la volatilidad positiva.
        """)

    # =====================================================================
    # MONTE CARLO — KDE SUAVIZADO, GRÁFICA ÚNICA
    # =====================================================================
    st.subheader("Simulación Monte Carlo – Análisis de riesgo forward-looking")
    st.dataframe(r["df_mc_stats"])

    _mc1, _mc2, _mc3 = st.columns([0.1, 3.5, 0.1])
    with _mc2:
        from scipy.stats import gaussian_kde

        strat_colors = [COLORS["sharpe"], COLORS["minvol"], COLORS["equal"]]
        strat_names  = ["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"]

        fig_mc, ax = plt.subplots(figsize=(13, 6))
        apply_dark_style(fig_mc, ax)
        fig_mc.suptitle("Distribución de Retornos Anuales Simulados – KDE (5,000 escenarios)",
                         color=COLORS["sharpe"], fontsize=12, fontweight="bold", y=1.01)

        # Usamos Monte Carlo paramétrico como fuente principal del KDE
        sims_dict = r["mc_simulations_mc"]
        var_dict  = r["mc_var_mc"]

        for name, color in zip(strat_names, strat_colors):
            sims = sims_dict[name]
            # Rango limpio: percentil 0.5 – 99.5
            x_min = np.percentile(sims, 0.5)
            x_max = np.percentile(sims, 99.5)
            x_grid = np.linspace(x_min, x_max, 400)

            kde = gaussian_kde(sims, bw_method=0.15)
            y_kde = kde(x_grid)

            # Curva KDE principal
            ax.plot(x_grid, y_kde, color=color, linewidth=2.2, label=name, zorder=3)
            # Área bajo la curva semitransparente
            ax.fill_between(x_grid, y_kde, alpha=0.12, color=color, zorder=2)

            # Línea VaR vertical
            var_val = var_dict[name]
            ax.axvline(var_val, color=color, linestyle="--", linewidth=1.4, alpha=0.85,
                       label=f"VaR {name[:6]} = {var_val:.1%}", zorder=4)

        # Línea del cero
        ax.axvline(0, color="white", linestyle="-", linewidth=1.6, alpha=0.45, zorder=5)
        ax.text(0, 1.0, "0%", color="white", fontsize=8, alpha=0.7,
                transform=ax.get_xaxis_transform(), ha="center", va="bottom")

        ax.set_xlabel("Retorno anual simulado", fontsize=10)
        ax.set_ylabel("Densidad de probabilidad", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

        ax.legend(fontsize=8, facecolor="#252d3f", edgecolor=COLORS["border"],
                  labelcolor=COLORS["text"], loc="upper left", ncol=2,
                  framealpha=0.9, bbox_to_anchor=(0.01, 0.99))

        plt.tight_layout()
        st.pyplot(fig_mc)
        plt.close(fig_mc)

    with st.expander("📖 Interpretación – Simulación Monte Carlo"):
        st.markdown("""
        **Interpretación analítica de la Simulación Monte Carlo:**

        La simulación genera 5,000 escenarios posibles de retorno anual para cada estrategia
        utilizando la media y la matriz de covarianza estimadas. Permite evaluar el
        comportamiento del portafolio bajo incertidumbre futura, no solo con datos históricos.

        **Monte Carlo vs Bootstrap:**
        - El gráfico izquierdo asume distribución normal multivariada.
        - El gráfico derecho remuestrea bloques de retornos históricos reales, sin asumir normalidad.
        - Las líneas punteadas indican el **VaR 95%** de cada estrategia.
        - Si ambos métodos coinciden, el resultado es más robusto.

        **Métricas clave:**
        - **VaR 95%:** pérdida máxima en el 5% de los peores escenarios.
        - **CVaR 95%:** promedio de las pérdidas en esos escenarios extremos.
        - **Probabilidad de pérdida:** porcentaje de escenarios con retorno negativo.
        """)

    # =====================================================================
    # ESTABILIDAD DE PESOS
    # =====================================================================
    if not r["df_stability"].empty:
        st.subheader("Estabilidad de pesos por horizonte temporal")
        st.dataframe(r["df_stability"], use_container_width=True)

        with st.expander("📖 Interpretación – Estabilidad de pesos por periodo"):
            st.markdown("""
                **Interpretación analítica de la estabilidad de pesos:**

                Esta tabla muestra cómo cambian los pesos óptimos cuando se re-optimiza
                con ventanas históricas de 3, 5 y todos los años disponibles.

                - Pesos **similares entre horizontes** → estrategia robusta y confiable.
                - Pesos **muy variables** → mayor sensibilidad al periodo de entrenamiento.

                En una defensa técnica, la estabilidad de pesos es un argumento clave:
                demuestra que la solución no es un artefacto del periodo de datos.
            """)

    # =====================================================================
    # COVID 2020
    # =====================================================================
    st.subheader("Comportamiento en periodo de crisis (COVID 2020)")
    crisis = (cum_sharpe.index.year == 2020)
    st.line_chart(pd.DataFrame({
        "Sharpe Máximo": cum_sharpe[crisis],
        "Mínima Volatilidad": cum_minvol[crisis],
        "Pesos Iguales": cum_equal[crisis]
    }))

    with st.expander("📖 Interpretación – Comportamiento en crisis (COVID 2020)"):
        st.markdown("""
            **Interpretación del comportamiento en periodo de crisis:**

            Permite evaluar la profundidad de la caída, la velocidad de recuperación
            y la resiliencia relativa de cada estrategia ante eventos extremos.
            Las estrategias de mínima volatilidad suelen contener mejor las caídas iniciales.
        """)

    # =====================================================================
    # BENCHMARKS
    # =====================================================================
    st.subheader("Comparación con benchmarks de mercado")
    st.dataframe(r["df_benchmarks"])

    with st.expander("📖 ¿Qué es un benchmark? – S&P 500, MSCI y NASDAQ explicados"):
        st.markdown("""
        ### ¿Qué es un benchmark?
        Un **benchmark** es un punto de referencia para evaluar si una estrategia es buena o mala.

        ### S&P 500 — referencia del mercado americano
        Agrupa ~500 empresas grandes de EE.UU. Si una estrategia no lo supera en el largo plazo,
        resulta difícil justificar su complejidad frente a una inversión pasiva.

        ### MSCI World — diversificación global
        Representa empresas de países desarrollados. Permite evaluar si la estrategia supera
        una cartera globalmente diversificada.

        ### NASDAQ 100 — referencia de crecimiento tecnológico
        Alta concentración en tecnología. Mayor potencial de crecimiento pero también mayor
        volatilidad, especialmente en momentos de crisis.
        """)

    st.subheader("Rendimiento acumulado: estrategias vs benchmarks")
    st.line_chart(r["comparison_cum"])

    with st.expander("📖 Interpretación – Rendimiento acumulado vs benchmarks"):
        st.markdown("""
        La línea que termina más arriba representa la estrategia con mayor crecimiento acumulado.
        Curvas más suaves indican menor volatilidad. Si una estrategia optimizada supera de forma
        consistente a los benchmarks, confirma que el modelo aporta valor frente a inversión pasiva.
        """)

    # =====================================================================
    # MEJOR PORTAFOLIO
    # =====================================================================
    st.subheader("Interpretación automática del mejor portafolio")
    st.dataframe(r["weighted_performance"].rename("Desempeño_Ponderado"))

    if best == "Pesos Iguales":
        st.markdown("### Mejor portafolio: Pesos Iguales\n\nEsta estrategia ha sido la más robusta y consistente, con menor dependencia de supuestos estadísticos.")
    elif best == "Sharpe Máximo":
        st.markdown("### Mejor portafolio: Sharpe Máximo\n\nOfrece el mejor equilibrio riesgo–retorno en el comportamiento histórico reciente.")
    else:
        st.markdown("### Mejor portafolio: Mínima Volatilidad\n\nDestaca por su estabilidad, aunque sacrifica retorno frente a las demás estrategias.")

    st.success(f"Portafolio recomendado según comportamiento real ponderado: {best}")

    # =====================================================================
    # PESOS ÓPTIMOS — GRÁFICO PREMIUM REDISEÑADO
    # =====================================================================
    st.subheader("Pesos óptimos del portafolio recomendado")

    df_weights = r["df_weights"]
    st.dataframe(df_weights)

    _pw1, _pw2, _pw3 = st.columns([0.3, 2.5, 0.3])
    with _pw2:
        tickers_w = df_weights["Ticker"].tolist()
        pesos_w   = df_weights["Peso (%)"].tolist()
        n_w       = len(tickers_w)

        # Paleta degradada cian → verde
        palette = [
            mcolors.to_hex(
                plt.cm.cool(0.15 + 0.7 * i / max(n_w - 1, 1))
            )
            for i in range(n_w)
        ]

        fig_w, ax_w = plt.subplots(figsize=(9, max(3.5, n_w * 0.7)))
        apply_dark_style(fig_w, ax_w)

        bars = ax_w.barh(
            tickers_w, pesos_w,
            color=palette, edgecolor=COLORS["bg"], linewidth=0.8,
            height=0.55
        )

        # Etiquetas de valor al final de cada barra
        for bar, val in zip(bars, pesos_w):
            x_pos = bar.get_width() + 0.5
            ax_w.text(
                x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center", ha="left", fontsize=9,
                color=COLORS["text"], fontweight="600"
            )

        ax_w.set_xlabel("Peso en el portafolio (%)", fontsize=9)
        ax_w.set_xlim(0, max(pesos_w) * 1.22)
        ax_w.set_title(
            f"Composición del portafolio recomendado\n{metodo}",
            fontsize=10, fontweight="bold", pad=10
        )
        ax_w.invert_yaxis()

        plt.tight_layout()
        st.pyplot(fig_w)
        plt.close(fig_w)

    with st.expander("📖 Interpretación – Pesos óptimos del portafolio recomendado"):
        st.markdown(f"""
            ### Interpretación de los pesos

            Los pesos mostrados corresponden al portafolio recomendado (**{best}**).
            La línea punteada blanca indica el peso que tendría cada activo en una distribución equitativa.

            - Barras a la **derecha de la línea** → el modelo sobrepondera ese activo.
            - Barras a la **izquierda de la línea** → el modelo subpondera ese activo.
            - Un **peso del 40%** significa que 40 de cada 100 unidades monetarias se asignan ahí.
        """)

    st.success("Análisis del portafolio ejecutado correctamente")

    # =====================================================================
    # RENDIMIENTOS ACUMULADOS
    # =====================================================================
    st.subheader("Rendimiento acumulado por acción")
    st.line_chart(r["cumulative_assets"])

    st.subheader("Comparación de rendimientos de estrategias")
    st.line_chart(pd.DataFrame({
        "Sharpe Máximo": cum_sharpe, "Mínima Volatilidad": cum_minvol, "Pesos Iguales": cum_equal
    }))

    with st.expander("📖 Interpretación – Rendimiento acumulado"):
        st.markdown("""
            El rendimiento acumulado refleja cómo habría evolucionado una inversión inicial
            en cada activo durante todo el periodo. Curvas más empinadas indican mayor crecimiento.
        """)

    st.subheader("Retornos diarios de los activos")
    st.line_chart(returns)

    with st.expander("📖 Interpretación – Retornos diarios"):
        st.markdown("""
            Muestra los retornos porcentuales diarios de cada activo.
            Picos positivos o negativos representan movimientos abruptos del mercado.
            Periodos de alta concentración de picos suelen coincidir con crisis financieras.
        """)

    st.subheader("Retornos diarios por activo")
    for ticker in returns.columns:
        st.markdown(f"### {ticker}")
        st.line_chart(returns[[ticker]])

    # =====================================================================
    # FRONTERA EFICIENTE — GRÁFICO PREMIUM REDISEÑADO
    # =====================================================================
    st.subheader("Frontera eficiente (Retorno vs Volatilidad)")

    _fe1, _fe2, _fe3 = st.columns([0.2, 3, 0.2])
    with _fe2:
        fig_fe, ax_fe = plt.subplots(figsize=(10, 6))
        apply_dark_style(fig_fe, ax_fe)

        # Nube de portafolios aleatorios coloreados por Sharpe
        sc = ax_fe.scatter(
            r["rand_vols"], r["rand_rets"],
            c=r["rand_sharpes"], cmap="plasma",
            s=12, alpha=0.35, linewidths=0, zorder=1
        )
        cbar = plt.colorbar(sc, ax=ax_fe, pad=0.02)
        cbar.set_label("Ratio de Sharpe", color=COLORS["text"], fontsize=8)
        cbar.ax.yaxis.set_tick_params(color=COLORS["text"])
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=COLORS["text"], fontsize=7)
        cbar.outline.set_edgecolor(COLORS["border"])

        # Línea de la frontera eficiente
        ax_fe.plot(
            r["efficient_vols"], r["efficient_rets"],
            color=COLORS["sharpe"], linewidth=2.5, zorder=3,
            label="Frontera eficiente",
            path_effects=[pe.withStroke(linewidth=5, foreground="#00d9ff20")]
        )

        # Puntos de las tres estrategias
        strategy_points = [
            (r["vol_sharpe"], r["ret_sharpe"], COLORS["sharpe"],  "★", "Sharpe Máximo"),
            (r["vol_minvol"], r["ret_minvol"], COLORS["minvol"],  "▲", "Mínima Volatilidad"),
            (r["vol_equal"],  r["ret_equal"],  COLORS["equal"],   "■", "Pesos Iguales"),
        ]

        for vx, ry, color, marker_char, label in strategy_points:
            ax_fe.scatter(vx, ry, s=180, color=color, zorder=5,
                          edgecolors="white", linewidths=1.2,
                          label=label)
            ax_fe.annotate(
                label,
                (vx, ry),
                xytext=(10, 8), textcoords="offset points",
                fontsize=8, color=color, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2.5, foreground=COLORS["bg"])]
            )

        ax_fe.set_xlabel("Volatilidad anual (riesgo)", fontsize=9)
        ax_fe.set_ylabel("Retorno anual esperado", fontsize=9)
        ax_fe.set_title(
            "Frontera Eficiente de Markowitz\nPortafolios aleatorios coloreados por Ratio de Sharpe",
            fontsize=10, fontweight="bold", pad=12
        )

        ax_fe.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax_fe.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

        legend = ax_fe.legend(
            fontsize=8, facecolor="#252d3f", edgecolor=COLORS["border"],
            labelcolor=COLORS["text"], loc="lower right", framealpha=0.9
        )

        plt.tight_layout()
        st.pyplot(fig_fe)
        plt.close(fig_fe)

    with st.expander("📖 Interpretación – Frontera eficiente de Markowitz"):
        st.markdown("""
            **Interpretación analítica de la frontera eficiente:**

            La **nube de puntos** representa 2,500 portafolios con pesos aleatorios.
            El color indica el Ratio de Sharpe: colores más claros (amarillo) = mayor eficiencia.

            La **línea continua** es la frontera eficiente: el conjunto de portafolios que no pueden
            mejorar simultáneamente en retorno y riesgo. Los portafolios por debajo y a la derecha
            son ineficientes porque existe siempre una alternativa mejor.

            - **Sharpe Máximo** se ubica en el punto de mayor eficiencia riesgo–retorno.
            - **Mínima Volatilidad** se posiciona en el extremo izquierdo de menor riesgo.
            - **Pesos Iguales** actúa como referencia neutral sin optimización.
        """)

    # =====================================================================
    # RESUMEN FINAL
    # =====================================================================
    st.subheader("Comparación de estrategias")
    st.dataframe(r["df_compare"])

    st.subheader("Pesos del portafolio recomendado")
    st.dataframe(r["df_weights"])

    df_retornos = pd.DataFrame(
        {"Retorno anual esperado": [r["retornos"]["Sharpe Máximo"], r["retornos"]["Mínima Volatilidad"], r["retornos"]["Pesos Iguales"]]},
        index=["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"]
    )
    st.subheader("Ratio / retorno esperado por estrategia")
    st.dataframe(df_retornos)

# ======================================================
# ASISTENTE INTELIGENTE (GEMINI)
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

    MODEL      = "gemini-2.5-flash-lite"
    GEMINI_URL = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{MODEL}:generateContent?key={GEMINI_API_KEY}"
    )

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Pregunta sobre los tickers, riesgos o el portafolio recomendado")

    if user_question:
        st.session_state.chat_messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        results       = st.session_state.analysis_results
        best_strategy = results["best"]
        weights_dict  = results["weights"][best_strategy]

        weights_text  = "\n".join(f"- {k}: {v:.2%}" for k, v in weights_dict.items())
        asset_text    = "\n".join(
            f"- {k}: retorno anual={v['retorno_anual']:.2%}, volatilidad={v['volatilidad']:.2%}"
            for k, v in results["asset_summary"].items()
        )
        strategy_text = "\n".join(
            f"- {k}: retorno={v['retorno']:.2%}, volatilidad={v['volatilidad']:.2%}, Sharpe={v['sharpe']:.2f}, drawdown={v['drawdown']:.2%}"
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
- No expliques teoría financiera innecesaria.
- Si preguntan por cifras, usa números concretos.
- No inventes datos.
- Termina siempre la respuesta.
"""

        payload = {
            "contents": [{"role": "user", "parts": [{"text": system_prompt + "\n\nPregunta del usuario:\n" + user_question}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 900}
        }

        response = requests.post(GEMINI_URL, json=payload)
        if response.status_code != 200:
            answer = "⚠️ Error al generar la respuesta con Gemini."
        else:
            data   = response.json()
            answer = (
                data.get("candidates", [{}])[0]
                .get("content", {}).get("parts", [{}])[0]
                .get("text", "No se obtuvo respuesta.")
            )

        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)



























































