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
# RISK_FREE_RATE — valor de respaldo estático.
# Dentro de cargar_y_optimizar se sobrescribe con el promedio
# histórico real descargado de Yahoo Finance (^IRX).
# =========================
RISK_FREE_RATE = 0.045  # fallback: T-Bill 3 meses ~4.5%

# =========================
# PALETA DE COLORES COMPARTIDA
# =========================
COLORS = {
    "sharpe":  "#00d9ff",
    "minvol":  "#66ffb2",
    "equal":   "#ff9966",
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
if "years_used" not in st.session_state:
    st.session_state.years_used = 6


@st.cache_data(show_spinner="Descargando datos y optimizando portafolio…")
def cargar_y_optimizar(tickers_tuple: tuple, years: int):

    # FIX 1 — Deduplicar tickers: columnas duplicadas en la matriz de
    # covarianza causan fallos silenciosos en la optimización SLSQP.
    seen = set(); tickers = []
    for t in tickers_tuple:
        if t not in seen:
            seen.add(t); tickers.append(t)
    n = len(tickers)

    LAMBDA_REG    = 0.01
    N_SIMULATIONS = 5000
    # FIX 2 — MAX_WEIGHT: el piso original de 0.40 permite 40% en un solo
    # activo con n=20 (sin justificación). Nuevo piso: max(2/n, 0.10).
    MAX_WEIGHT    = min(0.80, max(2.0 / n, 0.10))

    # Lista de avisos para mostrar en la UI (no podemos llamar st.warning
    # dentro de @st.cache_data, se retornan y se muestran fuera).
    optimization_warnings = []

    # =====================================================================
    # 1) DESCARGA DE DATOS
    # =====================================================================
    end_date   = datetime.today()
    start_date = end_date.replace(year=end_date.year - years)

    benchmark_tickers = ["SPY", "QQQ", "URTH"]
    all_tickers = tickers + benchmark_tickers + ["^IRX"]

    raw_data = yf.download(
        all_tickers, start=start_date, end=end_date,
        auto_adjust=False, progress=False
    )
    raw_data = raw_data["Adj Close"]
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data = raw_data.droplevel(0, axis=1)
    raw_data = raw_data.sort_index().ffill()

    # =====================================================================
    # 1.1) TASA LIBRE DE RIESGO DIARIA VARIABLE (^IRX)
    # ^IRX cotiza como tasa anualizada en % (ej. 4.5 = 4.5% anual).
    # Conversión a tasa DIARIA equivalente: (IRX/100) / 252
    # =====================================================================
    if "^IRX" in raw_data.columns:
        irx_raw = raw_data["^IRX"] / 100 / 252   # tasa diaria
        irx_raw = irx_raw.replace(0, np.nan).ffill().bfill()
        rf_historical_mean = float(irx_raw.dropna().mean() * 252)  # anual
        rf_daily_series = irx_raw
    else:
        rf_daily_series    = None
        rf_historical_mean = RISK_FREE_RATE

    raw_data = raw_data.drop(columns=["^IRX"], errors="ignore")

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

    # Alineación directa sobre returns.index (único reindex necesario)
    if rf_daily_series is not None:
        rf_daily = rf_daily_series.reindex(returns.index).ffill().bfill()
        rf_daily = rf_daily.fillna(RISK_FREE_RATE / trading_days)
    else:
        rf_daily = pd.Series(RISK_FREE_RATE / trading_days, index=returns.index)

    lw = LedoitWolf()
    lw.fit(returns)
    cov_daily  = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
    cov_annual = cov_daily * trading_days

    # =====================================================================
    # 3) FUNCIONES DE OPTIMIZACIÓN
    # =====================================================================
    def performance(weights, mean_ret, cov, rf=None):
        if rf is None:
            rf = rf_historical_mean
        ret    = np.dot(weights, mean_ret)
        vol    = np.sqrt(weights.T @ cov @ weights)
        sharpe = (ret - rf) / vol if vol > 0 else 0
        return ret, vol, sharpe

    def neg_sharpe(weights):
        ret, vol_val, sharpe = performance(weights, mean_returns_annual, cov_annual,
                                           rf=rf_historical_mean)
        penalty = LAMBDA_REG * np.sum(weights ** 2)
        return -(sharpe - penalty) if vol_val > 0 else 1e6

    def vol_obj(weights):
        return np.sqrt(weights.T @ cov_annual @ weights) + LAMBDA_REG * np.sum(weights ** 2)

    def max_drawdown(series):
        return ((series / series.cummax()) - 1).min()

    # FIX CALMAR — Guardia contra drawdown nulo (imposible con datos reales
    # pero posible en periodos muy cortos donde nunca hay caída).
    def safe_calmar(ret, dd):
        return ret / abs(dd) if dd != 0 else np.nan

    x0          = np.repeat(1 / n, n)
    bounds      = tuple((0, MAX_WEIGHT) for _ in range(n))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    # =====================================================================
    # 4) OPTIMIZACIONES
    # FIX CONVERGENCIA — Si SLSQP no converge, res.x es el último iterado
    # y puede no cumplir las restricciones (pesos que no suman 1).
    # Se añade fallback a pesos iguales para garantizar que los pesos
    # entregados siempre sumen 1 y sean válidos. Las advertencias se
    # acumulan en optimization_warnings y se muestran en la UI.
    # =====================================================================
    res_sharpe = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if res_sharpe.success:
        weights_sharpe = res_sharpe.x
    else:
        weights_sharpe = x0.copy()
        optimization_warnings.append(
            "⚠️ La optimización de **Sharpe Máximo** no convergió. "
            "Se usaron pesos iguales como alternativa. "
            "Esto puede ocurrir con activos muy correlacionados o periodos de datos muy cortos."
        )

    ret_sharpe, vol_sharpe, sharpe_sharpe = performance(
        weights_sharpe, mean_returns_annual, cov_annual, rf=rf_historical_mean)

    res_minvol = minimize(vol_obj, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if res_minvol.success:
        weights_minvol = res_minvol.x
    else:
        weights_minvol = x0.copy()
        optimization_warnings.append(
            "⚠️ La optimización de **Mínima Volatilidad** no convergió. "
            "Se usaron pesos iguales como alternativa."
        )

    ret_minvol, vol_minvol, sharpe_minvol = performance(
        weights_minvol, mean_returns_annual, cov_annual, rf=rf_historical_mean)

    weights_equal = np.repeat(1 / n, n)
    ret_equal, vol_equal, sharpe_equal = performance(
        weights_equal, mean_returns_annual, cov_annual, rf=rf_historical_mean)

    # =====================================================================
    # 5) RENDIMIENTOS ACUMULADOS
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
    # FIX FRONTERA — Se cuenta el número de puntos que SLSQP no resuelve.
    # Si hay muchos fallos, el usuario recibe un aviso en la UI explicando
    # que la frontera puede estar incompleta y por qué.
    # =====================================================================
    target_returns = np.linspace(mean_returns_annual.min(), mean_returns_annual.max(), 50)
    efficient_vols, efficient_rets = [], []
    frontier_failures = 0

    for targ in target_returns:
        cons = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, targ=targ: np.dot(w, mean_returns_annual) - targ}
        )
        res = minimize(vol_obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
        if res.success:
            r, v, _ = performance(res.x, mean_returns_annual, cov_annual, rf=rf_historical_mean)
            efficient_rets.append(r)
            efficient_vols.append(v)
        else:
            frontier_failures += 1

    # Si más del 30% de los puntos de la frontera fallaron, se genera aviso.
    if frontier_failures > len(target_returns) * 0.30:
        optimization_warnings.append(
            f"⚠️ La frontera eficiente está incompleta: {frontier_failures} de "
            f"{len(target_returns)} puntos no pudieron calcularse. "
            f"Esto suele ocurrir cuando MAX_WEIGHT ({MAX_WEIGHT:.0%}) es demasiado "
            f"restrictivo para alcanzar ciertos niveles de retorno con los activos seleccionados. "
            f"La curva mostrada es válida pero puede no cubrir todo el rango de retornos posibles."
        )

    np.random.seed(0)
    n_random       = 2500
    rand_w         = np.random.dirichlet(np.ones(n), size=n_random)
    rand_rets      = rand_w @ mean_returns_annual.values
    rand_vols      = np.array([np.sqrt(w @ cov_annual.values @ w) for w in rand_w])
    rand_sharpes   = (rand_rets - rf_historical_mean) / rand_vols

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

    # FIX 4 — Ventana adaptativa: con years=3 (~756 días) la ventana fija
    # de 252 deja el primer tercio como NaN. Se adapta: min(252, n_días/3).
    roll_window = max(21, min(252, len(returns) // 3))
    rolling_vol = pd.DataFrame({
        "Sharpe Máximo":      daily_sharpe.rolling(roll_window).std() * np.sqrt(252),
        "Mínima Volatilidad": daily_minvol.rolling(roll_window).std() * np.sqrt(252),
        "Pesos Iguales":      daily_equal.rolling(roll_window).std()  * np.sqrt(252)
    })

    # FIX CALMAR — Guardia contra drawdown nulo (usa safe_calmar definida arriba).
    df_calmar = pd.DataFrame({
        "Estrategia": ["Sharpe Máximo", "Mínima Volatilidad", "Pesos Iguales"],
        "Calmar": [
            safe_calmar(ret_sharpe, dd_sharpe),
            safe_calmar(ret_minvol, dd_minvol),
            safe_calmar(ret_equal,  dd_equal),
        ]
    })

    def sortino_ratio(ret_anual, daily_ret):
        # Exceso de retorno diario respecto a la tasa libre variable
        excess       = daily_ret - rf_daily.reindex(daily_ret.index).ffill().bfill()
        downside     = np.minimum(excess, 0)
        downside_dev = np.sqrt((downside ** 2).mean()) * np.sqrt(trading_days)
        rf_annual    = float(rf_daily.mean() * trading_days)
        return (ret_anual - rf_annual) / downside_dev if downside_dev > 0 else np.nan

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

    # FIX BOOTSTRAP VECTORIZADO + OFF-BY-ONE:
    # rng.integers(low, high) excluye `high`, por lo que sin el +1 el último
    # bloque válido (start = n_obs - block_size) nunca se muestrea,
    # sesgando sutilmente las simulaciones.
    n_obs            = len(returns)
    block_size       = 20
    n_blocks_per_sim = int(np.ceil(trading_days / block_size))
    ret_matrix       = returns.values

    def _bootstrap_portfolio_fast(weights_vec, seed_offset=0):
        rng    = np.random.default_rng(42 + seed_offset)
        # +1 corrige el off-by-one: integers(low, high) excluye high
        starts = rng.integers(0, n_obs - block_size + 1,
                              size=(N_SIMULATIONS, n_blocks_per_sim))
        # Índices día a día: (N_SIMULATIONS, n_blocks_per_sim, block_size)
        idx = starts[:, :, None] + np.arange(block_size)[None, None, :]
        # Aplanar y recortar a exactamente trading_days días
        idx = idx.reshape(N_SIMULATIONS, -1)[:, :trading_days]
        # Retornos simulados: (N_SIMULATIONS, trading_days, n_activos)
        sim = ret_matrix[idx]
        return (sim @ weights_vec).sum(axis=1)

    sim_sharpe_boot = _bootstrap_portfolio_fast(weights_sharpe, seed_offset=0)
    sim_minvol_boot = _bootstrap_portfolio_fast(weights_minvol, seed_offset=1)
    sim_equal_boot  = _bootstrap_portfolio_fast(weights_equal,  seed_offset=2)

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

    # FIX 3 — Alinear índices portafolio/benchmark: distintos feriados
    # producen NaNs silenciosos en el DataFrame combinado.
    common_idx = cum_sharpe.index.intersection(benchmark_cum.index)
    comparison_cum = pd.DataFrame({
        "Sharpe Máximo":      cum_sharpe.reindex(common_idx),
        "Mínima Volatilidad": cum_minvol.reindex(common_idx),
        "Pesos Iguales":      cum_equal.reindex(common_idx),
        "S&P 500 (SPY)":      benchmark_cum["SPY"].reindex(common_idx),
        "Nasdaq 100 (QQQ)":   benchmark_cum["QQQ"].reindex(common_idx),
        "MSCI World (URTH)":  benchmark_cum["URTH"].reindex(common_idx),
    })

    # =====================================================================
    # ESTABILIDAD DE PESOS
    # FIX CONVERGENCIA EN VENTANA — Se añade fallback a pesos iguales si
    # SLSQP no converge en alguna de las sub-ventanas de análisis.
    # =====================================================================
    def optimizar_en_ventana(ret_v):
        mr  = ret_v.mean() * trading_days
        lw_ = LedoitWolf(); lw_.fit(ret_v)
        cov_ = pd.DataFrame(lw_.covariance_ * trading_days, index=ret_v.columns, columns=ret_v.columns)
        n_  = len(ret_v.columns)
        x0_ = np.repeat(1/n_, n_)
        bds = tuple((0, min(0.80, max(2.0/n_, 0.10))) for _ in range(n_))
        con = {"type": "eq", "fun": lambda w: np.sum(w)-1}
        def ns_(w):
            r_ = np.dot(w, mr); v_ = np.sqrt(w.T @ cov_ @ w)
            sh = (r_ - rf_historical_mean)/v_ if v_ > 0 else 0
            return -(sh - LAMBDA_REG*np.sum(w**2)) if v_ > 0 else 1e6
        def vo_(w):
            return np.sqrt(w.T @ cov_ @ w) + LAMBDA_REG*np.sum(w**2)
        res_s = minimize(ns_, x0_, method="SLSQP", bounds=bds, constraints=con)
        res_v = minimize(vo_, x0_, method="SLSQP", bounds=bds, constraints=con)
        ws = res_s.x if res_s.success else x0_.copy()
        wm = res_v.x if res_v.success else x0_.copy()
        return ws, wm

    stability_rows = []
    for horizon in sorted(set([3, 5, years])):  # FIX 6 — sin duplicados si years=3 o 5
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
    # FIX WEIGHTED_PERFORMANCE — Corrección doble:
    #
    # PROBLEMA 1 (bug): El bloque anterior usaba .mul(weights_series).iloc[-1],
    # que tomaba solo la ÚLTIMA FILA del DataFrame multiplicado. Como esa fila
    # siempre cae en el año más reciente (peso = 1.0), la ponderación por año
    # no tenía efecto real: equivalía a elegir por retorno acumulado total.
    #
    # PROBLEMA 2 (Error 3): El criterio anterior solo consideraba retorno,
    # ignorando el riesgo. Una estrategia con alto retorno reciente pero
    # drawdowns severos podía ser seleccionada sobre una más estable.
    #
    # SOLUCIÓN: Se calcula el RATIO DE SHARPE anual de cada estrategia por
    # año natural, usando la RF variable por año, y se promedian con pesos
    # crecientes (años recientes pesan más). Esto implementa correctamente
    # la ponderación temporal Y ajusta por riesgo simultáneamente.
    # Cualquiera de las tres estrategias puede ganar según los datos.
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

    df_strategies = pd.DataFrame({
        "Sharpe Máximo":      daily_sharpe,
        "Mínima Volatilidad": daily_minvol,
        "Pesos Iguales":      daily_equal
    })

    # Retorno logarítmico total por año natural para cada estrategia
    annual_log_returns = df_strategies.groupby(df_strategies.index.year).sum()

    # Volatilidad anualizada por año natural
    annual_vol = df_strategies.groupby(df_strategies.index.year).std() * np.sqrt(trading_days)

    # Tasa libre de riesgo anualizada por año (promedio de la serie diaria × 252)
    rf_annual_by_year = rf_daily.groupby(rf_daily.index.year).mean() * trading_days
    # Alinear al mismo índice de años que las estrategias
    rf_annual_by_year = rf_annual_by_year.reindex(annual_log_returns.index).ffill().bfill()

    # Sharpe anual por estrategia por año:
    # (retorno_log_anual - rf_anual) / vol_anual
    # Se usa retorno log anual como aproximación del retorno aritmético anual
    # (diferencia < 2% en carteras diversificadas, consistente con el resto del código)
    annual_sharpe = annual_log_returns.sub(rf_annual_by_year, axis=0).div(annual_vol)
    # Reemplazar NaN/Inf que puedan surgir si vol == 0 en algún año
    annual_sharpe = annual_sharpe.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Pesos crecientes: año más antiguo recibe 1/n_años, más reciente recibe 1.0
    n_years = len(annual_sharpe)
    year_w  = np.arange(1, n_years + 1) / n_years   # shape (n_years,)

    # Promedio ponderado del Sharpe anual: criterio ajustado por riesgo + recencia
    weighted_performance = (annual_sharpe * year_w[:, None]).mean()
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
        "rf_daily": rf_daily,
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
        "frontier_failures": frontier_failures, "frontier_total": len(target_returns),
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
        "volatilidades":{"Sharpe Máximo": vol_sharpe, "Mínima Volatilidad": vol_minvol, "Pesos Iguales": vol_equal},
        "risk_free_rate": rf_historical_mean,
        # Lista de advertencias de optimización para mostrar en la UI
        "optimization_warnings": optimization_warnings,
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
    tickers_raw = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    # FIX 10 — Advertir duplicados al usuario antes de procesar
    duplicados = [t for t in set(tickers_raw) if tickers_raw.count(t) > 1]
    if duplicados:
        st.warning(f"Tickers duplicados detectados: {', '.join(set(duplicados))}. Se eliminarán automáticamente.")
    tickers = list(dict.fromkeys(tickers_raw))  # deduplica preservando orden
    if len(tickers) < 2:
        st.error("Ingrese al menos 2 tickers.")
    else:
        try:
            resultado = cargar_y_optimizar(tuple(tickers), years)
            st.session_state.analysis_results = resultado
            st.session_state.analysis_done    = True
            st.session_state.years_used       = years
            st.session_state.chat_messages    = []
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Error: {e}")

if st.session_state.analysis_done:

    r = st.session_state.analysis_results

    # =====================================================================
    # ADVERTENCIAS DE OPTIMIZACIÓN
    # Se muestran aquí (fuera de @st.cache_data) las advertencias recogidas
    # durante el proceso de optimización. En condiciones normales con datos
    # suficientes y activos diversificados, esta sección estará vacía.
    # =====================================================================
    for warn_msg in r.get("optimization_warnings", []):
        st.warning(warn_msg)

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

    st.subheader(f"Tendencia de precios (últimos {st.session_state.years_used} años)")
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

            Este análisis permite identificar activos con comportamientos más estables
            frente a otros con mayor variabilidad en el tiempo.
        """)

    st.subheader("Comparación sistemática de estrategias")
    st.dataframe(r["df_compare"])

    with st.expander("📖 Interpretación – Comparación de estrategias"):
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

            Este análisis respalda decisiones de asignación de activos
            alineadas con el horizonte temporal y el perfil de riesgo del inversor.
        """)

    st.subheader("Volatilidad histórica móvil")
    st.line_chart(r["rolling_vol"])

    with st.expander("📖 Interpretación – Volatilidad histórica móvil"):
        st.markdown("""
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
        """)

    st.subheader("Ratio Calmar (retorno vs drawdown)")
    st.dataframe(r["df_calmar"])

    with st.expander("📖 Interpretación – Ratio Calmar"):
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

            En el contexto del presente análisis, el Ratio Calmar permite
            identificar qué estrategia ofrece un **mejor equilibrio entre
            crecimiento del capital y control de pérdidas severas**,
            reforzando la robustez del proceso de selección de portafolios.
        """)

    st.subheader("Ratio Sortino")
    st.dataframe(r["df_sortino"])

    with st.expander("📖 Interpretación – Ratio Sortino"):
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

            En el contexto del análisis comparativo, el Ratio Sortino permite
            identificar qué estrategia ofrece una **mejor compensación entre
            retorno y riesgo negativo**, aportando una visión complementaria
            y más conservadora al proceso de toma de decisiones.
        """)

    # =====================================================================
    # MONTE CARLO — KDE suavizado, gráfica única
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

        sims_dict = r["mc_simulations_mc"]
        var_dict  = r["mc_var_mc"]

        for name, color in zip(strat_names, strat_colors):
            sims = sims_dict[name]
            x_min  = np.percentile(sims, 0.5)
            x_max  = np.percentile(sims, 99.5)
            x_grid = np.linspace(x_min, x_max, 400)

            kde   = gaussian_kde(sims, bw_method=0.15)
            y_kde = kde(x_grid)

            ax.plot(x_grid, y_kde, color=color, linewidth=2.2, label=name, zorder=3)
            ax.fill_between(x_grid, y_kde, alpha=0.12, color=color, zorder=2)

            var_val = var_dict[name]
            ax.axvline(var_val, color=color, linestyle="--", linewidth=1.4, alpha=0.85,
                       label=f"VaR {name[:6]} = {var_val:.1%}", zorder=4)

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

        La simulación genera 5.000 escenarios posibles de retorno anual para cada estrategia
        utilizando la media y la matriz de covarianza estimadas. Esto permite evaluar el
        comportamiento del portafolio bajo incertidumbre futura, no solo con datos históricos.

        **¿Cómo interpretar las distribuciones?**

        - Las curvas más desplazadas hacia la derecha indican mayor retorno esperado.
        - Las distribuciones más estrechas reflejan menor volatilidad y mayor estabilidad.
        - Una mayor concentración de valores a la izquierda del cero implica mayor probabilidad de pérdida.

        **Métricas clave de riesgo extremo:**
        - **VaR 95%:** pérdida máxima esperada en el 5% de los peores escenarios.
        - **CVaR 95%:** promedio de las pérdidas en esos escenarios extremos.
        - **Probabilidad de pérdida:** porcentaje de escenarios con retorno anual negativo.

        **Lectura estratégica:**

        - El portafolio de **Sharpe Máximo** tiende a mostrar mayor retorno esperado,
          aunque con mayor dispersión y exposición a escenarios adversos.
        - El portafolio de **Mínima Volatilidad** presenta una distribución más compacta,
          reduciendo la severidad de pérdidas extremas, pero con menor potencial de crecimiento.
        - La estrategia de **Pesos Iguales** actúa como referencia neutral sin optimización específica.

        En términos prácticos, la mejor estrategia dependerá del perfil del inversor:

        - Si se prioriza maximizar retorno ajustado por riesgo → **Sharpe Máximo**.
        - Si se prioriza estabilidad y control de pérdidas extremas → **Mínima Volatilidad**.

        La decisión óptima surge del equilibrio entre retorno esperado y tolerancia al riesgo extremo.
        """)

    # =====================================================================
    # ESTABILIDAD DE PESOS
    # =====================================================================
    if not r["df_stability"].empty:
        st.subheader("Estabilidad de pesos por horizonte temporal")
        st.dataframe(r["df_stability"], use_container_width=True)

    with st.expander("📖 Interpretación – Estabilidad de pesos por horizonte temporal"):
        st.markdown("""
            **¿Qué muestra esta tabla?**

            Esta tabla re-optimiza el portafolio tres veces usando ventanas de tiempo
            distintas: los últimos 3 años, los últimos 5 años y el periodo completo
            seleccionado. El objetivo es verificar si los pesos óptimos cambian mucho
            o poco dependiendo del periodo de datos utilizado.

            **¿Por qué es importante?**

            Uno de los problemas más conocidos del modelo de Markowitz es que sus
            resultados pueden ser muy sensibles al periodo de datos elegido. Si los
            pesos óptimos cambian drásticamente según la ventana de tiempo, significa
            que el modelo está aprovechando patrones históricos específicos que podrían
            no repetirse en el futuro. Esto se conoce como **sobreajuste** y es una
            señal de alerta.

            **¿Cómo interpretar los resultados?**

            - Si los pesos de un activo son **similares en los tres horizontes**
              (por ejemplo, siempre entre 20% y 25%), la estrategia es **robusta y
              confiable**. El modelo llega a la misma conclusión sin importar qué
              periodo se analice.
            - Si los pesos varían **de forma significativa** entre horizontes (por
              ejemplo, 5% en 3 años pero 45% en el periodo completo), la asignación
              es **inestable**. Esto indica que ese activo tuvo un comportamiento
              atípico en algún periodo puntual que distorsiona el resultado.
            - Los pesos de **Sharpe Máximo** tienden a ser más inestables que los de
              **Mínima Volatilidad**, ya que el Sharpe depende tanto del retorno como
              de la volatilidad, dos variables que cambian más en el tiempo.

            **Lectura recomendada para la defensa técnica:**

            Si los pesos son estables entre horizontes, esto demuestra que la solución
            no es un artefacto del periodo de datos elegido, sino una señal consistente
            del mercado. Es uno de los argumentos más sólidos para defender la validez
            del modelo frente a críticas metodológicas.

            Si existen variaciones importantes, se recomienda priorizar la estrategia de
            **Mínima Volatilidad**, que tiende a producir asignaciones más estables y
            predecibles a lo largo del tiempo.
        """)

    # =====================================================================
    # COVID 2020
    # =====================================================================
    st.subheader("Comportamiento en periodo de crisis (COVID 2020)")
    crisis = (cum_sharpe.index.year == 2020)
    if crisis.sum() > 0:
        st.line_chart(pd.DataFrame({
            "Sharpe Máximo": cum_sharpe[crisis],
            "Mínima Volatilidad": cum_minvol[crisis],
            "Pesos Iguales": cum_equal[crisis]
        }))
    else:
        st.info("El horizonte seleccionado no incluye datos de 2020.")

    with st.expander("📖 Interpretación – Comportamiento en crisis (COVID 2020)"):
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

            Este análisis refuerza la idea de que la eficiencia
            riesgo–retorno debe evaluarse no solo en condiciones normales,
            sino también bajo escenarios adversos.
        """)

    # =====================================================================
    # BENCHMARKS
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
    # MEJOR PORTAFOLIO
    # El criterio ahora usa Sharpe ponderado por recencia (no solo retorno),
    # ajustando por riesgo y evitando seleccionar estrategias con alto
    # retorno pero drawdowns severos.
    # =====================================================================
    st.subheader("Interpretación automática del mejor portafolio")
    st.dataframe(r["weighted_performance"].rename("Sharpe_Ponderado_Recencia"))

    if best == "Pesos Iguales":
        st.markdown(
            "### Mejor portafolio: Pesos Iguales\n\n"
            "El análisis del **Ratio de Sharpe anual ponderado por recencia** "
            "muestra que esta estrategia ha sido la **más robusta y consistente** "
            "en términos de retorno ajustado por riesgo.\n\n"
            "- Menor dependencia de supuestos estadísticos.\n"
            "- Mejor desempeño agregado a lo largo del tiempo.\n"
            "- Alta estabilidad frente a cambios de mercado."
        )
    elif best == "Sharpe Máximo":
        st.markdown(
            "### Mejor portafolio: Sharpe Máximo\n\n"
            "El análisis del Ratio de Sharpe anual ponderando más los años recientes "
            "indica que esta estrategia ofrece el mejor equilibrio riesgo–retorno "
            "en el comportamiento histórico reciente."
        )
    else:
        st.markdown(
            "### Mejor portafolio: Mínima Volatilidad\n\n"
            "El análisis del Ratio de Sharpe ponderado por recencia muestra que "
            "esta estrategia destaca por su estabilidad y eficiencia ajustada por riesgo."
        )

    st.success(f"Portafolio recomendado según Sharpe ponderado por recencia: {best}")

    # =====================================================================
    # PESOS ÓPTIMOS
    # =====================================================================
    st.subheader("Pesos óptimos del portafolio recomendado")

    df_weights = r["df_weights"]
    st.dataframe(df_weights)

    _pw1, _pw2, _pw3 = st.columns([0.3, 2.5, 0.3])
    with _pw2:
        tickers_w = df_weights["Ticker"].tolist()
        pesos_w   = df_weights["Peso (%)"].tolist()
        n_w       = len(tickers_w)

        palette = [
            mcolors.to_hex(plt.cm.cool(0.15 + 0.7 * i / max(n_w - 1, 1)))
            for i in range(n_w)
        ]

        fig_w, ax_w = plt.subplots(figsize=(9, max(3.5, n_w * 0.7)))
        apply_dark_style(fig_w, ax_w)

        bars = ax_w.barh(
            tickers_w, pesos_w,
            color=palette, edgecolor=COLORS["bg"], linewidth=0.8,
            height=0.55
        )

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

    with st.expander("📖 Interpretación – Rendimiento acumulado por acción"):
        st.markdown("""
            **Interpretación:**

            El rendimiento acumulado refleja cómo habría evolucionado una inversión inicial
            en cada activo si se hubiera mantenido durante todo el periodo de análisis.

            - Curvas más empinadas indican mayor crecimiento del capital.
            - Activos con mayor volatilidad suelen mostrar trayectorias más irregulares.
            - Diferencias significativas entre curvas evidencian distintos perfiles
              de riesgo y rentabilidad.

            Este gráfico facilita la comparación directa del desempeño histórico
            entre los activos analizados.
        """)

    st.subheader("Retornos diarios de los activos")
    st.line_chart(returns)

    with st.expander("📖 Interpretación – Retornos diarios de los activos"):
        st.markdown("""
            **Interpretación:**

            Este gráfico muestra los retornos porcentuales diarios de cada activo,
            evidenciando la volatilidad de corto plazo.

            - Picos positivos o negativos representan movimientos abruptos del mercado.
            - Mayor dispersión implica mayor riesgo.
            - Periodos de alta concentración de picos suelen coincidir con crisis financieras
              o eventos macroeconómicos relevantes.

            Este análisis es clave para evaluar el riesgo diario asumido por el inversor.
        """)

    st.subheader("Retornos diarios por activo")
    for ticker in returns.columns:
        st.markdown(f"### {ticker}")
        st.line_chart(returns[[ticker]])

    with st.expander("📖 Interpretación – Retornos diarios por activo individual"):
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
    # FRONTERA EFICIENTE — GRÁFICO PREMIUM
    # =====================================================================
    st.subheader("Frontera eficiente (Retorno vs Volatilidad)")

    # Aviso de frontera incompleta si aplica (Error 1)
    if r["frontier_failures"] > r["frontier_total"] * 0.30:
        st.warning(
            f"⚠️ La frontera eficiente está incompleta: {r['frontier_failures']} de "
            f"{r['frontier_total']} puntos no pudieron calcularse debido a restricciones "
            f"de peso (MAX_WEIGHT = {min(0.80, max(2.0/len(tickers), 0.10)):.0%}). "
            f"La curva es válida pero puede no cubrir todo el rango posible de retornos."
        )

    _fe1, _fe2, _fe3 = st.columns([0.2, 3, 0.2])
    with _fe2:
        fig_fe, ax_fe = plt.subplots(figsize=(10, 6))
        apply_dark_style(fig_fe, ax_fe)

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

        ax_fe.plot(
            r["efficient_vols"], r["efficient_rets"],
            color=COLORS["sharpe"], linewidth=2.5, zorder=3,
            label="Frontera eficiente",
            path_effects=[pe.withStroke(linewidth=5, foreground="#00d9ff20")]
        )

        strategy_points = [
            (r["vol_sharpe"], r["ret_sharpe"], COLORS["sharpe"],  "Sharpe Máximo"),
            (r["vol_minvol"], r["ret_minvol"], COLORS["minvol"],  "Mínima Volatilidad"),
            (r["vol_equal"],  r["ret_equal"],  COLORS["equal"],   "Pesos Iguales"),
        ]

        for vx, ry, color, label in strategy_points:
            ax_fe.scatter(vx, ry, s=180, color=color, zorder=5,
                          edgecolors="white", linewidths=1.2, label=label)
            ax_fe.annotate(
                label, (vx, ry),
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
        ax_fe.legend(fontsize=8, facecolor="#252d3f", edgecolor=COLORS["border"],
                     labelcolor=COLORS["text"], loc="lower right", framealpha=0.9)

        plt.tight_layout()
        st.pyplot(fig_fe)
        plt.close(fig_fe)

    with st.expander("📖 Interpretación – Frontera eficiente de Markowitz"):
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

            La **nube de puntos** representa 2,500 portafolios con pesos aleatorios.
            El color indica el Ratio de Sharpe: colores más claros (amarillo) = mayor eficiencia.

            La ubicación de las estrategias analizadas sobre la frontera
            permite identificar su perfil:
            - El portafolio de **Sharpe Máximo** se sitúa en una zona de
              mayor eficiencia, priorizando la rentabilidad ajustada por riesgo.
            - La estrategia de **Mínima Volatilidad** se posiciona en el
              extremo de menor riesgo, sacrificando retorno esperado.
            - La asignación de **Pesos Iguales** actúa como referencia
              neutral, sin optimización explícita.

            Esta visualización facilita la comprensión del trade-off
            riesgo–retorno y constituye una herramienta central para
            la toma de decisiones de inversión.
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

    st.info(
        f"📌 Tasa libre de riesgo utilizada (^IRX, diaria variable — promedio anualizado del periodo): "
        f"**{r['risk_free_rate']:.4%}**. "
        f"El Sortino y el Sharpe ex-post se calculan con la tasa variable día a día."
    )

# ======================================================
# ASISTENTE INTELIGENTE (GEMINI)
# ======================================================
st.divider()
st.subheader("🤖 Asistente inteligente del portafolio")

if not st.session_state.analysis_done:
    st.info("Ejecuta primero la optimización para habilitar el asistente.")
else:
    # FIX 9 — st.secrets.get() lanza FileNotFoundError en local sin secrets.toml
    GEMINI_API_KEY = None
    try:
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass
    GEMINI_API_KEY = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
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
- Cada párrafo debe aportar información distinta (no repetir ideas).
- No expliques teoría financiera innecesaria.
- Si aplica, menciona brevemente riesgo y retorno.
- Si preguntan por cifras, usa números concretos.
- No inventes datos.
- Termina siempre la respuesta.
"""

        payload = {
            "contents": [{"role": "user", "parts": [{"text": system_prompt + "\n\nPregunta del usuario:\n" + user_question}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 900}
        }

        try:
            response = requests.post(GEMINI_URL, json=payload, timeout=30)
            response.raise_for_status()
            data   = response.json()
            answer = (
                data.get("candidates", [{}])[0]
                .get("content", {}).get("parts", [{}])[0]
                .get("text", "No se obtuvo respuesta.")
            )
        except requests.exceptions.Timeout:
            answer = "⚠️ El asistente tardó demasiado en responder. Inténtalo de nuevo."
        except requests.exceptions.RequestException as e:
            answer = f"⚠️ Error de conexión con Gemini: {e}"

        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)




































































