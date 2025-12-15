import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.optimize import minimize


st.markdown("""
<style>
h2 {
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    color: #2b2d42 !important;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Análisis de Portafolios",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #f9fafc;
}

section[data-testid="stSidebar"] {
    background-color: #eef2f7;
}

div[data-testid="stSlider"] > div > div > div:nth-child(2) > div {
    background-color: #1f77b4 !important;
}


div[data-testid="stSlider"] div[role="slider"] {
    background-color: #1f77b4 !important;
    border-color: #1f77b4 !important;
}


div[data-testid="stRadio"] > label > div:first-child {
    border: 2px solid #7a3db8 !important;
}
div[data-testid="stRadio"] > label > div:first-child:hover {
    border-color: #a86dd8 !important;
}
div[data-testid="stRadio"] > label > div[aria-checked="true"] {
    background-color: #7a3db8 !important;
    border-color: #7a3db8 !important;
}

div.stButton > button {
    background-color: #7a3db8 !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    border: none !important;
    font-weight: 600 !important;
}
div.stButton > button:hover {
    background-color: #5c2c91 !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
.stApp {
    background-color: #f9fafc;
}

section[data-testid="stSidebar"] {
    background-color: #eef2f7;
}

div[data-testid="stSlider"] > div > div > div:nth-child(2) > div {
    background-color: #1f77b4 !important;
}

div[data-testid="stSlider"] div[role="slider"] {
    background-color: #1f77b4 !important;
    border-color: #1f77b4 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

div[data-testid="stRadio"] > label > div:first-child {
    border: 2px solid #7a3db8 !important;
}

div[data-testid="stRadio"] > label > div:first-child:hover {
    border-color: #a86dd8 !important;
}

div[data-testid="stRadio"] > label > div[aria-checked="true"] {
    background-color: #7a3db8 !important;
    border-color: #7a3db8 !important;
}

</style>
""", unsafe_allow_html=True)






# 1. Datos de los Tickers por Regiones y Sectores
# Se muestran los diversos ETF de Regiones como EUA, Cánada, Europa, entre otros.
# Se muestran los de Sectores como los de Comunicaciones, consumo discrecional, consumo básico, energía, etc.

TICKERS_REGIONES = ["SPLG", "EWC", "IEUR", "EEM", "EWJ"]

TICKERS_SECTORES = [
    "XLC","XLY","XLP","XLE","XLF",
    "XLV","XLI","XLB","XLRE","XLK","XLU"
]
#Se tienen los pesos por sector y región, datos que se dan en el pdf del proyecto.

PESOS_REGIONES = {
    "SPLG":0.7062,
    "EWC":0.0323,
    "IEUR":0.1176,
    "EEM":0.0902,
    "EWJ":0.0537
}
PESOS_SECTORES = {
    "XLC": 0.0999,
    "XLY": 0.1025,
    "XLP": 0.0482,
    "XLE":  0.0295,
    "XLF":  0.1307,
    "XLV":  0.0958,
    "XLI":  0.0809,
    "XLB":  0.0166,
    "XLRE": 0.0187,
    "XLK":  0.3535,
    "XLU":  0.0237
}

# 2.Se presentan dos tipos de funciones

# 2.1. Funciones de las métricas vistas en clase


def beta(port, benchmark):
    cov = np.cov(port, benchmark)[0, 1]
    var = np.var(benchmark)
    return cov / var

def media(r):
    return r.mean()

def volatilidad(r):
    return r.std()

def sharpe(r, rf=0.0):
    excess = r - rf/252
    return np.sqrt(252) * excess.mean() / excess.std()

def sortino(r, rf=0.0):
    excess = r - rf/252
    downside = excess[excess < 0].std()
    return np.sqrt(252) * excess.mean() / downside

def max_drawdown(r):
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()

def var_95(r):
    return np.percentile(r, 5)

def cvar_95(r):
    v = var_95(r)
    return r[r <= v].mean()

def sesgo(r):
    return skew(r)

def curtosis(r):
    return kurtosis(r)

def calcular_metricas(serie, rf=0.05):
    return {
        "Media diaria": media(serie),
        "Volatilidad diaria": volatilidad(serie),
        "Sharpe (5% rf)": sharpe(serie, rf=rf),
        "Sortino (5% rf)": sortino(serie, rf=rf),
        "Max Drawdown": max_drawdown(serie),
        "VaR 95%": var_95(serie),
        "CVaR 95%": cvar_95(serie),
        "Skew": sesgo(serie),
        "Kurtosis": curtosis(serie),
    }


# 2.2. Funciones que se encargan de:
#  Descargar precios, Calcular rendimientos, Construir portafolios a partir de pesos
#  Obtener medias y covarianzas.


def descargar_precios(tickers, years=4):
    data = yf.download(tickers, period=f"{years}y")["Close"]
    return data

# Construye un portafolio usando retornos ya calculados y pesos arbitrarios.


def construir_portafolio_arbitrario(retornos, pesos_dict):
    columnas = [t for t in pesos_dict.keys() if t in retornos.columns]
    r = retornos[columnas]
    pesos = np.array([pesos_dict[t] for t in columnas], dtype=float)
    pesos = pesos / pesos.sum()
    portafolio = (r * pesos).sum(axis=1)
    return portafolio

def obtener_mu_cov(retornos):
    mu = retornos.mean()          # media diaria
    cov = retornos.cov()          # covarianza diaria
    return mu, cov

#  Calcula rendimientos y portafolio dada una tabla de precios y pesos (diccionario).

def construir_portafolio(data_precios, pesos_dict):
    retornos = data_precios.pct_change().dropna()

    columnas = [t for t in pesos_dict.keys() if t in retornos.columns]
    retornos = retornos[columnas]

    pesos = np.array([pesos_dict[t] for t in columnas], dtype=float)
    pesos = pesos / pesos.sum()

    portafolio = (retornos * pesos).sum(axis=1)
    return retornos, portafolio



# 4. Optimización de portafolio

def port_vol(w, cov):
    w = np.array(w)
    return np.sqrt(w.T @ cov.values @ w)

def port_ret(w, mu):
    w = np.array(w)
    return w @ mu.values

def min_var_portfolio(mu, cov):
    n = len(mu)
    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    cons = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
    )

    def obj(w):
        return w.T @ cov.values @ w

    res = minimize(obj, w0, bounds=bounds, constraints=cons)
    return res.x if res.success else None

def max_sharpe_portfolio(mu, cov, rf_anual):
    n = len(mu)
    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    rf_diario = rf_anual / 252.0

    def neg_sharpe(w):
        r_p = port_ret(w, mu)
        v_p = port_vol(w, cov)
        if v_p == 0:
            return 1e6
        return - (r_p - rf_diario) / v_p

    cons = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
    )

    res = minimize(neg_sharpe, w0, bounds=bounds, constraints=cons)
    return res.x if res.success else None

def markowitz_target_portfolio(mu, cov, target_anual):
    n = len(mu)
    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    target_diario = target_anual / 252.0

    def obj(w):
        return w.T @ cov.values @ w

    cons = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'eq', 'fun': lambda w: port_ret(w, mu) - target_diario},
    )

    res = minimize(obj, w0, bounds=bounds, constraints=cons)
    return res.x if res.success else None

    # --------------------------- BLACK-LITTERMAN ----------------------------------------

def black_litterman(data, pesos_mercado, visiones, tau=0.05, delta=2.5, metodo_post="Markowitz",
                    rendimiento_objetivo=None, peso_min=0, peso_max=1):
    """
    metodo_post : str
        Método de optimización post-BL ("Mínima Varianza", "Máximo Sharpe", "Markowitz")
    rendimiento_objetivo : float
        Solo si metodo_post == "Markowitz"
    peso_min, peso_max : float
        Restricciones de pesos

    Retorna:
    --------
    tuple: (rendimientos_implicitos, rendimientos_bl, pesos_optimizados, metricas)
    """

    # --- 1. PREPARACIÓN DE DATOS ---
    returns = data.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    activos = list(data.columns)
    n = len(activos)

    # Convertir pesos de mercado a array (normalizado)
    w_market = np.array([pesos_mercado[activo] / 100 for activo in activos])
    w_market = w_market / w_market.sum()

    # CALCULAR RENDIMIENTOS IMPLÍCITOS (EQUILIBRIO) 
    pi = delta * (cov_matrix @ w_market)

    # CONSTRUIR MATRIZ P Y VECTORES Q
    k = len(visiones)  # Número de visiones

    if k == 0:
        # Sin visiones, usar directamente los rendimientos implícitos
        mu_bl = pi
        P = np.zeros((1, n))
        Q = np.zeros(1)
    else:
        P = np.zeros((k, n))
        Q = np.zeros(k)
        omega_diag = np.zeros(k)

        for i, vista in enumerate(visiones):
            activo_1 = vista['activo_1']
            activo_2 = vista['activo_2']
            valor = vista['valor'] / 100  # Convertir porcentaje a decimal
            confianza = vista['confianza']
            operador = vista['operador']

            # Índice del activo 1
            idx_1 = activos.index(activo_1)

            if activo_2 == "Rendimiento Absoluto":
                # Vista absoluta: R_activo1 = valor
                P[i, idx_1] = 1.0
                Q[i] = valor
            else:
                # Vista relativa: R_activo1 - R_activo2 = valor
                idx_2 = activos.index(activo_2)

                if operador == ">":
                    # activo_1 > activo_2 por valor%
                    P[i, idx_1] = 1.0
                    P[i, idx_2] = -1.0
                    Q[i] = valor
                elif operador == "<":
                    # activo_1 < activo_2 por valor%
                    P[i, idx_1] = 1.0
                    P[i, idx_2] = -1.0
                    Q[i] = -valor
                else:  # "="
                    # activo_1 = activo_2 (diferencia = 0)
                    P[i, idx_1] = 1.0
                    P[i, idx_2] = -1.0
                    Q[i] = 0.0

            # Calcular incertidumbre de la vista basada en confianza
            # Confianza más alta = menor incertidumbre
            # Ω_i = τ * P_i * Σ * P_i^T / confianza_normalizada
            confianza_normalizada = confianza / 10.0  # 1-10 -> 0.1-1.0
            omega_diag[i] = tau * (P[i] @ cov_matrix @ P[i].T) / (confianza_normalizada ** 2)

        Omega = np.diag(omega_diag)

        # --- 4. CALCULAR RENDIMIENTOS ESPERADOS BLACK-LITTERMAN ---
        # μ_BL = [(τΣ)^-1 + P'Ω^-1P]^-1 [(τΣ)^-1π + P'Ω^-1Q]

        tau_sigma = tau * cov_matrix
        tau_sigma_inv = np.linalg.inv(tau_sigma)
        omega_inv = np.linalg.inv(Omega)

        # Término izquierdo: [(τΣ)^-1 + P'Ω^-1P]^-1
        left_term = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)

        # Término derecho: [(τΣ)^-1π + P'Ω^-1Q]
        right_term = tau_sigma_inv @ pi + P.T @ omega_inv @ Q

        # Rendimientos esperados BL
        mu_bl = left_term @ right_term

    # Crear DataFrame temporal con los rendimientos BL para usar la función de optimización

    # Sustituir los retornos medios en la optimización
    mean_returns_bl = pd.Series(mu_bl, index=activos)

    # Función auxiliar para optimización con BL
    def portfolio_performance_bl(weights):
        ret = np.dot(weights, mean_returns_bl)
        vol = np.sqrt(weights @ cov_matrix @ weights.T)
        return ret, vol

    # Optimización según método
    if metodo_post == "Mínima Varianza":
        x0 = np.ones(n)/n
        bounds = tuple((peso_min, peso_max) for _ in range(n))
        constraints = ({'type':'eq','fun':lambda w: np.sum(w)-1})

        result = op.minimize(
            lambda w: portfolio_performance_bl(w)[1],
            x0, constraints=constraints, bounds=bounds, method='SLSQP'
        )
        pesos_opt = result.x
        ret_opt, vol_opt = portfolio_performance_bl(pesos_opt)

    elif metodo_post == "Máximo Sharpe":
        x0 = np.ones(n)/n
        bounds = tuple((peso_min, peso_max) for _ in range(n))
        constraints = ({'type':'eq','fun':lambda w: np.sum(w)-1})

        def neg_sharpe(w):
            r, vol = portfolio_performance_bl(w)
            return -(r - 0.0) / vol

        result = op.minimize(
            neg_sharpe, x0, constraints=constraints, bounds=bounds, method='SLSQP'
        )
        pesos_opt = result.x
        ret_opt, vol_opt = portfolio_performance_bl(pesos_opt)

    else:  # Markowitz
        if rendimiento_objetivo is None:
            rendimiento_objetivo = mean_returns_bl.mean()

        rendimiento_objetivo = rendimiento_objetivo / 100.0

        x0 = np.ones(n) / n
        bounds = tuple((peso_min, peso_max) for _ in range(n))

        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns_bl) - rendimiento_objetivo}
        )

        def variance(w):
            return w @ cov_matrix @ w.T

        result = op.minimize(
            variance, x0, method='SLSQP', bounds=bounds, constraints=constraints
        )
        pesos_opt = result.x
        ret_opt, vol_opt = portfolio_performance_bl(pesos_opt)

    # Convertir a porcentajes y diccionarios
    rendimientos_implicitos = {activo: round(pi[i] * 100, 2) for i, activo in enumerate(activos)}
    rendimientos_bl = {activo: round(mu_bl[i] * 100, 2) for i, activo in enumerate(activos)}
    pesos_optimizados = {activo: round(pesos_opt[i] * 100, 2) for i, activo in enumerate(activos)}

    # Calcular métricas
    sharpe_ratio = (ret_opt - 0.0) / vol_opt if vol_opt != 0 else 0

    # Calcular tracking error vs benchmark
    w_bench = np.array([pesos_mercado[activo] / 100 for activo in activos])
    w_bench = w_bench / w_bench.sum()
    ret_bench = np.dot(w_bench, mean_returns_bl)
    tracking_error = abs(ret_opt - ret_bench)

    metricas = {
        'rendimiento': round(ret_opt * 100, 2),
        'volatilidad': round(vol_opt * 100, 2),
        'sharpe': round(sharpe_ratio, 4),
        'tracking_error': round(tracking_error * 100, 2)
    }

    return rendimientos_implicitos, rendimientos_bl, pesos_optimizados, metricas


# 5. App

def main():
    st.markdown("## 📈 Cálculo de Métricas de Portafolios")

    st.markdown("""
    <div style="
        background-color:#ffffff;
        padding:18px;
        border-radius:10px;
        border-left:4px solid #7a3db8;
        box-shadow:0px 2px 6px rgba(0,0,0,0.05);
        margin-bottom:20px;">
        <p style="font-size:1rem; color:#333;">
            Aplicación para analizar portafolios de <b>Regiones</b> y <b>Sectores</b>:
        </p>
        <ul style="font-size:1rem; color:#333; line-height:1.6;">
            <li>Benchmark (pesos dados)</li>
            <li>Portafolio arbitrario (definido por el usuario)</li>
            <li>Portafolios optimizados: mínima varianza, máximo Sharpe y Markowitz con rendimiento objetivo.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)



    # Sidebar: parámetros generales
    estrategia = st.sidebar.selectbox("Estrategia", ["Regiones", "Sectores"])
    years = st.sidebar.slider("Años de datos históricos", 1, 10, 4)
    rf_anual = st.sidebar.number_input("Tasa libre de riesgo anual (rf)", 0.0, 0.20, 0.05, step=0.005)

    modo = st.sidebar.radio(
        "Portafios a calcular",
        ["Solo benchmark", "Solo arbitrario", "Benchmark y arbitrario", "Optimización"],
        index=2
    )

    if estrategia == "Regiones":
        tickers = TICKERS_REGIONES
        pesos_bench = PESOS_REGIONES
    else:
        tickers = TICKERS_SECTORES
        pesos_bench = PESOS_SECTORES

    # Pesos arbitrarios (si aplica)
    pesos_arbitrarios = {}
    if modo in ["Solo arbitrario", "Benchmark y arbitrario"]:
        st.sidebar.markdown("### Pesos portafolio arbitrario")
        st.sidebar.caption("Introduce los pesos (se normalizan automáticamente para sumar 1).")
        for t in tickers:
            default = float(pesos_bench.get(t, 0.0))
            w = st.sidebar.number_input(
                f"Peso {t}",
                min_value=0.0,
                max_value=1.0,
                value=default,
                step=0.01
            )
            pesos_arbitrarios[t] = w

    # Parámetro para Markowitz si estamos en optimización
    target_anual = None
    if modo == "Optimización":
        st.sidebar.markdown("### Markowitz – Rendimiento objetivo")
        target_anual = st.sidebar.number_input(
            "Rendimiento objetivo anual (en decimal, ej. 0.10 = 10%)",
            min_value=0.0,
            max_value=0.5,
            value=0.10,
            step=0.01
        )

    st.subheader(f"Estrategia seleccionada: {estrategia}")

    if st.button("Calcular métricas"):
        with st.spinner("Descargando datos y calculando…"):
            data = descargar_precios(tickers, years)
            retornos, portafolio_bench = construir_portafolio(data, pesos_bench)

            mu, cov = obtener_mu_cov(retornos)

            # Portafolio arbitrario (si aplica)
            portafolio_arbi = None
            if modo in ["Solo arbitrario", "Benchmark y arbitrario"]:
                if sum(pesos_arbitrarios.values()) == 0:
                    st.error("Los pesos del portafolio arbitrario no pueden ser todos cero.")
                    return
                portafolio_arbi = construir_portafolio_arbitrario(retornos, pesos_arbitrarios)

            # Portafolios optimizados (si aplica)
            w_minvar = w_maxsharpe = w_markowitz = None
            port_minvar = port_maxsharpe = port_markowitz = None

            if modo == "Optimización":
                # Mínima varianza
                w_minvar = min_var_portfolio(mu, cov)
                # Máximo Sharpe
                w_maxsharpe = max_sharpe_portfolio(mu, cov, rf_anual)
                # Markowitz con retorno objetivo
                w_markowitz = markowitz_target_portfolio(mu, cov, target_anual)

                if w_minvar is None or w_maxsharpe is None or w_markowitz is None:
                    st.error("No se pudo encontrar una solución óptima para alguna de las optimizaciones.")
                    return

                cols = retornos.columns
                # Series de retornos de cada portafolio optimizado
                port_minvar = (retornos[cols] * w_minvar).sum(axis=1)
                port_maxsharpe = (retornos[cols] * w_maxsharpe).sum(axis=1)
                port_markowitz = (retornos[cols] * w_markowitz).sum(axis=1)

        # ----------------------------
        # Mostrar datos básicos
        # ----------------------------
        st.markdown("### Precios de cierre (últimos 10 registros)")
        st.dataframe(data.tail(10))

        st.markdown("### Retornos diarios (primeros 5 registros)")
        st.dataframe(retornos.head())

        
                # ----------------------------
        # Métricas
        # ----------------------------
        if modo != "Optimización":
            metrics_dict = {}

            if modo in ["Solo benchmark", "Benchmark y arbitrario"]:
                metrics_dict["Benchmark"] = calcular_metricas(portafolio_bench, rf=rf_anual)

            if modo in ["Solo arbitrario", "Benchmark y arbitrario"] and portafolio_arbi is not None:
                metrics_dict["Arbitrario"] = calcular_metricas(portafolio_arbi, rf=rf_anual)

            df_metrics = pd.DataFrame(metrics_dict)
            st.markdown("### Métricas de portafolios")
            st.dataframe(df_metrics.style.format("{:.6f}"))

            # ----------------------------
            # Rendimiento acumulado
            # ----------------------------
            st.markdown("### Rendimiento acumulado")

            df_cum = pd.DataFrame()
            if "Benchmark" in metrics_dict:
                df_cum["Benchmark"] = (1 + portafolio_bench).cumprod()
            if "Arbitrario" in metrics_dict and portafolio_arbi is not None:
                df_cum["Arbitrario"] = (1 + portafolio_arbi).cumprod()

            st.line_chart(df_cum)

        else:
            # Métricas de los portafolios optimizados
            metrics_opt = {
                "MinVar": calcular_metricas(port_minvar, rf=rf_anual),
                "MaxSharpe": calcular_metricas(port_maxsharpe, rf=rf_anual),
                "Markowitz": calcular_metricas(port_markowitz, rf=rf_anual),
            }

       


            df_metrics_opt = pd.DataFrame(metrics_opt)
            st.markdown("### Métricas de portafolios optimizados")
            st.dataframe(df_metrics_opt.style.format("{:.6f}"))

            # Pesos de cada portafolio
            weights_df = pd.DataFrame(
                {
                    "MinVar": w_minvar,
                    "MaxSharpe": w_maxsharpe,
                    "Markowitz": w_markowitz,
                },
                index=retornos.columns
            )
            st.markdown("### Pesos de los portafolios optimizados")
            st.dataframe(weights_df.style.format("{:.4f}"))

            # Rendimiento acumulado de optimizados
            st.markdown("### Rendimiento acumulado – Portafolios optimizados")
            df_cum_opt = pd.DataFrame({
                "MinVar": (1 + port_minvar).cumprod(),
                "MaxSharpe": (1 + port_maxsharpe).cumprod(),
                "Markowitz": (1 + port_markowitz).cumprod(),
            })
            st.line_chart(df_cum_opt)

if __name__ == "__main__":
    main()






