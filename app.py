import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --------------------------------------------------
# Configuración básica de la página
# --------------------------------------------------
st.set_page_config(
    page_title="Creación de portafolios",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# 1. Universo de inversión y pesos de benchmark
# --------------------------------------------------

TICKERS_REGIONES = ["SPLG", "EWC", "IEUR", "EEM", "EWJ"]

TICKERS_SECTORES = [
    "XLC", "XLY", "XLP", "XLE", "XLF",
    "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"
]

PESOS_REGIONES = {
    "SPLG": 0.7062,
    "EWC": 0.0323,
    "IEUR": 0.1176,
    "EEM": 0.0902,
    "EWJ": 0.0537,
}

PESOS_SECTORES = {
    "XLC": 0.0999,
    "XLY": 0.1025,
    "XLP": 0.0482,
    "XLE": 0.0295,
    "XLF": 0.1307,
    "XLV": 0.0958,
    "XLI": 0.0809,
    "XLB": 0.0166,
    "XLRE": 0.0187,
    "XLK": 0.3535,
    "XLU": 0.0237,
}

# --------------------------------------------------
# 2. Funciones auxiliares
# --------------------------------------------------


@st.cache_data
def descargar_precios(tickers, years=4):
    """Descarga precios de cierre ajustados con yfinance."""
    data = yf.download(tickers, period=f"{years}y")["Close"]
    return data


def construir_portafolio(data_precios, pesos_dict):
    """
    A partir de precios y un diccionario de pesos, regresa:
    - DataFrame de rendimientos de cada activo
    - Serie con los rendimientos del portafolio
    """
    retornos = data_precios.pct_change().dropna()

    columnas = [t for t in pesos_dict.keys() if t in retornos.columns]
    retornos = retornos[columnas]

    pesos = np.array([pesos_dict[t] for t in columnas], dtype=float)
    pesos = pesos / pesos.sum()

    portafolio = (retornos * pesos).sum(axis=1)
    return retornos, portafolio


def construir_portafolio_arbitrario(retornos, pesos_dict):
    """Construye portafolio usando rendimientos ya calculados y pesos arbitrarios."""
    columnas = [t for t in pesos_dict.keys() if t in retornos.columns]
    r = retornos[columnas]
    pesos = np.array([pesos_dict[t] for t in columnas], dtype=float)
    pesos = pesos / pesos.sum()
    portafolio = (r * pesos).sum(axis=1)
    return portafolio


def obtener_mu_cov(retornos):
    """Media y matriz de covarianza diarias de los activos."""
    mu = retornos.mean()
    cov = retornos.cov()
    return mu, cov

# --------------------------------------------------
# 3. Métricas de portafolio
# --------------------------------------------------


def media(r):
    return r.mean()


def volatilidad(r):
    return r.std()


def sharpe(r, rf=0.0):
    excess = r - rf / 252
    return np.sqrt(252) * excess.mean() / excess.std()


def sortino(r, rf=0.0):
    excess = r - rf / 252
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

# --------------------------------------------------
# 4. Optimización de portafolios (Markowitz)
# --------------------------------------------------


def port_vol(w, cov):
    w = np.array(w)
    return np.sqrt(w.T @ cov.values @ w)


def port_ret(w, mu):
    w = np.array(w)
    return w @ mu.values


def min_var_portfolio(mu, cov, short=False):
    n = len(mu)
    w0 = np.ones(n) / n
    bounds = [(-1.0, 1.0)] * n if short else [(0.0, 1.0)] * n
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)

    def obj(w):
        return w.T @ cov.values @ w

    res = minimize(obj, w0, bounds=bounds, constraints=cons)
    return res.x if res.success else None


def max_sharpe_portfolio(mu, cov, rf_anual, short=False):
    n = len(mu)
    w0 = np.ones(n) / n
    rf_diario = rf_anual / 252.0
    bounds = [(-1.0, 1.0)] * n if short else [(0.0, 1.0)] * n

    def neg_sharpe(w):
        r_p = port_ret(w, mu)
        v_p = port_vol(w, cov)
        if v_p == 0:
            return 1e6
        return -(r_p - rf_diario) / v_p

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)

    res = minimize(neg_sharpe, w0, bounds=bounds, constraints=cons)
    return res.x if res.success else None


def markowitz_target_portfolio(mu, cov, target_anual, short=False):
    n = len(mu)
    w0 = np.ones(n) / n
    target_diario = target_anual / 252.0
    bounds = [(-1.0, 1.0)] * n if short else [(0.0, 1.0)] * n

    def obj(w):
        return w.T @ cov.values @ w

    cons = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'eq', 'fun': lambda w: port_ret(w, mu) - target_diario},
    )

    res = minimize(obj, w0, bounds=bounds, constraints=cons)
    return res.x if res.success else None


def efficient_frontier(mu, cov, ret_min_anual, ret_max_anual,
                       n_points=50, short=False):
    """
    Calcula la frontera eficiente entre ret_min y ret_max (anualizados).
    Regresa arrays de volatilidades y rendimientos anuales.
    """
    targets = np.linspace(ret_min_anual, ret_max_anual, n_points)
    vols = []
    rets = []

    for t in targets:
        w = markowitz_target_portfolio(mu, cov, t, short=short)
        if w is None:
            continue
        r_daily = port_ret(w, mu)
        v_daily = port_vol(w, cov)
        rets.append(r_daily * 252)
        vols.append(v_daily * np.sqrt(252))

    return np.array(vols), np.array(rets)

# --------------------------------------------------
# 5. Interfaz Streamlit
# --------------------------------------------------


def main():
    # Encabezado
    st.markdown(
        "<h1 style='text-align:center;'>📊 Creación y análisis de portafolios</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center;'>Estrategias por regiones y sectores, "
        "benchmark vs portafolio del usuario y portafolios optimizados.</p>",
        unsafe_allow_html=True
    )
    st.write("")

    # -------- Sidebar: parámetros globales --------
    st.sidebar.header("Configuración")

    estrategia = st.sidebar.selectbox(
        "Estrategia",
        ["Regiones", "Sectores"]
    )

    years = st.sidebar.slider("Años de datos históricos", 1, 10, 4)
    rf_anual = st.sidebar.number_input(
        "Tasa libre de riesgo anual (rf)",
        0.0, 0.20, 0.05, step=0.005
    )

    modo = st.sidebar.radio(
        "Portafolios a analizar",
        ["Solo benchmark", "Solo arbitrario",
         "Benchmark y arbitrario", "Optimización"],
        index=2
    )

    if estrategia == "Regiones":
        tickers = TICKERS_REGIONES
        pesos_bench = PESOS_REGIONES
    else:
        tickers = TICKERS_SECTORES
        pesos_bench = PESOS_SECTORES

    # Pesos arbitrarios
    pesos_arbitrarios = {}
    if modo in ["Solo arbitrario", "Benchmark y arbitrario"]:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Pesos portafolio arbitrario")
        st.sidebar.caption("Se normalizan automáticamente para sumar 1.")

        for t in tickers:
            valor_por_defecto = float(pesos_bench.get(t, 0.0))
            w = st.sidebar.number_input(
                f"Peso {t}",
                min_value=0.0,
                max_value=1.0,
                value=valor_por_defecto,
                step=0.01
            )
            pesos_arbitrarios[t] = w

    # Parámetros de optimización
    target_anual = None
    permitir_cortos = False
    if modo == "Optimización":
        st.sidebar.markdown("---")
        st.sidebar.subheader("Parámetros de optimización")

        target_anual = st.sidebar.number_input(
            "Rendimiento objetivo anual (Markowitz)",
            min_value=0.0, max_value=0.5, value=0.10, step=0.01
        )
        permitir_cortos = st.sidebar.checkbox(
            "Permitir posiciones cortas",
            value=False
        )

    # -------- Botón principal --------
    calcular = st.button("Calcular portafolios y métricas", type="primary")

    if not calcular:
        st.info("Seleccione los parámetros en la barra lateral y pulse el botón para calcular.")
        return

    # -------- Cálculos principales --------
    with st.spinner("Descargando datos y calculando..."):
        data = descargar_precios(tickers, years)
        retornos, portafolio_bench = construir_portafolio(data, pesos_bench)
        mu, cov = obtener_mu_cov(retornos)

        # Portafolio arbitrario
        portafolio_arbi = None
        if modo in ["Solo arbitrario", "Benchmark y arbitrario"]:
            if sum(pesos_arbitrarios.values()) == 0:
                st.error("Los pesos del portafolio arbitrario no pueden ser todos cero.")
                return
            portafolio_arbi = construir_portafolio_arbitrario(retornos, pesos_arbitrarios)

        # Portafolios optimizados
        w_minvar = w_maxsharpe = w_markowitz = None
        port_minvar = port_maxsharpe = port_markowitz = None
        front_vols = front_rets = None

        if modo == "Optimización":
            w_minvar = min_var_portfolio(mu, cov, short=permitir_cortos)
            w_maxsharpe = max_sharpe_portfolio(mu, cov, rf_anual, short=permitir_cortos)
            w_markowitz = markowitz_target_portfolio(mu, cov, target_anual, short=permitir_cortos)

            if (w_minvar is None) or (w_maxsharpe is None) or (w_markowitz is None):
                st.error("No se encontró solución factible para alguna de las optimizaciones.")
                return

            cols = retornos.columns
            port_minvar = (retornos[cols] * w_minvar).sum(axis=1)
            port_maxsharpe = (retornos[cols] * w_maxsharpe).sum(axis=1)
            port_markowitz = (retornos[cols] * w_markowitz).sum(axis=1)

            # Frontera eficiente (en anualizado)
            mu_anual = mu * 252
            ret_min = float(mu_anual.min())
            ret_max = float(mu_anual.max()) * 1.5
            front_vols, front_rets = efficient_frontier(
                mu, cov, ret_min, ret_max, n_points=80, short=permitir_cortos
            )

    st.success("Cálculos completados.")

    # -------- Tabs de presentación --------
    tab_resumen, tab_datos, tab_graficos = st.tabs(
        ["Resumen de métricas", "Datos", "Gráficos"]
    )

    # ===================== TAB RESUMEN =====================
    with tab_resumen:
        st.subheader("Métricas de los portafolios")

        if modo != "Optimización":
            metricas = {}

            if modo in ["Solo benchmark", "Benchmark y arbitrario"]:
                metricas["Benchmark"] = calcular_metricas(portafolio_bench, rf=rf_anual)

            if modo in ["Solo arbitrario", "Benchmark y arbitrario"] and portafolio_arbi is not None:
                metricas["Arbitrario"] = calcular_metricas(portafolio_arbi, rf=rf_anual)

            df_metrics = pd.DataFrame(metricas)
            st.dataframe(df_metrics.style.format("{:.6f}"), use_container_width=True)

            for nombre, serie in metricas.items():
                st.markdown(f"#### {nombre}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Media diaria", f"{serie['Media diaria']:.5f}")
                c2.metric("Volatilidad diaria", f"{serie['Volatilidad diaria']:.5f}")
                c3.metric("Sharpe", f"{serie['Sharpe (5% rf)']:.3f}")
                c4.metric("Max Drawdown", f"{serie['Max Drawdown']:.3f}")

        else:
            metricas_opt = {
                "Mínima varianza": calcular_metricas(port_minvar, rf=rf_anual),
                "Máximo Sharpe": calcular_metricas(port_maxsharpe, rf=rf_anual),
                "Markowitz": calcular_metricas(port_markowitz, rf=rf_anual),
            }
            df_metrics_opt = pd.DataFrame(metricas_opt)
            st.dataframe(df_metrics_opt.style.format("{:.6f}"), use_container_width=True)

            weights_df = pd.DataFrame(
                {
                    "MinVar": w_minvar,
                    "MaxSharpe": w_maxsharpe,
                    "Markowitz": w_markowitz,
                },
                index=retornos.columns
            )
            st.markdown("#### Pesos de los portafolios optimizados")
            st.dataframe(weights_df.style.format("{:.4f}"), use_container_width=True)

    # ===================== TAB DATOS =====================
    with tab_datos:
        st.subheader("Precios y retornos")

        st.markdown("**Precios de cierre (últimos 10 registros)**")
        st.dataframe(data.tail(10), use_container_width=True)

        st.markdown("**Retornos diarios (primeros 10 registros)**")
        st.dataframe(retornos.head(10), use_container_width=True)

    # ===================== TAB GRÁFICOS =====================
    with tab_graficos:
        st.subheader("Rendimiento acumulado")

        df_cum = pd.DataFrame()
        if modo != "Optimización":
            if modo in ["Solo benchmark", "Benchmark y arbitrario"]:
                df_cum["Benchmark"] = (1 + portafolio_bench).cumprod()
            if modo in ["Solo arbitrario", "Benchmark y arbitrario"] and portafolio_arbi is not None:
                df_cum["Arbitrario"] = (1 + portafolio_arbi).cumprod()
        else:
            df_cum["MinVar"] = (1 + port_minvar).cumprod()
            df_cum["MaxSharpe"] = (1 + port_maxsharpe).cumprod()
            df_cum["Markowitz"] = (1 + port_markowitz).cumprod()

        st.line_chart(df_cum, use_container_width=True)

        if modo == "Optimización":
            st.subheader("Frontera eficiente (riesgo vs rendimiento anual)")
            if front_vols is not None and len(front_vols) > 0:
                fig, ax = plt.subplots()
                ax.plot(front_vols, front_rets, label="Frontera eficiente")

                # Puntos de Min Var, Máx Sharpe y Markowitz
                rv_min = port_ret(w_minvar, mu) * 252
                sv_min = port_vol(w_minvar, cov) * np.sqrt(252)
                ax.scatter(sv_min, rv_min, label="Min Var")

                rv_s = port_ret(w_maxsharpe, mu) * 252
                sv_s = port_vol(w_maxsharpe, cov) * np.sqrt(252)
                ax.scatter(sv_s, rv_s, label="Máx Sharpe")

                rv_m = port_ret(w_markowitz, mu) * 252
                sv_m = port_vol(w_markowitz, cov) * np.sqrt(252)
                ax.scatter(sv_m, rv_m, label="Markowitz")

                ax.set_xlabel("Volatilidad anual")
                ax.set_ylabel("Retorno esperado anual")
                ax.legend()
                st.pyplot(fig)
            else:
                st.info("No se pudo construir la frontera eficiente con los parámetros actuales.")

        else:
            st.subheader("Retornos diarios")
            df_ret = pd.DataFrame()
            if modo in ["Solo benchmark", "Benchmark y arbitrario"]:
                df_ret["Benchmark"] = portafolio_bench
            if (modo in ["Solo arbitrario", "Benchmark y arbitrario"]) and (portafolio_arbi is not None):
                df_ret["Arbitrario"] = portafolio_arbi
            st.line_chart(df_ret, use_container_width=True)


if __name__ == "__main__":
    main()


