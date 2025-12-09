import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# ==========================
# 1. DATOS Y PESOS
# ==========================

TICKERS_REGIONES = ["SPLG", "EWC", "IEUR", "EEM", "EWJ"]

TICKERS_SECTORES = [
    "XLC","XLY","XLP","XLE","XLF",
    "XLV","XLI","XLB","XLRE","XLK","XLU"
]

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

# ==========================
# 2. FUNCIONES AUXILIARES
# ==========================

def descargar_precios(tickers, years=4):
    """Descarga precios de cierre ajustados con yfinance."""
    data = yf.download(tickers, period=f"{years}y")["Close"]
    return data

def construir_portafolio(data_precios, pesos_dict):
    """Calcula rendimientos y portafolio dada una tabla de precios y un diccionario de pesos."""
    retornos = data_precios.pct_change().dropna()

    columnas = [t for t in pesos_dict.keys() if t in retornos.columns]
    retornos = retornos[columnas]

    pesos = np.array([pesos_dict[t] for t in columnas], dtype=float)
    pesos = pesos / pesos.sum()

    portafolio = (retornos * pesos).sum(axis=1)
    return retornos, portafolio

def construir_portafolio_arbitrario(retornos, pesos_dict):
    """Construye portafolio usando retornos ya calculados y pesos arbitrarios."""
    columnas = [t for t in pesos_dict.keys() if t in retornos.columns]
    r = retornos[columnas]
    pesos = np.array([pesos_dict[t] for t in columnas], dtype=float)
    pesos = pesos / pesos.sum()
    portafolio = (r * pesos).sum(axis=1)
    return portafolio

# ==========================
# 3. FUNCIONES DE MÉTRICAS
# ==========================

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

# ==========================
# 4. APLICACIÓN STREAMLIT
# ==========================

def main():
    st.title("Cálculo de métricas – Portafolios Benchmark y Arbitrario")
    st.write("Usando ETFs por **regiones** y **sectores** con pesos de benchmark y un portafolio arbitrario definido por el usuario.")

    # Sidebar
    estrategia = st.sidebar.selectbox(
        "Estrategia",
        ["Regiones", "Sectores"]
    )

    years = st.sidebar.slider("Años de datos históricos", 1, 10, 4)
    rf_anual = st.sidebar.number_input("Tasa libre de riesgo anual (rf)", 0.0, 0.20, 0.05, step=0.005)

    modo = st.sidebar.radio(
        "Portafolios a calcular",
        ["Solo benchmark", "Solo arbitrario", "Benchmark y arbitrario"],
        index=2
    )

    if estrategia == "Regiones":
        tickers = TICKERS_REGIONES
        pesos_bench = PESOS_REGIONES
    else:
        tickers = TICKERS_SECTORES
        pesos_bench = PESOS_SECTORES

    # ---- Pesos arbitrarios en sidebar ----
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

    st.subheader(f"Estrategia seleccionada: {estrategia}")

    if st.button("Calcular métricas"):
        with st.spinner("Descargando datos y calculando…"):
            data = descargar_precios(tickers, years)

            # Portafolio benchmark
            retornos, portafolio_bench = construir_portafolio(data, pesos_bench)

            # Portafolio arbitrario (si aplica)
            portafolio_arbi = None
            if modo in ["Solo arbitrario", "Benchmark y arbitrario"]:
                # Si todos los pesos son 0, no se puede construir
                if sum(pesos_arbitrarios.values()) == 0:
                    st.error("Los pesos del portafolio arbitrario no pueden ser todos cero.")
                    return
                portafolio_arbi = construir_portafolio_arbitrario(retornos, pesos_arbitrarios)

        # --- Mostrar datos básicos ---
        st.markdown("### Precios de cierre (últimos 10 registros)")
        st.dataframe(data.tail(10))

        st.markdown("### Retornos diarios (primeros 5 registros)")
        st.dataframe(retornos.head())

        # --- Métricas ---
        metrics_dict = {}

        if modo in ["Solo benchmark", "Benchmark y arbitrario"]:
            metrics_dict["Benchmark"] = calcular_metricas(portafolio_bench, rf=rf_anual)

        if modo in ["Solo arbitrario", "Benchmark y arbitrario"] and portafolio_arbi is not None:
            metrics_dict["Arbitrario"] = calcular_metricas(portafolio_arbi, rf=rf_anual)

        df_metrics = pd.DataFrame(metrics_dict)
        st.markdown("### Métricas de portafolios")
        st.dataframe(df_metrics.style.format("{:.6f}"))

        # --- Rendimiento acumulado ---
        st.markdown("### Rendimiento acumulado")
        df_cum = pd.DataFrame()

        if "Benchmark" in metrics_dict:
            df_cum["Benchmark"] = (1 + portafolio_bench).cumprod()

        if "Arbitrario" in metrics_dict and portafolio_arbi is not None:
            df_cum["Arbitrario"] = (1 + portafolio_arbi).cumprod()

        st.line_chart(df_cum)

if __name__ == "__main__":
    main()
