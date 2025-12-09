import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis


# 1. BENCHMARK Tickers por regiones, sectores y los respectivos pesos.


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
    "XLE": 0.0295,
    "XLF": 0.1307,
    "XLV": 0.0958,
    "XLI": 0.0809,
    "XLB": 0.0166,
    "XLRE": 0.0187,
    "XLK": 0.3535,
    "XLU": 0.0237
}

# 2. Funciones
#Descarga precios
def descargar_precios(tickers, years=4):
    data = yf.download(tickers, period=f"{years}y")["Close"]
    return data

# Calcula rendimientos de los activos y del portafolio benchmark

def construir_portafolio(data_precios, pesos_dict):
    retornos = data_precios.pct_change().dropna()

    #Mismo orden y mismas columnas
    columnas = [t for t in pesos_dict.keys() if t in retornos.columns]
    retornos = retornos[columnas]

    pesos = np.array([pesos_dict[t] for t in columnas], dtype=float)
    pesos = pesos / pesos.sum()

    portafolio = (retornos * pesos).sum(axis=1)
    return retornos, portafolio

# 3. Métricas

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

# 4. Streamlit

def main():
    st.title("Cálculo de métricas – Portafolios Benchmark")
    st.write("Usando ETFs por **regiones** y **sectores** con sus pesos de benchmark.")

    # Sidebar
    estrategia = st.sidebar.selectbox(
        "Estrategia",
        ["Regiones", "Sectores"]
    )

    years = st.sidebar.slider("Años de datos históricos", 1, 10, 4)
    rf_anual = st.sidebar.number_input("Tasa libre de riesgo anual (rf)", 0.0, 0.20, 0.05, step=0.005)

    if estrategia == "Regiones":
        tickers = TICKERS_REGIONES
        pesos = PESOS_REGIONES
    else:
        tickers = TICKERS_SECTORES
        pesos = PESOS_SECTORES

    st.subheader(f"Estrategia seleccionada: {estrategia}")

    # Botón para ejecutar
    if st.button("Calcular métricas del benchmark"):
        with st.spinner("Descargando datos y calculando…"):
            data = descargar_precios(tickers, years)
            retornos, portafolio = construir_portafolio(data, pesos)

        st.markdown("### Precios de cierre (últimos 10 registros)")
        st.dataframe(data.tail(10))

        st.markdown("### Retornos diarios (primeros 5 registros)")
        st.dataframe(retornos.head())

        # Métricas
        metrics = calcular_metricas(portafolio, rf=rf_anual)
        df_metrics = pd.DataFrame(metrics, index=["Benchmark"]).T

        st.markdown("### Métricas del portafolio benchmark")
        st.dataframe(df_metrics.style.format("{:.6f}"))

        # Curva de rendimiento acumulado
        st.markdown("### Rendimiento acumulado del benchmark")
        rendimiento_acum = (1 + portafolio).cumprod()
        st.line_chart(rendimiento_acum)

        # Histograma de retornos
        st.markdown("### Distribución de retornos diarios")
        st.bar_chart(portafolio.value_counts(bins=30).sort_index())

if __name__ == "__main__":
    main()
