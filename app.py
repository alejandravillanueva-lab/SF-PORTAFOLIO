import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.optimize import minimize
import scipy.optimize as op

st.set_page_config(
    page_title="Análisis de Portafolios",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
h2 {
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    color: #2b2d42 !important;
}
.stApp { background-color: #f9fafc; }
section[data-testid="stSidebar"] { background-color: #eef2f7; }

/* Slider */
div[data-testid="stSlider"] > div > div > div:nth-child(2) > div {
    background-color: #1f77b4 !important;
}
div[data-testid="stSlider"] div[role="slider"] {
    background-color: #1f77b4 !important;
    border-color: #1f77b4 !important;
}

/* Radio */
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

/* Button */
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

# Tickers y pesos

TICKERS_REGIONES = ["SPLG", "EWC", "IEUR", "EEM", "EWJ"]
TICKERS_SECTORES = ["XLC","XLY","XLP","XLE","XLF","XLV","XLI","XLB","XLRE","XLK","XLU"]

PESOS_REGIONES = {"SPLG":0.7062,"EWC":0.0323,"IEUR":0.1176,"EEM":0.0902,"EWJ":0.0537}
PESOS_SECTORES = {"XLC":0.0999,"XLY":0.1025,"XLP":0.0482,"XLE":0.0295,"XLF":0.1307,
                  "XLV":0.0958,"XLI":0.0809,"XLB":0.0166,"XLRE":0.0187,"XLK":0.3535,"XLU":0.0237}

# Funciones de las métricas

def sharpe(r, rf=0.0):
    excess = r - rf/252
    std = excess.std()
    return np.sqrt(252) * excess.mean() / std if std != 0 else np.nan

def var_95(r):
    return np.percentile(r, 5)

def cvar_95(r):
    v = var_95(r)
    return r[r <= v].mean()
def sortino(r, rf=0.0):
    excess = r - rf/252
    downside = excess[excess < 0].std()
    return np.sqrt(252) * excess.mean() / downside if downside != 0 else np.nan

def max_drawdown(r):
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()

def calcular_metricas(serie, rf=0.05):
    return {
        "Media diaria": serie.mean(),
        "Volatilidad diaria": serie.std(),
        "Sharpe (rf anual)": sharpe(serie, rf=rf),
        "Sortino (rf anual)": sortino(serie, rf=rf),
        "Max Drawdown": max_drawdown(serie),
        "VaR 95%": var_95(serie),
        "CVaR 95%": cvar_95(serie),
        "Skew": skew(serie),
        "Kurtosis": kurtosis(serie),
    }


# Datos
def descargar_precios(tickers, years=4):
    data = yf.download(tickers, period=f"{years}y")["Close"]
    return data

def construir_portafolio(data_precios, pesos_dict):
    retornos = data_precios.pct_change().dropna()
    cols = [t for t in pesos_dict.keys() if t in retornos.columns]
    retornos = retornos[cols]
    pesos = np.array([pesos_dict[t] for t in cols], dtype=float)
    pesos = pesos / pesos.sum()
    portafolio = (retornos * pesos).sum(axis=1)
    return retornos, portafolio

def construir_portafolio_arbitrario(retornos, pesos_dict):
    cols = [t for t in pesos_dict.keys() if t in retornos.columns]
    r = retornos[cols]
    pesos = np.array([pesos_dict[t] for t in cols], dtype=float)
    pesos = pesos / pesos.sum()
    return (r * pesos).sum(axis=1)

def obtener_mu_cov(retornos):
    return retornos.mean(), retornos.cov()
    
# Funciones de optimización

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
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)

    def obj(w):
        return w.T @ cov.values @ w

    res = minimize(obj, w0, bounds=bounds, constraints=cons)
    return res.x if res.success else None

def max_sharpe_portfolio(mu, cov, rf_anual):
    n = len(mu)
    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    rf_diario = rf_anual / 252.0
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)

    def neg_sharpe(w):
        r_p = port_ret(w, mu)
        v_p = port_vol(w, cov)
        if v_p == 0:
            return 1e6
        return - (r_p - rf_diario) / v_p

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
    
# Black-Litterman
def black_litterman(data, pesos_mercado, visiones, tau=0.05, delta=2.5,
                    metodo_post="Markowitz", rendimiento_objetivo=None,
                    peso_min=0, peso_max=1):

    returns = data.pct_change().dropna()
    cov_matrix = returns.cov() * 252
    activos = list(data.columns)
    n = len(activos)

    w_market = np.array([pesos_mercado[a] for a in activos], dtype=float)
    w_market = w_market / w_market.sum()

    pi = delta * (cov_matrix @ w_market)

    k = len(visiones)
    if k == 0:
        mu_bl = pi
    else:
        P = np.zeros((k, n))
        Q = np.zeros(k)
        omega_diag = np.zeros(k)

        for i, vista in enumerate(visiones):
            a1 = vista['activo_1']
            a2 = vista['activo_2']
            valor = vista['valor'] / 100.0
            conf = vista['confianza']
            optr = vista['operador']

            idx1 = activos.index(a1)

            if a2 == "Rendimiento Absoluto":
                P[i, idx1] = 1.0
                Q[i] = valor
            else:
                idx2 = activos.index(a2)
                P[i, idx1] = 1.0
                P[i, idx2] = -1.0
                if optr == ">":
                    Q[i] = valor
                elif optr == "<":
                    Q[i] = -valor
                else:
                    Q[i] = 0.0

            conf_norm = conf / 10.0
            omega_diag[i] = tau * (P[i] @ cov_matrix @ P[i].T) / (conf_norm ** 2)

        Omega = np.diag(omega_diag)

        tau_sigma = tau * cov_matrix
        tau_sigma_inv = np.linalg.inv(tau_sigma)
        omega_inv = np.linalg.inv(Omega)

        left_term = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
        right_term = tau_sigma_inv @ pi + P.T @ omega_inv @ Q
        mu_bl = left_term @ right_term

    mean_returns_bl = pd.Series(mu_bl, index=activos)

    def perf(w):
        ret = np.dot(w, mean_returns_bl)
        vol = np.sqrt(w @ cov_matrix @ w.T)
        return ret, vol

    x0 = np.ones(n) / n
    bounds = tuple((peso_min, peso_max) for _ in range(n))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)

    if metodo_post == "Mínima Varianza":
        result = op.minimize(lambda w: perf(w)[1], x0, method="SLSQP",
                             bounds=bounds, constraints=constraints)
    elif metodo_post == "Máximo Sharpe":
        def neg_sh(w):
            r, v = perf(w)
            return -(r / v) if v != 0 else 1e6
        result = op.minimize(neg_sh, x0, method="SLSQP",
                             bounds=bounds, constraints=constraints)
    else:
        if rendimiento_objetivo is None:
            rendimiento_objetivo = float(mean_returns_bl.mean()) * 100.0
        target = rendimiento_objetivo / 100.0
        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns_bl) - target},
        )
        result = op.minimize(lambda w: (w @ cov_matrix @ w.T), x0, method="SLSQP",
                             bounds=bounds, constraints=constraints)

    pesos_opt = result.x
    ret_opt, vol_opt = perf(pesos_opt)

    rend_imp = {a: round(pi[i] * 100, 2) for i, a in enumerate(activos)}
    rend_bl = {a: round(mu_bl[i] * 100, 2) for i, a in enumerate(activos)}
    pesos_bl = {a: round(pesos_opt[i] * 100, 2) for i, a in enumerate(activos)}

    sharpe_ratio = (ret_opt / vol_opt) if vol_opt != 0 else 0

    ret_bench = np.dot(w_market, mean_returns_bl)
    tracking_error = abs(ret_opt - ret_bench)

    metricas = {
        "rendimiento (%)": round(ret_opt * 100, 2),
        "volatilidad (%)": round(vol_opt * 100, 2),
        "sharpe": round(sharpe_ratio, 4),
        "tracking_error (%)": round(tracking_error * 100, 2),
    }

    return rend_imp, rend_bl, pesos_bl, metricas

# Aplicación
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
            Aplicación para analizar portafolios de <b>Regiones</b> y <b>Sectores</b>.
        </p>
        <ul style="font-size:1rem; color:#333; line-height:1.6;">
            <li>Benchmark (pesos dados)</li>
            <li>Portafolio arbitrario</li>
            <li>Optimización (MinVar, MaxSharpe, Markowitz)</li>
            <li>Black-Litterman</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    estrategia = st.sidebar.selectbox("Estrategia", ["Regiones", "Sectores"])
    years = st.sidebar.slider("Años de datos históricos", 1, 10, 4)
    rf_anual = st.sidebar.number_input("Tasa libre de riesgo anual (rf)", 0.0, 0.20, 0.05, step=0.005)

    modo = st.sidebar.radio(
        "Portafolios a calcular",
        ["Solo benchmark", "Solo arbitrario", "Benchmark y arbitrario", "Optimización", "Black-Litterman"],
        index=2
    )

    if estrategia == "Regiones":
        tickers = TICKERS_REGIONES
        pesos_bench = PESOS_REGIONES
    else:
        tickers = TICKERS_SECTORES
        pesos_bench = PESOS_SECTORES

    pesos_arbitrarios = {}
    if modo in ["Solo arbitrario", "Benchmark y arbitrario"]:
        st.sidebar.markdown("### Pesos portafolio arbitrario")
        for t in tickers:
            pesos_arbitrarios[t] = st.sidebar.number_input(
                f"Peso {t}",
                min_value=0.0, max_value=1.0,
                value=float(pesos_bench.get(t, 0.0)),
                step=0.01
            )

    target_anual = None
    if modo == "Optimización":
        st.sidebar.markdown("### Markowitz – Rendimiento objetivo")
        target_anual = st.sidebar.number_input(
            "Rendimiento objetivo anual (decimal, ej. 0.10 = 10%)",
            min_value=0.0, max_value=0.5, value=0.10, step=0.01
        )

    # Black-Litterman params
    tau = 0.05
    delta = 2.5
    metodo_post_bl = "Markowitz"
    rendimiento_obj_bl = None
    visiones = []

    if modo == "Black-Litterman":
        st.sidebar.markdown("### Black-Litterman")
        tau = st.sidebar.number_input("Tau (τ)", 0.001, 1.0, 0.05, 0.01)
        delta = st.sidebar.number_input("Delta (δ)", 0.1, 10.0, 2.5, 0.1)
        metodo_post_bl = st.sidebar.selectbox("Optimización post-BL", ["Mínima Varianza", "Máximo Sharpe", "Markowitz"])
        if metodo_post_bl == "Markowitz":
            rendimiento_obj_bl = st.sidebar.number_input("Rendimiento objetivo post-BL (%)", 0.0, 50.0, 10.0, 0.5)

        st.sidebar.markdown("### Visiones")
        num_visiones = st.sidebar.number_input("Número de visiones", 0, len(tickers), 0, 1)

        for i in range(int(num_visiones)):
            st.sidebar.markdown(f"**Visión {i+1}**")
            a1 = st.sidebar.selectbox("Activo 1", tickers, key=f"a1_{i}")
            optr = st.sidebar.selectbox("Operador", [">", "<", "="], key=f"op_{i}")
            a2 = st.sidebar.selectbox("Activo 2 / Absoluto", ["Rendimiento Absoluto"] + tickers, key=f"a2_{i}")
            valor = st.sidebar.number_input("Valor (%)", value=2.0, step=0.5, key=f"val_{i}")
            conf = st.sidebar.slider("Confianza (1-10)", 1, 10, 5, key=f"conf_{i}")

            visiones.append({
                "activo_1": a1,
                "operador": optr,
                "activo_2": a2,
                "valor": float(valor),
                "confianza": int(conf),
            })

    st.subheader(f"Estrategia seleccionada: {estrategia}")

    if st.button("Calcular métricas"):
        port_arbi = None  # para que no reviente si no lo creas

        with st.spinner("Descargando datos y calculando…"):
            data = descargar_precios(tickers, years)
            retornos, port_bench = construir_portafolio(data, pesos_bench)
            mu, cov = obtener_mu_cov(retornos)

        st.markdown("### Precios de cierre (últimos 10 registros)")
        st.dataframe(data.tail(10))

        st.markdown("### Retornos diarios (primeros 5 registros)")
        st.dataframe(retornos.head())

        if modo in ["Solo benchmark", "Solo arbitrario", "Benchmark y arbitrario"]:
            metrics_dict = {}

            if modo in ["Solo benchmark", "Benchmark y arbitrario"]:
                metrics_dict["Benchmark"] = calcular_metricas(port_bench, rf=rf_anual)

            if modo in ["Solo arbitrario", "Benchmark y arbitrario"]:
                if sum(pesos_arbitrarios.values()) == 0:
                    st.error("Los pesos del portafolio arbitrario no pueden ser todos cero.")
                    return
                port_arbi = construir_portafolio_arbitrario(retornos, pesos_arbitrarios)
                metrics_dict["Arbitrario"] = calcular_metricas(port_arbi, rf=rf_anual)

            st.markdown("### Métricas de portafolios")

            # ✅ AQUÍ ESTABA TU BUG: df_metrics no existía
            df_metrics = pd.DataFrame(metrics_dict)
            df_show = df_metrics.copy()

            for fila in df_show.index:
                if ("Sharpe" in fila) or ("Sortino" in fila) or ("Skew" in fila) or ("Kurtosis" in fila):
                    df_show.loc[fila] = df_show.loc[fila].round(3)
                else:
                    df_show.loc[fila] = (df_show.loc[fila] * 100).round(2)

            st.dataframe(df_show, use_container_width=True)

            st.markdown("### Rendimiento acumulado")
            df_cum = pd.DataFrame()
            if "Benchmark" in metrics_dict:
                df_cum["Benchmark"] = (1 + port_bench).cumprod()
            if ("Arbitrario" in metrics_dict) and (port_arbi is not None):
                df_cum["Arbitrario"] = (1 + port_arbi).cumprod()
            st.line_chart(df_cum)

        elif modo == "Optimización":
            w_minvar = min_var_portfolio(mu, cov)
            w_maxsharpe = max_sharpe_portfolio(mu, cov, rf_anual)
            w_markowitz = markowitz_target_portfolio(mu, cov, target_anual)

            if w_minvar is None or w_maxsharpe is None or w_markowitz is None:
                st.error("No se pudo encontrar una solución óptima para alguna de las optimizaciones.")
                return

            cols = retornos.columns
            port_minvar = (retornos[cols] * w_minvar).sum(axis=1)
            port_maxsharpe = (retornos[cols] * w_maxsharpe).sum(axis=1)
            port_markowitz = (retornos[cols] * w_markowitz).sum(axis=1)

            metrics_opt = {
                "MinVar": calcular_metricas(port_minvar, rf=rf_anual),
                "MaxSharpe": calcular_metricas(port_maxsharpe, rf=rf_anual),
                "Markowitz": calcular_metricas(port_markowitz, rf=rf_anual),
            }
            st.markdown("### Métricas de portafolios optimizados")
            st.dataframe(pd.DataFrame(metrics_opt).style.format("{:.6f}"), use_container_width=True)

            weights_df = pd.DataFrame(
                {"MinVar": w_minvar, "MaxSharpe": w_maxsharpe, "Markowitz": w_markowitz},
                index=cols
            )
            st.markdown("### Pesos de los portafolios optimizados")
            st.dataframe(weights_df.style.format("{:.4f}"), use_container_width=True)

            st.markdown("### Rendimiento acumulado – Portafolios optimizados")
            df_cum_opt = pd.DataFrame({
                "MinVar": (1 + port_minvar).cumprod(),
                "MaxSharpe": (1 + port_maxsharpe).cumprod(),
                "Markowitz": (1 + port_markowitz).cumprod(),
            })
            st.line_chart(df_cum_opt)

        else:  # Black-Litterman
            rend_imp, rend_bl, pesos_bl, metricas_bl = black_litterman(
                data=data,
                pesos_mercado=pesos_bench,
                visiones=visiones,
                tau=tau,
                delta=delta,
                metodo_post=metodo_post_bl,
                rendimiento_objetivo=rendimiento_obj_bl,
                peso_min=0.0,
                peso_max=1.0
            )

            st.markdown("### Rendimientos implícitos vs Black Litterman")
            df_bl = pd.DataFrame({
                "Implicitos (%)": pd.Series(rend_imp),
                "BL (%)": pd.Series(rend_bl)
            })
            st.dataframe(df_bl.style.format("{:.2f}"), use_container_width=True)

            st.markdown("### Pesos optimizados")
            st.dataframe(pd.Series(pesos_bl).to_frame("Peso (%)").style.format("{:.2f}"),
                         use_container_width=True)

            st.markdown("### Métricas Black-Litterman")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rendimiento (%)", f"{metricas_bl['rendimiento (%)']:.2f}")
            c2.metric("Volatilidad (%)", f"{metricas_bl['volatilidad (%)']:.2f}")
            c3.metric("Sharpe", f"{metricas_bl['sharpe']:.4f}")
            c4.metric("Tracking error (%)", f"{metricas_bl['tracking_error (%)']:.2f}")

if __name__ == "__main__":
    main()


