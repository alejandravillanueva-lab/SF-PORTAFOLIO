import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sf_library as sfl      # Nuestra libreria

# Configuración de la página
st.set_page_config(
    page_title="Portfolio Manager Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CArgar los datos en caché
@st.cache_data
def cargar_datos():
    """Carga los datos de los archivos CSV"""
    try:
        # Intentar cargar los archivos CSV
        data_regiones = pd.read_csv('data_regiones.csv', index_col=0, parse_dates=True)
        data_sectores = pd.read_csv('data_sectores.csv', index_col=0, parse_dates=True)
        
        return data_regiones, data_sectores
    except FileNotFoundError:
        st.error("⚠️ No se encontraron los archivos CSV. Por favor, coloca 'data_regiones.csv' y 'data_sectores.csv' en la misma carpeta que este script.")
        return None, None, False
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return None, None, False
    
data_regiones, data_sectores = cargar_datos()

# Mapeo de estrategia a datos
DATOS_ESTRATEGIA = {
    "Regiones": data_regiones,
    "Sectores": data_sectores
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

# CSS personalizado
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<div class="main-header">📊 Portfolio Manager Pro</div>', unsafe_allow_html=True)
st.markdown("### Análisis Cuantitativo de Estrategias de Inversión")

# Benchmarks predefinidos
BENCHMARKS = {
    "Regiones": {
        "SPLG": 70.62,
        "EWC": 3.23,
        "IEUR": 11.76,
        "EEM": 9.02,
        "EWJ": 5.37
    },
    "Sectores": {
        "XLC": 9.99,
        "XLY": 10.25,
        "XLP": 4.82,
        "XLE": 2.95,
        "XLF": 13.07,
        "XLV": 9.58,
        "XLI": 8.09,
        "XLB": 1.66,
        "XLRE": 1.87,
        "XLK": 35.35,
        "XLU": 2.37
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

# Sidebar - Configuración General
with st.sidebar:
    st.header("⚙️ Configuración General")
    
    # Selección de estrategia
    estrategia = st.selectbox(
        "Estrategia de Inversión",
        ["Regiones", "Sectores"],
        help="Seleccione el universo invertible a analizar"
    )
    
    st.divider()
    
    # Período de análisis
    st.subheader("📅 Período de Análisis")
    fecha_inicio = st.date_input(
        "Fecha Inicio",
        value=datetime.now() - timedelta(days=365),
        help="Fecha de inicio para el análisis histórico"
    )
    fecha_fin = st.date_input(
        "Fecha Fin",
        value=datetime.now(),
        help="Fecha de fin para el análisis histórico"
    )
    
    st.divider()
    
    # Parámetros globales
    st.subheader("🎯 Parámetros")
    tasa_libre_riesgo = st.number_input(
        "Tasa Libre de Riesgo (%)",
        min_value=0.0,
        max_value=20.0,
        value=4.5,
        step=0.1,
        help="Tasa anualizada para el cálculo de Sharpe y Sortino Ratio"
    ) / 100
    
    nivel_confianza = st.slider(
        "Nivel de Confianza VaR/CVaR (%)",
        min_value=90,
        max_value=99,
        value=95,
        help="Nivel de confianza para el cálculo de VaR y CVaR"
    ) / 100

# Tabs principales
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Construcción de Portafolio",
    "📊 Métricas y Análisis",
    "📈 Optimización",
    "🔮 Black-Litterman"
])

# ==================== TAB 1: CONSTRUCCIÓN DE PORTAFOLIO ====================
with tab1:
    st.header("Construcción de Portafolio")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📌 Benchmark: {estrategia}")
        
        # Mostrar benchmark
        benchmark_df = pd.DataFrame({
            'ETF': list(BENCHMARKS[estrategia].keys()),
            'Peso (%)': list(BENCHMARKS[estrategia].values())
        })
        
        st.dataframe(
            benchmark_df,
            hide_index=True,
            use_container_width=True
        )
        
        # Gráfico del benchmark
        fig_benchmark = px.pie(
            benchmark_df,
            values='Peso (%)',
            names='ETF',
            title=f'Composición Benchmark - {estrategia}',
            hole=0.4
        )
        st.plotly_chart(fig_benchmark, use_container_width=True)
    
    with col2:
        st.subheader("📝 Portafolio Arbitrario")
        st.info("**Instrucciones:** Defina los pesos de su portafolio. La suma debe ser 100%.")
        
        # Inputs para portafolio arbitrario
        pesos_arbitrarios = {}
        suma_pesos = 0
        
        for etf in BENCHMARKS[estrategia].keys():
            peso = st.number_input(
                f"{etf} (%)",
                min_value=0.0,
                max_value=100.0,
                value=BENCHMARKS[estrategia][etf],
                step=0.01,
                key=f"peso_{etf}"
            )
            pesos_arbitrarios[etf] = peso
            suma_pesos += peso
        
        # Validación de pesos
        if abs(suma_pesos - 100) < 0.01:
            st.success(f"✅ Suma de pesos: {suma_pesos:.2f}%")
        else:
            st.error(f"❌ Suma de pesos: {suma_pesos:.2f}% (debe ser 100%)")
        
        if st.button("🔄 Normalizar Pesos", use_container_width=True):
            st.info("Los pesos serán normalizados automáticamente al calcular métricas")

# ==================== TAB 2: MÉTRICAS Y ANÁLISIS ====================
with tab2:
    st.header("Métricas y Análisis del Portafolio")
    
    # Inicializar session_state para métricas si no existe
    if 'metricas_calculadas' not in st.session_state:
        st.session_state.metricas_calculadas = False
        st.session_state.metricas = {}
    
    # Selector de portafolio a analizar
    tipo_portafolio = st.radio(
        "Seleccione el portafolio a analizar:",
        ["Benchmark", "Portafolio Arbitrario", "Portafolio Optimizado"],
        horizontal=True
    )
    # datos_filtrados = datos_estrategia.loc[fecha_inicio:fecha_fin]
       
    if st.button("📊 Calcular Métricas", type="primary", use_container_width=True):
        datos_estrategia = DATOS_ESTRATEGIA[estrategia]
        pesos_benchmark = [v/100 for v in BENCHMARKS[estrategia].values()]
        portafolio_benchmark = sfl.construir_portafolio(datos_estrategia, pesos_benchmark)

        if tipo_portafolio == "Portafolio Arbitrario":
          
            # Convertir pesos a diccionario normalizado (0-1)
            pesos_usar = [v/100 for v in pesos_arbitrarios.values()]
            
        elif tipo_portafolio == "Benchmark":

            # Usar pesos del benchmark (ya están en %)
            pesos_usar = pesos_benchmark
            
        elif tipo_portafolio == "Portafolio Optimizado":
            # Verificar que existe un portafolio optimizado
            if not st.session_state.get('optimizacion_realizada', False):
                st.warning("⚠️ Primero debe optimizar un portafolio en la pestaña 'Optimización'")
                st.stop()

            pesos_usar = [v/100 for k, v in st.session_state.pesos_optimizados.items()]
        
        portafolio = sfl.construir_portafolio(datos_estrategia, pesos_usar)

        # Calcular métricas
        st.session_state.metricas = sfl.calcular_metricas_portfolio(portafolio, portafolio_benchmark)
        st.session_state.metricas_calculadas = True
        st.session_state.tipo_portafolio_analizado = tipo_portafolio            
        st.success("✅ Métricas calculadas exitosamente!")
    
    st.divider()
    
    # Métricas principales
    st.subheader("📈 Métricas de Rendimiento")
    
    # Obtener valores de las métricas
    if st.session_state.metricas_calculadas:
        m = st.session_state.metricas
        rend_val = f"{m['rendimiento']:.2f}%"
        rend_delta = f"{m['delta_benchmark']:.2f}%"
        vol_val = f"{m['volatilidad']:.2f}%"
        beta_val = f"{m['beta']:.3f}"
        sharpe_val = f"{m['sharpe']:.3f}"
        sortino_val = f"{m['sortino']:.3f}"
        dd_val = f"{m['max_drawdown']:.2f}%"
        var_val = f"{m['var']:.2f}%"
        cvar_val = f"{m['cvar']:.2f}%"
        sesgo_val = f"{m['sesgo']:.3f}"
        curtosis_val = f"{m['curtosis']:.3f}"
        rend_pos_val = f"{m['rend_positivos']:.1f}%"
    else:
        rend_val = "---"
        rend_delta = None
        vol_val = "---"
        beta_val = "---"
        sharpe_val = "---"
        sortino_val = "---"
        dd_val = "---"
        var_val = "---"
        cvar_val = "---"
        sesgo_val = "---"
        curtosis_val = "---"
        rend_pos_val = "---"
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Rendimiento Anualizado",
            value=rend_val,
            delta=rend_delta,
            help="Rendimiento promedio anualizado del portafolio"
        )
        st.metric(
            label="Beta",
            value=beta_val,
            help="Sensibilidad del portafolio respecto al mercado"
        )
    
    with col2:
        st.metric(
            label="Volatilidad Anualizada",
            value=vol_val,
            help="Desviación estándar anualizada de los rendimientos"
        )
        st.metric(
            label="Sharpe Ratio",
            value=sharpe_val,
            help="Rendimiento ajustado por riesgo (exceso de rendimiento / volatilidad)"
        )
    
    with col3:
        st.metric(
            label="Max Drawdown",
            value=dd_val,
            help="Máxima caída desde un pico histórico"
        )
        st.metric(
            label="Sortino Ratio",
            value=sortino_val,
            help="Rendimiento ajustado por riesgo negativo"
        )
    
    with col4:
        st.metric(
            label=f"VaR {int(nivel_confianza*100)}%",
            value=var_val,
            help="Value at Risk: pérdida máxima esperada con un nivel de confianza dado"
        )
        st.metric(
            label=f"CVaR {int(nivel_confianza*100)}%",
            value=cvar_val,
            help="Conditional VaR: pérdida promedio cuando se supera el VaR"
        )
    
    st.divider()
    
    # Métricas de distribución
    st.subheader("📊 Características de la Distribución")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Sesgo (Skewness)",
            value=sesgo_val,
            help="Asimetría de la distribución. Negativo indica cola izquierda pesada"
        )
    
    with col2:
        st.metric(
            label="Curtosis (Kurtosis)",
            value=curtosis_val,
            help="Exceso de curtosis. >0 indica colas pesadas (más riesgo extremo)"
        )
    
    with col3:
        st.metric(
            label="Rendimientos Positivos",
            value=rend_pos_val,
            help="Porcentaje de períodos con rendimientos positivos"
        )
    
    st.divider()
    
    # Visualizaciones
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Evolución del Valor del Portafolio")
        # Placeholder para gráfico
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=[],
            y=[],
            mode='lines',
            name='Portafolio'
        ))
        fig1.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Valor Normalizado",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader("📊 Distribución de Rendimientos")
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=[],
            name='Rendimientos',
            nbinsx=50
        ))
        fig2.update_layout(
            xaxis_title="Rendimiento",
            yaxis_title="Frecuencia",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.subheader("📉 Drawdown a lo Largo del Tiempo")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=[],
            y=[],
            fill='tozeroy',
            name='Drawdown'
        ))
        fig3.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Drawdown (%)",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        st.subheader("📊 Rendimientos Mensuales")
        # Heatmap placeholder
        fig4 = go.Figure()
        fig4.add_trace(go.Heatmap(
            z=[[]],
            colorscale='RdYlGn',
            zmid=0
        ))
        fig4.update_layout(
            xaxis_title="Mes",
            yaxis_title="Año",
            height=400
        )
        st.plotly_chart(fig4, use_container_width=True)

# ==================== TAB 3: OPTIMIZACIÓN ====================
with tab3:
    st.header("Optimización de Portafolios")
    
    # Inicializar session_state para optimización
    if 'optimizacion_realizada' not in st.session_state:
        st.session_state.optimizacion_realizada = False
        st.session_state.pesos_optimizados = {}
        st.session_state.metricas_opt = {}
        st.session_state.frontera_data = {}
    
    st.info("""
    **Métodos de Optimización Disponibles:**
    - **Mínima Varianza:** Minimiza el riesgo del portafolio
    - **Máximo Sharpe:** Maximiza el ratio rendimiento/riesgo
    - **Markowitz:** Optimiza para un rendimiento objetivo específico
    """)
    
    # Selector de método
    metodo_opt = st.selectbox(
        "Método de Optimización",
        ["Mínima Varianza", "Máximo Sharpe", "Markowitz (Rendimiento Objetivo)"]
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
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Parámetros de Optimización")
        
        # Parámetros específicos según método
        if metodo_opt == "Markowitz (Rendimiento Objetivo)":
            rendimiento_objetivo = st.number_input(
                "Rendimiento Objetivo Anualizado (%)",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=0.5,
                help="Rendimiento esperado que desea alcanzar"
            ) / 100
        else:
            rendimiento_objetivo = None
        
        # Restricciones
        st.subheader("🔒 Restricciones")
        
        permitir_cortos = st.checkbox(
            "Permitir posiciones cortas",
            value=False,
            help="Si se desmarca, todos los pesos deben ser ≥ 0"
        )
        
        peso_min = st.number_input(
            "Peso Mínimo por Activo (%)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
            help="Peso mínimo permitido para cada activo"
        ) / 100
        
        peso_max = st.number_input(
            "Peso Máximo por Activo (%)",
            min_value=0.0,
            max_value=100.0,
            value=100.0,
            step=1.0,
            help="Peso máximo permitido para cada activo"
        ) / 100
        
        # Función para optimizar portafolio (simulación)
        def optimizar_portafolio(metodo, etfs):
            # AQUÍ DEBES IMPLEMENTAR TU OPTIMIZACIÓN REAL
            # Por ahora, generamos pesos aleatorios que sumen 100%
            
            n_activos = len(etfs)
            
            if metodo == "Mínima Varianza":
                # Simular pesos más uniformes para mínima varianza
                pesos_raw = np.random.dirichlet(np.ones(n_activos) * 5) * 100
            elif metodo == "Máximo Sharpe":
                # Simular concentración en algunos activos
                pesos_raw = np.random.dirichlet(np.ones(n_activos) * 2) * 100
            else:  # Markowitz
                # Simular pesos intermedios
                pesos_raw = np.random.dirichlet(np.ones(n_activos) * 3) * 100
            
            # Aplicar restricciones
            pesos_raw = np.clip(pesos_raw, peso_min * 100, peso_max * 100)
            pesos = pesos_raw / pesos_raw.sum() * 100  # Normalizar a 100%
            
            pesos_dict = {etf: peso for etf, peso in zip(etfs, pesos)}
            
            # Calcular métricas simuladas
            metricas = {
                'rendimiento': np.random.uniform(8, 15) if metodo == "Máximo Sharpe" else np.random.uniform(5, 10),
                'volatilidad': np.random.uniform(8, 12) if metodo == "Mínima Varianza" else np.random.uniform(12, 18),
                'sharpe': np.random.uniform(0.8, 1.5) if metodo == "Máximo Sharpe" else np.random.uniform(0.4, 1.0),
            }
            
            # Generar datos para frontera eficiente
            n_portfolios = 100
            volatilidades = np.linspace(8, 25, n_portfolios)
            rendimientos = []
            
            for vol in volatilidades:
                # Simular relación riesgo-retorno
                if vol < 12:
                    ret = 3 + vol * 0.5 + np.random.normal(0, 0.5)
                else:
                    ret = 6 + vol * 0.3 + np.random.normal(0, 0.8)
                rendimientos.append(ret)
            
            frontera = {
                'volatilidades': volatilidades,
                'rendimientos': rendimientos,
                'vol_opt': metricas['volatilidad'],
                'ret_opt': metricas['rendimiento']
            }
            
            return pesos_dict, metricas, frontera
        
        if st.button("🚀 Optimizar Portafolio", type="primary", use_container_width=True):
            with st.spinner("Optimizando portafolio..."):
                import time
                time.sleep(1.5)
                
                # Realizar optimización
                pesos, metricas, frontera = optimizar_portafolio(metodo_opt, list(BENCHMARKS[estrategia].keys()))
                
                st.session_state.pesos_optimizados = pesos
                st.session_state.metricas_opt = metricas
                st.session_state.frontera_data = frontera
                st.session_state.optimizacion_realizada = True
                st.session_state.metodo_usado = metodo_opt
                
            st.success(f"✅ Optimización completada usando: {metodo_opt}")
    
    with col2:
        st.subheader("📊 Resultados de la Optimización")
        
        if st.session_state.optimizacion_realizada:
            # Mostrar método usado
            st.info(f"**Método:** {st.session_state.metodo_usado}")
            
            # Tabla de pesos optimizados
            st.write("**Pesos Optimizados:**")
            pesos_opt_df = pd.DataFrame({
                'ETF': list(st.session_state.pesos_optimizados.keys()),
                'Peso (%)': [f"{v:.2f}" for v in st.session_state.pesos_optimizados.values()]
            })
            st.dataframe(pesos_opt_df, hide_index=True, use_container_width=True)
            
            # Gráfico de composición
            fig_pie_opt = px.pie(
                pesos_opt_df,
                values='Peso (%)',
                names='ETF',
                title='Composición del Portafolio Optimizado',
                hole=0.4
            )
            st.plotly_chart(fig_pie_opt, use_container_width=True)
            
            # Métricas del portafolio optimizado
            st.write("**Métricas del Portafolio Optimizado:**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    "Rendimiento Esperado", 
                    f"{st.session_state.metricas_opt['rendimiento']:.2f}%"
                )
                st.metric(
                    "Sharpe Ratio", 
                    f"{st.session_state.metricas_opt['sharpe']:.3f}"
                )
            with col_b:
                st.metric(
                    "Volatilidad", 
                    f"{st.session_state.metricas_opt['volatilidad']:.2f}%"
                )
                st.metric(
                    "Peso Total", 
                    f"{sum(st.session_state.pesos_optimizados.values()):.2f}%"
                )
        else:
            # Mostrar placeholders
            st.write("**Pesos Optimizados:**")
            pesos_opt_df = pd.DataFrame({
                'ETF': list(BENCHMARKS[estrategia].keys()),
                'Peso (%)': ['---'] * len(BENCHMARKS[estrategia])
            })
            st.dataframe(pesos_opt_df, hide_index=True, use_container_width=True)
            
            st.info("👆 Presione el botón 'Optimizar Portafolio' para ver los resultados")
            
            # Métricas vacías
            st.write("**Métricas del Portafolio Optimizado:**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Rendimiento Esperado", "---")
                st.metric("Sharpe Ratio", "---")
            with col_b:
                st.metric("Volatilidad", "---")
                st.metric("Peso Total", "---")
    
    st.divider()
    
    # Frontera eficiente
    st.subheader("📈 Frontera Eficiente")
    
    if st.session_state.optimizacion_realizada:
        frontera = st.session_state.frontera_data
        
        fig_frontera = go.Figure()
        
        # Frontera eficiente
        fig_frontera.add_trace(go.Scatter(
            x=frontera['volatilidades'],
            y=frontera['rendimientos'],
            mode='lines',
            name='Frontera Eficiente',
            line=dict(color='blue', width=3)
        ))
        
        # Portafolio optimizado
        fig_frontera.add_trace(go.Scatter(
            x=[frontera['vol_opt']],
            y=[frontera['ret_opt']],
            mode='markers',
            name='Portafolio Optimizado',
            marker=dict(color='red', size=15, symbol='star')
        ))
        
        fig_frontera.update_layout(
            xaxis_title="Volatilidad (Riesgo) %",
            yaxis_title="Rendimiento Esperado %",
            hovermode='closest',
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig_frontera, use_container_width=True)
        
        # Información adicional
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Ratio Rendimiento/Riesgo",
                f"{frontera['ret_opt']/frontera['vol_opt']:.3f}",
                help="Rendimiento dividido por volatilidad"
            )
        with col2:
            st.metric(
                "Posición en Frontera",
                "Óptimo" if st.session_state.metodo_usado == "Máximo Sharpe" else "Eficiente",
                help="Ubicación del portafolio en la frontera eficiente"
            )
        with col3:
            diversificacion = 100 - max(st.session_state.pesos_optimizados.values())
            st.metric(
                "Índice de Diversificación",
                f"{diversificacion:.1f}%",
                help="100% - peso del activo más grande"
            )

    years = st.sidebar.slider("Años de datos históricos", 1, 10, 4)
    rf_anual = st.sidebar.number_input("Tasa libre de riesgo anual (rf)", 0.0, 0.20, 0.05, step=0.005)

    if estrategia == "Regiones":
        tickers = TICKERS_REGIONES
        pesos = PESOS_REGIONES
    else:
        fig_frontera = go.Figure()
        fig_frontera.add_trace(go.Scatter(
            x=[],
            y=[],
            mode='lines',
            name='Frontera Eficiente'
        ))
        fig_frontera.update_layout(
            xaxis_title="Volatilidad (Riesgo) %",
            yaxis_title="Rendimiento Esperado %",
            hovermode='closest',
            height=500
        )
        st.plotly_chart(fig_frontera, use_container_width=True)
        st.info("La frontera eficiente se mostrará después de optimizar el portafolio")

# ==================== TAB 4: BLACK-LITTERMAN ====================
with tab4:
    st.header("Optimización Black-Litterman")
    
    st.info("""
    **Modelo Black-Litterman:**  
    Combina el equilibrio del mercado (rendimientos implícitos) con las visiones del gestor 
    para generar rendimientos esperados ajustados y construir portafolios más robustos.
    """)
    
    st.divider()
    
    # Parámetros del modelo
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Parámetros del Modelo")
        
        tau = st.number_input(
            "Tau (τ)",
            min_value=0.001,
            max_value=1.0,
            value=0.05,
            step=0.01,
            format="%.3f",
            help="Parámetro de incertidumbre del prior (típicamente 0.01-0.05)"
        )
        
        delta = st.number_input(
            "Delta (δ) - Aversión al Riesgo",
            min_value=0.1,
            max_value=10.0,
            value=2.5,
            step=0.1,
            help="Coeficiente de aversión al riesgo del mercado"
        )
        
        st.subheader("📋 Supuestos")
        st.write("""
        **Matriz P (Picking Matrix):**  
        - Se asume estructura de vista relativa
        - Cada vista compara dos activos
        - Matriz identidad modificada
        """)
        
        metodo_bl = st.selectbox(
            "Método de Optimización Post-BL",
            ["Mínima Varianza", "Máximo Sharpe", "Markowitz"]
        )
        
        if metodo_bl == "Markowitz":
            rend_obj_bl = st.number_input(
                "Rendimiento Objetivo (%)",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=0.5
            ) / 100
    
    with col2:
        st.subheader("👁️ Visiones del Gestor (Views)")
        
        st.write("**Ingrese sus visiones sobre los rendimientos futuros:**")
        
        # Número de visiones
        num_visiones = st.number_input(
            "Número de Visiones",
            min_value=0,
            max_value=len(BENCHMARKS[estrategia]),
            value=2,
            step=1,
            help="Cantidad de visiones o expectativas sobre los activos"
        )
        
        visiones = []
        
        for i in range(num_visiones):
            st.write(f"**Visión {i+1}:**")
            col_a, col_b, col_c, col_d = st.columns([2, 1, 2, 1])
            
            with col_a:
                activo_1 = st.selectbox(
                    "Activo 1",
                    list(BENCHMARKS[estrategia].keys()),
                    key=f"activo1_{i}"
                )
            
            with col_b:
                operador = st.selectbox(
                    "Operador",
                    [">", "<", "="],
                    key=f"op_{i}"
                )
            
            with col_c:
                activo_2 = st.selectbox(
                    "Activo 2 / Rendimiento Absoluto",
                    ["Rendimiento Absoluto"] + list(BENCHMARKS[estrategia].keys()),
                    key=f"activo2_{i}"
                )
            
            with col_d:
                if activo_2 == "Rendimiento Absoluto":
                    valor = st.number_input(
                        "Rendimiento (%)",
                        value=5.0,
                        step=0.5,
                        key=f"valor_{i}"
                    )
                else:
                    valor = st.number_input(
                        "Por (%)",
                        value=2.0,
                        step=0.5,
                        key=f"valor_{i}"
                    )
            
            col_conf1, col_conf2 = st.columns([1, 2])
            with col_conf1:
                confianza = st.slider(
                    "Confianza",
                    min_value=1,
                    max_value=10,
                    value=5,
                    key=f"conf_{i}",
                    help="1=Baja confianza, 10=Alta confianza"
                )
            
            visiones.append({
                'activo_1': activo_1,
                'operador': operador,
                'activo_2': activo_2,
                'valor': valor,
                'confianza': confianza
            })
            
            st.divider()
        
        if st.button("🔮 Aplicar Black-Litterman", type="primary", use_container_width=True):
            st.success("Calculando rendimientos ajustados y optimizando...")
    
    st.divider()
    
    # Resultados
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Rendimientos Esperados")
        
        # Comparación de rendimientos
        rend_comp_df = pd.DataFrame({
            'ETF': list(BENCHMARKS[estrategia].keys()),
            'Rendimiento Implícito (%)': ['---'] * len(BENCHMARKS[estrategia]),
            'Rendimiento BL (%)': ['---'] * len(BENCHMARKS[estrategia])
        })
        st.dataframe(rend_comp_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.subheader("💼 Portafolio Optimizado BL")
        
        pesos_bl_df = pd.DataFrame({
            'ETF': list(BENCHMARKS[estrategia].keys()),
            'Peso (%)': ['---'] * len(BENCHMARKS[estrategia])
        })
        st.dataframe(pesos_bl_df, hide_index=True, use_container_width=True)
        
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
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Rendimiento Esperado", "---")
            st.metric("Sharpe Ratio", "---")
        with col_b:
            st.metric("Volatilidad", "---")
            st.metric("Tracking Error vs Benchmark", "---")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: gray; padding: 2rem;'>
        <p>Portfolio Manager Pro | Análisis Cuantitativo de Inversiones</p>
        <p style='font-size: 0.8rem;'>Desarrollado con Streamlit | © 2024</p>
    </div>
""", unsafe_allow_html=True)
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
