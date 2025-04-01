import streamlit as st
st.set_page_config(page_title="🧠 Analista Financiero", layout="wide")  # ✅ Esto va primero

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from deep_translator import GoogleTranslator

# ----------------------------
# 🌈 CUSTOM CSS
# ----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto&display=swap');

html, body, [class*="css"]  {
    font-family: 'Roboto', sans-serif;
    background-color: #1e1e1e;
    color: #f0f0f0;
}

h1, h2, h3, h4 {
    color: #00c0ff;
}

.stSidebar {
    background-color: #2c2c2c;
}

.stButton > button {
    background-color: #00c0ff;
    color: black;
    border-radius: 5px;
    padding: 0.5rem 1rem;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #009ac9;
    color: white;
}

[data-testid="stMetricValue"] {
    font-size: 1.5em;
    font-weight: bold;
    color: #00ffe0;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# 🔎 Diccionario empresa → ticker
# ----------------------------
empresa_ticker = {
    "apple": "AAPL", "amazon": "AMZN", "microsoft": "MSFT", "tesla": "TSLA",
    "google": "GOOGL", "meta": "META", "coca cola": "KO", "nvidia": "NVDA",
    "netflix": "NFLX", "disney": "DIS"
}

# ----------------------------
# 🔧 Funciones
# ----------------------------
@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    info = stock.info
    return df, info

@st.cache_data
def traducir_descripcion(texto, idioma_destino):
    if not texto:
        return "No hay descripción disponible."
    try:
        return GoogleTranslator(source='auto', target=idioma_destino).translate(texto)
    except:
        return "No se pudo traducir la descripción."

def plot_stock_data(df):
    st.subheader("📈 Evolución del Precio")
    fig, ax = plt.subplots()
    ax.plot(df.index, df['Close'], label='Precio de Cierre')
    ax.set_ylabel('Precio de Cierre')
    ax.set_xlabel('Fecha')
    ax.legend()
    st.pyplot(fig)

    st.subheader("📊 Volumen de Transacciones")
    fig, ax = plt.subplots()
    ax.plot(df.index, df['Volume'], label='Volumen', color='gray')
    ax.set_ylabel('Volumen')
    ax.set_xlabel('Fecha')
    ax.legend()
    st.pyplot(fig)

def calculate_indicators(df):
    df['ROI'] = (df['Close'] - df['Open']) / df['Open']
    roi = df['ROI'].mean() * 100
    volatility = df['Close'].std()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return roi, volatility, df[['MA50', 'MA200', 'RSI']]

def advanced_analysis(df, sp500_df):
    df['Returns'] = df['Close'].pct_change()
    sp500_df['Returns'] = sp500_df['Close'].pct_change()
    beta = np.cov(df['Returns'].dropna(), sp500_df['Returns'].dropna())[0][1] / np.var(sp500_df['Returns'].dropna())
    sharpe_ratio = df['Returns'].mean() / df['Returns'].std() * np.sqrt(252)
    return beta, sharpe_ratio

def predict_stock_price(df):
    df = df.reset_index()
    df['Date_Ordinal'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)
    X = df[['Date_Ordinal']]
    y = df['Close']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    fig, ax = plt.subplots()
    ax.plot(df['Date'], y, label='Precio Real')
    ax.plot(df['Date'], y_pred, label='Predicción', linestyle='--', color='orange')
    ax.set_title("🔮 Predicción del Precio con Regresión Lineal")
    ax.legend()
    st.pyplot(fig)
    error = np.sqrt(mean_squared_error(y, y_pred))
    st.write(f"📉 RMSE (Error Cuadrático Medio): {error:.2f}")

# ----------------------------
# 🚀 INTERFAZ PRINCIPAL
# ----------------------------
st.sidebar.title("📊 Parámetros de Análisis")

# Inicializar ticker
if "ticker" not in st.session_state:
    st.session_state.ticker = "AAPL"

# Búsqueda por nombre
nombre_empresa = st.sidebar.text_input("🔍 Escribe el nombre de la empresa:")
nombre_normalizado = nombre_empresa.lower().strip()
if st.sidebar.button("🔎 Buscar Ticker"):
    sugerido = empresa_ticker.get(nombre_normalizado)
    if sugerido:
        st.session_state.ticker = sugerido
        st.sidebar.success(f"Ticker: {sugerido}")
    else:
        st.sidebar.warning("Empresa no encontrada.")

# Entrada de ticker
ticker = st.sidebar.text_input("🏷️ Ticker seleccionado o manual:", value=st.session_state.ticker)

# Selector de idioma
idioma = st.sidebar.selectbox("🌐 Idioma de la descripción", ["es", "en", "fr", "pt", "de", "it"],
    format_func=lambda x: {"es": "Español", "en": "Inglés", "fr": "Francés", "pt": "Portugués", "de": "Alemán", "it": "Italiano"}[x])

# Fechas
start_date = st.sidebar.date_input("📅 Desde:", value=date(2022, 1, 1))
end_date = st.sidebar.date_input("📅 Hasta:", value=date.today())
nivel = st.sidebar.radio("🎓 Nivel de Análisis:", ["Principiante", "Intermedio", "Experto"])

# Botón analizar
if st.sidebar.button("📊 Analizar"):
    try:
        df, info = get_stock_data(ticker, start_date, end_date)
        if df.empty:
            st.error("No se encontraron datos para este ticker.")
        else:
            st.title(f"📈 Análisis Financiero de {info.get('shortName', ticker)}")

            # Info básica traducida
            with st.expander("📘 Ver información de la empresa"):
                desc = info.get("longBusinessSummary", "")
                st.markdown(traducir_descripcion(desc, idioma))

            st.markdown(f"**💵 Precio actual:** ${info.get('currentPrice', 'N/D')}")
            st.markdown(f"**🏢 Capitalización:** {info.get('marketCap', 'N/D')}")
            st.markdown(f"**📊 P/E Ratio:** {info.get('trailingPE', 'N/D')}")
            st.markdown(f"**💸 Dividend Yield:** {info.get('dividendYield', 'N/D')}")

            # NIVEL PRINCIPIANTE
            if nivel == "Principiante":
                st.header("👶 Análisis para Principiantes")
                plot_stock_data(df)
                cambio = (df['Close'][-1] - df['Close'][0]) / df['Close'][0] * 100
                tendencia = "alcista 📈" if cambio > 0 else "bajista 📉"
                st.success(f"El precio cambió un {cambio:.2f}%. Tendencia: {tendencia}")
                promedio_vol = df['Volume'].mean()
                st.info(f"Volumen promedio: {promedio_vol:,.0f}")
            
            # NIVEL INTERMEDIO
            elif nivel == "Intermedio":
                st.header("🧭 Análisis para Intermedios")
                roi, volat, indicadores = calculate_indicators(df)
                st.metric("🔁 ROI Promedio", f"{roi:.2f}%")
                st.metric("📉 Volatilidad", f"{volat:.2f}")
                st.line_chart(indicadores[['MA50', 'MA200']])
                st.line_chart(indicadores['RSI'])

                with st.expander("ℹ️ ¿Qué significan estas gráficas?"):
                    st.markdown("""
### 📊 Medias Móviles (MA50 y MA200)
- MA50: Promedio 50 días.
- MA200: Promedio 200 días.

**Interpretación:**
- MA50 cruza arriba = señal alcista.
- MA50 cruza abajo = señal bajista.

---

### 📈 RSI (Índice de Fuerza Relativa)
- RSI > 70 → Posible sobrecompra.
- RSI < 30 → Posible sobreventa.
- 40-60 → Zona neutral.
                    """)

                rsi = indicadores['RSI'].iloc[-1]
                if rsi > 70:
                    st.warning("⚠️ RSI alto → posible sobrecompra.")
                elif rsi < 30:
                    st.success("💡 RSI bajo → posible oportunidad de compra.")

            # NIVEL EXPERTO
            elif nivel == "Experto":
                st.header("🏦 Análisis para Expertos")
                sp_df, _ = get_stock_data("^GSPC", start_date, end_date)
                beta, sharpe = advanced_analysis(df, sp_df)
                st.metric("📈 Beta", f"{beta:.2f}")
                st.metric("📊 Sharpe", f"{sharpe:.2f}")
                predict_stock_price(df)
                if beta > 1:
                    st.warning("⚠️ Alta volatilidad.")
                else:
                    st.info("✅ Riesgo controlado.")
                if sharpe > 1:
                    st.success("💰 Buen rendimiento ajustado al riesgo.")
                else:
                    st.warning("📉 Sharpe bajo.")
    except Exception as e:
        st.error(f"Ocurrió un error: {str(e)}")








