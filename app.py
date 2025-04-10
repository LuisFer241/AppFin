# === Archivo: app.py ===

import streamlit as st
from datetime import date
import yfinance as yf
import pandas as pd
import numpy as np
from deep_translator import GoogleTranslator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------- ESTILO PERSONALIZADO ----------------------------

def apply_custom_css():
    st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #f5f5f5;
    }
    @media (prefers-color-scheme: dark) {
      section[data-testid="stSidebar"] {
        background-color: #1e1e1e;
      }
    }
    .stButton > button {
        background-color: #1a73e8;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 18px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #0f5cd7;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------- FUNCIONES AUXILIARES ----------------------------

@st.cache_data(show_spinner=False)
def get_stock_data(ticker: str, start: str, end: str):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end)
        info = stock.info
        return df, info
    except Exception as e:
        st.error(f"Error al obtener datos: {e}")
        return pd.DataFrame(), {}

@st.cache_data(show_spinner=False)
def traducir_descripcion(texto: str, idioma_destino: str) -> str:
    if not texto:
        return "No hay descripción disponible."
    try:
        return GoogleTranslator(source='auto', target=idioma_destino).translate(texto)
    except:
        return "No se pudo traducir la descripción."

def show_company_info(info, idioma):
    with st.expander("📘 Información de la Empresa"):
        descripcion = info.get("longBusinessSummary", "")
        st.write(traducir_descripcion(descripcion, idioma))
        st.write(f"**📌 Industria:** {info.get('industry', 'N/D')}")
        st.write(f"**🏢 Capitalización:** {info.get('marketCap', 'N/D')}")
        st.write(f"**📊 P/E Ratio:** {info.get('trailingPE', 'N/D')}")
        st.write(f"**💸 Dividend Yield:** {info.get('dividendYield', 'N/D')}")

def explain_metric(nombre, descripcion):
    with st.expander(f"ℹ️ ¿Qué es {nombre}?"):
        st.write(descripcion)

# ---------------------------- FUNCIONES FINANCIERAS ----------------------------

@st.cache_data(show_spinner=False)
def calcular_cagr(df, años):
    try:
        fecha_final = df.index[-1]
        fecha_inicio = fecha_final.replace(year=fecha_final.year - años)
        df_filtrado = df[df.index >= fecha_inicio]
        if len(df_filtrado) < 2:
            return None
        inicio = df_filtrado['Close'].iloc[0]
        fin = df_filtrado['Close'].iloc[-1]
        cagr = (fin / inicio) ** (1 / años) - 1
        return cagr * 100
    except:
        return None

def calcular_volatilidad_anual(df):
    retornos = df['Close'].pct_change().dropna()
    volatilidad = np.std(retornos) * np.sqrt(252)
    return volatilidad * 100

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
    fig = px.line(df, x='Date', y=[y, y_pred], labels={"value": "Precio", "variable": "Serie"}, title="Predicción del Precio con Regresión Lineal")
    st.plotly_chart(fig)
    error = np.sqrt(mean_squared_error(y, y_pred))
    st.write(f"📉 Error RMSE: {error:.2f}")

# ---------------------------- GRÁFICOS ----------------------------

def plot_stock_data_interactive(df, ticker):
    st.subheader("📈 Evolución del Precio")
    fig = px.line(df, x=df.index, y='Close', title=f"Precio histórico de cierre - {ticker.upper()}")
    fig.update_layout(xaxis_title="Fecha", yaxis_title="Precio de Cierre")
    st.plotly_chart(fig, use_container_width=True)

def plot_moving_averages(indicators_df):
    st.subheader("📊 Medias Móviles")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=indicators_df.index, y=indicators_df['MA50'], name='MA50'))
    fig.add_trace(go.Scatter(x=indicators_df.index, y=indicators_df['MA200'], name='MA200'))
    fig.update_layout(title="Medias móviles 50 y 200 días", xaxis_title="Fecha", yaxis_title="Precio")
    st.plotly_chart(fig, use_container_width=True)

def plot_rsi_chart(indicators_df):
    st.subheader("📈 RSI (Índice de Fuerza Relativa)")
    fig = px.line(indicators_df, x=indicators_df.index, y='RSI', title='RSI')
    fig.add_hline(y=70, line_dash="dot", line_color="red")
    fig.add_hline(y=30, line_dash="dot", line_color="green")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------- INTERFAZ PRINCIPAL ----------------------------

st.set_page_config(page_title="Analista Financiero Pro", layout="wide")
apply_custom_css()

st.sidebar.title("Parámetros de Análisis")
TICKERS_EMPRESAS = {"apple": "AAPL", "amazon": "AMZN", "microsoft": "MSFT", "tesla": "TSLA", "google": "GOOGL"}

nombre_empresa = st.sidebar.text_input("Nombre de la empresa:")
ticker_default = TICKERS_EMPRESAS.get(nombre_empresa.lower().strip(), "AAPL")
ticker = st.sidebar.text_input("Ticker de la empresa:", value=ticker_default)
idioma = st.sidebar.selectbox("Idioma de la descripción:", ["es", "en"], format_func=lambda x: {"es": "Español", "en": "Inglés"}[x])
start_date = st.sidebar.date_input("Desde:", value=date(2019, 1, 1))
end_date = st.sidebar.date_input("Hasta:", value=date.today())
nivel = st.sidebar.radio("Nivel de análisis:", ["Principiante", "Intermedio", "Experto"])

if st.sidebar.button("Analizar"):
    with st.spinner("Obteniendo datos financieros..."):
        df, info = get_stock_data(ticker, start_date, end_date)
        if df.empty:
            st.error("No se encontraron datos para este ticker.")
        else:
            st.title(f"Análisis Financiero: {info.get('shortName', ticker)}")
            show_company_info(info, idioma)
            plot_stock_data_interactive(df, ticker)

            st.subheader("Crecimiento: Rendimiento Anual Compuesto (CAGR)")
            cagr1 = calcular_cagr(df, 1)
            cagr3 = calcular_cagr(df, 3)
            cagr5 = calcular_cagr(df, 5)
            explain_metric("CAGR", "Muestra cuánto ha crecido una inversión cada año, en promedio.")
            st.write("1 año:", f"{cagr1:.2f}%" if cagr1 else "N/D")
            st.write("3 años:", f"{cagr3:.2f}%" if cagr3 else "N/D")
            st.write("5 años:", f"{cagr5:.2f}%" if cagr5 else "N/D")

            st.subheader("Volatilidad")
            vol = calcular_volatilidad_anual(df)
            explain_metric("Volatilidad anual", "Qué tanto varía el precio. Alta volatilidad = más riesgo.")
            st.metric("Volatilidad histórica", f"{vol:.2f}%")

            if nivel == "Intermedio":
                st.subheader("Indicadores técnicos")
                roi, volat, indicadores = calculate_indicators(df)
                st.metric("ROI Promedio", f"{roi:.2f}%")
                plot_moving_averages(indicadores)
                plot_rsi_chart(indicadores)

            elif nivel == "Experto":
                st.subheader("Indicadores Avanzados")
                sp500_df, _ = get_stock_data("^GSPC", start_date, end_date)
                beta, sharpe = advanced_analysis(df, sp500_df)
                st.metric("Beta", f"{beta:.2f}")
                st.metric("Índice de Sharpe", f"{sharpe:.2f}")
                predict_stock_price(df)















