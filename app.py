import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import json
import time
import uuid
from datetime import date, datetime
from data_loader import download_data
import scipy.optimize as sco
from scipy.optimize import minimize

# Función para optimizar los pesos del portafolio
def optimize_portfolio_weights(returns_data):
    """
    Optimiza los pesos del portafolio usando el ratio de Sharpe
    
    Args:
        returns_data: DataFrame con los retornos diarios de los activos
        
    Returns:
        Lista con los pesos optimizados (porcentajes)
    """
    from scipy.optimize import minimize
    import numpy as np
    
    # Función objetivo: negativo del ratio de Sharpe (para maximizarlo)
    def neg_sharpe_ratio(weights, returns, risk_free_rate=0.02/252):
        weights = np.array(weights)
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
        return -sharpe
    
    # Restricciones: la suma de los pesos debe ser 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Límites: cada peso entre 0 y 1
    n_assets = len(returns_data.columns)
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Pesos iniciales iguales
    initial_weights = np.array([1/n_assets] * n_assets)
    
    # Optimización
    try:
        result = minimize(neg_sharpe_ratio, 
                          initial_weights, 
                          args=(returns_data,), 
                          method='SLSQP', 
                          bounds=bounds, 
                          constraints=constraints)
        
        # Convertir a porcentajes
        optimized_weights = [w * 100 for w in result['x']]

        # Normalizar para asegurar que sumen exactamente 100%
        total_weight = sum(optimized_weights)
        optimized_weights = [round(w * 100 / total_weight, 2) for w in optimized_weights]

        # Ajuste final para manejar errores de redondeo
        difference = 100 - sum(optimized_weights)
        if abs(difference) > 0:
            # Añadir la diferencia al peso más grande para minimizar el impacto
            max_index = optimized_weights.index(max(optimized_weights))
            optimized_weights[max_index] += difference
            optimized_weights[max_index] = round(optimized_weights[max_index], 2)
        
        return optimized_weights
    except Exception as e:
        # En caso de error, retornar pesos iguales
        equal_weight = round(100 / n_assets, 2)
        weights = [equal_weight] * n_assets
        weights[-1] = 100 - sum(weights[:-1])
        return weights
def calculate_efficient_frontier(returns_data, num_portfolios=50):
    """
    Calcula la frontera eficiente generando múltiples portafolios óptimos

    Args:
        returns_data: DataFrame con los retornos diarios
        num_portfolios: Número de portafolios a generar para la frontera

    Returns:
        Tuple con arrays de retornos, volatilidades y ratios de Sharpe
    """
    mean_returns = returns_data.mean() * 252
    cov_matrix = returns_data.cov() * 252
    num_assets = len(mean_returns)

    # Función para calcular estadísticas del portafolio
    def portfolio_stats(weights):
        weights = np.array(weights)
        ret = np.sum(mean_returns * weights)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = ret / vol if vol != 0 else 0
        return ret, vol, sharpe
    
    # Función objetivo para minimizar volatilidad dado un retorno objetivo
    def minimize_volatility(weights, target_return):
        ret, vol, sharpe = portfolio_stats(weights)
        return vol
    
    # Generar rango de retornos objetivo
    min_ret = mean_returns.min()
    max_ret = mean_returns.max()
    target_returns = np.linspace(min_ret, max_ret, num_portfolios)
    
    # Listas para almacenar resultados
    returns_list = []
    volatility_list = []
    sharpe_list = []
    
    # Restricciones y límites
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    for target in target_returns:
        # Restricciones para cada optimización
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x, target=target: portfolio_stats(x)[0] - target}
        ]
        
        # Optimización
        try:
            result = minimize(
                minimize_volatility,
                np.array([1/num_assets] * num_assets),
                args=(target,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'ftol': 1e-9, 'disp': False}
            )
            
            if result.success:
                ret, vol, sharpe = portfolio_stats(result.x)
                returns_list.append(ret)
                volatility_list.append(vol)
                sharpe_list.append(sharpe)
        except:
            continue
    
    # Convertir a arrays numpy
    if returns_list:
        returns_array = np.array(returns_list)
        volatility_array = np.array(volatility_list)
        sharpe_array = np.array(sharpe_list)
        
        # Ordenar por volatilidad para una frontera más suave
        sorted_indices = np.argsort(volatility_array)
        
        return (returns_array[sorted_indices], 
                volatility_array[sorted_indices], 
                sharpe_array[sorted_indices])
    else:
        return np.array([]), np.array([]), np.array([])
        
# ---------------- CONFIGURACIÓN ----------------
st.set_page_config(page_title="Portfolio Analyzer", layout="wide")

# Cargar traducción
def load_translation(lang):
    file_path = os.path.join("translations", f"{lang}.json")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error al cargar el archivo de traducción ({lang}.json): {e}")
        return {}  # Retorna diccionario vacío en lugar de None

# Función de traducción mejorada con cache
@st.cache_data
def get_translations(lang):
    return load_translation(lang)

# Inicializar estado de la sesión si no existe
if 'app_id' not in st.session_state:
    st.session_state.app_id = str(uuid.uuid4())[:8]  # ID único para evitar colisiones de key
    
if 'tickers' not in st.session_state:
    st.session_state.tickers = ["", "", ""]  # Valores iniciales para 3 tickers
if 'weights' not in st.session_state:
    st.session_state.weights = [0.0, 0.0, 0.0]  # Inicializar pesos en cero para luego optimizarlos
if 'num_assets' not in st.session_state:
    st.session_state.num_assets = 3
if 'cache_bust' not in st.session_state:
    st.session_state.cache_bust = 0  # Para forzar recarga cuando sea necesario

# Función para actualizar el número de activos
def update_num_assets():
    new_num = st.session_state.new_num_assets
    current_num = len(st.session_state.tickers)
    
    if new_num > current_num:
        st.session_state.tickers.extend([""] * (new_num - current_num))
        # Inicializar todos los pesos en cero
        st.session_state.weights = [0.0] * new_num
    elif new_num < current_num:
        st.session_state.tickers = st.session_state.tickers[:new_num]
        st.session_state.weights = [0.0] * new_num

    st.session_state.num_assets = new_num
    # Regenerar UUIDs para keys únicas
    st.session_state.cache_bust += 1

# Sidebar - idioma
st.sidebar.title("🌐 Language / Idioma")
language = st.sidebar.selectbox("Choose / Elige", ["es", "en"], index=0, key=f"lang_{st.session_state.app_id}")
# Cargar traducciones como diccionario
translations = get_translations(language)

# Función de traducción mejorada
def t(key):
    """Función para obtener la traducción de una clave con manejo inteligente de fallbacks"""
    if not translations:
        return key.replace("_", " ").title()
    
    # Buscar la clave exacta
    if key in translations:
        return translations[key]
    
    # Fallback 1: Intentar con la clave en minúsculas
    if key.lower() in translations:
        return translations[key.lower()]
    
    # Fallback 2: Convertir guiones bajos en espacios y capitalizar
    return key.replace("_", " ").title()

# Mostrar la fecha de actualización
current_date = datetime.now().strftime("%d-%m-%Y %H:%M")
st.sidebar.caption(f"{t('last_update')}: {current_date}")

# Sidebar - fecha
st.sidebar.markdown("### " + t("date_range"))
start_date = st.sidebar.date_input(t("start_date"), value=date(2018, 1, 1), key=f"start_date_{st.session_state.app_id}")
end_date = st.sidebar.date_input(t("end_date"), value=date.today(), key=f"end_date_{st.session_state.app_id}")

# ---------------- INTERFAZ PRINCIPAL ----------------
st.title(t("app_title"))
tab1, tab2, tab3 = st.tabs([t("tab_portfolio"), t("tab_performance"), t("tab_settings")])

with tab1:
    st.header(t("portfolio_inputs"))
    st.info(t("instructions"))

    # Usar un callback para actualizar el número de activos
    st.number_input(
        t("num_assets"), 
        min_value=1, 
        max_value=10, 
        value=st.session_state.num_assets,
        step=1, 
        key=f"new_num_assets_{st.session_state.cache_bust}",
        on_change=update_num_assets
    )

    # Ejemplos de tickers populares
    ticker_examples = {
        "US Stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
        "US ETFs": ["SPY", "QQQ", "VTI", "VOO", "VGT"],
        "Spanish": ["SAN.MC", "BBVA.MC", "ITX.MC", "REP.MC", "IBE.MC"],
        "Indices": ["^GSPC", "^IXIC", "^DJI", "^IBEX"]
    }
    
    # Ejemplos de tickers
    with st.expander(t("view_ticker_examples")):
        col1, col2, col3, col4 = st.columns(4)
        
        def insertar_ticker(ticker):
            for i, t in enumerate(st.session_state.tickers):
                if not t.strip():
                    st.session_state.tickers[i] = ticker
                    st.rerun()
                    break
            # Si no hay espacios vacíos, reemplazar el primero
            if all(t.strip() for t in st.session_state.tickers):
                st.session_state.tickers[0] = ticker
                st.rerun()

        with col1:
            st.write("**US Stocks**")
            for ticker in ticker_examples["US Stocks"]:
                if st.button(ticker, key=f"ex_us_{ticker}_{st.session_state.cache_bust}"):
                    insertar_ticker(ticker)
        
        with col2:
            st.write("**US ETFs**")
            for ticker in ticker_examples["US ETFs"]:
                if st.button(ticker, key=f"ex_etf_{ticker}_{st.session_state.cache_bust}"):
                    insertar_ticker(ticker)
        
        with col3:
            st.write("**España**")
            for ticker in ticker_examples["Spanish"]:
                if st.button(ticker, key=f"ex_es_{ticker}_{st.session_state.cache_bust}"):
                    insertar_ticker(ticker)
        
        with col4:
            st.write("**Índices**")
            for ticker in ticker_examples["Indices"]:
                if st.button(ticker, key=f"ex_idx_{ticker}_{st.session_state.cache_bust}"):
                    insertar_ticker(ticker)

    # Inputs de tickers y pesos
    col1, col2 = st.columns(2)

    def update_ticker(i):
        st.session_state.tickers[i] = st.session_state[f"ticker_{i}_{st.session_state.cache_bust}"]

    def update_weight(i):
        st.session_state.weights[i] = st.session_state[f"weight_{i}_{st.session_state.cache_bust}"]
        # Validar que los pesos sumen 100%
        if sum(st.session_state.weights) > 100:
            # Ajustar proporcionalmente
            excess = sum(st.session_state.weights) - 100
            st.session_state.weights[i] -= excess

    for i in range(st.session_state.num_assets):
        with col1:
            st.text_input(
                f"{t('ticker_label')} {i+1}",
                value=st.session_state.tickers[i] if i < len(st.session_state.tickers) else "",
                key=f"ticker_{i}_{st.session_state.cache_bust}",
                on_change=update_ticker,
                args=(i,)
            )
        with col2:
            st.number_input(
                f"{t('weight_label')} {i+1}",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.weights[i] if i < len(st.session_state.weights) else 0.0,
                format="%.2f",
                key=f"weight_{i}_{st.session_state.cache_bust}",
                on_change=update_weight,
                args=(i,)
            )
        # Botón para optimizar pesos después de ingresar tickers
    if any(ticker.strip() for ticker in st.session_state.tickers):  # Verificar que hay al menos un ticker
        if st.button(t("optimize_weights") if "optimize_weights" in translations else "Optimize Weights", 
                     key=f"optimize_weights_{st.session_state.cache_bust}"):
            with st.spinner(t("downloading_data") if "downloading_data" in translations else "Downloading data for optimization..."):
                try:
                    # Filtrar tickers vacíos
                    tickers = [t for t in st.session_state.tickers if t.strip()]
                    
                    if tickers:
                        # Descargar datos históricos
                        data = download_data(tickers, start=start_date, end=end_date)
                        
                        if not data.empty:
                            # Calcular retornos diarios
                            returns = data.pct_change().dropna()
                            
                            # Optimizar pesos
                            optimized_weights = optimize_portfolio_weights(returns)
                            
                            # Actualizar pesos en la sesión
                            for i, weight in enumerate(optimized_weights):
                                if i < len(st.session_state.weights):
                                    st.session_state.weights[i] = weight
                                    
                            st.success(t("weights_optimized") if "weights_optimized" in translations else "Portfolio weights optimized for maximum Sharpe ratio!")
                            st.rerun()
                        else:
                            st.error(t("no_data_warning") if "no_data_warning" in translations else "No data found for the selected tickers.")
                    else:
                        st.warning(t("no_tickers_error") if "no_tickers_error" in translations else "Please enter at least one ticker before optimizing.")
                except Exception as e:
                    st.error(f"{t('optimization_error') if 'optimization_error' in translations else 'Error during optimization'}: {e}")
                
    # Filtrar tickers vacíos
    tickers = [t for t in st.session_state.tickers if t.strip()]
    weights = st.session_state.weights[:len(tickers)]
    total_weight = sum(weights)

    if not tickers:
        st.error(t("no_tickers_error"))
    else:
        # Mostrar tabla de tickers actuales
        st.write(t("selected_tickers"))
        
        # Preparar DataFrame con mejor formato
        df_portfolio = pd.DataFrame({
            t("column_ticker"): tickers,
            t("column_weight"): [f"{w:.2f}%" for w in weights]
        })
        
        # Estilo de la tabla mejorado
        st.dataframe(
            df_portfolio,
            use_container_width=True,
            hide_index=True
        )
        
        if abs(total_weight - 100) > 0.01:  # Permitir un pequeño margen de error
            st.warning(t("weight_warning"))
            # Botón para normalizar pesos automáticamente
            if st.button(t("normalize_weights"), key=f"normalize_weights_{st.session_state.cache_bust}"):
                # Normalizar pesos para sumar 100%
                normalized_weights = [w * 100 / total_weight for w in weights]
                for i, w in enumerate(normalized_weights):
                    if i < len(st.session_state.weights):
                        st.session_state.weights[i] = round(w, 2)
                # Ajustar el último para asegurar suma exacta de 100
                if len(st.session_state.weights) > 0:
                    st.session_state.weights[-1] = round(100 - sum(st.session_state.weights[:-1]), 2)
                st.rerun()

        # Botón para descargar datos
        analyze_button = st.button(
            "📊 " + t("analyze_portfolio"),
            key=f"analyze_portfolio_button_{st.session_state.app_id}_{st.session_state.cache_bust}"
        )
        
        if analyze_button:
            with st.spinner(t("downloading_data")):
                try:
                    data = download_data(tickers, start=start_date, end=end_date)
                    
                    if data.empty:
                        st.error(t("no_data_warning"))
                        with st.expander(t("error_diagnostics")):
                            st.markdown(f"""
                            **{t("possible_error_causes")}:**
                            
                            1. **{t("ticker_format_problem")}**:
                               - {t("verify_format")}
                               - {t("us_stocks")}: `AAPL`, `MSFT`, `GOOGL`
                               - {t("spanish_stocks")}: `SAN.MC`, `BBVA.MC`, `ITX.MC`
                               - {t("indices")}: `^GSPC` (S&P 500), `^IXIC` (Nasdaq)
                            
                            2. **{t("internet_connection")}**:
                               - {t("verify_connection")}
                            
                            3. **{t("correct_installation")}**:
                               - {t("ensure_installation")} `pip install yfinance --upgrade`
                            
                            4. **{t("api_limits")}**:
                               - {t("yahoo_limits")}
                               - {t("try_fewer_tickers")}
                            """)
                    else:
                        st.write("📊 " + t("historical_data_downloaded"))
                        
                        # Mejorar visualización de datos históricos
                        st.dataframe(
                            data.style.format("{:.2f}"),
                            use_container_width=True,
                            height=250
                        )

                        # Crear datos normalizados para el gráfico
                        normalized_data = data / data.iloc[0] * 100

                        # Gráfico mejorado
                        fig = go.Figure()
                        colors = [
                            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
                        ]
                        
                        for i, col in enumerate(normalized_data.columns):
                            fig.add_trace(go.Scatter(
                                x=normalized_data.index,
                                y=normalized_data[col],
                                mode='lines',
                                name=col,
                                line=dict(width=2, color=colors[i % len(colors)])
                            ))
                        
                        fig.update_layout(
                            title=t("portfolio_chart"),
                            xaxis_title=t("date"),
                            yaxis_title=t("growth_base_100"),
                            template="plotly_dark",
                            height=450,
                            hovermode="x unified",
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Cálculo del retorno del portafolio
                        returns = data.pct_change().dropna()
                        
                        # Normalizar pesos
                        weights_array = np.array(weights) / sum(weights)
                        
                        # Calcular retornos del portafolio
                        portfolio_returns = returns.dot(weights_array)
                        
                        # Métricas mejoradas
                        days = (end_date - start_date).days
                        years = days / 365.25
                        
                        # Calcular CAGR correctamente
                        portfolio_value = (1 + portfolio_returns).cumprod()
                        if len(portfolio_value) > 0:
                            total_return = portfolio_value.iloc[-1] - 1
                            cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
                        else:
                            cagr = 0
                            
                        volatility = portfolio_returns.std() * np.sqrt(252)
                        sharpe_ratio = cagr / volatility if volatility != 0 else 0
                        
                        # Cálculo mejorado de drawdown
                        cumulative_returns = (1 + portfolio_returns).cumprod()
                        peak = cumulative_returns.cummax()
                        drawdowns = (cumulative_returns - peak) / peak
                        max_drawdown = drawdowns.min()
                        
                        # Cálculo de rendimiento mensual promedio
                        monthly_returns = cumulative_returns.resample('M').last().pct_change().dropna()
                        avg_monthly_return = monthly_returns.mean()
                        
                        # Porcentaje de meses positivos
                        positive_months = (monthly_returns > 0).sum() / len(monthly_returns) if len(monthly_returns) > 0 else 0

                        # Gráfico de rendimiento del portafolio
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(
                            x=cumulative_returns.index,
                            y=cumulative_returns,
                            mode='lines',
                            name=t("portfolio"),
                            line=dict(color='green', width=2)
                        ))
                        
                        # Añadir línea de tendencia
                        if len(cumulative_returns) > 1:
                            x_numeric = np.array(range(len(cumulative_returns)))
                            y = np.array(cumulative_returns)
                            fit = np.polyfit(x_numeric, y, 1)
                            trend = np.poly1d(fit)
                            
                            fig2.add_trace(go.Scatter(
                                x=cumulative_returns.index,
                                y=trend(x_numeric),
                                mode='lines',
                                name=t("trend"),
                                line=dict(color='white', width=1, dash='dash')
                            ))
                        
                        fig2.update_layout(
                            title=t("portfolio_cumulative_performance"),
                            xaxis_title=t("date"),
                            yaxis_title=t("growth"),
                            template='plotly_dark',
                            height=400,
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                        # Frontera Eficiente
                        with st.expander(t("efficient_frontier") if "efficient_frontier" in translations else "📈 Frontera Eficiente"):
                            with st.spinner(t("calculating_frontier") if "calculating_frontier" in translations else "Calculando frontera eficiente..."):
                                try:
                                    # Calcular frontera eficiente
                                    ef_returns, ef_volatility, ef_sharpe = calculate_efficient_frontier(returns)
                                    
                                    if len(ef_returns) > 0:
                                        # Crear gráfica de frontera eficiente
                                        fig_ef = go.Figure()
                                        
                                        # Línea de frontera eficiente
                                        fig_ef.add_trace(go.Scatter(
                                        x=ef_volatility,
                                        y=ef_returns,
                                        mode='lines+markers',
                                        name=t("efficient_frontier") if "efficient_frontier" in translations else "Frontera Eficiente",
                                        marker=dict(
                                            color=ef_sharpe,
                                            colorscale='Viridis',
                                            size=6,
                                            showscale=True,
                                            colorbar=dict(title="Sharpe Ratio")
                                        ),
                                        line=dict(width=2)
                                        )
                                    )

                                        
                                        # Punto del portafolio actual
                                        current_return = np.sum(returns.mean() * weights_array) * 252
                                        current_volatility = np.sqrt(np.dot(weights_array.T, np.dot(returns.cov() * 252, weights_array)))
                                        
                                        fig_ef.add_trace(go.Scatter(
                                            x=[current_volatility],
                                            y=[current_return],
                                            mode='markers',
                                            name=t("current_portfolio") if "current_portfolio" in translations else "Portafolio Actual",
                                            marker=dict(
                                                size=12,
                                                color='red',
                                                symbol='star',
                                                line=dict(width=2, color='white')
                                            )
                                        ))
                                        
                                        # Portafolio de máximo Sharpe
                                        max_sharpe_idx = np.argmax(ef_sharpe)
                                        fig_ef.add_trace(go.Scatter(
                                            x=[ef_volatility[max_sharpe_idx]],
                                            y=[ef_returns[max_sharpe_idx]],
                                            mode='markers',
                                            name=t("max_sharpe_portfolio") if "max_sharpe_portfolio" in translations else "Máximo Sharpe",
                                            marker=dict(
                                                size=10,
                                                color='green',
                                                symbol='diamond',
                                                line=dict(width=2, color='white')
                                            )
                                        ))
                                        
                                        # Portafolio de mínima volatilidad
                                        min_vol_idx = np.argmin(ef_volatility)
                                        fig_ef.add_trace(go.Scatter(
                                            x=[ef_volatility[min_vol_idx]],
                                            y=[ef_returns[min_vol_idx]],
                                            mode='markers',
                                            name=t("min_volatility_portfolio") if "min_volatility_portfolio" in translations else "Mínima Volatilidad",
                                            marker=dict(
                                                size=10,
                                                color='orange',
                                                symbol='square',
                                                line=dict(width=2, color='white')
                                            )
                                        ))
                                        
                                        fig_ef.update_layout(
                                            title=t("efficient_frontier_chart") if "efficient_frontier_chart" in translations else "Frontera Eficiente",
                                            xaxis_title=t("volatility") if "volatility" in translations else "Volatilidad (%)",
                                            yaxis_title=t("expected_return") if "expected_return" in translations else "Retorno Esperado (%)",
                                            template="plotly_dark",
                                            height=500,  # Aumentar altura para dar más espacio
                                            width=1000,
                                            hovermode="closest",
                                            # Configurar la leyenda para evitar superposición
                                            legend=dict(
                                                orientation="v",  # Vertical en lugar de horizontal
                                                yanchor="top",
                                                y=1,
                                                xanchor="left",
                                                x=1.01,  # Mover fuera del área del gráfico
                                                bgcolor="rgba(0,0,0,0.5)",  # Fondo semi-transparente
                                                bordercolor="rgba(255,255,255,0.3)",
                                                borderwidth=1,
                                                font=dict(size=10)  # Reducir tamaño de fuente
                                            ),
                                            # Ajustar márgenes para acomodar la leyenda
                                            margin=dict(l=60, r=180, t=60, b=60),
                                            # Formatear ejes como porcentajes
                                            xaxis=dict(
                                                tickformat='.1%',
                                                showgrid=True,
                                                gridcolor="rgba(255,255,255,0.1)",
                                                title_font=dict(size=12)
                                            ),
                                            yaxis=dict(
                                                tickformat='.1%',
                                                showgrid=True,
                                                gridcolor="rgba(255,255,255,0.1)",
                                                title_font=dict(size=12)
                                            ),
                                            # Configuración adicional para evitar superposición
                                            autosize=False,
                                            showlegend=True
                                        )
                                        
                                        st.plotly_chart(fig_ef, use_container_width=True)
                                        
                                        # Tabla con portafolios destacados
                                        st.markdown("#### " + (t("notable_portfolios") if "notable_portfolios" in translations else "Portafolios Destacados"))
                                        
                                        notable_portfolios = pd.DataFrame({
                                            t("portfolio_type") if "portfolio_type" in translations else "Tipo": [
                                                t("current_portfolio") if "current_portfolio" in translations else "Actual",
                                                t("max_sharpe_portfolio") if "max_sharpe_portfolio" in translations else "Máximo Sharpe",
                                                t("min_volatility_portfolio") if "min_volatility_portfolio" in translations else "Mínima Volatilidad"
                                            ],
                                            t("expected_return") if "expected_return" in translations else "Retorno": [
                                                f"{current_return:.2%}",
                                                f"{ef_returns[max_sharpe_idx]:.2%}",
                                                f"{ef_returns[min_vol_idx]:.2%}"
                                            ],
                                            t("volatility") if "volatility" in translations else "Volatilidad": [
                                                f"{current_volatility:.2%}",
                                                f"{ef_volatility[max_sharpe_idx]:.2%}",
                                                f"{ef_volatility[min_vol_idx]:.2%}"
                                            ],
                                            "Sharpe Ratio": [
                                                f"{current_return/current_volatility:.2f}" if current_volatility != 0 else "N/A",
                                                f"{ef_sharpe[max_sharpe_idx]:.2f}",
                                                f"{ef_sharpe[min_vol_idx]:.2f}"
                                            ]
                                        })
                                        
                                        st.dataframe(notable_portfolios, use_container_width=True, hide_index=True)
                                        
                                        # Análisis de eficiencia
                                        st.markdown("#### " + (t("efficiency_analysis") if "efficiency_analysis" in translations else "Análisis de Eficiencia"))
                                        
                                        # Verificar si el portafolio actual está en la frontera eficiente
                                        distance_to_frontier = float('inf')
                                        closest_efficient_idx = 0
                                        
                                        for i, (ef_ret, ef_vol) in enumerate(zip(ef_returns, ef_volatility)):
                                            distance = np.sqrt((current_return - ef_ret)**2 + (current_volatility - ef_vol)**2)
                                            if distance < distance_to_frontier:
                                                distance_to_frontier = distance
                                                closest_efficient_idx = i
                                        
                                        if distance_to_frontier < 0.01:  # Umbral de tolerancia
                                            st.success(t("portfolio_efficient") if "portfolio_efficient" in translations else 
                                                     "✅ Tu portafolio está cerca de la frontera eficiente")
                                        else:
                                            improvement_return = ef_returns[closest_efficient_idx] - current_return
                                            improvement_volatility = current_volatility - ef_volatility[closest_efficient_idx]
                                            
                                            if improvement_return > 0 and improvement_volatility > 0:
                                                st.info(f"📊 {t('improvement_possible') if 'improvement_possible' in translations else 'Mejora posible'}: "
                                                       f"+{improvement_return:.2%} {t('return') if 'return' in translations else 'retorno'}, "
                                                       f"-{improvement_volatility:.2%} {t('volatility') if 'volatility' in translations else 'volatilidad'}")
                                            elif improvement_return > 0:
                                                st.info(f"📈 {t('return_improvement') if 'return_improvement' in translations else 'Mejora en retorno'}: "
                                                       f"+{improvement_return:.2%}")
                                            elif improvement_volatility > 0:
                                                st.info(f"📉 {t('risk_reduction') if 'risk_reduction' in translations else 'Reducción de riesgo'}: "
                                                       f"-{improvement_volatility:.2%}")
                                    
                                    else:
                                        st.error(t("frontier_calculation_error") if "frontier_calculation_error" in translations else 
                                               "No se pudo calcular la frontera eficiente")
                                        
                                except Exception as e:
                                    st.error(f"{t('frontier_error') if 'frontier_error' in translations else 'Error calculando frontera'}: {str(e)}")

                        # Visualización de métricas
                        st.markdown("## 📊 " + t("portfolio_metrics"))
                        
                        metrics_col1, metrics_col2 = st.columns(2)
                        
                        with metrics_col1:
                            col1, col2 = st.columns(2)
                            col1.metric("📈 CAGR", f"{cagr:.2%}")
                            col2.metric("📉 " + t("volatility"), f"{volatility:.2%}")
                            
                            col1, col2 = st.columns(2)
                            col1.metric("⚖️ Sharpe Ratio", f"{sharpe_ratio:.2f}")
                            col2.metric("📉 Max Drawdown", f"{max_drawdown:.2%}")
                        
                        with metrics_col2:
                            col1, col2 = st.columns(2)
                            col1.metric("📊 " + t("avg_monthly_return"), f"{avg_monthly_return:.2%}")
                            col2.metric("✅ " + t("positive_months"), f"{positive_months:.1%}")
                            
                            col1, col2 = st.columns(2)
                            col1.metric("📆 " + t("investment_period"), f"{days} " + t("days"))
                            col2.metric("💰 " + t("total_return"), f"{total_return:.2%}")

                        # Análisis de drawdown
                        with st.expander(t("drawdown_analysis")):
                            # Gráfico de drawdown
                            fig_dd = go.Figure()
                            fig_dd.add_trace(go.Scatter(
                                x=drawdowns.index,
                                y=drawdowns,
                                fill='tozeroy',
                                mode='lines',
                                name=t("drawdown"),
                                line=dict(color='red')
                            ))
                            fig_dd.update_layout(
                                title=t("drawdown_chart"),
                                xaxis_title=t("date"),
                                yaxis_title=t("drawdown_percentage"),
                                template='plotly_dark',
                                height=300
                            )
                            st.plotly_chart(fig_dd, use_container_width=True)
                            
                            # Estadísticas de drawdown
                            drawdown_threshold = -0.05  # 5%
                            significant_drawdowns = drawdowns[drawdowns < drawdown_threshold]
                            num_significant_drawdowns = len(significant_drawdowns.groupby(
                                (significant_drawdowns.shift() >= drawdown_threshold) & 
                                (significant_drawdowns < drawdown_threshold)
                            ).cumcount())
                            
                            st.markdown(f"""
                            - **{t('max_drawdown')}**: {max_drawdown:.2%}
                            - **{t('significant_drawdowns')}**: {num_significant_drawdowns} ({t('greater_than')} 5%)
                            """)

                        # Correlación entre activos
                        with st.expander(t("correlation_analysis")):
                            corr_matrix = returns.corr()
                            fig_corr = go.Figure(data=go.Heatmap(
                                z=corr_matrix.values,
                                x=corr_matrix.columns,
                                y=corr_matrix.index,
                                colorscale='RdBu_r',
                                zmin=-1, zmax=1
                            ))
                            fig_corr.update_layout(
                                title=t("correlation_matrix"),
                                height=400
                            )
                            st.plotly_chart(fig_corr, use_container_width=True)
                            
                            # Análisis de diversificación
                            avg_correlation = (corr_matrix.sum().sum() - len(corr_matrix)) / (len(corr_matrix)**2 - len(corr_matrix))
                            
                            if avg_correlation > 0.7:
                                st.warning(t("high_correlation_warning"))
                            elif avg_correlation < 0.3:
                                st.success(t("good_diversification"))
                            else:
                                st.info(t("moderate_correlation"))

                        # Recomendaciones avanzadas
                        with st.expander(t("recommendations")):
                            recommendations = []
                            
                            # Análisis de concentración
                            max_weight = max(weights)
                            if max_weight > 60:
                                recommendations.append(f"⚠️ **{t('concentration_warning')}** {t('concentration_detail')}")
                            
                            # Análisis de volatilidad
                            if volatility > 0.25:
                                recommendations.append(f"⚠️ **{t('high_volatility')}** {t('volatility_detail')}")
                            
                            # Análisis de Sharpe Ratio
                            if sharpe_ratio < 0.5:
                                recommendations.append(f"⚠️ **{t('low_sharpe')}** {t('sharpe_detail')}")
                            elif sharpe_ratio > 1:
                                recommendations.append(f"✅ **{t('good_sharpe')}** {t('sharpe_good_detail')}")
                            
                            # Análisis de correlación
                            if avg_correlation > 0.7 and len(tickers) > 1:
                                recommendations.append(f"⚠️ **{t('diversification_warning')}** {t('diversification_detail')}")
                            
                            # Análisis de drawdown
                            if max_drawdown < -0.3:
                                recommendations.append(f"⚠️ **{t('high_drawdown_warning')}** {t('drawdown_detail')}")
                            
                            if recommendations:
                                for rec in recommendations:
                                    st.markdown(rec)
                            else:
                                st.success(t("portfolio_looks_good"))

                        # Exportar
                        with st.expander(t("download")):
                            excel_df = data.copy()
                            excel_df.index.name = t("date")
                            
                            # Botón para CSV
                            st.download_button(
                                label=t("download_csv"),
                                data=excel_df.to_csv().encode("utf-8"),
                                file_name="portfolio_data.csv",
                                mime="text/csv",
                                key=f"download_csv_{st.session_state.cache_bust}"
                            )
                            
                            # Información adicional para exportar
                            metrics_df = pd.DataFrame({
                                t("metric"): ["CAGR", t("volatility"), "Sharpe Ratio", "Max Drawdown", 
                                            t("avg_monthly_return"), t("positive_months"), t("total_return")],
                                t("value"): [f"{cagr:.2%}", f"{volatility:.2%}", f"{sharpe_ratio:.2f}", 
                                           f"{max_drawdown:.2%}", f"{avg_monthly_return:.2%}", 
                                           f"{positive_months:.1%}", f"{total_return:.2%}"]
                            })
                            
                            # Descargar métricas
                            st.download_button(
                                label=t("download_metrics"),
                                data=metrics_df.to_csv(index=False).encode("utf-8"),
                                file_name="portfolio_metrics.csv",
                                mime="text/csv",
                                key=f"download_metrics_{st.session_state.cache_bust}"
                            )

                        # Guardar config
                        with st.expander(t("save_config")):
                            if st.button(t("save_button"), key=f"save_config_{st.session_state.cache_bust}"):
                                config = {
                                    "tickers": tickers,
                                    "weights": weights,
                                    "start_date": str(start_date),
                                    "end_date": str(end_date),
                                    "saved_date": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                                }
                                try:
                                    with open("portfolio_config.json", "w") as f:
                                        json.dump(config, f, indent=4)
                                    st.success(t("config_saved"))
                                except Exception as e:
                                    st.error(f"{t('error_saving_config')}: {e}")
                except Exception as e:
                    st.error(f"{t('data_download_error')}: {e}")
                    st.info(t("try_other_tickers"))

with tab2:
    st.header(t("performance"))
    
    st.info(t("benchmark_tab_info"))
    
    # Mejorado: Selección de índice y comparación
    reference_indices = {
        "S&P 500": "^GSPC",
        "Nasdaq": "^IXIC", 
        "Dow Jones": "^DJI",
        "IBEX 35": "^IBEX",
        "Euro Stoxx 50": "^STOXX50E"
    }
    
    selected_index = st.selectbox(
        t("reference_index"), 
        list(reference_indices.keys()),
        key=f"reference_index_{st.session_state.app_id}"
    )
    
    # Verificar si hay tickers seleccionados
    portfolio_tickers = [t for t in st.session_state.tickers if t.strip()]
    portfolio_weights = st.session_state.weights[:len(portfolio_tickers)]
    
    if portfolio_tickers:
        if st.button(t("compare_performance"), key=f"compare_performance_{st.session_state.app_id}"):
            with st.spinner(t("downloading_data")):
                try:
                    # Combinar tickers del portafolio con el índice
                    all_tickers = portfolio_tickers + [reference_indices[selected_index]]
                    comparison_data = download_data(all_tickers, start=start_date, end=end_date)
                    
                    if comparison_data.empty:
                        st.error(t("no_data_warning"))
                    else:
                        # Normalizar datos para comparación
                        normalized_data = comparison_data / comparison_data.iloc[0] * 100
                        
                        # Calcular rendimiento del portafolio
                        returns = comparison_data[portfolio_tickers].pct_change().dropna()
                        weights_array = np.array(portfolio_weights) / sum(portfolio_weights)
                        portfolio_returns = returns.dot(weights_array)
                        
                        # Crear serie de retorno acumulado del portafolio
                        portfolio_cumulative = (1 + portfolio_returns).cumprod() * 100
                        
                        # Gráfico de comparación
                        fig_comp = go.Figure()
                        
                        # Línea del portafolio
                        fig_comp.add_trace(go.Scatter(
                            x=portfolio_cumulative.index,
                            y=portfolio_cumulative,
                            mode='lines',
                            name=t("portfolio"),
                            line=dict(color='green', width=2.5)
                        ))
                        
                        # Línea del índice
                        index_ticker = reference_indices[selected_index]
                        fig_comp.add_trace(go.Scatter(
                            x=normalized_data.index,
                            y=normalized_data[index_ticker],
                            mode='lines',
                            name=selected_index,
                            line=dict(color='gray', width=2, dash='dot')
                        ))
                        
                        fig_comp.update_layout(
                            title=f"{t('portfolio_vs')} {selected_index}",
                            xaxis_title=t("date"),
                            yaxis_title=t("growth_base_100"),
                            template="plotly_dark",
                            height=450,
                            hovermode="x unified",
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        st.plotly_chart(fig_comp, use_container_width=True)
                        
                        # Comparar métricas
                        st.markdown("### " + t("performance_comparison"))
                        
                        # Calcular métricas del portafolio
                        port_return = portfolio_cumulative.iloc[-1] / 100 - 1
                        days = (end_date - start_date).days
                        years = days / 365.25
                        port_cagr = (1 + port_return) ** (1 / years) - 1 if years > 0 else 0
                        port_vol = portfolio_returns.std() * np.sqrt(252)
                        port_sharpe = port_cagr / port_vol if port_vol != 0 else 0
                        
                        # Calcular métricas del índice
                        index_returns = comparison_data[index_ticker].pct_change().dropna()
                        index_cumulative = (1 + index_returns).cumprod().iloc[-1] - 1
                        index_cagr = (1 + index_cumulative) ** (1 / years) - 1 if years > 0 else 0
                        index_vol = index_returns.std() * np.sqrt(252)
                        index_sharpe = index_cagr / index_vol if index_vol != 0 else 0
                        
                        # Mostrar comparación de métricas
                        metrics_df = pd.DataFrame({
                            t("metric"): [t("total_return"), "CAGR", t("volatility"), "Sharpe Ratio"],
                            t("portfolio"): [f"{port_return:.2%}", f"{port_cagr:.2%}", 
                                            f"{port_vol:.2%}", f"{port_sharpe:.2f}"],
                            selected_index: [f"{index_cumulative:.2%}", f"{index_cagr:.2%}", 
                                            f"{index_vol:.2%}", f"{index_sharpe:.2f}"]
                        })
                        
                        st.dataframe(
                            metrics_df,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Análisis adicional
                        if port_cagr > index_cagr and port_sharpe > index_sharpe:
                            st.success(t("portfolio_outperforming"))
                        elif port_cagr < index_cagr and port_sharpe < index_sharpe:
                            st.warning(t("portfolio_underperforming"))
                        elif port_cagr > index_cagr and port_sharpe < index_sharpe:
                            st.info(t("portfolio_higher_return_higher_risk"))
                        elif port_cagr < index_cagr and port_sharpe > index_sharpe:
                            st.info(t("portfolio_lower_return_lower_risk"))
                except Exception as e:
                    st.error(f"{t('comparison_error')}: {e}")
    else:
        st.warning(t("need_portfolio_first"))

with tab3:
    st.header(t("settings"))
    st.caption(t("dark_mode_note"))
    
    st.subheader(t("troubleshooting"))
    
    with st.expander(t("connection_error")):
        st.markdown(f"""
        ### {t("common_problems")}
        
        #### 1. {t("verify_ticker_syntax")}
        {t("ticker_syntax_explanation")}:
        - **{t("us_stocks")}**: `AAPL`, `MSFT`, `GOOGL`
        - **{t("spanish_stocks")}**: {t("spanish_stocks_explanation")} `.MC`, {t("example")} `SAN.MC`, `BBVA.MC`
        - **{t("indices")}**: {t("indices_explanation")} `^`, {t("example")} `^GSPC` (S&P 500), `^IBEX` (IBEX 35)
        - **{t("us_etfs")}**: `SPY`, `QQQ`, `VGT`
        
        #### 2. {t("update_yfinance")}
        {t("run_terminal_command")}:
        ```
        pip install yfinance --upgrade
        ```
        
        #### 3. {t("network_problems")}
        - {t("corporate_proxy")}
        - {t("check_internet_firewall")}
        
        #### 4. {t("change_date_range")}
        - {t("limited_historical_data")}
        - {t("try_recent_dates")}
        
        #### 5. {t("api_limits")}
        - {t("too_many_queries")}
        - {t("wait_before_retry")}
        """)
    
    st.subheader(t("about_yahoo_finance"))
    st.markdown(f"""
    {t("app_uses_yahoo")}
    
    **{t("important_notes")}:**
    - {t("us_stocks_note")}
    - {t("european_stocks_note")}:
      - {t("spain_bme")}: ticker.MC ({t("example")}: SAN.MC, BBVA.MC)
      - {t("germany_xetra")}: ticker.DE
      - {t("uk_lse")}: ticker.L
    - {t("etfs_note")}: {t("use_symbols")} SPY, VTI, etc.
    - {t("indices_note")}: {t("use_symbols")} ^GSPC (S&P 500), ^IXIC (Nasdaq), etc.
    
    [Yahoo Finance {t("documentation")}](https://pypi.org/project/yfinance/)
    """)
    
    # Verificar instalación
    with st.expander(t("verify_installation")):
        st.code("""
# """ + t("run_terminal_commands") + """:
pip install --upgrade yfinance pandas numpy plotly streamlit

# """ + t("verify_yfinance_version") + """:
pip show yfinance
        """)

# Cargar configuración guardada si existe
if os.path.exists("portfolio_config.json"):
    with st.sidebar.expander(t("load_saved_config")):
        if st.button(t("load_last_config"), key=f"load_config_{st.session_state.app_id}"):
            try:
                with open("portfolio_config.json", "r") as f:
                    saved_config = json.load(f)
                    
                # Actualizar la sesión con los datos guardados
                st.session_state.tickers = saved_config["tickers"]
                st.session_state.weights = saved_config["weights"]
                st.session_state.num_assets = len(saved_config["tickers"])
                
                # Actualizar fecha si está disponible
                if "start_date" in saved_config:
                    try:
                        st.session_state[f"start_date_{st.session_state.app_id}"] = datetime.strptime(saved_config["start_date"], "%Y-%m-%d").date()
                    except:
                        pass
                
                if "end_date" in saved_config:
                    try:
                        st.session_state[f"end_date_{st.session_state.app_id}"] = datetime.strptime(saved_config["end_date"], "%Y-%m-%d").date()
                    except:
                        pass
                
                # Regenerar UUID para evitar conflictos de caché
                st.session_state.cache_bust += 1
                
                st.sidebar.success(t("config_loaded"))
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"{t('config_load_error')}: {e}")

















