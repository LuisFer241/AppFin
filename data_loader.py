import yfinance as yf
import pandas as pd
import time
import requests
from datetime import datetime

def download_data(tickers, start="2018-01-01", end=None):
    """
    Descarga datos históricos de precio para los tickers especificados desde Yahoo Finance.
    
    Args:
        tickers: Lista de tickers o un solo ticker como string
        start: Fecha de inicio en formato 'YYYY-MM-DD' o un objeto datetime
        end: Fecha de fin en formato 'YYYY-MM-DD' o un objeto datetime (opcional, por defecto es la fecha actual)
        
    Returns:
        DataFrame con los precios de cierre ajustados para todos los tickers
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    
    # Filtrar tickers vacíos
    tickers = [ticker.strip() for ticker in tickers if ticker.strip()]
    
    if not tickers:
        print("❌ No se proporcionaron tickers válidos.")
        return pd.DataFrame()
    
    # Convertir fechas a string si son objetos datetime
    if isinstance(start, datetime):
        start = start.strftime('%Y-%m-%d')
    if isinstance(end, datetime):
        end = end.strftime('%Y-%m-%d')
    
    print(f"🔍 Buscando datos para: {', '.join(tickers)}")
    print(f"📅 Rango de fechas: {start} hasta {end or 'hoy'}")
    
    all_data = pd.DataFrame()
    
    # Intentar descargar datos para cada ticker individualmente
    for ticker in tickers:
        try:
            # Verificar conectividad básica
            try:
                requests.get("https://query1.finance.yahoo.com", timeout=5)
            except requests.exceptions.RequestException:
                print("❌ No se puede conectar a Yahoo Finance. Verifica tu conexión a Internet.")
                continue
            
            # Intentar primero obtener información básica del ticker para verificar si existe
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            if not info or 'regularMarketPrice' not in info:
                print(f"❌ No se encontró información para el ticker: {ticker}. Verifica el símbolo.")
                continue

            # Descargar datos históricos
            print(f"⏳ Descargando datos para {ticker}...")
            
            # Especificar auto_adjust=False para mantener la columna 'Adj Close'
            ticker_data = ticker_obj.history(
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False
            )
            
            if ticker_data.empty:
                print(f"❌ No hay datos históricos disponibles para {ticker} en el rango de fechas especificado.")
                continue
            
            # Usamos 'Adj Close' ya que especificamos auto_adjust=False
            if 'Adj Close' in ticker_data.columns:
                close_data = ticker_data['Adj Close'].to_frame()
            else:
                # Fallback: si por alguna razón no está 'Adj Close', usar 'Close'
                close_data = ticker_data['Close'].to_frame()
                
            close_data.columns = [ticker]
            
            # Añadir al dataframe combinado
            if all_data.empty:
                all_data = close_data
            else:
                all_data = pd.concat([all_data, close_data], axis=1)
                
            print(f"✅ Datos obtenidos para {ticker}: {len(close_data)} días")
            # Esperar un poco entre peticiones para no sobrecargar la API
            time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ Error al procesar {ticker}: {str(e)}")
    
    # Verificar si se obtuvo algún dato
    if all_data.empty:
        print("⚠️ No se pudo obtener ningún dato histórico. Revisa los símbolos y tu conexión.")
        return pd.DataFrame()
    
    # SOLUCIÓN AL ERROR: Eliminar la zona horaria del índice antes de filtrar
    all_data.index = pd.to_datetime(all_data.index).tz_localize(None)
    
    # Filtrar por fecha usando fechas sin zona horaria
    if start:
        start_date = pd.to_datetime(start)
        all_data = all_data[all_data.index >= start_date]
    if end:
        end_date = pd.to_datetime(end)
        all_data = all_data[all_data.index <= end_date]
    
    print(f"📊 Resumen de datos descargados:")
    print(f"- Tickers: {all_data.columns.tolist()}")
    if not all_data.empty:
        print(f"- Periodo: {all_data.index.min().strftime('%Y-%m-%d')} a {all_data.index.max().strftime('%Y-%m-%d')}")
        print(f"- Total días: {len(all_data)}")
    else:
        print("- No hay datos disponibles en el rango especificado")
    
    return all_data







