import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title='VaR Dashboard',
    page_icon='📉',
)

@st.cache_data
def get_data():
    ticker = "AAPL"
    data = yf.download(ticker, start="2010-01-01")

    # Ver qué columnas hay
    if 'Adj Close' in data.columns:
        data['Returns'] = data['Adj Close'].pct_change()
    else:
        data['Returns'] = data['Close'].pct_change()

    data = data.dropna()

    return data

data = get_data()

# Página principal

st.title("📉 Análisis de Riesgo Financiero")
st.write("Estimación de Value at Risk (VaR) y Expected Shortfall (ES)")

# Mostrar datos
st.header("Datos históricos", divider="gray")
st.write(data.head())

# Rendimientos

st.header("Rendimientos diarios", divider="gray")

st.line_chart(data['Returns'])



st.header("Estadísticas", divider="gray")

media = data['Returns'].mean()
sesgo = data['Returns'].skew()
curtosis = data['Returns'].kurt()

col1, col2, col3 = st.columns(3)

col1.metric("Media", f"{media:.6f}")
col2.metric("Sesgo", f"{sesgo:.4f}")
col3.metric("Curtosis", f"{curtosis:.4f}")
