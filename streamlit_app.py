import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from scipy.stats import t

# Configuración de la página
st.set_page_config(
    page_title="Risk Dashboard",
    page_icon="📉",
    layout="wide"
)
@st.cache_data
def get_data():
    ticker = "BTC-USD"
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
#INCISO B
media = data['Returns'].mean()
sesgo = data['Returns'].skew()
curtosis = data['Returns'].kurt()

col1, col2, col3 = st.columns(3)

col1.metric("Media", f"{media:.6f}")
col2.metric("Sesgo", f"{sesgo:.4f}")
col3.metric("Curtosis", f"{curtosis:.4f}")
#INCISO C
st.header("VaR y Expected Shortfall", divider= 'red')
alpha = [0.95, 0.975, 0.99]

def var_es_normal(returns, alpha):
    mean = returns.mean()
    std = returns.std()

    z = norm.ppf(1 - alpha)

    VaR = mean + z * std
    ES = mean - std * (norm.pdf(z) / (1 - alpha))

    return VaR, ES

def var_es_t(returns, alpha):
    gl, med, disp = t.fit(returns)

    x = t.ppf(1 - alpha, gl)

    VaR = med + disp * x

    ES = med + disp * (
        (t.pdf(x, gl) / (1 - alpha)) * (gl + x**2) / (gl - 1)
    )

    return VaR, ES
def var_es_hist(returns, alpha):
    sorted_returns = returns.sort_values()

    index = int((1 - alpha) * len(sorted_returns))

    VaR = sorted_returns.iloc[index] #El inicio de las peores perdidas
    ES = sorted_returns.iloc[:index].mean()

    return VaR, ES
import numpy as np

def var_es_mc(returns, alpha, n_sim=10000):
    media = returns.mean()
    dev = returns.std()

   
    sim = np.random.normal(media, dev, n_sim)

    # VaR
    VaR = np.percentile(sim, (1 - alpha) * 100)

    # ES
    ES = sim[sim <= VaR].mean() #Seleccionamos solo los que son menores a VaR

    return VaR, ES

returns = data['Returns']

# DataFrame donde guardarás resultados
rolling_results = pd.DataFrame(index=returns.index) #Necesitamos indice de tiempo

# Inicializar columnas
rolling_results['Returns'] = returns
rolling_results['VaR_95_hist'] = np.nan
rolling_results['ES_95_hist'] = np.nan
rolling_results['VaR_99_hist'] = np.nan
rolling_results['ES_99_hist'] = np.nan
rolling_results['VaR_95_norm'] = np.nan
rolling_results['ES_95_norm'] = np.nan
rolling_results['VaR_99_norm'] = np.nan
rolling_results['ES_99_norm'] = np.nan


for t in range(252, len(returns)):

    window_data = returns.iloc[t-252:t]

    # HISTÓRICO
    sorted_r = window_data.sort_values()

    # 95%
    idx_95 = int(0.05 * len(sorted_r))
    VaR_95_h = sorted_r.iloc[idx_95]
    ES_95_h = sorted_r.iloc[:idx_95].mean()

    # 99%
    idx_99 = int(0.01 * len(sorted_r))
    VaR_99_h = sorted_r.iloc[idx_99]
    ES_99_h = sorted_r.iloc[:idx_99].mean()

    # NORMAL
    mu = window_data.mean()
    sigma = window_data.std()

    z_95 = norm.ppf(0.05)
    z_99 = norm.ppf(0.01)

    VaR_95_n = mu + sigma * z_95
    VaR_99_n = mu + sigma * z_99

    ES_95_n = mu - sigma * (norm.pdf(z_95) / 0.05)
    ES_99_n = mu - sigma * (norm.pdf(z_99) / 0.01)

    rolling_results.iloc[t, rolling_results.columns.get_loc('VaR_95_hist')] = VaR_95_h
    rolling_results.iloc[t, rolling_results.columns.get_loc('ES_95_hist')] = ES_95_h
    rolling_results.iloc[t, rolling_results.columns.get_loc('VaR_99_hist')] = VaR_99_h
    rolling_results.iloc[t, rolling_results.columns.get_loc('ES_99_hist')] = ES_99_h

    rolling_results.iloc[t, rolling_results.columns.get_loc('VaR_95_norm')] = VaR_95_n
    rolling_results.iloc[t, rolling_results.columns.get_loc('ES_95_norm')] = ES_95_n
    rolling_results.iloc[t, rolling_results.columns.get_loc('VaR_99_norm')] = VaR_99_n
    rolling_results.iloc[t, rolling_results.columns.get_loc('ES_99_norm')] = ES_99_n
    violations = rolling_results['Returns'] < rolling_results['VaR_95_hist']

try:
    import plotly.graph_objects as go
    use_plotly = True
except:
    use_plotly = False
fig = go.Figure()

# Returns
fig.add_trace(go.Scatter(
    x=rolling_results.index,
    y=rolling_results['Returns'],
    name='Returns'
))

# VaR
fig.add_trace(go.Scatter(
    x=rolling_results.index,
    y=rolling_results['VaR_95_hist'],
    name='VaR 95%',
    line=dict(dash='dash')
))

st.plotly_chart(fig)

st.line_chart(
    rolling_results[['Returns', 'VaR_95_hist']].dropna()
)


st.header("📊 VaR y Expected Shortfall", divider='red')

# ---------------------------
# 🎛️ SELECTORES
# ---------------------------
metodos = ["Normal", "t-Student", "Histórico", "Monte Carlo"]
metodo_sel = st.selectbox("Selecciona método", metodos)

alpha_sel = st.selectbox("Selecciona nivel de confianza", [0.95, 0.975, 0.99])

# ---------------------------
# 🧮 CALCULAR SEGÚN MÉTODO
# ---------------------------
if metodo_sel == "Normal":
    VaR, ES = var_es_normal(returns, alpha_sel)

elif metodo_sel == "t-Student":
    VaR, ES = var_es_t(returns, alpha_sel)

elif metodo_sel == "Histórico":
    VaR, ES = var_es_hist(returns, alpha_sel)

elif metodo_sel == "Monte Carlo":
    VaR, ES = var_es_mc(returns, alpha_sel)

# ---------------------------
# 📊 MOSTRAR RESULTADOS
# ---------------------------
col1, col2 = st.columns(2)

col1.metric("VaR", f"{VaR:.5f}")
col2.metric("ES", f"{ES:.5f}")

# ---------------------------
# 📋 TABLA COMPARATIVA (TODOS)
# ---------------------------
st.subheader("📋 Comparación de métodos")

resultados = []

for a in [0.95, 0.975, 0.99]:
    var_n, es_n = var_es_normal(returns, a)
    var_t, es_t = var_es_t(returns, a)
    var_h, es_h = var_es_hist(returns, a)
    var_mc, es_mc = var_es_mc(returns, a)

    resultados.append({
        "Alpha": a,
        "VaR Normal": var_n,
        "ES Normal": es_n,
        "VaR t": var_t,
        "ES t": es_t,
        "VaR Hist": var_h,
        "ES Hist": es_h,
        "VaR MC": var_mc,
        "ES MC": es_mc,
    })

df_results = pd.DataFrame(resultados)

st.dataframe(df_results)

# ---------------------------
# 📊 GRÁFICA COMPARATIVA
# ---------------------------
st.subheader("📊 Comparación VaR por método")

fig = go.Figure()

for col in ["VaR Normal", "VaR t", "VaR Hist", "VaR MC"]:
    fig.add_trace(go.Bar(
        x=df_results["Alpha"],
        y=df_results[col],
        name=col
    ))

st.plotly_chart(fig)

# ---------------------------
# 📉 ROLLING WINDOW VISUAL
# ---------------------------
st.subheader("📉 Rolling VaR (252 días)")

plot_data = rolling_results.dropna()

fig2 = go.Figure()

# Returns
fig2.add_trace(go.Scatter(
    x=plot_data.index,
    y=plot_data['Returns'],
    name='Returns'
))

# Selector dinámico
tipo_rolling = st.selectbox(
    "Selecciona tipo de VaR rolling",
    ["Histórico", "Normal"]
)

if tipo_rolling == "Histórico":
    fig2.add_trace(go.Scatter(
        x=plot_data.index,
        y=plot_data['VaR_95_hist'],
        name='VaR 95 Hist',
        line=dict(color='red', dash='dash')
    ))

    fig2.add_trace(go.Scatter(
        x=plot_data.index,
        y=plot_data['ES_95_hist'],
        name='ES 95 Hist',
        line=dict(color='orange', dash='dot')
    ))

else:
    fig2.add_trace(go.Scatter(
        x=plot_data.index,
        y=plot_data['VaR_95_norm'],
        name='VaR 95 Normal',
        line=dict(color='blue', dash='dash')
    ))

    fig2.add_trace(go.Scatter(
        x=plot_data.index,
        y=plot_data['ES_95_norm'],
        name='ES 95 Normal',
        line=dict(color='green', dash='dot')
    ))

# 🔥 Violaciones
violations = plot_data['Returns'] < plot_data['VaR_95_hist']

fig2.add_trace(go.Scatter(
    x=plot_data.index[violations],
    y=plot_data['Returns'][violations],
    mode='markers',
    name='Violaciones',
    marker=dict(color='black', size=6)
))

st.plotly_chart(fig2)
