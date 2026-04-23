import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

# ---------------------------
# ⚙️ CONFIG
# ---------------------------
st.set_page_config(
    page_title="Risk Dashboard",
    page_icon="📉",
    layout="wide"
)

# ---------------------------
# 📥 DATA
# ---------------------------
@st.cache_data
def get_data():
    ticker = "BTC-USD"
    data = yf.download(ticker, start="2015-01-01")

    if 'Adj Close' in data.columns:
        data['Returns'] = data['Adj Close'].pct_change()
    else:
        data['Returns'] = data['Close'].pct_change()

    return data.dropna()

data = get_data()
returns = data['Returns']

# ---------------------------
# 📊 HEADER
# ---------------------------
st.title("📉 Análisis de Riesgo Financiero")
st.write("Estimación de VaR y Expected Shortfall")

# ---------------------------
# 📉 SERIES
# ---------------------------
st.header("Rendimientos")
st.line_chart(returns)

# ---------------------------
# 📈 STATS
# ---------------------------
st.header("Estadísticas")

col1, col2, col3 = st.columns(3)

col1.metric("Media", f"{returns.mean():.6f}")
col2.metric("Sesgo", f"{returns.skew():.4f}")
col3.metric("Curtosis", f"{returns.kurt():.4f}")

# ---------------------------
# 📐 FUNCIONES VaR
# ---------------------------
def var_es_normal(r, alpha):
    mu = r.mean()
    sigma = r.std()

    z = norm.ppf(1 - alpha)

    VaR = mu + z * sigma
    ES = mu - sigma * (norm.pdf(z) / (1 - alpha))

    return VaR, ES


def var_es_t(r, alpha):
    # Convertir a numpy limpio
    r = pd.Series(r).astype(float)

    # Limpiar basura
    r = r.replace([np.inf, -np.inf], np.nan).dropna()

    # ⚠️ Si hay muy pocos datos, evitar crash
    if len(r) < 10:
        return np.nan, np.nan

    # Ajuste t
    df, loc, scale = student_t.fit(r.values)

    x = student_t.ppf(1 - alpha, df)

    VaR = loc + scale * x

    ES = loc + scale * (
        (student_t.pdf(x, df) / (1 - alpha)) * (df + x**2) / (df - 1)
    )

    return VaR, ES

def var_es_hist(r, alpha):
    sorted_r = r.sort_values()
    idx = int((1 - alpha) * len(sorted_r))

    VaR = sorted_r.iloc[idx]
    ES = sorted_r.iloc[:idx].mean()

    return VaR, ES


def var_es_mc(r, alpha, n_sim=10000):
    mu = r.mean()
    sigma = r.std()

    sim = np.random.normal(mu, sigma, n_sim)

    VaR = np.percentile(sim, (1 - alpha) * 100)
    ES = sim[sim <= VaR].mean()

    return VaR, ES


# ---------------------------
# 🔁 ROLLING VaR
# ---------------------------
rolling = pd.DataFrame(index=returns.index)

rolling['Returns'] = returns
rolling[['VaR_95_hist', 'ES_95_hist',
         'VaR_99_hist', 'ES_99_hist',
         'VaR_95_norm', 'ES_95_norm',
         'VaR_99_norm', 'ES_99_norm']] = np.nan

for i in range(252, len(returns)):
    window = returns.iloc[i-252:i]

    # HIST
    sorted_r = window.sort_values()

    idx_95 = int(0.05 * len(sorted_r))
    idx_99 = int(0.01 * len(sorted_r))

    rolling.iloc[i, rolling.columns.get_loc('VaR_95_hist')] = sorted_r.iloc[idx_95]
    rolling.iloc[i, rolling.columns.get_loc('ES_95_hist')] = sorted_r.iloc[:idx_95].mean()

    rolling.iloc[i, rolling.columns.get_loc('VaR_99_hist')] = sorted_r.iloc[idx_99]
    rolling.iloc[i, rolling.columns.get_loc('ES_99_hist')] = sorted_r.iloc[:idx_99].mean()

    # NORMAL
    mu = window.mean()
    sigma = window.std()

    z95 = norm.ppf(0.05)
    z99 = norm.ppf(0.01)

    rolling.iloc[i, rolling.columns.get_loc('VaR_95_norm')] = mu + sigma * z95
    rolling.iloc[i, rolling.columns.get_loc('ES_95_norm')] = mu - sigma * (norm.pdf(z95) / 0.05)

    rolling.iloc[i, rolling.columns.get_loc('VaR_99_norm')] = mu + sigma * z99
    rolling.iloc[i, rolling.columns.get_loc('ES_99_norm')] = mu - sigma * (norm.pdf(z99) / 0.01)

plot_data = rolling.dropna()

# ---------------------------
# 📉 GRÁFICA
# ---------------------------
st.subheader("Returns vs VaR")

st.line_chart(plot_data[['Returns', 'VaR_95_hist']])

# ---------------------------
# 🎯 SELECTOR
# ---------------------------
st.header("VaR y ES")

metodo = st.selectbox("Método", ["Normal", "t-Student", "Histórico", "Monte Carlo"])
alpha = st.selectbox("Confianza", [0.95, 0.975, 0.99])

if metodo == "Normal":
    VaR, ES = var_es_normal(returns, alpha)
elif metodo == "t-Student":
    VaR, ES = var_es_t(returns, alpha)
elif metodo == "Histórico":
    VaR, ES = var_es_hist(returns, alpha)
else:
    VaR, ES = var_es_mc(returns, alpha)

col1, col2 = st.columns(2)
col1.metric("VaR", f"{VaR:.5f}")
col2.metric("ES", f"{ES:.5f}")

# ---------------------------
# 📋 TABLA
# ---------------------------
st.subheader("Comparación")

res = []

for a in [0.95, 0.975, 0.99]:
    res.append({
        "Alpha": a,
        "Normal": var_es_normal(returns, a)[0],
        "t": var_es_t(returns, a)[0],
        "Hist": var_es_hist(returns, a)[0],
        "MC": var_es_mc(returns, a)[0],
    })

df = pd.DataFrame(res)
st.dataframe(df)

# ---------------------------
# 📊 BARRAS
# ---------------------------
st.bar_chart(df.set_index("Alpha"))
