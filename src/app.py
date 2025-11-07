# Hacer una aplicación de Streamlit para predecir el precio de Binance Coin (BNB) con el precio de Ethereum (ETH) usando el modelo entrenado
import streamlit as st
import pandas as pd
import numpy as np
from pickle import load
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Cargar el modelo entrenado y el scaler
from pickle import load
model = load(open('../models/best_model.pkl', 'rb'))
scaler = load(open('../models/scaler.pkl', 'rb'))

# Hacer un sidebar para que el usuario ingrese el precio de Ethereum (ETH)
st.sidebar.header('Input Ethereum (ETH) Price')
eth_price = st.sidebar.number_input('Ethereum (ETH) Price in USD', min_value=0.0, value=2000.0, step=1.0)

# Escalar el precio de Ethereum (ETH)
eth_price_scaled = scaler.transform(eth_price.reshape(-1, 1))

# Hacer la predicción
bnb_price_pred_scaled = model.predict(eth_price_scaled)

# Desescalar la predicción
bnb_price_pred = scaler.inverse_transform(bnb_price_pred_scaled.reshape(-1, 1))
st.header('Predicted Binance Coin (BNB) Price')