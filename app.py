import streamlit as st
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
import joblib
import xgboost as xgb
from processor import segmentar_letras, extraer_features
import os
from string import ascii_letters, digits

# ───────────────────────────────────────────────
# 📦 Cargar modelos y encoder
modelo_letras = load_model("modelo_letras.h5")

modelo_xgb_reg = xgb.XGBRegressor()
modelo_xgb_reg.load_model("xgb_model_reg.json")

modelo_xgb_clf = xgb.XGBClassifier()
modelo_xgb_clf.load_model("xgb_model_clf.json")

encoder = joblib.load("encoder.pkl")

# ───────────────────────────────────────────────

st.title("🔍 Clasificador de captchas")

uploaded_file = st.file_uploader("📤 Sube una imagen de captcha", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img_pil = Image.open(uploaded_file)
    st.image(img_pil, caption="Captcha subido", use_container_width=True)

    # Guardar temporalmente para leer con OpenCV
    ruta = "temp_captcha.png"
    img_pil.save(ruta)
    img = cv2.imread(ruta)

    # Verificar si tiene color
    if len(img.shape) == 3 and img.shape[2] == 3:
        b, g, r = cv2.split(img)
        if (b == g).all() and (b == r).all():
            img_bn = b
            color = "no"
        else:
            img_bn = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            color = "sí"
    else:
        img_bn = img
        color = "no"

    tiene_color = color == "sí"
    st.write(f"📌 ¿Tiene color?: **{color}**")

    # ───────────────────────────────────────────────
    # ✂️ Segmentar letras con imagen original
    texto_real = uploaded_file.name.split('.')[0]
    letras = segmentar_letras(img_bn, texto_real)

    if not letras:
        st.error("❌ No se pudieron segmentar letras.")
        st.stop()

    texto_predicho = ""
    for letra, _ in letras:
        if letra is None or letra.size == 0:
            st.warning("❗ Letra vacía o mal segmentada.")
            continue
        letra_redim = cv2.resize(letra, (32, 50)).astype(np.float32)
        letra_redim = np.expand_dims(letra_redim, axis=(0, -1))
        pred = modelo_letras.predict(letra_redim, verbose=0)
        idx = np.argmax(pred)
        letra_pred = encoder.inverse_transform([idx])[0]
        texto_predicho += letra_pred

    st.write(f"✏️ Texto detectado: `{texto_predicho}`")

    # ───────────────────────────────────────────────

    features = extraer_features(texto_predicho, tiene_color)
    features = features.drop("color")
    # 🧠 Predecir probabilidad
    probabilidad = modelo_xgb_reg.predict([features])[0]
    st.write(f"📈 Probabilidad de acierto estimada: `{probabilidad:.2%}`")

    # 🧠 Clasificar con los mismos features
    clase = modelo_xgb_clf.predict([features])[0]
    resultado = "Fácil ✅" if clase == 1 else "Difícil ❌"
    st.success(f"🧠 Clasificación del captcha: **{resultado}**")