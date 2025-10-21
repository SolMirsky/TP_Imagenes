"""
Predicción de incendios forestales con preprocesamiento interactivo
- El usuario carga una foto
- Puede aplicar: Blanco y Negro, Balance de blancos (Gray World),
  Aumentar saturación, Denoise (bilateral/mediana), Dehazing (Retinex), CLAHE, RGB
- Luego predice con el modelo .keras

Requisitos:
  pip install streamlit tensorflow==2.12.* opencv-python numpy altair
"""

import os
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt

# Personalización de la interfaz
st.markdown("""
    <style>
    /* Fondo general */
    .stApp {
        background: linear-gradient(135deg, #1b4332, #2d6a4f, #40916c);
        color: white;
        font-family: 'Poppins', sans-serif;
    }

    /* Título principal */
    h1 {
        text-align: center;
        color: #ffb703;
        text-shadow: 1px 1px 3px #000000;
        margin-bottom: 20px;
    }

    /* Subtítulos */
    h2, h3 {
        color: #fefae0;
    }

    /* Botón de carga */
    section[data-testid="stFileUploader"] > div {
        background-color: #081c15 !important;
        border-radius: 10px;
        padding: 10px;
    }

    /* Texto del botón */
    section[data-testid="stFileUploader"] label div span {
        color: #fefae0 !important;
        font-weight: 600;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #081c15;
        color: #fefae0;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #e63946;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="🔥 Detección de Incendios Forestales 🌲",
    page_icon="🔥",
    layout="wide",
)

st.markdown("""
    <style>
    /* Fondo temático: degradado oscuro -> naranja suave */
    .stApp {
        background: linear-gradient(180deg, #0f1724 0%, #1f2937 45%, #3b2b1d 100%);
        color: #f8fafc;
    }

    /* Contenedor principal */
    .main > div {
        background: rgba(255,255,255,0.02);
        border-radius: 12px;
    }

    /* Títulos */
    .big-title {
        font-size:34px;
        font-weight:800;
        color: #ff7a45; /* acento fuego */
        margin-bottom: 6px;
    }
    .subtitle {
        color: #d1d5db;
        margin-top: -6px;
        margin-bottom: 14px;
    }

    /* Botón principal */
    .stButton>button {
        background-color: #ff6b35;
        color: white;
        border-radius: 10px;
        font-weight: 700;
        padding: 8px 14px;
    }
    .stButton>button:hover {
        background-color: #ff4a1a;
        transform: translateY(-1px);
    }

    /* Sidebar */
    .css-1d391kg {  /* clase de sidebar en algunas versiones; si no aplica, no rompe */
        background: rgba(0,0,0,0.12);
        border-radius: 8px;
    }

    /* Footer oculto por defecto de Streamlit */
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# =====================================================
# Config general
# =====================================================
DEFAULT_MODEL_PATH = "forest_fire_model_final.keras"
IMG_SIZE = (160, 160)
CLASS_NAMES = ["Fuego", "Sin Fuego"]

# =====================================================
# Funciones de procesamiento de imagen
# =====================================================
def to_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)

def apply_clahe_bgr(bgr, clip=2.0, tiles=8):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tiles), int(tiles)))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def white_balance_gray_world(bgr, eps=1e-6):
    img = bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0) + eps
    gray = means.mean()
    gain = gray / means
    out = img * gain
    return to_uint8(out)

def single_scale_retinex(channel, sigma=80):
    blurred = cv2.GaussianBlur(channel, (0, 0), sigma)
    return np.log(channel + 1e-6) - np.log(blurred + 1e-6)

def retinex_dehaze(bgr, sigma=80.0):
    img = bgr.astype(np.float32) / 255.0
    out = np.zeros_like(img)
    for c in range(3):
        rr = single_scale_retinex(img[:, :, c], sigma=sigma)
        rmin, rmax = np.percentile(rr, (1, 99))
        rr = np.clip((rr - rmin) / (rmax - rmin + 1e-6), 0.0, 1.0)
        out[:, :, c] = rr
    return to_uint8(out * 255.0)

def denoise_bilateral(bgr, d=7, sigmaColor=50, sigmaSpace=50):
    return cv2.bilateralFilter(bgr, int(d), float(sigmaColor), float(sigmaSpace))

def denoise_median(bgr, ksize=3):
    k = int(ksize)
    if k % 2 == 0:
        k += 1
    return cv2.medianBlur(bgr, k)

def increase_saturation(bgr, factor=1.2):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * float(factor), 0, 255)
    hsv2 = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

def to_grayscale(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

def adjust_rgb_channels(bgr, r_factor=1.0, g_factor=1.0, b_factor=1.0):
    #Ajusta la intensidad de los canales R, G y B de forma independiente
    b, g, r = cv2.split(bgr.astype(np.float32))
    r = np.clip(r * r_factor, 0, 255)
    g = np.clip(g * g_factor, 0, 255)
    b = np.clip(b * b_factor, 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)

def preprocess_for_model(rgb_u8):
    rgb = cv2.resize(rgb_u8, IMG_SIZE)
    x = rgb.astype(np.float32) / 255.0
    return x

# =====================================================
# Cargar modelo
# =====================================================
@st.cache_resource(show_spinner=False)
def load_default_model():
    p = Path(DEFAULT_MODEL_PATH).resolve()
    if not p.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {p}")
    try:
        m = tf.keras.models.load_model(p.as_posix())
        return m, str(p), False
    except ValueError as e:
        if "Lambda" in str(e):
            m = tf.keras.models.load_model(p.as_posix(), safe_mode=False)
            return m, str(p), True
        raise

# =====================================================
# Interfaz
# =====================================================
st.markdown('<div class="big-title">🔥 Detector de Incendios Forestales</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Subí una imagen, probá preprocesados interactivos y evaluá la predicción del modelo.</div>', unsafe_allow_html=True)


try:
    model, model_path_used, used_unsafe = load_default_model()
    st.success(f"Modelo cargado correctamente desde: {model_path_used}")
except Exception as e:
    st.error(str(e))
    st.stop()

uploaded = st.file_uploader(" Elegí una imagen (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if bgr is None:
        st.error("No se pudo leer la imagen.")
        st.stop()

    # Tabs
    tab1, tab2 = st.tabs(["Procesamiento", "Predicción"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original")
            st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        with st.sidebar:
            #st.image("", width=80) para agregar icono referencial
            st.title("Ajustes de Procesamiento")
            st.markdown("---")

            do_gray = st.checkbox("Blanco y Negro")
            do_wb = st.checkbox("Balance de blancos (Gray World)")
            do_sat = st.checkbox("Más saturación")
            do_denoise = st.checkbox("Denoise")
            do_dehaze = st.checkbox("Dehazing (Retinex)")
            do_clahe = st.checkbox("CLAHE")
            do_rgb = st.checkbox("Realce RGB (canales de color)", value=False)

            st.subheader("Parámetros")
            sat_factor = st.slider("Factor de saturación", 0.5, 2.5, 1.2, 0.1)
            denoise_mode = st.selectbox("Modo denoise", ["bilateral", "median"])
            bilateral_d = st.slider("bilateral d", 3, 15, 7, 2)
            bilateral_sigmaColor = st.slider("bilateral sigmaColor", 10, 100, 50, 5)
            bilateral_sigmaSpace = st.slider("bilateral sigmaSpace", 10, 100, 50, 5)
            median_ksize = st.slider("median ksize", 3, 11, 3, 2)
            retinex_sigma = st.slider("Retinex sigma", 10.0, 120.0, 80.0, 1.0)
            clahe_clip = st.slider("CLAHE clipLimit", 0.5, 5.0, 2.0, 0.1)
            clahe_tiles = st.slider("CLAHE tiles", 4, 16, 8, 1)
            r_factor = st.slider("Intensidad canal Rojo (R)", 0.0, 2.5, 1.0, 0.1)
            g_factor = st.slider("Intensidad canal Verde (G)", 0.0, 2.5, 1.0, 0.1)
            b_factor = st.slider("Intensidad canal Azul (B)", 0.0, 2.5, 1.0, 0.1)
          

        # Aplicar procesamientos
        bgr_proc = bgr.copy()
        if do_gray: bgr_proc = to_grayscale(bgr_proc)
        if do_wb: bgr_proc = white_balance_gray_world(bgr_proc)
        if do_rgb:bgr_proc = adjust_rgb_channels(bgr_proc, r_gain=r_gain, g_gain=g_gain, b_gain=b_gain)
        if do_sat: bgr_proc = increase_saturation(bgr_proc, factor=sat_factor)
        if do_denoise:
            if denoise_mode == "bilateral":
                bgr_proc = denoise_bilateral(bgr_proc, bilateral_d, bilateral_sigmaColor, bilateral_sigmaSpace)
            else:
                bgr_proc = denoise_median(bgr_proc, median_ksize)
        if do_dehaze: bgr_proc = retinex_dehaze(bgr_proc, retinex_sigma)
        if do_clahe: bgr_proc = apply_clahe_bgr(bgr_proc, clahe_clip, clahe_tiles)
        if do_rgb:bgr_proc = adjust_rgb_channels(bgr_proc, r_factor, g_factor, b_factor)

        with col2:
            st.subheader("Vista Previa")
            applied = []
            if do_gray: applied.append("Gray")
            if do_wb: applied.append("WB")
            if do_rgb: applied.append(f"RGB(R{r_gain},G{g_gain},B{b_gain})")
            if do_sat: applied.append(f"Sat×{sat_factor}")
            if do_denoise: applied.append(f"Denoise:{denoise_mode}")
            if do_dehaze: applied.append("Retinex")
            if do_clahe: applied.append("CLAHE")
            caption = " → ".join(applied) if applied else "Ningún preprocesado aplicado"
            st.image(cv2.cvtColor(bgr_proc, cv2.COLOR_BGR2RGB), caption=caption,use_container_width=True)

    with tab2:
        st.header("Predicción del modelo")
        rgb_for_model = cv2.cvtColor(bgr_proc, cv2.COLOR_BGR2RGB)
        x = np.expand_dims(preprocess_for_model(rgb_for_model), axis=0)

        if st.button("🔎 Predecir"):
            probs = model.predict(x, verbose=0)[0]
            top = int(np.argmax(probs))
            st.success(f"Predicción: **{CLASS_NAMES[top]}** ({probs[top]:.2%})")

            # Gráfico de barras
            df_probs = pd.DataFrame({
                "Clase": CLASS_NAMES,
                "Probabilidad": [float(p) for p in probs]
            })
            chart = alt.Chart(df_probs).mark_bar(cornerRadiusTopLeft=10, cornerRadiusTopRight=10).encode(
                x=alt.X("Probabilidad:Q", axis=alt.Axis(format='%')),
                y=alt.Y("Clase:N", sort='-x'),
                color=alt.Color("Clase:N", legend=None, scale=alt.Scale(range=["#ff6b6b", "#1dd1a1"]))
            ).properties(height=200)
            st.altair_chart(chart, use_container_width=True)

else:
    st.info("⬆ Subí una imagen para comenzar")

# =====================================================

st.markdown("""
---
<div style='text-align: center; color: gray;'>
    🔥 Proyecto de detección de incendios forestales — desarrollado por <b>Grupo 2</b><br>
    <small>Streamlit • TensorFlow • OpenCV • >
</div>
""", unsafe_allow_html=True)


