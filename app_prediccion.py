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

"""
Predicción de incendios forestales con preprocesamiento interactivo

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
import matplotlib.pyplot as plt # Aunque no se usa en el script, se mantiene por si es necesario en el futuro


# Personalización de la interfaz 

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
        font-size:38px; /* Aumentado */
        font-weight:800;
        color: #ff7a45; /* Acento fuego */
        margin-bottom: 6px;
    }
    .subtitle {
        color: #d1d5db;
        margin-top: -6px;
        margin-bottom: 14px;
        font-size: 16px; /* Ajustado */
    }

    /* Botón principal (Predecir) */
    .stButton>button {
        background-color: #ff6b35;
        color: white;
        border-radius: 10px;
        font-weight: 700;
        padding: 10px 20px; /* Un poco más grande */
        transition: all 0.2s; /* Animación */
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff4a1a;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }

    /* Sidebar - Estilos específicos */
    .sidebar .stCheckbox > label {
        color: #f8fafc; /* Color del texto para checkboxes */
        font-weight: 500;
    }
    .sidebar .stSlider label {
        font-weight: 500;
    }
    .css-1d391kg {  /* Estilo para la barra lateral en algunas versiones */
        background: rgba(0,0,0,0.12);
        border-radius: 8px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #2c3e50; /* Base oscura */
        color: #fefae0;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        border: 1px solid #4a657c;
        transition: all 0.2s;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #e63946; /* Rojo/Fuego para activa */
        color: white;
        font-weight: bold;
        border: 1px solid #e63946;
    }

    /* Footer oculto por defecto de Streamlit */
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="🔥 Detección de Incendios Forestales 🌲",
    page_icon="🔥",
    layout="wide",
)


# Configuración general

DEFAULT_MODEL_PATH = "forest_fire_model_final.keras"
IMG_SIZE = (160, 160)
CLASS_NAMES = ["Fuego", "Sin Fuego"]


# Funciones de procesamiento de imagen

def to_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)

def apply_clahe_bgr(bgr, clip=2.0, tiles=8):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # Convertir a int para evitar errores con cv2.createCLAHE
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
    # Ajusta la intensidad de los canales R, G y B de forma independiente
    b, g, r = cv2.split(bgr.astype(np.float32))
    r = np.clip(r * r_factor, 0, 255)
    g = np.clip(g * g_factor, 0, 255)
    b = np.clip(b * b_factor, 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)

def preprocess_for_model(rgb_u8):
    rgb = cv2.resize(rgb_u8, IMG_SIZE)
    x = rgb.astype(np.float32) / 255.0
    return x


# Cargar modelo

@st.cache_resource(show_spinner="Cargando modelo de detección de incendios...")
def load_default_model():
    p = Path(DEFAULT_MODEL_PATH).resolve()
    if not p.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {p}. Asegúrate de que '{DEFAULT_MODEL_PATH}' esté en el directorio.")
    try:
        m = tf.keras.models.load_model(p.as_posix())
        return m, str(p), False
    except ValueError as e:
        if "Lambda" in str(e):
            # Carga el modelo con safe_mode=False si hay problemas con capas custom o Lambdas
            m = tf.keras.models.load_model(p.as_posix(), safe_mode=False)
            return m, str(p), True
        raise


# Interfaz principal

st.markdown('<div class="big-title">🔥 Detector de Incendios Forestales 🌲</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Sube una imagen, prueba preprocesados interactivos y evalúa la predicción del modelo.</div>', unsafe_allow_html=True)


try:
    model, model_path_used, used_unsafe = load_default_model()
    st.success(f"Modelo cargado correctamente desde: {os.path.basename(model_path_used)}")
except Exception as e:
    st.error(f"❌ Error al cargar el modelo. {str(e)}")
    st.stop()

uploaded = st.file_uploader("🖼️ Elegí una imagen (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if bgr is None:
        st.error("No se pudo leer la imagen. Asegúrate de que es un formato de imagen válido.")
        st.stop()

    # Variables para almacenar los parámetros de los sliders (inicialización)
    sat_factor = 1.2
    bilateral_d = 7
    bilateral_sigmaColor = 50
    bilateral_sigmaSpace = 50
    median_ksize = 3
    retinex_sigma = 80.0
    clahe_clip = 2.0
    clahe_tiles = 8
    r_factor = 1.0
    g_factor = 1.0
    b_factor = 1.0
    denoise_mode = "bilateral" # Inicialización para evitar errores

    # --- Barra Lateral (Controles)
    with st.sidebar:
        st.title("⚙️ Ajustes")
        st.markdown("**Selecciona los filtros que deseas aplicar:**")
        
        # 1. Checkboxes
        do_gray = st.checkbox("Escala de Grises")
        do_wb = st.checkbox("Corrección de Color (Balance de Blancos)")
        do_sat = st.checkbox("Saturación")
        do_rgb = st.checkbox("Ajuste de Color (RGB)")
        do_clahe = st.checkbox("Contraste (CLAHE)")
        do_dehaze = st.checkbox("Eliminación de Niebla (Retinex)")
        do_denoise = st.checkbox("Reducción de Ruido")
        
        st.markdown("---")
        st.markdown("### Parámetros Fijos")
        
        # 2. Parámetros (Usando Expander)
        
        # Saturación
        if do_sat:
            with st.expander("Saturación", expanded=True):
                sat_factor = st.slider("Factor de Saturación", 0.5, 2.5, 1.2, 0.1)

        # RGB
        if do_rgb:
            with st.expander("Ajustes de Intensidad RGB", expanded=True):
                r_factor = st.slider("Rojo (R)", 0.0, 2.5, 1.0, 0.1)
                g_factor = st.slider("Verde (G)", 0.0, 2.5, 1.0, 0.1)
                b_factor = st.slider("Azul (B)", 0.0, 2.5, 1.0, 0.1)
        
        # CLAHE
        if do_clahe:
            with st.expander("Contraste (CLAHE)", expanded=True):
                clahe_clip = st.slider("Límite de Contraste (ClipLimit)", 0.5, 5.0, 2.0, 0.1)
                clahe_tiles = st.slider("Tamaño de la Malla (Tiles)", 4, 16, 8, 1)

        # Dehazing (Retinex)
        if do_dehaze:
            with st.expander("Ajustes de Niebla (Retinex)", expanded=True):
                retinex_sigma = st.slider("Desempañamiento ($\sigma$)", 10.0, 120.0, 80.0, 1.0)
        
        # Denoise
        if do_denoise:
            with st.expander("Ajustes de Reducción de Ruido", expanded=True):
                denoise_mode = st.selectbox("Modo de Ruido", ["Bilateral", "Mediana"])
                
                if denoise_mode == "Bilateral":
                    bilateral_d = st.slider("Diámetro (d)", 3, 15, 7, 2)
                    bilateral_sigmaColor = st.slider("Suavizado de Color ($\sigma_c$)", 10, 100, 50, 5)
                    bilateral_sigmaSpace = st.slider("Suavizado Espacial ($\sigma_s$)", 10, 100, 50, 5)
                else: # Mediana
                    # st.slider requiere valores impares para kernel size
                    median_ksize = st.slider("Tamaño del Kernel", 3, 11, 3, 2)

    # --- Pestañas (Tabs) ---
    tab1, tab2 = st.tabs(["🖼️ Procesamiento Interactivo", "📊 Predicción del Modelo"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Imagen Cargada")
            st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        # Aplicar procesamientos
        bgr_proc = bgr.copy()
        applied = []
        
        # Orden de aplicación: Grises -> Color/Balance -> Mejora -> Ruido
        if do_gray: 
            bgr_proc = to_grayscale(bgr_proc)
            applied.append("Grises")
        if do_wb: 
            bgr_proc = white_balance_gray_world(bgr_proc)
            applied.append("WB")
        if do_rgb:
            # Importante: los factores deben estar definidos (ya lo están arriba)
            bgr_proc = adjust_rgb_channels(bgr_proc, r_factor, g_factor, b_factor)
            applied.append(f"RGB(R×{r_factor},G×{g_factor},B×{b_factor})")
        if do_sat: 
            bgr_proc = increase_saturation(bgr_proc, factor=sat_factor)
            applied.append(f"Sat×{sat_factor}")
        if do_dehaze: 
            bgr_proc = retinex_dehaze(bgr_proc, retinex_sigma)
            applied.append("Retinex")
        if do_clahe: 
            bgr_proc = apply_clahe_bgr(bgr_proc, clahe_clip, clahe_tiles)
            applied.append("CLAHE")
        if do_denoise:
            if denoise_mode == "Bilateral":
                bgr_proc = denoise_bilateral(bgr_proc, bilateral_d, bilateral_sigmaColor, bilateral_sigmaSpace)
                applied.append(f"Denoise(Bilat)")
            else:
                bgr_proc = denoise_median(bgr_proc, median_ksize)
                applied.append(f"Denoise(Med)")


        with col2:
            st.subheader("Vista Previa")
            caption = " → ".join(applied) if applied else "Ningún filtro aplicado"
            st.image(cv2.cvtColor(bgr_proc, cv2.COLOR_BGR2RGB), caption=caption,use_container_width=True)

    with tab2:
        st.header("Análisis del modelo")
        
        # Preparar la imagen para el modelo
        rgb_for_model = cv2.cvtColor(bgr_proc, cv2.COLOR_BGR2RGB)
        x = np.expand_dims(preprocess_for_model(rgb_for_model), axis=0)

        if st.button("🔎 Ejecutar Predicción", use_container_width=True):
            
            # Placeholder para mostrar el resultado rápidamente mientras se calcula
            with st.spinner('Analizando imagen...'):
                probs = model.predict(x, verbose=0)[0]
                top_index = int(np.argmax(probs))
                
            # Muestra el resultado de forma impactante
            if CLASS_NAMES[top_index] == "Fuego":
                st.warning(f"🔥🔥 ¡ALERTA DE FUEGO DETECTADO! 🔥🔥")
                st.markdown(f"**Probabilidad de Fuego:** **<span style='color:#e63946; font-size:24px;'>{probs[top_index]:.2%}</span>**", unsafe_allow_html=True)
            else:
                st.success("✅ Sin Fuego Detectado")
                st.markdown(f"**Probabilidad Sin Fuego:** **<span style='color:#1dd1a1; font-size:24px;'>{probs[top_index]:.2%}</span>**", unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("Distribución de Probabilidades")

            # Gráfico de barras
            df_probs = pd.DataFrame({
                "Clase": CLASS_NAMES,
                "Probabilidad": [float(p) for p in probs]
            })
            
            # Definir colores para el gráfico
            color_scale = alt.Scale(domain=CLASS_NAMES, range=["#e63946", "#1dd1a1"])
            
            chart = alt.Chart(df_probs).mark_bar(cornerRadiusTopLeft=10, cornerRadiusTopRight=10).encode(
                x=alt.X("Probabilidad:Q", axis=alt.Axis(format='%', title='Probabilidad')),
                y=alt.Y("Clase:N", sort='-x', title='Clase'),
                color=alt.Color("Clase:N", legend=None, scale=color_scale),
                tooltip=["Clase", alt.Tooltip("Probabilidad", format='.2%')]
            ).properties(height=200)
            
            # Estilo del gráfico
            chart = chart.configure_axis(
                grid=False,
                labelColor='#d1d5db',
                titleColor='#fefae0'
            ).configure_view(
                strokeWidth=0
            )

            st.altair_chart(chart, use_container_width=True)

else:
    st.info("⬆️ Sube una imagen para comenzar el análisis y activar los controles de preprocesamiento.")

# =====================================================
st.markdown("""
---
<div style='text-align: center; color: #9ca3af;'>
    🔥 Proyecto de detección de incendios forestales — desarrollado por <b>Grupo 2</b><br>
    <small>Streamlit • TensorFlow • OpenCV</small>
</div>
""", unsafe_allow_html=True)


