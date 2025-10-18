
"""
Predicción de incendios forestales con preprocesamiento interactivo
- El usuario carga una foto
- Puede aplicar: Blanco y Negro, Balance de blancos (Gray World),
  Aumentar saturación, Denoise (bilateral/mediana), Dehazing (Retinex), CLAHE
- Luego predice con el modelo .keras

Requisitos:
  pip install streamlit tensorflow==2.12.* opencv-python numpy
"""
# app_prediccion.py
import os
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

# ===================== Config =====================
DEFAULT_MODEL_PATH = "forest_fire_model_final.keras"  # mismo directorio que este script
IMG_SIZE = (160, 160)
CLASS_NAMES = ["Fuego", "Sin Fuego"]  # ajustá el orden si tu modelo fue entrenado distinto

# ================== Utils imagen ==================
def to_uint8(img):
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

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
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        rr = single_scale_retinex(img[:, :, c], sigma=sigma)
        rmin, rmax = np.percentile(rr, (1, 99))
        rr = np.clip((rr - rmin) / (rmax - rmin + 1e-6), 0.0, 1.0)
        out[:, :, c] = rr
    return to_uint8(out * 255.0)

def denoise_bilateral(bgr, d=7, sigmaColor=50, sigmaSpace=50):
    return cv2.bilateralFilter(bgr, d=int(d), sigmaColor=float(sigmaColor), sigmaSpace=float(sigmaSpace))

def denoise_median(bgr, ksize=3):
    k = int(ksize)
    if k % 2 == 0:
        k += 1
    k = max(3, k)
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

def preprocess_for_model(rgb_u8):
    rgb = cv2.resize(rgb_u8, IMG_SIZE, interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32) / 255.0
    return x  # (H, W, 3) en [0,1]

# =============== Carga de modelo (cache) ===========
@st.cache_resource(show_spinner=False)
def load_default_model():
    """
    Carga el modelo por defecto. Si contiene Lambda (modelos viejos),
    reintenta con safe_mode=False.
    """
    p = Path(DEFAULT_MODEL_PATH).resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo por defecto en: {p}\n"
            f"Colocá 'forest_fire_model_final.keras' en la misma carpeta de esta app."
        )
    try:
        m = tf.keras.models.load_model(p.as_posix())  # modelos nuevos (sin Lambda)
        return m, str(p), False
    except ValueError as e:
        if "deserialization of a `Lambda` layer" in str(e):
            m = tf.keras.models.load_model(p.as_posix(), safe_mode=False)
            return m, str(p), True
        raise

# ========================= UI ======================
st.set_page_config(page_title="Incendios - Predicción con Preprocesado", page_icon="🔥", layout="wide")
st.title("🔥 Clasificador de Incendios — Preprocesamiento y predicción")

# Cargar modelo automáticamente
try:
    model, model_path_used, used_unsafe = load_default_model()
    if used_unsafe:
        st.warning(f"Modelo cargado con `safe_mode=False` (contiene Lambda):\n{model_path_used}")
    else:
        st.success(f"Modelo cargado: {model_path_used}")
except Exception as e:
    st.error(str(e))
    st.stop()

# Panel imagen
st.header("Imagen")
uploaded = st.file_uploader("Elegí una imagen (JPG/PNG)", type=["jpg", "jpeg", "png"])

# Panel de procesado (lateral)
with st.sidebar:
    st.header("Procesamientos de imagen")
    do_gray    = st.checkbox("Blanco y Negro", value=False)
    do_wb      = st.checkbox("Balance de blancos (Gray World)", value=False)
    do_sat     = st.checkbox("Más saturación", value=False)
    do_denoise = st.checkbox("Denoise", value=False)
    do_dehaze  = st.checkbox("Dehazing (Retinex)", value=False)
    do_clahe   = st.checkbox("CLAHE", value=False)

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

if uploaded:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("No se pudo leer la imagen.")
        st.stop()

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Original")
        st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    # Aplicar procesamientos
    bgr_proc = bgr.copy()
    if do_gray:
        bgr_proc = to_grayscale(bgr_proc)
    if do_wb:
        bgr_proc = white_balance_gray_world(bgr_proc)
    if do_sat:
        bgr_proc = increase_saturation(bgr_proc, factor=sat_factor)
    if do_denoise:
        if denoise_mode == "bilateral":
            bgr_proc = denoise_bilateral(bgr_proc, d=bilateral_d,
                                         sigmaColor=bilateral_sigmaColor, sigmaSpace=bilateral_sigmaSpace)
        else:
            bgr_proc = denoise_median(bgr_proc, ksize=median_ksize)
    if do_dehaze:
        bgr_proc = retinex_dehaze(bgr_proc, sigma=retinex_sigma)
    if do_clahe:
        bgr_proc = apply_clahe_bgr(bgr_proc, clip=clahe_clip, tiles=clahe_tiles)

    with col2:
        st.subheader("Procesada (previa a la predicción)")
        st.image(cv2.cvtColor(bgr_proc, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    st.markdown("---")
    # Preparar para el modelo
    rgb_for_model = cv2.cvtColor(bgr_proc, cv2.COLOR_BGR2RGB)
    x = preprocess_for_model(rgb_for_model)  # [0,1]
    x = np.expand_dims(x, axis=0)

    if st.button("🔎 Predecir"):
        probs = model.predict(x, verbose=0)[0]
        top = int(np.argmax(probs))
        st.success(f"Predicción: **{CLASS_NAMES[top]}** ({probs[top]:.2%})")
        st.write({CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)})
else:
    st.info("Subí una imagen para comenzar.")
