import base64
import io
import cv2
import numpy as np
from PIL import Image

# =========================
# CONFIGURACIÓN GENERAL
# =========================
IMG_SIZE = (160, 160)
CLASS_NAMES = ["Fuego", "Sin Fuego"]

# =========================
# FUNCIONES AUXILIARES
# =========================

def to_uint8(img):
    """Asegura que los valores estén en rango [0,255] y convierte a uint8."""
    return np.clip(img, 0, 255).astype(np.uint8)

# -------------------------
# FILTROS / PROCESAMIENTOS
# -------------------------

def apply_clahe_bgr(bgr, clip=2.0, tiles=8):
    """Aplica CLAHE (mejora local de contraste) en el canal L de LAB."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tiles), int(tiles)))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def white_balance_gray_world(bgr, eps=1e-6):
    """Corrige balance de blancos según el método Gray World."""
    img = bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0) + eps
    gray = means.mean()
    gain = gray / means
    out = img * gain
    return to_uint8(out)

def single_scale_retinex(channel, sigma=80):
    """Aplica Retinex a un canal individual."""
    blurred = cv2.GaussianBlur(channel, (0, 0), sigma)
    return np.log(channel + 1e-6) - np.log(blurred + 1e-6)

def retinex_dehaze(bgr, sigma=80.0):
    """Reduce la neblina (dehazing) usando Single Scale Retinex."""
    img = bgr.astype(np.float32) / 255.0
    out = np.zeros_like(img)
    for c in range(3):
        rr = single_scale_retinex(img[:, :, c], sigma=sigma)
        rmin, rmax = np.percentile(rr, (1, 99))
        rr = np.clip((rr - rmin) / (rmax - rmin + 1e-6), 0.0, 1.0)
        out[:, :, c] = rr
    return to_uint8(out * 255.0)

def denoise_bilateral(bgr, d=7, sigmaColor=50, sigmaSpace=50):
    """Reducción de ruido mediante filtro bilateral."""
    return cv2.bilateralFilter(bgr, int(d), float(sigmaColor), float(sigmaSpace))

def denoise_median(bgr, ksize=3):
    """Reducción de ruido mediante filtro de mediana."""
    k = int(ksize)
    if k % 2 == 0:
        k += 1
    return cv2.medianBlur(bgr, k)

def increase_saturation(bgr, factor=1.2):
    """Aumenta la saturación en espacio HSV."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * float(factor), 0, 255)
    hsv2 = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

def to_grayscale(bgr):
    """Convierte a escala de grises manteniendo 3 canales."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

def adjust_rgb_channels(bgr, r_factor=1.0, g_factor=1.0, b_factor=1.0):
    """Ajusta cada canal RGB por un factor."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    r = np.clip(r * r_factor, 0, 255)
    g = np.clip(g * g_factor, 0, 255)
    b = np.clip(b * b_factor, 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)

# -------------------------
# FUNCIONES DE PREDICCIÓN
# -------------------------

def preprocess_for_model(rgb_u8):
    """Prepara la imagen RGB para el modelo (redimensiona y normaliza)."""
    rgb = cv2.resize(rgb_u8, IMG_SIZE)
    x = rgb.astype(np.float32) / 255.0
    return x

def bgr_to_data_uri(bgr):
    """Convierte imagen BGR a formato base64 (para mostrar en HTML)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"
