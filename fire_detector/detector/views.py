import os
import cv2
import numpy as np
from pathlib import Path
from django.shortcuts import render
from django.conf import settings
import tensorflow as tf

from .forms import PredictForm
from .utils import (
    preprocess_for_model, bgr_to_data_uri,
    to_grayscale, white_balance_gray_world, increase_saturation,
    adjust_rgb_channels, retinex_dehaze, apply_clahe_bgr,
    denoise_bilateral, denoise_median
)

# =========================
# CONFIG GENERAL
# =========================
BASE_DIR = Path(settings.BASE_DIR)
MODEL_PATH = BASE_DIR / "forest_fire_model_final.keras"
CLASS_NAMES = ["Fuego", "Sin Fuego"]
IMG_SIZE = (160, 160)

# =========================
# CARGA ÚNICA DEL MODELO
# =========================
def _load_model():
    """Carga el modelo Keras (.keras) una sola vez."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")
    try:
        model = tf.keras.models.load_model(MODEL_PATH.as_posix(), compile=False)
    except Exception:
        # fallback si hay capas Lambda u otras incompatibilidades
        model = tf.keras.models.load_model(MODEL_PATH.as_posix(), compile=False, safe_mode=False)
    return model

# Cache del modelo en memoria
MODEL = _load_model()

# =========================
# VISTA PRINCIPAL
# =========================
def index(request):
    """Vista principal: formulario, preprocesamiento y predicción."""
    print("✅ Renderizando index.html")

    # Mensaje inicial si no hay imagen cargada
    context = {
        "form": PredictForm(),
        "message": "Subí una imagen para analizar si hay fuego 🔥"
    }

    if request.method == "POST":
        form = PredictForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # --- Leer imagen subida ---
                file_bytes = np.frombuffer(request.FILES["image"].read(), np.uint8)
                bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if bgr is None:
                    context["error"] = "No se pudo leer la imagen subida."
                    return render(request, "detector/index.html", context)

                bgr_proc = bgr.copy()
                applied = []

                # --- Flags de filtros ---
                do_gray = form.cleaned_data.get("do_gray")
                do_wb = form.cleaned_data.get("do_wb")
                do_sat = form.cleaned_data.get("do_sat")
                do_rgb = form.cleaned_data.get("do_rgb")
                do_clahe = form.cleaned_data.get("do_clahe")
                do_dehaze = form.cleaned_data.get("do_dehaze")
                do_denoise = form.cleaned_data.get("do_denoise")

                # --- Parámetros ---
                sat_factor = form.cleaned_data.get("sat_factor") or 1.2
                r_factor = form.cleaned_data.get("r_factor") or 1.0
                g_factor = form.cleaned_data.get("g_factor") or 1.0
                b_factor = form.cleaned_data.get("b_factor") or 1.0
                clahe_clip = form.cleaned_data.get("clahe_clip") or 2.0
                clahe_tiles = form.cleaned_data.get("clahe_tiles") or 8
                retinex_sigma = form.cleaned_data.get("retinex_sigma") or 80.0
                denoise_mode = form.cleaned_data.get("denoise_mode") or "bilateral"
                bilateral_d = form.cleaned_data.get("bilateral_d") or 7
                bilateral_sigmaColor = form.cleaned_data.get("bilateral_sigmaColor") or 50
                bilateral_sigmaSpace = form.cleaned_data.get("bilateral_sigmaSpace") or 50
                median_ksize = form.cleaned_data.get("median_ksize") or 3

                # --- Aplicar filtros ---
                if do_gray:
                    bgr_proc = to_grayscale(bgr_proc); applied.append("Grises")
                if do_wb:
                    bgr_proc = white_balance_gray_world(bgr_proc); applied.append("WB")
                if do_rgb:
                    bgr_proc = adjust_rgb_channels(bgr_proc, r_factor, g_factor, b_factor)
                    applied.append(f"RGB({r_factor:.1f},{g_factor:.1f},{b_factor:.1f})")
                if do_sat:
                    bgr_proc = increase_saturation(bgr_proc, factor=sat_factor)
                    applied.append(f"Saturación x{sat_factor}")
                if do_dehaze:
                    bgr_proc = retinex_dehaze(bgr_proc, retinex_sigma); applied.append("Retinex")
                if do_clahe:
                    bgr_proc = apply_clahe_bgr(bgr_proc, clahe_clip, clahe_tiles); applied.append("CLAHE")
                if do_denoise:
                    if denoise_mode == "bilateral":
                        bgr_proc = denoise_bilateral(bgr_proc, bilateral_d, bilateral_sigmaColor, bilateral_sigmaSpace)
                        applied.append("Denoise Bilateral")
                    else:
                        bgr_proc = denoise_median(bgr_proc, median_ksize)
                        applied.append("Denoise Mediana")

                # =========================
                # PREDICCIÓN ORIGINAL
                # =========================
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                x0 = np.expand_dims(preprocess_for_model(rgb), axis=0)
                probs0 = MODEL.predict(x0, verbose=0)[0]
                idx0 = int(np.argmax(probs0))
                pred0 = {"label": CLASS_NAMES[idx0], "prob": float(probs0[idx0])}

                # =========================
                # PREDICCIÓN PROCESADA
                # =========================
                rgbp = cv2.cvtColor(bgr_proc, cv2.COLOR_BGR2RGB)
                xp = np.expand_dims(preprocess_for_model(rgbp), axis=0)
                probsp = MODEL.predict(xp, verbose=0)[0]
                idxp = int(np.argmax(probsp))
                predp = {"label": CLASS_NAMES[idxp], "prob": float(probsp[idxp])}

                # --- Pasar datos al template ---
                context.update({
                    "form": form,
                    "original_data_uri": bgr_to_data_uri(bgr),
                    "processed_data_uri": bgr_to_data_uri(bgr_proc),
                    "applied": " → ".join(applied) if applied else "Ningún filtro aplicado",
                    "pred_original": pred0,
                    "pred_processed": predp,
                    "message": None,
                })

            except Exception as e:
                context["error"] = f"Error al procesar la imagen: {e}"

        else:
            context["error"] = "Formulario inválido."

    return render(request, "detector/index.html", context)
