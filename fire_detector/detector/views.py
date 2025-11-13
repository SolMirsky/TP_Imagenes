import base64
import cv2
import numpy as np
import tensorflow as tf
from django.shortcuts import render
from django.http import HttpResponse
from django.template.loader import render_to_string
from django.views.decorators.csrf import csrf_exempt
from .forms import FireForm


# =========================
# 🔹 Página principal
# =========================
def fire_detector_view(request):
    form = FireForm()
    return render(request, "detector/index.html", {"form": form})


# =========================
# 🔹 Codificar imagen en base64
# =========================
def encode_image(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


# =========================
# 🔹 Aplicar filtros
# =========================
def apply_filters(image, form):
    img = image.copy()

    # Escala de grises
    if form.cleaned_data.get("do_gray"):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Balance de blancos
    if form.cleaned_data.get("do_wb"):
        avg_b = np.mean(img[:, :, 0])
        avg_g = np.mean(img[:, :, 1])
        avg_r = np.mean(img[:, :, 2])
        img[:, :, 0] = np.clip(img[:, :, 0] * (avg_g / avg_b), 0, 255)
        img[:, :, 2] = np.clip(img[:, :, 2] * (avg_g / avg_r), 0, 255)

    # Saturación
    if form.cleaned_data.get("do_sat"):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s * form.cleaned_data.get("sat_factor", 1.2), 0, 255)
        hsv = cv2.merge([h, s, v])
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Ajuste RGB
    if form.cleaned_data.get("do_rgb"):
        img[:, :, 2] = np.clip(img[:, :, 2] * form.cleaned_data.get("r_factor"), 0, 255)
        img[:, :, 1] = np.clip(img[:, :, 1] * form.cleaned_data.get("g_factor"), 0, 255)
        img[:, :, 0] = np.clip(img[:, :, 0] * form.cleaned_data.get("b_factor"), 0, 255)

    # CLAHE
    if form.cleaned_data.get("do_clahe"):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=form.cleaned_data.get("clahe_clip"),
            tileGridSize=(form.cleaned_data.get("clahe_tiles"), form.cleaned_data.get("clahe_tiles"))
        )
        l2 = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)

    # Dehaze (Retinex)
    if form.cleaned_data.get("do_dehaze"):
        sigma = form.cleaned_data.get("retinex_sigma")
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        img = cv2.addWeighted(img, 4, blur, -4, 128)

    # Reducción de ruido
    if form.cleaned_data.get("do_denoise"):
        mode = form.cleaned_data.get("denoise_mode")

        if mode == "bilateral":
            img = cv2.bilateralFilter(
                img,
                d=form.cleaned_data.get("bilateral_d"),
                sigmaColor=form.cleaned_data.get("bilateral_sigmaColor"),
                sigmaSpace=form.cleaned_data.get("bilateral_sigmaSpace")
            )
        else:
            img = cv2.medianBlur(img, form.cleaned_data.get("median_ksize"))

    return img


# =========================
# 🔹 HTMX — procesar imagen y mostrar preview
# =========================
def process_image_ajax(request):
    form = FireForm(request.POST or None, request.FILES or None)
    image_data = None

    # Imagen subida
    if form.is_valid() and "image" in request.FILES:
        image_file = request.FILES["image"].read()
        request.session["image_cache"] = base64.b64encode(image_file).decode("utf-8")
        np_arr = np.frombuffer(image_file, np.uint8)
        image_data = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Imagen que ya estaba en sesión
    elif "image_cache" in request.session:
        cached = base64.b64decode(request.session["image_cache"])
        np_arr = np.frombuffer(cached, np.uint8)
        image_data = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image_data is None:
        return HttpResponse("<p>⚠️ No se pudo cargar la imagen.</p>")

    # Aplicar filtros
    processed = apply_filters(image_data, form)

    # Guardar imágenes en sesión con nombres correctos
    request.session["orig_image"] = encode_image(image_data)
    request.session["proc_image"] = encode_image(processed)

    # Render preview
    html = render_to_string("detector/_preview.html", {
        "original_data_uri": f"data:image/jpeg;base64,{encode_image(image_data)}",
        "processed_data_uri": f"data:image/jpeg;base64,{encode_image(processed)}"
    })

    return HttpResponse(html)


# =========================
# 🔹 Predicción doble (original + procesada)
# =========================
@csrf_exempt
def predict_view(request):

    if "orig_image" not in request.session or "proc_image" not in request.session:
        return HttpResponse("<p style='color:#f87171;'>⚠️ No hay imágenes para predecir.</p>")

    try:
        model = tf.keras.models.load_model("forest_fire_model_final.keras", compile=False)

        def decode_image(b64_string):
            data = base64.b64decode(b64_string)
            np_arr = np.frombuffer(data, np.uint8)
            return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        def prepare(img):
            img = cv2.resize(img, (160, 160))
            return np.expand_dims(img / 255.0, axis=0)

        img_orig = decode_image(request.session["orig_image"])
        img_proc = decode_image(request.session["proc_image"])

        prob_o = float(model.predict(prepare(img_orig))[0][0])
        prob_p = float(model.predict(prepare(img_proc))[0][0])

        label_o = "🔥 Incendio" if prob_o > 0.5 else "🌲 Sin Incendio"
        label_p = "🔥 Incendio" if prob_p > 0.5 else "🌲 Sin Incendio"

        html = f"""
        <div style='margin-top:15px'>
          <h3 style='color:#fbbf24;'>Resultados:</h3>
          <div style='display:flex;gap:40px'>

            <div style='width:50%;text-align:center'>
              <h4>Imagen Original</h4>
              <p><b>{label_o}</b> ({prob_o*100:.2f}%)</p>
            </div>

            <div style='width:50%;text-align:center'>
              <h4>Imagen Procesada</h4>
              <p><b>{label_p}</b> ({prob_p*100:.2f}%)</p>
            </div>
          </div>
        </div>
        """
        return HttpResponse(html)

    except Exception as e:
        return HttpResponse(f"<pre style='color:red'>❌ Error en predicción:\n{str(e)}</pre>")
