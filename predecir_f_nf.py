"""
Predicción de incendios forestales con modelo entrenado
Usa el archivo forest_fire_model_final.keras generado por incendios.py
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from incendios import preprocess_bgr, CONFIG  # reutilizamos funciones y parámetros del script principal


# === CONFIGURACIÓN ===

# Ruta del modelo entrenado
MODEL_PATH = "forest_fire_model_final.keras"

# Carpeta con imágenes nuevas para probar
TEST_IMG_DIR = "imagenes_prueba"   # debe existir y contener JPG o PNG


def cargar_modelo():
    """Carga el modelo entrenado"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}. Ejecutá incendios.py primero.")
    print(f"Cargando modelo desde {MODEL_PATH} ...")
    return tf.keras.models.load_model(MODEL_PATH)


def predecir_imagen(modelo, ruta_img):
    """Predice si hay fuego o no en una imagen individual"""
    img = cv2.imread(ruta_img)
    if img is None:
        print(f"No se pudo leer {ruta_img}")
        return None

    # Preprocesar (igual que en entrenamiento)
    proc = preprocess_bgr(img, CONFIG["PREPROC"])
    rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    rgb = np.expand_dims(rgb, axis=0)  # agregar dimensión batch

    # Predicción
    pred = modelo.predict(rgb, verbose=0)
    clase = np.argmax(pred)
    prob = float(np.max(pred))
    etiqueta = "🔥 FUEGO" if clase == 1 else "🌳 SIN FUEGO"

    print(f"{os.path.basename(ruta_img)} → {etiqueta} ({prob:.2%})")
    return clase, prob


def main():
    modelo = cargar_modelo()

    if not os.path.exists(TEST_IMG_DIR):
        print(f"No existe la carpeta {TEST_IMG_DIR}. Creala y agregá imágenes para probar.")
        return

    imagenes = [os.path.join(TEST_IMG_DIR, f)
                for f in os.listdir(TEST_IMG_DIR)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not imagenes:
        print("No se encontraron imágenes en la carpeta de prueba.")
        return

    print(f"\n📸 Probando {len(imagenes)} imágenes...\n")
    for img_path in imagenes:
        predecir_imagen(modelo, img_path)


if __name__ == "__main__":
    main()
