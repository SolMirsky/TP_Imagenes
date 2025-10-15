"""
Clasificación de incendios en paisajes forestales con preprocesamiento avanzado
- CLAHE (corrección de iluminación local)
- Dehazing suave (Retinex)
- Balance de blancos (Gray World)
- Reducción de ruido (bilateral/mediana)
- Downscale al tamaño de inferencia

Dataset: https://www.kaggle.com/datasets/alik05/forest-fire-dataset
pip install  tensorflow>=2.12, opencv-python, numpy, scikit-image, matplotlib
"""

import os
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# CONFIGURACION GENERAL

CONFIG = {
    "DATA_DIR": r"C:\Users\nuria\Downloads\Forest Fire Dataset",
    "IMG_SIZE": (160, 160),         # tamaño de inferencia (downscale)
    "BATCH_SIZE": 32,
    "VAL_SPLIT": 0.2,
    "TEST_SPLIT": 0.1,              # del total (se toma desde el conjunto completo)
    "SEED": 42,
    "EPOCHS": 15,
    "BASE_LEARNING_RATE": 1e-4,
    "USE_IMAGENET_WEIGHTS": True,   
    "AUGMENTATION": True,

    # Activar/ajustar pasos de preprocesado
    "PREPROC": {
        "apply_clahe": True,
        "clahe_clip": 2.0,
        "clahe_tiles": 8,

        "apply_dehaze_soft": True,   # Retinex (suave)
        "retinex_sigma": 80.0,       # suavizado gaussiano para Retinex

        "apply_white_balance": True, # Gray World
        "wb_eps": 1e-6,

        "denoise": "bilateral",      # "bilateral", "median" o None
        "bilateral_d": 7,
        "bilateral_sigmaColor": 50,
        "bilateral_sigmaSpace": 50,
        "median_ksize": 3,
    }
}


# UTILIDADES DE PREPROCESAMIENTO

def to_uint8(img):
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def apply_clahe_bgr(bgr, clip=2.0, tiles=8):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return out

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

def white_balance_gray_world(bgr, eps=1e-6):
    img = bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0) + eps
    gray = means.mean()
    gain = gray / means
    balanced = img * gain
    return to_uint8(balanced)

def denoise_image(bgr, method="bilateral", cfg=None):
    if method is None:
        return bgr
    cfg = cfg or {}
    if method.lower() == "bilateral":
        d = cfg.get("bilateral_d", 7)
        sc = cfg.get("bilateral_sigmaColor", 50)
        ss = cfg.get("bilateral_sigmaSpace", 50)
        return cv2.bilateralFilter(bgr, d=d, sigmaColor=sc, sigmaSpace=ss)
    if method.lower() == "median":
        k = cfg.get("median_ksize", 3)
        k = max(3, int(k) if int(k) % 2 == 1 else int(k) + 1)  # impar
        return cv2.medianBlur(bgr, k)
    return bgr

def preprocess_bgr(bgr, cfg):
    """
    1) CLAHE -> 2) Retinex -> 3) Gray World -> 4) Denoise -> 5) Resize
    """
    if cfg["apply_clahe"]:
        bgr = apply_clahe_bgr(bgr, clip=cfg["clahe_clip"], tiles=cfg["clahe_tiles"])
    if cfg["apply_dehaze_soft"]:
        bgr = retinex_dehaze(bgr, sigma=cfg["retinex_sigma"])
    if cfg["apply_white_balance"]:
        bgr = white_balance_gray_world(bgr, eps=cfg["wb_eps"])
    if cfg["denoise"] is not None:
        bgr = denoise_image(bgr, method=cfg["denoise"], cfg=cfg)
    bgr = cv2.resize(bgr, CONFIG["IMG_SIZE"], interpolation=cv2.INTER_AREA)
    return bgr

def tf_preprocess_image(image):
    """
    Recibe tensor [H,W,3] uint8 (RGB) y devuelve float32 [0,1] (IMG_SIZE, 3).
    Nota: convertimos a numpy, aseguramos contigüidad y evitamos cvtColor usando slicing.
    """
    def _py(img_np):
        # Asegurar numpy array uint8 y contiguo
        try:
            import tensorflow as _tf  # para detectar tf.Tensor
            if isinstance(img_np, _tf.Tensor):
                img_np = img_np.numpy()
        except Exception:
            pass
        img_np = np.asarray(img_np)
        if img_np.dtype != np.uint8:
            img_np = img_np.astype(np.uint8)
        img_np = np.ascontiguousarray(img_np)

        # RGB -> BGR sin cv2.cvtColor (evita error de tipos)
        bgr = img_np[..., ::-1]

        # Preprocesado con OpenCV (en BGR)
        bgr = preprocess_bgr(bgr, CONFIG["PREPROC"])

        # Volver a RGB (slicing)
        rgb = bgr[..., ::-1].astype(np.float32) / 255.0
        return rgb

    rgb = tf.py_function(func=_py, inp=[image], Tout=tf.float32)
    rgb.set_shape((*CONFIG["IMG_SIZE"], 3))
    return rgb

# train/val/test

def make_splits():
    # 1) train+val
    train_val = tf.keras.utils.image_dataset_from_directory(
        CONFIG["DATA_DIR"],
        labels="inferred",
        label_mode="int",
        image_size=(250, 250),   
        batch_size=CONFIG["BATCH_SIZE"],
        validation_split=CONFIG["VAL_SPLIT"],
        subset="training",
        seed=CONFIG["SEED"],
        shuffle=True
    )
    val = tf.keras.utils.image_dataset_from_directory(
        CONFIG["DATA_DIR"],
        labels="inferred",
        label_mode="int",
        image_size=(250, 250),
        batch_size=CONFIG["BATCH_SIZE"],
        validation_split=CONFIG["VAL_SPLIT"],
        subset="validation",
        seed=CONFIG["SEED"],
        shuffle=True
    )
    class_names = train_val.class_names

    # 2) set completo para muestrear test
    full = tf.keras.utils.image_dataset_from_directory(
        CONFIG["DATA_DIR"],
        labels="inferred",
        label_mode="int",
        image_size=(250, 250),
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True,
        seed=CONFIG["SEED"]
    )

    total_batches = tf.data.experimental.cardinality(full).numpy()
    test_batches = max(1, int(round(total_batches * CONFIG["TEST_SPLIT"])))
    test = full.take(test_batches)
    return train_val, val, test, class_names


# DATA AUGMENTATION

def build_augmentation():
    if not CONFIG["AUGMENTATION"]:
        return tf.keras.Sequential(name="no_aug")
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augment")


# MODELO

def build_model(num_classes):
    weights = "imagenet" if CONFIG["USE_IMAGENET_WEIGHTS"] else None

    inputs = layers.Input(shape=(*CONFIG["IMG_SIZE"], 3))
    x = build_augmentation()(inputs)

    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=weights,
        input_shape=(*CONFIG["IMG_SIZE"], 3),
        pooling="avg"
    )
    base.trainable = False  # warmup del head primero

    x = base(x, training=False)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CONFIG["BASE_LEARNING_RATE"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, base


# DATA PIPELINE con preprocesado (por batch)

def add_preprocessing(ds):
    def _pp(x, y):
        x = tf.cast(x, tf.uint8)  # aseguramos uint8
        x = tf.map_fn(
            lambda img: tf_preprocess_image(img),  # img: (H,W,3) -> (IMG_SIZE,3)
            x,
            fn_output_signature=tf.float32
        )
        return x, y

    return ds.map(_pp, num_parallel_calls=tf.data.AUTOTUNE) \
             .cache() \
             .prefetch(tf.data.AUTOTUNE)


# ENTRENAMIENTO

def train():
    train_val_raw, val_raw, test_raw, class_names = make_splits()
    num_classes = len(class_names)
    print("Clases:", class_names)

    train_ds = add_preprocessing(train_val_raw)
    val_ds   = add_preprocessing(val_raw)
    test_ds  = add_preprocessing(test_raw)

    model, base = build_model(num_classes)

    cb = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ModelCheckpoint("forest_fire_best.keras", monitor="val_accuracy", save_best_only=True)
    ]

    print("Entrenando head (base congelada)...")
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG["EPOCHS"],
        callbacks=cb
    )

    # Fine-tuning (descongelamos parte del backbone)
    print("Fine-tuning parcial...")
    base.trainable = True
    for layer in base.layers[:-30]:  # deja libres ~30 capas finales
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(CONFIG["BASE_LEARNING_RATE"] / 10),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    hist_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max(6, CONFIG["EPOCHS"] // 3),
        callbacks=cb
    )

    # Evaluación
    print("Evaluando en test...")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test acc: {test_acc:.4f}")

    # Reporte detallado
    y_true, y_pred = [], []
    for xb, yb in test_ds:
        preds = model.predict(xb, verbose=0)
        y_true.append(yb.numpy())
        y_pred.append(np.argmax(preds, axis=1))
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))

    model.save("forest_fire_model_final.keras")
    with open("train_config.json", "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2, ensure_ascii=False)

    # VISUALIZACIONES
    
    def _plot_hist(h, title):
        plt.figure()
        plt.plot(h.history["accuracy"], label="acc")
        plt.plot(h.history["val_accuracy"], label="val_acc")
        plt.title(title)
        plt.xlabel("Epoch"); plt.ylabel("Accuracy")
        plt.legend(); plt.tight_layout()
        plt.show()

    _plot_hist(hist, "Head training")
    _plot_hist(hist_ft, "Fine-tuning")

if __name__ == "__main__":
    tf.keras.utils.set_random_seed(CONFIG["SEED"])
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    train()
