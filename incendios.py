"""
Clasificación de incendios en paisajes forestales (sin preprocesamientos de imagen)
- Sin CLAHE, sin Retinex, sin balance de blancos, sin denoise
- Resize a tamaño de inferencia (hecho por image_dataset_from_directory)
- Normalización a [0,1] y conversión a lo que espera MobileNetV2

Dataset: https://www.kaggle.com/datasets/alik05/forest-fire-dataset
Instalar: pip install "tensorflow>=2.12" numpy scikit-learn matplotlib
"""

import os
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# =========================
# CONFIGURACION GENERAL
# =========================
CONFIG = {
    "DATA_DIR": r"Forest Fire Dataset",   # <-- ajustá esta ruta a tu carpeta
    "IMG_SIZE": (160, 160),               # tamaño de inferencia
    "BATCH_SIZE": 32,
    "VAL_SPLIT": 0.2,
    "TEST_SPLIT": 0.1,                    # fracción del total para test
    "SEED": 42,
    "EPOCHS": 15,
    "BASE_LEARNING_RATE": 1e-4,
    "USE_IMAGENET_WEIGHTS": True,
    "AUGMENTATION": True,
}

# Carpeta y rutas de guardado (evita OSError [Errno 22] en Windows)
SAVE_DIR = Path.cwd() / "models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
BEST_PATH  = SAVE_DIR / "forest_fire_best.keras"
FINAL_PATH = SAVE_DIR / "forest_fire_model_final.keras"

# =========================
# DATASETS (train/val/test)
# =========================
def make_splits():
    """
    Carga train/val/test con resize directo a IMG_SIZE.
    Sin preprocesado adicional; solo normalización más adelante.
    """
    train = tf.keras.utils.image_dataset_from_directory(
        CONFIG["DATA_DIR"],
        labels="inferred",
        label_mode="int",
        image_size=CONFIG["IMG_SIZE"],      # resize aquí
        batch_size=CONFIG["BATCH_SIZE"],
        validation_split=CONFIG["VAL_SPLIT"],
        subset="training",
        seed=CONFIG["SEED"],
        shuffle=True,
    )
    val = tf.keras.utils.image_dataset_from_directory(
        CONFIG["DATA_DIR"],
        labels="inferred",
        label_mode="int",
        image_size=CONFIG["IMG_SIZE"],
        batch_size=CONFIG["BATCH_SIZE"],
        validation_split=CONFIG["VAL_SPLIT"],
        subset="validation",
        seed=CONFIG["SEED"],
        shuffle=True,
    )
    class_names = train.class_names

    # set completo para muestrear test (nota: puede solapar con train/val si no separás por carpeta)
    full = tf.keras.utils.image_dataset_from_directory(
        CONFIG["DATA_DIR"],
        labels="inferred",
        label_mode="int",
        image_size=CONFIG["IMG_SIZE"],
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True,
        seed=CONFIG["SEED"],
    )
    total_batches = tf.data.experimental.cardinality(full).numpy()
    test_batches = max(1, int(round(total_batches * CONFIG["TEST_SPLIT"])))
    test = full.take(test_batches)

    return train, val, test, class_names

# =========================
# DATA AUGMENTATION
# =========================
def build_augmentation():
    if not CONFIG["AUGMENTATION"]:
        return tf.keras.Sequential(name="no_aug")
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augment",
    )

# =========================
# MODELO (Transfer Learning)
# =========================
def build_model(num_classes):
    weights = "imagenet" if CONFIG["USE_IMAGENET_WEIGHTS"] else None

    inputs = layers.Input(shape=(*CONFIG["IMG_SIZE"], 3), dtype=tf.float32)  # recibirá [0,1]
    x = build_augmentation()(inputs)  # aug en [0,1]

    # MobileNetV2 espera entrada preprocesada de [0,255] -> [-1,1]
    # Convertimos dentro del grafo para mantener el pipeline en un solo modelo.
    x = layers.Lambda(lambda t: tf.keras.applications.mobilenet_v2.preprocess_input(t * 255.0))(x)

    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=weights,
        input_shape=(*CONFIG["IMG_SIZE"], 3),
        pooling="avg",
    )
    base.trainable = False  # warmup del head primero

    x = base(x, training=False)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CONFIG["BASE_LEARNING_RATE"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base

# =========================
# DATA PIPELINE (normalización + performance)
# =========================
def add_preprocessing(ds):
    # Normaliza a [0,1] y optimiza pipeline
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.cache().prefetch(tf.data.AUTOTUNE)

# =========================
# ENTRENAMIENTO
# =========================
def train():
    train_raw, val_raw, test_raw, class_names = make_splits()
    num_classes = len(class_names)
    print("Clases:", class_names)

    train_ds = add_preprocessing(train_raw)
    val_ds   = add_preprocessing(val_raw)
    test_ds  = add_preprocessing(test_raw)

    model, base = build_model(num_classes)

    cb = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ModelCheckpoint(BEST_PATH.as_posix(), monitor="val_accuracy", save_best_only=True),
    ]

    print("Entrenando head (base congelada)...")
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG["EPOCHS"],
        callbacks=cb,
    )

    # Fine-tuning parcial
    print("Fine-tuning parcial...")
    base.trainable = True
    for layer in base.layers[:-30]:  # deja libres ~30 capas finales
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(CONFIG["BASE_LEARNING_RATE"] / 10),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    hist_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max(6, CONFIG["EPOCHS"] // 3),
        callbacks=cb,
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

    # Guardado seguro en Windows
    model.save(FINAL_PATH.as_posix())
    with open((SAVE_DIR / "train_config.json").as_posix(), "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2, ensure_ascii=False)

    # Plots
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

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    tf.keras.utils.set_random_seed(CONFIG["SEED"])
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    train()
