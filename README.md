# TP_Imagenes 

## INTEGRANTES GRUPO

* ARNAUD, NURIA
* HAMRA, HERNAN
* SPANDONARI, VICTORIA
* WLADIMIRSKY, SOLANGE

## Clasificación de incendios en paisajes forestales

Este proyecto entrena un modelo de clasificación binaria (incendio / no-incendio) usando Transfer Learning con MobileNetV2 sobre un conjunto de imágenes de bosques. El foco está en un pipeline de preprocesamiento de imágenes pensado para escenas reales (iluminación irregular, niebla/humo, dominante de color, ruido), aplicado dentro de tf.data para mantener un flujo eficiente.

Dataset: https://www.kaggle.com/datasets/alik05/forest-fire-dataset
Requisitos:  tensorflow>=2.12, opencv-python, numpy, scikit-image, matplotlib


## Tecnicas:

- CLAHE (corrección de iluminación local)
- Dehazing suave (Retinex) para atenuar bruma/humo y realzar detalle
- Balance de blancos (Gray World) para neutralizar dominantes de color
- Reducción de ruido (bilateral o mediana, preservando bordes)
- Downscale al tamaño de inferencia (160×160 por defecto)
