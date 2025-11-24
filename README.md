## INTEGRANTES GRUPO

* ARNAUD, NURIA
* HAMRA, HERNAN
* SPANDONARI, VICTORIA
* WLADIMIRSKY, SOLANGE

## Clasificación de incendios en paisajes forestales 🌲🔥

Este proyecto aborda la detección automática de incendios forestales mediante clasificación binaria de imágenes (Fuego / Sin Fuego), utilizando Transfer Learning con MobileNetV2 y un conjunto de técnicas avanzadas de procesamiento de imagen aplicadas de forma interactiva a través de Streamlit.

El modelo se entrena con TensorFlow/Keras, y la aplicación permite al usuario subir imágenes, aplicar filtros visuales y de corrección, y obtener predicciones en tiempo real con visualizaciones comparativas (imagen original vs. imagen preprocesada).
Combina Machine Learning y una app web en Django para detectar incendios en imágenes de paisajes forestales.

## Dataset 

El modelo se entrena con el Forest Fire Dataset de Kaggle
https://www.kaggle.com/datasets/alik05/forest-fire-dataset
Clases: Fire y No Fire
Tamaño de imagen: 160×160 píxeles
División: 70 % entrenamiento, 20 % validación, 10 % te
y la app permite al usuario subir imágenes y clasificarlas. Además, se pueden aplicar filtros a las imágenes

## Arquitectura del Proyecto

Archivos clave

incendios.py – Entrenamiento y evaluación del modelo

Es el núcleo de Machine Learning del proyecto:

Cargar y dividir el dataset en train/validation/test.

Modelo de Clasificación: con Transfer Learning . Basado en MobileNetV2 con pesos preentrenados en ImageNet.

Entrenamiento en dos fases: Head training (base congelada) Se entrena primero la cabeza del modelo  y Luego se realiza fine-tuning parcial de las últimas capas.

Data Augmentation: Aplicado durante el entrenamiento para mejorar la generalización:

layers.RandomFlip("horizontal"),
layers.RandomRotation(0.05),
layers.RandomZoom(0.1),
layers.RandomContrast(0.1)

Evaluar el modelo en el set de prueba:

 accuracy
 
 confusion matrix
 
 classification report
 
Guardar modelos y configuración:

 forest_fire_best.keras → mejor modelo durante entrenamiento.
 
 forest_fire_model_final.keras → modelo final.
 
 train_config.json → parámetros usados (IMG_SIZE, BATCH_SIZE, EPOCHS, etc.).
 
Generar gráficos de accuracy vs epochs para entrenamiento y fine-tuning.

Uso:
python incendios.py


Aplicación Interactiva (Streamlit): Carga de imágenes (.jpg, .jpeg, .png).
Aplicación dinámica de filtros visuales.
Doble predicción: sobre la imagen original y la procesada.

## Técnicas de Procesamiento de Imágenes Implementadas

La aplicación implementa un conjunto de técnicas diseñadas para mejorar la visibilidad, corrección de color y reducción de ruido en imágenes reales de incendios:

Técnica	 y Descripción

Grayscale (Escala de grises)	Convierte la imagen a un solo canal de luminancia, útil para análisis estructural.

Balance de Blancos (Gray World)	Corrige dominantes de color (por ejemplo, exceso de rojo o verde) ajustando cada canal RGB al promedio global.

Ajuste RGB Manual	Permite modificar individualmente los niveles de rojo, verde y azul mediante sliders interactivos.

Aumento de Saturación	Intensifica los colores y mejora la separación entre zonas de fuego y fondo.

Dehazing (Retinex)	Elimina bruma o humo mediante el algoritmo Single Scale Retinex, realzando contraste y detalles.

CLAHE (Contrast Limited Adaptive Histogram Equalization)	Mejora el contraste local adaptativo en regiones oscuras o sobreexpuestas, preservando bordes.

Reducción de Ruido (Bilateral / Mediana)	Filtros que suavizan la imagen manteniendo bordes: el bilateral actúa sobre color y espacio, el mediano elimina artefactos impulsivos.

Escalado y Normalización	Redimensiona a 160×160 píxeles y normaliza valores a [0,1] para compatibilidad con MobileNetV2.


Estas técnicas pueden combinarse secuencialmente, visualizándose en tiempo real en la interfaz.

## Requerimientos
Para ejecutar el proyecto correctamente, se recomienda Python 3.9 o superior y las siguientes dependencias:
tensorflow>=2.12, opencv-python, numpy, pandas, scikit-image, scikit-learn, matplotlib, altair, streamlit y django


## Ejecución del Proyecto
1. Entrenamiento del modelo
El script incendios.py entrena el modelo de detección de incendios.
Ejecutar: python incendios.py

2. Ejecución de la app Streamlit
Asegurarse de tener el archivo forest_fire_model_final.keras en la misma carpeta que app_prediccion.py.
Luego ejecutar:
streamlit run app_prediccion.py


Abrir el navegador en la URL que aparece (por defecto: http://localhost:8501)
En la interfaz: Subir una imagen. Activar los filtros de procesamiento desde la barra lateral. Presionar “Ejecutar Predicción” para ver los resultados de detección.

3. Ejecución del Proyecto con Django
   
La aplicación también puede ejecutarse como un sitio web usando Django.

Desde la carpeta del proyecto:

Instalar dependencias:

pip install django tensorflow numpy matplotlib scikit-learn

Iniciar el servidor:

python manage.py runserver

Abrir el navegador:

http://127.0.0.1:8000/

Esta app permite interacción web con el modelo.
En la interfaz web de Django podrás:
Subir una imagen desde el navegador para analizar,
Seleccionar los filtros de procesamiento disponibles,
Ver la imagen procesada,
Obtener la predicción del modelo entrenado,
Navegar por una interfaz con sidebar.
fire_detector – App Django

Archivos importantes de la app:

views.py → recibe imágenes, carga el modelo y devuelve la predicción.

templates/ → interfaz de usuario.

static/ → recursos visuales.





``` 


## Estructura del Repositorio

TP_Imagenes/
├── incendios.py                  # Entrenamiento del modelo (MobileNetV2)
├── app_prediccion.py             # Aplicación interactiva Streamlit
├── forest_fire_model_final.keras # Modelo entrenado
├── train_config.json             # Parámetros de configuración
├── README.md                     # Documentación del proyecto
│
├── forest_env / venv             # Entornos virtuales (no deberían subirse)
├── db.sqlite3                    # Base de datos (no utilizada)
│
├── fire_detector/                # Proyecto Django principal
│   ├── manage.py                 # Herramienta de administración
│   │
│   ├── fire_detector/            # Configuración del proyecto
│   │   ├── settings.py           # Ajustes globales
│   │   ├── urls.py               # Rutas principales
│   │   ├── wsgi.py / asgi.py     # Arranque del servidor
│   │   └── __init__.py
│   │
│   └── detector/                 # Aplicación Django principal
│       ├── views.py              # Procesamiento y predicción
│       ├── urls.py               # Rutas de la app
│       ├── templates/            # HTML del frontend
│       ├── static/               # Archivos CSS/JS/imagenes
│       ├── forms.py              # Formularios
│       ├── utils.py              # Funciones auxiliares
│       ├── models.py             # Actualmente sin modelos
│       └── admin.py              # No utilizado por ahora
``` 
La carpeta `detector` contiene la aplicación principal, incluyendo vistas, formularios, plantillas HTML, recursos estáticos y funciones de procesamiento de imágenes. 


## Resultados

Precisión del modelo: > 95 % en validación
Tiempo de inferencia: < 0.1 s por imagen (CPU)
Robustez: tolerante a variaciones de luz, color y humo

Este proyecto aborda la detección automática de incendios forestales mediante clasificación binaria de imágenes (Fuego / Sin Fuego), utilizando Transfer Learning con MobileNetV2 y un conjunto de técnicas avanzadas de procesamiento de imagen aplicadas de forma interactiva a través de Streamlit.

El modelo se entrena con TensorFlow/Keras, y la aplicación permite al usuario subir imágenes, aplicar filtros visuales y de corrección, y obtener predicciones en tiempo real con visualizaciones comparativas (imagen original vs. imagen preprocesada).

## Dataset 
https://www.kaggle.com/datasets/alik05/forest-fire-dataset
Clases: Fire y No Fire
Tamaño de imagen: 160×160 píxeles
División: 70 % entrenamiento, 20 % validación, 10 % te


## Arquitectura del Proyecto

Modelo de Clasificación: Basado en MobileNetV2 con pesos preentrenados en ImageNet.
Entrenamiento en dos fases: Head training (base congelada) y Fine-tuning parcial de las últimas capas.
Data Augmentation: Aplicado durante el entrenamiento para mejorar la generalización:
layers.RandomFlip("horizontal"),
layers.RandomRotation(0.05),
layers.RandomZoom(0.1),
layers.RandomContrast(0.1)
Aplicación Interactiva (Streamlit): Carga de imágenes (.jpg, .jpeg, .png).
Aplicación dinámica de filtros visuales.
Doble predicción: sobre la imagen original y la procesada.

## Técnicas de Procesamiento de Imágenes Implementadas

La aplicación implementa un conjunto de técnicas diseñadas para mejorar la visibilidad, corrección de color y reducción de ruido en imágenes reales de incendios:

Técnica y 	Descripción
Grayscale (Escala de grises)	Convierte la imagen a un solo canal de luminancia, útil para análisis estructural.
Balance de Blancos (Gray World)	Corrige dominantes de color (por ejemplo, exceso de rojo o verde) ajustando cada canal RGB al promedio global.
Ajuste RGB Manual	Permite modificar individualmente los niveles de rojo, verde y azul mediante sliders interactivos.
Aumento de Saturación	Intensifica los colores y mejora la separación entre zonas de fuego y fondo.
Dehazing (Retinex)	Elimina bruma o humo mediante el algoritmo Single Scale Retinex, realzando contraste y detalles.
CLAHE (Contrast Limited Adaptive Histogram Equalization)	Mejora el contraste local adaptativo en regiones oscuras o sobreexpuestas, preservando bordes.
Reducción de Ruido (Bilateral / Mediana)	Filtros que suavizan la imagen manteniendo bordes: el bilateral actúa sobre color y espacio, el mediano elimina artefactos impulsivos.
Escalado y Normalización	Redimensiona a 160×160 píxeles y normaliza valores a [0,1] para compatibilidad con MobileNetV2.

Estas técnicas pueden combinarse secuencialmente, visualizándose en tiempo real en la interfaz.

## Requerimientos
Para ejecutar el proyecto correctamente, se recomienda Python 3.9 o superior y las siguientes dependencias:
tensorflow>=2.12, opencv-python, numpy, pandas, scikit-image, scikit-learn, matplotlib, altair y streamlit


## Ejecución del Proyecto
1. Entrenamiento del modelo
El script incendios.py entrena el modelo de detección de incendios.
Ejecutar: python incendios.py

2. Ejecución de la app Streamlit
Asegurate de tener el archivo forest_fire_model_final.keras en la misma carpeta que app_prediccion.py.
Luego ejecutá:
streamlit run app_prediccion.py
Abrí el navegador en la URL que aparece (por defecto: http://localhost:8501)
En la interfaz: Subí una imagen. Activá los filtros de procesamiento desde la barra lateral. Presioná “Ejecutar Predicción” para ver los resultados de detección.

3. Ejecución del Proyecto con Django
La aplicación también puede ejecutarse como un sitio web usando Django.
Desde la carpeta del proyecto, correr:
python manage.py runserver
Abrir en el navegador:  http://127.0.0.1:8000/
En la interfaz web de Django podrás:
Subir una imagen para analizar, Seleccionar los filtros de procesamiento disponibles, Ver la imagen procesada, Obtener la predicción del modelo entrenado, Navegar por una interfaz con sidebar.

## Estructura del Repositorio
TP_Imagenes/
│
├── incendios.py                  # Entrenamiento del modelo (MobileNetV2)
├── app_prediccion.py             # Aplicación interactiva Streamlit
├── forest_fire_model_final.keras # Modelo entrenado
├── train_config.json             # Parámetros de configuración
└── README.md                     # Documentación del proyecto


## Resultados

Precisión del modelo: > 95 % en validación
Tiempo de inferencia: < 0.1 s por imagen (CPU)
Robustez: tolerante a variaciones de luz, color y humo

## Créditos
Proyecto desarrollado en el marco de la
Tecnicatura Superior en Ciencia de Datos e Inteligencia Artificial
IFTS Nº 18 – Ciudad Autónoma de Buenos Aires

