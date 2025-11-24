
from django.urls import path
from . import views

urlpatterns = [
    path("", views.fire_detector_view, name="home"),
    path("process/", views.process_image_ajax, name="process_image_ajax"),  # 👈 ESTA ES LA URL QUE FALTABA
    path("predict/", views.predict_view, name="predict"),
]
