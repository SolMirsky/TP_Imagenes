from django import forms

class FireForm(forms.Form):
    # Imagen principal
    image = forms.ImageField(
        label="Seleccionar imagen",
        required=False
    )

    # ===== Filtros principales =====
    do_gray = forms.BooleanField(label="Escala de grises", required=False)
    do_wb = forms.BooleanField(label="Balance de blancos (Gray World)", required=False)
    do_sat = forms.BooleanField(label="Aumentar saturación", required=False)
    do_rgb = forms.BooleanField(label="Ajuste RGB", required=False)
    do_clahe = forms.BooleanField(label="Corrección CLAHE", required=False)
    do_dehaze = forms.BooleanField(label="Dehazing (Retinex)", required=False)
    do_denoise = forms.BooleanField(label="Reducción de ruido", required=False)

    # ===== Parámetros ajustables =====
    # Saturación
    sat_factor = forms.FloatField(
        label="Factor de saturación",
        min_value=0.5, max_value=2.5,
        initial=1.2,
        required=False
    )

    # Ajuste RGB individual
    r_factor = forms.FloatField(
        label="Canal R", min_value=0.5, max_value=2.0,
        initial=1.0, required=False
    )
    g_factor = forms.FloatField(
        label="Canal G", min_value=0.5, max_value=2.0,
        initial=1.0, required=False
    )
    b_factor = forms.FloatField(
        label="Canal B", min_value=0.5, max_value=2.0,
        initial=1.0, required=False
    )

    # CLAHE
    clahe_clip = forms.FloatField(
        label="CLAHE clip limit",
        min_value=1.0, max_value=4.0,
        initial=2.0,
        required=False
    )
    clahe_tiles = forms.IntegerField(
        label="CLAHE tiles",
        min_value=4, max_value=16,
        initial=8,
        required=False
    )

    # Retinex (dehazing)
    retinex_sigma = forms.FloatField(
        label="Sigma Retinex",
        min_value=10.0, max_value=150.0,
        initial=80.0,
        required=False
    )

    # ===== Reducción de ruido =====
    denoise_mode = forms.ChoiceField(
        label="Modo de reducción de ruido",
        choices=[("bilateral", "Bilateral"), ("median", "Mediana")],
        initial="bilateral",
        required=False
    )

    bilateral_d = forms.IntegerField(
        label="Bilateral D",
        min_value=1, max_value=15,
        initial=7,
        required=False
    )
    bilateral_sigmaColor = forms.IntegerField(
        label="Sigma Color",
        min_value=10, max_value=150,
        initial=50,
        required=False
    )
    bilateral_sigmaSpace = forms.IntegerField(
        label="Sigma Espacio",
        min_value=10, max_value=150,
        initial=50,
        required=False
    )
    median_ksize = forms.IntegerField(
        label="Kernel Mediana",
        min_value=1, max_value=9,
        initial=3,
        required=False
    )
