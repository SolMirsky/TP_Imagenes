from django import forms

class PredictForm(forms.Form):
    image = forms.ImageField(label="Seleccionar imagen")

    do_gray = forms.BooleanField(label="Escala de grises", required=False)
    do_wb = forms.BooleanField(label="Balance de blancos (Gray World)", required=False)
    do_sat = forms.BooleanField(label="Aumentar saturación", required=False)
    do_rgb = forms.BooleanField(label="Ajuste RGB", required=False)
    do_clahe = forms.BooleanField(label="Corrección CLAHE", required=False)
    do_dehaze = forms.BooleanField(label="Dehazing (Retinex)", required=False)
    do_denoise = forms.BooleanField(label="Reducción de ruido", required=False)

    sat_factor = forms.FloatField(label="Factor saturación", initial=1.2, required=False)
    r_factor = forms.FloatField(label="Canal R", initial=1.0, required=False)
    g_factor = forms.FloatField(label="Canal G", initial=1.0, required=False)
    b_factor = forms.FloatField(label="Canal B", initial=1.0, required=False)

    clahe_clip = forms.FloatField(label="CLAHE clip", initial=2.0, required=False)
    clahe_tiles = forms.IntegerField(label="CLAHE tiles", initial=8, required=False)
    retinex_sigma = forms.FloatField(label="Sigma Retinex", initial=80.0, required=False)

    denoise_mode = forms.ChoiceField(
        label="Modo de reducción de ruido",
        choices=[("bilateral", "Bilateral"), ("median", "Mediana")],
        initial="bilateral",
        required=False
    )

    bilateral_d = forms.IntegerField(label="Denoise D", initial=7, required=False)
    bilateral_sigmaColor = forms.IntegerField(label="Sigma Color", initial=50, required=False)
    bilateral_sigmaSpace = forms.IntegerField(label="Sigma Espacio", initial=50, required=False)
    median_ksize = forms.IntegerField(label="Kernel Mediana", initial=3, required=False)
