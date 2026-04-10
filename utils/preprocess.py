import pandas as pd
from sklearn.preprocessing import LabelEncoder

EXPECTED_COLUMNS = [
    "id_estudiante", "edad", "grado", "tiempo_sesion_min",
    "intentos", "errores", "puntaje", "actividades_completadas",
    "tasa_exito", "dias_inactivo", "nivel_logico",
    "uso_bloques", "uso_codigo", "interacciones_ia",
    "ayuda_solicitada", "emocion_detectada",
    "riesgo_abandono", "rendimiento",
    "complejidad_actividad", "tasa_exito_ajustada_intentos",
    "frecuencia_interaccion_ia_error", "dias_activos_recientes"
]

def preprocess(df):
    # Validar columnas
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")

    df = df[EXPECTED_COLUMNS]

    # Eliminar nulos
    df = df.dropna()

    # Encoding de variables categóricas
    categorical_cols = ["nivel_logico", "emocion_detectada", "rendimiento"]

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders