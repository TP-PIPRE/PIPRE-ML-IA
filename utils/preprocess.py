import pandas as pd
from sklearn.preprocessing import LabelEncoder

EXPECTED_COLUMNS = [
    "id_estudiante", "edad", "grado", "tiempo_sesion_min",
    "intentos", "errores", "puntaje", "actividades_completadas",
    "tasa_exito", "dias_inactivo", "nivel_logico",
    "uso_bloques", "uso_codigo", "interacciones_ia",
    "ayuda_solicitada", "emocion_detectada",
    "riesgo_abandono", "rendimiento",
]

def preprocess(df, encoders=None, is_training=True):

    # Validar columnas
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")

    df = df[EXPECTED_COLUMNS]

    # Eliminar nulos
    df = df.dropna()

    categorical_cols = ["nivel_logico", "emocion_detectada", "rendimiento"]

    if is_training:
        encoders = {}

        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    else:
        for col in categorical_cols:
            le = encoders[col]

            def safe_transform(val):
                if val in le.classes_:
                    return le.transform([val])[0]
                else:
                    return 0  # valor por defecto

            df[col] = df[col].apply(safe_transform)

    return df, encoders
