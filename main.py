import pandas as pd

from utils.loader import load_data
from utils.preprocess import preprocess  # 🔥 IMPORTANTE

from models.ria01_desempeño import ClasificadorDesempeno
from models.ria03_recomendador import RecomendadorActividades
from models.ria08_anomalias import DetectorAnomalias
from models.ria11_tiempo import PrediccionTiempo
from models.ria12_codigo import EvaluadorCodigo


def main():
    # =========================
    # 🔥 CARGA Y PREPROCESS
    # =========================
    df = load_data("data/dataset.xlsx")
    df, encoders = preprocess(df, is_training=True)

    print("✅ Datos preprocesados\n")

    # =========================
    # 🔹 RIA-01 DESEMPEÑO
    # =========================
    ria01 = ClasificadorDesempeno()
    ria01.train(df)

    ria01_input = pd.DataFrame([{
        "id_estudiante": 1,
        "edad": 10,
        "grado": 5,
        "tiempo_sesion_min": 30,
        "intentos": 3,
        "errores": 1,
        "puntaje": 80,
        "actividades_completadas": 5,
        "tasa_exito": 0.8,
        "dias_inactivo": 1,
        "nivel_logico": "medio",
        "uso_bloques": 1,
        "uso_codigo": 40,
        "interacciones_ia": 2,
        "ayuda_solicitada": 1,
        "emocion_detectada": "feliz",
        "riesgo_abandono": 0,
        "rendimiento": "medio",
        "complejidad_actividad": 10,
        "tasa_exito_ajustada_intentos": 0.25,
        "frecuencia_interaccion_ia_error": 0.3,
        "dias_activos_recientes": 3
    }])

    ria01_input, _ = preprocess(ria01_input, encoders=encoders, is_training=False)

    print("RIA-01 Nivel:", ria01.predict(ria01_input))
    print(f"Accuracy: {ria01.accuracy:.2f}")
    print(f"Precision: {ria01.precision:.2f}\n")

    # =========================
    # 🔹 RIA-03 RECOMENDADOR
    # =========================
    rec = RecomendadorActividades()
    rec.train(df)

    rec_input = ria01_input.copy()  # 🔥 reutilizas input ya limpio

    print("RIA-03:", rec.predict(rec_input))
    print(f"Accuracy: {rec.accuracy:.2f}")
    print(f"Precision: {rec.precision:.2f}\n")

    # =========================
    # 🔹 RIA-08 ANOMALÍAS
    # =========================
    anom = DetectorAnomalias()
    anom.train(df)

    anom_input = ria01_input.copy()

    print("RIA-08:", anom.predict(anom_input))
    print(f"Tasa de anomalías: {anom.anomaly_ratio:.2f}\n")

    # =========================
    # 🔹 RIA-11 TIEMPO
    # =========================
    tiempo = PrediccionTiempo()
    tiempo.train(df)

    tiempo_input = ria01_input.copy()

    print("RIA-11 Tiempo estimado:", tiempo.predict(tiempo_input))
    print(f"R2 Score: {tiempo.r2:.2f}")
    print(f"MAE: {tiempo.mae:.2f}")
    print(f"Exactitud (±5 min): {tiempo.accuracy:.2f}\n")

    # =========================
    # 🔹 RIA-12 CÓDIGO
    # =========================
    codigo = EvaluadorCodigo()
    codigo.train(df)

    codigo_input = ria01_input.copy()

    print("RIA-12:", codigo.predict(codigo_input))
    print(f"Accuracy: {codigo.accuracy:.2f}")
    print(f"Precision: {codigo.precision:.2f}")


if __name__ == "__main__":
    main()
