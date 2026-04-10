import pandas as pd

from utils.loader import load_data

from models.ria03_recomendador import RecomendadorActividades
from models.ria08_anomalias import DetectorAnomalias
from models.ria11_tiempo import PrediccionTiempo
from models.ria12_codigo import EvaluadorCodigo


def main():
    # 🔥 Cargar datos SIN preprocess global
    df = load_data("data/dataset.xlsx")

    print("✅ Datos cargados\n")

    # =========================
    # 🔹 RIA-03 RECOMENDADOR
    # =========================
    rec = RecomendadorActividades()
    rec.train(df)
    
    rec_input = pd.DataFrame([{
        "tiempo_sesion_min": 30,
        "intentos": 3,
        "errores": 1,
        "puntaje": 80,
        "tasa_exito": 0.8,
        "nivel_logico": 2,
        "uso_codigo": 40,
        "complejidad_actividad": 10,
        "tasa_exito_ajustada_intentos": 0.25,
        "frecuencia_interaccion_ia_error": 0.5,
        "dias_activos_recientes": 0.8
    }])

    print("RIA-03:", rec.predict(rec_input))
    print(f"Accuracy: {rec.accuracy:.2f}")
    print(f"Precision: {rec.precision:.2f}\n")

    # =========================
    # 🔹 RIA-08 ANOMALÍAS
    # =========================
    anom = DetectorAnomalias()
    anom.train(df)

    anom_input = pd.DataFrame([{
        "tiempo_sesion_min": 30,
        "intentos": 3,
        "errores": 1,
        "puntaje": 80,
        "dias_inactivo": 2
    }])

    print("RIA-08:", anom.predict(anom_input))
    print(f"Tasa de anomalías: {anom.anomaly_ratio:.2f}\n")

    # =========================
    # 🔹 RIA-11 TIEMPO
    # =========================
    tiempo = PrediccionTiempo()
    tiempo.train(df)

    tiempo_input = pd.DataFrame([{
        "intentos": 3,
        "errores": 1,
        "nivel_logico": 2,
        "tasa_exito": 0.8,
        "actividades_completadas": 5
    }])

    print("RIA-11 Tiempo estimado:", tiempo.predict(tiempo_input))
    print(f"R2 Score: {tiempo.r2:.2f}")
    print(f"MAE: {tiempo.mae:.2f}")
    print(f"Exactitud (±5 min): {tiempo.accuracy:.2f}\n")
    # =========================
    # 🔹 RIA-12 CÓDIGO
    # =========================
    codigo = EvaluadorCodigo()
    codigo.train(df)

    codigo_input = pd.DataFrame([{
        "errores": 1,
        "intentos": 3,
        "uso_codigo": 1,
        "uso_bloques": 0,
        "nivel_logico": 2
    }])

    print("RIA-12:", codigo.predict(codigo_input))
    print(f"Accuracy: {codigo.accuracy:.2f}")
    print(f"Precision: {codigo.precision:.2f}")


if __name__ == "__main__":
    main()