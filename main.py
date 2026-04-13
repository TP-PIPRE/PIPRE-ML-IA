import pandas as pd

from utils.loader import load_data

from models.ria01_desempeño import ClasificadorDesempeno
from models.ria03_recomendador import RecomendadorActividades
from models.ria08_anomalias import DetectorAnomalias
from models.ria11_tiempo import ClasificadorTiempo
from models.ria12_codigo import EvaluadorCodigo


def main():
    # =========================
    # CARGA Y PREPROCESS GLOBAL
    # =========================
    df = load_data("data/dataset.xlsx")

    print("✅ Datos preprocesados\n")

    # =========================
    # INPUT BASE (CRUDO)
    # =========================
    input_data = pd.DataFrame([{
        "tiempo_sesion_min": 30,
        "intentos": 3,
        "errores": 1,
        "nivel_logico": "medio",
        "uso_codigo": 40,
        "interacciones_ia": 2,
        "puntaje": 75,
        "dias_inactivo": 1
    }])

    # =========================
    # RIA-01 DESEMPEÑO
    # =========================
    ria01 = ClasificadorDesempeno()
    ria01.train(df)

    ria01_input = ria01.preprocess_data(input_data.copy(), is_training=False)

    print("RIA-01 Nivel:", ria01.predict(ria01_input))
    print(f"Accuracy: {ria01.accuracy:.2f}")
    print(f"Precision: {ria01.precision:.2f}\n")

    # =========================
    # RIA-03 RECOMENDADOR (🔥 RANDOMIZED)
    # =========================
    rec = RecomendadorActividades()
    rec.train(df)
    rec.evaluar(df)

    rec_input = df.mean(numeric_only=True).to_frame().T
    rec_input["nivel_logico"] = df["nivel_logico"].mode()[0]

    print("RIA-03:", rec.predict(rec_input))
    print(f"Accuracy: {rec.accuracy:.2f}")
    print(f"Precision: {rec.precision:.2f}\n")

    # =========================
    # RIA-08 ANOMALÍAS
    # =========================
    det = DetectorAnomalias()

    det.train(df)

    ejemplo = df.sample(1, random_state=42)

    print("RIA-08:", det.predict(ejemplo))
    print(f"Tasa de anomalías: {det.anomaly_ratio:.2f}")
    
    # =========================
    # RIA-11 TIEMPO (rápido/normal/lento)
    # =========================
    tiempo = ClasificadorTiempo()
    tiempo.train(df)

    tiempo_input = tiempo.preprocess(input_data.copy(), is_training=False)

    print("RIA-11 Tiempo:", tiempo.predict(tiempo_input))
  

   # =========================
    # RIA-12 CÓDIGO
    # =========================
    codigo = EvaluadorCodigo()
    codigo.train(df)

    # 🔥 AHORA SÍ usar preprocess_data
    codigo_input = codigo.preprocess_data(input_data.copy(), is_training=False)

    print("RIA-12:", codigo.predict(codigo_input))
    print(f"Accuracy: {codigo.accuracy:.2f}")
    print(f"Precision: {codigo.precision:.2f}")


if __name__ == "__main__":
    main()