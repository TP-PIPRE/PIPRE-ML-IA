import pandas as pd

# 🔥 IMPORTAR MODELOS
from models.ria01_desempeño import ClasificadorDesempeno
from models.ria03_recomendador import RecomendadorActividades
from models.ria08_anomalias import DetectorAnomalias
from models.ria11_tiempo import ClasificadorTiempo
from models.ria12_codigo import EvaluadorCodigo

# 🔥 UI
from ui.ui_resultados import mostrar_resultados


def main():

    # =========================
    # 📊 CARGAR DATASET
    # =========================
    df = pd.read_excel("data/dataset.xlsx")

    # =========================
    # 🔹 ENTRENAR MODELOS
    # =========================
    ria1 = ClasificadorDesempeno()
    ria1.train(df)

    ria3 = RecomendadorActividades()
    ria3.train(df)
    ria3.evaluar(df)

    ria8 = DetectorAnomalias()
    ria8.train(df)

    ria11 = ClasificadorTiempo()
    ria11.train(df)

    ria12 = EvaluadorCodigo()
    ria12.train(df)

    # =========================
    # 🔹 INPUT DE PRUEBA
    # =========================
    data = df.sample(1)

    # =========================
    # 📊 RESULTADOS
    # =========================
    resultados = {

        # 🔥 RIA 1
        "RIA1 - Desempeño": {
            "resultado": ria1.predict(data),
            "accuracy": ria1.accuracy,
            "precision": ria1.precision,
            "importancias": dict(zip(
                ria1.feature_columns,
                ria1.model.feature_importances_
            ))
        },

        # 🔥 RIA 3
        "RIA3 - Recomendación": {
            "resultado": ria3.predict(data),
            "accuracy": ria3.accuracy,
            "precision": ria3.precision,
            "importancias": dict(zip(
                ria3.feature_columns,
                ria3.model_stage1.feature_importances_
            ))
        },

        # 🔥 RIA 8 (anomalias con importancia)
        "RIA8 - Anomalías": {
            "resultado": ria8.predict(data),
            "anomalias": f"{ria8.anomaly_ratio:.2%} del dataset detectado como anómalo",
            "importancias": ria8.calcular_importancia(df)
        },

        # 🔥 RIA 11
        "RIA11 - Tiempo": {
            "resultado": ria11.predict(data),
            "accuracy": ria11.accuracy,
            "precision": ria11.precision,
            "importancias": dict(zip(
                ria11.feature_columns,
                ria11.model.feature_importances_
            ))
        },

        # 🔥 RIA 12
        "RIA12 - Código": {
            "resultado": ria12.predict(data),
            "accuracy": ria12.accuracy,
            "precision": ria12.precision,
            "importancias": dict(zip(
                ria12.feature_columns,
                ria12.model.feature_importances_
            ))
        }
    }

    # =========================
    # MOSTRAR UI
    # =========================
    mostrar_resultados(resultados)


if __name__ == "__main__":
    main()