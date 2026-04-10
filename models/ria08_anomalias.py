from sklearn.ensemble import IsolationForest
import pandas as pd

class DetectorAnomalias:

    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)

        # 🔥 Definir columnas base (clave)
        self.feature_columns = [
            "tiempo_sesion_min",
            "intentos",
            "errores",
            "puntaje",
            "dias_inactivo"
        ]

    def train(self, df):
        # 🔥 Seleccionar SOLO columnas necesarias
        X = df[self.feature_columns]

        self.model.fit(X)

        preds = self.model.predict(X)
        self.anomaly_ratio = (preds == -1).mean()

    def predict(self, data):

        # 🔥 Asegurar columnas correctas
        for col in self.feature_columns:
            if col not in data.columns:
                data[col] = 0

        X = data[self.feature_columns]

        result = self.model.predict(X)[0]

        if result == -1:
            return "Anomalía detectada"
        return "Comportamiento normal"
