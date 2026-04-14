from sklearn.ensemble import IsolationForest
import pandas as pd

class DetectorAnomalias:

    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)

        self.feature_columns = [
            "tiempo_sesion_min",
            "intentos",
            "errores",
            "puntaje",
            "dias_inactivo"
        ]

        self.anomaly_ratio = 0

    def preprocess(self, df):
        df = df.copy()

        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df[self.feature_columns]

    def train(self, df):
        X = self.preprocess(df)

        self.model.fit(X)

        preds = self.model.predict(X)
        self.anomaly_ratio = (preds == -1).mean()
    
    def predict(self, data):
        X = self.preprocess(data)

        result = self.model.predict(X)[0]

        return "Anomalía detectada" if result == -1 else "Comportamiento normal"
    
    
    def calcular_importancia(self, df):
        X = self.preprocess(df)

        # 🔥 usar desviación estándar como proxy de importancia
        importancias = X.std().to_dict()

        # 🔥 normalizar
        total = sum(importancias.values())
        if total > 0:
            importancias = {k: v / total for k, v in importancias.items()}

        return importancias
        