from sklearn.ensemble import IsolationForest

class DetectorAnomalias:

    def __init__(self):
        self.model = IsolationForest(contamination=0.1)

    def train(self, df):
        X = df[[
            "tiempo_sesion_min", "intentos", "errores",
            "puntaje", "dias_inactivo"
        ]]

        self.model.fit(X)

        preds = self.model.predict(X)
        self.anomaly_ratio = (preds == -1).mean()

    def predict(self, data):
        result = self.model.predict(data)[0]

        if result == -1:
            return "Anomalía detectada"
        return "Comportamiento normal"