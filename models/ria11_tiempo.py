from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import pandas as pd

class PrediccionTiempo:

    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )

        # 🔥 COLUMNAS FIJAS
        self.feature_columns = [
            "intentos",
            "errores",
            "nivel_logico",
            "tasa_exito",
            "actividades_completadas"
        ]

    def train(self, df):

        # 🔥 TARGET
        df["tiempo_estimado"] = (
            df["intentos"] * 4 +
            df["errores"] * 3 +
            df["nivel_logico"] * 10 +
            (1 - df["tasa_exito"]) * 20 +
            df["actividades_completadas"] * 2
        )

        X = df[self.feature_columns]
        y = df["tiempo_estimado"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        self.r2 = r2_score(y_test, y_pred)
        self.mae = mean_absolute_error(y_test, y_pred)

        error = np.abs(y_test - y_pred)
        tolerancia = 5
        correctos = (error <= tolerancia).sum()
        self.accuracy = correctos / len(y_test)

    def predict(self, data):

        # 🔥 ASEGURAR MISMAS COLUMNAS
        for col in self.feature_columns:
            if col not in data.columns:
                data[col] = 0

        X = data[self.feature_columns]

        return self.model.predict(X)[0]
