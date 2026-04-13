from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class EvaluadorCodigo:

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        # 🔥 Encoder
        self.le_nivel = LabelEncoder()

        # 🔥 COLUMNAS FIJAS
        self.feature_columns = [
            "errores",
            "intentos",
            "uso_codigo",
            "uso_bloques",
            "nivel_logico"
        ]

    # =========================
    # 🔥 PREPROCESS
    # =========================
    def preprocess_data(self, df, is_training=False):

        df = df.copy()

        # 🔧 asegurar columnas necesarias
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        # 🔥 asegurar tipo string
        df["nivel_logico"] = df["nivel_logico"].astype(str)

        # 🔥 encoding
        if is_training:
            df["nivel_logico"] = self.le_nivel.fit_transform(df["nivel_logico"])
        else:
            df["nivel_logico"] = df["nivel_logico"].apply(
                lambda x: x if x in self.le_nivel.classes_ else self.le_nivel.classes_[0]
            )
            df["nivel_logico"] = self.le_nivel.transform(df["nivel_logico"])

        return df

    # =========================
    # 🔥 TRAIN
    # =========================
    def train(self, df):

        df = self.preprocess_data(df, is_training=True)

        # 🔥 TARGET (sin leakage real)
        df["calidad_codigo"] = (
            df["uso_codigo"] * 2 -
            df["errores"] * 3 +
            df["intentos"] * 1.5 +
            df["nivel_logico"] * 5
        )

        df["calidad_codigo"] = pd.cut(
            df["calidad_codigo"],
            bins=3,
            labels=[0, 1, 2]
        )

        X = df[self.feature_columns]
        y = df["calidad_codigo"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        self.accuracy = accuracy_score(y_test, y_pred)
        self.precision = precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        )

    # =========================
    # 🔥 PREDICT
    # =========================
    def predict(self, data):

        data = self.preprocess_data(data, is_training=False)

        X = data[self.feature_columns]

        pred = self.model.predict(X)[0]

        # 🔥 interpretación
        if pred == 0:
            return "Código básico"
        elif pred == 1:
            return "Código intermedio"
        else:
            return "Código avanzado"