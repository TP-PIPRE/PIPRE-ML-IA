import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder


class ClasificadorTiempo:

    def __init__(self, verbose=False):
        self.model = RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42
        )

        self.verbose = verbose
        self.le_nivel = LabelEncoder()
        self.le_target = LabelEncoder()

        self.accuracy = 0
        self.precision = 0
        self.recall = 0

        # 🔥 FEATURES MEJORADAS (SIN LEAK)
        self.feature_columns = [
            "intentos",
            "errores",
            "interacciones_ia",
            "uso_codigo",
            "nivel_logico",
            "ratio_error",
            "dependencia_ia",
            "actividad_total",
            "eficiencia",
            "carga_error",
            "uso_relativo"
        ]

    def construir_clases_tiempo(self, df):
        df = df.copy()

        # 🔥 mejor separación
        q1 = df["tiempo_sesion_min"].quantile(0.25)
        q2 = df["tiempo_sesion_min"].quantile(0.75)

        condiciones = [
            df["tiempo_sesion_min"] <= q1,
            (df["tiempo_sesion_min"] > q1) & (df["tiempo_sesion_min"] <= q2),
            df["tiempo_sesion_min"] > q2
        ]

        categorias = ["rapido", "normal", "lento"]

        df["categoria_tiempo"] = np.select(
            condiciones,
            categorias,
            default="normal"
        ).astype(str)

        return df

    def preprocess(self, df, is_training=False):
        df = df.copy()

        base_cols = [
            "intentos",
            "errores",
            "interacciones_ia",
            "uso_codigo",
            "nivel_logico",
            "tiempo_sesion_min"
        ]

        # 🔧 asegurar columnas
        for col in base_cols:
            if col not in df.columns:
                df[col] = 0

        # 🔧 asegurar tipo numérico
        numeric_cols = [
            "intentos",
            "errores",
            "interacciones_ia",
            "uso_codigo",
            "tiempo_sesion_min"
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # 🎯 crear target SOLO en entrenamiento
        if is_training:
            df = self.construir_clases_tiempo(df)

        # 🔥 FEATURES DERIVADAS (SIN LEAK)
        df["ratio_error"] = df["errores"] / (df["intentos"] + 1)
        df["dependencia_ia"] = df["interacciones_ia"] / (df["intentos"] + 1)

        df["actividad_total"] = df["uso_codigo"] + df["interacciones_ia"]
        df["eficiencia"] = df["intentos"] / (df["errores"] + 1)
        df["carga_error"] = df["errores"] / (df["intentos"] + 1)
        df["uso_relativo"] = df["uso_codigo"] / (df["interacciones_ia"] + 1)

        # 🔧 limpiar valores
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)

        # 🔧 encoding
        df["nivel_logico"] = df["nivel_logico"].astype(str)

        if is_training:
            df["nivel_logico"] = self.le_nivel.fit_transform(df["nivel_logico"])
            df["categoria_tiempo"] = self.le_target.fit_transform(df["categoria_tiempo"])
        else:
            df["nivel_logico"] = df["nivel_logico"].apply(
                lambda x: x if x in self.le_nivel.classes_ else self.le_nivel.classes_[0]
            )
            df["nivel_logico"] = self.le_nivel.transform(df["nivel_logico"])

        return df

    def train(self, df):
        df = self.preprocess(df, is_training=True)

        X = df[self.feature_columns]
        y = df["categoria_tiempo"]

        if self.verbose:
            print("Distribución clases:")
            print(pd.Series(y).value_counts())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        self.accuracy = accuracy_score(y_test, y_pred)
        self.precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        self.recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    def predict(self, data):
        data = self.preprocess(data, is_training=False)

        for col in self.feature_columns:
            if col not in data.columns:
                data[col] = 0

        X = data[self.feature_columns]

        pred = self.model.predict(X)[0]

        return self.le_target.inverse_transform([pred])[0]