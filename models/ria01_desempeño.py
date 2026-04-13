import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


class ClasificadorDesempeno:

    def __init__(self, verbose=False):
        self.model = None
        self.le_nivel = LabelEncoder()
        self.le_target = LabelEncoder()
        self.verbose = verbose

        self.feature_columns = [
            "tiempo_sesion_min",
            "intentos",
            "errores",
            "nivel_logico",
            "uso_codigo",
            "interacciones_ia",
            "ratio_error",
            "intensidad_uso",
            "dependencia_ia"
        ]

    def construir_rendimiento(self, df):
        df = df.copy()

        score = (
            (df["puntaje"] * 0.5) +
            (df["tasa_exito"] * 50 * 0.5)
        )

        df["rendimiento"] = pd.cut(
            score,
            bins=[-float("inf"), 50, 75, float("inf")],
            labels=["bajo", "medio", "alto"]
        )

        df["rendimiento"] = df["rendimiento"].fillna("medio")

        return df

    def preprocess_data(self, df, is_training=False):
        df = df.copy()  # 🔥 NO modificar original

        base_cols = [
            "tiempo_sesion_min", "errores", "intentos",
            "nivel_logico", "uso_codigo", "interacciones_ia"
        ]

        # asegurar columnas base
        for col in base_cols:
            if col not in df.columns:
                df[col] = 0

        # 🔥 SOLO en entrenamiento se usa puntaje/tasa_exito
        if is_training:
            if "puntaje" not in df.columns or "tasa_exito" not in df.columns:
                raise ValueError("Faltan columnas necesarias para construir el target")

            df = self.construir_rendimiento(df)

        # 🔥 FEATURES (SIN usar puntaje ni tasa_exito)
        df["ratio_error"] = df["errores"] / (df["intentos"] + 1)
        df["intensidad_uso"] = df["tiempo_sesion_min"] / (df["intentos"] + 1)
        df["dependencia_ia"] = df["interacciones_ia"] / (df["intentos"] + 1)

        # 🔥 asegurar tipo
        df["nivel_logico"] = df["nivel_logico"].astype(str)

        # 🔥 encoding
        if is_training:
            df["nivel_logico"] = self.le_nivel.fit_transform(df["nivel_logico"])
            df["rendimiento"] = self.le_target.fit_transform(df["rendimiento"])
        else:
            df["nivel_logico"] = df["nivel_logico"].apply(
                lambda x: x if x in self.le_nivel.classes_ else self.le_nivel.classes_[0]
            )
            df["nivel_logico"] = self.le_nivel.transform(df["nivel_logico"])

        return df

    def train(self, df):
        df = self.preprocess_data(df, is_training=True)

        X = df[self.feature_columns]
        y = df["rendimiento"]

        if self.verbose:
            print("Distribución clases:")
            print(pd.Series(y).value_counts())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        # 🔥 RandomForest NO necesita scaler → eliminado
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=10,
            class_weight="balanced",
            random_state=42
        )

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        self.accuracy = accuracy_score(y_test, y_pred)
        self.precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        self.recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

        cm = confusion_matrix(y_test, y_pred)
        fn = cm.sum(axis=1) - cm.diagonal()
        self.fn_rate = fn.sum() / cm.sum()

        if self.verbose:
            print("\n📊 Importancia de variables:")
            for name, val in zip(self.feature_columns, self.model.feature_importances_):
                print(f"{name}: {round(val, 3)}")

    def predict(self, data):
        data = self.preprocess_data(data, is_training=False)

        for col in self.feature_columns:
            if col not in data.columns:
                data[col] = 0

        data = data[self.feature_columns]

        pred = self.model.predict(data)[0]

        return self.le_target.inverse_transform([pred])[0]