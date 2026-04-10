import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier


class ClasificadorDesempeno:

    def __init__(self, verbose=False):
        self.model = None
        self.scaler = StandardScaler()
        self.le_nivel = LabelEncoder()
        self.le_target = LabelEncoder()
        self.verbose = verbose

        # 🔥 FEATURES REALES + DERIVADAS
        self.feature_columns = [
            "tiempo_sesion_min",
            "errores",
            "intentos",
            "puntaje",
            "tasa_exito",
            "nivel_logico",
            "uso_codigo",
            "frecuencia_interaccion_ia_error",
            "eficiencia",
            "error_ratio",
            "intensidad"
        ]

    def preprocess_data(self, df, is_training=False):

        # 🔹 Validar columnas base
        base_cols = [
            "tiempo_sesion_min", "errores", "intentos", "puntaje",
            "tasa_exito", "nivel_logico", "uso_codigo",
            "frecuencia_interaccion_ia_error"
        ]

        for col in base_cols:
            if col not in df.columns:
                df[col] = 0

        # 🔥 Feature Engineering (clave)
        df["eficiencia"] = df["puntaje"] / (df["intentos"] + 1)
        df["error_ratio"] = df["errores"] / (df["intentos"] + 1)
        df["intensidad"] = df["tiempo_sesion_min"] / (df["intentos"] + 1)

        # 🔥 Encoding SOLO en entrenamiento
        if is_training:
            df["nivel_logico"] = self.le_nivel.fit_transform(df["nivel_logico"])
            df["rendimiento"] = self.le_target.fit_transform(df["rendimiento"])
        else:
            df["nivel_logico"] = self.le_nivel.transform(df["nivel_logico"])

        return df

    def train(self, df):

        df = self.preprocess_data(df, is_training=True)

        X = df[self.feature_columns]
        y = df["rendimiento"]  # 🔥 TARGET REAL

        if self.verbose:
            print("Distribución clases:")
            print(pd.Series(y).value_counts())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        # 🔥 Escalado
        X_train = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=self.feature_columns
        )
        X_test = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=self.feature_columns
        )

        # 🔥 Gradient Boosting optimizado
        param_grid = {
            "n_estimators": [150, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5]
        }

        grid_search = GridSearchCV(
            GradientBoostingClassifier(),
            param_grid,
            cv=5,
            scoring="f1_weighted",
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

        y_pred = self.model.predict(X_test)

        # 🔥 Métricas reales
        self.accuracy = accuracy_score(y_test, y_pred)
        self.precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        self.recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

        cm = confusion_matrix(y_test, y_pred)
        fn = cm.sum(axis=1) - cm.diagonal()
        self.fn_rate = fn.sum() / cm.sum()

        if self.verbose:
            print("\nMejores parámetros:", grid_search.best_params_)

            print("\n📊 Importancia de variables:")
            for name, val in zip(self.feature_columns, self.model.feature_importances_):
                print(f"{name}: {round(val, 3)}")

    def predict(self, data):

        data = self.preprocess_data(data, is_training=False)

        for col in self.feature_columns:
            if col not in data.columns:
                data[col] = 0

        data = data[self.feature_columns]

        data = pd.DataFrame(
            self.scaler.transform(data),
            columns=self.feature_columns
        )

        pred = self.model.predict(data)[0]

        # 🔥 Decodificar resultado
        return self.le_target.inverse_transform([pred])[0]
