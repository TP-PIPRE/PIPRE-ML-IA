import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

class RecomendadorActividades:
    def __init__(self, verbose=False):  
        self.model = None
        self.scaler = StandardScaler()
        self.le_nivel = LabelEncoder()
        self.le_target = LabelEncoder()
        self.verbose = verbose  # 

        self.feature_columns = [
            "nivel_logico",
            "tasa_exito",
            "errores",
            "intentos"
        ]

    def preprocess_data(self, df, is_training=False):

        df["score_final"] = (
            df["puntaje"] * 0.6 +
            df["tasa_exito"] * 40 -
            df["errores"] * 2 +
            df["intentos"] * 1.5
        )

        df["rendimiento"] = pd.cut(
            df["score_final"],
            bins=3,
            labels=["bajo", "medio", "alto"]
        )

        if is_training:
            df["nivel_logico"] = self.le_nivel.fit_transform(df["nivel_logico"])
            df["rendimiento"] = self.le_target.fit_transform(df["rendimiento"])
        else:
            df["nivel_logico"] = self.le_nivel.transform(df["nivel_logico"])

        return df

    def train(self, df):

        df = self.preprocess_data(df, is_training=True)

        X = df[self.feature_columns]
        y = df["rendimiento"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        X_train = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=self.feature_columns
        )
        X_test = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=self.feature_columns
        )

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [10, 20],
        }

        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring="f1_weighted",
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

        y_pred = self.model.predict(X_test)

        self.accuracy = accuracy_score(y_test, y_pred)
        self.precision = precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        )

        if self.verbose:
            print("Mejores parámetros:", grid_search.best_params_)

            print("\n📊 Importancia de variables:")
            for name, val in zip(self.feature_columns, self.model.feature_importances_):
                print(f"{name}: {round(val, 3)}")

    def predict(self, data):

        for col in self.feature_columns:
            if col not in data.columns:
                data[col] = 0

        data = data[self.feature_columns]

        data = pd.DataFrame(
            self.scaler.transform(data),
            columns=self.feature_columns
        )

        pred = self.model.predict(data)[0]

        if pred == 0:
            return "Recomendar actividades básicas"
        elif pred == 1:
            return "Recomendar actividades intermedias"
        else:
            return "Recomendar actividades avanzadas"