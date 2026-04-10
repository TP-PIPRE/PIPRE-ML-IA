from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

class EvaluadorCodigo:

    def __init__(self):
        # 🔥 modelo mejorado
        self.model = RandomForestClassifier(class_weight="balanced")

    def train(self, df):
        X = df[[
            "errores", "intentos", "uso_codigo",
            "uso_bloques", "nivel_logico"
        ]]

        # 🔥 NUEVA CLASIFICACIÓN (más balanceada)
        y = df["puntaje"].apply(
            lambda x: 2 if x > 70 else (1 if x > 40 else 0)
        )

        # 🔥 split con stratify
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y
        )

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        # 🔥 métricas sin warning
        self.accuracy = accuracy_score(y_test, y_pred)
        self.precision = precision_score(
            y_test, y_pred,
            average="weighted",
            zero_division=0
        )

    def predict(self, data):
        score = self.model.predict(data)[0]

        if score == 2:
            return "Código de alta calidad"
        elif score == 1:
            return "Código aceptable"
        else:
            return "Código deficiente"