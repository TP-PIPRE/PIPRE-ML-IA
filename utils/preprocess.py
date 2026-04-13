def preprocess_data(self, df, is_training=False):
    df = df.copy()

    # 🔧 columnas base
    base_cols = [
        "nivel_logico",
        "dias_inactivo",
        "uso_codigo",
        "interacciones_ia",
        "intentos"
    ]

    for col in base_cols:
        if col not in df.columns:
            df[col] = 0

    # 🔧 asegurar tipo numérico
    numeric_cols = [
        "dias_inactivo", "uso_codigo",
        "interacciones_ia", "intentos"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 🎯 SOLO entrenamiento
    if is_training:
        required_cols = ["puntaje", "tasa_exito", "errores", "intentos"]

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Falta columna: {col}")

        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        df["score_final"] = (
            df["puntaje"] * 0.5 +
            df["tasa_exito"] * 50 -
            df["errores"] * 2 +
            df["intentos"] * 1.5
        )

        df["rendimiento"] = pd.qcut(
            df["score_final"],
            q=3,
            labels=["bajo", "medio", "alto"],
            duplicates="drop"
        )

        # 🔥 evitar NaN en target
        df["rendimiento"] = df["rendimiento"].astype(str).fillna("medio")

    # 🔥 FEATURES ROBUSTAS
    df["ratio_ia"] = df["interacciones_ia"] / (df["intentos"] + 1)
    df["actividad_total"] = df["uso_codigo"] + df["interacciones_ia"]
    df["inactividad_relativa"] = df["dias_inactivo"] / (df["dias_inactivo"] + df["intentos"] + 1)

    # 🔧 limpiar posibles inf / NaN
    df.replace([float("inf"), -float("inf")], 0, inplace=True)
    df.fillna(0, inplace=True)

    # 🔧 encoding
    df["nivel_logico"] = df["nivel_logico"].astype(str)

    if is_training:
        df["nivel_logico"] = self.le_nivel.fit_transform(df["nivel_logico"])
        df["rendimiento"] = self.le_target.fit_transform(df["rendimiento"])
    else:
        df["nivel_logico"] = df["nivel_logico"].apply(
            lambda x: x if x in self.le_nivel.classes_ else self.le_nivel.classes_[0]
        )
        df["nivel_logico"] = self.le_nivel.transform(df["nivel_logico"])

    return df