import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class AppResultados:

    def __init__(self, root, resultados):
        self.root = root
        self.root.title("Resultados del Sistema IA")
        self.root.geometry("1000x700")  # 🔥 más grande

        self.resultados = resultados

        self.crear_interfaz()

    def crear_interfaz(self):

        titulo = tk.Label(
            self.root,
            text="Dashboard de Resultados IA",
            font=("Arial", 18, "bold")
        )
        titulo.pack(pady=10)

        notebook = ttk.Notebook(self.root)
        notebook.pack(expand=True, fill="both")

        for ria, data in self.resultados.items():
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=ria)

            self.crear_panel(frame, ria, data)

    def crear_panel(self, frame, ria, data):

        # 🔹 Resultado
        tk.Label(frame, text="Resultado:", font=("Arial", 12, "bold")).pack(pady=5)
        tk.Label(frame, text=data.get("resultado", "N/A"), font=("Arial", 12)).pack()

        # 🔹 RIA8 (solo anomalías)
        if "RIA8" in ria:
            tk.Label(frame, text="Tasa de anomalías:", font=("Arial", 12, "bold")).pack(pady=5)
            tk.Label(frame, text=data.get("anomalias", "N/A")).pack()

            # 🔥 gráfico de importancia también
            importancias = data.get("importancias", None)

            if importancias:
                self.crear_grafico_importancia(frame, importancias)

            return

        # 🔹 MÉTRICAS SIMPLES
        accuracy = data.get("accuracy", 0)
        precision = data.get("precision", 0)

        tk.Label(frame, text="Métricas:", font=("Arial", 12, "bold")).pack(pady=5)
        tk.Label(frame, text=f"Accuracy: {accuracy:.2f}").pack()
        tk.Label(frame, text=f"Precision: {precision:.2f}").pack()

        # 🔥 GRÁFICO DE IMPORTANCIA
        importancias = data.get("importancias", None)

        if importancias:
            self.crear_grafico_importancia(frame, importancias)

    def crear_grafico_importancia(self, frame, importancias):

        fig, ax = plt.subplots(figsize=(7, 4))

        # 🔥 eliminar nivel_logico si existe
        importancias_filtradas = {
            k: v for k, v in importancias.items()
            if k != "nivel_logico"
        }

        columnas = list(importancias_filtradas.keys())
        valores = list(importancias_filtradas.values())

        # 🔥 ordenar de mayor a menor
        columnas_valores = sorted(zip(valores, columnas), reverse=True)
        valores_ordenados, columnas_ordenadas = zip(*columnas_valores)

        # 🔥 gráfico horizontal (mejor para muchas variables)
        ax.barh(columnas_ordenadas, valores_ordenados)

        ax.set_title("Importancia de Variables", fontsize=12, fontweight="bold")
        ax.set_xlabel("Importancia")

        # 🔥 invertir eje para que el más importante esté arriba
        ax.invert_yaxis()

        # 🔥 grid suave
        ax.grid(axis='x', linestyle='--', alpha=0.5)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)


# 🔥 función launcher
def mostrar_resultados(resultados):
    root = tk.Tk()
    app = AppResultados(root, resultados)
    root.mainloop()