import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class AppResultados:

    def __init__(self, root, resultados, evaluar_otro=None):
        self.root = root
        self.root.title("Resultados del Sistema IA")
        self.root.geometry("1200x800")

        self.resultados = resultados
        self.evaluar_otro = evaluar_otro  # 🔥 importante mover aquí

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

        # =========================
        # 🔥 SCROLL
        # =========================
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)

        container = tk.Frame(canvas)

        container.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=container, anchor="n")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # =========================
        # 🎯 FRAME CENTRAL (CENTRADO REAL)
        # =========================
        wrapper = tk.Frame(container)
        wrapper.pack(expand=True)

        center_frame = tk.Frame(wrapper)
        center_frame.pack()

        # =========================
        # 🔝 DATOS
        # =========================
        input_data = data.get("input_data", None)

        if input_data:
            box = tk.LabelFrame(center_frame, text="Datos evaluados", font=("Arial", 11, "bold"))
            box.pack(pady=10)

            tabla = ttk.Treeview(
                box,
                columns=("Variable", "Valor"),
                show="headings",
                height=6
            )

            tabla.heading("Variable", text="Variable")
            tabla.heading("Valor", text="Valor")

            tabla.column("Variable", width=250, anchor="center")
            tabla.column("Valor", width=150, anchor="center")

            for k, v in input_data.items():
                tabla.insert("", "end", values=(k, v))

            tabla.pack(padx=20, pady=10)

        # =========================
        # 🧠 RESULTADO
        # =========================
        box_result = tk.LabelFrame(center_frame, text="Resultado", font=("Arial", 11, "bold"))
        box_result.pack(pady=10)

        tk.Label(
            box_result,
            text=data.get("resultado", "N/A"),
            font=("Arial", 14, "bold"),
            fg="#2c3e50"
        ).pack(padx=20, pady=10)

        # =========================
        # 🔹 RIA8
        # =========================
        if "RIA8" in ria:
            box_anom = tk.LabelFrame(center_frame, text="Análisis de anomalías", font=("Arial", 11, "bold"))
            box_anom.pack(pady=10)

            tk.Label(
                box_anom,
                text=data.get("anomalias", "N/A"),
                font=("Arial", 11)
            ).pack(padx=20, pady=10)

            importancias = data.get("importancias", None)
            if importancias:
                self.crear_grafico_importancia(center_frame, importancias)

        else:
            # =========================
            # 📊 MÉTRICAS
            # =========================
            box_metricas = tk.LabelFrame(center_frame, text="Métricas", font=("Arial", 11, "bold"))
            box_metricas.pack(pady=10)

            accuracy = data.get("accuracy", 0)
            precision = data.get("precision", 0)

            frame_metrics = tk.Frame(box_metricas)
            frame_metrics.pack()

            tk.Label(
                frame_metrics,
                text=f"Accuracy: {accuracy:.2f}",
                width=20
            ).grid(row=0, column=0, padx=20, pady=5)

            tk.Label(
                frame_metrics,
                text=f"Precision: {precision:.2f}",
                width=20
            ).grid(row=0, column=1, padx=20, pady=5)

            # =========================
            # 📈 GRÁFICO
            # =========================
            importancias = data.get("importancias", None)
            if importancias:
                self.crear_grafico_importancia(center_frame, importancias)

        # =========================
        # 🔘 BOTÓN (AHORA SÍ FUNCIONA)
        # =========================
        if self.evaluar_otro:
            btn_frame = tk.Frame(center_frame)
            btn_frame.pack(pady=20)

            tk.Button(
                btn_frame,
                text="🔄 Evaluar otra fila",
                font=("Arial", 11, "bold"),
                bg="#3498db",
                fg="white",
                padx=15,
                pady=6,
                command=self.ejecutar_evaluacion
            ).pack()

    # =========================
    # 🔁 REEVALUAR
    # =========================
    def ejecutar_evaluacion(self):
        nuevos_resultados = self.evaluar_otro()

        # 🔥 limpiar la interfaz actual
        for widget in self.root.winfo_children():
            widget.destroy()

        # 🔥 actualizar datos
        self.resultados = nuevos_resultados

        # 🔥 reconstruir UI
        self.crear_interfaz()

    # =========================
    # 📈 GRÁFICO
    # =========================
    def crear_grafico_importancia(self, frame, importancias):

        fig, ax = plt.subplots(figsize=(7, 4))

        importancias_filtradas = {
            k: v for k, v in importancias.items()
            if k != "nivel_logico"
        }

        if not importancias_filtradas:
            return

        columnas = list(importancias_filtradas.keys())
        valores = list(importancias_filtradas.values())

        columnas_valores = sorted(zip(valores, columnas), reverse=True)
        valores_ordenados, columnas_ordenadas = zip(*columnas_valores)

        ax.barh(columnas_ordenadas, valores_ordenados)

        ax.set_title("Importancia de Variables", fontsize=12, fontweight="bold")
        ax.set_xlabel("Importancia")

        ax.invert_yaxis()
        ax.grid(axis='x', linestyle='--', alpha=0.5)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

        plt.close(fig)
# =========================
# 🚀 LAUNCHER
# =========================
def mostrar_resultados(resultados, evaluar_otro=None):
    root = tk.Tk()
    app = AppResultados(root, resultados, evaluar_otro)
    root.mainloop()