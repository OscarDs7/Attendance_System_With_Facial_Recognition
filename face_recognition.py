# ---- Sistema de Reconocimiento Facial para Registro de Asistencias ----
# ---- Basado en MediaPipe Face Mesh y SVM ----
# English version:
# ---- Facial Recognition System for Attendance Logging ----
# ---- Based on MediaPipe Face Mesh and SVM ----

import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
import pandas as pd
import time
import threading
import joblib
import tkinter as tk
import openpyxl
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from tkinter import ttk
from tkinter import messagebox
from datetime import datetime, timedelta
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- Configuration ----------------
DB_PATH = "known_faces.pkl"
SVM_PATH = "svm_model.pkl" # Ruta del modelo SVM guardado
EXCEL_PATH = "asistencias.xlsx" # Ruta del archivo Excel de asistencias
LOGO_PATH = "logo_ceti.jpg"

NUM_LANDMARKS = 420             # número de puntos de referencia
CAPTURE_SECONDS_PER_ANGLE = 4   # segundos que tarda en capturar cada ángulo
EXIT_SECONDS_AFTER_ENTRY = 30   # segundos que tarda en tomar la salida después de haber tomado la entrada
DIST_FALLBACK_THRESHOLD = 0.60  # umbral de retroceso de distancia
GROUP_OPTIONS = ["7O", "7P"]    # opciones de grupo
SUBJECT_OPTIONS = ["ML", "PDI"] # opciones de materia
SELECTED_EYE_IDX = (33, 263)    # índices de ojos izquierdo y derecho
DRAW_BOX_COLOR = (255, 0, 0)    # azul BGR (detección en angulos de registro)
YAW_THRESHOLD_DEGREES = 12      # umbral de yaw para ángulos de registro
PCA_VARIANCE = 0.90             # varianza explicada para PCA

# --- Constantes optimizadas (ajusta si quieres) ---
SMOOTH_ALPHA = 0.85            # EMA alpha (más suavizado)
CLAHE_CLIP = 3.5            # parámetros CLAHE
CLAHE_GRID = (8, 8) 

PCA_FIXED_COMPONENTS = 40      # componentes fijos PCA (si se desea usar en lugar de varianza)
SVM_PROB_THRESHOLD = 0.40      # probabilidad mínima SVM para aceptar predicción
CONFIRM_FRAMES = 4             # cuántos frames concordantes para confirmar identidad
CONFIRM_RATIO = 0.66           # ratio mínimo de votos iguales dentro del buffer
RESET_STATE_SECONDS = 30       # si no se ve a la persona en este tiempo, reiniciar estado
MORE_SAMPLES_ON_REGISTER = True
MORE_SAMPLES_COUNT = 20        # si se usa capture_more_samples (mejora reentreno)

# ---------------- Configuración MediaPipe ----------------
mp_face_mesh = mp.solutions.face_mesh # Módulo de MediaPipe Face Mesh7
# ---------------- Global timers ----------------
timers = {}       # Timers para cada persona (entrada/salida)          

# ---------------- Diálogo no bloqueante para datos del alumno ----------------
class NonBlockingDialog:
    def __init__(self, title="Dialog", ask_name=True, default_group=None, callback=None):
        self.title = title
        self.ask_name = ask_name
        self.default_group = default_group
        self.callback = callback
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            root = tk.Tk()
            root.title(self.title)
            root.resizable(False, False)
            pad = 8
            r = 0
            if self.ask_name:
                tk.Label(root, text="Nombre (completo):").grid(row=r, column=0, padx=pad, pady=6, sticky="w")
                name_entry = tk.Entry(root, width=30)
                name_entry.grid(row=r, column=1, padx=pad, pady=6)
                r += 1
                tk.Label(root, text="Registro / Matrícula:").grid(row=r, column=0, padx=pad, pady=6, sticky="w")
                reg_entry = tk.Entry(root, width=30)
                reg_entry.grid(row=r, column=1, padx=pad, pady=6)
                r += 1
            else:
                name_entry = None
                reg_entry = None

            tk.Label(root, text="Grupo:").grid(row=r, column=0, padx=pad, pady=6, sticky="w")
            combo_group = ttk.Combobox(root, values=GROUP_OPTIONS, state="readonly", width=27)
            combo_group.grid(row=r, column=1, padx=pad, pady=6)
            if self.default_group and self.default_group in GROUP_OPTIONS:
                combo_group.set(self.default_group)
            else:
                combo_group.current(0)
            r += 1

            tk.Label(root, text="Materia:").grid(row=r, column=0, padx=pad, pady=6, sticky="w")
            combo_subject = ttk.Combobox(root, values=SUBJECT_OPTIONS, state="readonly", width=27)
            combo_subject.grid(row=r, column=1, padx=pad, pady=6)
            combo_subject.current(0)
            r += 1

            def on_ok():
                nm = name_entry.get().strip() if name_entry else None
                reg = reg_entry.get().strip() if reg_entry else None
                grp = combo_group.get().strip()
                subj = combo_subject.get().strip()
                try:
                    root.destroy()
                except:
                    pass
                if self.callback:
                    try:
                        self.callback(nm, reg, grp, subj)
                    except Exception as e:
                        print("Callback error in NonBlockingDialog:", e)

            btn = tk.Button(root, text="Aceptar", command=on_ok, width=12)
            btn.grid(row=r, column=0, columnspan=2, pady=10)
            root.eval('tk::PlaceWindow %s center' % root.winfo_toplevel())
            root.mainloop()
        except Exception as e:
            print("Error launching dialog:", e)
            if self.callback:
                try:
                    self.callback(None, GROUP_OPTIONS[0], SUBJECT_OPTIONS[0])
                except:
                    pass
# fin-NonBlockingDialog

# ---------------- Cargar BD Pickle ----------------
def load_database(path=DB_PATH):
    """Carga la DB de pickle de forma segura; si está corrupta devuelve {}."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    try:
        with open(path, "rb") as f:
            db = pickle.load(f)
        # normalizar embeddings a numpy arrays (proteger corruptos)
        if not isinstance(db, dict):
            print("[DB] Formato inesperado en pickle, inicializando DB vacía.")
            return {}
        for k, v in list(db.items()):
            if isinstance(v, dict):
                samples = v.get("samples")
                if isinstance(samples, dict):
                    for ang, s in samples.items():
                        if s is not None:
                            try:
                                samples[ang] = np.array(s, dtype=float)
                            except Exception:
                                samples[ang] = None
                    db[k]["samples"] = samples
                else:
                    # formato antiguo: el valor completo es un vector -> lo convertimos en samples
                    try:
                        emb = np.array(v, dtype=float)
                        db[k] = {"group": "-", "subject": "-", "samples": {"frontal": emb, "derecha": None, "izquierda": None}}
                    except Exception:
                        print(f"[DB] Registro inválido para clave {k}, se elimina.")
                        db.pop(k, None)
            else:
                # entrada no dict; intentar convertir
                try:
                    emb = np.array(v, dtype=float)
                    db[k] = {"group": "-", "subject": "-", "samples": {"frontal": emb, "derecha": None, "izquierda": None}}
                except Exception:
                    print(f"[DB] Entrada inválida para clave {k}, se elimina.")
                    db.pop(k, None)
        return db
    except Exception as e:
        print(f"[DB] Error al leer '{path}': {e} — se inicializa nueva DB.")
        return {}
# fin load_database

# ---------------- Guardar BD Pickle ----------------
def save_database(db, path=DB_PATH):
    """Guarda db (pickle). Convertir numpy -> listas para evitar problemas con versiones."""
    safe_db = {}
    for name, info in db.items():
        safe_info = dict(info)
        samples = safe_info.get("samples", {}) or {}
        safe_samples = {}
        for ang, vec in samples.items():
            if vec is None:
                safe_samples[ang] = None
            else:
                try:
                    safe_samples[ang] = np.array(vec).tolist()
                except Exception:
                    safe_samples[ang] = None
        safe_info["samples"] = safe_samples
        safe_db[name] = safe_info
    try:
        with open(path, "wb") as f:
            pickle.dump(safe_db, f)
    except Exception as e:
        print(f"[DB] Error guardando DB en '{path}': {e}")
# fin-save_database

# ---------------- Encabezado personalizado por grupo para exportar asistencia a PDF --------------
CUSTOM_HEADERS = {
    "7O": {
        "escuela": "CETI Colomos",
        "grado_grupo": "7O",
        "materia": "Machine Learning",
        "turno": "Matutino",
        "clave_materia": "19SDSIA02",
        "docente": "Dr. C. Gerardo Gil García"
    },
    "7P": {
        "escuela": "CETI Colomos",
        "grado_grupo": "7P",
        "materia": "Procesamiento Digital de Imágenes",
        "turno": "Matutino",
        "clave_materia": "19SDS32",
        "docente": "Dr. C. Gerardo Gil García"
    }
}

def exportar_pdf_grupo(grupo: str, ruta_excel="asistencias.xlsx"):
    """
    Exporta a PDF el listado filtrado por grupo desde un archivo Excel.
    El PDF se genera con el nombre: Reporte_{grupo}_{YYYY-MM-DD}.pdf
    """

    grupo = grupo.upper()  # Normalizar

    # -------------------------------
    # Seleccionar encabezado correcto
    # -------------------------------
    if grupo not in CUSTOM_HEADERS:
        print(f"[ERROR] No hay encabezados definidos para el grupo '{grupo}'.")
        print("Grupos disponibles:", ", ".join(CUSTOM_HEADERS.keys()))
        return

    encabezado = CUSTOM_HEADERS[grupo]

    # -------------------------------
    # Cargar Excel
    # -------------------------------
    try:
        dfs = pd.read_excel(ruta_excel, sheet_name=None)  # Carga TODAS las hojas
        df = pd.concat(dfs.values(), ignore_index=True)   # Une todas en un solo DataFrame

    except Exception as e:
        print(f"[ERROR] No se puede abrir el archivo Excel: {e}")
        return

    if "Grupo" not in df.columns:
        print("[ERROR] La tabla no contiene la columna 'Grupo'.")
        return

    # -------------------------------
    # Filtrar registros del grupo
    # -------------------------------
    df_grupo = df[df["Grupo"].str.upper() == grupo]

    if df_grupo.empty:
        print(f"[ADVERTENCIA] No hay registros para el grupo '{grupo}'.")
        return

    # -------------------------------
    # Construir nombre del PDF
    # -------------------------------
    fecha = datetime.now().strftime("%Y-%m-%d")
    nombre_pdf = f"Reporte_{grupo}_{fecha}.pdf"

    print(f"[INFO] Generando archivo: {nombre_pdf}")

    # -------------------------------
    # Crear documento PDF
    # -------------------------------
    doc = SimpleDocTemplate(nombre_pdf, pagesize=letter)

    elementos = []
    estilos = getSampleStyleSheet()

    # Logo
    try:
        img = Image(LOGO_PATH, width=90, height=90)
        elementos.append(img)
        elementos.append(Spacer(1, 12))
    except Exception as e:
        print(f"[WARN] No se pudo cargar el logo: {e}")

    # Título
    titulo = Paragraph(f"<b>Reporte de Asistencias - Grupo {grupo}</b>", estilos["Title"])
    elementos.append(titulo)
    elementos.append(Spacer(1, 12))

    # Encabezados personalizados según grupo
    for k, v in encabezado.items():
        texto = Paragraph(f"<b>{k.replace('_',' ').title()}:</b> {v}", estilos["Normal"])
        elementos.append(texto)

    elementos.append(Spacer(1, 18))

    # Convertir a tabla
    tabla_datos = [df_grupo.columns.tolist()] + df_grupo.astype(str).values.tolist()

    tabla = Table(tabla_datos)
    tabla.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))

    elementos.append(tabla)

    # Finalizar PDF
    doc.build(elementos)

    print(f"[OK] PDF creado correctamente: {nombre_pdf}\n")
# fin exportar_pdf_grupo

def exportar_pdf_grupo_materia(grupo: str, materia: str, ruta_excel="asistencias.xlsx"):
    """
    Exporta a PDF el listado filtrado primero por grupo y luego por materia.
    Nombre del PDF: Reporte_{grupo}_{materia}_{YYYY-MM-DD}.pdf
    """

    grupo = grupo.upper().strip()
    materia = materia.upper().strip()

    # -------------------------------
    # Verificar encabezados por grupo
    # -------------------------------
    if grupo not in CUSTOM_HEADERS:
        print(f"[ERROR] No hay encabezados definidos para el grupo '{grupo}'.")
        print("Grupos disponibles:", ", ".join(CUSTOM_HEADERS.keys()))
        return

    encabezado = CUSTOM_HEADERS[grupo]

    # -------------------------------
    # Cargar Excel
    # -------------------------------
    try:
        dfs = pd.read_excel(ruta_excel, sheet_name=None)
        df = pd.concat(dfs.values(), ignore_index=True)
    except Exception as e:
        print(f"[ERROR] No se puede abrir el archivo Excel: {e}")
        return

    # -------------------------------
    # Validar columnas requeridas
    # -------------------------------
    if "Grupo" not in df.columns or "Materia" not in df.columns:
        print("[ERROR] La tabla no contiene las columnas necesarias ('Grupo', 'Materia').")
        return

    # -------------------------------
    # Filtro 1: Por grupo
    # -------------------------------
    df_filtrado = df[df["Grupo"].str.upper() == grupo]

    if df_filtrado.empty:
        print(f"[ADVERTENCIA] No hay registros para el grupo '{grupo}'.")
        return

    # -------------------------------
    # Filtro 2: Por materia
    # -------------------------------
    df_filtrado = df_filtrado[df_filtrado["Materia"].str.upper() == materia]

    if df_filtrado.empty:
        print(f"[ADVERTENCIA] No hay registros para la materia '{materia}' en el grupo '{grupo}'.")
        return

    # -------------------------------
    # Nombre del PDF
    # -------------------------------
    fecha = datetime.now().strftime("%Y-%m-%d")
    nombre_pdf = f"Reporte_{grupo}_{materia}_{fecha}.pdf"

    print(f"[INFO] Generando archivo: {nombre_pdf}")

    # -------------------------------
    # Crear documento PDF
    # -------------------------------
    doc = SimpleDocTemplate(nombre_pdf, pagesize=letter)
    elementos = []
    estilos = getSampleStyleSheet()

    # Logo
    try:
        img = Image(LOGO_PATH, width=90, height=90)
        elementos.append(img)
        elementos.append(Spacer(1, 12))
    except Exception as e:
        print(f"[WARN] No se pudo cargar el logo: {e}")

    # Título
    titulo = Paragraph(f"<b>Reporte de Asistencias</b>", estilos["Title"])
    subtitulo = Paragraph(f"<b>Grupo:</b> {grupo} &nbsp;&nbsp;&nbsp; <b>Materia:</b> {materia}", estilos["Heading2"])
    elementos.append(titulo)
    elementos.append(subtitulo)
    elementos.append(Spacer(1, 18))

    # Encabezados personalizados
    for k, v in encabezado.items():
        texto = Paragraph(f"<b>{k.replace('_',' ').title()}:</b> {v}", estilos["Normal"])
        elementos.append(texto)

    elementos.append(Spacer(1, 12))

    # Tabla
    tabla_datos = [df_filtrado.columns.tolist()] + df_filtrado.astype(str).values.tolist()
    tabla = Table(tabla_datos)

    tabla.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))

    elementos.append(tabla)
    doc.build(elementos)

    print(f"[OK] PDF creado correctamente: {nombre_pdf}\n")
# fin exportar_pdf_grupo_materia

def exportar_pdf_grupo_materia_fecha(grupo: str, materia: str, fecha: str = "", ruta_excel="asistencias.xlsx"):
    """
    Exporta a PDF filtrando por:
      - Grupo
      - Materia
      - Fecha (si está en blanco usa la fecha actual)
    
    Nombre PDF: Reporte_{grupo}_{materia}_{fecha}.pdf
    """

    grupo = grupo.upper().strip()
    materia = materia.upper().strip()

    # Si no se dio fecha → usar fecha actual
    if fecha.strip() == "":
        fecha = datetime.now().strftime("%Y-%m-%d")

    # Normalizar fecha
    fecha = fecha.strip()

    # -------------------------------
    # Verificar encabezados por grupo
    # -------------------------------
    if grupo not in CUSTOM_HEADERS:
        print(f"[ERROR] No hay encabezados definidos para el grupo '{grupo}'.")
        print("Grupos disponibles:", ", ".join(CUSTOM_HEADERS.keys()))
        return

    encabezado = CUSTOM_HEADERS[grupo]

    # -------------------------------
    # Cargar Excel
    # -------------------------------
    try:
        dfs = pd.read_excel(ruta_excel, sheet_name=None)
        df = pd.concat(dfs.values(), ignore_index=True)
    except Exception as e:
        print(f"[ERROR] No se puede abrir el archivo Excel: {e}")
        return

    # -------------------------------
    # Validar columnas requeridas
    # -------------------------------
    for col in ["Grupo", "Materia", "Fecha"]:
        if col not in df.columns:
            print(f"[ERROR] La tabla no contiene la columna '{col}'.")
            return

    # -------------------------------
    # Filtro 1: Por grupo
    # -------------------------------
    df_f = df[df["Grupo"].str.upper() == grupo]
    if df_f.empty:
        print(f"[ADVERTENCIA] No hay registros para el grupo '{grupo}'.")
        return

    # -------------------------------
    # Filtro 2: Por materia
    # -------------------------------
    df_f = df_f[df_f["Materia"].str.upper() == materia]
    if df_f.empty:
        print(f"[ADVERTENCIA] No hay registros en '{materia}' para el grupo '{grupo}'.")
        return

    # -------------------------------
    # Filtro 3: Por fecha
    # -------------------------------
    df_f = df_f[df_f["Fecha"].astype(str) == fecha]
    if df_f.empty:
        print(f"[ADVERTENCIA] No hay registros para la fecha '{fecha}' en el grupo {grupo} - materia {materia}.")
        return

    # -------------------------------
    # Crear nombre PDF
    # -------------------------------
    nombre_pdf = f"Reporte_{grupo}_{materia}_{fecha}.pdf"
    print(f"[INFO] Generando: {nombre_pdf}")

    # -------------------------------
    # Crear documento PDF
    # -------------------------------
    doc = SimpleDocTemplate(nombre_pdf, pagesize=letter)
    elementos = []
    estilos = getSampleStyleSheet()

    # Logo
    try:
        img = Image(LOGO_PATH, width=90, height=90)
        elementos.append(img)
        elementos.append(Spacer(1, 12))
    except:
        pass

    # Título
    titulo = Paragraph(f"<b>Reporte de Asistencias</b>", estilos["Title"])
    subtitulo = Paragraph(
        f"<b>Grupo:</b> {grupo} &nbsp;&nbsp; "
        f"<b>Materia:</b> {materia} &nbsp;&nbsp; "
        f"<b>Fecha:</b> {fecha}",
        estilos["Heading2"]
    )

    elementos.append(titulo)
    elementos.append(subtitulo)
    elementos.append(Spacer(1, 16))

    # Encabezados personalizados
    for k, v in encabezado.items():
        txt = Paragraph(f"<b>{k.replace('_',' ').title()}:</b> {v}", estilos["Normal"])
        elementos.append(txt)

    elementos.append(Spacer(1, 16))

    # Tabla
    tabla_datos = [df_f.columns.tolist()] + df_f.astype(str).values.tolist()
    tabla = Table(tabla_datos)
    tabla.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))

    elementos.append(tabla)
    doc.build(elementos)

    print(f"[OK] PDF creado correctamente: {nombre_pdf}\n")
# fin exportar_pdf_grupo_materia_fecha

# ---------------- Función para asegurar existencia de archivo Excel ----------------
def ensure_excel_exists(path=EXCEL_PATH):
    if not os.path.exists(path):
        try:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                for g in GROUP_OPTIONS:
                    df = pd.DataFrame(columns=["Fecha", "Alumno", "Registro", "Grupo", "Materia", "Entrada", "Salida"])
                    df.to_excel(writer, sheet_name=g, index=False)
        except Exception as e:
            print(f"[EXCEL] Error creando {path}: {e}")

# ---------------- Función para leer hoja de grupo desde Excel ----------------
def read_group_sheet(group, path=EXCEL_PATH):
    ensure_excel_exists(path)
    try:
        df = pd.read_excel(path, sheet_name=group)
    except Exception:
        df = pd.DataFrame(columns=["Fecha", "Alumno", "Registro", "Grupo", "Materia", "Entrada", "Salida"])
    return df

# ---------------- Función para guardar hoja de grupo en Excel ----------------
def write_group_sheet(df, group, path=EXCEL_PATH):
    try:
        # Si el archivo existe, cargar todas las hojas
        if os.path.exists(path):
            with pd.ExcelWriter(path.replace(".xlsx", "_tmp.xlsx"), engine="openpyxl") as writer:
                try:
                    existing = pd.read_excel(path, sheet_name=None)
                except:
                    existing = {}

                # Reescribir cada hoja, reemplazando solo la del grupo
                for sheet_name, old_df in existing.items():
                    if sheet_name == group:
                        df.to_excel(writer, sheet_name=group, index=False)
                    else:
                        old_df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            # Crear archivo nuevo con todas las hojas
            with pd.ExcelWriter(path.replace(".xlsx", "_tmp.xlsx"), engine="openpyxl") as writer:
                for g in GROUP_OPTIONS:
                    if g == group:
                        df.to_excel(writer, sheet_name=g, index=False)
                    else:
                        empty = pd.DataFrame(columns=["Fecha","Alumno","Registro","Grupo","Materia","Entrada","Salida"])
                        empty.to_excel(writer, sheet_name=g, index=False)

        # Reemplazar archivo original
        os.replace(path.replace(".xlsx", "_tmp.xlsx"), path)

    except Exception as e:
        print(f"[EXCEL] Error guardando sheet {group}: {e}")
 # fin write_group_sheet

# # ---------------- Mensaje emergente de estado de asistencia del alumno ----------------
def popup_info(texto):
    messagebox.showinfo("Asistencia", texto)

# ---------------- Añadir o actualizar asistencia en Excel ----------------
def add_or_update_attendance(person_name, registro, group, subject, tipo, path=EXCEL_PATH):
    """
    Actualiza o crea una fila de asistencia según alumno + materia.
    Controla entradas duplicadas, salidas duplicadas y salidas sin entrada.
    Devuelve códigos estándar para registrar_entrada() y registrar_salida().
    """

    ensure_excel_exists(path)
    df = read_group_sheet(group, path)

    hoy = datetime.now().strftime("%Y-%m-%d")
    hora_actual = datetime.now().strftime("%H:%M:%S")

    # Asegurar columnas correctas
    expected_cols = ["Fecha", "Alumno", "Registro", "Grupo", "Materia", "Entrada", "Salida"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = "-"

    # Normalización
    person_name = str(person_name).strip()
    registro = str(registro).strip()
    subject = str(subject).strip()

    # Buscar registros del día + alumno + materia
    mask = (
        (df["Fecha"] == hoy) &
        (df["Alumno"] == person_name) &
        (df["Materia"] == subject)
    )
    rows = df[mask]

    # -------------------------------------------------------------
    #                           ENTRADA
    # -------------------------------------------------------------
    if tipo == "entrada":

        # Ya existe entrada → no duplicar
        if not rows.empty and any(rows["Entrada"] != "-"):
            print(f"[Asistencia] Entrada YA existe para {person_name} en {subject}.")
            popup_info("¡La entrada ya fue registrada!")
            return "entrada_duplicada"

        # Crear nueva fila de entrada
        new_row = {
            "Fecha": hoy,
            "Alumno": person_name,
            "Registro": registro,
            "Grupo": group,
            "Materia": subject,
            "Entrada": hora_actual,
            "Salida": "-",
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        write_group_sheet(df, group, path)

        print(f"[ENTRY] Entrada registrada para {person_name} ({subject})")
        popup_info(f"Entrada registrada\nHora: {hora_actual}")
        return "entrada_ok"

    # -------------------------------------------------------------
    #                           SALIDA
    # -------------------------------------------------------------
    elif tipo == "salida":

        if rows.empty:
            print(f"[WARN] No hay entrada previa para {person_name} en {subject}.")
            popup_info(f"No puede registrar salida sin entrada previa en {subject}.")
            return "sin_entrada"

        # Buscar fila con entrada registrada pero sin salida
        pendiente = rows[(rows["Entrada"] != "-") & (rows["Salida"] == "-")]

        if pendiente.empty:
            print(f"[Asistencia] Salida YA existe para {person_name} en {subject}.")
            popup_info("Asistencia del día ya está completa.")
            return "salida_duplicada"

        # Registrar la salida
        idx = pendiente.index[0] # tomar la primera fila pendiente
        df.at[idx, "Salida"] = hora_actual # actualizar salida
        write_group_sheet(df, group, path)

        print(f"[EXIT] Salida registrada para {person_name} ({subject})")
        popup_info(f"Salida registrada\nHora: {hora_actual}")
        return "salida_ok"

    # -------------------------------------------------------------
    #                        ERROR DE TIPO
    # -------------------------------------------------------------
    else:
        print(f"[ERROR] Tipo inválido en add_or_update_attendance(): {tipo}")
        return "error"  
    # fin add_or_update_attendance
# -----------------------------------------------------------------------------

def registrar_entrada(label, db, state):
    alumno = db.get(label, {})
    registro = alumno.get("registro", "-")
    grupo = alumno.get("group", GROUP_OPTIONS[0])
    
    # Materia activa
    materia = (
        state.get(label, {}).get("subject_session") or 
        alumno.get("subject", "-")
    )

    # Registrar asistencia
    res = add_or_update_attendance(
        person_name=label,
        registro=registro,
        group=grupo,
        subject=materia,
        tipo="entrada"
    )

    # -------- Manejo de resultados --------
    if res == "entrada_duplicada":
        return

    if res != "entrada_ok":
        print(f"[registrar_entrada] Resultado inesperado: {res}")
        return

    # -------- Entrada válida --------
    st = state.setdefault(label, {})

    # Reseteo correcto del ciclo entrada-salida
    st["entry_marked"] = True
    st["exit_marked"] = False        # ← MUY IMPORTANTE (evita entradas dobles)

    st["entry_time"] = time.time()
    st["last_subject"] = materia
    state[label] = st

    # Timer
    timers[label] = {
        "state": "entrada",
        "start_time": time.time()
    }

    print(f"[registrar_entrada] Timer iniciado para {label}")

    # fin registrar_entrada


def registrar_salida(label, db, state):
    alumno = db.get(label, {})
    registro = alumno.get("registro", "-")
    grupo = alumno.get("group", GROUP_OPTIONS[0])

    # Recuperar materia correctamente
    materia = (
        state.get(label, {}).get("subject_session")
        or state.get(label, {}).get("last_subject")
        or alumno.get("subject", "-")
    )

    # No permitir salida sin entrada previa
    if not state.get(label, {}).get("entry_marked", False):
        popup_info("No puede registrar salida sin tener una entrada previa")
        return

    # Registrar asistencia
    add_or_update_attendance(
        person_name=label,
        registro=registro,
        group=grupo,
        subject=materia,
        tipo="salida"
    )

    # Actualizar estado interno
    st = state.setdefault(label, {})
    st["exit_marked"] = True
    st["entry_marked"] = False      # ← IMPORTANTE (evita salidas dobles)
    st["last_subject"] = materia    # ← Mantiene la materia activa
    state[label] = st

    # Timer
    timers[label] = {
        "state": "salida",
        "start_time": time.time()
    }
    print(f"[registrar_salida] Timer iniciado para {label}")

    # fin registrar_salida

# --------------------------------------------------------------------------------

# ---------------- Selección de puntos de referencia ----------------
def build_selected_indices(num=NUM_LANDMARKS, total=468, include=SELECTED_EYE_IDX):
    idxs = list(np.linspace(0, total-1, num, dtype=int))
    for e in include:
        if 0 <= e < total and e not in idxs:
            idxs.append(e)
    idxs = sorted(set(idxs))
    return idxs
SELECTED_IDX = build_selected_indices()

# ---------------- Extracción de puntos de referencia ----------------
def extract_selected_landmarks(face_landmarks, selected_idx=SELECTED_IDX):
    coords = []
    for i in selected_idx:
        try:
            lm = face_landmarks.landmark[i]
            coords.append(float(lm.x))
            coords.append(float(lm.y))
        except Exception:
            # si falla algún índice, rellenar con ceros (defensa)
            coords.append(0.0)
            coords.append(0.0)
    return np.array(coords, dtype=float)

# ---------------- Normalización y ajuste de longitud en embeddings ----------------
def normalize_vector(vec):
    vec = np.array(vec, dtype=float)
    x = vec[::2].copy()
    y = vec[1::2].copy()
    try:
        pos33 = SELECTED_IDX.index(SELECTED_EYE_IDX[0])
        pos263 = SELECTED_IDX.index(SELECTED_EYE_IDX[1])
        eye_x0, eye_y0 = x[pos33], y[pos33]
        eye_x1, eye_y1 = x[pos263], y[pos263]
        cx = (eye_x0 + eye_x1) / 2.0
        cy = (eye_y0 + eye_y1) / 2.0
        eye_dist = np.sqrt((eye_x1 - eye_x0)**2 + (eye_y1 - eye_y0)**2)
    except Exception:
        cx = np.mean(x)
        cy = np.mean(y)
        eye_dist = np.std(x) + 1e-6
    x = x - cx
    y = y - cy
    if eye_dist > 0:
        x = x / (eye_dist + 1e-12)
        y = y / (eye_dist + 1e-12)
    return np.concatenate([x, y])

def fix_length(vec, target_len):
    vec = np.array(vec, dtype=float)
    if vec.size == target_len:
        return vec
    if vec.size < target_len:
        pad = np.zeros(target_len - vec.size, dtype=float)
        return np.concatenate([vec, pad])
    return vec[:target_len]

# ---------------- Estimación de yaw (giro horizontal) ----------------
def estimate_yaw_deg(face_landmarks):
    try:
        left_eye_x = face_landmarks.landmark[SELECTED_EYE_IDX[0]].x
        right_eye_x = face_landmarks.landmark[SELECTED_EYE_IDX[1]].x
        nose_x = face_landmarks.landmark[1].x
    except Exception:
        return 0.0
    eye_center_x = (left_eye_x + right_eye_x) / 2.0
    eye_dist = abs(right_eye_x - left_eye_x) + 1e-6
    dx = nose_x - eye_center_x
    yaw_rad = np.arctan2(dx, eye_dist)
    yaw_deg = np.degrees(yaw_rad)
    return yaw_deg

# ---------------- Construcción de matriz de entrenamiento desde BD Pickle ----------------
def build_training_matrix_from_db(db):
    X = []
    y = []
    for name, info in db.items():
        samples = info.get("samples", {}) or {}
        vecs = []
        for ang in ("frontal", "derecha", "izquierda"):
            v = samples.get(ang)
            if v is not None:
                try:
                    va = np.array(v, dtype=float)
                    va = fix_length(va, len(va))  # aseguramos array numpy
                    vecs.append(va)
                except Exception:
                    pass
        if len(vecs) == 0:
            continue
        avg = np.mean(np.stack(vecs), axis=0)
        X.append(avg)
        y.append(name)
    if len(X) == 0:
        return np.array([]), np.array([])
    return np.vstack(X), np.array(y)

# ---------------- Entrenamiento y guardado del modelo SVM + PCA ----------------
def train_and_save_model(db, svm_path=SVM_PATH):
    X, y = build_training_matrix_from_db(db)
    if X.size == 0 or len(y) < 2:
        print("[MODEL] No hay suficientes clases/muestras para entrenar.")
        return None, None, None
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # decidir componentes PCA de forma segura
    n_samples, n_feats = Xs.shape
    max_components = min(n_feats, max(1, n_samples - 1))
    # PCA temporal para decidir número por varianza, solo si hay al menos 2 muestras
    if n_samples < 2:
        print("[MODEL] Muy pocas muestras para PCA, se omite PCA.")
        pca = None
        Xp = Xs
    else:
        pca_tmp = PCA(n_components=max_components, svd_solver='full')
        pca_tmp.fit(Xs)
        cumvar = np.cumsum(pca_tmp.explained_variance_ratio_)
        n_comp = np.searchsorted(cumvar, PCA_VARIANCE) + 1
        n_comp = max(1, n_comp)
        # regla: intentar mínimo 20 si es posible
        if max_components >= 20:
            n_comp = max(20, n_comp)
        n_comp = min(n_comp, max_components)
        pca = PCA(n_components=n_comp)
        Xp = pca.fit_transform(Xs)

    clf = SVC(kernel="rbf", probability=True, class_weight='balanced', gamma='scale', C=1.0)
    clf.fit(Xp, y)

    try:
        joblib.dump((clf, scaler, pca), svm_path)
    except Exception as e:
        print("[MODEL] Error guardando modelo:", e)

    print(f"[MODEL] Entrenado SVM + PCA({'NO PCA' if pca is None else pca.n_components_}) y guardado en {svm_path}")
    return clf, scaler, pca

# ---------------- Carga del modelo SVM + PCA ----------------
def load_model(svm_path=SVM_PATH):
    if os.path.exists(svm_path):
        try:
            clf, scaler, pca = joblib.load(svm_path)
            return clf, scaler, pca
        except Exception as e:
            print("[MODEL] Error cargando modelo:", e)
            return None, None, None
    return None, None, None

# ---------------- Evaluación del modelo sobre la BD ----------------
def evaluate_model_on_db(clf, scaler, pca, db):
    """
    Evalúa el modelo sobre las muestras usadas para entrenar.
    Retorna: accuracy, número de muestras y dict de {clase: cantidad}
    """
    try:
        X, y = build_training_matrix_from_db(db) # matriz de entrenamiento
        if X.size == 0 or len(y) == 0:
            return 0.0, 0, {}

        Xs = scaler.transform(X) # normalización

        if pca is not None:
            Xp = pca.transform(Xs)
        else:
            Xp = Xs

        preds = clf.predict(Xp) # predicciones
        acc = (preds == y).sum() / len(y) # accuracy

        per_class = {}
        # contar por clase
        for label in y:
            per_class[label] = per_class.get(label, 0) + 1 

        return float(acc), len(y), per_class

    except Exception as e:
        print("[EVAL] ERROR:", e)
        return 0.0, 0, {}

# ---------------- Fallback distance match (versión robusta) ----------------
def fallback_match(vec_norm, db, threshold=DIST_FALLBACK_THRESHOLD):
    """
    Compara el embedding normalizado contra la BD sin PCA (distancia cruda).
    Esta versión es segura y evita errores cuando:
      - Hay vectores de diferente longitud
      - Algún ángulo está en None
      - Los vectores contienen NaN
      - Se intenta convertir algo que no es array
    """

    if vec_norm is None:
        return None, None

    best = None
    best_d = float("inf")
    target_len = len(vec_norm)

    # Iteramos por cada persona y sus muestras de ángulos
    for name, info in db.items():
        samples = info.get("samples", {}) or {}

        for ang, s in samples.items():
            if s is None:
                continue

            # ---- Convertir a array de forma segura ----
            try:
                s_arr = np.array(s, dtype=float)
            except Exception:
                continue

            # ---- Evitar vectores vacíos ----
            if s_arr.size == 0:
                continue

            # ---- Empatar longitud si difieren ----
            if s_arr.size != target_len:
                m = min(s_arr.size, target_len)
                v1 = vec_norm[:m]
                v2 = s_arr[:m]
            else:
                v1 = vec_norm
                v2 = s_arr

            # ---- Evitar vectores dañados ----
            if np.isnan(v1).any() or np.isnan(v2).any():
                continue

            # ---- Distancia euclidiana ----
            try:
                d = np.linalg.norm(v1 - v2)
            except Exception:
                continue

            # ---- Guardar mejor coincidencia ----
            if d < best_d:
                best_d = d
                best = name

    # ---- Validar umbral ----
    if best is not None and best_d <= threshold:
        similarity = 1 - min(best_d, 1)   # normaliza 0–1
        return best, similarity

    return None, None

# ---------------- UI helper: draw label robusto ----------------
def draw_detection_label(frame, bbox, name, registro="-", score=None):
    """
    Dibuja una barra fija en la parte inferior y centrada con el nombre y registro.
    No se relaciona con el bounding box; es global a la ventana.
    """
    h, w = frame.shape[:2]

    # -------- Texto --------
    if name == "Desconocido":
        text = "Desconocido - presiona 'n' para registrar"
        color = (0, 255, 255)
    else:
        if score is not None:
            text = f"{name} | {registro} ({score*100:.1f}%)"
        else:
            text = f"{name} | {registro}"
        color = (255, 255, 255)

    # Medidas del texto
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

    # -------- Posición centrada en el borde inferior --------
    box_width = text_w + 30
    box_height = text_h + 20

    x1 = (w - box_width) // 2
    y1 = h - box_height - 10   # 10px arriba del borde inferior
    x2 = x1 + box_width
    y2 = y1 + box_height

    # -------- Dibujar fondo con transparencia --------
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # -------- Dibujar texto centrado --------
    text_x = x1 + (box_width - text_w) // 2
    text_y = y1 + (box_height + text_h) // 2 - 5

    cv2.putText(
        frame, text, 
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    return frame

    # fin del draw_detection_label

# ---------------- Capture 3 angles new person (with yaw validation + retries) ----------------
def capture_three_angles_new_person(name, cap, face_mesh, seconds=CAPTURE_SECONDS_PER_ANGLE, max_retries_per_angle=2):
    win = "Registro 3 ángulos"
    angles = ["frontal", "derecha", "izquierda"]
    collected = {}
    try:
        for angle in angles:
            retries = 0
            while retries <= max_retries_per_angle:
                print(f"[Registro] Prepárate: {angle} - presiona 'c' para comenzar ({seconds}s). Intento {retries+1}/{max_retries_per_angle+1}")
                # esperar 'c'
                started = False
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        try: cv2.destroyWindow(win) 
                        except: pass
                        return None
                    disp = frame.copy()
                    cv2.putText(disp, f"Preparado: {angle} - presiona 'c' (Captura {retries+1})", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                    cv2.imshow(win, disp)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord('c'):
                        started = True
                        break
                    if k == 27:
                        try: cv2.destroyWindow(win)
                        except: pass
                        return None
                if not started:
                    retries += 1
                    continue
                t0 = time.time()
                vecs = []
                while time.time() - t0 < seconds:
                    ret, frame = cap.read()
                    if not ret: break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    try:
                        results = face_mesh.process(rgb)
                    except Exception as e:
                        print("[FaceMesh] Error durante process():", e)
                        results = None
                    disp = frame.copy()
                    elapsed = int(time.time()-t0)
                    cv2.putText(disp, f"Capturando {angle}: {elapsed}s/{seconds}s", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    if results and getattr(results, 'multi_face_landmarks', None):
                        fl = results.multi_face_landmarks[0]
                        yaw = estimate_yaw_deg(fl)
                        accept = True
                        if angle == "derecha" and yaw > -YAW_THRESHOLD_DEGREES:
                            accept = False
                        if angle == "izquierda" and yaw < YAW_THRESHOLD_DEGREES:
                            accept = False
                        if accept:
                            raw = extract_selected_landmarks(fl)
                            vec_norm = normalize_vector(raw)
                            vecs.append(vec_norm)
                            xs = [lm.x for lm in fl.landmark]
                            ys = [lm.y for lm in fl.landmark]
                            h, w = frame.shape[:2]
                            x1 = max(int(min(xs)*w)-10, 0)
                            y1 = max(int(min(ys)*h)-10, 0)
                            x2 = min(int(max(xs)*w)+10, w-1)
                            y2 = min(int(max(ys)*h)+10, h-1)
                            cv2.rectangle(disp, (x1,y1), (x2,y2), DRAW_BOX_COLOR, 2)
                            cv2.putText(disp, f"Yaw {yaw:.1f}deg", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
                        else:
                            cv2.putText(disp, f"Pose no adecuada (yaw {yaw:.1f}deg)", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    cv2.imshow(win, disp)
                    if cv2.waitKey(1) & 0xFF == 27:
                        try: cv2.destroyWindow(win)
                        except: pass
                        return None
                if len(vecs) == 0:
                    print(f"[Registro] No se obtuvieron frames válidos para ángulo {angle} en intento {retries+1}. Reintentando...")
                    retries += 1
                    time.sleep(0.3)
                    continue
                collected[angle] = np.mean(np.stack(vecs), axis=0)
                print(f"[Registro] Ángulo {angle} completado ({len(vecs)} frames válidos).")
                break
            else:
                # si agotamos reintentos para este ángulo
                print(f"[Registro] Fallaron todos los intentos para ángulo {angle}. Cancelando registro.")
                try: cv2.destroyWindow(win)
                except: pass
                return None
        try: cv2.destroyWindow(win)
        except: pass
        return collected
    except Exception as e:
        print("[Registro] Error durante captura:", e)
        try: cv2.destroyWindow(win)
        except: pass
        return None
    # fin-capture_three_angles_new_person
    
# ---------------- Small helpers --------------------
def apply_clahe(frame):
    """Aplica CLAHE sobre la luminancia para condiciones bajas de luz."""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
        gray_clahe = clahe.apply(gray)
        return cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2BGR)
    except Exception:
        return frame.copy()

def capture_more_samples(name, cap, face_mesh, target_n=MORE_SAMPLES_COUNT, timeout_sec=20):
    """
    Captura target_n vectores válidos (no por ángulo; útil para inicializar rápidamente).
    Devuelve dict samples con 'frontal' promedio (y deja 'derecha'/'izquierda' None).
    """
    t0 = time.time()
    vecs = []
    print(f"[Registro-Multi] Capturando hasta {target_n} vectores válidos para {name} (timeout {timeout_sec}s)")
    while len(vecs) < target_n and (time.time() - t0) < timeout_sec:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            results = face_mesh.process(rgb)
        except Exception as e:
            results = None
        if results and getattr(results, "multi_face_landmarks", None):
            fl = results.multi_face_landmarks[0]
            raw = extract_selected_landmarks(fl)
            vec_norm = normalize_vector(raw)
            if vec_norm is not None:
                vecs.append(vec_norm)
        # pequeña pausa para no saturar
        cv2.waitKey(5)
    if len(vecs) == 0:
        print("[Registro-Multi] No se obtuvieron vectores válidos.")
        return None
    avg = np.mean(np.stack(vecs), axis=0)
    print(f"[Registro-Multi] Capturados {len(vecs)} vectores. Promediando y guardando como 'frontal'.")
    return {"frontal": avg, "derecha": None, "izquierda": None}

def get_center_key(x1, y1, x2, y2, grid=60):
    """
    Crea una key grosera basada en la posición del bbox (coarsening por grid px).
    Permite agrupar predicciones por la misma cara aproximada en frames consecutivos.
    """
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return (cx // grid, cy // grid)

def confirm_identity(buff):
    """
    Dado un buffer de predicciones (lista de (name,score)), devuelve (name_confirmed, score_avg) o (None, None).
    Requiere CONFIRM_FRAMES y CONFIRM_RATIO.
    """
    if not buff or len(buff) < 1:
        return None, None
    names = [b[0] for b in buff]
    # ignorar 'Desconocido' para confirmar
    names_filtered = [n for n in names if n != "Desconocido"]
    if len(names_filtered) == 0:
        return None, None
    # cuenta más frecuente
    from collections import Counter
    cnt = Counter(names_filtered)
    most, count = cnt.most_common(1)[0]
    if count >= max(CONFIRM_FRAMES, int(len(buff) * CONFIRM_RATIO)):
        # calcular score promedio de entradas que coinciden
        scores = [s for (n,s) in buff if n == most and s is not None]
        avg_score = (sum(scores) / len(scores)) if scores else None
        return most, avg_score
    return None, None
# --------------------------------------------------


# ---------------- Recognition loop (versión corregida) ----------------
def recognition_loop():
    db = load_database()
    ensure_excel_exists(EXCEL_PATH)
    # cargar modelo (puede ser None si no hay)
    clf, scaler, pca = load_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return

    # estructuras auxiliares
    vote_buffers = {}   # key -> deque de últimas predicciones [(name, score), ...]
    last_seen_global = {}  # label -> last seen ts (por seguridad reset)
    try:
        # Mediapipe Face Mesh para detección y landmarks
        with mp_face_mesh.FaceMesh(
                static_image_mode=False, 
                max_num_faces=5, 
                refine_landmarks=True,
                min_detection_confidence=0.5, 
                min_tracking_confidence=0.5) as face_mesh:

            state = {}
            subject_dialogs = {}
            pending_registration = {"active": False, "name": None, "registro": None, "group": None, "subject": None}
            last_raw_vec = None

            def registration_callback(name, registro, group, subject):
                pending_registration["active"] = True
                pending_registration["name"] = name if name and name.strip() else None
                pending_registration["registro"] = registro
                pending_registration["group"] = group
                pending_registration["subject"] = subject
                print(f"[Dialog] Registro pedido: name={pending_registration['name']}, registro={registro}, group={group}, subject={subject}")

            def subject_callback_factory(person_name):
                def cb(nm, reg, grp, subj):
                    s = state.get(person_name, {})
                    real_reg = db.get(person_name, {}).get("registro", "-")
                    s["registro_session"] = real_reg or "-"
                    s["subject_session"] = subj or "-"
                    s["group_session"] = grp or db.get(person_name, {}).get("group", GROUP_OPTIONS[0])
                    s["last_seen"] = time.time()
                    state[person_name] = s
                    print(f"[Dialog] {person_name}: registro = {s['registro_session']}, materia = {s['subject_session']}, grupo = {s['group_session']}")
                return cb

            print("Comandos: 'n' registrar nuevo, 'q' o ESC salir.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[WARN] Frame no leído, saliendo.")
                    break

                # preprocesado (CLAHE) para baja luz
                frame_proc = apply_clahe(frame)

                rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)

                try:
                    results = face_mesh.process(rgb)
                except Exception as e:
                    print("[FaceMesh] Error en process():", e)
                    results = None

                display = frame.copy()

                # Si hay registro pendiente, iniciar captura (mejor: multisamples si activado)
                if pending_registration["active"]:
                    reg = pending_registration.copy()
                    pending_registration["active"] = False
                    name_reg = reg["name"] or "SinNombre"
                    registro = reg["registro"] or "SinRegistro"
                    grp_reg = reg["group"] or GROUP_OPTIONS[0]
                    subj_reg = reg["subject"] or "-"
                    print(f"[Registro] Iniciando captura para {name_reg} con el registro = {registro} (grupo {grp_reg}, materia {subj_reg})")
                    samples = None
                    if MORE_SAMPLES_ON_REGISTER:
                        samples = capture_more_samples(name_reg, cap, face_mesh, target_n=MORE_SAMPLES_COUNT)
                        samples = capture_three_angles_new_person(name_reg, cap, face_mesh)
                    # si no hubo multisamples, caer en la captura por 3 ángulos ya implementada
                    if samples is None:
                        samples = capture_three_angles_new_person(name_reg, cap, face_mesh)
                    if samples is None:
                        print("[Registro] Captura cancelada o fallida.")
                    else:
                        db[name_reg] = {"registro": registro, "group": grp_reg, "subject": subj_reg, "samples": samples}
                        save_database(db)
                        print(f"[Registro] Guardado {name_reg} en base local.")
                        # reentrenar modelo si hay >=2 clases
                        if len(db) >= 2:
                            print("[MODEL] Re-entrenando modelo...")
                            # entrenar con PCA fija si quieres (opcional)
                            clf_new, scaler_new, pca_new = train_and_save_model(db)
                            if clf_new is None:
                                print("[MODEL] Reentrenado fallido: insuficientes datos.")
                            else:
                                try:
                                    acc, nsamp, per_class = evaluate_model_on_db(clf_new, scaler_new, pca_new, db)
                                except:
                                    acc, nsamp, per_class = 0.0, 0, {}
                                print(f"[EVAL] accuracy={acc:.3f} | muestras={nsamp} | clases={len(per_class)}")
                                ACCEPT_ACC = 0.70
                                if acc >= ACCEPT_ACC:
                                    clf, scaler, pca = clf_new, scaler_new, pca_new
                                    print("[MODEL] Nuevo modelo aceptado y cargado.")
                                else:
                                    ts = int(time.time())
                                    try:
                                        backup_path = SVM_PATH.replace(".pkl", f"_weak_{ts}.pkl")
                                        joblib.dump((clf_new, scaler_new, pca_new), backup_path)
                                        print(f"[MODEL] Modelo débil guardado en {backup_path} (no se cargó).")
                                    except Exception:
                                        pass
                                    # no tocar global DIST_FALLBACK_THRESHOLD de forma peligrosa aquí

                # detección y reconocimiento
                if results and getattr(results, 'multi_face_landmarks', None):

                    for fl in results.multi_face_landmarks:

                        # calcular bounding box de los landmarks
                        xs = [lm.x for lm in fl.landmark]
                        ys = [lm.y for lm in fl.landmark]
                        h, w = frame.shape[:2]

                        x1 = max(int(min(xs) * w) - 10, 0)
                        y1 = max(int(min(ys) * h) - 10, 0)
                        x2 = min(int(max(xs) * w) + 10, w - 1)
                        y2 = min(int(max(ys) * h) + 10, h - 1)


                        # ----------------------------------------
#                        EXTRAER Y PREPARAR EL VECTOR DEL ROSTRO
                        # ----------------------------------------
                        raw_vec = extract_selected_landmarks(fl)

                        # EMA smoothing
                        if last_raw_vec is None:
                            smooth_raw = raw_vec
                        else:
                            smooth_raw = SMOOTH_ALPHA * raw_vec + (1.0 - SMOOTH_ALPHA) * last_raw_vec
                        last_raw_vec = smooth_raw.copy()

                        vec_norm = normalize_vector(smooth_raw)
                        if vec_norm is None:
                            continue
                        vec_norm = fix_length(vec_norm, max(len(vec_norm), 1))

                        # ----------------------------------------
                        # RECONOCIMIENTO DEL ROSTRO (SVM + fallback)
                        # ----------------------------------------
                        name_pred = None
                        score_pred = None

                        # INTENTO SVM
                        if clf is not None and scaler is not None and pca is not None:
                            try:
                                if vec_norm.size == scaler.mean_.shape[0]:
                                    Xs = scaler.transform([vec_norm])
                                    Xp = pca.transform(Xs)
                                    probs = clf.predict_proba(Xp)[0]
                                    idx = int(np.argmax(probs))
                                    score_pred = float(probs[idx])
                                    if score_pred >= SVM_PROB_THRESHOLD:
                                        name_pred = str(clf.classes_[idx])
                            except Exception as e:
                                print("[MODEL] Error predict:", e)
                                name_pred = None

                        # FALLBACK SI SVM NO CONFIRMA
                        d_fallback = None   # importante!

                        if name_pred is None:
                            fmatch, d_tmp = fallback_match(vec_norm, db, threshold=DIST_FALLBACK_THRESHOLD)
                            if fmatch:
                                name_pred = fmatch
                                d_fallback = d_tmp
                                score_pred = max(score_pred or 0.0, 1.0 - d_tmp)

                        # VALOR FINAL USADO EN LA ETIQUETA
                        if name_pred is not None:
                            if d_fallback is not None:
                                dist = d_fallback
                            else:
                                dist = 1.0 - (score_pred if score_pred is not None else 0.0)
                        else:
                            dist = 1.0

                        name = name_pred if name_pred is not None else "Desconocido"

                        # COLOR FINAL SEGÚN EL ORIGEN DEL RECONOCIMIENTO
                        if name_pred is not None:
                            # Si vino del SVM
                            if d_fallback is None:
                                if score_pred is not None and score_pred >= SVM_PROB_THRESHOLD:
                                    box_color = (0, 255, 0)   # verde
                                else:
                                    box_color = (0, 0, 255)   # rojo
                            else:
                                # Si vino de fallback
                                if d_fallback <= DIST_FALLBACK_THRESHOLD:
                                    box_color = (0, 255, 0)   # verde
                                else:
                                    box_color = (0, 0, 255)   # rojo
                        else:
                            box_color = (0, 0, 255)

                        cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)

                        # -----------------------------------------
                        # MOSTRAR TIMER (Entrada→Salida o Salida→Entrada)
                        # -----------------------------------------
                        timer = timers.get(name)

                        # mostrar temporizador si existe
                        if timer:
                            elapsed = time.time() - timer["start_time"]
                            remaining = int(EXIT_SECONDS_AFTER_ENTRY - elapsed)
                            # -------------------------------------------------------------
                            # SOLO REGISTRAR cuando:
                            # - remaining <= 0
                            # - hay rostro detectado
                            # - el nombre coincide (sigue siendo la misma persona)
                            # -------------------------------------------------------------
                            if remaining <= 0 and not timer.get("done", False):

                                # Verifica que el rostro siga presente y reconocido
                                if name_pred == name:
                                    tipo_auto = "salida" if timer["state"] == "entrada" else "entrada"

                                    print(f"[AUTO] Tiempo cumplido y rostro presente → Registrando {tipo_auto} para {name}")

                                    res = add_or_update_attendance(
                                        person_name=name,
                                        registro=db[name]["registro"],
                                        group=db[name]["group"],
                                        subject=db[name]["subject"],
                                        tipo=tipo_auto
                                    )

                                    timer["done"] = True
                                    timers[name] = timer
                                else:
                                    # No registrar aún → sigue esperando a que el rostro reaparezca
                                    print(f"[WAIT] Tiempo cumplido pero {name} NO está presente. Esperando...")

                            # -------------------------------------------------------------
                            # Mensaje del temporizador
                            # -------------------------------------------------------------
                            if remaining > 0:
                                tipo_msg = "Salida" if timer["state"] == "entrada" else "Entrada"
                                txt = f"{tipo_msg} en: {remaining}s"
                            else:
                                tipo_msg = "salida" if timer["state"] == "entrada" else "entrada"
                                txt = f"Listo para {tipo_msg}"

                            # Mostrar texto
                            h, w = display.shape[:2]
                            (tw, th), baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                            x = (w - tw) // 2
                            y = th + 20

                            cv2.putText(
                                display, txt, (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 255), 2, cv2.LINE_AA
                            )


                        # fin_timer display

                        # preparar etiqueta temporal
                        label_shown = name_pred if name_pred is not None else "Desconocido"
                        registro = db.get(label_shown, {}).get("registro", "-") if label_shown != "Desconocido" else "-"

                        # voting key (coarse spatial key to follow same face)
                        key = get_center_key(x1, y1, x2, y2)
                        if key not in vote_buffers:
                            from collections import deque
                            vote_buffers[key] = deque(maxlen=CONFIRM_FRAMES + 2)
                        vote_buffers[key].append((label_shown, score_pred))

                        # confirmar identidad desde buffer
                        confirmed_name, confirmed_score = confirm_identity(list(vote_buffers[key]))
                        # si no confirmada, tratar como desconocido al dibujar/registrar
                        final_label = confirmed_name if confirmed_name else "Desconocido"
                        # usar score de confirmed si disponible
                        draw_detection_label(display, (x1, y1, x2, y2), final_label, registro if final_label!="Desconocido" else "-", score=confirmed_score)

                        # si confirmado -> control de asistencia
                        if final_label != "Desconocido":
                            now_ts = time.time()
                            # crear estado si no existe
                            st = state.setdefault(final_label, {
                                "entry_marked": False,
                                "exit_marked": False,
                                "entry_time": None,
                                "last_seen": 0,
                                "subject_session": None,
                                "registro_session": db.get(final_label, {}).get("registro", "-")
                            })

                            # abrir dialogo materia si falta
                            if not st.get("subject_session") and final_label not in subject_dialogs:
                                subject_dialogs[final_label] = True
                                NonBlockingDialog(title=f"Materia para {final_label}", ask_name=False, default_group=None, callback=subject_callback_factory(final_label))

                            # si no tiene materia aún, actualizar last_seen y saltar
                            if not st.get("subject_session"):
                                st["last_seen"] = now_ts
                                state[final_label] = st
                                continue

                            # debounce: evitar múltiples registros por frames muy seguidos
                            if now_ts - st.get("last_seen", 0) < 1.2:
                                st["last_seen"] = now_ts
                                state[final_label] = st
                                continue

                            # registrar entrada (si no hay)
                            if not st.get("entry_marked", False):
                                registrar_entrada(final_label, db, state)
                                st["entry_marked"] = True
                                st["entry_time"] = now_ts
                                print(f"[ENTRY] {final_label}: entrada marcada")

                            # si ya había entrada y no salida -> intentar marcar salida si pasó tiempo
                            elif st.get("entry_marked") and not st.get("exit_marked", False):
                                if st.get("entry_time") and (now_ts - st["entry_time"] >= EXIT_SECONDS_AFTER_ENTRY):
                                    registrar_salida(final_label, db, state)
                                    st["exit_marked"] = True
                                    print(f"[EXIT] {final_label}: salida marcada")

                            # actualizar tiempos
                            st["last_seen"] = now_ts
                            state[final_label] = st
                            last_seen_global[final_label] = now_ts

                # limpiar subject_dialogs si ya seleccionaron materia
                for name_in_state in list(subject_dialogs.keys()):
                    if state.get(name_in_state, {}).get("subject_session"):
                        subject_dialogs.pop(name_in_state, None)

                # Reiniciar estados stale (si no se ha visto a la persona en RESET_STATE_SECONDS)
                now_all = time.time()
                for lbl, st in list(state.items()):
                    if now_all - st.get("last_seen", 0) > RESET_STATE_SECONDS:
                        # reset medio: mantener registro_session pero permitir nueva entrada mañana
                        state[lbl] = {
                            "entry_marked": False,
                            "exit_marked": False,
                            "entry_time": None,
                            "last_seen": 0,
                            "subject_session": None,
                            "registro_session": db.get(lbl, {}).get("registro", "-")
                        }

                cv2.imshow("Asistencia - Webcam (presiona tecla 'n' para registrar)", display)
                k = cv2.waitKey(1) & 0xFF
                if k == 27 or k == ord('q'):
                    break
                if k == ord('n'):
                    print("[UI] Abriendo dialogo para registrar nuevo alumno (no bloqueante).")
                    NonBlockingDialog(title="Registrar Nuevo Alumno", ask_name=True, default_group=None, callback=registration_callback)

    except Exception as e:
        print("[ERROR] Error en reconocimiento principal:", e)
    finally:
        try:
            cap.release()
        except:
            pass
        try:
            cv2.destroyAllWindows()
        except:
            pass

# ---  Mostrar tabla de asistencias / usuarios en bd local  ---
def mostrar_tabla():
    print("\n===== TABLAS DISPONIBLES =====")
    print("1) Tabla de asistencias (Excel)")
    print("2) Usuarios registrados (DB local)")
    
    opcion = input("Seleccione una opción: ")

    if opcion == "1":
        mostrar_tabla_excel()
    elif opcion == "2":
        mostrar_tabla_usuarios()
    else:
        print("Opción inválida.")


# --- Funciones de las tablas ---
def mostrar_tabla_excel(ruta_excel="asistencias.xlsx"):
    try:
        dfs = pd.read_excel(ruta_excel, sheet_name=None)  # Cargar TODAS las hojas
        df = pd.concat(dfs.values(), ignore_index=True)   # Unificar
    except Exception as e:
        print(f"[ERROR] No se pudo leer el archivo Excel: {e}")
        return

    if df.empty:
        print("[TABLA] No hay registros de asistencias.")
        return

    print("\n===== CONSULTAR ASISTENCIAS =====")
    print("1) Mostrar grupo 7O")
    print("2) Mostrar grupo 7P")
    print("3) Mostrar TODOS los grupos")
    
    opcion = input("Seleccione una opción: ").strip()

    # Normalizamos columna
    if "Grupo" not in df.columns:
        print("[ERROR] La tabla no contiene la columna 'Grupo'.")
        return
    
    df["Grupo"] = df["Grupo"].astype(str).str.upper()

    # ------------------------------
    # Filtrar según opción elegida
    # ------------------------------
    if opcion == "1":
        df_filtrado = df[df["Grupo"] == "7O"]
        titulo = "REGISTROS GRUPO 7O"

    elif opcion == "2":
        df_filtrado = df[df["Grupo"] == "7P"]
        titulo = "REGISTROS GRUPO 7P"

    elif opcion == "3":
        df_filtrado = df
        titulo = "REGISTROS DE TODOS LOS GRUPOS"

    else:
        print("Opción inválida.")
        return

    # ------------------------------
    # Mostrar resultados
    # ------------------------------
    if df_filtrado.empty:
        print("[TABLA] No hay registros para la selección realizada.")
        return

    print(f"\n===== {titulo} =====")
    print(df_filtrado.to_string(index=False))
    print("\n")
# fin-mostrar_tabla_excel

def mostrar_tabla_usuarios():
    db = load_database()
    if not db:
        print("[TABLA] No hay usuarios registrados en la base de datos local.")
        return
    print("\n===== USUARIOS REGISTRADOS =====")
    print(f"{'Nombre':<20} {'Registro':<15} {'Grupo':<10} {'Materia':<20}")
    print("-" * 65)
    for name, info in db.items():
        registro = info.get("registro", "-")
        group = info.get("group", "-")
        subject = info.get("subject", "-")
        print(f"{name:<20} {registro:<15} {group:<10} {subject:<20}")
    print("\n")
# fin-mostrar_tabla_usuarios

ADMIN_PASSWORD = "ceti2025"

def admin_menu():
    print("\n===== PANEL DE ADMINISTRACIÓN =====")

    pwd = input("Ingrese la contraseña de administrador: ").strip()
    if pwd != ADMIN_PASSWORD:
        print("[ERROR] Contraseña incorrecta.")
        return

    while True:
        print("\n--- ADMINISTRAR USUARIOS ---")
        print("1) Buscar usuario")
        print("2) Actualizar usuario")
        print("3) Eliminar usuario")
        print("4) Listar todos los usuarios")
        print("5) Salir")

        opcion = input("Seleccione una opción: ").strip()

        if opcion == "1":
            admin_buscar_usuario()
        elif opcion == "2":
            admin_actualizar_usuario()
        elif opcion == "3":
            admin_eliminar_usuario()
        elif opcion == "4":
            admin_mostrar_usuarios()
        elif opcion == "5":
            break
        else:
            print("Opción inválida.")
# fin-admin_menu

# ------------ Operaciones de admin ------------
def admin_buscar_usuario():
    db = load_database()

    print("\n=== BUSCAR USUARIO ===")
    criterio = input("Ingrese nombre o registro: ").strip().lower()

    encontrados = {
        k: v for k, v in db.items()
        if criterio in k.lower() or criterio in str(v.get("registro", "")).lower()
    }

    if not encontrados:
        print("[INFO] No se encontraron coincidencias.")
        return

    print("\nUsuarios encontrados:")
    for nombre, data in encontrados.items():
        print(f"- {nombre} → {data}")
# fin-admin_buscar_usuario

def admin_actualizar_usuario():
    db = load_database()
    nombre = input("\nNombre exacto del usuario a actualizar: ").strip()

    if nombre not in db:
        print("[ERROR] Usuario no encontrado.")
        return

    alumno = db[nombre]

    print("\nCampos actuales del usuario:")

    # Mostrar solo campos permitidos para actualizar
    campos_permitidos = ["nombre", "registro", "group", "subject"]

    for k in campos_permitidos:
        valor = alumno.get(k, "-")
        print(f"{k}: {valor}")


    print("\nPresione ENTER para no cambiar un campo.")

    nuevo_nombre = input("Nuevo nombre: ").strip()
    nuevo_registro = input("Nuevo registro: ").strip()
    nuevo_grupo = input("Nuevo grupo: ").strip()
    nueva_materia = input("Nueva materia: ").strip()

    if nuevo_registro:
        alumno["registro"] = nuevo_registro
    if nuevo_nombre:
        db[nuevo_nombre] = alumno
        if nuevo_nombre != nombre:
            del db[nombre]
        nombre = nuevo_nombre
    if nuevo_grupo:
        alumno["group"] = nuevo_grupo
    if nueva_materia:
        alumno["subject"] = nueva_materia

    db[nombre] = alumno
    save_database(db)

    print("[OK] Usuario actualizado con éxito.")
# fin-admin_actualizar_usuario

def admin_eliminar_usuario():
    db = load_database()
    nombre = input("\nNombre exacto del usuario a eliminar: ").strip()

    if nombre not in db:
        print("[ERROR] Usuario no encontrado.")
        return

    conf = input(f"¿Seguro que desea eliminar a '{nombre}'? (s/n): ").strip().lower()
    if conf != "s":
        print("Cancelado.")
        return

    del db[nombre]
    save_database(db)
    print("[OK] Usuario eliminado correctamente.")
# fin-admin_eliminar_usuario

def admin_mostrar_usuarios():
    db = load_database()

    if not db:
        print("[INFO] No hay usuarios registrados.")
        return

    print("\n=== LISTA DE USUARIOS ===\n")

    # Encabezado
    print(f"{'Nombre':20} | {'Registro':10} | {'Grupo':6} | {'Materia':15}")
    print("-" * 60)

    # Filas
    for nombre, u in db.items():
        print(f"{nombre:20} | {u.get('registro','-'):10} | {u.get('group','-'):6} | {u.get('subject','-'):15}")

# ---------------- Main ----------------
def main():
    print("Iniciando sistema de asistencias (PCA + SVM, yaw check, dialogs no bloqueantes).")
    df = pd.read_excel(EXCEL_PATH)
    print(df.columns)  
    # Garantizar que el archivo Excel existe antes de comenzar
    ensure_excel_exists(EXCEL_PATH)

    while True:
        print("\n===== SISTEMA DE ASISTENCIAS =====")
        print("1) Iniciar reconocimiento")
        print("2) Mostrar registros")
        print("3) Exportar PDF por grupo")
        print("4) Exportar PDF por grupo/materia")
        print("5) Exportar PDF por grupo/materia/fecha")
        print("6) Panel de administración")
        print("7) Salir")

        opcion = input("Seleccione una opción: ")

        if opcion == "1": # Iniciar reconocimiento
            recognition_loop()

        elif opcion == "2": # Mostrar registros
            try:
                mostrar_tabla()
            except Exception as e:
                print(f"[ERROR] No se pudo mostrar la tabla: {e}")

        elif opcion == "3": # Exportar PDF por grupo
            grupo = input("Ingrese el grupo a exportar (ej. 7O, 7P): ").strip()
            try:
                exportar_pdf_grupo(grupo)
            except Exception as e:
                print(f"[ERROR] No se pudo generar el PDF: {e}")

        elif opcion == "4": # Exportar PDF por grupo/materia
            grupo = input("Ingrese el grupo a exportar (7O / 7P): ").strip()
            materia = input("Ingrese la materia a exportar (ML / PDI): ").strip()
            try:
                exportar_pdf_grupo_materia(grupo, materia)
            except Exception as e:
                print(f"[ERROR] No se pudo generar el PDF: {e}")

        elif opcion == "5":  # Exportar PDF por grupo, materia y fecha
            grupo = input("Ingrese el grupo (7O / 7P): ").strip()
            materia = input("Ingrese la materia (ML / PDI): ").strip()
            fecha = input("Ingrese la fecha (YYYY-MM-DD) o deje vacío para hoy: ").strip()
            try:
                exportar_pdf_grupo_materia_fecha(grupo, materia, fecha)
            except Exception as e:
                print(f"[ERROR] No se pudo generar el PDF: {e}")

        elif opcion == "6": # Panel de administración
            admin_menu()

        elif opcion == "7":
            print("Saliendo del sistema de asistencias.")
            break
        else:
            print("Opción inválida. Intente nuevamente.")

if __name__ == "__main__":
    main()