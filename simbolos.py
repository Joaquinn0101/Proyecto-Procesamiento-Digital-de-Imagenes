"""
TEL-328 · Lectura de símbolos dibujados en los post-its (proyecto_v3)

Paso 1 (proyecto_v3.py): el detector de amarillo encuentra cada post-it.
Paso 2 (este archivo): se recorta SOLO ese post-it y un clasificador YOLO11n-cls
dice qué tiene dibujado:
    corona         -> meta
    flecha_arriba  -> trampolín
    limpio         -> post-it sin dibujo
    otro           -> cualquier otra cosa (X, flecha abajo o de lado, garabato, dedo)

Una sola lectura puede fallar (reflejo, desenfoque), así que cada post-it guarda
sus últimas lecturas y recién cambia de tipo cuando la votación es clara.

El modelo se entrena con entrenar_simbolos.py y queda en modelos/simbolos.pt.
Sin modelo, todo sigue funcionando: los post-its son "limpio" y el tipo se puede
marcar a mano (clic derecho en el juego).
"""
import os
import time
from collections import deque

import cv2
import numpy as np

CARPETA = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(CARPETA, "modelos", "simbolos.pt")
MSG_SIN_ARCHIVO = "sin modelo (entrena con entrenar_simbolos.py)"

# Orden alfabético = el orden en que Ultralytics numera las carpetas de clases
CLASES = ("corona", "flecha_arriba", "limpio", "otro")
SIMBOLOS = ("corona", "flecha_arriba")         # los que cambian las reglas del juego

# --- Recorte ---
TAM_RECORTE = 128      # px del recorte guardado / clasificado
IMGSZ = 96             # px con que trabaja el modelo
MARGEN_RECORTE = 0.08  # se quita 8% de borde por lado (sombra, pared)

# --- Cuándo vale la pena leer un post-it ---
LADO_MIN = 40          # px: más chico que esto, el dibujo no se distingue
PROPORCION_MIN_SIMBOLO = 0.75  # lado corto / lado largo: una FILA de post-its pegados no se lee
                               # (distinto de proyecto_v3.PROPORCION_MIN = 0.2, que SÍ acepta filas
                               # como plataforma)

# --- Votación ---
CADA_CUADROS = 10        # cada 10 cuadros: ~6 veces/s si la cámara da 60 fps, ~3 veces/s a 30 fps
LECTURAS_GUARDADAS = 5
LECTURAS_MIN = 3
UMBRAL_CONFIANZA = 0.85  # promedio mínimo de la clase ganadora para cambiar de tipo

ROTACION_MANUAL = ("auto", "corona", "flecha_arriba", "limpio")

# --- Captura de datos para entrenar (tecla S) ---
CARPETA_CAPTURA = os.path.join(CARPETA, "dataset", "sin_clasificar")
CAPTURA_CADA = 30        # cuadros entre capturas: ~0,5 s a 60 fps, ~1 s a 30 fps


def _xywh(rect):
    """Acepta (x, y, w, h) o pygame.Rect."""
    return int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])


def recortar(frame_bgr, rect):
    """Recorte del post-it, sin su borde, a TAM_RECORTE x TAM_RECORTE (BGR).
    Se ajusta a los límites de la imagen; None si no queda nada."""
    x, y, w, h = _xywh(rect)
    mx, my = int(round(w * MARGEN_RECORTE)), int(round(h * MARGEN_RECORTE))
    alto, ancho = frame_bgr.shape[:2]
    x0, y0 = max(0, x + mx), max(0, y + my)
    x1, y1 = min(ancho, x + w - mx), min(alto, y + h - my)
    if x1 <= x0 or y1 <= y0:
        return None
    return cv2.resize(frame_bgr[y0:y1, x0:x1], (TAM_RECORTE, TAM_RECORTE),
                      interpolation=cv2.INTER_AREA)


def es_clasificable(rect):
    """¿Es un post-it suelto y lo bastante grande como para leer su dibujo?"""
    _, _, w, h = _xywh(rect)
    if min(w, h) < LADO_MIN:
        return False
    return min(w, h) / max(w, h) >= PROPORCION_MIN_SIMBOLO


def recortes_legibles(frame_bgr, plataformas, quietos):
    """(ids, recortes) de los post-its quietos donde se puede leer el dibujo."""
    ids, recortes = [], []
    for pid in sorted(quietos):
        rect = plataformas.get(pid)
        if rect is None or not es_clasificable(rect):
            continue
        recorte = recortar(frame_bgr, rect)
        if recorte is not None:
            ids.append(pid)
            recortes.append(recorte)
    return ids, recortes


def cargar_clasificador(ruta=RUTA_MODELO):
    """Carga el modelo entrenado. Devuelve (clasificar, mensaje).

    clasificar(lista de recortes BGR) -> array (n, 4) con las probabilidades en
    el orden de CLASES. Si algo falla devuelve (None, motivo) y el juego sigue
    sin símbolos.
    """
    if not os.path.exists(ruta):
        return None, MSG_SIN_ARCHIVO
    hilos = cv2.getNumThreads()
    try:
        from ultralytics import YOLO
    except ImportError:
        return None, "sin modelo: falta instalar ultralytics"
    except Exception as e:   # p. ej. OSError de Windows con un torch dañado (WinError 126)
        return None, f"sin modelo: no se pudo cargar ultralytics/torch ({e})"
    finally:
        # Ultralytics hace cv2.setNumThreads(0) al importarse (lo necesita para entrenar);
        # en el juego eso dejaría al detector de post-its con 1 solo hilo.
        cv2.setNumThreads(hilos)
    try:
        modelo = YOLO(ruta)
        nombres = [modelo.names[i] for i in sorted(modelo.names)]
        if tuple(nombres) != CLASES:
            return None, f"modelo con clases {nombres}, se esperaban {list(CLASES)}"

        def clasificar(recortes):
            resultados = modelo.predict(recortes, imgsz=IMGSZ, verbose=False)
            return np.stack([r.probs.data.cpu().numpy() for r in resultados])

        # La primera predicción es lenta (carga en GPU): se hace ahora y no en pleno juego
        clasificar([np.zeros((TAM_RECORTE, TAM_RECORTE, 3), np.uint8)])
        return clasificar, "modelo OK"
    except Exception as e:
        return None, f"sin modelo: {e}"


class LectorSimbolos:
    """Decide el tipo de cada post-it votando con sus últimas lecturas."""

    def __init__(self, clasificar=None, mensaje=""):
        self.clasificar = clasificar   # función de cargar_clasificador (o None)
        self.mensaje = mensaje
        self._lecturas = {}            # id -> últimas filas de probabilidades
        self._decididos = {}           # id -> tipo decidido por la votación
        self.manual = {}               # id -> tipo forzado con clic derecho

    @property
    def activo(self):
        return self.clasificar is not None

    def actualizar(self, frame_bgr, plataformas, quietos, cuadro):
        """Llamar una vez por cuadro. plataformas: {id: Rect}; quietos: ids a la
        vista y sin movimiento encima (SeguidorPlataformas.quietos())."""
        for datos in (self._lecturas, self._decididos, self.manual):   # post-its que ya no están
            for pid in [p for p in datos if p not in plataformas]:
                del datos[pid]
        if not self.activo or cuadro % CADA_CUADROS != 0:
            return

        ids, recortes = recortes_legibles(frame_bgr, plataformas, quietos)
        if not recortes:
            return
        try:
            probs = np.asarray(self.clasificar(recortes), dtype=float)
        except Exception as e:   # el juego no se cae por el modelo
            print(f"[simbolos] error al clasificar: {e}; se desactiva")
            self.clasificar = None
            self.mensaje = f"sin modelo: error al clasificar ({e})"
            return
        for pid, fila in zip(ids, probs):
            self._votar(pid, fila)

    def _votar(self, pid, fila):
        lecturas = self._lecturas.setdefault(pid, deque(maxlen=LECTURAS_GUARDADAS))
        lecturas.append(fila)
        if len(lecturas) < LECTURAS_MIN:
            return
        promedio = np.mean(lecturas, axis=0)
        k = int(np.argmax(promedio))
        if promedio[k] < UMBRAL_CONFIANZA:
            return
        nuevo, actual = CLASES[k], self._decididos.get(pid, "limpio")
        if actual in SIMBOLOS and nuevo not in SIMBOLOS:
            return   # un dibujo en papel no desaparece solo: otro/limpio no borran un símbolo
        self._decididos[pid] = nuevo

    def tipos(self, plataformas):
        """{id: tipo} para cada plataforma: manual > votación > "limpio"."""
        return {pid: self.manual.get(pid) or self._decididos.get(pid, "limpio")
                for pid in plataformas}

    def rotar_manual(self, pid):
        """Clic derecho: auto -> corona -> flecha_arriba -> limpio -> auto."""
        actual = self.manual.get(pid, "auto")
        nuevo = ROTACION_MANUAL[(ROTACION_MANUAL.index(actual) + 1) % len(ROTACION_MANUAL)]
        if nuevo == "auto":
            self.manual.pop(pid, None)
        else:
            self.manual[pid] = nuevo
        return nuevo


class Capturador:
    """Guarda recortes de los post-its para armar el dataset (tecla S).

    Nombre: <sesion>_p<id>_<n>.png. Todos los recortes del mismo post-it en una
    sesión comparten '<sesion>_p<id>', así se ordenan juntos a mano en
    dataset/corona, flecha_arriba, limpio u otro. El entrenamiento usa ese
    prefijo para no mezclar el mismo post-it en train y val.
    """

    def __init__(self, carpeta=None):
        self.carpeta = carpeta or CARPETA_CAPTURA
        self.activo = False
        self.sesion = None
        self.guardados = 0

    def alternar(self):
        """Enciende o apaga la captura; al encender empieza una sesión nueva."""
        self.activo = not self.activo
        if self.activo:
            os.makedirs(self.carpeta, exist_ok=True)
            sesion = time.strftime("%Y%m%d_%H%M%S")
            if sesion != self.sesion:   # en el mismo segundo sigue numerando: no sobrescribe
                self.sesion, self.guardados = sesion, 0
        return self.activo

    def actualizar(self, frame_bgr, plataformas, quietos, cuadro):
        """Cada CAPTURA_CADA cuadros guarda los post-its quietos y legibles.
        Devuelve cuántos recortes guardó."""
        if not self.activo or cuadro % CAPTURA_CADA != 0:
            return 0
        ids, recortes = recortes_legibles(frame_bgr, plataformas, quietos)
        for pid, recorte in zip(ids, recortes):
            ruta = os.path.join(self.carpeta, f"{self.sesion}_p{pid}_{self.guardados:04d}.png")
            cv2.imencode(".png", recorte)[1].tofile(ruta)   # tolera rutas con tildes en Windows
            self.guardados += 1
        return len(recortes)
