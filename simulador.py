"""
TEL-328 · Simulador de zona de juego (sin webcam)

Arma una pizarra VIRTUAL con recortes reales de post-its (de dataset_prueba/, que
el modelo no vio al entrenar) pegados sobre la foto de la pared, y corre el juego
completo encima: detector de amarillo -> seguidor -> lector de símbolos (modelo
real) -> votación -> reglas (meta / trampolín).

Uso:
  python simulador.py                                   (escenario por defecto)
  python simulador.py "limpio flecha_arriba _ limpio corona"
  python simulador.py "limpio limpio@-100 corona" --entrenamiento

  Cada palabra es un post-it de izquierda a derecha: corona, flecha_arriba,
  limpio u otro. "_" deja un hueco (falta un post-it). "@-100" lo sube 100 px
  (o "@60" lo baja). --entrenamiento toma los recortes de dataset/ en vez de
  dataset_prueba/.

Las teclas del juego funcionan igual (clic derecho, R, D, ESC...).
"""
import os
import random
import sys

import cv2
import numpy as np

import proyecto_v3 as P

CARPETA = os.path.dirname(os.path.abspath(__file__))
FOTO_PARED = os.path.join(CARPETA, "desafios", "pizarra_base.jpg")
ESCENARIO_DEFECTO = "limpio flecha_arriba _ limpio otro corona"
LADO, PASO, Y_BASE, X0 = 70, 130, 380, 30   # post-its de 70 px con huecos de 60 px


def pared():
    """La pizarra real sin sus post-its (se borran con inpaint)."""
    foto = cv2.resize(cv2.imread(FOTO_PARED), (P.ANCHO, P.ALTO))
    mascara = np.zeros(foto.shape[:2], np.uint8)
    mascara[225:320, 175:410] = 255
    return cv2.inpaint(foto, mascara, 7, cv2.INPAINT_TELEA)


def armar_escena(escenario, origen, semilla=0):
    """Imagen tal como debe VERSE en el juego y lista de (clase, archivo) usados."""
    rng = random.Random(semilla)
    img, usados = pared(), []
    for i, palabra in enumerate(escenario.split()):
        clase, _, dy = palabra.partition("@")
        if clase == "_":
            continue
        carpeta = os.path.join(origen, clase)
        archivos = sorted(f for f in os.listdir(carpeta) if f.lower().endswith((".png", ".jpg")))
        f = rng.choice(archivos)
        recorte = cv2.resize(cv2.imread(os.path.join(carpeta, f)), (LADO, LADO))
        x, y = X0 + i * PASO, Y_BASE + int(dy or 0)
        sombra = img[y + 4:y + LADO + 4, x + 3:x + LADO + 3]
        img[y + 4:y + LADO + 4, x + 3:x + LADO + 3] = (sombra * 0.75).astype(np.uint8)
        img[y:y + LADO, x:x + LADO] = recorte
        usados.append((clase, f))
    return img, usados


class CamaraVirtual:
    """Imita cv2.VideoCapture: entrega siempre la misma escena con un poco de ruido."""

    def __init__(self, escena):
        # el juego espeja la imagen de la cámara: se entrega espejada para que se vea derecha
        self.cuadro = cv2.flip(escena, 1)
        self.rng = np.random.default_rng(0)

    def read(self):
        ruido = self.rng.normal(0, 3, self.cuadro.shape)
        return True, np.clip(self.cuadro + ruido, 0, 255).astype(np.uint8)

    def isOpened(self):
        return True

    def set(self, *args):
        return True

    def get(self, *args):
        return 0

    def release(self):
        pass


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    escenario = args[0] if args else ESCENARIO_DEFECTO
    origen = os.path.join(CARPETA, "dataset" if "--entrenamiento" in sys.argv else "dataset_prueba")
    escena, usados = armar_escena(escenario, origen)
    print("Escenario:", escenario)
    for letra, (clase, f) in zip("ABCDEFGHIJ", usados):
        print(f"  {letra}: {clase:14s} ({f})")
    P.abrir_camara = lambda: CamaraVirtual(escena)
    P.main()


if __name__ == "__main__":
    main()
