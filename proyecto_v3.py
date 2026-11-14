"""
TEL-328 · Proyecto PDI versión 3 (post-its con símbolos)

Idea: la webcam mira una pizarra/pared. Cada post-it AMARILLO se convierte
en una plataforma donde el personaje puede caminar.

Qué cambia respecto a proyecto_menu.py:
  1. Un solo modo. Se eliminan el menú de gestos, el modo aleatorio,
     la lava roja y la meta verde detectadas por color.
  2. El amarillo se detecta en espacio Lab y RELATIVO a la escena
     (b* del píxel contra la mediana de b* del cuadro), lo que compensa
     cambios de luz y de balance de blancos.
  3. Filtro de forma: solo se aceptan manchas que parezcan un post-it
     (casi cuadradas, sólidas, que llenen su rectángulo, ~4 esquinas).
  4. Estabilidad temporal: una plataforma aparece tras varios cuadros
     seguidos y no desaparece por un parpadeo o una mano que pasa.
  5. Calibración opcional: clic sobre un post-it para aprender su color.
  6. Tapado vs. sacado: si un post-it deja de verse, se compara lo que hay
     en su lugar con la pared que lo rodeaba. Si es pared y la escena está
     quieta -> lo sacaron. Si es otra cosa o hay movimiento -> está tapado
     (p. ej. una mano) y se RECUERDA. Se dibuja solo con borde.
  7. Personaje automático (cubo azul):
       - SEMÁFORO (arriba a la derecha):
           rojo     -> menos de 2 post-its, o desde el inicio no puede saltar a ninguna parte.
           amarillo -> preparado: el cubo espera en el inicio hasta que se presione ESPACIO.
           verde    -> en marcha.
       - En marcha salta solo de post-it en post-it, de izquierda a derecha y
         de vuelta (A->B->C->B->A...), y SOLO si el salto es físicamente
         posible (altura y distancia tipo Mario Bros). Si un salto es
         imposible, se devuelve por donde vino.
  8. Símbolos dibujados (simbolos.py): se recorta cada post-it y un
     clasificador YOLO11n-cls (modelos/simbolos.pt) lee qué tiene dibujado.
       - CORONA         -> meta: al pararse ahí el cubo termina el camino (¡META!).
       - FLECHA ARRIBA  -> trampolín: desde ahí sube ~300 px y salta ~410 px
                           (un salto normal: ~95 px y ~117 px).
       - limpio / otro  -> plataforma normal.
     El cubo parte en el post-it más a la izquierda que no sea corona y, si la
     corona es un vecino alcanzable, salta directo a ella.
     Sin modelo entrenado el juego funciona igual (clic derecho marca a mano).
     El post-it con símbolo debe estar SEPARADO de los demás y el dibujo no debe
     tocar sus bordes.

Controles:
  ESPACIO  dar la partida (con el semáforo en amarillo)
  Clic     calibrar el color sobre un post-it
  Clic der marcar a mano el tipo de un post-it (auto -> corona -> flecha -> limpio)
  S        capturar recortes para entrenar (dataset/sin_clasificar/)
  C        borrar calibración (vuelve al modo automático)
  F        congelar / descongelar plataformas
  E        fijar / soltar exposición de la cámara (usar ya apuntando a la pizarra)
  D        mostrar / ocultar ventana de depuración (máscara + contornos)
  G        grabar / detener video de la cámara en grabaciones/ (para desafios.py --video)
  R        devolver el personaje al inicio (el semáforo vuelve a amarillo)
  ESC      salir

Librerías: pip install opencv-python pygame numpy ultralytics
"""
import os
import sys
import time

import cv2
import numpy as np
import pygame

import simbolos

# ============================================================
# CONFIGURACIÓN
# ============================================================
ANCHO, ALTO = 800, 600
FUENTE_CAMARA = 0

# --- Detección de color (Lab, valores 0-255 de OpenCV; 128 = neutro) ---
DELTA_B = 15      # cuánto más "amarillo" (b*) que la escena debe ser un píxel
A_MIN, A_MAX = -35, 8   # rango de a* (verde-rojo): el amarillo tira a verde,
                        # la piel y los naranjos tiran a rojo (a* positivo)
L_MINIMO = 90     # brillo mínimo: descarta sombras y zonas oscuras
TOL_CALIB_B = 10  # tolerancias al usar calibración por clic (estrechas:
TOL_CALIB_A = 10  # un post-it pálido y una pared crema difieren poco)

# --- Filtro de forma ---
AREA_MIN = 800        # px² (post-it lejano)
AREA_MAX = 45000      # px² (post-it muy cerca)
PROPORCION_MIN = 0.2  # lado corto / lado largo. 0.2 acepta una FILA de hasta
                      # ~5 post-its pegados como una sola plataforma larga
EXTENSION_MIN = 0.80  # área del contorno / área de su rectángulo mínimo
SOLIDEZ_MIN = 0.88    # área del contorno / área de su envolvente convexa
VERTICES_MIN, VERTICES_MAX = 4, 6

# --- Estabilidad temporal ---
CUADROS_PARA_APARECER = 8     # ~0.13 s a 60 fps
CUADROS_PARA_DESAPARECER = 20  # (solo si no hay imagen para razonar oclusión)
CANDIDATO_PERDIDOS_MAX = 5     # un candidato aún no confirmado se olvida rápido

# --- Tapado vs. sacado (razonamiento de oclusión) ---
# Cuando un post-it confirmado deja de detectarse, se mira QUÉ hay en su lugar:
#   - parece la pared que lo rodeaba y la escena está quieta -> SACADO (se borra)
#   - cualquier otra cosa (mano, manga, sombra) o hay movimiento -> TAPADO (se recuerda)
ANILLO_PX = 12              # grosor del anillo de pared alrededor del post-it
UMBRAL_PARED = 6            # distancia de color (Lab relativo) para decir "es pared".
                            # Medido en desafios.py: pared 0.5-1.1, piel 12.6, mano 20-30
UMBRAL_TEXTURA_PARED = 12   # desviación de brillo máxima de una zona "lisa como pared"
UMBRAL_MOVIMIENTO = 12      # diferencia de gris entre cuadros para contar movimiento
                            # (piel sobre pizarra clara difiere solo ~13 niveles de gris)
FRACCION_MOVIMIENTO = 0.03  # % de la zona con movimiento para decir "se está moviendo algo"
CUADROS_CONFIRMAR_SACADO = 15  # ~0.25 s viendo pared quieta antes de borrar
TAPADO_MAXIMO = 20 * 60        # tras 20 s tapado se olvida igual (evita fantasmas)
REENGANCHE = 0.8               # si un post-it reaparece corrido (cámara movida) a menos de
                               # 0.8 x su tamaño de donde estaba, se reconoce como el mismo
SOLAPE_DUPLICADO = 0.25        # un "tapado" que se solapa >25% con uno visible es un fantasma
# Movimiento de CÁMARA (no de una mano): hay movimiento en los 4 cuadrantes a la vez
CAMARA_FRACCION_CUADRANTE = 0.12
CAMARA_ENFRIAMIENTO = 20       # cuadros tras mover la cámara en que no se confía en "tapados"
IOU_MINIMO = 0.3
# Post-it confirmado que deja de pasar el filtro de forma (sombra, perspectiva, luz,
# borde): si una mancha amarilla calza con él, sigue "visible" en vez de "tapado"
IOU_MANCHA = 0.5
AREA_MANCHA = (0.6, 1.6)       # área de la mancha / área del post-it
SUAVIZADO = 0.25               # 0 = no se mueve, 1 = sigue la detección cruda

# --- Personaje y saltos (1 "cuerpo" = TAM_PERSONAJE px) ---
TAM_PERSONAJE = 35
GRAVEDAD = 0.8          # px/cuadro²
VEL_SALTO = 13          # px/cuadro hacia arriba -> sube 13²/(2·0.8) ≈ 105 px ≈ 3 cuerpos
VEL_AVANCE = 4          # px/cuadro en el aire  -> en plano alcanza ≈ 130 px ≈ 3.7 cuerpos
CAIDA_MAXIMA = 5 * TAM_PERSONAJE  # no se tira desde más de 5 cuerpos de altura
MARGEN = 0.9            # usa solo el 90% del salto teórico (saltos "cómodos")
CUADROS_ESPERA = 45     # ~0.75 s parado en cada post-it antes del siguiente salto
CUADROS_SALTO = 36      # ~0.6 s de animación de cada salto

# --- Trampolín: saltos que salen de un post-it con FLECHA HACIA ARRIBA ---
SUPER_VEL_SALTO = 23    # sube 23²/(2·0.8) ≈ 330 px; útil (×0.9) ≈ 300 px ≈ 8.5 cuerpos (normal: ~95)
SUPER_VEL_AVANCE = 8    # en plano alcanza ≈ 460 px; útil ≈ 410 px ≈ 12 cuerpos (normal: ~117)
CUADROS_SALTO_SUPER = 56  # ~0.9 s: el salto de trampolín se ve más alto y lento

PALETA = {
    "amarillo": (241, 196, 15),
    "verde": (46, 204, 113),
    "blanco": (255, 255, 255),
    "gris": (70, 70, 70),
    "personaje": (52, 152, 219),
    "rojo": (231, 76, 60),
    "naranjo": (243, 156, 18),
}


# ============================================================
# DETECCIÓN DE POST-ITS
# ============================================================
def mascara_amarillo(frame_bgr, calibracion=None):
    """Máscara binaria de píxeles amarillos, robusta a cambios de luz.

    Se trabaja en Lab: L = brillo, a = verde↔rojo, b = azul↔amarillo.
    Restar la mediana de la escena a 'a' y 'b' compensa que la cámara
    tiña todo el cuadro más cálido o más frío.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    a_rel = a.astype(np.int16) - int(np.median(a[::4, ::4]))   # mediana sobre 1 de cada 16
    b_rel = b.astype(np.int16) - int(np.median(b[::4, ::4]))   # píxeles: igual de robusta, 16x más rápida

    if calibracion is None:
        mascara = (b_rel > DELTA_B) & (a_rel > A_MIN) & (a_rel < A_MAX) & (L > L_MINIMO)
    else:
        ref_a, ref_b = calibracion
        mascara = ((np.abs(b_rel - ref_b) < TOL_CALIB_B)
                   & (np.abs(a_rel - ref_a) < TOL_CALIB_A)
                   & (L > L_MINIMO))

    mascara = mascara.astype(np.uint8) * 255
    k = np.ones((5, 5), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, k)   # quita puntitos
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, k)  # rellena huecos
    return mascara


def es_postit(contorno):
    """Condición explícita de forma: ¿esta mancha parece un post-it?"""
    area = cv2.contourArea(contorno)
    if not (AREA_MIN < area < AREA_MAX):
        return False
    (_, _), (w, h), _ = cv2.minAreaRect(contorno)
    if w == 0 or h == 0:
        return False
    if min(w, h) / max(w, h) < PROPORCION_MIN:          # post-it o fila de post-its
        return False
    if area / (w * h) < EXTENSION_MIN:                  # llena su rectángulo
        return False
    envolvente = cv2.contourArea(cv2.convexHull(contorno))
    if envolvente == 0 or area / envolvente < SOLIDEZ_MIN:  # sin entrantes
        return False
    aprox = cv2.approxPolyDP(contorno, 0.04 * cv2.arcLength(contorno, True), True)
    return VERTICES_MIN <= len(aprox) <= VERTICES_MAX   # ~4 esquinas


def detectar_postits(frame_bgr, calibracion=None):
    """Devuelve (rectángulos aceptados, máscara, contornos aceptados, rechazados)."""
    mascara = mascara_amarillo(frame_bgr, calibracion)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    aceptados, rechazados, rects = [], [], []
    alto, ancho = mascara.shape
    for c in contornos:
        x, y, w, h = cv2.boundingRect(c)
        toca_borde = x <= 2 or y <= 2 or x + w >= ancho - 2 or y + h >= alto - 2
        if not toca_borde and es_postit(c):  # lo que toca el borde (pared, objetos cortados) se ignora
            aceptados.append(c)
            rects.append(cv2.boundingRect(c))
        elif cv2.contourArea(c) > 200:
            rechazados.append(c)
    return rects, mascara, aceptados, rechazados


def color_en_punto(frame_bgr, x, y, radio=7):
    """Lee el color (a, b relativos a la escena) alrededor de un punto."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    h, w = lab.shape[:2]
    x0, x1 = max(0, x - radio), min(w, x + radio + 1)
    y0, y1 = max(0, y - radio), min(h, y + radio + 1)
    parche = lab[y0:y1, x0:x1].reshape(-1, 3)
    _, a_p, b_p = np.median(parche, axis=0)
    a_rel = int(a_p) - int(np.median(lab[:, :, 1]))
    b_rel = int(b_p) - int(np.median(lab[:, :, 2]))
    return a_rel, b_rel


# ============================================================
# ESTABILIDAD TEMPORAL
# ============================================================
def iou(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def _firma_color(lab, rect, medianas, anillo=False):
    """Color típico (L, a, b relativos a la escena) y textura de una zona.
    Si anillo=True mide el marco de pared alrededor del rectángulo."""
    alto, ancho = lab.shape[:2]
    x, y, w, h = [int(v) for v in rect]
    if anillo:
        x0, y0 = max(0, x - ANILLO_PX), max(0, y - ANILLO_PX)
        x1, y1 = min(ancho, x + w + ANILLO_PX), min(alto, y + h + ANILLO_PX)
        zona = lab[y0:y1, x0:x1].copy().astype(np.int16)
        zona[(y - y0):(y - y0 + h), (x - x0):(x - x0 + w)] = -1   # quita el centro
        pix = zona.reshape(-1, 3)
        pix = pix[pix[:, 0] >= 0]
    else:  # interior del post-it, recortando 15% de borde (bordes y sombras)
        mx, my = int(w * 0.15), int(h * 0.15)
        pix = lab[max(0, y + my):min(alto, y + h - my),
                  max(0, x + mx):min(ancho, x + w - mx)].reshape(-1, 3).astype(np.int16)
    if len(pix) == 0:
        return None
    med = np.median(pix, axis=0) - medianas
    return med, float(np.std(pix[:, 0]))


def _distancia_color(f1, f2):
    """Distancia en Lab relativo; el brillo pesa la mitad (tolera sombras suaves)."""
    d = f1 - f2
    return float(np.sqrt((0.5 * d[0]) ** 2 + d[1] ** 2 + d[2] ** 2))


class SeguidorPlataformas:
    """Mantiene plataformas estables y RECUERDA los post-its tapados.

    Estados de cada pista:
      visible  -> el detector lo está viendo.
      tapado   -> no se ve, pero en su lugar hay algo que no es pared
                  (o hay movimiento): se recuerda y sigue siendo plataforma.
      (borrada)-> se vio pared quieta en su lugar durante varios cuadros.
    """

    def __init__(self):
        self.pistas = []   # {"id","rect","vistos","perdidos","firma_pared","evid_sacado","estado"}
        self._siguiente_id = 0
        self._gris_anterior = None
        self._camara_reciente = 0
        self.camara_movida = False   # para mostrarlo en pantalla

    # ---------- utilidades de imagen ----------
    def _movimiento(self, frame_bgr):
        gris = cv2.GaussianBlur(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (7, 7), 0)
        if self._gris_anterior is None or self._gris_anterior.shape != gris.shape:
            mov = np.zeros_like(gris)
        else:
            mov = (cv2.absdiff(gris, self._gris_anterior) > UMBRAL_MOVIMIENTO).astype(np.uint8)
        self._gris_anterior = gris
        return mov

    @staticmethod
    def _movio_camara(mov):
        """Una mano mueve una zona; la cámara mueve TODA la imagen."""
        h, w = mov.shape
        cuadrantes = (mov[:h // 2, :w // 2], mov[:h // 2, w // 2:],
                      mov[h // 2:, :w // 2], mov[h // 2:, w // 2:])
        return all(c.mean() > CAMARA_FRACCION_CUADRANTE for c in cuadrantes)

    @staticmethod
    def _hay_movimiento(mov, rect):
        x, y, w, h = [int(v) for v in rect]
        x0, y0 = max(0, x - ANILLO_PX), max(0, y - ANILLO_PX)
        zona = mov[y0:y + h + ANILLO_PX, x0:x + w + ANILLO_PX]
        return zona.size > 0 and zona.mean() > FRACCION_MOVIMIENTO

    @staticmethod
    def _mancha_propia(rect, manchas):
        """Mancha amarilla rechazada por forma que calza con este post-it: se solapa
        bastante (IoU) y tiene un tamaño parecido. None si no hay ninguna."""
        if not manchas:
            return None
        area = rect[2] * rect[3]
        mejor, mejor_iou = None, IOU_MANCHA
        for m in manchas:
            v = iou(rect, m)
            if v >= mejor_iou and AREA_MANCHA[0] <= (m[2] * m[3]) / max(area, 1) <= AREA_MANCHA[1]:
                mejor, mejor_iou = m, v
        return mejor

    # ---------- ciclo por cuadro ----------
    def actualizar(self, detecciones, frame_bgr=None, manchas=None):
        if frame_bgr is not None:
            lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
            medianas = np.median(lab[::4, ::4].reshape(-1, 3), axis=0).astype(np.int16)
            mov = self._movimiento(frame_bgr)
        else:
            lab = None

        if lab is not None and self._movio_camara(mov):
            self._camara_reciente = CAMARA_ENFRIAMIENTO
        else:
            self._camara_reciente = max(0, self._camara_reciente - 1)
        self.camara_movida = self._camara_reciente > 0

        # Emparejar pistas con detecciones: 1° por solapamiento (IoU); 2° las que
        # quedaron solas, por cercanía de centros (post-it corrido por la cámara)
        usadas, asignacion = set(), {}
        for k, pista in enumerate(self.pistas):
            mejor, mejor_iou = None, IOU_MINIMO
            for i, d in enumerate(detecciones):
                if i in usadas:
                    continue
                v = iou(pista["rect"], d)
                if v > mejor_iou:
                    mejor, mejor_iou = i, v
            if mejor is not None:
                usadas.add(mejor)
                asignacion[k] = mejor
        for k, pista in enumerate(self.pistas):
            if k in asignacion or pista["vistos"] < CUADROS_PARA_APARECER:
                continue
            x, y, w, h = pista["rect"]
            cx, cy, limite = x + w / 2, y + h / 2, REENGANCHE * max(w, h)
            mejor, mejor_d = None, limite
            for i, (dx, dy, dw, dh) in enumerate(detecciones):
                if i in usadas:
                    continue
                dist = ((dx + dw / 2 - cx) ** 2 + (dy + dh / 2 - cy) ** 2) ** 0.5
                if dist < mejor_d:
                    mejor, mejor_d = i, dist
            if mejor is not None:
                usadas.add(mejor)
                asignacion[k] = mejor

        for k, pista in enumerate(self.pistas):
            mejor = asignacion.get(k)
            if mejor is not None:
                usadas.add(mejor)
                nuevo = np.array(detecciones[mejor], dtype=float)
                pista["rect"] = (1 - SUAVIZADO) * pista["rect"] + SUAVIZADO * nuevo
                pista["vistos"] += 1
                pista["perdidos"] = 0
                pista["evid_sacado"] = 0
                pista["estado"] = "visible"
                # ¿Hay algo moviéndose encima? (una mano daría una mala lectura del símbolo)
                pista["quieto"] = lab is None or not self._hay_movimiento(mov, pista["rect"])
                # Aprende cómo es la pared alrededor, SOLO con la escena quieta
                # (si no, una mano que se acerca contamina la firma de la pared)
                # y solo si el anillo se sigue pareciendo a la pared ya aprendida
                # (rechaza manos quietas al lado del post-it); se actualiza de a
                # poco para seguir cambios lentos de luz.
                if lab is not None and pista["quieto"]:
                    f = _firma_color(lab, pista["rect"], medianas, anillo=True)
                    if f is not None:
                        anterior = pista["firma_pared"]
                        if anterior is None:
                            pista["firma_pared"] = f[0].astype(float)
                        elif _distancia_color(f[0], anterior) < UMBRAL_PARED:
                            pista["firma_pared"] = 0.9 * anterior + 0.1 * f[0]
                continue

            # el detector no lo aceptó. ¿Sigue ahí, solo que deformado?
            # Si un post-it YA confirmado deja de pasar el filtro de forma (sombra,
            # perspectiva, cambio de luz, tocar el borde) pero en su lugar hay una mancha
            # amarilla de tamaño parecido y nada se mueve encima, sigue a la vista: no
            # está tapado. (Una mano encima borra el amarillo o se mueve.)
            confirmado = pista["vistos"] >= CUADROS_PARA_APARECER
            mancha = self._mancha_propia(pista["rect"], manchas) if confirmado else None
            if mancha is not None and (lab is None or not self._hay_movimiento(mov, pista["rect"])):
                pista["rect"] = (1 - SUAVIZADO) * pista["rect"] + SUAVIZADO * np.array(mancha, dtype=float)
                pista["perdidos"] = 0
                pista["evid_sacado"] = 0
                pista["estado"] = "visible"
                pista["quieto"] = True
                continue

            pista["perdidos"] += 1
            if not confirmado:
                pista["borrar"] = pista["perdidos"] > CANDIDATO_PERDIDOS_MAX
            elif lab is None or pista.get("firma_pared") is None:
                pista["borrar"] = pista["perdidos"] > CUADROS_PARA_DESAPARECER
            elif self.camara_movida:
                # La cámara se movió: su posición guardada ya no vale. Si no se
                # reenganchó con ninguna detección, se olvida (evita fantasmas).
                pista["borrar"] = True
            else:
                interior = _firma_color(lab, pista["rect"], medianas)
                parece_pared = (interior is not None
                                and _distancia_color(interior[0], pista["firma_pared"]) < UMBRAL_PARED
                                and interior[1] < UMBRAL_TEXTURA_PARED)
                if parece_pared and not self._hay_movimiento(mov, pista["rect"]):
                    pista["evid_sacado"] += 1           # pared quieta: probablemente lo sacaron
                else:
                    pista["evid_sacado"] = 0            # algo lo tapa
                pista["estado"] = "tapado"
                pista["borrar"] = (pista["evid_sacado"] >= CUADROS_CONFIRMAR_SACADO
                                   or pista["perdidos"] > TAPADO_MAXIMO)

        self.pistas = [p for p in self.pistas if not p.pop("borrar", False)]

        # Un "tapado" encima de uno visible es un fantasma del mismo post-it
        visibles = [p for p in self.pistas if p["estado"] == "visible"]
        def solape(r1, r2):
            x1, y1, w1, h1 = r1
            x2, y2, w2, h2 = r2
            ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            return ix * iy / max(1.0, min(w1 * h1, w2 * h2))
        self.pistas = [p for p in self.pistas
                       if p["estado"] != "tapado"
                       or not any(solape(p["rect"], v["rect"]) > SOLAPE_DUPLICADO for v in visibles)]
        for i, d in enumerate(detecciones):
            if i not in usadas:
                self.pistas.append({"id": self._siguiente_id, "rect": np.array(d, dtype=float),
                                    "vistos": 1, "perdidos": 0, "evid_sacado": 0,
                                    "estado": "visible", "firma_pared": None, "quieto": False})
                self._siguiente_id += 1

    def plataformas(self):
        """Plataformas fijas (visibles o tapadas): {id: pygame.Rect}. El id no cambia
        mientras el post-it siga ahí, así el personaje sabe sobre cuál está parado."""
        return {p["id"]: pygame.Rect(*[int(round(v)) for v in p["rect"]])
                for p in self.pistas if p["vistos"] >= CUADROS_PARA_APARECER}

    def tapados(self):
        return {p["id"] for p in self.pistas
                if p["vistos"] >= CUADROS_PARA_APARECER and p["estado"] == "tapado"}

    def quietos(self):
        """Ids de post-its confirmados, a la vista y sin nada moviéndose encima:
        los únicos donde vale la pena leer el símbolo dibujado."""
        if self.camara_movida:
            return set()
        return {p["id"] for p in self.pistas
                if p["vistos"] >= CUADROS_PARA_APARECER and p["estado"] == "visible"
                and p.get("quieto", False)}

    def candidatos(self):
        return [pygame.Rect(*[int(round(v)) for v in p["rect"]])
                for p in self.pistas if p["vistos"] < CUADROS_PARA_APARECER]


# ============================================================
# PERSONAJE
# ============================================================
def salto_posible(origen, destino, fuerte=False):
    """¿Puede un personaje tipo Mario saltar desde 'origen' hasta 'destino'?

    Se usa la parábola real del salto:
      altura máxima  = V² / (2·GRAVEDAD)
      tiempo en aire = (V + sqrt(V² - 2·g·subida)) / g   (rama de bajada)
      alcance        = avance * tiempo
    fuerte=True: el salto sale de un trampolín (post-it con flecha hacia arriba).
    """
    vel = SUPER_VEL_SALTO if fuerte else VEL_SALTO
    avance = SUPER_VEL_AVANCE if fuerte else VEL_AVANCE
    subida = origen.top - destino.top          # > 0 si el destino está más arriba
    altura_max = vel ** 2 / (2 * GRAVEDAD)
    if subida > MARGEN * altura_max:
        return False                           # demasiado alto
    if -subida > CAIDA_MAXIMA:
        return False                           # caída exagerada
    if destino.centerx >= origen.centerx:
        hueco = destino.left - origen.right
    else:
        hueco = origen.left - destino.right
    hueco = max(0, hueco)                      # si se solapan en x, no hay hueco
    tiempo = (vel + (vel ** 2 - 2 * GRAVEDAD * subida) ** 0.5) / GRAVEDAD
    return hueco <= MARGEN * avance * tiempo


class PersonajeAutomatico:
    """Cubo azul que se para y salta solo sobre los post-its.

    Tipos de post-it (los entrega simbolos.LectorSimbolos):
      "corona"        -> META: al pararse ahí el recorrido termina.
      "flecha_arriba" -> TRAMPOLÍN: los saltos que salen de ahí llegan más lejos y más alto.
      "limpio"/"otro" -> plataforma normal.
    """

    def __init__(self):
        self.rect = pygame.Rect(0, 0, TAM_PERSONAJE, TAM_PERSONAJE)
        self.visible = False
        self.plataforma = None   # id del post-it donde está parado
        self.direccion = 1       # +1 = hacia la derecha, -1 = hacia la izquierda
        self.espera = CUADROS_ESPERA
        self.salto = None        # {"desde": (x, y), "hacia": id, "t": cuadro, "fuerte": bool}
        self.meta = False        # True = está parado sobre la corona: camino terminado
        self.llego = False       # True = llegó a su post-it saltando (no lo pusieron ahí)
        self.en_marcha = False   # False = espera en el inicio a que se dé la partida (ESPACIO)

    def _volver_al_inicio(self):
        self.plataforma = None
        self.salto = None
        self.meta = False
        self.llego = False

    def reiniciar(self):
        """Tecla R: vuelve al inicio y espera otra partida (semáforo en amarillo)."""
        self._volver_al_inicio()
        self.en_marcha = False

    @staticmethod
    def _inicio(ordenadas, tipos):
        """Post-it de partida: el de más a la izquierda que NO sea la meta."""
        no_meta = [i for i in ordenadas if tipos.get(i) != "corona"]
        return (no_meta or ordenadas)[0]

    @staticmethod
    def _vecinos_alcanzables(i, ordenadas, plataformas, tipos, direcciones=(1, -1)):
        """Direcciones (+1/-1) hacia vecinos a los que puede saltar desde ordenadas[i]."""
        origen = plataformas[ordenadas[i]]
        fuerte = tipos.get(ordenadas[i]) == "flecha_arriba"   # parado en un trampolín
        return [d for d in direcciones
                if 0 <= i + d < len(ordenadas)
                and salto_posible(origen, plataformas[ordenadas[i + d]], fuerte)]

    def semaforo(self, plataformas, tipos=None):
        """"rojo": menos de 2 post-its, o desde el inicio no puede saltar a ninguna parte.
        "amarillo": preparado, esperando la partida. "verde": en marcha."""
        tipos = tipos or {}
        if len(plataformas) < 2:
            return "rojo"
        if self.en_marcha:
            return "verde"
        ordenadas = sorted(plataformas, key=lambda i: plataformas[i].centerx)
        i = ordenadas.index(self._inicio(ordenadas, tipos))
        return "amarillo" if self._vecinos_alcanzables(i, ordenadas, plataformas, tipos) else "rojo"

    def arrancar(self, plataformas, tipos=None):
        """Tecla ESPACIO: da la partida solo si el semáforo está en amarillo."""
        if self.semaforo(plataformas, tipos) != "amarillo":
            return False
        self.en_marcha = True
        self.espera = CUADROS_ESPERA // 3     # primer salto al tiro, sin la espera completa
        return True

    def actualizar(self, plataformas, tipos=None):
        """plataformas: {id: Rect}; tipos: {id: tipo} (si falta un id, es "limpio").
        Llamar una vez por cuadro."""
        tipos = tipos or {}
        ordenadas = sorted(plataformas, key=lambda i: plataformas[i].centerx)
        n = len(ordenadas)

        # Menos de 2 post-its: no hay personaje (y el semáforo vuelve a rojo)
        if n < 2:
            self.visible = False
            self.reiniciar()
            return
        self.visible = True

        # En pleno salto: avanzar la animación
        if self.salto is not None:
            destino = plataformas.get(self.salto["hacia"])
            if destino is None:            # sacaron el post-it de destino en el aire
                self._volver_al_inicio()
            else:
                self._animar_salto(destino)
                return

        # Sin plataforma (recién aparece o le sacaron la suya): al de más a la
        # izquierda que NO sea la meta (si todos son coronas, al de más a la izquierda).
        # También si lo PUSIERON sobre un post-it que recién se leyó como corona: el
        # símbolo se lee ~30 cuadros después de que aparece el cubo, y la meta hay que
        # alcanzarla saltando.
        hay_no_meta = any(tipos.get(i) != "corona" for i in ordenadas)
        puesto_en_meta = (not self.llego and tipos.get(self.plataforma) == "corona" and hay_no_meta)
        # Sin partida dada, el cubo espera siempre en el inicio (que puede cambiar si
        # se agregan o sacan post-its)
        esperando_en_otro = not self.en_marcha and self.plataforma != self._inicio(ordenadas, tipos)
        if self.plataforma not in plataformas or puesto_en_meta or esperando_en_otro:
            self.plataforma = self._inicio(ordenadas, tipos)
            self.direccion = 1
            self.espera = CUADROS_ESPERA
            self.llego = False

        actual = plataformas[self.plataforma]
        self.rect.midbottom = (actual.centerx, actual.top)

        # Parado sobre la corona: llegó a la meta y ya no salta más
        self.meta = tipos.get(self.plataforma) == "corona"
        # Sin partida (semáforo amarillo o rojo) se queda quieto en el inicio
        if self.meta or not self.en_marcha:
            return
        self.espera -= 1
        if self.espera > 0:
            return
        self.espera = CUADROS_ESPERA
        i = ordenadas.index(self.plataforma)
        fuerte = tipos.get(self.plataforma) == "flecha_arriba"   # parado en un trampolín
        # Vecinos alcanzables: primero sigue en su dirección, si no puede se devuelve
        vecinos = self._vecinos_alcanzables(i, ordenadas, plataformas, tipos,
                                            (self.direccion, -self.direccion))
        # Si la meta está al lado y la alcanza, va directo a ella
        hacia_meta = [d for d in vecinos if tipos.get(ordenadas[i + d]) == "corona"]
        if vecinos:
            d = (hacia_meta or vecinos)[0]
            self.direccion = d
            self.salto = {"desde": self.rect.midbottom, "hacia": ordenadas[i + d], "t": 0,
                          "fuerte": fuerte}
        # Ningún vecino alcanzable: se queda esperando en su post-it

    def _animar_salto(self, destino):
        self.salto["t"] += 1
        fuerte = self.salto.get("fuerte", False)
        s = min(1.0, self.salto["t"] / (CUADROS_SALTO_SUPER if fuerte else CUADROS_SALTO))
        x0, y0 = self.salto["desde"]
        x1, y1 = destino.centerx, destino.top
        extra = 60 if fuerte else 30           # el trampolín se eleva más
        arco = max(extra, y0 - y1 + extra)     # sube un poco más que el desnivel
        x = x0 + (x1 - x0) * s
        y = y0 + (y1 - y0) * s - 4 * arco * s * (1 - s)
        self.rect.midbottom = (int(x), int(y))
        if s >= 1.0:
            self.plataforma = self.salto["hacia"]
            self.salto = None
            self.espera = CUADROS_ESPERA
            self.llego = True


# ============================================================
# CÁMARA
# ============================================================
def abrir_camara():
    if isinstance(FUENTE_CAMARA, int) and sys.platform.startswith("win"):
        cam = cv2.VideoCapture(FUENTE_CAMARA, cv2.CAP_DSHOW)
        if cam.isOpened():
            return cam
    cam = cv2.VideoCapture(FUENTE_CAMARA)
    return cam if cam.isOpened() else None


def restaurar_automatico(cam):
    """Devuelve exposición y balance de blancos a automático.
    Se llama al iniciar porque algunos drivers recuerdan el modo manual
    de una ejecución anterior (la cámara quedaría oscura)."""
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # DirectShow: 0.75 = automático
    cam.set(cv2.CAP_PROP_AUTO_WB, 1)


def fijar_exposicion_actual(cam):
    """Congela la exposición en el valor que la cámara usa AHORA (tecla E).
    Úsese solo con la cámara ya apuntando a la pizarra. No todas las
    webcams lo soportan."""
    valor = cam.get(cv2.CAP_PROP_EXPOSURE)
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # DirectShow: 0.25 = manual
    cam.set(cv2.CAP_PROP_EXPOSURE, valor)
    cam.set(cv2.CAP_PROP_AUTO_WB, 0)


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================
def texto(pantalla, fuente, msg, pos, color=PALETA["blanco"]):
    pantalla.blit(fuente.render(msg, True, (0, 0, 0)), (pos[0] + 2, pos[1] + 2))
    pantalla.blit(fuente.render(msg, True, color), pos)


def mostrar_ventana_cv(nombre, imagen):
    """cv2.imshow que no se cae si OpenCV no tiene ventanas (pasa cuando están
    instalados opencv-python y opencv-python-headless a la vez). True si se mostró."""
    try:
        cv2.imshow(nombre, imagen)
        cv2.waitKey(1)
        return True
    except cv2.error:
        return False


def cerrar_ventanas_cv():
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def dibujar_semaforo(pantalla, estado):
    """Semáforo arriba a la derecha: rojo = no preparado, amarillo = listo (ESPACIO), verde = en marcha."""
    caja = pygame.Rect(ANCHO - 52, 10, 40, 112)
    pygame.draw.rect(pantalla, (30, 30, 30), caja, border_radius=8)
    pygame.draw.rect(pantalla, PALETA["blanco"], caja, 2, border_radius=8)
    luces = (("rojo", (231, 76, 60)), ("amarillo", (241, 196, 15)), ("verde", (46, 204, 113)))
    for k, (nombre, color) in enumerate(luces):
        centro = (caja.centerx, caja.y + 20 + 36 * k)
        encendida = nombre == estado
        pygame.draw.circle(pantalla, color if encendida else tuple(c // 4 for c in color), centro, 13)
        if encendida:   # un brillo para que se note cuál está prendida
            pygame.draw.circle(pantalla, PALETA["blanco"], centro, 13, 2)


def lineas_simbolos(lector, capturador):
    """Líneas de estado de los símbolos para la pantalla. Si no hay modelo por algo
    distinto de "falta entrenarlo", agrega el motivo."""
    if lector.activo:
        lineas = ["símbolos: modelo OK"]
    else:
        lineas = ["símbolos: sin modelo — clic derecho para marcar"]
        if lector.mensaje and lector.mensaje != simbolos.MSG_SIN_ARCHIVO:
            lineas.append(f"motivo: {lector.mensaje[:70]}")
    if capturador.activo:
        lineas[0] += f"  | CAPTURANDO ({capturador.guardados})"
    return lineas


def dibujar_icono(pantalla, rect, tipo):
    """Dibuja el símbolo del post-it: corona (meta) o flecha hacia arriba (trampolín)."""
    negro = (20, 20, 20)
    x, y, w, h = rect
    g = max(2, min(w, h) // 14)
    if tipo == "corona":
        m = w // 5
        puntos = [(x + m, y + h - m), (x + m, y + h // 3), (x + w // 3, y + h // 2),
                  (x + w // 2, y + m), (x + 2 * w // 3, y + h // 2), (x + w - m, y + h // 3),
                  (x + w - m, y + h - m)]
        pygame.draw.polygon(pantalla, negro, puntos, g)
    elif tipo == "flecha_arriba":
        cx, arriba, abajo, punta = x + w // 2, y + h // 5, y + h - h // 5, w // 5
        pygame.draw.line(pantalla, negro, (cx, abajo), (cx, arriba), g)
        pygame.draw.polygon(pantalla, negro, [(cx, arriba - g), (cx - punta, arriba + punta),
                                              (cx + punta, arriba + punta)])


def main():
    pygame.init()
    pantalla = pygame.display.set_mode((ANCHO, ALTO))
    pygame.display.set_caption("TEL-328: Plataformas con Post-it Amarillo")
    reloj = pygame.time.Clock()
    fuente = pygame.font.SysFont(None, 28)
    fuente_grande = pygame.font.SysFont(None, 52)

    # Lector de símbolos (sin modelo entrenado, el juego sigue sin símbolos)
    clasificar, msg_modelo = simbolos.cargar_clasificador()
    print("Símbolos:", msg_modelo)
    lector = simbolos.LectorSimbolos(clasificar, msg_modelo)
    capturador = simbolos.Capturador()
    cuadro = 0

    camara = abrir_camara()
    if camara is None:
        print("Error: no se pudo abrir la cámara.")
        pygame.quit()
        sys.exit()

    restaurar_automatico(camara)
    seguidor = SeguidorPlataformas()
    jugador = PersonajeAutomatico()

    calibracion = None
    congelado = False
    depuracion = False
    exposicion_fija = False
    grabador = None
    frame = None

    while True:
        ok, crudo = camara.read()
        if ok:
            frame = cv2.flip(cv2.resize(crudo, (ANCHO, ALTO)), 1)
            if grabador is not None:
                grabador.write(frame)

        # --- Eventos ---
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                if grabador is not None:
                    grabador.release()
                camara.release()
                cerrar_ventanas_cv()
                pygame.quit()
                sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_r:
                    jugador.reiniciar()
                elif ev.key == pygame.K_SPACE:   # semáforo en amarillo -> verde: parte el cubo
                    actuales = seguidor.plataformas()
                    if not jugador.arrancar(actuales, lector.tipos(actuales)):
                        print("No se puede partir: el semáforo está en", jugador.semaforo(actuales, lector.tipos(actuales)))
                elif ev.key == pygame.K_f:
                    congelado = not congelado
                elif ev.key == pygame.K_c:
                    calibracion = None
                elif ev.key == pygame.K_e:
                    exposicion_fija = not exposicion_fija
                    if exposicion_fija:
                        fijar_exposicion_actual(camara)
                    else:
                        restaurar_automatico(camara)
                elif ev.key == pygame.K_g:
                    if grabador is None:
                        os.makedirs("grabaciones", exist_ok=True)
                        ruta = os.path.join("grabaciones", time.strftime("desafio_%Y%m%d_%H%M%S.mp4"))
                        grabador = cv2.VideoWriter(ruta, cv2.VideoWriter_fourcc(*"mp4v"), 30, (ANCHO, ALTO))
                        print("Grabando en", ruta)
                    else:
                        grabador.release()
                        grabador = None
                        print("Grabación detenida")
                elif ev.key == pygame.K_d:
                    depuracion = not depuracion
                    if not depuracion:
                        cerrar_ventanas_cv()
                elif ev.key == pygame.K_s:
                    if capturador.alternar():
                        print("Captura de recortes ENCENDIDA en", capturador.carpeta)
                    else:
                        print(f"Captura apagada ({capturador.guardados} recortes en la sesión)")
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and frame is not None:
                calibracion = color_en_punto(frame, *ev.pos)
                print(f"Calibrado: a_rel={calibracion[0]}, b_rel={calibracion[1]}")
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:   # clic derecho: tipo a mano
                actuales = seguidor.plataformas()
                orden = sorted(actuales, key=lambda i: actuales[i].centerx)
                for pid in orden:
                    if actuales[pid].collidepoint(ev.pos):
                        letra = chr(ord("A") + orden.index(pid) % 26)
                        print(f"Post-it {letra}: {lector.rotar_manual(pid)}")
                        break

        # --- Detección ---
        if frame is not None and not congelado:
            rects, mascara, aceptados, rechazados = detectar_postits(frame, calibracion)
            seguidor.actualizar(rects, frame, [cv2.boundingRect(c) for c in rechazados])
            if depuracion:
                vista = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(vista, rechazados, -1, (0, 0, 255), 2)
                cv2.drawContours(vista, aceptados, -1, (0, 255, 0), 2)
                if not mostrar_ventana_cv("Depuracion: verde=post-it, rojo=rechazado", vista):
                    depuracion = False
                    print("La ventana de depuración no está disponible: OpenCV sin ventanas "
                          "(desinstala opencv-python-headless y reinstala opencv-python)")

        plataformas = seguidor.plataformas()

        # --- Símbolos (y captura de recortes para entrenar) ---
        if frame is not None and not congelado:
            quietos = seguidor.quietos()
            lector.actualizar(frame, plataformas, quietos, cuadro)
            capturador.actualizar(frame, plataformas, quietos, cuadro)
        cuadro += 1
        tipos = lector.tipos(plataformas)

        tapados = seguidor.tapados()
        ordenadas = sorted(plataformas, key=lambda i: plataformas[i].centerx)

        # --- Personaje ---
        jugador.actualizar(plataformas, tipos)

        # --- Dibujo ---
        if frame is not None:
            fondo = (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) * 0.6).astype(np.uint8)
            pantalla.blit(pygame.image.frombuffer(fondo.tobytes(), (ANCHO, ALTO), "RGB"), (0, 0))
        else:
            pantalla.fill((25, 25, 30))

        # Enlaces entre post-its vecinos: verde = ida y vuelta, naranjo = solo un
        # sentido (p. ej. solo bajar, o solo con trampolín), rojo = imposible
        for a, b in zip(ordenadas, ordenadas[1:]):
            ra, rb = plataformas[a], plataformas[b]
            ida = salto_posible(ra, rb, tipos[a] == "flecha_arriba")
            vuelta = salto_posible(rb, ra, tipos[b] == "flecha_arriba")
            color = PALETA["verde"] if ida and vuelta else PALETA["naranjo"] if ida or vuelta else PALETA["rojo"]
            pygame.draw.line(pantalla, color, ra.midtop, rb.midtop, 3)
        for c in seguidor.candidatos():  # post-its "por confirmar"
            pygame.draw.rect(pantalla, PALETA["amarillo"], c, 1, border_radius=3)
        for k, pid in enumerate(ordenadas):
            p = plataformas[pid]
            if pid in tapados:   # recordado: se dibuja hueco, con borde amarillo grueso
                pygame.draw.rect(pantalla, PALETA["amarillo"], p, 4, border_radius=3)
            else:
                pygame.draw.rect(pantalla, PALETA["amarillo"], p, border_radius=3)
                pygame.draw.rect(pantalla, PALETA["blanco"], p, 2, border_radius=3)
            letra = fuente.render(chr(ord("A") + k % 26), True, (0, 0, 0))
            if tipos[pid] in simbolos.SIMBOLOS:      # con ícono, la letra va a la esquina
                dibujar_icono(pantalla, p, tipos[pid])
                pantalla.blit(letra, (p.x + 4, p.y + 2))
            else:
                pantalla.blit(letra, letra.get_rect(center=p.center))

        if jugador.visible:
            pygame.draw.rect(pantalla, PALETA["personaje"], jugador.rect, border_radius=4)
            pygame.draw.rect(pantalla, PALETA["blanco"], jugador.rect, 2, border_radius=4)
        if jugador.meta:
            msg = fuente_grande.render("¡META!", True, PALETA["verde"])
            pantalla.blit(msg, msg.get_rect(center=(ANCHO // 2, 150)))

        estado = ("CONGELADO" if congelado else "en vivo") + " | " + \
                 ("calibrado" if calibracion else "auto") + " | expo " + \
                 ("FIJA" if exposicion_fija else "auto") + (" | ● GRABANDO" if grabador else "") + \
                 (" | CÁMARA MOVIÉNDOSE" if seguidor.camara_movida else "")
        n = len(plataformas)
        luz = jugador.semaforo(plataformas, tipos)
        modo = ("sin personaje (faltan post-its)" if n < 2 else "META" if jugador.meta
                else "en marcha" if luz == "verde" else "en espera")
        texto(pantalla, fuente, f"Post-its: {n} ({len(tapados)} tapados) | personaje: {modo} | {estado}", (15, 12))
        texto(pantalla, fuente, "Clic: calibrar  C: auto  F: congelar  E: expo  D: debug  G: grabar  R: reiniciar",
              (15, 36), (200, 200, 200))
        texto(pantalla, fuente, "ESPACIO: partir  Clic der: tipo de post-it  S: capturar recortes",
              (15, 60), (200, 200, 200))
        for i, linea in enumerate(lineas_simbolos(lector, capturador)):   # estado (y motivo si falla)
            texto(pantalla, fuente, linea, (15, 84 + 24 * i), (200, 200, 200) if i == 0 else PALETA["naranjo"])
        dibujar_semaforo(pantalla, luz)
        if n < 2:
            aviso = "Pega al menos 2 post-its amarillos"
        elif luz == "rojo":
            aviso = "Camino imposible desde el inicio"
        elif luz == "amarillo":
            aviso = "Listo: presiona ESPACIO para partir"
        else:
            aviso = None
        if aviso:
            msg = fuente_grande.render(aviso, True, PALETA["blanco"])
            pantalla.blit(msg, msg.get_rect(center=(ANCHO // 2, ALTO - 40)))

        pygame.display.flip()
        reloj.tick(60)


if __name__ == "__main__":
    main()
