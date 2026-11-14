"""
TEL-328 · Banco de desafíos y pruebas de proyecto_v3.py

Un detector clásico no se "entrena" con pesos como una red neuronal, pero sí se
evalúa como uno: se le pasa un conjunto de pruebas con resultado esperado y se
mide cuántas aprueba. Este archivo hace eso.

Modos:

  python desafios.py
      1) Pruebas rápidas (reglas del cubo, lector de símbolos, captura,
         entrenamiento) con datos sintéticos: no necesitan cámara ni modelo.
      2) Desafíos sintéticos del detector (construidos sobre una foto real de la
         pizarra: desafios/pizarra_base.jpg) y compara:
        - "anterior": el seguidor sin razonamiento de oclusión
        - "tapado/sacado": el seguidor nuevo
      Imprime una tabla APRUEBA / FALLA por prueba y por desafío.

  python desafios.py --rapidas
      Solo las pruebas rápidas (sale con código 1 si alguna falla).

  python desafios.py --video grabaciones/archivo.mp4
      Pasa un video REAL (grabado con la tecla G en proyecto_v3.py) por el
      mismo pipeline e imprime, segundo a segundo, cuántos post-its ve, cuántos
      recuerda tapados y cuántos acaba de borrar. Ustedes anotan qué debía
      pasar en cada segundo y comparan.

  Agregar --guardar para escribir los videos de los desafíos en desafios/ y
  poder verlos.
"""
import importlib
import os
import re
import subprocess
import sys
import tempfile
import types

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # no abre ventana al importar
import cv2
import numpy as np
import pygame

import proyecto_v3 as P
import simbolos as S

ANCHO, ALTO = P.ANCHO, P.ALTO
FPS = 60
CARPETA = os.path.dirname(os.path.abspath(__file__))
FOTO_BASE = os.path.join(CARPETA, "desafios", "pizarra_base.jpg")
rng = np.random.default_rng(1)


# ============================================================
# ESCENA
# ============================================================
def preparar_escena():
    """Pared limpia (foto real sin los post-its) y un parche de post-it real."""
    if os.path.exists(FOTO_BASE):
        foto = cv2.resize(cv2.imread(FOTO_BASE), (ANCHO, ALTO))
        parche = foto[244:302, 194:252].copy()             # post-it real, sin bordes
        mascara = np.zeros(foto.shape[:2], np.uint8)
        mascara[225:320, 175:410] = 255                    # zona de los post-its originales
        pared = cv2.inpaint(foto, mascara, 7, cv2.INPAINT_TELEA)
    else:  # respaldo sintético si no está la foto
        pared = np.full((ALTO, ANCHO, 3), (177, 174, 175), np.uint8)
        parche = np.full((58, 58, 3), (123, 176, 184), np.uint8)
    return pared, parche


PARED, PARCHE = preparar_escena()
POSICIONES = [(150, 380), (290, 380), (430, 380)]   # A, B, C (x, y esquina sup. izq.)


def pegar_postit(img, pos, parche=PARCHE):
    x, y = pos
    h, w = parche.shape[:2]
    sombra = img[y + 4:y + h + 4, x + 3:x + w + 3]
    img[y + 4:y + h + 4, x + 3:x + w + 3] = (sombra * 0.75).astype(np.uint8)
    img[y:y + h, x:x + w] = parche


def dibujar_mano(img, cx, cy, color=(110, 150, 205), escala=1.0):
    """Mano esquemática: palma + 4 dedos + pulgar + antebrazo que sale por abajo."""
    s = escala
    capa = img.copy()
    cv2.rectangle(capa, (int(cx - 45 * s), int(cy + 40 * s)), (int(cx + 45 * s), ALTO), color, -1)
    cv2.ellipse(capa, (cx, cy), (int(55 * s), int(65 * s)), 0, 0, 360, color, -1)
    for i, dx in enumerate((-36, -12, 12, 36)):
        largo = (70, 85, 80, 62)[i] * s
        cv2.rectangle(capa, (int(cx + dx * s - 10 * s), int(cy - 40 * s - largo)),
                      (int(cx + dx * s + 10 * s), int(cy - 30 * s)), color, -1)
    cv2.ellipse(capa, (int(cx - 62 * s), int(cy + 5 * s)), (int(14 * s), int(40 * s)), 30, 0, 360, color, -1)
    sombreado = np.clip(capa.astype(np.int16) + rng.normal(0, 4, capa.shape), 0, 255).astype(np.uint8)
    diff = np.any(capa != img, axis=2)
    img[diff] = sombreado[diff]


def ruido(img):
    return np.clip(img.astype(np.int16) + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)


def recorrido(desde, hasta, t, t0, t1):
    """Interpola una posición entre t0 y t1 (en segundos)."""
    if t <= t0:
        return desde
    if t >= t1:
        return hasta
    s = (t - t0) / (t1 - t0)
    return (int(desde[0] + (hasta[0] - desde[0]) * s), int(desde[1] + (hasta[1] - desde[1]) * s))


FUERA = (900, 800)          # mano fuera de cuadro
SOBRE_B = (319, 430)        # mano tapando el post-it B


# ============================================================
# DESAFÍOS
# Cada desafío: función cuadro(t) -> imagen, duración (s) y una verificación
# sobre la historia de plataformas ("historia" = lista por cuadro de dicts).
# ============================================================
def escena_base(postits=(0, 1, 2), ganancia=1.0):
    img = PARED.copy()
    for i in postits:
        pegar_postit(img, POSICIONES[i])
    if ganancia != 1.0:
        img = np.clip(img * ganancia, 0, 255).astype(np.uint8)
    return img


def d_mano_rapida(t):
    img = escena_base()
    dibujar_mano(img, *recorrido(FUERA, SOBRE_B, t, 2.0, 2.3))
    if t > 2.3:
        img = escena_base()
        dibujar_mano(img, *recorrido(SOBRE_B, FUERA, t, 2.5, 2.8))
    return img


def d_mano_quieta(t):
    img = escena_base()
    if 2.0 <= t < 5.5:
        pos = recorrido(FUERA, SOBRE_B, t, 2.0, 2.4) if t < 5.0 else recorrido(SOBRE_B, FUERA, t, 5.0, 5.4)
        dibujar_mano(img, *pos)
    return img


def d_sacar_postit(t):
    return escena_base((0, 1, 2) if t < 2.0 else (0, 2))


def d_mano_saca_postit(t):
    """La mano llega, tapa B, se lo lleva y se va."""
    img = escena_base((0, 1, 2) if t < 2.6 else (0, 2))
    if 2.0 <= t < 3.4:
        pos = recorrido(FUERA, SOBRE_B, t, 2.0, 2.4) if t < 2.8 else recorrido(SOBRE_B, FUERA, t, 2.8, 3.3)
        dibujar_mano(img, *pos)
    return img


def d_manga_blanca(t):
    """Peor caso: algo del color de la pizarra tapa B por 2 s (manga blanca)."""
    img = escena_base()
    if 2.0 <= t < 4.5:
        pos = recorrido(FUERA, SOBRE_B, t, 2.0, 2.3) if t < 4.0 else recorrido(SOBRE_B, FUERA, t, 4.0, 4.4)
        dibujar_mano(img, *pos, color=(178, 176, 176), escala=1.3)
    return img


def d_cambio_luz(t):
    g = 1.0 if t < 2.0 else max(0.6, 1.0 - 0.4 * (t - 2.0))   # baja 40% en 1 s
    return escena_base(ganancia=g)


def d_agregar_postit(t):
    return escena_base((0, 2) if t < 2.0 else (0, 1, 2))


def d_mano_grande(t):
    """Mano grande tapa A y B a la vez por 2 s."""
    img = escena_base()
    if 2.0 <= t < 4.6:
        destino = (250, 450)
        pos = recorrido(FUERA, destino, t, 2.0, 2.4) if t < 4.2 else recorrido(destino, FUERA, t, 4.2, 4.6)
        dibujar_mano(img, *pos, escala=1.6)
    return img


def d_camara_se_mueve(t):
    """Se mueve el notebook de golpe: toda la imagen se corre ~110 px en 0.15 s
    (como en el video real), con desenfoque de movimiento."""
    s = min(1.0, max(0.0, (t - 2.0) / 0.15))
    m = np.float32([[1, 0, -100 * s], [0, 1, -45 * s]])
    if 0 < s < 1:
        return cv2.blur(cv2.warpAffine(escena_base(), m, (ANCHO, ALTO), borderMode=cv2.BORDER_REPLICATE), (25, 5))
    return cv2.warpAffine(escena_base(), m, (ANCHO, ALTO), borderMode=cv2.BORDER_REPLICATE)


# --- verificaciones ---
def nunca_menos_de(n):
    def v(hist):
        minimo = min(len(h) for h in hist[90:])   # tras 1.5 s (ya estabilizado)
        return minimo >= n, f"mínimo visto/recordado: {minimo} (esperado ≥ {n})"
    return v


def baja_a(n, desde_s, max_s):
    def v(hist):
        i0 = int(desde_s * FPS)
        for i in range(i0, len(hist)):
            if len(hist[i]) <= n:
                demora = (i - i0) / FPS
                return demora <= max_s, f"bajó a {n} en {demora:.2f} s (límite {max_s} s)"
        return False, f"nunca bajó a {n}"
    return v


def sube_a(n, desde_s, max_s):
    def v(hist):
        i0 = int(desde_s * FPS)
        for i in range(i0, len(hist)):
            if len(hist[i]) >= n:
                demora = (i - i0) / FPS
                return demora <= max_s, f"subió a {n} en {demora:.2f} s (límite {max_s} s)"
        return False, f"nunca subió a {n}"
    return v


def exactamente(n, desde_s):
    def v(hist):
        i0 = int(desde_s * FPS)
        cuentas = [len(h) for h in hist[90:]]
        maximo, final = max(cuentas), [len(h) for h in hist[i0:]]
        ok = maximo <= n and all(c == n for c in final)
        return ok, f"máximo {maximo} (sin duplicados si ≤ {n}); desde {desde_s} s: {min(final)}-{max(final)}"
    return v


DESAFIOS = [
    ("1. Mano pasa rápido sobre B (0.8 s)", d_mano_rapida, 4.0, nunca_menos_de(3)),
    ("2. Mano se queda 3 s sobre B", d_mano_quieta, 7.0, nunca_menos_de(3)),
    ("3. Mano grande tapa A y B (2 s)", d_mano_grande, 6.0, nunca_menos_de(3)),
    ("4. Cambio de luz (-40% en 1 s)", d_cambio_luz, 5.0, nunca_menos_de(3)),
    ("5. Se saca B (sin mano visible)", d_sacar_postit, 4.0, baja_a(2, 2.0, 1.0)),
    ("6. La mano se lleva B", d_mano_saca_postit, 5.0, baja_a(2, 2.8, 1.0)),
    ("7. Se agrega B", d_agregar_postit, 3.5, sube_a(3, 2.0, 0.5)),
    ("8. Manga blanca tapa B 2 s (peor caso)", d_manga_blanca, 6.0, nunca_menos_de(3)),
    ("9. Se mueve la cámara (sin fantasmas)", d_camara_se_mueve, 4.5, exactamente(3, 3.0)),
]


# ============================================================
# PRUEBAS RÁPIDAS
# Cada prueba es una función sin argumentos que devuelve (ok, detalle).
# ============================================================
TINTA = (25, 25, 25)  # plumón negro


def dibujar_corona(img, x, y, w, h, grosor, margen):
    """Corona de 3 puntas dentro del rectángulo (x, y, w, h), dejando 'margen' (fracción)."""
    m = int(margen * w)
    pts = np.array([[x + m, y + h - m], [x + m, y + m + h // 6], [x + w // 3, y + h // 2],
                    [x + w // 2, y + m], [x + 2 * w // 3, y + h // 2], [x + w - m, y + m + h // 6],
                    [x + w - m, y + h - m]], np.int32)
    cv2.polylines(img, [pts], True, TINTA, grosor, cv2.LINE_AA)


def dibujar_flecha(img, x, y, w, h, grosor, margen):
    """Flecha hacia arriba centrada en el rectángulo, dejando 'margen' (fracción)."""
    m = int(margen * h)
    cv2.arrowedLine(img, (x + w // 2, y + h - m), (x + w // 2, y + m), TINTA, grosor,
                    cv2.LINE_AA, tipLength=0.35)


def _postit_detectado(img, x, y):
    rects = P.detectar_postits(ruido(img))[0]
    return any(abs(rx - x) < 6 and abs(ry - y) < 6 for rx, ry, rw, rh in rects), f"rects={rects}"


def p_detector_con_corona():
    img = PARED.copy()
    pegar_postit(img, (150, 380))
    dibujar_corona(img, 150, 380, 58, 58, grosor=4, margen=0.22)
    return _postit_detectado(img, 150, 380)


def p_detector_con_flecha():
    img = PARED.copy()
    pegar_postit(img, (150, 380))
    dibujar_flecha(img, 150, 380, 58, 58, grosor=4, margen=0.18)
    return _postit_detectado(img, 150, 380)


def fila(presentes="ABCDE", x0=100, paso=130, y=400, lado=70):
    """{id: Rect}; id = posición (A=0). Post-its de 70 px con huecos de 60 px."""
    return {i: pygame.Rect(x0 + i * paso, y, lado, lado)
            for i, letra in enumerate("ABCDE") if letra in presentes}


def p_fisica_trampolin():
    f, alto = fila(), pygame.Rect(230, 250, 70, 70)            # 150 px más arriba que A
    casos = [P.salto_posible(f[0], f[1]),                       # hueco 60: normal sí
             not P.salto_posible(f[1], f[3]),                   # hueco 190 (sin C): normal no
             P.salto_posible(f[1], f[3], fuerte=True),          # trampolín sí
             not P.salto_posible(f[0], f[4], fuerte=True),      # hueco 450: ni con trampolín
             not P.salto_posible(f[0], alto),                   # 150 px de subida: normal no
             P.salto_posible(f[0], alto, fuerte=True)]          # trampolín sí
    return all(casos), str(casos)


def p_trampolin_mas_fuerte():
    """La flecha permite subir ~300 px y saltar ~410 px (el normal: ~95 y ~117)."""
    a = pygame.Rect(100, 400, 70, 70)
    casos = [P.salto_posible(a, pygame.Rect(230, 150, 70, 70), fuerte=True),      # sube 250 px
             not P.salto_posible(a, pygame.Rect(230, 150, 70, 70)),               # normal no
             P.salto_posible(a, pygame.Rect(550, 400, 70, 70), fuerte=True),      # hueco 380 px
             not P.salto_posible(a, pygame.Rect(230, 60, 70, 70), fuerte=True)]   # 340 px: ni así
    return all(casos), str(casos)


def p_semaforo():
    """Rojo: <2 post-its o el cubo no puede moverse desde el inicio. Amarillo: listo,
    esperando ESPACIO (no se mueve). Verde: en marcha. R vuelve a amarillo."""
    j, r = P.PersonajeAutomatico(), {}
    uno = fila("A")
    j.actualizar(uno, {})
    r["uno"] = j.semaforo(uno, {})
    sin_camino = fila("AC")                                   # hueco de 190 px desde el inicio
    j.actualizar(sin_camino, {})
    r["sin_camino"] = j.semaforo(sin_camino, {})
    r["sin_camino_flecha"] = j.semaforo(sin_camino, {0: "flecha_arriba"})
    dos = fila("AB")
    for _ in range(200):
        j.actualizar(dos, {})
    r["dos_espera"] = (j.semaforo(dos, {}), j.plataforma)     # amarillo y sigue en A
    r["parte"] = j.arrancar(dos, {})
    for _ in range(200):
        j.actualizar(dos, {})
    r["dos_marcha"] = (j.semaforo(dos, {}), j.plataforma != 0 or j.salto is not None)
    j.reiniciar()
    j.actualizar(dos, {})
    r["tras_R"] = j.semaforo(dos, {})
    esperado = {"uno": "rojo", "sin_camino": "rojo", "sin_camino_flecha": "amarillo",
                "dos_espera": ("amarillo", 0), "parte": True, "dos_marcha": ("verde", True),
                "tras_R": "amarillo"}
    return r == esperado, str(r)


def deformar(img, pos):
    """Pega una 'esquina' amarilla al post-it (sombra/perspectiva/luz, como en el video real
    del 29/10): la mancha deja de llenar su rectángulo (extensión ~0.71 < 0.80)."""
    x, y = pos
    h, w = PARCHE.shape[:2]
    color = tuple(int(c) for c in PARCHE[h // 2, w // 2])
    cv2.fillPoly(img, [np.array([[x + w, y + h - 20], [x + w + 30, y + h + 4], [x + w, y + h]], np.int32)], color)


def p_postit_deformado():
    """Un post-it ya confirmado que se ve deformado (no pasa el filtro de forma) pero sigue
    a la vista y sin mano encima NO debe quedar como 'tapado': sigue visible y legible."""
    seg = P.SeguidorPlataformas()
    for k in range(int(4 * FPS)):
        f = escena_base()
        if k >= 2 * FPS:
            deformar(f, POSICIONES[1])
        f = ruido(f)
        rects, _, _, rechazados = P.detectar_postits(f)
        seg.actualizar(rects, f, [cv2.boundingRect(c) for c in rechazados])
    plats = seg.plataformas()
    id_b = min(plats, key=lambda i: abs(plats[i].x - POSICIONES[1][0]))
    ok = len(plats) == 3 and not seg.tapados() and id_b in seg.quietos()
    return ok, f"plataformas={len(plats)} tapados={len(seg.tapados())} B quieto={id_b in seg.quietos()}"


def p_quietos():
    seg, quietos, id_b = P.SeguidorPlataformas(), [], None
    for k in range(int(2.4 * FPS)):
        frame = ruido(d_mano_quieta(k / FPS))
        seg.actualizar(P.detectar_postits(frame)[0], frame)
        quietos.append(seg.quietos())
        if k == 96:
            plats = seg.plataformas()
            id_b = min(plats, key=lambda i: abs(plats[i].x - 290))
    tapa_b = any(id_b not in q for q in quietos[120:144])      # mano entrando (2.0–2.4 s)
    return (len(quietos[96]) == 3 and tapa_b,
            f"quietos a 1.6 s = {len(quietos[96])}, B deja de estar quieto al moverse la mano = {tapa_b}")


def simular(plats, tipos, cuadros=900):
    """Lista de ids donde aterriza el cubo (el primero es el de inicio) y el personaje.
    Se da la partida (como apretar ESPACIO con el semáforo en amarillo) al primer cuadro."""
    j, aterr = P.PersonajeAutomatico(), []
    for k in range(cuadros):
        j.actualizar(plats, tipos)
        if k == 0:
            j.arrancar(plats, tipos)
        if j.plataforma is not None and (not aterr or aterr[-1] != j.plataforma):
            aterr.append(j.plataforma)
    return aterr, j


def p_r1_ida_y_vuelta():
    a, _ = simular(fila(), {})
    return a[:9] == [0, 1, 2, 3, 4, 3, 2, 1, 0], str(a)


def p_r2_sin_puente():
    a, _ = simular(fila("ABDE"), {})
    return set(a) == {0, 1}, str(a)


def p_r3_trampolin():
    a, _ = simular(fila("ABDE"), {1: "flecha_arriba"})
    return 3 in a and 1 not in a[a.index(3):], str(a)


def p_r4_meta_y_sacar_corona():
    plats = fila()
    a, j = simular(plats, {3: "corona"})
    ok1 = a == [0, 1, 2, 3] and j.meta
    del plats[3]
    j.actualizar(plats, {})
    return ok1 and j.plataforma == 0 and not j.meta, f"{a} meta={j.meta} plataforma={j.plataforma}"


def p_r5_corona_a_la_izquierda():
    a, j = simular(fila(), {0: "corona"})
    return a == [1, 0] and j.meta, str(a)


def p_r6_corona_leida_tarde():
    """Como en vivo: el cubo aparece antes de que se lea la corona (que está en A).
    No debe declarar META sin haber saltado: se cambia de inicio y llega saltando."""
    plats, j, a = fila(), P.PersonajeAutomatico(), []
    for k in range(900):
        j.actualizar(plats, {} if k < 40 else {0: "corona"})
        if k == 0:
            j.arrancar(plats, {})
        if j.plataforma is not None and (not a or a[-1] != j.plataforma):
            a.append(j.plataforma)
    return a[:3] == [0, 1, 0] and j.meta, f"{a} meta={j.meta}"


# --- lector de símbolos (con un clasificador FALSO: no necesita modelo ni GPU) ---
CUADRADO = pygame.Rect(150, 380, 58, 58)
GRIS = np.full((ALTO, ANCHO, 3), 170, np.uint8)
CORONA, FLECHA = (0.95, 0.02, 0.02, 0.01), (0.02, 0.95, 0.02, 0.01)
LIMPIO, OTRO = (0.01, 0.02, 0.95, 0.02), (0.01, 0.02, 0.02, 0.95)


def falso(*probs):
    """Clasificador falso: misma fila de probabilidades para cada recorte;
    guarda el tamaño de cada lote en f.lotes."""
    def f(recortes):
        f.lotes.append(len(recortes))
        return np.tile(np.array(probs, float), (len(recortes), 1))
    f.lotes = []
    return f


def leer(L, veces, desde=0):
    for k in range(veces):
        L.actualizar(GRIS, {0: CUADRADO}, {0}, desde + 10 * k)


def p_s1_votacion():
    L = S.LectorSimbolos(falso(*CORONA))
    leer(L, 2)
    antes = L.tipos({0: CUADRADO})[0]
    L.actualizar(GRIS, {0: CUADRADO}, {0}, 25)                       # no es múltiplo de 10
    leer(L, 1, desde=30)
    r = (antes, L.tipos({0: CUADRADO})[0], L.clasificar.lotes)
    return r == ("limpio", "corona", [1, 1, 1]), str(r)


def p_s2_simbolo_no_se_borra():
    L = S.LectorSimbolos(falso(*CORONA))
    leer(L, 5)
    r = []
    for probs, desde in ((OTRO, 50), (LIMPIO, 100), (FLECHA, 150)):
        L.clasificar = falso(*probs)
        leer(L, 5, desde)
        r.append(L.tipos({0: CUADRADO})[0])
    return r == ["corona", "corona", "flecha_arriba"], str(r)


def p_s3_no_clasifica():
    plats = {0: CUADRADO, 1: pygame.Rect(300, 380, 58, 58),          # 1: tapado (no está en quietos)
             2: pygame.Rect(450, 380, 200, 60),                       # 2: fila de post-its
             3: pygame.Rect(100, 100, 30, 30)}                        # 3: muy chico
    L = S.LectorSimbolos(falso(*CORONA))
    for k in range(3):
        L.actualizar(GRIS, plats, {0, 2, 3}, 10 * k)
    t = L.tipos(plats)
    ok = L.clasificar.lotes == [1, 1, 1] and t == {0: "corona", 1: "limpio", 2: "limpio", 3: "limpio"}
    return ok, f"lotes={L.clasificar.lotes} tipos={t}"


def p_s4_manual_y_olvido():
    L = S.LectorSimbolos(falso(*FLECHA))
    leer(L, 3)
    r1 = L.rotar_manual(0)
    t1 = L.tipos({0: CUADRADO})[0]
    resto = [L.rotar_manual(0) for _ in range(3)]
    t4 = L.tipos({0: CUADRADO})[0]
    L.actualizar(GRIS, {}, set(), 40)                                 # desapareció: se olvida
    t5 = L.tipos({0: CUADRADO})[0]
    r = ([r1] + resto, t1, t4, t5)
    return r == (["corona", "flecha_arriba", "limpio", "auto"], "corona", "flecha_arriba", "limpio"), str(r)


def p_s5_sin_modelo():
    L = S.LectorSimbolos(None)
    L.actualizar(GRIS, {0: CUADRADO}, {0}, 0)
    clas, msg = S.cargar_clasificador("no_existe.pt")
    ok = not L.activo and L.tipos({0: CUADRADO}) == {0: "limpio"} and clas is None and "sin modelo" in msg
    return ok, msg


def p_s6_error_en_vivo():
    def roto(recortes):
        raise RuntimeError("falla de prueba")
    L = S.LectorSimbolos(roto)
    L.actualizar(GRIS, {0: CUADRADO}, {0}, 0)
    return not L.activo, "lector desactivado sin caerse"


def p_s7_recortar():
    a = S.recortar(GRIS, CUADRADO)
    b = S.recortar(GRIS, (780, 10, 50, 50))                           # cortado por el borde
    c = S.recortar(GRIS, (900, 900, 50, 50))                          # fuera de la imagen
    ok = (a is not None and a.shape == (128, 128, 3) and b is not None and b.shape == (128, 128, 3)
          and c is None and S.es_clasificable(CUADRADO)
          and not S.es_clasificable(pygame.Rect(0, 0, 200, 60))
          and not S.es_clasificable(pygame.Rect(0, 0, 30, 30)))
    return ok, f"a={None if a is None else a.shape} b={None if b is None else b.shape} c={c}"


def p_s8_modelo_equivocado():
    # Un clasificador con OTRAS clases (el del Taller 2: taza/control/cuaderno), si está en este PC
    ruta = os.path.join(os.path.expanduser("~"), "yolo_runs_taller2", "clasificacion_experimento_1",
                        "weights", "best.pt")
    if not os.path.exists(ruta):
        return True, "omitida (no hay un modelo con otras clases en este PC)"
    clas, msg = S.cargar_clasificador(ruta)
    return clas is None and "clases" in msg, msg


def p_s10_torch_roto():
    """En Windows, un torch dañado lanza OSError (WinError 126) al importar ultralytics."""
    roto = types.ModuleType("ultralytics")

    def _falla(nombre):
        raise OSError("[WinError 126] fbgemm.dll (simulado)")
    roto.__getattr__ = _falla
    real = sys.modules.get("ultralytics")
    sys.modules["ultralytics"] = roto
    try:
        clas, msg = S.cargar_clasificador(os.path.join(S.CARPETA, "modelos", "yolo11n-cls.pt"))
    finally:
        if real is None:
            del sys.modules["ultralytics"]
        else:
            sys.modules["ultralytics"] = real
    return clas is None and "sin modelo" in msg, msg


def p_s11_hilos_opencv():
    """Importar Ultralytics deja a OpenCV con 1 hilo; cargar el modelo debe devolverle los suyos.
    Corre aparte (otro proceso) porque en este Ultralytics ya está importado."""
    codigo = ("import os, cv2, simbolos as S; cv2.setNumThreads(4); "
              "S.cargar_clasificador(os.path.join(S.CARPETA, 'modelos', 'yolo11n-cls.pt')); "
              "print(cv2.getNumThreads())")
    r = subprocess.run([sys.executable, "-c", codigo], cwd=CARPETA, capture_output=True, text=True, timeout=180)
    salida = r.stdout.strip().splitlines()
    hilos = salida[-1] if salida else r.stderr.strip()[-200:]
    return hilos == "4", f"hilos de OpenCV tras cargar el modelo: {hilos} (esperado 4)"


def p_s12_captura_misma_sesion():
    """Apagar y prender S dentro del mismo segundo no debe sobrescribir recortes."""
    original = S.time.strftime
    S.time.strftime = lambda *a: "20261108_120000"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            C = S.Capturador(tmp)
            C.alternar()
            C.actualizar(GRIS, {0: CUADRADO}, {0}, 0)
            C.alternar()
            C.alternar()
            C.actualizar(GRIS, {0: CUADRADO}, {0}, 0)
            nombres = sorted(os.listdir(tmp))
    finally:
        S.time.strftime = original
    return len(nombres) == 2, str(nombres)


def p_hud_motivo():
    """La pantalla dice POR QUÉ no hay modelo cuando no es simplemente que falta entrenarlo."""
    C = S.Capturador(tempfile.gettempdir())   # apagado: no escribe nada
    con = P.lineas_simbolos(S.LectorSimbolos(falso(*CORONA), "modelo OK"), C)
    sin = P.lineas_simbolos(S.LectorSimbolos(None, S.MSG_SIN_ARCHIVO), C)
    malo = P.lineas_simbolos(S.LectorSimbolos(None, "modelo con clases ['taza'], se esperaban [...]"), C)
    ok = (con == ["símbolos: modelo OK"] and sin == ["símbolos: sin modelo — clic derecho para marcar"]
          and len(malo) == 2 and "clases" in malo[1])
    return ok, f"{con} | {sin} | {malo}"


def p_s9_captura():
    with tempfile.TemporaryDirectory() as tmp:
        C = S.Capturador(tmp)
        plats, q = {0: CUADRADO, 2: pygame.Rect(450, 380, 200, 60)}, {0, 2}
        n0 = C.actualizar(GRIS, plats, q, 0)                          # apagado: no guarda
        C.alternar()
        n = (n0, C.actualizar(GRIS, plats, q, 0), C.actualizar(GRIS, plats, q, 1),
             C.actualizar(GRIS, plats, q, 30))
        nombres = sorted(os.listdir(tmp))
    ok = (n == (0, 1, 0, 1) and len(nombres) == 2
          and all(re.fullmatch(r"\d{8}_\d{6}_p0_000[01]\.png", x) for x in nombres))
    return ok, f"{n} {nombres}"


# --- entrenamiento ---
def p_e1_grupo():
    E = importlib.import_module("entrenar_simbolos")
    r = (E.grupo("20261108_153000_p3_0012.png"), E.grupo("foto_celular.jpg"))
    return r == ("20261108_153000_p3", "foto_celular"), str(r)


def p_e2_division_por_grupos():
    E = importlib.import_module("entrenar_simbolos")
    with tempfile.TemporaryDirectory() as tmp:
        for clase, grupos in (("corona", 5), ("limpio", 1)):
            os.makedirs(os.path.join(tmp, clase))
            for g in range(grupos):
                for n in range(4):
                    open(os.path.join(tmp, clase, f"20261108_15000{g}_p{g}_{n:04d}.png"), "wb").close()
        items = E.listar(tmp)
    train, val = E.dividir(items)
    errores, _ = E.revisar(items)
    g = lambda lst: {E.grupo(os.path.basename(r)) for r, c in lst if c == "corona"}
    ok = len(g(val)) == 1 and not (g(train) & g(val)) and any("limpio" in e for e in errores)
    return ok, f"val={g(val)} errores={errores}"


def p_e3_test_repetido():
    """Avisa si dataset_prueba repite post-its (grupos) de dataset: el test saldría inflado."""
    E = importlib.import_module("entrenar_simbolos")
    items = [("x/corona/20261108_150000_p1_0000.png", "corona"),
             ("x/corona/20261108_150000_p2_0000.png", "corona")]
    test = [("y/corona/20261108_150000_p2_0007.png", "corona"),
            ("y/corona/20261109_100000_p1_0000.png", "corona")]
    r = E.grupos_repetidos(items, test)
    return r == {"20261108_150000_p2"}, str(r)


def p_e5_grupo_roboflow():
    """Roboflow exporta '<nombre>_png.rf.<hash>.jpg': el grupo debe seguir siendo el mismo."""
    E = importlib.import_module("entrenar_simbolos")
    r = (E.grupo("20261108_120000_p3_0012_png.rf.c1fbdb86fdb5e09a120dff6ef1d534df.jpg"),
         E.grupo("foto_celular_jpg.rf.b66449efcbac95c044d07a959cf98f7f.jpg"))
    return r == ("20261108_120000_p3", "foto_celular"), str(r)


def p_e6_leer_roboflow():
    """Exportación 'Folder Structure': train+valid se juntan (se re-dividen por post-it);
    test se respeta."""
    E = importlib.import_module("entrenar_simbolos")
    archivos = {"train/corona": ["20261108_120000_p0_0000", "20261108_120000_p1_0003"],
                "valid/corona": ["20261108_120000_p0_0001"],
                "test/corona": ["20261110_090000_p0_0000"]}
    with tempfile.TemporaryDirectory() as tmp:
        for carpeta, nombres in archivos.items():
            os.makedirs(os.path.join(tmp, carpeta))
            for n in nombres:
                open(os.path.join(tmp, carpeta, f"{n}_png.rf.0a1b2c.jpg"), "wb").close()
        open(os.path.join(tmp, "README.roboflow.txt"), "w").close()
        items, test = E.leer_roboflow(tmp)
    grupos = sorted({E.grupo(r) for r, _ in items})
    ok = (len(items) == 3 and len(test) == 1
          and grupos == ["20261108_120000_p0", "20261108_120000_p1"])
    return ok, f"items={len(items)} test={len(test)} grupos={grupos}"


def p_e4_parametros():
    """Aumentos que no deben cambiar el símbolo: sin volteo vertical, sin borrado,
    y recorte aleatorio suave (80–100 % del área, para no cortar la punta de la flecha)."""
    E = importlib.import_module("entrenar_simbolos")
    p = E.PARAMETROS
    ok = (p["flipud"] == 0.0 and p["erasing"] == 0.0 and p["scale"] == 0.2
          and p["fliplr"] == 0.5 and p["imgsz"] == 96 and p["workers"] == 0)
    return ok, str(p)


PRUEBAS_RAPIDAS = [
    ("Detector: post-it con corona dibujada", p_detector_con_corona),
    ("Detector: post-it con flecha dibujada", p_detector_con_flecha),
    ("Física: trampolín alcanza más lejos y más alto", p_fisica_trampolin),
    ("Física: trampolín sube ~300 px y salta ~410 px", p_trampolin_mas_fuerte),
    ("Semáforo: rojo / amarillo / verde y ESPACIO", p_semaforo),
    ("Seguidor: post-its quietos", p_quietos),
    ("Seguidor: post-it deformado a la vista no es 'tapado'", p_postit_deformado),
    ("Reglas: A B C D E ida y vuelta", p_r1_ida_y_vuelta),
    ("Reglas: A B _ D E sin flecha no cruza", p_r2_sin_puente),
    ("Reglas: flecha en B cruza a D (sin volver)", p_r3_trampolin),
    ("Reglas: corona = META; sacarla vuelve al inicio", p_r4_meta_y_sacar_corona),
    ("Reglas: corona a la izquierda de todo", p_r5_corona_a_la_izquierda),
    ("Reglas: corona leída después de aparecer el cubo", p_r6_corona_leida_tarde),
    ("Símbolos: votación (≥3 lecturas, ≥0.85)", p_s1_votacion),
    ("Símbolos: otro/limpio no borran una corona", p_s2_simbolo_no_se_borra),
    ("Símbolos: no lee tapados, filas ni chicos", p_s3_no_clasifica),
    ("Símbolos: tipo manual y olvido", p_s4_manual_y_olvido),
    ("Símbolos: sin modelo el juego sigue", p_s5_sin_modelo),
    ("Símbolos: error en vivo desactiva el lector", p_s6_error_en_vivo),
    ("Símbolos: recorte y post-its clasificables", p_s7_recortar),
    ("Símbolos: rechaza modelo con otras clases", p_s8_modelo_equivocado),
    ("Símbolos: torch dañado no cae el juego", p_s10_torch_roto),
    ("Símbolos: cargar el modelo no frena a OpenCV", p_s11_hilos_opencv),
    ("Captura: tecla S guarda recortes cada 30 cuadros", p_s9_captura),
    ("Captura: apagar/prender S no sobrescribe", p_s12_captura_misma_sesion),
    ("Pantalla: muestra por qué no hay modelo", p_hud_motivo),
    ("Entrenamiento: grupo = post-it en una sesión", p_e1_grupo),
    ("Entrenamiento: división por grupos y revisión", p_e2_division_por_grupos),
    ("Entrenamiento: avisa si el test repite post-its", p_e3_test_repetido),
    ("Entrenamiento: aumentos que no cambian el símbolo", p_e4_parametros),
    ("Roboflow: reconoce los nombres que exporta", p_e5_grupo_roboflow),
    ("Roboflow: lee la exportación y respeta su test", p_e6_leer_roboflow),
]


# ============================================================
# EJECUCIÓN
# ============================================================
def correr(funcion, duracion, con_oclusion, guardar_como=None):
    seguidor = P.SeguidorPlataformas()
    historia = []
    escritor = None
    if guardar_como:
        escritor = cv2.VideoWriter(guardar_como, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (ANCHO, ALTO))
    for k in range(int(duracion * FPS)):
        frame = ruido(funcion(k / FPS))
        rects, _, _, rechazados = P.detectar_postits(frame)
        if con_oclusion:
            seguidor.actualizar(rects, frame, [cv2.boundingRect(c) for c in rechazados])
        else:   # "anterior": el seguidor de base, sin oclusión ni manchas
            seguidor.actualizar(rects, None)
        historia.append(seguidor.plataformas())
        if escritor:
            escritor.write(frame)
    if escritor:
        escritor.release()
    return historia


def modo_desafios(guardar=False):
    print(f"Base: {'foto real de la pizarra' if os.path.exists(FOTO_BASE) else 'pared sintética'}\n")
    print(f"{'Desafío':42s} | {'anterior':8s} | {'tapado/sacado':13s} | detalle (tapado/sacado)")
    print("-" * 110)
    total = {False: 0, True: 0}
    for nombre, funcion, dur, verificar in DESAFIOS:
        res = {}
        for oclusion in (False, True):
            archivo = None
            if guardar and oclusion:
                archivo = os.path.join(CARPETA, "desafios", nombre.split(".")[0] + "_desafio.mp4")
            ok, detalle = verificar(correr(funcion, dur, oclusion, archivo))
            res[oclusion] = (ok, detalle)
            total[oclusion] += ok
        marca = lambda ok: "APRUEBA" if ok else "FALLA"
        print(f"{nombre:42s} | {marca(res[False][0]):8s} | {marca(res[True][0]):13s} | {res[True][1]}")
    print("-" * 110)
    print(f"{'TOTAL':42s} | {total[False]}/{len(DESAFIOS)}      | {total[True]}/{len(DESAFIOS)}")


def modo_rapidas():
    """Corre PRUEBAS_RAPIDAS. Devuelve True si todas aprueban."""
    print(f"{'Prueba rápida':48s} | {'resultado':9s} | detalle")
    print("-" * 110)
    aprobadas = 0
    for nombre, prueba in PRUEBAS_RAPIDAS:
        try:
            ok, detalle = prueba()
        except Exception as e:  # una prueba que se cae cuenta como FALLA
            ok, detalle = False, f"{type(e).__name__}: {e}"
        aprobadas += bool(ok)
        print(f"{nombre:48s} | {'APRUEBA' if ok else 'FALLA':9s} | {str(detalle)[:140]}")
    print("-" * 110)
    print(f"{'TOTAL':48s} | {aprobadas}/{len(PRUEBAS_RAPIDAS)}\n")
    return aprobadas == len(PRUEBAS_RAPIDAS)


def modo_video(ruta):
    cap = cv2.VideoCapture(ruta)
    if not cap.isOpened():
        print("No se pudo abrir", ruta)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    seguidor = P.SeguidorPlataformas()
    clasificar, msg = S.cargar_clasificador()
    print("Símbolos:", msg)
    lector = S.LectorSimbolos(clasificar, msg)
    k, ids_previos = 0, set()
    print(f"{'seg':>5s} | {'plataformas':>11s} | {'tapados':>7s} | {'eventos':24s} | tipos")
    eventos = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (ANCHO, ALTO))
        rects, _, _, rechazados = P.detectar_postits(frame)
        seguidor.actualizar(rects, frame, [cv2.boundingRect(c) for c in rechazados])
        plats = seguidor.plataformas()
        lector.actualizar(frame, plats, seguidor.quietos(), k)
        ids = set(plats)
        eventos += [f"+{i}" for i in ids - ids_previos] + [f"-{i}" for i in ids_previos - ids]
        ids_previos = ids
        k += 1
        if k % int(round(fps)) == 0:
            tipos = " ".join(f"{i}:{t}" for i, t in sorted(lector.tipos(plats).items()) if t != "limpio")
            print(f"{k / fps:5.0f} | {len(ids):11d} | {len(seguidor.tapados()):7d} | "
                  f"{' '.join(eventos):24s} | {tipos}")
            eventos = []
    cap.release()
    print("\n+N = apareció la plataforma N, -N = se borró; tipos = post-its con símbolo "
          "(los demás son 'limpio'). Comparen con lo que hicieron en el video.")


if __name__ == "__main__":
    if "--video" in sys.argv:
        modo_video(sys.argv[sys.argv.index("--video") + 1])
    elif "--rapidas" in sys.argv:
        sys.exit(0 if modo_rapidas() else 1)
    else:
        modo_rapidas()
        modo_desafios(guardar="--guardar" in sys.argv)
