# Post-its con símbolos (proyecto_v3): plan de implementación

> **Para agentes:** SUB-SKILL REQUERIDA: usar superpowers:subagent-driven-development (recomendado) o superpowers:executing-plans para implementar este plan tarea por tarea. Los pasos usan casillas (`- [ ]`).

**Objetivo:** que el juego reconozca coronas (meta) y flechas arriba (trampolín) dibujadas en los post-its, con un clasificador YOLO11n-cls sobre el recorte de cada post-it, todo dentro de una carpeta `proyecto_v3/` ordenada.

**Arquitectura:** `proyecto_v3.py` (copia de v2) sigue detectando y siguiendo post-its por color. A `simbolos.LectorSimbolos` le pasa los post-its quietos y clasificables, y el lector devuelve un tipo por post-it. Con esos tipos, `PersonajeAutomatico` aplica las reglas de meta y trampolín. `entrenar_simbolos.py` divide el dataset por grupos y entrena. Todo se prueba con `desafios.py`.

**Tecnologías:** Python 3.13.3, OpenCV 5.0, pygame 2.6.1, NumPy, Ultralytics 8.4.197, PyTorch 2.14.1+cu126 (RTX 3050 Ti).

**Diseño:** `proyecto_v3/docs/diseno_simbolos.md`

**Sin git:** la carpeta no es un repositorio, así que no hay pasos de commit. Nada se borra.

**Rutas:**
- `RAIZ` = `<carpeta>\Proyecto_actualizado`
- `V3` = `RAIZ\proyecto_v3`
- Los comandos se corren con `V3` como carpeta actual, salvo que se indique otra.

## Restricciones globales

- **Clases**, en este orden (el alfabético, que es el que usa Ultralytics): `CLASES = ("corona", "flecha_arriba", "limpio", "otro")`.
- **Recorte:** quita 8 % de borde por lado, sin rotar, a 128×128. El modelo usa `imgsz=96`.
- **Clasificable:** lado menor ≥ 40 px y lado corto / lado largo ≥ 0,75. Además, el post-it debe estar confirmado, visible y quieto.
- **Frecuencia:** se clasifica cada 10 cuadros, todos los recortes en un solo lote.
- **Votación:** últimas 5 lecturas, mínimo 3, promedio de la clase ganadora ≥ 0,85. Un símbolo decidido (corona o flecha_arriba) solo lo reemplaza el otro símbolo.
- **Trampolín:** `SUPER_VEL_SALTO = 18`, `SUPER_VEL_AVANCE = 6`, `CUADROS_SALTO_SUPER = 48`. El salto normal se queda en 13 y 4, y `MARGEN = 0.9`.
- **Captura:** cada 30 cuadros, en `dataset/sin_clasificar/<sesion>_p<id>_<n:04d>.png`, con `sesion = time.strftime("%Y%m%d_%H%M%S")` y `n` contando las imágenes de la sesión.
- **Entrenamiento:**
  - Carpeta de salida fuera de OneDrive: `%USERPROFILE%\yolo_runs_proyecto`.
  - Parámetros: `imgsz=96, epochs=40, patience=10, batch=32, workers=0, seed=0, deterministic=True, flipud=0.0, fliplr=0.5, erasing=0.0`.
- **Textos en pantalla:** `símbolos: modelo OK` / `símbolos: sin modelo — clic derecho para marcar` y el banner `¡META!`.
- **Desafíos de v2:** los 9 siguen en 8/9. Solo falla "8. Manga blanca tapa B 2 s (peor caso)".
- **El juego nunca se cae por el modelo.** Si falta ultralytics, faltan los pesos, las clases no calzan o hay un error en vivo, corre sin símbolos.
- **Estilo:** código, comentarios y mensajes en español, igual que v2. Sin dependencias nuevas (no se usa pytest: las pruebas son funciones `p_*` en `desafios.py` que devuelven `(ok, detalle)`).

## Foco de revisión

1. **Mano quieta sobre una corona ya decidida por más de 1 s:** la corona se mantiene. Lo prueba `p_s2` (Tarea 4).
2. **Sacan la corona con el cubo en META:** el cubo vuelve al inicio y sale de META. Lo prueba `p_r4` (Tarea 3).
3. **`modelos/simbolos.pt` equivocado** (por ejemplo, el clasificador del Taller 2): se avisa con un mensaje que contiene "clases" y el juego sigue sin símbolos. Lo prueba `p_s8` (Tarea 4).
4. **Post-it cortado por el borde de la imagen:** `recortar` no se cae. Lo prueba `p_s7` (Tarea 4).
5. **Una clase con un solo grupo:** el entrenamiento se detiene con un error claro en vez de entrenar con val incompleto. Lo prueba `p_e2` (Tarea 6).

---

### Tarea 1: Carpetas, base v3 y corredor de pruebas rápidas

**Archivos:**
- Crear: `V3\modelos\`, `V3\dataset\sin_clasificar\` y `RAIZ\versiones_anteriores\`.
- Copiar:
  - `RAIZ\proyecto_v2.py` → `V3\proyecto_v3.py`
  - `RAIZ\desafios.py` → `V3\desafios.py`
  - `RAIZ\desafios\` → `V3\desafios\`
  - `RAIZ\grabaciones\` → `V3\grabaciones\`
  - `<carpeta>\TEL328_Taller_2_2026\yolo11n-cls.pt` → `V3\modelos\`
- Mover:
  - `RAIZ\dataset\` y `RAIZ\dataset_prueba\` → `V3\`
  - `proyecto.py`, `proyecto_menu.py`, `game.py`, `generador_obstaculos.py`, `pizarra.py`, `proyecto_v2.py`, `desafios.py`, `desafios\` y `grabaciones\` → `RAIZ\versiones_anteriores\`
- `Preguntas 3.pdf` se queda en `RAIZ`.
- Modificar:
  - `V3\proyecto_v3.py`, línea 2 del docstring: `TEL-328 · Proyecto PDI — versión 3 (post-its con símbolos)`.
  - `V3\desafios.py`: `import proyecto_v3 as P` y el docstring.

**Interfaces:**
- Produce, en `desafios.py`:
  - `PRUEBAS_RAPIDAS: list[tuple[str, Callable[[], tuple[bool, str]]]]`
  - `modo_rapidas() -> bool`: imprime una tabla APRUEBA/FALLA; si una prueba lanza una excepción, cuenta como FALLA con el texto de la excepción.
  - `dibujar_corona(img, x, y, w, h, grosor, margen)` y `dibujar_flecha(img, x, y, w, h, grosor, margen)`: las mismas figuras del spike, en negro (25, 25, 25).
- Línea de comandos:
  - `python desafios.py --rapidas`: solo las pruebas rápidas; sale con código 1 si alguna falla.
  - Sin argumentos: primero las rápidas y luego los 9 desafíos.
  - `--video` y `--guardar` siguen igual.

- [ ] **Paso 1: Copiar y mover.** Primero todas las copias, después los movimientos (`Copy-Item -Recurse`, `Move-Item`). Comprobar con `Get-ChildItem RAIZ, V3, RAIZ\versiones_anteriores` que la estructura coincide con la sección 2 del diseño.
- [ ] **Paso 2: Comprobar que v2 sigue funcionando.** Desde `RAIZ\versiones_anteriores` correr `python desafios.py`. Esperado: `TOTAL | 5/9 | 8/9`.
- [ ] **Paso 3: Editar `V3\desafios.py`.** Cambiar el import, agregar `dibujar_corona`, `dibujar_flecha`, `PRUEBAS_RAPIDAS`, `modo_rapidas` y los flags. Escribir estas pruebas de caracterización (prueban el detector actual, deberían aprobar de inmediato):

```python
def p_detector_con_corona():
    img = PARED.copy(); pegar_postit(img, (150, 380))
    dibujar_corona(img, 150, 380, 58, 58, grosor=4, margen=0.22)
    rects = P.detectar_postits(ruido(img))[0]
    return any(abs(x - 150) < 6 and abs(y - 380) < 6 for x, y, w, h in rects), f"rects={rects}"
# p_detector_con_flecha: igual, con dibujar_flecha(..., grosor=4, margen=0.18)
```

- [ ] **Paso 4: Correr todo.** Desde `V3`: `python desafios.py`. Esperado: 2/2 rápidas APRUEBA y `TOTAL | 5/9 | 8/9`. Si una rápida falla, el problema está en el dibujo de prueba (comparar con `prueba_dibujo.png` del spike), no en el detector.

---

### Tarea 2: Post-its quietos y física de trampolín (`proyecto_v3.py`)

**Archivos:**
- Modificar: `V3\proyecto_v3.py`, en la sección de configuración, en `SeguidorPlataformas` y en `salto_posible`.
- Pruebas: `V3\desafios.py`.

**Interfaces:**
- Produce:
  - `SeguidorPlataformas.quietos() -> set[int]`: ids confirmados, en estado "visible", sin movimiento en su zona y con `camara_movida` en False.
  - `SUPER_VEL_SALTO = 18`, `SUPER_VEL_AVANCE = 6` y `CUADROS_SALTO_SUPER = 48`.
  - `salto_posible(origen: pygame.Rect, destino: pygame.Rect, fuerte: bool = False) -> bool`: con `fuerte=True` usa las constantes SUPER en la parábola. Las llamadas que ya existen no cambian.

- [ ] **Paso 1: Escribir las pruebas que fallan** en `desafios.py`:

```python
def fila(presentes="ABCDE", x0=100, paso=130, y=400, lado=70):
    """{id: Rect}; id = posición (A=0). Post-its de 70 px con huecos de 60 px."""
    return {i: pygame.Rect(x0 + i * paso, y, lado, lado) for i, l in enumerate("ABCDE") if l in presentes}

def p_fisica_trampolin():
    f, alto = fila(), pygame.Rect(230, 250, 70, 70)            # 150 px más arriba que A
    casos = [P.salto_posible(f[0], f[1]),                       # hueco 60: normal sí
             not P.salto_posible(f[1], f[3]),                   # hueco 190 (sin C): normal no
             P.salto_posible(f[1], f[3], fuerte=True),          # trampolín sí
             not P.salto_posible(f[0], f[4], fuerte=True),      # hueco 450: ni con trampolín
             not P.salto_posible(f[0], alto), P.salto_posible(f[0], alto, fuerte=True)]
    return all(casos), str(casos)

def p_quietos():
    seg, quietos, ids_b = P.SeguidorPlataformas(), [], None
    for k in range(int(2.4 * FPS)):
        frame = ruido(d_mano_quieta(k / FPS))
        seg.actualizar(P.detectar_postits(frame)[0], frame)
        quietos.append(seg.quietos())
        if k == 96:
            plats = seg.plataformas()
            ids_b = min(plats, key=lambda i: abs(plats[i].x - 290))
    tapa_b = any(ids_b not in q for q in quietos[120:144])      # mano entrando (2.0–2.4 s)
    return len(quietos[96]) == 3 and tapa_b, f"quietos@1.6s={len(quietos[96])}, B sale al moverse={tapa_b}"
```

- [ ] **Paso 2:** `python desafios.py --rapidas`. Esperado: las dos pruebas nuevas en FALLA (`TypeError` por `fuerte` y `AttributeError` por `quietos`).
- [ ] **Paso 3: Implementar en `proyecto_v3.py`.**
  - En `SeguidorPlataformas.actualizar`, cada pista emparejada guarda `pista["quieto"] = lab is None or not self._hay_movimiento(mov, pista["rect"])`. Las pistas nuevas parten con `quieto` en False.
  - Agregar `quietos()`.
  - Agregar el parámetro `fuerte` a `salto_posible`.
- [ ] **Paso 4:** `python desafios.py`. Esperado: todas las rápidas APRUEBA y `TOTAL | 5/9 | 8/9`.

---

### Tarea 3: Reglas del personaje (meta, trampolín, inicio, prioridad a la corona)

**Archivos:**
- Modificar: `V3\proyecto_v3.py`, clase `PersonajeAutomatico`.
- Pruebas: `V3\desafios.py`.

**Interfaces:**
- Usa: `salto_posible(..., fuerte)` y `CUADROS_SALTO_SUPER` (Tarea 2).
- Produce:
  - `PersonajeAutomatico.actualizar(plataformas: dict[int, pygame.Rect], tipos: dict[int, str] | None = None) -> None`. Si un id no aparece en `tipos`, cuenta como `"limpio"`.
  - El atributo `meta: bool`.
  - `reiniciar()` ahora también pone `meta = False`.

**Reglas** (de la sección 5 del diseño):
- **Inicio:** el post-it más a la izquierda que no sea corona; si todos son coronas, el más a la izquierda.
- **META:** en cada cuadro en que el cubo no está en el aire, se cumple `meta = tipos.get(plataforma) == "corona"`. Con `meta` activo, o con menos de 3 post-its, no salta.
- **Elegir el salto:**
  1. Los candidatos son los vecinos `i + d`, con `d` igual a `direccion` o `-direccion`, que se pueden alcanzar con `salto_posible(actual, destino, fuerte=tipos.get(actual_id) == "flecha_arriba")`.
  2. Si algún candidato es corona, salta a ese.
  3. Si no, al primer candidato, igual que en v2.
- **Animación:** el salto guarda `"fuerte"`. Si es fuerte, dura `CUADROS_SALTO_SUPER` y el arco es `max(60, y0 - y1 + 60)`; si no, sigue igual que antes.

- [ ] **Paso 1: Escribir las pruebas que fallan:**

```python
def simular(plats, tipos, cuadros=900):
    """Lista de ids donde aterriza el cubo (el primero es el de inicio) y el personaje."""
    j, aterr = P.PersonajeAutomatico(), []
    for _ in range(cuadros):
        j.actualizar(plats, tipos)
        if j.plataforma is not None and (not aterr or aterr[-1] != j.plataforma):
            aterr.append(j.plataforma)
    return aterr, j

def p_r1_ida_y_vuelta():
    a, _ = simular(fila(), {});                       return a[:9] == [0, 1, 2, 3, 4, 3, 2, 1, 0], str(a)
def p_r2_sin_puente():
    a, _ = simular(fila("ABDE"), {});                 return set(a) == {0, 1}, str(a)
def p_r3_trampolin():
    a, _ = simular(fila("ABDE"), {1: "flecha_arriba"})
    return 3 in a and 1 not in a[a.index(3):], str(a)
def p_r4_meta_y_sacar_corona():
    plats = fila(); a, j = simular(plats, {3: "corona"})
    ok1 = a == [0, 1, 2, 3] and j.meta
    del plats[3]; j.actualizar(plats, {})
    return ok1 and j.plataforma == 0 and not j.meta, f"{a} meta={j.meta} plataforma={j.plataforma}"
def p_r5_corona_a_la_izquierda():
    a, j = simular(fila(), {0: "corona"});            return a == [1, 0] and j.meta, str(a)
```

- [ ] **Paso 2:** `python desafios.py --rapidas`. Esperado: `p_r1` y `p_r2` APRUEBA (v2 ya lo hacía) y `p_r3`, `p_r4` y `p_r5` en FALLA.
- [ ] **Paso 3:** implementar las reglas en `PersonajeAutomatico`.
- [ ] **Paso 4:** `python desafios.py`. Esperado: todas las rápidas APRUEBA y `TOTAL | 5/9 | 8/9`.

---

### Tarea 4: `simbolos.py`, recorte, clasificador y votación

**Archivos:**
- Crear: `V3\simbolos.py`.
- Pruebas: `V3\desafios.py` (`import simbolos as S`).

**Interfaces que produce** (`simbolos.py`):
- Constantes:
  - `CARPETA = os.path.dirname(os.path.abspath(__file__))`
  - `RUTA_MODELO = os.path.join(CARPETA, "modelos", "simbolos.pt")`
  - `CLASES` y las demás constantes de las Restricciones globales.
- `recortar(frame_bgr: np.ndarray, rect) -> np.ndarray | None`:
  - `rect` puede ser una tupla `(x, y, w, h)` o un `pygame.Rect`.
  - Quita el 8 % de borde, ajusta el recorte a los límites de la imagen y lo redimensiona a 128×128×3 BGR.
  - Devuelve `None` si queda vacío.
- `es_clasificable(rect) -> bool`.
- `cargar_clasificador(ruta: str = RUTA_MODELO) -> tuple[Callable | None, str]`:
  - Devuelve `clasificar(recortes: list[np.ndarray]) -> np.ndarray`, de forma `(n, 4)` y en el orden de `CLASES`, más un mensaje.
  - Sin archivo: `(None, "sin modelo (entrena con entrenar_simbolos.py)")`.
  - Si `import ultralytics` falla: `(None, "sin modelo: falta instalar ultralytics")`.
  - Si los nombres del modelo (ordenados por índice) no coinciden con `CLASES`: `(None, f"modelo con clases {nombres}, se esperaban {list(CLASES)}")`.
  - Si todo está bien: predice con `modelo.predict(recortes, imgsz=96, verbose=False)` y `r.probs.data.cpu().numpy()`, hace una predicción de calentamiento con un recorte negro y devuelve `(clasificar, "modelo OK")`.
  - Cualquier otra excepción al cargar: `(None, f"sin modelo: {e}")`.
  - Ultralytics se importa dentro de la función.
- `class LectorSimbolos`:
  - `__init__(self, clasificar=None, mensaje="")`.
  - Atributos: `clasificar` (se puede reasignar) y `mensaje`. Propiedad `activo` = `clasificar is not None`.
  - `actualizar(frame_bgr, plataformas: dict[int, Rect], quietos: set[int], cuadro: int) -> None`:
    - En cada llamada olvida el historial, el tipo decidido y el tipo manual de los ids que ya no están.
    - Si `activo` y `cuadro % 10 == 0`, clasifica en un lote los ids que están en `quietos` y son `es_clasificable`, y aplica la votación.
    - Si `clasificar` lanza una excepción: imprime `[simbolos] error al clasificar: …; se desactiva` y deja `clasificar = None`.
  - `tipos(plataformas) -> dict[int, str]`: para cada id, el tipo manual si lo hay; si no, el decidido; si no, `"limpio"`.
  - `rotar_manual(pid) -> str`: recorre auto → `"corona"` → `"flecha_arriba"` → `"limpio"` → `"auto"` y devuelve el valor nuevo.

- [ ] **Paso 1: Escribir las pruebas que fallan:**

```python
CUADRADO, GRIS = pygame.Rect(150, 380, 58, 58), np.full((ALTO, ANCHO, 3), 170, np.uint8)
CORONA, FLECHA = (0.95, 0.02, 0.02, 0.01), (0.02, 0.95, 0.02, 0.01)
LIMPIO, OTRO = (0.01, 0.02, 0.95, 0.02), (0.01, 0.02, 0.02, 0.95)
def falso(*probs):            # misma fila para cada recorte; guarda el tamaño de cada lote
    def f(recortes):
        f.lotes.append(len(recortes)); return np.tile(np.array(probs, float), (len(recortes), 1))
    f.lotes = []; return f
def leer(L, veces, desde=0):
    for k in range(veces): L.actualizar(GRIS, {0: CUADRADO}, {0}, desde + 10 * k)

def p_s1_votacion():
    L = S.LectorSimbolos(falso(*CORONA)); leer(L, 2); antes = L.tipos({0: CUADRADO})[0]
    L.actualizar(GRIS, {0: CUADRADO}, {0}, 25)                       # no es múltiplo de 10
    leer(L, 1, desde=30); r = (antes, L.tipos({0: CUADRADO})[0], L.clasificar.lotes)
    return r == ("limpio", "corona", [1, 1, 1]), str(r)
def p_s2_simbolo_no_se_borra():
    L = S.LectorSimbolos(falso(*CORONA)); leer(L, 5); r = []
    for probs, desde in ((OTRO, 50), (LIMPIO, 100), (FLECHA, 150)):
        L.clasificar = falso(*probs); leer(L, 5, desde); r.append(L.tipos({0: CUADRADO})[0])
    return r == ["corona", "corona", "flecha_arriba"], str(r)
def p_s3_no_clasifica():
    plats = {0: CUADRADO, 1: pygame.Rect(300, 380, 58, 58),          # 1: tapado (no está en quietos)
             2: pygame.Rect(450, 380, 200, 60), 3: pygame.Rect(100, 100, 30, 30)}   # fila, chico
    L = S.LectorSimbolos(falso(*CORONA))
    for k in range(3): L.actualizar(GRIS, plats, {0, 2, 3}, 10 * k)
    t = L.tipos(plats)
    return L.clasificar.lotes == [1, 1, 1] and t == {0: "corona", 1: "limpio", 2: "limpio", 3: "limpio"}, str(t)
def p_s4_manual_y_olvido():
    L = S.LectorSimbolos(falso(*FLECHA)); leer(L, 3)
    r1 = L.rotar_manual(0); t1 = L.tipos({0: CUADRADO})[0]
    resto = [L.rotar_manual(0) for _ in range(3)]; t4 = L.tipos({0: CUADRADO})[0]
    L.actualizar(GRIS, {}, set(), 40); t5 = L.tipos({0: CUADRADO})[0]      # desapareció: se olvida
    r = ([r1] + resto, t1, t4, t5)
    return r == (["corona", "flecha_arriba", "limpio", "auto"], "corona", "flecha_arriba", "limpio"), str(r)
def p_s5_sin_modelo():
    L = S.LectorSimbolos(None); L.actualizar(GRIS, {0: CUADRADO}, {0}, 0)
    clas, msg = S.cargar_clasificador("no_existe.pt")
    return not L.activo and L.tipos({0: CUADRADO}) == {0: "limpio"} and clas is None and "sin modelo" in msg, msg
def p_s6_error_en_vivo():
    def roto(recortes): raise RuntimeError("falla de prueba")
    L = S.LectorSimbolos(roto); L.actualizar(GRIS, {0: CUADRADO}, {0}, 0)
    return not L.activo, "lector desactivado sin caerse"
def p_s7_recortar():
    a, b, c = S.recortar(GRIS, CUADRADO), S.recortar(GRIS, (780, 10, 50, 50)), S.recortar(GRIS, (900, 900, 50, 50))
    ok = (a.shape == (128, 128, 3) and b.shape == (128, 128, 3) and c is None and S.es_clasificable(CUADRADO)
          and not S.es_clasificable(pygame.Rect(0, 0, 200, 60)) and not S.es_clasificable(pygame.Rect(0, 0, 30, 30)))
    return ok, "formas y bordes"
def p_s8_modelo_equivocado():
    ruta = r"%USERPROFILE%\yolo_runs_taller2\clasificacion_experimento_1\weights\best.pt"
    if not os.path.exists(ruta): return True, "omitida (no está el modelo del Taller 2)"
    clas, msg = S.cargar_clasificador(ruta)
    return clas is None and "clases" in msg, msg
```

- [ ] **Paso 2:** `python desafios.py --rapidas`. Esperado: `p_s1` a `p_s8` en FALLA (`ModuleNotFoundError: simbolos`).
- [ ] **Paso 3:** crear `simbolos.py` con las interfaces de arriba. La votación promedia las últimas 5 filas de probabilidad (las guarda un `deque(maxlen=5)` por id) y aplica la regla de las Restricciones globales.
- [ ] **Paso 4:** `python desafios.py --rapidas`. Esperado: todas las rápidas APRUEBA. `p_s8` tarda unos segundos porque carga Ultralytics.

---

### Tarea 5: Integración en el juego y Capturador

**Archivos:**
- Modificar: `V3\simbolos.py` (agrega `Capturador`) y `V3\proyecto_v3.py` (docstring, `main` y dibujo).
- Pruebas: `V3\desafios.py`.

**Interfaces:**
- Usa: `LectorSimbolos`, `cargar_clasificador` (Tarea 4), `quietos()` (Tarea 2) y `actualizar(plataformas, tipos)` y `meta` (Tarea 3).
- Produce:
  - `simbolos.CARPETA_CAPTURA = os.path.join(CARPETA, "dataset", "sin_clasificar")`.
  - `class Capturador`:
    - `__init__(self, carpeta: str | None = None)`: con `None` usa `CARPETA_CAPTURA` tal como esté al construir.
    - Atributos: `activo`, `guardados` y `sesion`.
    - `alternar() -> bool`: al activar crea la carpeta y una sesión nueva.
    - `actualizar(frame_bgr, plataformas, quietos, cuadro) -> int`: si está activa y `cuadro % 30 == 0`, guarda `recortar(...)` de los ids quietos y clasificables, y devuelve cuántos guardó.
  - `proyecto_v3.dibujar_icono(pantalla, rect: pygame.Rect, tipo: str) -> None`: corona = polígono negro de 3 puntas; flecha_arriba = línea y punta negras; los demás tipos no dibujan nada.

- [ ] **Paso 1: Escribir la prueba que falla:**

```python
def p_s9_captura():
    tmp = tempfile.mkdtemp(); C = S.Capturador(tmp)
    plats, q = {0: CUADRADO, 2: pygame.Rect(450, 380, 200, 60)}, {0, 2}
    n0 = C.actualizar(GRIS, plats, q, 0); C.alternar()
    n = (n0, C.actualizar(GRIS, plats, q, 0), C.actualizar(GRIS, plats, q, 1), C.actualizar(GRIS, plats, q, 30))
    nombres = sorted(os.listdir(tmp))
    ok = n == (0, 1, 0, 1) and len(nombres) == 2 and all(re.fullmatch(r"\d{8}_\d{6}_p0_000[01]\.png", x) for x in nombres)
    return ok, f"{n} {nombres}"
```

- [ ] **Paso 2:** `python desafios.py --rapidas`. Esperado: `p_s9` en FALLA.
- [ ] **Paso 3:** implementar `Capturador`.
- [ ] **Paso 4: Integrar en `proyecto_v3.main`.**
  - **Al iniciar:** `clasificar, msg = simbolos.cargar_clasificador()`; imprimir `Símbolos: {msg}`; crear `LectorSimbolos(clasificar, msg)`, `Capturador()` y un contador `cuadro = 0`, que sube 1 por vuelta del ciclo.
  - **Eventos:**
    - `K_s`: `capturador.alternar()` e imprimir el estado.
    - Clic derecho (`button == 3`): buscar el post-it con `rect.collidepoint(ev.pos)` e imprimir `Post-it {letra}: {lector.rotar_manual(pid)}`.
  - **En cada cuadro** (si hay frame y no está congelado), después del seguidor: `lector.actualizar(frame, plataformas, seguidor.quietos(), cuadro)` y `capturador.actualizar(...)` con los mismos argumentos. Luego `tipos = lector.tipos(plataformas)` y `jugador.actualizar(plataformas, tipos)`.
  - **Enlaces entre vecinos:** `ida = salto_posible(ra, rb, tipos[a] == "flecha_arriba")` y `vuelta = salto_posible(rb, ra, tipos[b] == "flecha_arriba")`.
  - **Dibujo:**
    - `dibujar_icono` sobre cada plataforma. Si hay ícono, la letra pasa a `p.topleft + (4, 2)`.
    - Tercera línea de texto en (15, 60): `símbolos: modelo OK` o `símbolos: sin modelo — clic derecho para marcar`, más `  ● CAPTURANDO ({guardados})` cuando la captura está activa.
    - Segunda línea: sumar `Clic der: tipo  S: capturar`.
    - Si `jugador.meta`: `¡META!` con `fuente_grande` en verde, centrado en (ANCHO // 2, 90).
  - **Docstring:** agregar los controles nuevos y las reglas de símbolos.
- [ ] **Paso 5:** `python desafios.py`. Esperado: todas las rápidas APRUEBA y `TOTAL | 5/9 | 8/9`.
- [ ] **Paso 6: Prueba de humo sin ventana**, con un script en el scratchpad:
  - Configuración: `SDL_VIDEODRIVER=dummy`; `P.FUENTE_CAMARA = r"V3\desafios\1_desafio.mp4"`; `S.CARPETA_CAPTURA = <tmp>`.
  - Eventos con `threading.Timer`: `KEYDOWN K_s` a los 2 s, `MOUSEBUTTONDOWN button=3 pos=(621, 409)` a los 2,5 s (el post-it A queda espejado) y `QUIT` a los 4 s.
  - Corre `P.main()` capturando `SystemExit`.
  - Esperado: sin traceback; en la salida aparecen `Símbolos: sin modelo` y `Post-it`; `<tmp>` tiene al menos 1 `.png`.

---

### Tarea 6: Entrenamiento, `--video` con tipos y README

**Archivos:**
- Crear: `V3\entrenar_simbolos.py` y `V3\README.md`.
- Modificar: `V3\desafios.py` (`modo_video` y pruebas).

**Interfaces que produce** (`entrenar_simbolos.py`; Ultralytics se importa solo dentro de `main`):
- `grupo(nombre: str) -> str`:
  - `"20261108_153000_p3_0012.png"` → `"20261108_153000_p3"` (regex `^(\d{8}_\d{6}_p\d+)_\d+$` sobre el nombre sin extensión).
  - Cualquier otro nombre → el nombre sin extensión.
- `listar(carpeta: str) -> list[tuple[str, str]]`: pares `(ruta, clase)` de `carpeta/<clase>/*.png|jpg|jpeg`, solo de las carpetas de `CLASES`. Ignora `sin_clasificar/`.
- `dividir(items, frac_val=0.2, semilla=0) -> tuple[list, list]`: por clase, baraja los grupos con `random.Random(semilla)`; los primeros `ceil(frac_val * n_grupos)` grupos van a val y el resto a train.
- `revisar(items) -> tuple[list[str], list[str]]`: devuelve `(errores, avisos)`. Es error que una clase tenga menos de 2 grupos (incluye 0 imágenes), y el mensaje nombra la clase. Es aviso que tenga menos de 30 imágenes.
- `armar_carpetas(train, val, test, destino) -> None`: borra `destino` y copia a `destino/{train,val}/<clase>/`, y a `test/` solo si `test` no está vacío.
- `main(argv=None) -> int`:
  - Opciones: `--dataset V3\dataset`, `--prueba V3\dataset_prueba`, `--epocas 40`, `--runs %USERPROFILE%\yolo_runs_proyecto`, `--salida V3\modelos\simbolos.pt`.
  - Si `revisar` da errores: los imprime y devuelve 1.
  - Solo usa el test si las 4 clases tienen imágenes en `--prueba`; si no, avisa "sin test".
  - Arma `runs\datos_simbolos` y entrena `YOLO(V3\modelos\yolo11n-cls.pt).train(...)` con los parámetros globales, `project=runs` y `name="simbolos"`.
  - Si hay test: `YOLO(best).val(data=..., split="test", imgsz=96, workers=0)` e imprime top-1 y la carpeta de la matriz de confusión.
  - Copia `best.pt` a `--salida` y devuelve 0.

- [ ] **Paso 1: Escribir las pruebas que fallan:**

```python
def p_e1_grupo():
    E = importlib.import_module("entrenar_simbolos")
    r = (E.grupo("20261108_153000_p3_0012.png"), E.grupo("foto_celular.jpg"))
    return r == ("20261108_153000_p3", "foto_celular"), str(r)
def p_e2_division_por_grupos():
    E = importlib.import_module("entrenar_simbolos"); tmp = tempfile.mkdtemp()
    for clase, grupos in (("corona", 5), ("limpio", 1)):
        os.makedirs(os.path.join(tmp, clase))
        for g in range(grupos):
            for n in range(4): open(os.path.join(tmp, clase, f"20261108_15000{g}_p{g}_{n:04d}.png"), "wb").close()
    items = E.listar(tmp); train, val = E.dividir(items); errores, _ = E.revisar(items)
    g = lambda lst: {E.grupo(os.path.basename(r)) for r, c in lst if c == "corona"}
    ok = len(g(val)) == 1 and not (g(train) & g(val)) and any("limpio" in e for e in errores)
    return ok, f"val={g(val)} errores={errores}"
```

- [ ] **Paso 2:** `python desafios.py --rapidas`. Esperado: `p_e1` y `p_e2` en FALLA.
- [ ] **Paso 3:** crear `entrenar_simbolos.py`.
- [ ] **Paso 4:** `python desafios.py --rapidas`. Esperado: todas las rápidas APRUEBA.
- [ ] **Paso 5: Prueba de humo del entrenamiento con datos sintéticos.** Un script en el scratchpad genera `<tmp>\dataset` y `<tmp>\dataset_prueba`:
  - 3 sesiones × 2 post-its × 6 recortes por clase.
  - Base: `PARCHE`, con brillo y posición al azar.
  - Por clase: corona y flecha con los helpers de `desafios`; para `otro`, una X y una flecha de lado; `limpio` vacío.
  - Nombres con el formato de captura.

  Después correr:

  ```
  python entrenar_simbolos.py --dataset <tmp>\dataset --prueba <tmp>\dataset_prueba --epocas 2 --runs <tmp>\runs --salida <tmp>\simbolos.pt
  ```

  Esperado: código 0, se imprime el top-1 del test y existe `<tmp>\simbolos.pt`. Luego `S.cargar_clasificador(r"<tmp>\simbolos.pt")` debe dar `"modelo OK"` y una salida de forma `(1, 4)`. Con esto se confirma que esta versión de Ultralytics acepta `erasing`, `flipud` y `split="test"`.
- [ ] **Paso 6: `modo_video`.**
  - Al iniciar: `clasificar, msg = S.cargar_clasificador()` e imprimir `msg`.
  - Si hay modelo, en cada cuadro: `lector.actualizar(frame, seguidor.plataformas(), seguidor.quietos(), k)`.
  - En la línea de cada segundo, agregar la columna `tipos` con los ids que no son `limpio` (por ejemplo, `3:corona 5:flecha_arriba`).
  - Comprobar con `python desafios.py --video grabaciones\desafio_20261029_230641.mp4`: sin error, con el mensaje `sin modelo` y la tabla de siempre.
- [ ] **Paso 7: `README.md`**, en español y corto:
  - Requisitos: `pip install opencv-python pygame numpy ultralytics`, y PyTorch con CUDA instalado antes, como en el Taller 2.
  - `python proyecto_v3.py` y la tabla de controles.
  - Cómo juntar datos: las reglas de la sección 6 del diseño, más "el post-it con símbolo separado de los demás" y "dibujo con margen".
  - `python entrenar_simbolos.py`.
  - `python desafios.py` / `--rapidas` / `--video`.
- [ ] **Paso 8: Verificación final.** `python desafios.py`. Esperado: todas las rápidas APRUEBA (20 pruebas) y `TOTAL | 5/9 | 8/9`. Repetir la prueba de humo de la Tarea 5.

---

## Lo que queda para el usuario después del plan

Estos pasos necesitan la pizarra real y no se pueden automatizar:
1. Dibujar los post-its.
2. Capturar recortes con `S`.
3. Ordenarlos en las 4 carpetas.
4. Correr `python entrenar_simbolos.py`.
5. Probar en vivo con la webcam.
