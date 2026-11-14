# Plataformas con post-its: un juego de visión por computador

**TEL-328 · Procesamiento Digital de Imágenes · Proyecto**

La webcam mira una pizarra. Cada **post-it amarillo** pegado ahí se convierte en una
plataforma, y un cubo azul salta solo de una a otra con física tipo Mario Bros. Lo que se
dibuja en cada post-it cambia las reglas del juego:

| Dibujo en el post-it | Efecto |
|---|---|
| **Corona** | **Meta.** Si el cubo llega ahí, aparece **¡META!** y el camino termina. |
| **Flecha hacia arriba** | **Trampolín.** Desde ahí el cubo sube ~300 px y salta ~410 px, unas 3 veces un salto normal. |
| Vacío o cualquier otro dibujo | Plataforma normal. |

>  **Video de demostración:** _(agregar enlace)_

<!-- Agregar aquí una captura o GIF del juego: ![demo](docs/demo.gif) -->

## Cómo funciona

```
webcam ─► detector de amarillo ─► seguidor ─► lector de símbolos ─► reglas del juego
          (Lab + forma)           (tapado /   (recorte + YOLO11n-cls  (física, meta,
                                   sacado)      + votación)             trampolín, semáforo)
```

1. **Detectar los post-its** (procesamiento clásico):
   - El amarillo se segmenta en el espacio **Lab**, comparando cada píxel contra la mediana de la escena para tolerar cambios de luz.
   - Un **filtro de forma** acepta solo manchas que parezcan un post-it: área, proporción, extensión, solidez y unas 4 esquinas.
2. **Seguirlos en el tiempo:**
   - Un post-it aparece después de varios cuadros seguidos.
   - Si deja de verse, se compara su lugar con la pared que lo rodeaba: si ahora se ve pared quieta, **lo sacaron**; si hay otra cosa, como una mano, está **tapado** y se recuerda.
   - Si se movió la cámara, se olvidan las posiciones.
3. **Leer el símbolo:**
   - Cada post-it quieto se recorta y lo clasifica un **YOLO11n-cls** (Ultralytics) entrenado con recortes propios.
   - Las clases son `corona`, `flecha_arriba`, `limpio` y `otro`.
   - El tipo solo cambia si el promedio de las últimas lecturas supera el **85 %** de confianza.
4. **Jugar:**
   - Un **semáforo** indica si el cubo puede partir.
   - Cada salto se calcula con la parábola real (altura y distancia), así que un salto imposible no se intenta.

Más detalles en [docs/diseno_simbolos.md](docs/diseno_simbolos.md).

## Instalación

Requiere Python 3.10 o superior y una webcam. Con GPU NVIDIA, instala PyTorch con CUDA
**antes** que el resto; si no, se instala la versión solo-CPU, que igual funciona:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

> Si tienes instalados `opencv-python` **y** `opencv-python-headless` a la vez, OpenCV queda
> sin ventanas y la tecla `D` no abre la depuración. Se arregla con
> `pip uninstall -y opencv-python-headless opencv-python` y luego `pip install opencv-python`.

## Jugar

```bash
python proyecto_v3.py
```

1. Pega al menos 2 post-its amarillos **separados** entre sí, frente a la cámara.
2. Dibuja en ellos con plumón grueso, **sin tocar los bordes** del post-it.
3. Cuando el semáforo esté en 🟡, presiona **ESPACIO**.

**Semáforo** (arriba a la derecha):
- 🔴 **Rojo:** hay menos de 2 post-its, o desde el inicio el cubo no puede saltar a ninguna parte.
- 🟡 **Amarillo:** preparado. El cubo espera en el post-it de la izquierda.
- 🟢 **Verde:** en marcha.

**Controles:**

| Tecla | Acción |
|---|---|
| ESPACIO | dar la partida (con el semáforo en amarillo) |
| R | devolver el cubo al inicio |
| Clic derecho | marcar a mano el tipo de un post-it (auto → corona → flecha → limpio) |
| Clic | calibrar el color sobre un post-it (C la borra) |
| S | capturar recortes para entrenar (ver [cómo agregar imágenes](docs/agregar_imagenes.md)) |
| F | congelar las plataformas |
| E | fijar o soltar la exposición de la cámara |
| D | ventana de depuración (máscara y contornos) |
| G | grabar video en `grabaciones/` |
| ESC | salir |

**En la pantalla**, las líneas entre post-its muestran qué saltos se pueden hacer:
- **Verde:** ida y vuelta.
- **Naranjo:** un solo sentido, por ejemplo solo con el trampolín o solo bajando.
- **Rojo:** imposible.

### Probar sin cámara (simulador)

Arma una pizarra virtual con recortes reales y corre el juego completo encima:

```bash
python simulador.py
python simulador.py "limpio flecha_arriba limpio@-250 corona@-250"
```

- Cada palabra es un post-it, de izquierda a derecha: `corona`, `flecha_arriba`, `limpio` u `otro`. Caben hasta 6.
- `_` deja un hueco.
- `@-150` sube ese post-it 150 px y `@60` lo baja.

## Estructura del proyecto

```
proyecto_v3.py          juego: detección, seguimiento, personaje, reglas, semáforo
simbolos.py             recorte, clasificador YOLO, votación y captura (tecla S)
entrenar_simbolos.py    entrena el clasificador (carpetas locales o exportación de Roboflow)
simulador.py            pizarra virtual para probar sin webcam
desafios.py             pruebas automáticas
modelos/simbolos.pt     modelo entrenado que carga el juego
dataset/<clase>/        recortes de entrenamiento (corona, flecha_arriba, limpio, otro)
dataset_prueba/<clase>/ recortes de prueba: post-its físicos distintos
desafios/               foto de la pizarra que usan las pruebas
docs/                   diseño, plan y guía para agregar imágenes
```

## Modelo y resultados

| | |
|---|---|
| Modelo | YOLO11n-cls, entrada de 96 px, preentrenado en ImageNet |
| Datos | 525 recortes de entrenamiento y validación, 139 de prueba (post-its distintos) |
| Exactitud en prueba | **79,9 %** (antes de agregar coronas: 74,1 %) |
| Mejor clase | `limpio`: 21/21 · `corona`: 41/44 |
| Puntos débiles | flecha dentro de un círculo (no hay ejemplos parecidos) y letras en zigzag leídas como corona |

**Para mejorarlo:** agrega diseños distintos (no más fotos del mismo post-it) siguiendo
[docs/agregar_imagenes.md](docs/agregar_imagenes.md).

## Pruebas

```bash
python desafios.py --rapidas    # 32 pruebas de reglas, símbolos, captura y entrenamiento (~1 min)
python desafios.py              # lo anterior + 9 desafíos del detector sobre la foto real (~5 min)
python desafios.py --video grabaciones/archivo.mp4   # analiza una grabación segundo a segundo
```

Resultado esperado: **32/32** pruebas rápidas y **8/9** desafíos. El que falla, "manga
blanca", es el peor caso conocido: una manga del mismo color de la pizarra tapa un post-it.

## Limitaciones

- El post-it con símbolo debe estar **separado** de los demás: post-its pegados se unen en una sola plataforma y no se leen.
- Un dibujo que toca el borde del post-it hace que el detector lo pierda.
- El modelo se entrenó con unas 11 coronas, 6 flechas y 10 "otros" distintos. Dibujos muy diferentes pueden no reconocerse; para esos casos está el clic derecho.

## Uso de IA

Se usó un asistente de IA (Claude, de Anthropic) como apoyo para diseñar y programar el
lector de símbolos, las reglas del juego, el entrenamiento y las pruebas, y para redactar
la documentación. La captura y clasificación de las imágenes, las pruebas con la pizarra
real y la verificación de los resultados las hizo el grupo.
