# Diseño — Post-its con símbolos (proyecto_v3)

Fecha: 2026-11-08 · Estado: **borrador para revisión**

## 1. Objetivo

Extender `proyecto_v2.py` para que, además de detectar post-its amarillos como
plataformas, reconozca un **símbolo dibujado** en cada post-it y cambie las reglas
del recorrido automático del cubo azul.

**Lo que pediste**
- Post-it con **corona** = meta: si el cubo llega ahí, el camino termina.
- Post-it con **flecha hacia arriba** = el cubo puede saltar mucho más y alcanzar
  bloques que antes eran inalcanzables.
- Usar un modelo tipo YOLO, como en el Taller 2 (elegiste la opción A:
  detector de color actual + clasificador YOLO11n-cls sobre el recorte).
- Una carpeta nueva con todos los cambios, dejando fuera lo que estorba.

**Lo que asumí (no lo corregiste)**
- Al aterrizar sobre la corona se muestra "¡META!" y el cubo se queda ahí; `R` reinicia.
- El cubo nunca parte sobre una corona.
- La flecha funciona como **trampolín**: todo salto que *sale* de ese post-it usa la
  física fuerte. Es de un solo sentido (volver depende del post-it de destino).
- Un post-it `otro` (X, garabato, flecha abajo o de lado) es una plataforma normal.
- Se mantienen las reglas actuales: < 2 post-its sin cubo, 2 quieto, ≥ 3 automático,
  saltos solo entre vecinos en orden izquierda→derecha.

**Criterio de éxito**
1. Con el modelo entrenado, una corona y una flecha bien dibujadas se reconocen en
   vivo y el cubo cumple las reglas.
2. Sin modelo, el juego funciona igual que v2 (y se pueden marcar tipos a mano).
3. El banco de desafíos actual sigue en **8/9** (hoy falla solo "manga blanca").

## 2. Reorganización de carpetas

Nada se borra. Lo antiguo se mueve a una carpeta aparte; lo nuevo vive en `proyecto_v3/`.

```
Proyecto_actualizado/
├── proyecto_v3/                  ← TRABAJO ACTUAL
│   ├── proyecto_v3.py            juego (copia de proyecto_v2.py + reglas nuevas)
│   ├── simbolos.py               NUEVO: recorte, clasificador y votación
│   ├── entrenar_simbolos.py      NUEVO: arma train/val/test y entrena
│   ├── desafios.py               copia de v2 + desafíos de reglas y símbolos
│   ├── README.md                 cómo jugar, capturar y entrenar
│   ├── modelos/                  yolo11n-cls.pt (copiado del Taller 2) y simbolos.pt (tras entrenar)
│   ├── dataset/                  MOVIDA desde la raíz (+ sin_clasificar/)
│   ├── dataset_prueba/           MOVIDA desde la raíz
│   ├── desafios/                 COPIA (videos + pizarra_base.jpg)
│   ├── grabaciones/              COPIA
│   └── docs/diseno_simbolos.md   este documento
├── versiones_anteriores/         ← MOVIDO: proyecto.py, proyecto_menu.py, game.py,
│                                   generador_obstaculos.py, pizarra.py, proyecto_v2.py,
│                                   desafios.py, desafios/, grabaciones/
└── Preguntas 3.pdf               no es del proyecto (tarea de física); queda donde está
```

`proyecto_v2.py`, `desafios.py` y `desafios/` se mueven juntos, así v2 sigue
funcionando desde `versiones_anteriores/`.

## 3. Arquitectura

| Unidad | Qué hace | Depende de |
|---|---|---|
| `proyecto_v3.py` | Detección de post-its, seguimiento, personaje, reglas, dibujo, teclas | `simbolos.py` (opcional) |
| `simbolos.py` | `recortar(frame, rect)`, `LectorSimbolos` (clasifica y vota), `Capturador` (guarda recortes) | `ultralytics` (opcional), OpenCV |
| `entrenar_simbolos.py` | Divide el dataset por grupos, entrena YOLO11n-cls, evalúa, copia `best.pt` a `modelos/simbolos.pt` | `ultralytics` |
| `desafios.py` | Pruebas: los 9 desafíos de v2 + reglas + símbolos | `proyecto_v3`, `simbolos` |

`LectorSimbolos` recibe el modelo desde afuera (o lo carga de `modelos/simbolos.pt`),
así las pruebas pueden usar un modelo falso sin GPU ni pesos.

**Ciclo por cuadro**

```
cámara → detectar_postits → SeguidorPlataformas.actualizar
       → LectorSimbolos.actualizar(frame, plataformas, quietos)   (cada 10 cuadros)
       → tipos = {id: "limpio" | "corona" | "flecha_arriba" | "otro"}
       → PersonajeAutomatico.actualizar(plataformas, tipos)
       → dibujo (+ Capturador si está activo)
```

## 4. Reconocimiento de símbolos (`simbolos.py`)

**Recorte.** Rectángulo del post-it en el cuadro actual, quitando 8 % de borde por lado,
sin rotar (los post-its en la pared están casi derechos; no girar evita confundir flecha
arriba con flecha de lado). Se guarda y clasifica a 128×128; el modelo usa `imgsz=96`.
Es la **misma función** para capturar datos y para clasificar en vivo.

**Cuándo se clasifica un post-it** (si no, conserva su tipo actual):
- confirmado y **visible** (no "tapado");
- **sin movimiento** en su zona (una mano encima daría `otro`). El seguidor expone
  qué pistas están quietas;
- lado menor ≥ 40 px (a la distancia de tu foto miden ~70 px);
- casi cuadrado (lado corto / lado largo ≥ 0,75). Una fila de post-its pegados no se
  clasifica: **el post-it con símbolo debe estar separado de los demás.**

**Cada cuántos cuadros.** Cada 10 cuadros (~6 veces/s a 60 fps, ~3 veces/s a 30 fps), todos los recortes en un
solo lote. Medido: 7,6 ms por lote de 5 en GPU y 10 ms en CPU.

**Votación (umbral 0,85).** Por post-it se guardan las probabilidades de las últimas 5
lecturas. El tipo cambia solo si hay al menos 3 lecturas y el promedio de la clase
ganadora es ≥ 0,85. Mientras no se decide, el post-it es `limpio`. Un símbolo ya
decidido (corona o flecha) **solo lo reemplaza el otro símbolo**: lecturas de `otro` o
`limpio` (mano quieta encima, reflejo) no lo borran, porque un dibujo en papel no
desaparece solo. Si el post-it desaparece, se olvida su historial.

**Tipo manual.** Clic derecho sobre un post-it rota su tipo:
auto → corona → flecha_arriba → limpio → auto. El tipo manual manda sobre el
clasificador. Sirve para probar las reglas antes de tener modelo y como respaldo en una demo.

## 5. Reglas del juego (`proyecto_v3.py`)

**Física del trampolín.** `salto_posible(origen, destino, fuerte=False)`:

| | Velocidad de salto | Avance en el aire | Altura útil (×0,9) | Alcance útil en plano (×0,9) |
|---|---|---|---|---|
| normal (actual) | 13 | 4 | ~95 px (≈ 2,7 cubos) | ~117 px (≈ 3,3 cubos) |
| trampolín | 18 | 6 | ~182 px (≈ 5,2 cubos) | ~243 px (≈ 6,9 cubos) |

Con post-its de ~70 px y huecos de ~60 px, quitar C deja B→D a ~190 px: imposible
normal, posible con trampolín. Los valores quedan como constantes ajustables.
El salto de trampolín se anima más alto y un poco más lento.

**Recorrido.**
1. Inicio: el post-it más a la izquierda que **no** sea corona.
2. Cada espera, si un vecino alcanzable es corona, salta a ella primero
   (evita alejarse de una meta que estaba al lado). Si no, sigue igual que v2: continúa
   en su dirección y, si no puede, se devuelve.
3. Un salto que sale de un post-it `flecha_arriba` usa la física de trampolín.
4. Si el cubo está parado (no en el aire) sobre una corona → estado **META**: no salta más
   y aparece "¡META!". `R` lo devuelve al inicio. Si sacan la corona, vuelve al inicio.

**Dibujo.**
- La corona y la flecha se dibujan como íconos sobre la plataforma; la letra (A, B, C…)
  pasa a una esquina.
- Las líneas entre vecinos siguen el código verde / naranjo / rojo, calculadas con la
  física de cada origen.
- La barra de estado muestra el estado del modelo ("símbolos: modelo OK" o
  "sin modelo — clic derecho para marcar") y la captura activa.

## 6. Captura de datos y entrenamiento

**Captura (tecla `S`).** Activa o desactiva la captura. Mientras está activa, cada 30 cuadros (~0,5–1 s)
guarda el recorte de cada post-it clasificable en
`dataset/sin_clasificar/<sesion>_p<id>_<n>.png`. Los archivos del mismo post-it físico
quedan juntos por nombre, así que se ordenan rápido a mano en
`corona/`, `flecha_arriba/`, `limpio/`, `otro/`.

**Cómo juntar datos** (guía en el README, basada en `ejemplo_dataset.png`):
- Dibuja con plumón grueso y deja margen: si el dibujo toca el borde, el detector pierde
  el post-it (comprobado).
- Usa varios post-its físicos, luces, distancias y personas que dibujen.
- Pon en `otro/` lo que no es corona ni flecha arriba (X, flechas abajo o de lado,
  garabatos, dedo encima).
- `dataset_prueba/` debe tener post-its físicos **distintos** a los de `dataset/` y venir de
  otra sesión (en el Taller 2 los mismos objetos en train y test inflaron las métricas).
- Meta inicial: ~80–150 recortes por clase.

**Entrenamiento (`python entrenar_simbolos.py`).**
- `dataset/` se divide en train/val 80/20 **por grupo** (`<sesion>_p<id>` = un post-it
  físico en una sesión), nunca por cuadro. `dataset_prueba/` es test.
- Avisa si una clase tiene < 30 recortes. **Se detiene** si una clase tiene < 2 grupos:
  a esa clase le faltarían ejemplos en train o en val. (Ultralytics 8.4 empareja las carpetas
  con las clases por nombre, así que no las numera mal.)
- Avisa si `dataset_prueba` repite grupos de `dataset` (el test saldría inflado).
- **Dataset compartido en Roboflow** (`--roboflow <carpeta>`): lee la exportación *Folder
  Structure*. Junta train y valid, que Roboflow dividió al azar, y los vuelve a dividir por grupo.
  Respeta su test. El grupo se obtiene también de los nombres renombrados por Roboflow
  (`<nombre>_png.rf.<hash>.jpg`).
- Arma las carpetas y entrena fuera de OneDrive (`%USERPROFILE%\yolo_runs_proyecto\`),
  porque en el Taller 2 OneDrive dio `PermissionError`.
- Parámetros: YOLO11n-cls, `imgsz=96`, `epochs=40`, `patience=10`, `batch=32`,
  `workers=0`, `seed=0`, `flipud=0` (voltear invierte la flecha), `fliplr=0.5`
  (corona y flecha son simétricas), `erasing=0` (el borrado aleatorio puede borrar el
  símbolo y dejar la etiqueta mal), `scale=0.2` (recorte aleatorio de 80–100 % del área; con
  el valor por defecto, 0,5, podía cortar la punta de la flecha).
- Al final imprime la exactitud y la ruta de la matriz de confusión en test, y copia
  `best.pt` a `proyecto_v3/modelos/simbolos.pt`.

## 7. Manejo de errores

- Sin `ultralytics`/`torch`, sin `modelos/simbolos.pt`, o con un modelo de clases
  distintas a las 4 esperadas: el juego corre sin símbolos y lo avisa en pantalla
  (clic derecho sigue funcionando).
- Si la clasificación falla en vivo, se avisa una vez por consola y se desactiva el lector.
  El juego no se cae.
- La primera predicción se hace al iniciar (calentamiento), para no trabar el primer
  cuadro.

## 8. Pruebas (`python desafios.py`)

1. **Los 9 desafíos de v2** sin cambios: deben seguir en 8/9.
2. **Reglas** (con rectángulos sintéticos, sin cámara):
   - A B C D E normal: ida y vuelta completa;
   - A B _ D E sin flecha: B no llega a D y se devuelve;
   - A B _ D E con flecha en B: llega a D, pero D no vuelve a B;
   - corona en D: termina en META y no salta más;
   - corona a la izquierda de todo: parte en el siguiente y salta a la corona si la alcanza.
3. **Símbolos** (modelo falso):
   - la votación cambia el tipo solo con ≥ 3 lecturas y promedio ≥ 0,85;
   - no clasifica tapados, filas, post-its chicos ni zonas con movimiento;
   - lecturas de `otro` o `limpio` no borran una corona; una flecha sí la reemplaza.
4. **Detector con símbolo**: sobre la foto real, un post-it con corona o flecha dibujada
   con margen se sigue detectando.
5. **Con video real** (después de entrenar): `python desafios.py --video ...` imprime
   también los tipos por segundo.

## 9. Fuera de alcance (por ahora)

- Enderezar post-its muy inclinados con sus 4 esquinas (se agrega si las pruebas reales
  muestran fallas).
- Avisar si la meta es alcanzable antes de que el cubo llegue.
- Reemplazar el detector de color por un detector YOLO.
