"""
TEL-328 · Entrena el clasificador de símbolos (YOLO11n-cls) de proyecto_v3

Uso:
  python entrenar_simbolos.py                 (40 épocas, carpetas por defecto)
  python entrenar_simbolos.py --epocas 60
  python entrenar_simbolos.py --roboflow roboflow_export   (dataset compartido en Roboflow)

Lee:
  dataset/<clase>/*.png          -> se divide en train / val (80/20) POR GRUPO
  dataset_prueba/<clase>/*.png   -> test (post-its físicos distintos, otra sesión)
o, con --roboflow, la exportación 'Folder Structure' de Roboflow:
  train/ y valid/<clase>/        -> se juntan y se vuelven a dividir POR GRUPO
  test/<clase>/                  -> test (subir ahí la sesión de prueba)
Clases: corona, flecha_arriba, limpio, otro. dataset/sin_clasificar se ignora.

Grupo = '<sesion>_p<id>' del nombre que pone la tecla S: todos los recortes del
mismo post-it en una sesión. Un grupo nunca queda repartido entre train y val;
si no, val tendría casi las mismas fotos que train y la exactitud se inflaría
(lo que se vio en el Taller 2 con los mismos objetos en train y test).

Entrena FUERA de OneDrive (en el Taller 2 OneDrive dio PermissionError al guardar
last.pt) y al final copia best.pt a modelos/simbolos.pt, que el juego carga solo.
"""
import argparse
import math
import os
import random
import re
import shutil
import sys
from collections import defaultdict

from simbolos import CARPETA, CLASES, IMGSZ, RUTA_MODELO

EXTENSIONES = (".png", ".jpg", ".jpeg")
MIN_IMAGENES = 30   # menos que esto por clase: se avisa
MIN_GRUPOS = 2      # menos que esto por clase: no se puede armar val -> se detiene
RUNS = os.path.join(os.path.expanduser("~"), "yolo_runs_proyecto")   # fuera de OneDrive
MODELO_BASE = os.path.join(CARPETA, "modelos", "yolo11n-cls.pt")
PATRON_GRUPO = re.compile(r"^(\d{8}_\d{6}_p\d+)_\d+$")
# Roboflow exporta '<nombre>_png.rf.<hash>.jpg' en carpetas train/ valid/ test/
SUFIJO_ROBOFLOW = re.compile(r"_(?:png|jpe?g)\.rf\.[0-9a-f]+$", re.IGNORECASE)
SPLITS_ROBOFLOW = ("train", "valid", "val", "test")

# Parámetros de YOLO11n-cls. Los aumentos no deben cambiar qué símbolo hay en la imagen.
PARAMETROS = dict(
    imgsz=IMGSZ, patience=10, batch=32, workers=0, seed=0, deterministic=True, plots=True,
    flipud=0.0,   # voltear en vertical convierte la flecha arriba en flecha abajo
    fliplr=0.5,   # corona y flecha arriba son simétricas: voltear de lado vale
    erasing=0.0,  # el borrado aleatorio puede tapar el dibujo y dejar la etiqueta mal
    scale=0.2,    # recorte aleatorio de 80–100 % del área (el valor por defecto, 0.5,
                  # podía cortar la punta de la flecha y dejar solo una raya)
)


def grupo(nombre):
    """'20261108_153000_p3_0012.png' -> '20261108_153000_p3'; otro nombre -> sin extensión.
    También reconoce el nombre que pone Roboflow al exportar ('..._0012_png.rf.<hash>.jpg')."""
    base = os.path.splitext(os.path.basename(nombre))[0]
    base = SUFIJO_ROBOFLOW.sub("", base)
    m = PATRON_GRUPO.match(base)
    return m.group(1) if m else base


def listar(carpeta):
    """[(ruta, clase)] de carpeta/<clase>/ para cada clase de CLASES."""
    items = []
    for clase in CLASES:
        d = os.path.join(carpeta, clase)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(EXTENSIONES):
                items.append((os.path.join(d, f), clase))
    return items


def leer_roboflow(carpeta):
    """Exportación de Roboflow en formato 'Folder Structure' -> (items, test).

    train y valid se juntan en 'items' porque Roboflow los dividió al azar (mezclando
    recortes del mismo post-it); este script los vuelve a dividir POR GRUPO.
    test se respeta tal cual: ahí deben ir los post-its de la sesión de prueba.
    """
    items, test = [], []
    for split in SPLITS_ROBOFLOW:
        d = os.path.join(carpeta, split)
        if os.path.isdir(d):
            (test if split == "test" else items).extend(listar(d))
    return items, test


def _grupos_por_clase(items):
    por_clase = defaultdict(lambda: defaultdict(list))
    for ruta, clase in items:
        por_clase[clase][grupo(ruta)].append((ruta, clase))
    return por_clase


def dividir(items, frac_val=0.2, semilla=0):
    """Divide en (train, val) por grupos: un grupo va entero a uno de los dos."""
    rng = random.Random(semilla)
    train, val = [], []
    for _, grupos in sorted(_grupos_por_clase(items).items()):
        nombres = sorted(grupos)
        rng.shuffle(nombres)
        n_val = math.ceil(frac_val * len(nombres))
        for i, g in enumerate(nombres):
            (val if i < n_val else train).extend(grupos[g])
    return train, val


def revisar(items):
    """(errores, avisos) sobre el dataset antes de entrenar."""
    por_clase = _grupos_por_clase(items)
    errores, avisos = [], []
    for clase in CLASES:
        grupos = por_clase.get(clase, {})
        n = sum(len(v) for v in grupos.values())
        if len(grupos) < MIN_GRUPOS:
            errores.append(f"'{clase}': {n} recortes de {len(grupos)} post-it(s)/sesión(es); "
                           f"se necesitan al menos {MIN_GRUPOS} para armar val")
        elif n < MIN_IMAGENES:
            avisos.append(f"'{clase}': solo {n} recortes (se recomiendan {MIN_IMAGENES} o más)")
    return errores, avisos


def grupos_repetidos(items, test):
    """Grupos (un post-it en una sesión) que aparecen en dataset y también en dataset_prueba."""
    return {grupo(r) for r, _ in items} & {grupo(r) for r, _ in test}


def armar_carpetas(train, val, test, destino):
    """Borra 'destino' y copia a destino/{train,val[,test]}/<clase>/ (formato Ultralytics)."""
    if os.path.isdir(destino):
        shutil.rmtree(destino)
    for nombre, items in (("train", train), ("val", val), ("test", test)):
        if not items:
            continue
        for clase in CLASES:
            os.makedirs(os.path.join(destino, nombre, clase), exist_ok=True)
        for ruta, clase in items:
            shutil.copy2(ruta, os.path.join(destino, nombre, clase, os.path.basename(ruta)))


def main(argv=None):
    p = argparse.ArgumentParser(description="Entrena el clasificador de símbolos de proyecto_v3")
    p.add_argument("--dataset", default=os.path.join(CARPETA, "dataset"))
    p.add_argument("--prueba", default=os.path.join(CARPETA, "dataset_prueba"))
    p.add_argument("--epocas", type=int, default=40)
    p.add_argument("--runs", default=RUNS, help="carpeta de resultados (fuera de OneDrive)")
    p.add_argument("--salida", default=RUTA_MODELO)
    p.add_argument("--roboflow", default=None,
                   help="carpeta de una exportación de Roboflow (formato Folder Structure); "
                        "reemplaza a --dataset, y su test reemplaza a --prueba")
    a = p.parse_args(argv)

    if a.roboflow:
        items, test = leer_roboflow(a.roboflow)
        print(f"Datos: exportación de Roboflow en {a.roboflow} "
              f"({len(items)} de train+valid, {len(test)} de test)")
        if not items:
            print("No encontré train/<clase>/ ni valid/<clase>/ ahí. Exporta desde Roboflow "
                  "en formato 'Folder Structure' y descomprime el zip en esa carpeta.")
            return 1
        if {c for _, c in test} != set(CLASES):
            print("Aviso: el test de Roboflow no tiene las 4 clases; se usa dataset_prueba")
            test = listar(a.prueba)
    else:
        items, test = listar(a.dataset), listar(a.prueba)

    errores, avisos = revisar(items)
    conteo = defaultdict(int)
    for _, clase in items:
        conteo[clase] += 1
    print("Recortes por clase:", ", ".join(f"{c}={conteo[c]}" for c in CLASES))
    for x in avisos:
        print("Aviso:", x)
    if errores:
        print("No se puede entrenar todavía:")
        for e in errores:
            print("  -", e)
        return 1

    train, val = dividir(items)
    if {c for _, c in test} != set(CLASES):
        print("Aviso: el test no tiene las 4 clases; se entrena sin test")
        test = []
    repetidos = grupos_repetidos(items, test)
    if repetidos:
        print(f"Aviso: dataset_prueba repite {len(repetidos)} post-it(s) de dataset "
              f"(p. ej. {sorted(repetidos)[:3]}); la exactitud en test saldrá inflada. "
              "Usa post-its físicos y sesiones distintas para el test.")
    destino = os.path.join(a.runs, "datos_simbolos")
    armar_carpetas(train, val, test, destino)
    print(f"train {len(train)} | val {len(val)} | test {len(test)}  ->  {destino}")

    from ultralytics import YOLO   # se importa aquí: el resto del script no necesita torch
    modelo = YOLO(MODELO_BASE if os.path.exists(MODELO_BASE) else "yolo11n-cls.pt")
    modelo.train(data=destino, epochs=a.epocas, project=a.runs, name="simbolos", **PARAMETROS)
    best = str(modelo.trainer.best)
    if test:
        m = YOLO(best).val(data=destino, split="test", imgsz=IMGSZ, workers=0, plots=True,
                           project=a.runs, name="simbolos_test")   # también fuera de OneDrive
        print(f"Exactitud en test (top-1): {m.top1:.3f}   |   matriz de confusión en: {m.save_dir}")
    os.makedirs(os.path.dirname(os.path.abspath(a.salida)), exist_ok=True)
    shutil.copy2(best, a.salida)
    print("Modelo copiado a", a.salida, "(el juego lo carga al iniciar)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
