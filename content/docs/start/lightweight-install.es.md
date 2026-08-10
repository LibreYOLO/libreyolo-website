---
title: Instalación ligera
seo_title: "Ejecuta inferencia ONNX de LibreYOLO sin PyTorch"
description: "Instala LibreYOLO con --no-deps y ejecuta detección ONNX solo con numpy, sin torch en disco. La técnica, sus límites y la lista exacta de paquetes."
lead: "La ruta de inferencia ONNX de LibreYOLO es numpy de principio a fin, incluidos el decode y el NMS. Nada de esa ruta necesita PyTorch en tiempo de ejecución, así que una instalación que se salte la resolución de dependencias puede ejecutar detección con torch ausente de la máquina."
keywords: [libreyolo sin pytorch, inferencia onnx sin torch, instalación ligera libreyolo, pip install no-deps, onnxruntime inferencia python, libreyolo espacio en disco, detección de objetos sin torch, torch solo cpu]
last_verified: "1.5.0"
meta:
  - label: Se aplica a
    value: Detección ONNX, siete familias de modelos
  - label: Punto de entrada
    value: libreyolo.backends.onnx.OnnxBackend
    mono: true
  - label: Nivel de soporte
    value: Best effort, no es una distribución aparte
snippets:
  install:
    - label: Ligera
      language: bash
      code: |
        # Instala el paquete sin su lista de dependencias y aporta después
        # los cuatro paquetes que la ruta de detección ONNX importa de verdad.
        pip install --no-deps libreyolo
        pip install numpy pillow opencv-python-headless onnxruntime
    - label: Torch solo CPU
      language: bash
      code: |
        # Prueba esto primero. Conserva todas las funcionalidades y evita el
        # wheel de CUDA, que es donde se va la mayor parte del disco.
        pip install libreyolo --index-url https://download.pytorch.org/whl/cpu
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo.backends.onnx import OnnxBackend

        model = OnnxBackend("libreyolo9t.onnx")
        result = model.predict("https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg")

        # Aquí xyxy es un ndarray de numpy, no un tensor de torch.
        print(result.boxes.xyxy)
        print(result.boxes.conf)
        print(result.boxes.cls)
---

## Por qué funciona

`pip install --no-deps libreyolo` instala el paquete y se salta por completo su
lista de dependencias. No se resuelve nada por ti, y pasas a ser responsable de
instalar lo que realmente uses.

Eso solo sirve si la ruta de código que quieres de verdad no necesita las
dependencias que te has saltado, y en el caso de la detección ONNX no las
necesita. El decode, incluida la non-maximum suppression, es numpy. Las recetas
de preprocesado son numpy. PyTorch es una dependencia de entrenamiento e
inferencia eager, y en esta ruta nunca se llama.

Antes de esta versión la importación fallaba de todas formas: importar
cualquier cosa bajo `libreyolo.models` construía todas las clases de modelo
para poblar el registro de autodetección de checkpoints, y esas clases son
subclases de `torch.nn.Module`. Las recetas de preprocesado viven ahora en su
propio paquete, `libreyolo.preprocess`, y la importación de torch se aplaza
hasta que algo toca un atributo de torch, así que la ruta ONNX se importa con
torch ausente de la máquina. Ese paquete contiene un preprocesador nativo de
numpy por familia: `yolo9`, `yolonas`, `yolox`, `ec`, `rtdetr`, `rfdetr`,
`dfine`, `deim` y `deimv2`, dos más que las siete familias verificadas de
principio a fin más abajo. Cada `libreyolo/models/<family>/utils.py` reexporta
desde él, así que las rutas de importación existentes siguen funcionando.

## Prueba primero el wheel solo para CPU

La mayoría de quienes piden esto quieren evitar una instalación de varios
gigabytes, y el tamaño se concentra en un solo sitio: el wheel de `torch` por
defecto incluye CUDA. Una compilación solo para CPU es una fracción de eso y no
necesita ninguna ruta de instalación especial.

<code-tabs name="install" />

La opción solo para CPU conserva todas las funcionalidades de LibreYOLO: el
entrenamiento, la validación, todas las tareas, todas las familias, la CLI.
Elige la ruta ligera cuando quieras cero torch en la máquina, no simplemente
menos torch.

## Qué cubre la instalación ligera

| | |
|---|---|
| Tarea | Detección |
| Formato | ONNX |
| Punto de entrada | `OnnxBackend` |
| Interfaz | Biblioteca de Python |

Se verificaron siete familias en esta ruta: [YOLOv9](/docs/models/yolov9),
[YOLO-NAS](/docs/models/yolo-nas), [EdgeCrafter](/docs/models/edgecrafter),
[RT-DETR](/docs/models/rt-detr), [RF-DETR](/docs/models/rf-detr),
[D-FINE](/docs/models/d-fine) y [DEIM](/docs/models/deim), contando con ellas
las variantes de cada familia.

Ese es el alcance verificado, no un límite que la biblioteca imponga. Otras
tareas y otras familias simplemente quedan fuera de lo que se comprobó: algunas
tirarán de torch cuando las llames, y unas pocas puede que funcionen por
casualidad. Trata todo lo que quede fuera de esta lista como no probado, no como
soportado ni como roto.

Dentro de ella, los resultados son idénticos a los de la instalación normal, no
solo parecidos. Cada familia se exportó a ONNX y se ejecutó dos veces, una
normalmente y otra con torch bloqueado; los bounding boxes, las puntuaciones y
las clases coincidieron exactamente. Un test de paridad en la suite evita que
ese contrato se desvíe.

## Las cinco cosas en las que suele tropezar la gente

**Usa `OnnxBackend`, no las clases de modelo.** `LibreYOLO9("model.onnx")`
sigue requiriendo torch, porque `LibreYOLO9` es en sí misma una subclase de
`nn.Module`. Este es el error más probable, ya que todas las demás páginas de
esta documentación cargan un modelo a través de su clase o de `LibreYOLO()`.

**Exporta en otro sitio.** Producir el fichero `.onnx` requiere torch, así que
la máquina ligera no puede crear uno. Exporta en una máquina de desarrollo o de
CI y envía el artefacto al destino reducido.

**Los resultados llevan arrays de numpy.** Aquí `result.boxes.xyxy` es un
`ndarray`. Los contenedores aceptan cualquiera de los dos tipos, así que los
nombres de los atributos no cambian, pero el código que llame a `.cpu()` o
`.numpy()` sobre un resultado fallará.

**Una sola imagen devuelve un solo `Results`.** `predict()` devuelve un
`Results` para una imagen y una lista para varias. Indexar un resultado único
con `[0]` selecciona la primera detección, no la primera imagen, lo que te da
en silencio un resultado de un solo box en lugar de lanzar un error.

**La CLI no funcionará.** `typer` y `click` no están entre los cuatro paquetes,
así que el comando `libreyolo` no está disponible. Esto es una instalación de
biblioteca.

## Predicción

<code-tabs name="predict" />

Cambia `onnxruntime` por `onnxruntime-gpu` para ejecutar en CUDA. Los cuatro
paquetes son los que un `predict()` completo sin torch importa realmente,
registrados durante la llamada en lugar de deducidos sobre el papel.
`opencv-python-headless` sustituye al `opencv-python` declarado: el mismo
módulo, sin bibliotecas de GUI, más pequeño en disco.

Del resto de dependencias declaradas, `requests` solo hace falta para cargar
una imagen desde una URL, `pycocotools` y `scipy` son validación y evaluación, y
`typer` y `click` son la CLI.

## Esta lista se quedará desfasada, por diseño

La lista de paquetes de arriba es correcta para la versión indicada al principio
de esta página. `--no-deps` te saca de la resolución de dependencias, así que
nada lo comprueba por ti, y una versión posterior puede importar algo que no
esté listado aquí.

Si te encuentras con un `ModuleNotFoundError`, ya entiendes la técnica: instala
el paquete que falte. Ese es el modelo de mantenimiento previsto, no un motivo
para abrir un informe de error. Esta ruta es best effort y no es una
distribución soportada por separado, que es también la razón de que no haya un
segundo paquete ligero en PyPI ni planes de crearlo.

Para confirmar que tu entorno está realmente libre de torch y no recurriendo en
silencio a una copia instalada, compruébalo con un assert:

```python
import importlib.util

assert importlib.util.find_spec("torch") is None, "torch is installed"
```

Merece la pena mantener esa comprobación en CI para la imagen reducida. Sin
ella, un entorno que resulte tener torch pasará todos los tests y no te dirá
nada.
