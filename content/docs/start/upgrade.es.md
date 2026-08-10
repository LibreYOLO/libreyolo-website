---
title: Actualizar a 1.5.0
seo_title: "Actualizar LibreYOLO de 1.4.0 a 1.5.0"
description: "Los cuatro cambios de código que exige 1.5.0, los tres cambios que mueven las métricas y los ajustes de comportamiento menores que conviene conocer antes de comparar ejecuciones."
lead: "No se ha eliminado nada de la API pública de modelos: todas las clases y funciones que funcionaban en 1.4.0 se siguen importando. Cuatro argumentos han cambiado de forma, y tres valores por defecto mueven números con los que quizá estés comparando."
keywords: [actualizar libreyolo, migrar libreyolo 1.5.0, libreyolo cambios incompatibles, allow_experimental eliminado, yolox bn eps, faster-coco-eval por defecto]
last_verified: "1.5.0"
meta:
  - label: Se aplica a
    value: De 1.4.0 a 1.5.0
  - label: Cambios de código necesarios
    value: Cuatro, todos acotados
  - label: Resultados que cambian
    value: Backend de COCO, eps de BN en YOLOX, multiescala de D-FINE
  - label: Eliminaciones de la API pública
    value: Ninguna
---

Esta página trata sobre actualizar LibreYOLO en sí. Si lo que buscas es cómo
cargar un checkpoint de un proyecto upstream, eso es
[importar pesos existentes](/docs/migrate), otro tema distinto.

La entrada completa de la versión está en el [changelog](/docs/changelog). Lo
que sigue es solo la parte que te pide algo a ti.

## Cambios de código que debes hacer

### `allow_experimental=True` ya no existe

La barrera de reconocimiento ha desaparecido, junto con el mecanismo
`ddp_aware(experimental_key=...)` que había detrás. El entrenamiento y la
exportación de EC, RTMDet, PicoDet y FOMO exigían antes el argumento, así que
cualquier script que entrene una de esas familias está afectado.

```python
# 1.4.0
model.train(data="data.yaml", epochs=100, allow_experimental=True)

# 1.5.0: elimina el argumento
model.train(data="data.yaml", epochs=100)
```

No hay ningún shim de deprecación. Una llamada que lo siga pasando lanza
`TypeError`. `BaseModel.EXPERIMENTAL_WEIGHT_FILENAMES` se eliminó con él. El
hook `get_download_notice()` sobrevive, y lo siguen sobrescribiendo MiDaS,
SegFormer y YOLO9-P2.

Los niveles de soporte se siguen publicando, simplemente ya no son un
argumento: consulta [niveles de estabilidad](/docs/reference/stability-tiers).

### El nivel de exportación `"experimental"` ya no existe

```python
from libreyolo.export.support import Tier

# 1.4.0: Literal["validated", "experimental", "blocked"]
# 1.5.0: Literal["validated", "available", "blocked"]
```

El código que se ramifica según la cadena del nivel debe leer `"available"`
donde leía `"experimental"`. `BaseExporter` ya no emite un `RuntimeWarning`
para esos formatos. El estado de cada formato está en la
[matriz de exportación](/docs/reference/export-matrix).

### `pretrained=False` junto con `resume` ahora se rechaza

La combinación seguía adelante de forma incoherente. Ahora lanza:

```
ValueError: pretrained=False cannot be combined with resume.
```

Elige una. `pretrained=False` parte de una inicialización nueva con semilla,
que en 1.5.0 funciona para todas las familias entrenables en lugar de para
tres de ellas, y `resume` continúa una ejecución interrumpida desde su
checkpoint. Ambas están documentadas en [entrenamiento](/docs/train).

### En la CLI, `--imgsz` es un string, no un int

Es más acotado de lo que parece. Estos dos casos no se ven afectados:

```bash
libreyolo predict --model yolo9-t --source img.jpg --imgsz 640   # sigue bien
```

```python
model.predict("img.jpg", imgsz=640)   # sigue bien
```

Solo tiene que cambiar el código que llama a las funciones de comando de la
[CLI](/docs/cli) directamente desde Python, porque `predict`, `train` y `val`
ampliaron `--imgsz` de `int` a `str` para que pueda aceptar tamaños
rectangulares:

```python
from libreyolo.cli.commands.predict import predict_cmd

predict_cmd(..., imgsz=640)      # 1.4.0
predict_cmd(..., imgsz="640")    # 1.5.0, y ahora "480x640" también funciona
```

El valor por defecto de `train` es ahora el string `"640"`. `export --imgsz` ya
era un string, y `profile` no cambia.

## Números que cambian

Tres cambios mueven las métricas con la configuración por defecto. Si sigues
los resultados entre versiones, léelos antes de comparar una ejecución de 1.5.0
con una de 1.4.0.

### faster-coco-eval es el backend de métricas COCO por defecto

`val()` y la validación por época durante el entrenamiento calculan ahora las
métricas COCO con el backend en C++ faster-coco-eval en lugar de pycocotools.

El cambio se decidió sobre una paridad medida en los 100 splits de test de
RF100-VL: 1381 de 1400 valores de métricas idénticos bit a bit, desviación
máxima 2.22e-16, deltas principales exactamente 0, con 15,6x más velocidad en
conjunto y 56x en datasets con muchas detecciones. Tus números no deberían
moverse. Aun así, los produce una implementación distinta, y esa es la razón
por la que esto está en la lista.

pycocotools sigue siendo el respaldo automático cuando faster-coco-eval no
está instalado. Para forzarlo:

```bash
libreyolo val --model yolo9-t --data coco.yaml --no-faster-coco-eval
```

```python
model.val(data="coco.yaml", faster_coco_eval=False)
```

`LIBREYOLO_FASTER_COCO_EVAL=0` hace lo mismo de forma global. El backend que se
usa realmente se registra a nivel INFO, se expone como
`model.last_eval_backend` después de `val()`, y se incluye como `eval_backend`
en el payload JSON de la [CLI](/docs/cli/val). Instala la vía rápida con
`pip install libreyolo[fast-eval]`.

### Los checkpoints de YOLOX entrenados antes de 1.5.0 necesitan un override de eps

Esta es la trampa de la versión. Léela si has hecho fine-tuning de
[YOLOX](/docs/models/yolox).

YOLOX especifica `eps=1e-3` y `momentum=0.03` para BatchNorm. Hasta 1.5.0 esos
valores se aplicaban como un arreglo posterior que no sobrevivía a la
reconstrucción por número de clases que hace `train()` cuando el `nc` de tu
dataset difiere del del checkpoint. Un fine-tune así se entrenaba y reportaba
la validación durante el entrenamiento con el `eps=1e-5` por defecto de torch,
y luego se recargaba para inferencia con `1e-3`: los mismos tensores bajo una
normalización distinta.

Los tamaños con convolución normal apenas se mueven. El `n` depthwise se mueve
mucho, porque su `running_var` por canal es lo bastante pequeño como para que
eps domine. En el `ball` de RF100-VL, el mismo checkpoint nano obtiene
**0.566** de mAP50-95 evaluado con el eps con el que se entrenó y **0.151**
tras una recarga estándar.

Un checkpoint entrenado antes de 1.5.0 arrastra la semántica de eps=1e-5. Para
reportar números fieles, o bien lo evalúas con el eps de BN forzado a 1e-5:

```python
import torch
from libreyolo import LibreYOLOX

model = LibreYOLOX("my-yolox-finetune.pt")
for module in model.model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eps = 1e-5

model.val(data="data.yaml")
```

o incorporas `sqrt((var + 1e-3) / (var + 1e-5))` a los pesos de BN una sola vez
y guardas el resultado. Los checkpoints entrenados en 1.5.0 y posteriores no
necesitan ninguna de las dos cosas.

### El entrenamiento multiescala de D-FINE usa la receta por tamaño de upstream

`base_size_repeat` estaba fijado a 3 para todos los tamaños. Ahora se resuelve
por tamaño tal como especifica upstream: **n** entrena a tamaño fijo con la
multiescala desactivada, **s** 20, **m** 6, **l** 4, **x** 3. Antes solo
coincidía x, así que n, s, m y l ven una distribución de escalas distinta y
convergen a métricas distintas.

Para recuperar el comportamiento anterior, fíjalo explícitamente:

```python
from libreyolo.training.config import DFINEConfig

config = DFINEConfig(base_size_repeat=3)
```

DEIM sigue usando el 3 fijo. Los detalles de la familia están en
[D-FINE](/docs/models/d-fine).

## Conviene saberlo, no requiere acción

- **Los resultados con `imgsz` rectangular han cambiado porque antes estaban
  mal.** Las coordenadas de los bounding boxes, el redimensionado de máscaras
  de RTMDet, el reescalado de YOLO-NAS y el escalado del ground truth en el
  validador usan ahora alto y ancho por eje en lugar de un único escalar. Con
  `imgsz` cuadrado no cambia ni un bit. La inferencia o la validación
  rectangulares ejecutadas en 1.4.0 estaban mal escaladas. YOLO-NAS ahora
  rechaza directamente el `imgsz` rectangular en lugar de producir una salida
  incorrecta en silencio.
- **Los diccionarios de métricas han ganado claves.** `max_det`, `ar_max_det` y
  `AR_max_det` del evaluador COCO, y `metrics/loss` más `metrics/loss/ce` de
  FOMO. Los valores con la configuración por defecto no cambian, pero cualquier
  cosa que itere sobre las claves de métricas, incluidos los
  [loggers](/docs/train/loggers) personalizados y las cabeceras CSV, ve
  columnas nuevas.
- **Las ejecuciones de YOLO9 con semilla que provocan una reconstrucción de la
  cabeza** parten de una inicialización distinta, porque la semilla se aplica
  ahora antes de la reconstrucción y no después. Un fine-tune con semilla hecho
  en 1.4.0 hacia un número de clases distinto no es reproducible bit a bit en
  1.5.0.
- **`libreyolo[hub-kernels]` en CUDA ahora sí activa el kernel nativo
  MS-deform-attn.** 1.4.0 lo condicionaba a algo que RF-DETR nunca cumplía, así
  que el kernel nunca llegaba a ejecutarse. Las predicciones pueden variar
  dentro de la tolerancia de float en RF-DETR y en el resto de familias con
  deformable attention. Las instalaciones estándar no se ven afectadas, y
  `LIBREYOLO_HUB_KERNELS=0` lo desactiva.
- **`libreyolo predict` descarta las opciones no soportadas en lugar de lanzar
  un error.** La CLI filtra los kwargs contra la firma de `__call__` del
  modelo, así que una opción que una familia no acepta se ignora en vez de
  lanzar `TypeError`. Una errata en el nombre de un flag ahora se ignora en
  silencio.
- **Las fuentes en vivo cambian la forma de la salida JSON.** Las webcams, los
  streams RTSP y la captura de pantalla activan el streaming de forma
  implícita, lo que emite un registro por frame en lugar de uno por llamada.
  Estas [fuentes](/docs/predict/sources) son nuevas en 1.5.0, así que no afecta
  a ningún script de 1.4.0.
- **Reexportar `rfdetr-pose` o `yolonas-pose` a ONNX produce nombres de salida
  distintos.** 1.4.0 interpretaba mal sus cabezas de pose multitensor como
  segmentación mediante una heurística basada en el número de salidas. Los
  archivos `.onnx` que ya tengas en disco no se tocan.
- **En una instalación sin torch**, los resultados contienen arrays de numpy en
  lugar de `torch.Tensor`, así que `.boxes.data` devuelve un tipo distinto y el
  desempate de NMS puede diferir del de torchvision. Con torch instalado, el
  comportamiento es idéntico byte a byte. Consulta
  [instalación ligera](/docs/lightweight-install).
- **Los objetos de configuración validan más en la construcción.** `TrainConfig`
  ha ganado un `__post_init__` donde no tenía ninguno, así que una
  configuración que ya era inválida ahora lanza el error de inmediato en lugar
  de fallar en mitad de una ejecución. La serialización de `ValidationConfig`
  ha ganado una clave `edge_thresholds`, lo que rompe el round-trip estricto
  `ValidationConfig(**dump)` a partir de un dump de 1.4.0.
- **Los nombres de archivo de los pesos de las familias con sufijo de tarea se
  resuelven de otra forma.** `segformer-b0` se resuelve ahora a
  `LibreSegformerb0-sem.pt`. Esto arregla los 404 de la descarga automática, y
  rompe cualquier script que tuviera fijado el nombre de archivo antiguo sin
  sufijo.
- **El marcador de pytest `experimental_backend` ahora es `extended_backend`.**
  Solo importa si ejecutas la suite de tests con `-m`.

## Checkpoints y datasets

Los checkpoints escritos por 1.4.0 se cargan sin cambios. El
[esquema](/docs/reference/checkpoint-schema) ha ganado `imgsz_h` e `imgsz_w`
para los modelos rectangulares, y sigue escribiendo el escalar
`imgsz = max(h, w)` para los lectores antiguos. Las exportaciones a
[ExecuTorch](/docs/export/executorch) y [MNN](/docs/export/mnn) requieren ahora
un archivo adjunto, `<program>.pte.json` y `<model>.mnn.json` respectivamente, y
las exportaciones de HRNet llevan `pose_input: "person_crop"`. Los formatos de
dataset no cambian.
