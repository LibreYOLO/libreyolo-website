---
title: Actualizar a 1.7.0
seo_title: Actualizar LibreYOLO de 1.6.0 a 1.7.0
description: >-
  Pasos de migración para checkpoints de YOLO9, nombres de módulos y recetas de
  entrenamiento, carpetas de ejecución del CLI, la matriz de confusión y los
  nuevos errores en LibreYOLO 1.7.0.
lead: >-
  La versión 1.7.0 mantiene la carga de los checkpoints antiguos de YOLO9 con
  los mismos boxes en float. Cambia el entrenamiento de la familia YOLO9, los
  nombres que el código ve dentro de la red y algunos contratos de validación y
  predicción.
keywords:
  - actualizar libreyolo
  - migrar libreyolo 1.6.0
  - libreyolo cambios incompatibles
  - nombres de módulos yolo9
  - clip_max_norm
  - matriz de confusión
last_verified: 1.7.0
source_hash: 895bcd7425f0052f
---

## De 1.6.0 a 1.7.0

- Los checkpoints de YOLO9, YOLO9-E2E y YOLO9-P2 guardados por 1.7.0 usan nombres de tensor nuevos y no cargan en 1.6.x: 1.6.x carga los de YOLO9 y YOLO9-P2 con un aviso de claves ausentes y no predice ningún box, y lanza un error de discrepancia de `model_family` con los de YOLO9-E2E. Actualiza todos los entornos que los lean. Los checkpoints float, cuantizados y de entrenamiento escritos por versiones anteriores cargan en 1.7.0, y las exportaciones ONNX no se ven afectadas.

- El código que accede a `model.model` de un modelo de la familia YOLO9 ve los nombres de módulo de MultimediaTechLab/YOLO, sin alias. `DDetect` es `YOLO9Head` (`anchor_convs` / `class_convs`), `RepConvN` es `RepConv`, `RepNBottleneck` es `Bottleneck`, `DFL` es `Anchor2Vec` (`anchor2vec`), `nc` es `num_classes`, y las subcapas de bloque son `conv1`/`conv2`/`bottleneck`. En YOLO9-E2E, `YOLO9E2EDetect` es `YOLO9E2EHead` y `one2one_cv2` / `one2one_cv3` son `one_to_one_anchor_convs` / `one_to_one_class_convs`. `LibreYOLO9Model.fuse()`, `RepConvN.fuse_convs()`, `Conv.forward_without_bn()`, `Concat` y `yolo9.utils.decode_boxes` se han eliminado. El [changelog](/docs/changelog) lista cada renombrado.

- Los nombres de nodo de ONNX siguen los nuevos nombres de módulo en la cabeza, el backbone y el neck, por ejemplo `/head/anchor_convs.0/...`. Actualiza las herramientas que dependen de nombres de nodo, como las configuraciones de nodos finales de Hailo. Las entradas, las salidas y los nombres de salida del modelo no cambian.

- Un fine-tuning de YOLO9 a partir de pesos sin tensores PGI, lo que incluye los `LibreYOLO9{t,s,m,c}.pt` publicados, entrena solo la cabeza principal, como en 1.5.0; 1.6.0 añadía una rama PGI con inicialización aleatoria con `aux_weight=0.25`. Los pesos que traen tensores PGI, y las ejecuciones desde cero, siguen entrenando con PGI. Pasa `aux_weight=0.25` para añadir una rama nueva de todos modos, o `aux_weight=0` para forzar que no se añada.

- Cuando se añade PGI, YOLO9-m y YOLO9-c construyen la rama auxiliar de MultimediaTechLab: 12.8M (m) y 25.7M (c) de parámetros exclusivos del entrenamiento con 80 clases, frente a 7.5M y 10.2M, así que el entrenamiento PGI de estos tamaños necesita más memoria y tiempo por paso. Un checkpoint de entrenamiento m/c de 1.6.0 se reanuda y se ajusta con la rama con la que se entrenó. La inferencia y YOLO9-t/s no cambian.

- YOLO9, YOLO9-E2E y YOLO9-P2 recortan la norma L2 del gradiente a 10, y su asignador de entrenamiento puntúa los anchors en fp32, así que los positivos elegidos bajo AMP en fp16 pueden diferir. Pasa `clip_max_norm=0` para el recorte de 1.6.0; una ejecución iniciada en 1.6.0 y reanudada en 1.7.0 también recorta salvo que lo pases. La asignación en fp32 no tiene interruptor.

- Un `perspective` distinto de cero deforma por desplazamiento de esquinas en cada familia que usa el warp afín compartido (YOLO9 y sus variantes E2E y P2, YOLOX, YOLOv7, YOLO-NAS, PP-YOLOE, RTMDet, PicoDet, RT-DETR y RT-DETRv2), incluidas las ejecuciones reanudadas desde un checkpoint de 1.6.0 que lo fijaba. Los valores de `0.002` y superiores dan la misma deformación máxima, y los valores negativos lanzan `ValueError`. Ningún ajuste reproduce el warp de 1.6.0; el valor por defecto `perspective=0` no cambia.

- El pipeline de mosaic de YOLO9 toma la proporción de mixup de Beta(1, 1) en lugar de Beta(32, 32), sin ajuste para el valor antiguo. El entrenamiento por defecto de YOLOv7 cambia, porque YOLOv7 activa mixup por defecto. YOLO9, YOLO9-E2E, YOLO9-P2, RT-DETR y RT-DETRv2 cambian solo cuando mixup está activado.

- `libreyolo train` sin `name=` escribe las ejecuciones de EC, D-FINE, SegFormer y LingBot-Vision en `runs/train/ec_exp`, `dfine_exp`, `segformer_exp` y `lingbotvision_exp` en lugar de `runs/train/exp`, los probes de V-JEPA 2 en `runs/train/train`, y las ejecuciones de YOLO9-P2 en `runs/train/yolo9_exp`. Pasa `name=exp` para la ruta antigua, o `name=yolo9_p2_exp` para YOLO9-P2.

- `libreyolo train` arranca D-FINE, SegFormer y LingBot-Vision desde sus propias recetas. Para D-FINE eso es 132 épocas, `lr0=2e-4`, `weight_decay=1e-4`, 2 épocas de calentamiento y sin aumento de mosaic, mixup, HSV ni afín. Para enviar los valores que enviaba 1.6.0, pasa `epochs=300 optimizer=sgd lr0=0.01 weight_decay=5e-4 warmup_epochs=5 mosaic=1.0 mixup=1.0 hsv_prob=1.0 degrees=10 translate=0.1 shear=2.0`.

- `libreyolo.validation.ConfusionMatrix` indexa `matrix[predicted, true]` (antes `[actual, predicted]`), cuenta las detecciones con `conf_thres=0.25` (antes 0.15), recibe `names=` y `task=` solo como argumentos de palabra clave, y su `matrix` ya no se puede reasignar. Transpón el código que indexaba la matriz antigua, y pasa `conf_thres=0.15` para los recuentos antiguos. `plots=True` escribe los recuentos en `confusion_matrix.png` y el gráfico normalizado en `confusion_matrix_normalized.png`.

- `predict(agnostic_nms=True)` suprime los boxes solapados entre clases en lugar de avisar. Las tareas sin boxes lanzan `ValueError`, `LibreEnsemble` y HRNet lanzan `NotImplementedError`, y el nivel SAM rechaza la palabra clave con `TypeError`; quita el argumento de esas llamadas.

- Un `interpolation` de clasificador desconocido lanza `ValueError`, incluidos los modos de torchvision `LANCZOS`, `BOX`, `HAMMING` y `NEAREST_EXACT`. Usa `bilinear`, `bicubic`, `nearest` o `bilinear_no_antialias`.

- Vuelve a ejecutar la validación antes de comparar con los números de 1.6.0 para `val(single_cls=True)` en un modelo entrenado con varias clases, los datasets de segmentación que mezclan filas solo de box y de polígono, y RTMDet-Ins, que conserva hasta `max_det=300` máscaras por imagen por defecto; pasa `max_det=100` para el ajuste de MMDetection.

- Sin `libreyolo[rfdetr]`, las llamadas del CLI con un nombre de modelo RF-DETR o DINOv2 salen con el código de error JSON `model_load_failed` en lugar de `model_not_found`, y el mensaje nombra `pip install libreyolo[rfdetr]`.

Consulta el [changelog](/docs/changelog) para la versión completa.

## De 1.5.0 a 1.6.0

- Actualiza los entornos fijados a una versión de ONNX Runtime anterior a 1.18 a `onnxruntime>=1.18.0` antes de instalar el extra ONNX.

- SAM 3D Body acepta el conjunto de archivos del snapshot revisado. Instala `libreyolo[hf]`, consigue acceso al repositorio restringido y usa la descarga automática, o proporciona el directorio del snapshot sin cambios y el archivo MHR fijado en un sistema de archivos local de confianza.

- Vuelve a ejecutar la validación tras las correcciones de redimensionado de RF-DETR para tareas distintas de pose y de normalización de clasificadores. Revisa los umbrales de confianza desplegados antes de comparar con resultados de 1.5.0; no hay un flag para el preprocesamiento anterior.

- D-FINE, DEIM, RT-DETRv4 y la detección de YOLO-NAS activan AMP FP16 por defecto. Pasa `amp=False` para conservar FP32.

- Para recuperar los ajustes anteriores de fine-tuning de YOLO9, fija `aux_weight=0`, `max_labels=100` y `warmup_momentum=0.937`. Fija `letterbox_pad="topleft"` cuando una nueva conversión marcada como centrada deba reproducir la geometría anterior. Los checkpoints antiguos de una sola cabeza se reanudan con su grafo original. Las nuevas ejecuciones de YOLO9 usan más memoria de GPU, así que baja `batch` si un tamaño de batch de 1.5.0 se queda sin memoria.

- RF-DETR y DINOv2 crean directorios de ejecución incrementados con el nombre de la familia. Actualiza los consumidores de rutas de artefactos, o fija `output_dir="runs/train", exist_ok=True` para conservar la ubicación y la reutilización anteriores.

- Mantén `mixup + cutmix <= 1` en clasificación; las combinaciones inválidas ahora lanzan un error durante la configuración.

- QAT desactiva EMA, SyncBatchNorm y el promedio de checkpoints. Usa los checkpoints best/last de QAT sin depender de esos estados.

- Los cargadores personalizados con hooks que modifican el dataset deben usar `persistent_workers=False` o reconstruir los workers tras la modificación. Las copias persistentes incompatibles con varios workers ahora lanzan un error.

- Los arrays de imagen de NumPy se leen como BGR, el orden de OpenCV. Pasa `color_format="rgb"` donde pases un array RGB como `np.asarray(pil_image)`; la salida de `cv2.imread()` y los frames de vídeo no necesitan cambios.

- Un array de NumPy o un tensor 4D es un batch: `predict()` devuelve una lista con un `Results` por imagen en lugar de usar solo la primera imagen.

- `train(resume=True)` y `train(resume="path/to/last.pt")` restauran los argumentos de entrenamiento guardados de la ejecución y siguen escribiendo en su directorio de ejecución; los argumentos que pases explícitamente tienen prioridad. Reanudar pesos publicados, o una ejecución que ya alcanzó sus `epochs`, lanza `ValueError`.

- Con la validación activada, la última época siempre se valida, así que las ejecuciones más cortas que `eval_interval` ahora informan de métricas y escriben `best.pt`. `val=False` desactiva la validación, última época incluida. La validación durante el entrenamiento escribe en `<run>/val` en lugar de `runs/val/`.

- Los fallos de `train` en la CLI informan de su tipo de error, así que los errores de configuración salen con código 2 en lugar de 1 con `io_error`.

Consulta el [changelog](/docs/changelog) para la versión completa e [importar pesos](/docs/migrate) para la conversión de checkpoints.

## De 1.4.0 a 1.5.0

Esta página trata sobre actualizar LibreYOLO en sí. Si lo que buscas es cómo
cargar un checkpoint de un proyecto upstream, eso es
[importar pesos existentes](/docs/migrate), otro tema distinto.

La entrada completa de la versión está en el [changelog](/docs/changelog). Lo
que sigue es solo la parte que te pide algo a ti.

### Cambios de código que debes hacer

#### `allow_experimental=True` ya no existe

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

No hay ningún shim de deprecación. Una llamada que lo siga pasando recibe un
aviso `Unknown training config keys (ignored)`, y el argumento no tiene ningún
efecto. `BaseModel.EXPERIMENTAL_WEIGHT_FILENAMES` se eliminó con él. El
hook `get_download_notice()` sobrevive, y lo siguen sobrescribiendo MiDaS,
SegFormer y YOLO9-P2.

Los niveles de soporte se siguen publicando, simplemente ya no son un
argumento: consulta [niveles de estabilidad](/docs/reference/stability-tiers).

#### El nivel de exportación `"experimental"` ya no existe

```python
from libreyolo.export.support import Tier

# 1.4.0: Literal["validated", "experimental", "blocked"]
# 1.5.0: Literal["validated", "available", "blocked"]
```

El código que se ramifica según la cadena del nivel debe leer `"available"`
donde leía `"experimental"`. `BaseExporter` ya no emite un `RuntimeWarning`
para esos formatos. El estado de cada formato está en la
[matriz de exportación](/docs/reference/export-matrix).

#### `pretrained=False` junto con `resume` ahora se rechaza

La combinación seguía adelante de forma incoherente. Ahora lanza:

```
ValueError: pretrained=False cannot be combined with resume.
```

Elige una. `pretrained=False` parte de una inicialización nueva con semilla,
que en 1.5.0 funciona para todas las familias entrenables en lugar de para
tres de ellas, y `resume` continúa una ejecución interrumpida desde su
checkpoint. Ambas están documentadas en [entrenamiento](/docs/train).

#### En la CLI, `--imgsz` es un string, no un int

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

### Números que cambian

Tres cambios mueven las métricas con la configuración por defecto. Si sigues
los resultados entre versiones, léelos antes de comparar una ejecución de 1.5.0
con una de 1.4.0.

#### faster-coco-eval es el backend de métricas COCO por defecto

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

#### Los checkpoints de YOLOX entrenados antes de 1.5.0 necesitan un override de eps

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
from libreyolo import LibreYOLO

model = LibreYOLO("my-yolox-finetune.pt")
for module in model.model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eps = 1e-5

model.val(data="data.yaml")
```

o incorporas `sqrt((var + 1e-3) / (var + 1e-5))` a los pesos de BN una sola vez
y guardas el resultado. Los checkpoints entrenados en 1.5.0 y posteriores no
necesitan ninguna de las dos cosas.

#### El entrenamiento multiescala de D-FINE usa la receta por tamaño de upstream

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

### Conviene saberlo, no requiere acción

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
  lanzar `TypeError`. Un nombre de flag desconocido se sigue rechazando con
  `No such option`.
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

### Checkpoints y datasets

Los checkpoints escritos por 1.4.0 se cargan sin cambios. El
[esquema](/docs/reference/checkpoint-schema) ha ganado `imgsz_h` e `imgsz_w`
para los modelos rectangulares, y sigue escribiendo el escalar
`imgsz = max(h, w)` para los lectores antiguos. Las exportaciones a
[ExecuTorch](/docs/export/executorch) y [MNN](/docs/export/mnn) requieren ahora
un archivo adjunto, `<program>.pte.json` y `<model>.mnn.json` respectivamente, y
las exportaciones de HRNet llevan `pose_input: "person_crop"`. Los formatos de
dataset no cambian.
