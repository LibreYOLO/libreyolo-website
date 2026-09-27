---
title: libreyolo train
seo_title: referencia del comando libreyolo train
description: >-
  Entrena un modelo desde la línea de comandos: los 59 argumentos con sus
  valores por defecto, cómo los sustituyen los valores por defecto de cada
  familia y qué argumentos ignora cada familia.
lead: >-
  Entrena un modelo sobre un dataset y escribe checkpoints, métricas y logs en
  un directorio de ejecución. Cada argumento de abajo tiene un valor por defecto
  tomado de la definición del comando, que la configuración de entrenamiento
  propia de cada familia de modelos puede sustituir.
keywords:
  - libreyolo train cli
  - entrenar yolo linea de comandos
  - comando libreyolo train
  - argumentos libreyolo train
  - entrenar yolo dataset propio
  - congelar capas yolo
last_verified: 1.5.0
meta:
  - label: Comando
    value: libreyolo train
    mono: true
  - label: Requerido
    value: data
    mono: true
  - label: Salida
    value: 'Checkpoints, métricas y logs en runs/train/exp'
snippets:
  examples:
    - label: Básico
      language: bash
      code: >
        # coco8.yaml viene con el paquete y descarga sus 8 imágenes en el primer
        uso.

        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml epochs=10 imgsz=640
        batch=8
    - label: Comprobar primero la configuración resuelta
      language: bash
      code: >
        # Imprime lo que usaría la ejecución, incluidos los valores por defecto
        de

        # la familia, y sale sin entrenar ni cargar datos.

        libreyolo train model=LibreDFINEn.pt data=coco8.yaml epochs=10
        dry_run=true
    - label: Ejecución con nombre y receta explícita
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml \
          epochs=50 batch=8 optimizer=adamw lr0=0.001 weight_decay=0.0001 \
          patience=20 save_period=5 project=runs/train name=yolo9s-coco8 exist_ok=true
source_hash: 3aad4298310d3081
---

## Sinopsis

```bash
libreyolo train data=<dataset.yaml> [model=<name|path>] [key=value ...]
```

Los argumentos son pares `key=value`, y la forma POSIX también funciona, de modo
que `epochs=50` y `--epochs 50` son el mismo argumento. Los booleanos aceptan
`true` y `false`: `amp=false` se convierte en `--no-amp` cuando el flag tiene
forma negativa.

## Argumentos

### Modelo y datos

| Argumento | Por defecto | Significado |
|---|---|---|
| `data` | | Ruta al YAML del dataset (formato YOLO, p. ej. `coco8.yaml`). Requerido |
| `model` | `yolox-s` | Nombre del modelo o ruta a los pesos |
| `task` | | Tarea explícita que sustituye a la detectada: `detect`, `segment`, `semantic`, `pose`, `classify`, `gaze`, `obb`, `point`, `depth` |
| `pretrained` | `true` | Usar pesos preentrenados. Con `false` se construye la arquitectura y se entrena desde cero |
| `allow_download_scripts` | `false` | Permitir Python embebido en los bloques de descarga del YAML del dataset |

### Bucle de entrenamiento

| Argumento | Por defecto | Significado |
|---|---|---|
| `epochs` | `300` | Épocas de entrenamiento |
| `batch` | `16` | Tamaño de batch por dispositivo |
| `imgsz` | `640` | Tamaño de imagen de entrenamiento: `640` (cuadrada) o `480x640` (alto x ancho) |
| `device` | `auto` | Dispositivo: `0`, `cpu`, `mps`, `auto` |
| `workers` | `4` | Workers del dataloader |
| `cache` | `false` | Cachear imágenes para acelerar la carga de datos: `ram`, `disk`, `true`, `false` |
| `seed` | `0` | Semilla aleatoria |
| `resume` | | Reanudar el entrenamiento: `true`, o una ruta a un checkpoint |
| `amp` | `true` | Precisión mixta automática (AMP) |
| `amp_dtype` | `float16` | dtype de AMP en CUDA: `float16` o `bfloat16` |
| `cuda_graph` | `false` | Captura el forward y el backward del entrenamiento en CUDA graphs. Solo una GPU y solo familias compatibles; el resto se ejecutan en modo eager |
| `lora` | `false` | Fine-tuning con LoRA, para las familias transformer listadas en Notas |
| `freeze` | | Congelar capas: un número entero, una lista de índices o nombres de módulos |

### Destilación

| Argumento | Por defecto | Significado |
|---|---|---|
| `distill_model` | | Teacher: un checkpoint de detector, o un id de foundation teacher como `dinov2` para destilación de características del backbone |
| `dis` | | Peso de la loss de destilación. Si no se indica, el valor publicado por defecto para ese tipo de loss |
| `distill_loss_type` | `mgd` | Loss de características para teachers de tipo detector: `mgd`, `cwd`. Los foundation teachers usan siempre `feat_mse` |

### Optimizador

| Argumento | Por defecto | Significado |
|---|---|---|
| `optimizer` | `sgd` | Optimizador: `sgd`, `adam`, `adamw` |
| `lr0` | `0.01` | Learning rate inicial |
| `momentum` | `0.937` | Momentum de SGD, y el coeficiente de primer momento para los optimizadores Adam |
| `weight_decay` | `0.0005` | Regularización L2 |
| `nesterov` | `true` | Momentum de Nesterov |

### Scheduler

| Argumento | Por defecto | Significado |
|---|---|---|
| `scheduler` | `yoloxwarmcos` | Tipo de planificación del LR |
| `warmup_epochs` | `5` | Duración del warmup |
| `warmup_lr_start` | `0.0` | LR inicial del warmup |
| `min_lr_ratio` | `0.05` | Ratio de LR mínimo |
| `lr_drop` | `100` | Época de caída escalonada del LR en RF-DETR |

### Aumento de datos

| Argumento | Por defecto | Significado |
|---|---|---|
| `mosaic` | `1.0` | Probabilidad de mosaic |
| `mixup` | `1.0` | Probabilidad de mixup |
| `hsv_prob` | `1.0` | Probabilidad de jitter HSV |
| `flip_prob` | `0.5` | Probabilidad de volteo horizontal |
| `degrees` | `10.0` | Rango de rotación, en más y en menos, en grados |
| `translate` | `0.1` | Ratio de traslación |
| `shear` | `2.0` | Ángulo de cizalladura |
| `mosaic_scale` | `(0.1,2.0)` | Rango de escala del mosaic |
| `mixup_scale` | `(0.5,1.5)` | Rango de escala del mixup |
| `no_aug_epochs` | `15` | Desactivar el aumento de datos durante las últimas N épocas |

### EMA

| Argumento | Por defecto | Significado |
|---|---|---|
| `ema` | `true` | Media móvil exponencial |
| `ema_decay` | `0.9998` | Factor de decaimiento de la EMA |

### Validación durante el entrenamiento

| Argumento | Por defecto | Significado |
|---|---|---|
| `val` | `true` | Validar durante el entrenamiento |
| `eval_interval` | `10` | Validar cada N épocas |
| `max_det` | `300` | Máximo de predicciones por imagen tras el NMS de validación |
| `eval_max_det` | | Tope del evaluador COCO. Si no se indica, la convención AP@100 de pycocotools |
| `faster_coco_eval` | `true` | Usar el backend C++ faster-coco-eval para las métricas COCO cuando esté instalado; si no, recurre a pycocotools |
| `save_plots` | `false` | Guardar las gráficas finales de validación durante el entrenamiento |
| `patience` | `50` | Paciencia del early stopping. `0` lo desactiva |

### Salida

| Argumento | Por defecto | Significado |
|---|---|---|
| `project` | `runs/train` | Raíz del directorio de salida |
| `name` | `exp` | Nombre del experimento |
| `exist_ok` | `false` | Reutilizar el directorio de salida existente |
| `save_period` | `10` | Guardar un checkpoint cada N épocas |
| `log_interval` | `10` | Registrar la loss cada N batches |

### Flags de agente

| Argumento | Por defecto | Significado |
|---|---|---|
| `json` | `false` | Salida JSON por stdout |
| `quiet` | `false` | Silenciar stderr |
| `dry_run` | `false` | Resolver e imprimir la configuración sin ejecutar |
| `help_json` | `false` | Volcar el esquema del comando como JSON y salir |

## Ejemplos

<code-tabs name="examples" />

## Notas

### Los valores por defecto de arriba no siempre son los que se usan

Cada familia de modelos lleva su propia configuración de entrenamiento y, allí
donde esa configuración difiere de la base, su valor sustituye al valor por
defecto del comando para cualquier argumento que no hayas fijado explícitamente.
Fijar tú mismo el argumento gana siempre. `libreyolo cfg` imprime los valores
por defecto base y las sustituciones de cada familia, que es la forma de ver qué
usará realmente una familia dada.

`imgsz` es el argumento donde más importa. El valor por defecto del comando es
`640`, que no es la entrada nativa de todos los checkpoints: los tamaños de
detección publicados de RF-DETR son 384, 512, 576 y 704, y los checkpoints `n` y
`t` de YOLOX son de 416. RF-DETR y DEIMv2 se tratan pasándoles `imgsz` solo
cuando se ha fijado explícitamente, de modo que en los demás casos sigue vigente
su propio tamaño. A las otras familias se les entrega el valor tal cual y
entrenan con él. FOMO es la estricta: cada tamaño acepta únicamente su entrada
nativa (96, 192 y 224), así que una ejecución de FOMO necesita `imgsz` fijado
para que coincida o se detiene con un error. RF-DETR exige además que el valor
sea divisible por su tamaño de patch multiplicado por su número de ventanas, e
informa de los dos tamaños válidos más cercanos cuando no lo es.

### Argumentos que una familia ignora

No todas las familias leen todos los argumentos, y donde más se nota es en los
del aumento de datos. RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETRv4 y DINOv2 entrenan
con pipelines de paso directo, sin mosaic, sin mixup y sin deformación afín, así
que ahí `mosaic`, `mixup`, `hsv_prob`, `degrees`, `translate`, `shear`,
`mosaic_scale` y `mixup_scale` no llegan a nada. EC comparte ese pipeline, pero
sí lee `hsv_prob`, `degrees` y `translate` cuando su tarea es pose. Las familias
de clasificación, SegFormer y NAFNet ignoran todo ese conjunto y `flip_prob` con
él, porque su volteo se aplica con una probabilidad fija en lugar de una
configurable. YOLO-NAS ignora solo `mosaic`, ya que en su lugar aumenta con una
transformación afín por muestra siempre activa. RF-DETR ignora tres más además
de esa lista: `optimizer`, `momentum` y `nesterov`.

Fijar uno de estos no es un error. La ejecución escribe una línea en stderr con
el nombre de la familia y los argumentos que va a ignorar, y a continuación
entrena; esa línea es la lista autoritativa para la versión instalada. También es
la única señal, así que una ejecución con `quiet=true` dentro de un script
silencia el aviso junto con todo lo demás que va a stderr.

`val=false` es un caso relacionado. Pone `eval_interval` a `0` en la mayoría de
familias; RF-DETR no puede desactivar la validación de esa forma y registra que
ha ignorado la petición.

### Otros comportamientos que conviene conocer

`lora=true` lo aceptan RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETR v1, v2 y v4, EC y
ConvNeXt. Cualquier otra familia sale con `config_unsupported` en lugar de
entrenar sin ello.

`pretrained=false` combinado con `resume` se rechaza en las familias que admiten
el entrenamiento desde cero, ya que ambos piden cosas opuestas.

`mosaic` y `mixup` son las grafías de línea de comandos de los campos de
configuración `mosaic_prob` y `mixup_prob`. En las familias cuyo mixup solo se
aplica a las muestras de mosaic, un `mixup` por encima de cero con `mosaic` a
cero no se activa nunca, y la ejecución lo indica.

`dry_run=true` resuelve la referencia del modelo, aplica los valores por defecto
de la familia e imprime la configuración con la que entrenaría. No carga el
dataset, así que es la forma barata de confirmar que un argumento ha llegado al
valor que esperabas.

stdout lleva el objeto de resultado final; el progreso y los avisos van a stderr.
El código de salida es `0` si todo va bien, `2` ante un error de uso o de
configuración, `3` cuando no se encuentra o no se puede leer el dataset, `4`
cuando no se puede cargar el modelo, y `1` para el resto de fallos en tiempo de
ejecución.

Relacionado: [`libreyolo doctor`](/docs/cli/doctor) para comprobar un dataset
antes de comprometerte con una ejecución, [`libreyolo monitor`](/docs/cli/monitor)
para seguir una ejecución en el navegador, [`libreyolo val`](/docs/cli/val) para
medir el resultado.
