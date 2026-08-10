---
title: libreyolo val
seo_title: "referencia del comando libreyolo val"
description: "Evalúa un checkpoint sobre un split de un dataset desde la línea de comandos: cada argumento con su valor por defecto y las claves de métricas que devuelve cada tarea."
lead: "Evalúa un modelo contra un split de un dataset e imprime las métricas. El conjunto de métricas depende de la tarea del modelo, y los números son los mismos con los que se construye una fila de benchmark."
keywords: [libreyolo val cli, comando validacion libreyolo, evaluar modelo yolo linea de comandos, calcular mAP50-95 terminal, argumentos libreyolo val]
last_verified: "1.5.0"
meta:
  - label: Comando
    value: libreyolo val
    mono: true
  - label: Requerido
    value: model, data
    mono: true
  - label: Salida
    value: "Métricas por stdout. Gráficas y JSON COCO en runs/val/exp cuando se piden"
snippets:
  examples:
    - label: Básico
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml
    - label: Gráficas y JSON COCO
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml \
          imgsz=640 batch=8 save_json=true save_plots=true \
          project=runs/val name=yolo9s-coco8 exist_ok=true
    - label: Legible por máquina
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml json=true quiet=true
---

## Sinopsis

```bash
libreyolo val model=<name|path> data=<dataset.yaml> [key=value ...]
```

Los argumentos son pares `key=value`, y la forma POSIX también funciona, así que
`batch=8` y `--batch 8` son el mismo argumento.

## Argumentos

| Argumento | Por defecto | Significado |
|---|---|---|
| `model` | | Ruta de los pesos del modelo o nombre de CLI. Requerido |
| `data` | | Ruta al YAML del dataset (formato YOLO, p. ej. `coco8.yaml`). Requerido |
| `data_dir` | | Directorio del dataset directo, ignorando la ruta indicada en el YAML |
| `split` | `val` | Split del dataset: `val`, `test`, `train` |
| `batch` | `16` | Tamaño de batch |
| `imgsz` | | Tamaño de la imagen: `640` (cuadrada) o `480x640` (alto x ancho). El tamaño de entrada propio del modelo si no se indica |
| `conf` | `0.001` | Umbral de confianza |
| `iou` | `0.6` | Umbral de IoU para NMS |
| `max_det` | `300` | Máximo de predicciones por imagen tras el NMS |
| `eval_max_det` | | Tope del evaluador COCO. La convención AP@100 de pycocotools si no se indica |
| `faster_coco_eval` | `true` | Usa el backend C++ de faster-coco-eval para las métricas COCO cuando está instalado; si no, recurre a pycocotools |
| `half` | `false` | Inferencia en FP16 |
| `amp_dtype` | `float16` | Dtype del autocast de CUDA cuando `half=true`: `float16` o `bfloat16` |
| `save_json` | `false` | Guarda los resultados en JSON con formato COCO |
| `save_plots` | `false` | Guarda las gráficas de validación: métricas, AP por clase, matriz de confusión, muestras |
| `workers` | `4` | Workers del dataloader |
| `device` | `auto` | Dispositivo |
| `project` | `runs/val` | Raíz del directorio de salida |
| `name` | `exp` | Nombre del experimento |
| `exist_ok` | `false` | Reutiliza el directorio de salida |
| `allow_download_scripts` | `false` | Permite Python embebido en los bloques de descarga del YAML del dataset |
| `json` | `false` | Salida JSON por stdout |
| `quiet` | `false` | Silencia stderr |
| `verbose` | `true` | Salida detallada |
| `help_json` | `false` | Vuelca el esquema del comando como JSON y sale |

## Ejemplos

<code-tabs name="examples" />

## Notas

### Qué son las métricas

El conjunto que se imprime sigue a la tarea del modelo, y la salida JSON usa las
mismas claves.

Detección, segmentación y cajas orientadas reportan `mAP50`, `mAP50_95`,
`precision` y `recall`. Cuando un modelo predice más de un tipo de salida, los
grupos por tipo aparecen al lado como `box_metrics`, `mask_metrics` y
`obb_metrics`, cada uno con esas mismas cuatro claves.

La clasificación reporta `accuracy_top1` y `accuracy_top5`. La detección de
puntos reporta `precision`, `recall`, `f1`, `MLE`, `MAE`, `RMSE` y `mAP_sweep`.
La profundidad reporta `abs_rel`, `rmse`, `delta1`, `delta2` y `delta3`. La
segmentación semántica reporta `mIoU` y `pixel_accuracy`. La restauración
reporta `PSNR` y `SSIM`.

El resultado JSON lleva además `eval_backend`, que nombra la biblioteca de
evaluación COCO y la versión que produjeron los números, de modo que dos
ejecuciones se pueden comparar sabiendo si el mismo backend puntuó ambas.

### Umbrales

Los valores por defecto de aquí son valores de evaluación, no de predicción:
`conf` es `0.001` e `iou` es `0.6`, mientras que
[`libreyolo predict`](/docs/cli/predict) usa `0.25` y `0.45`. Subir `conf` a un
umbral de visualización baja el recall y con él la mAP, así que un número
obtenido de esa forma no es comparable con uno publicado.

`imgsz` no está fijado por defecto, lo que significa el tamaño de entrada propio
del modelo. Fijarlo evalúa al tamaño indicado, que es la manera de medir un
checkpoint fuera de su resolución nativa.

### Datasets que se descargan

Un YAML de dataset cuyo campo `download` es una URL se descarga en el primer uso
sin ningún permiso adicional. Uno que lleva un script de descarga en Python
embebido necesita `allow_download_scripts=true`, y el comando avisa por stderr de
que se ha habilitado la ejecución de código local. Los `coco8.yaml` y
`coco128.yaml` incluidos están basados en URL, así que no necesitan nada.

### Salida y códigos de salida

stdout lleva las métricas; el progreso va a stderr. `json=true` imprime un único
objeto con `schema_version`, y `quiet=true` silencia stderr.

El código de salida es `0` en caso de éxito, `2` para un error de uso o de
configuración, `3` cuando el dataset no se encuentra, `4` cuando el modelo no se
puede cargar y `1` para otros fallos en tiempo de ejecución.

Relacionado: [`libreyolo train`](/docs/cli/train), que ejecuta esta misma
evaluación según su propio calendario a través de `eval_interval`.
