---
title: Esquema de checkpoint
seo_title: Esquema de metadatos de checkpoint de LibreYOLO v1.0
description: >-
  Los metadatos que lleva todo checkpoint .pt de LibreYOLO: claves obligatorias,
  añadidos por tarea, claves de runtime de exportación, manifiestos de
  cuantización y campos de entrenamiento.
lead: >-
  Un archivo .pt de LibreYOLO es un diccionario plano guardado con torch.save.
  La clave model contiene el state dict; el resto de claves de primer nivel son
  metadatos que identifican el checkpoint sin analizar el nombre del archivo ni
  inspeccionar el state dict.
keywords:
  - esquema checkpoint libreyolo
  - metadatos checkpoint pytorch
  - schema_version 1.0
  - model_family
  - manifiesto cuantización quant
  - wrap_libreyolo_checkpoint
last_verified: 1.5.0
verification: >-
  Refleja docs/checkpoint_schema.md del repositorio libreyolo en la v1.5.0,
  contrastado con libreyolo/utils/serialization.py y BaseModel.save.
snippets:
  usage:
    - label: Leer los metadatos de un checkpoint
      language: python
      code: >
        from libreyolo import LibreYOLO

        from libreyolo.utils.serialization import unwrap_libreyolo_checkpoint

        import torch


        # Descarga un checkpoint y vuelve a guardarlo para que exista una ruta
        local.

        LibreYOLO("LibreYOLO9t.pt").save("roundtrip.pt")


        loaded = torch.load("roundtrip.pt", map_location="cpu",
        weights_only=False)

        state_dict, metadata = unwrap_libreyolo_checkpoint(loaded)


        print(metadata["schema_version"], metadata["model_family"])

        print(metadata["size"], metadata["task"], metadata["nc"],
        metadata["imgsz"])

        print(len(state_dict), "tensors")
source_hash: ce760f1bed97bfd0
---

## Esquema v1.0

Todo checkpoint `.pt` oficial de LibreYOLO contiene:

```python
{
    "model": state_dict,
    "schema_version": "1.0",
    "libreyolo_version": "0.x.y",
    "model_family": "yolo9",
    "size": "t",
    "task": "detect",
    "nc": 80,
    "names": {0: "cat", 1: "dog"},
    "imgsz": 640,
}
```

| Clave | Tipo | Significado |
|---|---|---|
| `model` | state dict | Los pesos del modelo |
| `schema_version` | str | Versión del contrato de metadatos; la v1.0 usa la cadena `"1.0"` |
| `libreyolo_version` | str | La versión que generó el checkpoint |
| `model_family` | str | Una familia registrada, como `yolo9`, `rfdetr`, `dfine`, `ec` |
| `size` | str | Variante dentro de la familia, como `t`, `s`, `r18`, `atto` |
| `task` | str | Nombre canónico de la tarea |
| `nc` | int | Número de clases, positivo |
| `names` | dict | `dict[int, str]` con claves en `0..nc-1` |
| `imgsz` | int | Resolución de entrada cuadrada y positiva, o el escalar heredado para un contrato rectangular |

`task` es uno de `detect`, `segment`, `semantic`, `panoptic`, `pose`,
`classify`, `gaze`, `obb`, `point`, `depth`, `edge`, `normal`, `restore`,
`matte`, `ocr`, `embed` o `mesh`.

Los checkpoints oficiales escriben todas las claves de `names`. Los lectores
pueden rellenar las claves ausentes con etiquetas `class_i` para mapeos
dispersos heredados, pero las claves fuera de rango no son válidas.

Los checkpoints rectangulares mantienen un `imgsz` escalar para los lectores
antiguos, con el valor `max(imgsz_h, imgsz_w)`, y además escriben `imgsz_h` e
`imgsz_w` con las dimensiones reales. Un lector que entienda los campos
rectangulares debe darles preferencia sobre el escalar. Las familias con un
contrato rectangular fijo, como la pose de HRNet, rechazan los tamaños de
runtime incompatibles.

El esquema es plano de forma deliberada, y `model` es un state dict también de
forma deliberada.

<code-tabs name="usage" />

## Añadidos de pose

La pose suele ser de una sola clase, `nc: 1` con `person`, pero la cabeza de
pose de YOLO-NAS también admite pose multiclase con un único esqueleto de
keypoints compartido, en cuyo caso `nc` y `names` describen las clases igual
que en detección. Las exportaciones de pose para runtime emiten `scores` con
forma `[batch, anchors, nc]`.

| Clave | Significado |
|---|---|
| `num_keypoints` | Número positivo de keypoints que usa la cabeza de pose |
| `keypoint_dim` | `2` para etiquetas `x,y` o `3` para etiquetas `x,y,visibility`; las salidas del modelo siempre exponen `x,y,visibility` |
| `oks_sigmas` | Sigmas OKS opcionales por keypoint; cuando no están, se usa el valor por defecto de la tarea para `num_keypoints` |
| `num_keypoints_per_class` | Recuentos opcionales de keypoints por clase para cabezas de estilo GroupPose cuyo tensor de keypoints está rellenado por clase; `0` para las clases sin keypoints |

## Añadidos de mesh

Los checkpoints de mesh usan `task: "mesh"`, `nc: 1` y `names: {0: "person"}`.
La disposición de los parámetros varía entre modelos de cuerpo, así que las
dimensiones se registran en lugar de darse por supuestas.

| Clave | Significado |
|---|---|
| `body_model` | La parametrización, como `mhr`; obligatoria, y se usa para interpretar todos los campos siguientes |
| `num_betas` | Número de coeficientes de identidad y forma; 45 en MHR |
| `num_body_pose` | Anchura del bloque de parámetros de pose corporal; 130 en MHR. Es un vector plano, no un triplete por articulación, porque las articulaciones del rig tienen grados de libertad distintos |
| `num_vertices` | Número de vértices que emite el decodificador; 18439 en MHR |
| `num_joints` | Número de articulaciones que emite el decodificador; 127 en MHR |
| `rotation_format` | Cómo se codifican las rotaciones, como `euler_zyx` en MHR o `axis_angle`. Nunca se infiere de la forma del tensor, porque un vector de 3 es ambiguo |

## Marcadores para tareas densas

Varias tareas predicen mapas densos en lugar de clases, así que los campos de
tipo clase existen solo por compatibilidad con el esquema.

| Tarea | `nc` | `names` |
|---|---|---|
| `depth` | 1 | `{0: "depth"}` |
| `edge` | 1 | `{0: "edge"}` |
| `restore` | 1 | `{0: "image"}` |
| `ocr` | 1 | `{0: "text"}` |

Las predicciones de edge son mapas densos de probabilidad en float32 dentro de
`[0, 1]`.

Los checkpoints de restore pueden añadir `degradation`, una etiqueta corta de
degradación como `deblur`, `denoise` o `super-resolution`; `dataset`, una
etiqueta de procedencia como `GoPro` o `SIDD`; y `scale`, un factor entero
positivo de ampliación de la salida respecto a la entrada, por ejemplo `4`
para un modelo de superresolución x4. Si falta o vale `1`, la imagen
restaurada conserva la resolución de entrada. El runtime también deduce la
escala a partir de la familia y el tamaño, así que `scale` es metadato de
procedencia y no un requisito en tiempo de carga.

## Añadidos de OCR

La familia `ppocr` distribuye un checkpoint compuesto por cada nivel, cuyo
state dict `model` contiene dos submodelos bajo los espacios de nombres de
claves `det.*` y `rec.*`.

| Clave | Significado |
|---|---|
| `charset` | El alfabeto CTC completo en orden de índice de salida: el índice 0 es el blank de CTC, después el diccionario de reconocimiento y después el carácter de espacio. Los cargadores deben leerlo del checkpoint, nunca de un archivo aparte |
| `pipeline` | Valores por defecto del pipeline fijados en el momento de la conversión: `det_limit_side_len`, `det_db_thresh`, `det_db_box_thresh`, `det_db_unclip_ratio`, `rec_image_shape`. Los argumentos de runtime pueden sobrescribirlos en cada llamada |
| `components` | Reservado para etapas opcionales del pipeline como la orientación del documento, el desdoblado y la rotación de las líneas de texto. Vacío en la v1 |

## Metadatos de runtime en la exportación

Los artefactos exportados usan la misma convención de doble escritura
rectangular: `imgsz_h` e `imgsz_w` se escriben junto al escalar heredado
`imgsz`, y un lector que no entienda los campos rectangulares no debe tratar
el escalar en silencio como un contrato cuadrado.

El soporte rectangular en runtime está acotado por familia y por formato. Las
exportaciones de la familia YOLO9, HRNet, NAFNet y Real-ESRGAN pueden usar
`imgsz_h` e `imgsz_w` no cuadrados en los formatos compatibles; las familias o
los formatos sin soporte rectangular explícito rechazan esos metadatos en
lugar de preprocesar esos artefactos como cuadrados. Las exportaciones de
HRNet son cabezas fijas de recorte de persona, con batch uno y en FP32, donde
W32 acepta 256x192 y W48 acepta 384x288, y el detector de personas no va
embebido en el grafo.

Las exportaciones con NMS embebido pueden añadir estas claves planas:

| Clave | Significado |
|---|---|
| `nms` | Booleano en forma de cadena; `"true"` significa que el grafo incluye una salida de postprocesado embebida |
| `nms_conf` | Umbral de confianza fijado en la salida embebida |
| `nms_iou` | Umbral de IoU fijado en la salida embebida |
| `max_det` | Número máximo de filas de detección post-NMS que emite la salida embebida |
| `nms_raw_output` | Booleano en forma de cadena; `"true"` significa que el grafo expone además una salida cruda auxiliar del detector |

En las exportaciones ONNX de detección de YOLO9 con `nms=true`, la salida `0`
(llamada `output`) es el tensor post-NMS autónomo con los umbrales fijados en
la exportación. Cuando `nms_raw_output=true`, la salida `1` (llamada `raw`)
queda reservada para los backends de LibreYOLO, de modo que puedan aplicar el
recorte nativo al lienzo original y la semántica de
`predict(conf=..., iou=..., max_det=...)` en runtime. Los consumidores de
terceros deberían usar la primera salida.

Las exportaciones de pose pueden añadir `num_keypoints`; `keypoint_dim`, donde
las exportaciones crudas de estilo GroupPose pueden usar valores mayores como
`8` cuando el tensor incluye campos de precisión o de logits de clase;
`num_keypoints_per_class` como lista codificada en JSON, donde deben
conservarse los huecos de clase con cero keypoints porque definen el esquema;
y `pose_input`, donde `"person_crop"` significa que el grafo consume un
recorte ya extraído y no contiene detector. Las exportaciones de HRNet para
runtime exigen ese valor.

Las exportaciones de clasificación pueden añadir `crop_pct`, un ratio float de
recorte central cuyo objetivo de redimensionado previo al recorte es
`round(imgsz / crop_pct)` y que vale `0.875` por defecto cuando no está, e
`interpolation`, `"bilinear"` o `"bicubic"`, con `"bilinear"` por defecto.

Las exportaciones de ExecuTorch escriben los metadatos planos en un sidecar
`<program>.pte.json` obligatorio. El contrato v1 es CPU, FP32, batch 1 y un
lienzo de entrada fijo, y además exige `executorch_version`,
`executorch_delegate` igual a `"xnnpack"` y un
`executorch_delegate_partitions` positivo. El cargador rechaza un sidecar que
declare otro delegate, formas dinámicas o una precisión distinta de FP32.

Las exportaciones de MNN escriben los metadatos planos en un sidecar
`<model>.mnn.json` obligatorio. El contrato v1 es CPU, FP32, solo detección y
una forma de entrada NCHW fija, y además exige `mnn_version`, `mnn_backend`
igual a `"cpu"`, `mnn_input_names` y `mnn_output_names` ordenados y no vacíos,
`mnn_input_shape` como cuatro enteros positivos en el orden
`[batch, channels, height, width]`, y `mnn_batch` igual a
`mnn_input_shape[0]`. El cargador rechaza los metadatos dinámicos, los que no
son FP32, los que no son de detección, los de una familia no soportada o los
que tienen formas inconsistentes.

Un `.pte` y un `.mnn` son artefactos específicos de un backend, no checkpoints
de PyTorch.

## Checkpoints cuantizados

Un modelo cuantizado añade una clave plana opcional, `quant`, que contiene un
diccionario de manifiesto con `schema`, `recipe`, `keep_high_precision`,
`execution`, la procedencia de la calibración, `module_count` y `state`. Los
manifiestos FP8 pueden llevar además `fp8_tensorwise_weights`, la lista exacta
de nombres de módulos `QuantLinear` cuya escala de pesos es por tensor en
lugar de por canal de salida. Un cargador que vea `quant` reconstruye la
estructura de módulos cuantizados y la política de escalado antes de
`load_state_dict`.

`state` distingue las dos formas del artefacto.

`"prepared"`, la opción por defecto, contiene los pesos maestros en FP32 más
los buffers de escala `_q_*` y se puede entrenar. Un lector sin soporte de
cuantización puede ignorar la clave `quant` y cargar los maestros como un
modelo en coma flotante.

`"finalized"` es la forma de despliegue que escribe `export(format="pt")`. Se
eliminan los maestros y cada módulo cuantizado lleva en su lugar los pesos
empaquetados:

| Receta | Tensores empaquetados | Descuantización |
|---|---|---|
| int8 | `weight_packed` int8 con la forma original de los pesos, `_q_w_scale` FP32 por canal | `weight_packed * scale` |
| fp8 | `weight_packed` float8_e4m3fn con la forma original, `_q_w_scale` FP32 con una entrada por canal de salida | `weight_packed * scale` |
| w4a16, w4a8 | `weight_packed` uint8, dos códigos de 4 bits por byte, primero el nibble bajo, código `q + 8`; `_q_w_gscale` FP32 `[out, ngroups]`, grupo de 128 a lo largo de in_features | Escala por grupos |
| int2 | Cuatro códigos de 2 bits por byte, código `q + 2`, grupo de 64 | Escala por grupos |
| nvfp4 | `weight_packed` uint8 `[out, ceil(in/16)*8]`, código `sign<<3 \| E2M1 level`; `weight_block_scale` float8_e4m3fn `[out, ceil(in/16)]`; `_q_w_amax` FP32 por tensor | `block_scale * amax / (448 * 6)` |
| mxfp4 | Igual que nvfp4 pero con bloques de 32 elementos, más `weight_block_exp` int8 `[out, ceil(in/32)]` | `2 ** exponent` |

Los buffers de rango de activación `_q_act_lo`, `_q_act_hi` y `_q_calibrated`
se conservan para int8. El manifiesto registra `remainder`, `"fp16"` o
`"fp32"`, para los tensores no cuantizados. El desempaquetado reproduce la
simulación bit a bit, así que la inferencia finalizada coincide exactamente
con la inferencia preparada en el dispositivo donde se finalizó. Esta
disposición es el contrato estable para exportadores y runtimes externos.

## Checkpoints de entrenamiento

Los checkpoints del entrenador usan el mismo núcleo obligatorio de metadatos y
pueden añadir campos planos de entrenamiento y de reanudación:

```python
{
    "model": state_dict,
    "epoch": 42,
    "optimizer": optimizer_state_dict,
    "config": {},
    "loss": 1.23,
    "best_metric_key": "metrics/mAP50-95",
    "best_metric_value": 0.51,
    "best_epoch": 39,
    "is_ema_weights": True,
    "train_model": raw_state_dict,
    "ema": ema_state_dict,
    "ema_updates": 12345,
}
```

`is_ema_weights` declara si el `model` de primer nivel está suavizado con EMA.
Cuando EMA está activado, `train_model`, `ema` y `ema_updates` conservan el
estado de reanudación. Los pesos de inferencia que se publican deberían ser
ligeros y no incluir el optimizador, la epoch, la config, la loss ni el estado
de reanudación de EMA, salvo que se distribuyan a propósito como checkpoints
de entrenamiento.

Por compatibilidad entre versiones, los lectores aceptan los alias heredados
de mejor métrica `best_mAP50_95`, `best_mAP50`, `best_metric` y
`best_metric_name`.

## Snapshots externos

El esquema rige los archivos `.pt` creados por LibreYOLO. No renombra ni
envuelve los snapshots upstream de varios archivos que usan los niveles de
modelos independientes.

El tamaño `14b-a7b` de LibreMODUS es una excepción explícita: el alias se
resuelve a través de `LibreVLM(...)` hacia un directorio de archivos upstream
fijados, y LibreYOLO ni le añade metadatos v1.0 ni lo republica como un `.pt`.

## Pesos heredados y de terceros

Los escritores nuevos validan de forma estricta y deben emitir metadatos v1.0.
Cuando los metadatos faltan o están incompletos, los checkpoints antiguos con
aspecto de LibreYOLO se cargan por la vía de compatibilidad con un aviso e
instrucciones de conversión, y los checkpoints upstream de terceros se dirigen
a la conversión automática. Consulta
[checkpoints upstream](/docs/reference/upstream-checkpoints).

## Helpers

Los helpers del esquema están en `libreyolo.utils.serialization`:

```python
wrap_libreyolo_checkpoint(
    state_dict,
    *,
    model_family,
    size,
    task,
    nc,
    names=None,
    imgsz=None,
    libreyolo_version=None,
    schema_version="1.0",
    **extra_metadata,
) -> dict

validate_checkpoint_metadata(checkpoint, *, strict=False) -> list[str]

unwrap_libreyolo_checkpoint(loaded, *, strict=False) -> tuple[dict, dict]
```

`validate_checkpoint_metadata` no muta nada y devuelve la lista de errores;
con `strict=True` lanza `CheckpointMetadataError` en su lugar.
`model.save(path)` es la forma soportada de escribir un checkpoint conforme.
