---
title: NVIDIA DeepStream
seo_title: Ejecutar modelos YOLO en NVIDIA DeepStream
description: >-
  Exporta un modelo LibreYOLO para NVIDIA DeepStream: un grafo ONNX más un
  config de nvinfer generado. Comandos exactos para compilar el parser y para el
  pipeline.
lead: >-
  NVIDIA DeepStream ejecuta la inferencia a través de su elemento nvinfer, que
  necesita un grafo ONNX, un archivo de configuración que le corresponda y un
  parser de bounding boxes. Poner deepstream=True en la exportación a ONNX
  escribe los dos primeros y los conecta con el tercero.
keywords:
  - deepstream yolo
  - exportar yolo a deepstream
  - nvinfer config yolo
  - parser de bounding boxes deepstream
  - config_infer_primary
  - NvDsInferParseYolo
  - deepstream-app
  - motor tensorrt deepstream
  - jetson deepstream
meta:
  - label: Flag
    value: 'export(format="onnx", deepstream=True)'
    mono: true
  - label: Escribe
    value: 'Un grafo ONNX, config_infer_primary_<stem>.txt y <stem>_labels.txt'
  - label: Cobertura
    value: 43 combinaciones de familia y tarea repartidas en nueve tareas
  - label: Parser
    value: >-
      NvDsInferParseYolo, del proyecto DeepStream-Yolo de Marcos Luciano, con
      licencia MIT. Se compila una vez por dispositivo.
    links:
      - label: github.com/marcoslucianops/DeepStream-Yolo
        href: 'https://github.com/marcoslucianops/DeepStream-Yolo'
  - label: Disponibilidad
    value: Llega en la v1.5.0. Fusionado en dev el 2026-08-08 en la pull request 728.
    links:
      - label: pull request 728
        href: 'https://github.com/LibreYOLO/libreyolo/pull/728'
      - label: issue 648
        href: 'https://github.com/LibreYOLO/libreyolo/issues/648'
  - label: Validado en ejecución
    value: 'DeepStream 8.0.0 en una RTX 5070 Ti, solo detección, 2026-08-08'
verification: >-
  Escrito a partir de la validación en ejecución del 2026-08-08. Las listas de
  familias, las claves de configuración y los valores por defecto están leídos
  de libreyolo/export/deepstream.py y libreyolo/export/exporter.py en el commit
  5f81e11e, que se fusionó en dev ese mismo día en la pull request 728.
snippets:
  install:
    - label: Instalación
      language: bash
      code: |
        pip install "libreyolo[onnx]"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO9, LibreDFINE


        # Escribe libreyolo9s.onnx, config_infer_primary_libreyolo9s.txt

        # y libreyolo9s_labels.txt en el directorio de trabajo.

        LibreYOLO9("libreyolo9s.pt", size="s").export(format="onnx",
        deepstream=True)


        # Mantén cada modelo de detección en su propio directorio: todos los
        configs

        # nombran el mismo archivo de caché del motor. Ver "Trampas conocidas".

        LibreDFINE("LibreDFINEs.pt", size="s").export(format="onnx",
        deepstream=True)
    - label: Argumentos
      language: python
      code: >
        model.export(
            format="onnx",     # deepstream=True se rechaza en cualquier otro formato
            deepstream=True,
            conf=0.25,         # alimenta pre-cluster-threshold (y classifier-threshold,
                               # segmentation-threshold en esas tareas)
            iou=0.45,          # alimenta nms-iou-threshold, se omite con cluster-mode=4
            batch=1,           # alimenta batch-size y el nombre de la caché del motor
            half=False,        # True marca el config con network-mode=2 (build fp16)
            int8=False,        # True marca el config con network-mode=1
            dynamic=True,      # eje de batch dinámico en el grafo ONNX
            imgsz=640,         # alimenta infer-dims=3;H;W
        )


        # deepstream=True y nms=True son mutuamente excluyentes: DeepStream
        ejecuta

        # la supresión en su etapa de clustering, así que no se embebe nada en
        el grafo.
    - label: Descarga primero los pesos de D-FINE
      language: bash
      code: |
        curl -L -o LibreDFINEs.pt \
          https://huggingface.co/LibreYOLO/LibreDFINEs/resolve/main/LibreDFINEs.pt
  gpu:
    - label: Confirma el acceso a la GPU antes que nada
      language: bash
      code: |
        docker run --rm --gpus all nvcr.io/nvidia/tritonserver:26.04-py3 \
          nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
      expect: |
        name, driver_version, compute_cap
        NVIDIA GeForce RTX 5070 Ti, 591.86, 12.0
  parser:
    - label: 'build_parser.sh, ejecútalo dentro del contenedor de DeepStream'
      language: bash
      code: >
        set -e

        git clone --depth 1
        https://github.com/marcoslucianops/DeepStream-Yolo.git


        # /usr/local/cuda-12 en esta imagen es un stub y la compilación muere
        ahí con

        # "fatal error: crt/host_defines.h: No such file or directory". Resuelve
        un

        # toolkit que sí lleve la cabecera; en la imagen 8.0 ese es cuda-12.5.

        CUDA_DIR=$(readlink -f /usr/local/cuda)

        [ -f "$CUDA_DIR/include/crt/host_defines.h" ] || \
          CUDA_DIR=$(ls -d /usr/local/cuda-*.* | sort -Vr | \
                     while read d; do [ -f "$d/include/crt/host_defines.h" ] && echo "$d" && break; done)

        # La imagen trae libcublas.so.12 y libcublas.so.12.8.4.1 pero no el

        # libcublas.so sin versión que -lcublas necesita, así que el enlazado
        falla con

        # "/usr/bin/ld: cannot find -lcublas". Dale al enlazador los nombres que
        quiere.

        mkdir -p /tmp/cudalibs

        for lib in cublas cublasLt cudart; do
          real=$(find /usr/local -name "lib${lib}.so.1*" | grep -v stubs | sort -V | tail -1)
          ln -sf "$real" "/tmp/cudalibs/lib${lib}.so"
        done

        export LIBRARY_PATH="/tmp/cudalibs:$LIBRARY_PATH"


        make -C DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
        CUDA_VER="${CUDA_DIR##*/cuda-}"
    - label: La segmentación de instancias usa otro parser
      language: bash
      code: >
        git clone --depth 1
        https://github.com/marcoslucianops/DeepStream-Yolo-Seg.git

        make -C DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg \
          CUDA_VER="${CUDA_DIR##*/cuda-}"
  run:
    - label: deepstream_app_config.txt
      language: text
      code: >
        [application]

        enable-perf-measurement=1

        perf-measurement-interval-sec=5

        gie-kitti-output-dir=kitti


        [tiled-display]

        enable=0


        [source0]

        enable=1

        type=3

        uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4

        num-sources=1

        gpu-id=0


        [streammux]

        gpu-id=0

        batch-size=1

        batched-push-timeout=40000

        width=1920

        height=1080

        live-source=0


        [primary-gie]

        enable=1

        gpu-id=0

        gie-unique-id=1

        config-file=config_infer_primary_libreyolo9s.txt


        [osd]

        enable=1

        border-width=2

        text-size=15


        [sink0]

        enable=1

        type=1

        sync=0


        [tests]

        file-loop=0
    - label: Ejecútalo
      language: bash
      code: |
        deepstream-app -c deepstream_app_config.txt
      expect: |
        App run successful
    - label: Los dos pasos en un solo contenedor
      language: bash
      code: |
        docker run --rm --gpus all -v "$PWD:/work" -w /work \
          nvcr.io/nvidia/deepstream:8.0-samples-multiarch \
          bash -c "bash build_parser.sh && deepstream-app -c deepstream_app_config.txt"
source_hash: 1ee91c265753dd9a
---

## Disponibilidad

La exportación a DeepStream llega en la v1.5.0. Se fusionó en `dev` el 2026-08-08
en la pull request 728, así que una instalación actual ya la tiene y no hace falta
fijar ninguna rama.

<code-tabs name="install" />

Si clonaste la rama `deepstream-export` antes del 2026-08-08, reemplázala. Esa
rama se rebaseó y se subió con force-push, y al historial antiguo le falta el
arreglo que permite que estas exportaciones funcionen siquiera en una máquina con
CUDA.

## Qué escribe la exportación

`model.export(format="onnx", deepstream=True)` escribe tres archivos uno al lado
del otro. Para `libreyolo9s.pt`:

- `libreyolo9s.onnx`, el grafo de detección, con un tensor de salida de forma
  `(batch, num_detections, 6)`, donde cada fila es
  `[x1, y1, x2, y2, score, class_id]` en coordenadas de píxel de la entrada de la
  red.
- `config_infer_primary_libreyolo9s.txt`, una configuración de `nvinfer` que lleva
  las constantes de preprocesado de la familia, el número de clases, los umbrales
  y el cableado del parser.
- `libreyolo9s_labels.txt`, un nombre de clase por línea.

Aparece un archivo de etiquetas siempre que el checkpoint lleve nombres de clase.
Los modelos de profundidad no tienen ninguno, así que no reciben ni el archivo ni
una clave `labelfile-path`.

LibreYOLO no emite ningún `.so`. El `.so` que carga DeepStream es el parser de
bounding boxes de `marcoslucianops/DeepStream-Yolo`, compilado una vez por
dispositivo, y es el mismo binario apuntes al detector de LibreYOLO que apuntes.
El modelo es el ONNX. La clasificación y la segmentación semántica no necesitan
parser alguno, porque `nvinfer` hace ese postprocesado por su cuenta.

## Exportar el modelo

<code-tabs name="export" />

`LibreDFINE._load_weights` lanza `FileNotFoundError` cuando el archivo no está ya
en disco, sin intentar descargarlo, así que descarga `LibreDFINEs.pt` tú antes. Ese
hueco está registrado en el
[issue #727](https://github.com/LibreYOLO/libreyolo/issues/727). Los pesos de YOLO9
se descargan en el primer uso.

El flag es solo de Python. `libreyolo export` en esta rama no tiene opción
`deepstream`, y la CLI construye sus argumentos de exportación a partir de una
lista fija en lugar de dejar pasar claves desconocidas.

## Compilar el parser de bounding boxes

La detección necesita la biblioteca del parser, la segmentación de instancias
necesita otra distinta, y el resto de tareas no necesitan ninguna. Dos cosas de la
imagen de DeepStream 8.0 rompen el comando de compilación documentado, y ambas son
problemas del entorno más que de LibreYOLO.

La imagen trae `cuda`, `cuda-12`, `cuda-12.5`, `cuda-12.8` y `cuda-12.9` bajo
`/usr/local`. Solo `cuda-12.5` tiene un toolkit completo. También trae
`libcublas.so.12` y `libcublas.so.12.8.4.1` pero no el `libcublas.so` sin versión
contra el que resuelve `-lcublas`. El script de abajo esquiva las dos cosas.

<code-tabs name="parser" />

Después apunta `custom-lib-path` en el config generado al
`libnvdsinfer_custom_impl_Yolo.so` ya compilado. El valor generado es la ruta
relativa `nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so`, que
resuelve bien cuando `deepstream-app` se ejecuta desde el checkout de
`DeepStream-Yolo` y hay que editarla en cualquier otro caso.

## Ejecutar el pipeline

Comprueba que el contenedor ve la GPU antes de gastar tiempo en nada más. Es la
comprobación que hizo primero la ejecución de validación, sobre una tarjeta
Blackwell bajo WSL2.

<code-tabs name="gpu" />

La ejecución de validación movió `deepstream-app` con una única fuente de archivo,
sin sink de display, con el on-screen display activado y con `gie-kitti-output-dir`
puesto para que las detecciones de cada frame acabaran en disco como texto KITTI.
Un config con esos ajustes:

<code-tabs name="run" />

`nvinfer` construye el motor de TensorRT a partir del ONNX en la primera ejecución
y lo cachea junto al modelo, de modo que la primera ejecución paga la construcción
del motor y las siguientes cargan la caché.

## El config generado

Los dos configs de abajo los escribió el exportador para la ejecución de
validación, sin editarlos después.

| Clave | YOLO9-s | D-FINE-s |
|---|---|---|
| `net-scale-factor` | 0.003921568627 | 0.003921568627 |
| `model-color-format` | 0 | 0 |
| `infer-dims` | 3;640;640 | 3;640;640 |
| `maintain-aspect-ratio` | 1 | 0 |
| `symmetric-padding` | 0 | 0 |
| `network-type` | 0 | 0 |
| `num-detected-classes` | 80 | 80 |
| `cluster-mode` | 2 | 4 |
| `parse-bbox-func-name` | NvDsInferParseYolo | NvDsInferParseYolo |
| `pre-cluster-threshold` | 0.25 | 0.25 |
| `nms-iou-threshold` | 0.45 | |
| `topk` | 300 | 300 |

Los dos configs se diferencian en tres puntos: `maintain-aspect-ratio`,
`cluster-mode`, y si `nms-iou-threshold` está presente siquiera. El config de
D-FINE omite esa clave por completo, que es lo que pide `cluster-mode=4`.

Las cabezas que emiten como mucho una predicción por objeto reciben
`cluster-mode=4`, así que DeepStream no ejecuta clustering sobre ellas; el
clustering fusionaría detecciones genuinamente distintas. Eso cubre `rfdetr`,
`dfine`, `deim`, `deimv2`, `ec`, `rtdetr`, `rtdetrv2`, `rtdetrv4` y `yolo9_e2e`.
Las cabezas de grid y de anchors reciben `cluster-mode=2` más
`nms-iou-threshold`.

Los configs de detección también llevan
`engine-create-func-name=NvDsInferYoloCudaEngineGet`, que delega la construcción
del motor a la biblioteca del parser. Eso es lo que fija el nombre del archivo de
caché del motor, y es el origen de la colisión que se describe en las trampas
conocidas.

## Tareas y familias soportadas

Exportan cuarenta y tres combinaciones de familia y tarea.
`deepstream_supported_tasks()` y `deepstream_supported_families(task)` en
`libreyolo/export/deepstream.py` devuelven las mismas listas en tiempo de
ejecución.

| Tarea | `network-type` | Biblioteca del parser | Familias |
|---|---|---|---|
| Detección | 0 | DeepStream-Yolo | yolo9, yolo9_p2, yolo9_e2e, yolo1, yolo2, yolo3, yolo4, yolo7, yolox, yolonas, rtmdet, picodet, rfdetr, dfine, deim, deimv2, ec, rtdetr, rtdetrv2, rtdetrv4 |
| Clasificación | 1 | No hace falta | mobilenetv4, convnext, efficientnetv2, resnet, dinov2 |
| Segmentación semántica | 2 | No hace falta | pidnet, eomt, dinov2, lingbotvision |
| Segmentación de instancias | 3 | DeepStream-Yolo-Seg | rfdetr, dfine, ec |
| Pose | 100 | No hace falta | yolo9, yolonas, rfdetr, ec |
| Profundidad | 100 | No hace falta | depth_anything, zipdepth |
| Restauración | 100 | No hace falta | nafnet, realesrgan, swinir |
| Matting | 100 | No hace falta | birefnet |
| Gaze | 100 | No hace falta | l2cs |

`network-type=100` significa que DeepStream no tiene postprocesador para esa
tarea. Esos configs ponen `output-tensor-meta=1`, las salidas nativas del grafo
pasan sin tocarse, y la aplicación las decodifica desde los metadatos del tensor.
Ahí los grafos con varias salidas no dan problema: todas las capas de salida
llegan a los metadatos con los mismos nombres de salida y los mismos ejes
dinámicos que en una exportación ONNX normal.

Las filas de segmentación de instancias son la fila de detección seguida de la
máscara de esa instancia, aplanada a `(netH / 4, netW / 4)`, que es la resolución
que el parser de segmentación tiene fijada en el código, como probabilidades para
`segmentation-threshold`.

La clasificación y el gaze corren como inferencia secundaria. Pon `process-mode=2`
y `operate-on-gie-id` en el config generado para colocar un clasificador detrás de
un detector. El gaze es un contrato de solo cabeza, un recorte de cara por entrada,
así que necesita un detector de caras delante.

Faltan tres familias a propósito. `segformer` no está conectada al contrato
compartido de exportación semántica y no puede exportarse a ONNX en ningún
formato. RTMDet-Ins y YOLO9 tienen bloqueada la exportación de segmentación de
instancias dentro de la propia LibreYOLO. `depth_anything3` no tiene
implementación de exportación.

Dos filas de la tabla tienen huecos de checkpoint detrás. Solo está publicado el
checkpoint semántico `l` de EoMT, y la clasificación con DINOv2 no tiene ningún
checkpoint publicado, así que esa combinación necesita pesos ajustados por ti.

## Diferencias de preprocesado

`nvinfer` calcula `net-scale-factor * (x - offsets)` por canal con una escala
escalar, que no puede expresar una desviación estándar por canal. Las familias que
la necesitan (`rfdetr`, `ec`, los tamaños de `deimv2` con backbone DINO, `rtmdet`,
`picodet` y todas las familias de clasificación) llevan la normalización horneada
en el grafo exportado, y el config generado alimenta al grafo con el espacio de
entrada crudo que le corresponde.

La geometría es donde los pipelines de Python de la propia LibreYOLO y `nvinfer`
todavía divergen:

- Las familias con letterbox (`yolo9`, `yolox`, `yolonas`, `rtmdet`, `yolo2`,
  `yolo3`, `yolo4`, `yolo7`) rellenan con gris de forma nativa. `nvinfer` rellena
  con negro.
- La detección con `yolonas` redimensiona de forma nativa el lado más largo a 636
  dentro de su lienzo de 640. El `maintain-aspect-ratio` de `nvinfer` usa los 640
  completos.
- La clasificación redimensiona de forma nativa el lado más corto y luego recorta
  por el centro. `nvinfer` estira el frame o la ROI del objeto hasta la entrada de
  la red, así que los sujetos recortados de cerca salen distintos.
- EoMT ejecuta de forma nativa tiles con ventana deslizante para la segmentación
  semántica. El grafo exportado es un único lienzo estirado, que es más rápido y
  menos preciso.
- `pidnet` emite un mapa de clases a 1/8 de la resolución de entrada y
  `lingbotvision` a 1/16. DeepStream sobremuestrea el mapa de clases para
  mostrarlo.

La comprobación de paridad de ONNX alimenta tensores ya preprocesados, así que
comprueba las salidas del grafo y no puede detectar un orden de color o una
política de padding equivocados en el config. Valida con tus propios datos antes
de desplegar una carga de trabajo que exija paridad exacta.

## Trampas conocidas

### Dos modelos de detección en un mismo directorio cargan el motor del otro

Todos los configs de detección llevan la misma línea:

```ini
model-engine-file=model_b1_gpu0_fp32.engine
```

El constructor de motores del parser exige ese nombre base y no varía según el
modelo. Exporta un segundo modelo de detección al mismo directorio y la segunda
ejecución cargará el motor cacheado del primer modelo. Nada peta; simplemente los
boxes salen mal. Dale a cada modelo de detección su propio directorio. La
ejecución de validación tuvo que aislar D-FINE en uno antes de poder probarlo
siquiera.

### Un box solo puede llevar una clase

El formato de fila de `nvinfer` es `[x1, y1, x2, y2, score, class_id]`, una clase
por box, así que la exportación colapsa las puntuaciones de clase a su argmax. Un
box que `predict` reporta bajo dos clases sobrevive bajo una. Caso medido:
LibreYOLO reporta `vase 0.773` y `bottle 0.383` sobre el mismo box, y el grafo de
DeepStream se queda con `vase`. Esto se deriva del formato de fila del parser y no
se puede cambiar sin salirse de ese contrato, así que es comportamiento esperado y
no una regresión.

## Validado

`deepstream-app` llegó hasta EOS con `App run successful` en los dos tipos de
cabeza de detector, sobre el `sample_1080p_h264.mp4` que trae NVIDIA (1443
frames), con los volcados KITTI por frame activados.

| | YOLO9-s | D-FINE-s |
|---|---|---|
| Tipo de cabeza | grid | uno a uno |
| `cluster-mode` | 2 | 4 |
| `maintain-aspect-ratio` | 1 | 0 |
| Frames con detecciones | 1443 | 1443 |
| Detecciones totales | 18031 | 71105 |

Los histogramas de clase sobre los 1443 frames ponen los coches primero y las
personas segundas en los dos modelos, que es lo correcto para una escena de calle.
La diferencia de cuatro veces en el número de detecciones es la diferencia de
`cluster-mode` haciendo su trabajo: D-FINE con `cluster-mode=4` no ejecuta
clustering, así que sobrevive toda query por encima del umbral, casi duplicadas
incluidas.

Dos modelos entrenados de forma independiente ponen el objeto dominante en el
mismo sitio:

```text
YOLO9  bus  [706.72,  0.82, 1916.34, 1062.97]  conf 0.965
D-FINE bus  [702.73,  2.93, 1916.24, 1069.32]  conf 0.965
```

Esa ejecución establece cinco cosas: TensorRT construye un motor a partir del ONNX
exportado en sm_120, `nvinfer` acepta todas las claves del config generado,
`NvDsInferParseYolo` lee correctamente la disposición del tensor, los boxes caen en
coordenadas 1920x1080 de la resolución de origen, y las etiquetas resuelven contra
el archivo de etiquetas generado.

El entorno en el que se ejecutó:

| Componente | Valor |
|---|---|
| SO anfitrión | Windows 11 Pro 26200 |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB |
| Driver | 591.86 |
| Capacidad de cómputo | 12.0 (Blackwell, sm_120) |
| Runtime de contenedores | Docker Desktop 29.4.3, backend WSL2 |
| Imagen de DeepStream | `nvcr.io/nvidia/deepstream:8.0-samples-multiarch` |
| Versión de DeepStream | 8.0.0 |
| CUDA del contenedor | 12.8.1 |
| Parser | `marcoslucianops/DeepStream-Yolo` en HEAD |

Junto a la ejecución del pipeline, `tests/unit/test_deepstream_export.py` cubre los
adaptadores de grafo y las claves del config generado, y sus 35 tests pasan en este
commit.

## Sin validar

Se indica para que el alcance de arriba no se lea más amplio de lo que es.

- Jetson y aarch64. El contrato de exportación no depende de la arquitectura, pero
  el pipeline solo se ha ejecutado sobre una GPU discreta x86.
- Cuarenta y una de las 43 combinaciones. Solo detección con `yolo9` y detección
  con `dfine` pasaron por DeepStream. La clasificación, la segmentación semántica,
  la segmentación de instancias y las tareas de tensor crudo están cubiertas por
  tests unitarios y comprobaciones de paridad ONNX, no por una ejecución del
  pipeline.
- FP16 e INT8. Solo se ejercitó `network-mode=0`.
- Multi-stream y batching. Una fuente, `batch-size=1`.
- Precisión contra un dataset de ground truth. Las detecciones se comprobaron por
  plausibilidad semántica y por acuerdo entre modelos, no se puntuaron como mAP a
  través de DeepStream.
