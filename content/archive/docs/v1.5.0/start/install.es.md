---
title: Instalación
seo_title: Instalar LibreYOLO
description: >-
  Instala LibreYOLO desde PyPI, elige los extras opcionales que necesita una
  familia de modelos o un destino de exportación, y confirma que PyTorch ve tu
  GPU.
lead: >-
  LibreYOLO se publica en PyPI como libreyolo. El paquete base cubre la
  predicción, el entrenamiento, la validación y las familias de modelos que no
  necesitan nada más allá de PyTorch; los extras opcionales añaden el resto.
keywords:
  - instalar libreyolo
  - pip install libreyolo
  - libreyolo extras
  - libreyolo cuda
  - libreyolo gpu
  - requisitos libreyolo
last_verified: 1.5.0
meta:
  - label: Paquete
    value: libreyolo
    mono: true
  - label: Python
    value: 3.10 o superior
  - label: Licencia del código
    value: MIT
  - label: Dependencia principal
    value: PyTorch 2.4 o superior
snippets:
  install:
    - label: pip
      language: bash
      code: |
        pip install libreyolo
    - label: Con extras
      language: bash
      code: |
        # Sepáralos con comas para combinar varios en una sola instalación.
        pip install "libreyolo[rfdetr,onnx]"
    - label: Todo
      language: bash
      code: |
        pip install "libreyolo[all]"
    - label: Desde el código fuente
      language: bash
      code: |
        git clone https://github.com/LibreYOLO/libreyolo.git
        cd libreyolo
        pip install -e .
  verify:
    - label: CLI
      language: bash
      code: |
        # Python, Torch, CUDA, cuDNN, cada GPU visible y qué
        # paquetes opcionales están instalados.
        libreyolo checks
    - label: Python
      language: python
      code: |
        import libreyolo

        print(libreyolo.__version__)
    - label: Inventario de modelos
      language: bash
      code: |
        # Cada familia registrada con sus tareas, tamaños y resoluciones
        # de entrada. Las familias cuyo extra falta se listan con el
        # comando pip que las habilita.
        libreyolo models
source_hash: 34fc6d3e24d03fb4
---

## Instalación

<code-tabs name="install" />

Se requiere Python 3.10 o superior. La instalación base instala PyTorch,
torchvision, NumPy, Pillow, OpenCV, PyYAML, requests, mss, tqdm, pycocotools,
typer, click, safetensors y SciPy, de modo que YOLOv9 y las demás familias que
no necesitan nada más funcionan directamente tras `pip install libreyolo`.

Un clon deja activa la rama `release`, la rama estable cuyo código coincide con
esta documentación. La rama de integración, que contiene trabajo aún no
publicado, es `dev`.

## Extras opcionales

Un extra es un nombre entre corchetes que añade las dependencias que necesita
una familia de modelos o un destino de exportación. Nada más cambia: la API es
la misma con o sin el extra.

### Familias de modelos

| Extra | Añade |
|---|---|
| `rfdetr` | `transformers`, que proporciona el backbone de RF-DETR |
| `eomt` | `transformers` |
| `midas` | `timm` 1.0.x, que proporciona los encoders ViT-L/16 y EfficientNet-Lite3 de MiDaS |
| `vlm` | `transformers`, `num2words`, `decord`, `lmdb`, `peft` |
| `sam` | `transformers`, `timm` |
| `openvocab` | `transformers`, `timm`, `regex`, `ftfy` |
| `sensenova` | `transformers`, `accelerate` y `bitsandbytes` fuera de macOS |
| `modus` | `transformers`, `accelerate` |
| `clip` | `regex` y `ftfy`, necesarios para el tokenizador de texto de CLIP incluido en la librería |
| `siglip2` | `sentencepiece`, necesario para el tokenizador multilingüe de SigLIP 2 |
| `gaze` | `gdown`, que activa la descarga automática del checkpoint de L2CS |
| `rtdetr` | Nada. RT-DETR no necesita ninguna dependencia extra; el nombre se mantiene por estabilidad |

### Exportación y runtimes

| Extra | Añade |
|---|---|
| `onnx` | `onnx`, `onnxsim`, `onnxruntime` |
| `tensorrt` | `tensorrt-cu12` 10.16.1.11 y `pycuda`, fuera de macOS |
| `openvino` | `openvino` |
| `coreml` | `coremltools` |
| `coreai` | `coreai-torch`, solo macOS |
| `tflite`, alias `litert` | `libreyolo[onnx]` más `onnx2tf`, `ai-edge-litert`, `onnx-graphsurgeon` y `onnx-simplifier` |
| `mnn` | `libreyolo[onnx]` más `MNN` |
| `ncnn` | `pnnx` y `ncnn` |
| `paddle` | `libreyolo[onnx]` más `paddlepaddle` 2.6.2 y `x2paddle` 1.6.0 |
| `executorch` | `executorch` |
| `triton` | `tritonclient[http]` para inferencia V2 por HTTP y HTTPS |

### Entrenamiento, evaluación y logging

| Extra | Añade |
|---|---|
| `lora` | `libreyolo[rfdetr]` más `peft`, para hacer fine-tuning con `lora=True` |
| `plots` | `matplotlib` |
| `fast-eval` | `faster-coco-eval`, el backend de evaluación COCO en C++ |
| `tensorboard` | `tensorboard` |
| `mlflow` | `mlflow` |
| `wandb` | `wandb` |
| `comet` | `comet-ml` |
| `clearml` | `clearml` |
| `neptune` | `neptune-scale` |
| `dvclive`, alias `dvc` | `dvclive` |

`fast-eval` es opcional en lugar de una dependencia obligatoria para que una
plataforma sin wheel precompilada no pueda romper una instalación básica. Cuando
el paquete no está presente, la evaluación COCO recurre a pycocotools y la
ejecución continúa.

### Herramientas

| Extra | Añade |
|---|---|
| `stream` | `yt-dlp`, necesario solo para resolver URLs de páginas de YouTube |
| `tracking` | Nada. Todas las dependencias de tracking ya son dependencias del núcleo |
| `label` | `libreyolo[sam]`, que habilita la asistencia de clic a máscara en `libreyolo label` |
| `hub-kernels` | `kernels`, el cargador opcional de kernels compilados del Hub. Consulta [kernels](/docs/reference/kernels), donde se indica que instalarlo puede desplazar las predicciones de RF-DETR a tolerancia de float |
| `clip-convert` | `libreyolo[clip]` más `open_clip_torch`, para la conversión de pesos y las comprobaciones de paridad |
| `siglip2-convert` | `libreyolo[siglip2]` más `transformers`, por el mismo motivo |

Las webcams, RTSP, RTMP, TCP, UDP, HLS y las listas locales multi-stream no
necesitan ningún extra. Solo las URLs de páginas de YouTube lo necesitan.

### El extra agregado

`libreyolo[all]` instala los extras de modelos, exportación, tracking y logging
en un solo comando. Algunos quedan deliberadamente fuera. `neptune` se excluye
porque la versión estable de `neptune-scale` requiere protobuf inferior a 7
mientras que la ruta de TFLite requiere protobuf 7. `executorch` se excluye
porque ExecuTorch restringe con qué versión de PyTorch se empareja, y `coreai`
porque `coreai-torch` fija PyTorch en 2.11.x y arrastraría todo el entorno a esa
versión. `fast-eval`, `hub-kernels`, `clip-convert` y `siglip2-convert` también
quedan fuera. Instala cualquiera de ellos por su nombre.

## Restricciones de plataforma

Tres extras están acotados por plataforma mediante sus marcadores de
dependencia, de modo que la instalación tiene éxito en todas partes y
simplemente instala menos donde no existe una wheel.

| Extra | Restricción |
|---|---|
| `coreai` | Solo macOS. La toolchain de Core AI ni convierte ni ejecuta en otro sistema |
| `tensorrt` | Se omite en macOS, que no tiene CUDA |
| `tflite`, `litert` | `onnx2tf` y `ai-edge-litert` requieren Python 3.12 o superior |

`sensenova` omite `bitsandbytes` en macOS, donde no se publica ninguna wheel; el
resto del extra se instala con normalidad.

Si el disco es la limitación, la mayor parte es PyTorch, y la mayor parte de
PyTorch es la carga de CUDA que su wheel por defecto incluye. Una wheel solo de
CPU la elimina sin renunciar a nada. Para detección con ONNX en una máquina que
no debe llevar torch en absoluto, consulta la
[instalación ligera](/docs/lightweight-install).

## GPU y CUDA

La selección de dispositivo ocurre al construir el modelo. El valor por defecto,
`device="auto"`, usa CUDA cuando `torch.cuda.is_available()` es verdadero,
después Metal Performance Shaders cuando `torch.backends.mps.is_available()` es
verdadero, y CPU en caso contrario. Nada más en la librería inspecciona el
hardware, así que si PyTorch no puede ver una GPU, LibreYOLO tampoco.

Para fijar el dispositivo, pasa `device` al modelo o a `predict`, `train`,
`val` y `export`. Acepta `"cpu"`, `"cuda"`, `"cuda:0"`, `"mps"`, un entero
simple como `0`, o una cadena de dígitos como `"0"`; los dos últimos se expanden
a `cuda:<n>`.

Empieza con `libreyolo checks`, que imprime la versión de Torch, las versiones
de CUDA y cuDNN con las que se compiló Torch, y cada GPU visible con su memoria.
Cuando informa de que no hay CUDA en una máquina que tiene una tarjeta NVIDIA,
la wheel de PyTorch que pip resolvió es una build de CPU. Instala primero una
build con CUDA desde el índice de PyTorch y después instala LibreYOLO:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install libreyolo
```

Ese es el mismo índice que el repositorio fija para su propio entorno gestionado
con uv en Linux y Windows. Necesita el driver de NVIDIA 555 o superior, que es
el requisito del runtime de CUDA 12.8. macOS mantiene la wheel de PyPI, ya que
el host de descargas de PyTorch no publica builds para Darwin.

## Comprueba la instalación

<code-tabs name="verify" />

`libreyolo models` es la forma más rápida de ver si un extra surtió efecto: una
familia cuya dependencia falta se imprime con el comando pip exacto que la
habilita. Ambos comandos aceptan también `--json`, que imprime los mismos datos
como un objeto legible por máquina en stdout.
