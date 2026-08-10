---
title: NVIDIA Jetson
seo_title: "Instalar LibreYOLO y PyTorch en NVIDIA Jetson"
description: "Instala LibreYOLO en una NVIDIA Jetson: las cuatro bibliotecas CUDA que JetPack no incluye, el paso --no-deps que necesita PyTorch y cifras medidas en una Orin Nano."
lead: "Las placas NVIDIA Jetson ejecutan LibreYOLO con los wheels estándar de PyTorch para aarch64. No interviene ninguna compilación de torch específica para Jetson, pero JetPack omite cuatro bibliotecas contra las que torch enlaza, y la instalación tiene que aportarlas."
keywords:
  - NVIDIA Jetson
  - Jetson Orin Nano
  - JetPack 7.2
  - instalar pytorch en jetson
  - nvidia-cudnn-cu13
  - nvidia-nccl-cu13
  - nvidia-cusparselt-cu13
  - nvidia-nvshmem-cu13
  - torch.cuda.is_available
  - no kernel image is available for execution on the device
  - tensorrt en jetson
  - wheels aarch64
last_verified: "1.4.0"
meta:
  - label: Placa
    value: "Jetson Orin Nano Super Developer Kit, 8 GB, capacidad de cómputo de GPU 8.7"
  - label: Plataforma
    value: "JetPack 7.2 (L4T R39.2), Ubuntu 24.04, CUDA 13, Python 3.12.3, aarch64"
  - label: Stack probado
    value: "libreyolo 1.4.0, torch 2.13.0+cu130, torchvision 0.28.0+cu130, opencv 5.0.0, numpy 2.5.1, el 2026-07-27"
  - label: Ausentes en JetPack
    value: "nvidia-cudnn-cu13, nvidia-nccl-cu13, nvidia-cusparselt-cu13, nvidia-nvshmem-cu13"
    mono: true
  - label: Benchmarks
    value: "223 ejecuciones verificadas en esta placa, 58 modelos de 12 familias, en PyTorch, ONNX Runtime y TensorRT"
    links:
      - label: visionanalysis.org/hardware/jetson_orin
        href: https://www.visionanalysis.org/hardware/jetson_orin
  - label: Seguimiento en
    value: "La mitad Jetson de la issue 648"
    links:
      - label: issue 648
        href: https://github.com/LibreYOLO/libreyolo/issues/648
verification: "Receta de instalación y salida esperada tomadas de la instalación del 2026-07-27 en una Jetson Orin Nano Super. Las filas de latencia y precisión vienen del snapshot de resultados verificados que hay detrás de visionanalysis.org, filtrado por el hardware jetson_orin, medido en junio de 2026 sobre libreyolo 1.2.0.dev0. Comportamiento de exportación y del cargador leído de libreyolo/export/exporter.py, libreyolo/export/tensorrt.py y libreyolo/models/__init__.py."
snippets:
  prep:
    - label: Paquetes del sistema y un entorno virtual
      language: bash
      code: |
        # JetPack no trae preinstalados ni pip ni el módulo venv.
        sudo apt update
        sudo apt install -y python3.12-venv python3-pip

        python3 -m venv ~/libreyolo
        source ~/libreyolo/bin/activate
        pip install -U pip wheel setuptools
  torch:
    - label: PyTorch, desde el índice de wheels de CUDA 13
      language: bash
      code: |
        pip install torch torchvision \
          --index-url https://download.pytorch.org/whl/cu130 \
          --extra-index-url https://pypi.org/simple
    - label: Las cuatro bibliotecas que JetPack no incluye
      language: bash
      code: |
        pip install nvidia-cudnn-cu13 nvidia-nccl-cu13 \
                    nvidia-cusparselt-cu13 nvidia-nvshmem-cu13
    - label: Si pip exige cuda-toolkit 13.0.3, instala con --no-deps
      language: bash
      code: |
        # --no-deps implica nombrar también a mano las dependencias Python de torch.
        pip install --no-deps \
          torch torchvision \
          nvidia-cudnn-cu13 nvidia-nccl-cu13 \
          nvidia-cusparselt-cu13 nvidia-nvshmem-cu13 \
          filelock typing_extensions sympy networkx jinja2 markupsafe mpmath \
          fsspec numpy pillow
  ldd:
    - label: Identifica la siguiente biblioteca que falta en vez de adivinar
      language: bash
      code: |
        ldd "$VIRTUAL_ENV/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so" \
          | grep "not found"

        # Todo lo que sigue faltando en todas las bibliotecas de torch, de una pasada:
        ldd "$VIRTUAL_ENV"/lib/python3.12/site-packages/torch/lib/*.so 2>/dev/null \
          | grep "not found" | sort -u
  install:
    - label: Instala LibreYOLO después de torch, no antes
      language: bash
      code: |
        # torch ya está satisfecho, así que pip deja intacta la compilación CUDA.
        pip install libreyolo

        # El extra de ONNX solo hace falta para exportar. Una exportación a TensorRT
        # pasa por ONNX, así que instálalo antes de la sección de exportación de abajo.
        pip install "libreyolo[onnx]"
  verify:
    - label: Versiones y dispositivo
      language: python
      code: |
        import cv2
        import numpy
        import torch

        import libreyolo

        print("torch", torch.__version__, "cuda", torch.cuda.is_available())
        print("gpu", torch.cuda.get_device_name(0))
        print("libreyolo", libreyolo.__version__)
        print("cv2", cv2.__version__, "numpy", numpy.__version__)
      expect: |
        torch 2.13.0+cu130 cuda True
        gpu Orin
        libreyolo 1.4.0
        cv2 5.0.0 numpy 2.5.1
    - label: Después, ejecuta un kernel real
      language: python
      code: |
        import torch

        x = torch.rand(2000, 2000, device="cuda")
        print(float((x @ x).sum()))
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO9, SAMPLE_IMAGE

        # Descarga el checkpoint en el primer uso.
        model = LibreYOLO9("libreyolo9s.pt", size="s")

        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes)
    - label: CLI
      language: bash
      code: |
        libreyolo predict --source https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg --model libreyolo9s.pt --save
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, LibreYOLO9, SAMPLE_IMAGE

        # Escribe libreyolo9s.onnx y luego construye libreyolo9s.engine a partir de él.
        LibreYOLO9("libreyolo9s.pt", size="s").export(format="tensorrt", half=True)

        # El engine se recarga por el mismo punto de entrada.
        result = LibreYOLO("libreyolo9s.engine").predict(SAMPLE_IMAGE)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model libreyolo9s.pt --format tensorrt --half
  power:
    - label: Modo de potencia y frecuencias
      language: bash
      code: |
        sudo nvpmodel -q      # qué modos expone esta placa, y cuál está activo
        sudo nvpmodel -m 0    # el modo más alto en la placa probada aquí
        sudo jetson_clocks

        tegrastats            # carga en vivo; nvidia-smi es limitado en Tegra
---

## Qué documenta esta página

Esta página documenta una configuración que se verificó de principio a fin, no
una matriz de compatibilidad. La placa fue una Jetson Orin Nano Super Developer
Kit con 8 GB de memoria ejecutando JetPack 7.2 (L4T R39.2, Ubuntu 24.04, CUDA
13, Python 3.12.3), y el stack que levantó en ella fue `libreyolo 1.4.0` con
`torch 2.13.0+cu130`, OpenCV 5.0.0 y NumPy 2.5.1. `torch.cuda.is_available()`
devolvió `True` y la GPU se identificó como `Orin`.

Otras versiones de JetPack, otras placas Jetson y otras versiones de CUDA no se
probaron. La receta de abajo es la que funcionó en esa combinación.

Esa ejecución fue el 2026-07-27 contra LibreYOLO 1.4.0, y no se ha repetido en
hardware con 1.5.0: esta es la única página del árbol de 1.5.0 que todavía lleva
una verificación de 1.4.0, y por eso su front matter dice
`last_verified: "1.4.0"`. Nada de lo que cambia en 1.5.0 toca la ruta de
instalación, las cuatro bibliotecas ausentes ni los flags de exportación
descritos aquí, así que se espera que los comandos sigan siendo válidos, pero los
números de versión de las salidas de abajo son los que imprimió 1.4.0, no una
medición de 1.5.0.

Hay dos cosas en todo esto que van en contra de lo que dicen la mayoría de las
guías de Jetson. Los wheels son las compilaciones aarch64 corrientes publicadas
para CUDA 13, así que no hace falta ninguna compilación de torch específica para
Jetson. Y JetPack no incluye cuatro bibliotecas contra las que enlazan esos
wheels, así que `import torch` falla de una biblioteca en una hasta que están
instaladas las cuatro.

## Instalación

Las imágenes de JetPack llegan sin pip y sin el módulo `venv`, así que ambos van
primero.

<code-tabs name="prep" />

Una placa de 8 GB va justa con los checkpoints más grandes. Añadir swap en el
NVMe antes de cargarlos evita que el sistema mate el proceso por falta de
memoria a mitad de ejecución.

Después, PyTorch. El índice de CUDA 13 lleva los wheels aarch64; el índice extra
aporta las dependencias de Python puro desde PyPI.

<code-tabs name="torch" />

Los cuatro wheels `nvidia-*-cu13` son la parte que es fácil pasar por alto.
JetPack proporciona el driver de la GPU, no cuDNN, NCCL, cuSPARSELt ni NVSHMEM, y
torch se niega a importarse sin ellos. Instalar los cuatro de una vez es más
rápido que ir descubriéndolos una excepción cada vez.

El tercer snippet cubre un fallo concreto: los metadatos de dependencias de torch
para la compilación de CUDA 13 piden `cuda-toolkit==13.0.3`, que no tiene wheel
aarch64 en PyPI, así que la resolución falla antes de descargar nada. `--no-deps`
se salta el resolutor, lo que significa que hay que nombrar cada dependencia en
la línea de comandos.

LibreYOLO se instala en último lugar. Instalarlo primero deja que pip elija su
propio torch, que en esta plataforma no es la compilación CUDA.

<code-tabs name="install" />

Todas las dependencias restantes se resuelven a un wheel aarch64 precompilado,
incluidos OpenCV, NumPy, SciPy, pycocotools y safetensors. No se compila nada
desde el código fuente.

## Comprobar que CUDA funciona

<code-tabs name="verify" />

El segundo snippet importa tanto como el primero. Un wheel compilado para la
arquitectura de GPU equivocada sigue informando de
`torch.cuda.is_available() == True` y luego falla en la primera operación real
con `CUDA error: no kernel image is available for execution on the device`. Una
multiplicación de matrices en el dispositivo es la comprobación que lo detecta.

## Ejecutar una predicción

<code-tabs name="predict" />

`predict` devuelve el mismo objeto `Results` que en cualquier otra plataforma,
así que las páginas de los modelos se aplican sin cambios.

## Exportar a TensorRT

En esta placa, TensorRT fue más rápido que PyTorch y que ONNX Runtime en los 55
modelos que se midieron en todos los runtimes.

<code-tabs name="export" />

`format="tensorrt"` escribe primero un grafo ONNX y construye el engine a partir
de él, así que el extra `onnx` tiene que estar instalado. `LibreYOLO()` despacha
según el sufijo del archivo, así que un archivo `.engine` se carga con la misma
llamada que un checkpoint `.pt`.

No uses el extra de pip `tensorrt` en una Jetson. Fija `tensorrt-cu12`, una
compilación para CUDA 12, contra una plataforma CUDA 13. Usa en su lugar el
TensorRT que instala JetPack. Si `import tensorrt` falla dentro del entorno
virtual mientras funciona fuera, recrea el entorno con `--system-site-packages`
para que el módulo del sistema sea visible.

Los engines de TensorRT serializados están atados al dispositivo, a la
arquitectura de GPU y a la versión de TensorRT que los construyó. Un engine
construido en una estación de trabajo no cargará en una Jetson, así que el paso
de construcción se ejecuta en la propia placa.

## Medido en esta placa

Latencia por imagen, tamaño de batch 1, de principio a fin incluyendo
preprocesado y postprocesado, sobre COCO val2017 (subconjunto de 500 imágenes)
con `conf=0.001` y `max_det=300`. Cinco modelos de los 58 medidos:

| Modelo | Entrada (px) | PyTorch FP32 (ms) | ONNX FP32 (ms) | TensorRT FP32 (ms) | TensorRT FP16 (ms) | mAP 50-95 |
|---|---:|---:|---:|---:|---:|---:|
| DEIMv2-Atto | 320 | 64.9 | 22.8 | 12.3 | 11.2 | 27.49 |
| YOLOX-Tiny | 416 | 49.2 | 31.8 | 23.0 | 19.4 | 35.45 |
| YOLO9-t | 640 | 101.2 | 53.8 | 36.0 | 29.1 | 41.78 |
| RT-DETR-r18 | 640 | 98.3 | 103.7 | 45.3 | 25.7 | 49.72 |
| D-FINE-s | 640 | 96.8 | 96.1 | 44.7 | 33.1 | 53.45 |

La columna de mAP es la puntuación de la propia ejecución en TensorRT FP16. En
los 55 modelos medidos en los cuatro runtimes, la mayor diferencia entre la
puntuación de PyTorch FP32 y la de TensorRT FP16 fue de 0.59 puntos, en
DEIMv2-X. Los runtimes se diferencian en velocidad, no en precisión.

TensorRT FP32 fue más rápido que PyTorch y que ONNX Runtime en los 55 modelos.
TensorRT FP16 también fue más rápido que PyTorch FP32 en los 55, entre 1.68x y
6.22x, con una mediana de 3.39x. ONNX Runtime es el que varía: fue más lento que
PyTorch en 23 de los 55, la fila de RT-DETR-r18 entre ellos.

Condiciones detrás de cada número: `libreyolo 1.2.0.dev0`, `torch 2.12.0+cu130`,
Python 3.12.3, CUDA 13, driver 595.78, ONNX Runtime 1.24.0, medido en junio de
2026. La latencia en una Jetson depende además del modo de potencia activo, que
los registros del benchmark no recogen.

<code-tabs name="power" />

Las 223 ejecuciones, incluidos los otros 53 modelos y las columnas completas de
precisión, están publicadas en
[la página de la Jetson Orin en Vision Analysis](https://www.visionanalysis.org/hardware/jetson_orin).

## Resolución de problemas

### import torch falla nombrando una biblioteca compartida

Falta una de las cuatro bibliotecas de arriba. En vez de adivinar cuál, léela
directamente del binario:

<code-tabs name="ldd" />

Cada entrada que falta se corresponde con un wheel:

| Biblioteca ausente | Wheel |
|---|---|
| cuDNN | `nvidia-cudnn-cu13` |
| NCCL | `nvidia-nccl-cu13` |
| cuSPARSELt | `nvidia-cusparselt-cu13` |
| NVSHMEM | `nvidia-nvshmem-cu13` |

### torch avisa de que ninguna compilación soporta esta GPU

La primera llamada a CUDA en la configuración que funciona imprime esto:

```text
UserWarning: Found GPU0 Orin which is of compute capability (CC) 8.7.
The following list shows the CCs this version of PyTorch was built for and the hardware CCs it supports:
- 8.0 which supports hardware CC >=8.0,<9.0 except {8.7}
- 9.0 which supports hardware CC >=9.0,<10.0
- 10.0 which supports hardware CC >=10.0,<11.0 except {10.1}
- 11.0 which supports hardware CC >=11.0,<12.0
- 12.0 which supports hardware CC >=12.0,<13.0
No published PyTorch CUDA builds for release 2.13.0+cu130 support this GPU.
```

El aviso es cosmético en esta placa. El wheel lleva kernels `sm_80` y la Orin los
ejecuta. El mismo aviso apareció con el wheel anterior de ese índice, el que
produjo todas las filas de benchmark de arriba. Confírmalo con la multiplicación
de matrices de la comprobación de CUDA en lugar de fiarte o desconfiar del
mensaje.

### CUDA error: no kernel image is available for execution on the device

El wheel instalado se compiló para otra arquitectura de GPU. Es lo que ocurre con
los wheels del índice `sbsa` de NVIDIA, que apuntan a GPUs ARM de servidor y no
al silicio de Jetson. Reinstala desde el índice de CUDA 13 de la sección de
instalación.

### pip no encuentra cuda-toolkit 13.0.3

No existe un wheel aarch64 para él. Usa la forma con `--no-deps` de la sección de
instalación y nombra explícitamente las dependencias de torch.

### libnvpl_lapack_lp64_gomp.so.0: cannot open shared object file

El wheel aarch64 de torch enlaza con las NVIDIA Performance Libraries para el
cálculo en CPU. Instálalas y ponlas en la ruta de bibliotecas:

```bash
pip install nvpl-lapack nvpl-blas --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130/
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvpl/lib:$LD_LIBRARY_PATH"
```

Ese índice está bien para estas dos bibliotecas de CPU. Sus compilaciones de
torch son las que producen el fallo de "no kernel image" de arriba.

### Fuentes de wheels que no encajan con JetPack 7.2

| Fuente | Resultado en la Orin Nano Super |
|---|---|
| torch de `pypi.jetson-ai-lab.io/sbsa/cu130` | Compilado para GPUs ARM de servidor. Importa, informa de que CUDA está disponible y luego falla con "no kernel image is available for execution on the device". |
| torch de `pypi.jetson-ai-lab.io/jp6/*` | Compilaciones para CUDA 12 y Python 3.10. No se instalan en el Python 3.12 de esta imagen. |
| Contenedores de PyTorch de JetPack 6 | La inicialización de CUDA falla con el error 801 en un host con JetPack 7. |
| Compilar torch desde el código fuente | Funciona, pero lleva horas en una placa de 8 GB y es innecesario una vez instalados los wheels de CUDA 13. |

## DeepStream

Para un pipeline de vídeo completo en lugar de un bucle en Python, exporta con
`deepstream=True` y ejecuta el grafo a través de `nvinfer`. Esa ruta tiene su
propia página, que incluye el config de `nvinfer` generado, la compilación del
parser de bounding boxes y las trampas conocidas:
[DeepStream](/docs/export/deepstream).

El pipeline de DeepStream en sí se validó en una GPU discreta x86, no en una
Jetson. El contrato de exportación no depende de la arquitectura, pero la
ejecución del pipeline en aarch64 sigue pendiente.

## Sin verificar

- Versiones de JetPack distintas de la 7.2, y versiones de L4T distintas de la
  R39.2.
- Placas Jetson que no sean la Orin Nano Super de 8 GB.
- El entrenamiento en la placa. Se ejercitaron la inferencia y la exportación; no
  se hizo ninguna ejecución de entrenamiento.
- Los engines INT8. Para esta placa solo existen filas de FP32 y FP16.
- Tamaños de batch por encima de 1. Todas las mediciones de arriba son con batch 1.
