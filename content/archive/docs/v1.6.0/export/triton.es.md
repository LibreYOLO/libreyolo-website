---
title: Triton Inference Server
seo_title: Servir un modelo LibreYOLO en NVIDIA Triton
description: >-
  Sirve una exportación ONNX de LibreYOLO a través de NVIDIA Triton: la
  estructura del model repository, el config.pbtxt generado y cómo predecir
  contra una URL de modelo HTTP.
lead: >-
  Triton Inference Server aloja un model repository y responde peticiones de
  inferencia por HTTP. LibreYOLO exporta el grafo ONNX, genera un config.pbtxt
  que lleva los metadatos de la exportación como un único parámetro de Triton, y
  trata una URL de modelo como una ruta de modelo cargable.
keywords:
  - libreyolo triton
  - triton inference server
  - servir modelo yolo con triton
  - config.pbtxt yolo
  - tritonclient http
  - model repository triton
  - inferencia yolo remota
last_verified: 1.5.0
meta:
  - label: Llamada
    value: 'LibreYOLO("http://127.0.0.1:8000/yolo9")'
    mono: true
  - label: Función auxiliar
    value: >-
      create_triton_config(onnx_path, config_path, model_name=...,
      max_batch_size=8)
    mono: true
  - label: Extra
    value: 'pip install "libreyolo[onnx,triton]"'
    mono: true
  - label: Protocolo
    value: >-
      Solo inferencia V2 por HTTP y HTTPS. Sin gRPC, sin autenticación, sin
      memoria compartida y sin carga ni descarga de modelos.
  - label: Timeouts
    value: Los timeouts de conexión y de red son de 30 segundos por defecto
verification: >-
  Leído de libreyolo/backends/triton.py, libreyolo/models/__init__.py,
  docs/triton.md y pyproject.toml en la rama dev. Los comandos del contenedor
  son los fijados en docs/triton.md.
snippets:
  install:
    - label: Instalación
      language: bash
      code: |
        pip install "libreyolo[onnx,triton]"
  repo:
    - label: Exportar a la estructura del repositorio
      language: python
      code: |
        from pathlib import Path

        from libreyolo import LibreYOLO

        model_dir = Path("triton_repo/yolo9/1")
        model_dir.mkdir(parents=True, exist_ok=True)

        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            output_path=str(model_dir / "model.onnx"),
            dynamic=True,
            simplify=False,
        )
    - label: Generar config.pbtxt
      language: python
      code: |
        from libreyolo import create_triton_config

        create_triton_config(
            "triton_repo/yolo9/1/model.onnx",
            "triton_repo/yolo9/config.pbtxt",
            model_name="yolo9",
            max_batch_size=8,
        )
    - label: Estructura resultante
      language: text
      code: |
        triton_repo/
          yolo9/
            config.pbtxt
            1/
              model.onnx
  serve:
    - label: Arrancar el servidor
      language: bash
      code: |
        docker run --rm --name libreyolo-triton \
          -p 8000:8000 -p 8002:8002 \
          -v "$(pwd)/triton_repo:/models:ro" \
          nvcr.io/nvidia/tritonserver:26.04-py3 \
          tritonserver --model-repository=/models --exit-on-error=true
    - label: Esperar a que esté listo
      language: bash
      code: >
        until curl --fail --silent http://127.0.0.1:8000/v2/health/ready; do
        sleep 1; done
    - label: Pararlo
      language: bash
      code: |
        docker stop libreyolo-triton
  run:
    - label: Predecir contra el modelo servido
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        remote = LibreYOLO("http://127.0.0.1:8000/yolo9")
        result = remote.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Comparar con el modelo local
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        remote = LibreYOLO("http://127.0.0.1:8000/yolo9").predict(SAMPLE_IMAGE)
        native = LibreYOLO("LibreYOLO9t.pt").predict(SAMPLE_IMAGE)

        print(len(remote.boxes), len(native.boxes))
        print(remote.boxes.xyxy[:3])
        print(native.boxes.xyxy[:3])
    - label: 'Fijar una versión, o cambiar el timeout'
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.backends.triton import TritonBackend

        # Un segundo segmento de ruta selecciona la versión del modelo. Sin él,
        # decide la política de versiones configurada en Triton.
        pinned = LibreYOLO("http://127.0.0.1:8000/yolo9/1")

        # Los timeouts de conexión y de red son de 30 segundos por defecto.
        patient = TritonBackend("http://127.0.0.1:8000/yolo9", timeout=120)
source_hash: 0652e4faf0224df3
---

## Instalación

<code-tabs name="install" />

El extra `triton` instala `tritonclient[http]`. Los extras de gRPC y de memoria
compartida se excluyen a propósito: esta integración es solo inferencia V2 por
HTTP y HTTPS. `onnx` hace falta porque tanto el artefacto servido como el
generador del config trabajan a partir de un grafo ONNX.

## Construir el model repository

Exporta con un eje de batch dinámico, dentro de la estructura de directorios que
Triton espera.

<code-tabs name="repo" />

Triton no conserva los metadatos personalizados de ONNX en su respuesta de
configuración del modelo, así que los metadatos exportados completos tienen que
viajar de otra forma. `create_triton_config` los codifica como un único parámetro
de tipo string JSON llamado `libreyolo_metadata` en `config.pbtxt`, emite las
declaraciones de entradas y salidas en el orden del grafo, se encarga del
escapado del JSON y fija el modelo a `KIND_CPU`.

La función auxiliar valida antes de escribir. Exige exactamente una entrada en el
grafo ONNX, al menos una salida, formas de tensor resolubles y unos metadatos
cuyo mapa `names` defina todos los índices de clase desde 0 hasta `nc - 1`. Un
modelo que falle cualquiera de esas comprobaciones se rechaza al generar el
config, no en la primera petición.

`max_batch_size: 8` encaja con una exportación dinámica y permite al servidor
agrupar hasta ocho imágenes por petición. Para un grafo ONNX con batch fijo de 1
usa `max_batch_size=0`; LibreYOLO envía entonces las imágenes de una en una.

## Arrancar el servidor

<code-tabs name="serve" />

Los comandos fijan Triton Server 26.04 y omiten deliberadamente los flags de GPU
de Docker, ya que `KIND_CPU` en el config generado impide de todas formas que el
modelo se coloque en la GPU.

## Ejecutar el artefacto

Una URL de modelo de Triton es una ruta de modelo. `LibreYOLO()` comprueba si hay
un esquema `http` o `https` antes de cualquier manejo de rutas locales y devuelve
un backend que habla con el servidor, así que el punto de llamada es idéntico al
de un checkpoint local, y también lo es el objeto `Results` que se recibe.

<code-tabs name="run" />

La forma de la URL es `http(s)://host:port/model` con un segmento de versión
opcional. El puerto tiene que ser explícito. Se rechazan las credenciales
embebidas, la query string y el fragmento, igual que una ruta con más de dos
segmentos.

`device` se acepta y se ignora con una línea de log, porque la colocación es
decisión del servidor.

## Restricciones

El backend falla con un error directo en lugar de dar un resultado degradado
cuando no se cumple el contrato: faltan los metadatos de LibreYOLO en la
configuración del modelo, hay más de una entrada de modelo, hay un desajuste
entre las salidas configuradas y los metadatos del modelo, hay un tipo de dato de
entrada que no soporta, o el servidor o el modelo no están listos.

Fuera del contrato en esta versión: gRPC, autenticación, memoria compartida y
cargar o descargar modelos a través de la API.

Se puede servir cualquier formato que Triton soporte de por sí, pero el parámetro
de metadatos y el config generado aquí tienen forma de ONNX, así que el camino de
LibreYOLO es [ONNX](/docs/export/onnx) dentro del repositorio. Para un pipeline
de vídeo completo en lugar de un servidor de petición-respuesta, mira
[DeepStream](/docs/export/deepstream).
