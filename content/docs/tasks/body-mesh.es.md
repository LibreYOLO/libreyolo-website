---
title: Malla corporal
seo_title: Recuperación de malla corporal en LibreYOLO
description: >-
  Recupera una malla corporal 3D paramétrica por persona en LibreYOLO. Predice a
  partir de boxes de personas o de un detector, y lee vértices, articulaciones y
  la traslación de cámara.
lead: >-
  La recuperación de malla corporal convierte una sola imagen y un conjunto de
  boxes de personas en un cuerpo 3D paramétrico por persona: parámetros de forma
  y de pose, vértices posados, articulaciones 3D y la traslación de cámara que
  los sitúa delante del objetivo.
keywords:
  - malla corporal 3d python
  - human mesh recovery python
  - pose 3d cuerpo humano
  - SAM 3D Body
  - MHR
  - modelo corporal paramétrico
  - tarea mesh libreyolo
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import SAMPLE_IMAGE

        from libreyolo.models.sam3dbody import LibreSAM3DBody


        # Esta familia no está registrada en la factory LibreYOLO(), así que

        # se construye directamente. model_path=None dispara la descarga

        # restringida de Hugging Face; una cadena se trata como un checkpoint

        # local ya existente y nunca se descarga. La inferencia requiere CUDA.

        model = LibreSAM3DBody(None, size="d3", device="cuda")

        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])


        meshes = result.meshes

        print(meshes.body_model)      # la parametrización que usan estos
        tensores

        print(meshes.vertices.shape)  # (N, V, 3), frame de cámara, metros

        print(meshes.joints3d.shape)  # (N, J, 3)

        print(meshes.joints2d.shape)  # (N, J, 2), píxeles sobre la imagen
        original
    - label: Con un detector de personas
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        from libreyolo.models.sam3dbody import LibreSAM3DBody


        # person_detector acepta un detector LibreYOLO ya construido, un

        # callable normal o una instancia de PersonDetector. No hay atajo por
        nombre.

        detector = LibreYOLO("LibreYOLO9s.pt")

        model = LibreSAM3DBody(None, size="d3", device="cuda")


        result = model(SAMPLE_IMAGE, person_detector=detector)
source_hash: 31c5b44171cbcd0e
---

## Definición

La recuperación de malla corporal devuelve un payload `Meshes` por imagen,
alineado fila a fila con `result.boxes`: la fila `i` describe a la persona del
box `i`, el mismo contrato que la tarea de pose usa para los keypoints.

Todo se expresa en el frame de cámara de la imagen original. `transl` es métrica,
en metros, con +z apuntando en dirección contraria a la cámara. `vertices` y
`joints3d` son métricos y ya incluyen `transl`, así que no necesitan ninguna
composición adicional. `joints2d` está en píxeles sobre el lienzo de la imagen
original, no sobre el recorte que vio la red. `faces` guarda la topología de la
malla una sola vez para toda la imagen en lugar de por fila, porque todas las
personas la comparten. En esta versión no hay frame de mundo ni de gravedad, y
ningún campo hace de sustituto silencioso de uno.

Los layouts de parámetros difieren entre modelos corporales, así que nada
relativo a las formas es fijo: `body_model` nombra la parametrización y los
recuentos se leen de vuelta desde los tensores. Para `"mhr"`, el Momentum Human
Rig, las rotaciones son ángulos de Euler en radianes en lugar de axis-angle,
`body_pose` es un vector de parámetros plano por articulación en lugar de un
triplete por articulación, y `betas` son coeficientes de blendshape de identidad.
La escala del esqueleto, la pose de las manos y la expresión facial viven en
`extras`.

La clave canónica de la tarea es `mesh`. `body-mesh`, `hmr` y
`human-mesh-recovery` se normalizan a ella.

## Modelos

[SAM 3D Body](/docs/models/sam-3d-body) es la única familia que sirve esta
tarea, y es un wrapper más que un port: el paquete `sam-3d-body` de Meta se
publica bajo la SAM License, de la que el código propio de LibreYOLO no puede
derivar, así que no se incorpora nada de él. Dos backbones comparten el mismo
modelo corporal MHR, `d3` sobre un encoder DINOv3 ViT-H/16+ y `h` sobre el ViT-H
original.

Antes de una primera predicción se aplican tres requisitos, y ninguno de ellos es
opcional.

El paquete upstream lo instalas tú, no LibreYOLO:

```bash
git clone https://github.com/facebookresearch/sam-3d-body
pip install roma einops yacs omegaconf braceexpand pytorch-lightning timm
```

Apunta la biblioteca al clon con `sam_3d_body_path=` o con la variable de entorno
`SAM_3D_BODY_PATH`. Un usuario que nunca construya esta familia nunca dispara la
importación.

El mirror del checkpoint está restringido. Acepta la licencia en la página del
modelo en Hugging Face y autentícate con `hf auth login`, o la primera descarga
fallará. El propio modelo corporal MHR es una publicación Apache-2.0 aparte, que
se descarga desde su propia ubicación pública y se cachea localmente.

La inferencia necesita un dispositivo CUDA. El estimador upstream mueve su batch
a la GPU sin comprobar nada, así que no hay una ruta de CPU a la que recurrir y
`device="cpu"` lanza una excepción.

## Predicción

<code-tabs name="predict" />

Las personas llegan al modelo por una de dos vías. `person_boxes` pasa boxes que
ya tienes, solo para una única imagen: un conjunto fijo de boxes no puede seguir a
las personas a lo largo de los frames de un vídeo, así que pasarlo con una fuente
de vídeo lanza una excepción en lugar de reutilizar en silencio los boxes del
primer frame. `person_detector` acepta un detector LibreYOLO ya construido, un
callable o un `PersonDetector`, y es la vía para vídeo. `focal_length` aporta un
intrínseco de cámara conocido; si se deja sin definir, el modelo usa su propia
estimación, que es lo que reporta `meshes.focal_length`.

Esta familia no está conectada a la factory `LibreYOLO()` ni al comando de CLI
`libreyolo predict`. `LibreSAM3DBody` es el único punto de entrada. Consulta
[predicción](/docs/predict) para fuentes, streaming y manejo de resultados.

## Entrenamiento

Ninguna familia de esta tarea entrena dentro de LibreYOLO.
`LibreSAM3DBody.train()` lanza una excepción: entrena en el proyecto upstream y
carga aquí el checkpoint resultante.

## Validación

No hay validador de mallas, y `val()` lanza una excepción. Los benchmarks
habituales son de licencia solo para investigación, así que no se incluye ninguno
y ninguno puede descargarse por ti.

Las métricas en sí están disponibles como `libreyolo.validation.mesh_metrics`,
para evaluar contra un dataset que ya tengas. Toma articulaciones predichas y
objetivo, opcionalmente vértices predichos y objetivo, y devuelve un diccionario
con exactamente las mismas claves que el de un validador:

`metrics/mpjpe` es el error medio de posición por articulación tras alinear la
articulación raíz, así que puntúa la pose ignorando dónde está situada la persona
en la escena. `metrics/pa_mpjpe` es la misma magnitud tras un alineamiento de
Procrustes completo, rotación, escala uniforme y traslación, que elimina el error
de orientación global y de tamaño corporal y deja la pose articulada.
`metrics/pve` es el error medio por vértice sobre la superficie de la malla tras
alinear en el centroide de vértices; a diferencia de las métricas de
articulaciones, es sensible a la forma del cuerpo, y solo aparece cuando se
aportan ambos arrays de vértices. En las tres, cuanto más bajo mejor. Se asume
que las entradas son métricas, en metros, y `scale_to_mm` convierte los
resultados a los milímetros que reporta la literatura.

## Exportación

La exportación de mallas no está implementada. LibreYOLO no ha definido un
contrato de metadatos de grafo exportado para esta tarea, incluido cómo llevar el
layout de parámetros de MHR fuera de PyTorch, así que `export()` lanza una
excepción en lugar de emitir un grafo cuya salida no podría interpretarse.
