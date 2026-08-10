---
title: SAM 3D Body
families:
  - sam3dbody
seo_title: 'SAM 3D Body: recuperación de malla de cuerpo completo en LibreYOLO'
description: >-
  Usa SAM 3D Body en LibreYOLO para recuperar la malla 3D de un cuerpo humano
  completo. Instala y predice; la SAM License de Meta restringe los checkpoints
  y hace falta CUDA.
lead: >-
  SAM 3D Body es el modelo de Meta guiado por prompts para recuperar una malla
  3D de cuerpo completo, manos y pies incluidos, a partir de una sola imagen y
  de cajas de personas. LibreYOLO envuelve el paquete upstream en lugar de
  portarlo.
keywords:
  - SAM 3D Body
  - MHR
  - Momentum Human Rig
  - malla 3d del cuerpo humano
  - reconstrucción 3d de personas
  - human mesh recovery
  - pose 3d python
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import SAMPLE_IMAGE

        from libreyolo.models.sam3dbody import LibreSAM3DBody


        # Esta familia no está registrada en la factoría LibreYOLO(), así

        # que se construye directamente. model_path=None es lo que dispara

        # la descarga restringida de Hugging Face; una cadena, en cambio, se

        # trata como la ruta de un checkpoint local ya existente y nunca se

        # descarga automáticamente.

        # La inferencia requiere un dispositivo CUDA; no hay camino por CPU.

        model = LibreSAM3DBody(None, size="d3", device="cuda")

        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])


        meshes = result.meshes

        print(meshes.vertices.shape)    # (N, V, 3), sistema de la cámara,
        metros

        print(meshes.joints3d.shape)    # (N, J, 3)
    - label: Con un detector de personas
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # Aquí no hay atajo por cadena con nombre: pasa un detector
        # LibreYOLO ya construido, un callable normal o una instancia de
        # PersonDetector.
        detector = LibreYOLO("LibreRFDETRn.pt")
        model = LibreSAM3DBody(None, size="d3", device="cuda")

        result = model(SAMPLE_IMAGE, person_detector=detector)
source_hash: 8edc8d7872f3f875
---

## Instalación

```bash
pip install libreyolo
```

Eso te da solo el adaptador de LibreYOLO. SAM 3D Body en sí no viene incluido,
porque su licencia no es una de la que pueda derivarse el código propio de
LibreYOLO: clona el repositorio upstream e instala sus dependencias por tu
cuenta, y luego apunta LibreYOLO al clon.

```bash
git clone https://github.com/facebookresearch/sam-3d-body
pip install roma einops yacs omegaconf braceexpand pytorch-lightning timm
```

```python
from libreyolo.models.sam3dbody import LibreSAM3DBody

model = LibreSAM3DBody(
    None,
    size="d3",
    sam_3d_body_path="/path/to/sam-3d-body",
    device="cuda",
)
```

o define la variable de entorno `SAM_3D_BODY_PATH` en lugar de pasar
`sam_3d_body_path` en cada llamada. Quien nunca construya esta familia no
dispara nunca el import, y nunca se topa con la SAM License. Esta familia no
está conectada a la factoría `LibreYOLO()` ni al comando de CLI
`libreyolo predict`; `LibreSAM3DBody` es el único punto de entrada.

## Predicción

<code-tabs name="predict" />

La descarga del checkpoint está restringida: exige aceptar la licencia de Meta
en la página del modelo en Hugging Face y autenticarse con `hf auth login` para
que la primera descarga funcione. La inferencia en sí necesita un dispositivo
CUDA sin excepciones: el estimador upstream mueve su batch a la GPU sin
comprobar nada, así que una máquina solo con CPU lanza un error en vez de hacer
fallback. `result.meshes` es un payload `Meshes`, alineado fila a fila con
`result.boxes` (una fila por persona detectada): `vertices` y `joints3d` son
métricos y ya incluyen la traslación de cámara estimada, `joints2d` va en
píxeles sobre la imagen original, y las rotaciones siguen la convención de MHR,
ángulos de Euler en lugar de axis-angle. Consulta
[predicción](/docs/predict) para las fuentes, el streaming y el manejo de
resultados.

## Variantes

Dos backbones detrás del mismo modelo corporal MHR: `d3` usa un encoder DINOv3
ViT-H/16+ y `h` usa el encoder ViT-H original.

## Exportación

<export-matrix />

La exportación de mallas corporales no está implementada: LibreYOLO todavía no
ha definido un contrato de grafo exportado para la tarea de malla, incluido cómo
representar la disposición de parámetros de MHR fuera de PyTorch.

## Checkpoints

Todos los archivos de pesos publicados de esta familia.

<checkpoint-table />

## Licencia

<provenance-box>

El modelo corporal que accionan los checkpoints, MHR (Momentum Human Rig), es
una publicación aparte de Meta bajo Apache-2.0. LibreYOLO descarga su asset de
TorchScript desde la propia release pública de MHR en tiempo de ejecución y lo
cachea en local; ese archivo no lo replica LibreYOLO y se rige por sus propios
términos Apache-2.0, no por la SAM License.

</provenance-box>

## Cita

<citation-block />
