---
title: "Mejores alternativas a YOLOv8 en 2026: RF-DETR y YOLO-NAS"
description: "RF100-VL mide cómo se adapta un detector mediante fine-tuning fuera de COCO. RF-DETR y YOLO-NAS superan a YOLOv8. Aprende a ejecutar ambos con LibreYOLO."
date: 2026-08-20
author: Xuban
tags: [LibreYOLO, YOLOv8 alternative, YOLO8 alternative, RF-DETR, YOLO-NAS, object detection, RF100-VL]
faq:
  - q: "¿Cuál es la mejor alternativa a YOLOv8 en 2026?"
    a: "RF-DETR-S para una GPU. Obtuvo 60.41 mAP50-95 en la campaña RF100-VL de LibreYOLO. El artículo de RF100-VL indica 56.9 para YOLOv8m. YOLO-NAS-S es la opción convolucional, con 58.00."
  - q: "¿RF-DETR se transfiere mejor que YOLOv8?"
    a: "En RF100-VL, sí. LibreYOLO midió 60.41 para RF-DETR-S y 61.13 para RF-DETR-M. Roboflow informa de 60.2 y 61.2. El artículo indica 56.9 para YOLOv8m."
  - q: "¿Puedo usar YOLO-NAS con fines comerciales?"
    a: "Los pesos preentrenados de Deci no son para uso comercial. La arquitectura tiene licencia Apache-2.0. Consulta el artículo sobre YOLO-NAS para conocer los detalles de la licencia."
---

COCO es la forma habitual de elegir un detector. Otra perspectiva es medir hasta qué punto el modelo se adapta mediante fine-tuning con otros datos. [RF100-VL](https://arxiv.org/abs/2505.20612) lo mide así: se entrena durante 100 épocas un checkpoint preentrenado con COCO en cada uno de 100 datasets y, después, se evalúa en cada partición de prueba. El resultado es la media de mAP50-95.

YOLOv8m obtiene 56.9, según el [apéndice del artículo](https://papers.neurips.cc/paper_files/paper/2025/file/1013f8ff40a194f3f12a6bcc5221bb34-Paper-Datasets_and_Benchmarks_Track.pdf). Dos modelos que hemos ejecutado obtienen una puntuación superior: RF-DETR y YOLO-NAS. Los resultados sin procesar están en [huggingface.co/datasets/LibreYOLO/rf100-vl-results](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results).

## RF-DETR

RF-DETR-S obtiene 60.41 y RF-DETR-M, 61.13 en la [campaña de LibreYOLO](/articles/rf100vl-benchmark). [Roboflow informa](https://rfdetr.roboflow.com/latest/learn/benchmarks/) de 60.2 y 61.2. Las versiones Nano a Large tienen licencia Apache-2.0.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=512, batch=8)
```

[Documentación](/docs/models/rf-detr) | [Pesos](https://huggingface.co/LibreYOLO/LibreRFDETRs) | [Ejecución `20260814-rfdetr-s-1b1190ee`](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main/rfdetr-s/20260814-rfdetr-s-1b1190ee)

## YOLO-NAS

YOLO-NAS-S obtiene 58.00 y es la opción convolucional si buscas una CNN de estilo YOLO. Los pesos publicados por Deci no son para uso comercial. LibreYOLO no aloja ninguno de ellos. Más información: [YOLO-NAS sigue en mantenimiento](/articles/yolo-nas-with-libreyolo).

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
```

[Documentación](/docs/models/yolo-nas) | [Ejecución `20260809-yolonas-s-02926964`](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main/yolonas-s/20260809-yolonas-s-02926964)

Instala RF-DETR con `pip install "libreyolo[rfdetr]"` o YOLO-NAS con `pip install libreyolo`. Página de la campaña: [RF100-VL](/articles/rf100vl-benchmark).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentación](https://www.libreyolo.com/docs)
