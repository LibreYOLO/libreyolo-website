---
title: "Le migliori alternative a YOLOv8 nel 2026: confronto RF-DETR e YOLO-NAS"
description: "RF100-VL misura quanto bene un detector si adatta con il fine-tuning fuori da COCO. RF-DETR e YOLO-NAS ottengono risultati superiori a YOLOv8. Ecco come eseguirli con LibreYOLO."
date: 2026-08-20
author: Xuban
tags: [LibreYOLO, YOLOv8 alternative, YOLO8 alternative, RF-DETR, YOLO-NAS, object detection, RF100-VL]
faq:
  - q: "Qual è la migliore alternativa a YOLOv8 nel 2026?"
    a: "RF-DETR-S per una GPU. Ha ottenuto 60.41 mAP50-95 nella campagna RF100-VL di LibreYOLO. Il paper RF100-VL riporta 56.9 per YOLOv8m. YOLO-NAS-S è l'opzione convoluzionale, con 58.00."
  - q: "RF-DETR trasferisce meglio di YOLOv8?"
    a: "Su RF100-VL, sì. LibreYOLO ha misurato 60.41 per RF-DETR-S e 61.13 per RF-DETR-M. Roboflow riporta 60.2 e 61.2. Il paper riporta 56.9 per YOLOv8m."
  - q: "Posso usare YOLO-NAS a fini commerciali?"
    a: "I pesi preaddestrati di Deci non sono per uso commerciale. L'architettura è distribuita con licenza Apache-2.0. Consulta l'articolo su YOLO-NAS per i dettagli sulla licenza."
---

COCO è il riferimento abituale per scegliere un detector. Un altro criterio è quanto bene il modello fa fine-tuning su altri dati. [RF100-VL](https://arxiv.org/abs/2505.20612) lo misura così: un checkpoint preaddestrato su COCO viene addestrato per 100 epoche su ciascuno di 100 dataset, poi valutato su ogni split di test. Il valore è la mAP50-95 media.

YOLOv8m ottiene 56.9, secondo l'[appendice del paper](https://papers.neurips.cc/paper_files/paper/2025/file/1013f8ff40a194f3f12a6bcc5221bb34-Paper-Datasets_and_Benchmarks_Track.pdf). Due modelli che abbiamo eseguito ottengono risultati superiori: RF-DETR e YOLO-NAS. I risultati grezzi sono disponibili su [huggingface.co/datasets/LibreYOLO/rf100-vl-results](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results).

## RF-DETR

RF-DETR-S ottiene 60.41 e RF-DETR-M 61.13 nella [campagna di LibreYOLO](/articles/rf100vl-benchmark). [Roboflow riporta](https://rfdetr.roboflow.com/latest/learn/benchmarks/) 60.2 e 61.2. Le varianti da Nano a Large sono distribuite con licenza Apache-2.0.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=512, batch=8)
```

[Documentazione](/docs/models/rf-detr) | [Pesi](https://huggingface.co/LibreYOLO/LibreRFDETRs) | [Run `20260814-rfdetr-s-1b1190ee`](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main/rfdetr-s/20260814-rfdetr-s-1b1190ee)

## YOLO-NAS

YOLO-NAS-S ottiene 58.00 ed è l'opzione convoluzionale se vuoi una CNN in stile YOLO. I pesi pubblicati da Deci non sono per uso commerciale. LibreYOLO non ne ospita. Dettagli: [YOLO-NAS è ancora mantenuto](/articles/yolo-nas-with-libreyolo).

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
```

[Documentazione](/docs/models/yolo-nas) | [Run `20260809-yolonas-s-02926964`](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main/yolonas-s/20260809-yolonas-s-02926964)

Installa RF-DETR con `pip install "libreyolo[rfdetr]"` oppure YOLO-NAS con `pip install libreyolo`. Pagina della campagna: [RF100-VL](/articles/rf100vl-benchmark).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentazione](https://www.libreyolo.com/docs)
