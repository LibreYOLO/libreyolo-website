---
title: "Alternatives à YOLOv8 en 2026 : RF-DETR et YOLO-NAS"
description: "RF100-VL mesure la capacité d'un détecteur à s'adapter à de nouveaux datasets par fine-tuning après un pré-entraînement sur COCO. RF-DETR et YOLO-NAS obtiennent de meilleurs scores que YOLOv8. Découvrez comment lancer les deux avec LibreYOLO."
date: 2026-08-20
author: Xuban
tags: [LibreYOLO, YOLOv8 alternative, YOLO8 alternative, RF-DETR, YOLO-NAS, object detection, RF100-VL]
faq:
  - q: "Quelle est la meilleure alternative à YOLOv8 en 2026 ?"
    a: "RF-DETR-S pour un GPU. Il a obtenu 60.41 mAP50-95 lors de la campagne RF100-VL de LibreYOLO. L'article RF100-VL rapporte 56.9 pour YOLOv8m. YOLO-NAS-S est l'option convolutionnelle, avec 58.00."
  - q: "RF-DETR se transfère-t-il mieux que YOLOv8 ?"
    a: "Sur RF100-VL, oui. LibreYOLO a mesuré 60.41 pour RF-DETR-S et 61.13 pour RF-DETR-M. Roboflow rapporte 60.2 et 61.2. L'article rapporte 56.9 pour YOLOv8m."
  - q: "Puis-je utiliser YOLO-NAS à des fins commerciales ?"
    a: "Les poids pré-entraînés de Deci ne sont pas destinés à un usage commercial. L'architecture est sous Apache-2.0. Consultez l'article sur YOLO-NAS pour les détails de licence."
---

COCO est la référence habituelle pour choisir un détecteur. Une autre approche consiste à mesurer la qualité du fine-tuning du modèle sur d'autres données. [RF100-VL](https://arxiv.org/abs/2505.20612) évalue cela : un checkpoint pré-entraîné sur COCO est entraîné pendant 100 époques sur chacun de 100 datasets, puis évalué sur chaque split de test. Le chiffre correspond à la moyenne du mAP50-95.

YOLOv8m obtient 56.9, selon l'[annexe de l'article](https://papers.neurips.cc/paper_files/paper/2025/file/1013f8ff40a194f3f12a6bcc5221bb34-Paper-Datasets_and_Benchmarks_Track.pdf). Deux modèles que nous avons exécutés obtiennent un meilleur score : RF-DETR et YOLO-NAS. Les résultats bruts sont disponibles sur [huggingface.co/datasets/LibreYOLO/rf100-vl-results](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results).

## RF-DETR

RF-DETR-S obtient 60.41 et RF-DETR-M 61.13 dans la [campagne de LibreYOLO](/articles/rf100vl-benchmark). [Roboflow rapporte](https://rfdetr.roboflow.com/latest/learn/benchmarks/) 60.2 et 61.2. Les versions Nano à Large sont sous Apache-2.0.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=512, batch=8)
```

[Documentation](/docs/models/rf-detr) | [Poids](https://huggingface.co/LibreYOLO/LibreRFDETRs) | [Exécution `20260814-rfdetr-s-1b1190ee`](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main/rfdetr-s/20260814-rfdetr-s-1b1190ee)

## YOLO-NAS

YOLO-NAS-S obtient 58.00. C'est l'option convolutionnelle si vous recherchez un CNN de style YOLO. Les poids publiés par Deci ne sont pas destinés à un usage commercial. LibreYOLO n'en héberge aucun. Détails : [YOLO-NAS est toujours maintenu](/articles/yolo-nas-with-libreyolo).

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
```

[Documentation](/docs/models/yolo-nas) | [Exécution `20260809-yolonas-s-02926964`](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main/yolonas-s/20260809-yolonas-s-02926964)

Installez RF-DETR avec `pip install "libreyolo[rfdetr]"`, ou YOLO-NAS avec `pip install libreyolo`. Page de la campagne : [RF100-VL](/articles/rf100vl-benchmark).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentation](https://www.libreyolo.com/docs)
