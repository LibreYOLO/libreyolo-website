---
title: "Melhores alternativas ao YOLOv8 em 2026"
description: "O RF100-VL mede o desempenho do fine-tuning de um detector fora do COCO. RF-DETR e YOLO-NAS superam o YOLOv8. Veja como executar ambos no LibreYOLO."
date: 2026-08-20
author: Xuban
tags: [LibreYOLO, YOLOv8 alternative, YOLO8 alternative, RF-DETR, YOLO-NAS, object detection, RF100-VL]
faq:
  - q: "Qual é a melhor alternativa ao YOLOv8 em 2026?"
    a: "RF-DETR-S para uma GPU. Ele obteve 60.41 mAP50-95 na campanha RF100-VL do LibreYOLO. O artigo do RF100-VL informa 56.9 para o YOLOv8m. O YOLO-NAS-S é a opção convolucional, com 58.00."
  - q: "O RF-DETR transfere melhor que o YOLOv8?"
    a: "No RF100-VL, sim. O LibreYOLO mediu 60.41 para o RF-DETR-S e 61.13 para o RF-DETR-M. A Roboflow informa 60.2 e 61.2. O artigo informa 56.9 para o YOLOv8m."
  - q: "Posso usar o YOLO-NAS comercialmente?"
    a: "Os pesos pré-treinados da Deci não podem ser usados comercialmente. A arquitetura usa Apache-2.0. O LibreYOLO não hospeda esses pesos. Veja os detalhes da licença no artigo sobre YOLO-NAS."
---

O COCO é o critério habitual para escolher um detector. Outra perspectiva é avaliar o desempenho do modelo ao fazer fine-tuning com outros dados. O [RF100-VL](https://arxiv.org/abs/2505.20612) mede isso: um checkpoint pré-treinado no COCO é treinado por 100 épocas em cada um de 100 datasets e, em seguida, avaliado em cada conjunto de teste. O número corresponde à média de mAP50-95.

O YOLOv8m obtém 56.9, segundo o [apêndice do artigo](https://papers.neurips.cc/paper_files/paper/2025/file/1013f8ff40a194f3f12a6bcc5221bb34-Paper-Datasets_and_Benchmarks_Track.pdf). Dois modelos que executamos têm pontuações maiores: RF-DETR e YOLO-NAS. Os resultados brutos estão em [huggingface.co/datasets/LibreYOLO/rf100-vl-results](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results).

## RF-DETR

O RF-DETR-S obteve 60.41 e o RF-DETR-M, 61.13 na [campanha do LibreYOLO](/articles/rf100vl-benchmark). A [Roboflow informa](https://rfdetr.roboflow.com/latest/learn/benchmarks/) 60.2 e 61.2. As variantes de Nano a Large usam Apache-2.0.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=512, batch=8)
```

[Documentação](/docs/models/rf-detr) | [Pesos](https://huggingface.co/LibreYOLO/LibreRFDETRs) | [Execução `20260814-rfdetr-s-1b1190ee`](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main/rfdetr-s/20260814-rfdetr-s-1b1190ee)

## YOLO-NAS

O YOLO-NAS-S obteve 58.00 e é a opção convolucional para quem quer uma CNN no estilo YOLO. Os pesos publicados pela Deci não podem ser usados comercialmente. O LibreYOLO não hospeda nenhum deles. Saiba mais: [YOLO-NAS continua sendo mantido](/articles/yolo-nas-with-libreyolo).

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
```

[Documentação](/docs/models/yolo-nas) | [Execução `20260809-yolonas-s-02926964`](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main/yolonas-s/20260809-yolonas-s-02926964)

Instale com `pip install "libreyolo[rfdetr]"` para usar RF-DETR ou com `pip install libreyolo` para usar YOLO-NAS. Página da campanha: [RF100-VL](/articles/rf100vl-benchmark).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Documentação](https://www.libreyolo.com/docs)
