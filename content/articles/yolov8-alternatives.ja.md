---
title: "2026年版：YOLOv8の代替モデル、RF-DETRとYOLO-NASを比較"
description: "RF100-VLは、COCO以外のデータセットで検出器をファインチューニングした性能を測定します。RF-DETRとYOLO-NASはYOLOv8を上回ります。LibreYOLOで両方を実行する方法を紹介します。"
date: 2026-08-20
author: Xuban
tags: [LibreYOLO, YOLOv8 alternative, YOLO8 alternative, RF-DETR, YOLO-NAS, object detection, RF100-VL]
faq:
  - q: "2026年に最適なYOLOv8の代替モデルは何ですか？"
    a: "GPUを使うならRF-DETR-Sです。LibreYOLOのRF100-VLキャンペーンで60.41 mAP50-95を記録しました。RF100-VLの論文ではYOLOv8mが56.9と報告されています。畳み込み型の選択肢はYOLO-NAS-Sで、スコアは58.00です。"
  - q: "RF-DETRはYOLOv8より転移性能が高いですか？"
    a: "RF100-VLでは高いです。LibreYOLOはRF-DETR-Sで60.41、RF-DETR-Mで61.13を測定しました。Roboflowの報告値は60.2と61.2です。論文ではYOLOv8mが56.9と報告されています。"
  - q: "YOLO-NASを商用利用できますか？"
    a: "Deciの学習済み重みは商用利用できません。アーキテクチャのライセンスはApache-2.0です。YOLO-NASの記事でライセンスの詳細を確認してください。"
---

検出器を選ぶときは、通常COCOが基準になります。別の見方として、ほかのデータでモデルをどれだけうまくファインチューニングできるかがあります。[RF100-VL](https://arxiv.org/abs/2505.20612)はその性能を測定します。COCOで事前学習したチェックポイントを、100個のデータセットそれぞれで100エポック学習させ、その後、各テスト分割で評価します。数値はmAP50-95の平均です。

[論文の付録](https://papers.neurips.cc/paper_files/paper/2025/file/1013f8ff40a194f3f12a6bcc5221bb34-Paper-Datasets_and_Benchmarks_Track.pdf)によると、YOLOv8mは56.9です。私たちが実行した2つのモデル、RF-DETRとYOLO-NASはこれを上回ります。実行結果は[こちら](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results)で確認できます。

## RF-DETR

[LibreYOLOのキャンペーン](/articles/rf100vl-benchmark)ではRF-DETR-Sが60.41、RF-DETR-Mが61.13です。[Roboflowの報告](https://rfdetr.roboflow.com/latest/learn/benchmarks/)では60.2と61.2です。NanoからLargeまでApache-2.0です。

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=512, batch=8)
```

[ドキュメント](/docs/models/rf-detr) | [重み](https://huggingface.co/LibreYOLO/LibreRFDETRs) | [実行 `20260814-rfdetr-s-1b1190ee`](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main/rfdetr-s/20260814-rfdetr-s-1b1190ee)

## YOLO-NAS

YOLO-NAS-Sは58.00です。YOLOスタイルのCNNを使いたい場合の畳み込み型モデルです。Deciが公開した重みは商用利用できません。LibreYOLOはその重みを提供していません。詳細は[YOLO-NASは現在もメンテナンスされています](/articles/yolo-nas-with-libreyolo)をご覧ください。

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
```

[ドキュメント](/docs/models/yolo-nas) | [実行 `20260809-yolonas-s-02926964`](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main/yolonas-s/20260809-yolonas-s-02926964)

RF-DETRには`pip install "libreyolo[rfdetr]"`を、YOLO-NASには`pip install libreyolo`を実行してインストールします。キャンペーンページ：[RF100-VL](/articles/rf100vl-benchmark)。

[GitHub](https://github.com/LibreYOLO/libreyolo) | [ドキュメント](https://www.libreyolo.com/docs)
