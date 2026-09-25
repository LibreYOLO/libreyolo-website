---
title: "Alternatif YOLOv8 Terbaik 2026: RF-DETR dan YOLO-NAS"
description: "RF100-VL mengukur seberapa baik detector melakukan fine-tuning di luar COCO. RF-DETR dan YOLO-NAS mendapat skor lebih tinggi daripada YOLOv8. Cara menjalankan keduanya di LibreYOLO."
date: 2026-08-20
author: Xuban
tags: [LibreYOLO, YOLOv8 alternative, YOLO8 alternative, RF-DETR, YOLO-NAS, object detection, RF100-VL]
faq:
  - q: "Apa alternatif YOLOv8 terbaik pada 2026?"
    a: "RF-DETR-S untuk GPU. Skornya 60.41 mAP50-95 dalam campaign RF100-VL LibreYOLO. Paper RF100-VL melaporkan skor 56.9 untuk YOLOv8m. YOLO-NAS-S adalah opsi konvolusional, dengan skor 58.00."
  - q: "Apakah RF-DETR melakukan transfer lebih baik daripada YOLOv8?"
    a: "Pada RF100-VL, ya. LibreYOLO mengukur skor RF-DETR-S sebesar 60.41 dan RF-DETR-M sebesar 61.13. Roboflow melaporkan 60.2 dan 61.2. Paper melaporkan skor 56.9 untuk YOLOv8m."
  - q: "Bisakah YOLO-NAS digunakan secara komersial?"
    a: "Bobot pretrained dari Deci tidak boleh digunakan secara komersial. Arsitekturnya berlisensi Apache-2.0. Lihat artikel YOLO-NAS untuk detail lisensinya."
---

COCO adalah cara yang umum untuk memilih detector. Sudut pandang lain adalah seberapa baik model melakukan fine-tuning pada data lain. [RF100-VL](https://arxiv.org/abs/2505.20612) mengukurnya: checkpoint pretrained pada COCO dilatih selama 100 epoch pada masing-masing dari 100 dataset, lalu dinilai pada setiap test split. Angkanya adalah nilai rata-rata mAP50-95.

YOLOv8m mendapat skor 56.9, menurut [lampiran paper](https://papers.neurips.cc/paper_files/paper/2025/file/1013f8ff40a194f3f12a6bcc5221bb34-Paper-Datasets_and_Benchmarks_Track.pdf). Dua model yang kami jalankan mendapat skor lebih tinggi: RF-DETR dan YOLO-NAS. Hasil mentah tersedia di [huggingface.co/datasets/LibreYOLO/rf100-vl-results](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results).

## RF-DETR

RF-DETR-S mendapat skor 60.41 dan RF-DETR-M 61.13 dalam [campaign LibreYOLO](/articles/rf100vl-benchmark). [Roboflow melaporkan](https://rfdetr.roboflow.com/latest/learn/benchmarks/) skor 60.2 dan 61.2. Varian Nano hingga Large berlisensi Apache-2.0.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=512, batch=8)
```

[Dokumentasi](/docs/models/rf-detr) | [Bobot](https://huggingface.co/LibreYOLO/LibreRFDETRs) | [Run `20260814-rfdetr-s-1b1190ee`](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main/rfdetr-s/20260814-rfdetr-s-1b1190ee)

## YOLO-NAS

YOLO-NAS-S mendapat skor 58.00. Ini opsi konvolusional jika Anda menginginkan CNN bergaya YOLO. Bobot yang dipublikasikan Deci tidak boleh digunakan secara komersial. LibreYOLO tidak menyediakan bobot tersebut. Detailnya: [YOLO-NAS masih dipelihara](/articles/yolo-nas-with-libreyolo).

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
```

[Dokumentasi](/docs/models/yolo-nas) | [Run `20260809-yolonas-s-02926964`](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main/yolonas-s/20260809-yolonas-s-02926964)

Instal dengan `pip install "libreyolo[rfdetr]"` untuk RF-DETR, atau `pip install libreyolo` untuk YOLO-NAS. Halaman campaign: [RF100-VL](/articles/rf100vl-benchmark).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Dokumentasi](https://www.libreyolo.com/docs)
