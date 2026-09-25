---
title: "LibreYOLO di CVPR 2026: Demo Edge AI di Hailo-8L dan Snapdragon"
description: "Di CVPR 2026 di Denver, tim Jabra dan IT University of Copenhagen menggunakan LibreYOLOXs sebagai model contoh dalam tutorial \"Edge AI in Action\", lalu menjalankannya di Hailo-8L dan Snapdragon."
date: 2026-06-30
author: Xuban
tags: [LibreYOLO, yolox, edge-ai, cvpr]
faq:
  - q: "Seberapa cepat LibreYOLOXs berjalan di perangkat edge?"
    a: "Dalam tutorial CVPR 2026, LibreYOLOXs berjalan pada 19.0 ms per frame (52.6 FPS) di Raspberry Pi 5 dengan Hailo-8L, dan 6.69 ms di HTP/DSP Qualcomm QCS6490 setelah kuantisasi INT8."
  - q: "Bagaimana cara menerapkan model LibreYOLO ke akselerator edge?"
    a: "Ekspor ke ONNX dengan pemanggilan ekspor LibreYOLO, lalu kompilasi untuk target: Hailo Dataflow Compiler menghasilkan HEF untuk chip Hailo, sedangkan tumpukan SNPE / QAIRT / AI Hub mendukung Qualcomm Snapdragon. Inilah pipeline persis yang dijelaskan dari awal hingga akhir dalam tutorial CVPR."
---

![CVPR 2026, Denver, Colorado, 3-7 Juni](/articles/libreyolo-at-cvpr-2026/cvpr-denver-banner.png)

LibreYOLO hadir di CVPR 2026.

CVPR dikenal luas sebagai konferensi computer vision paling bergengsi di dunia. Tahun ini konferensi tersebut berlangsung di Denver, dan tim dari Jabra serta IT University of Copenhagen mengadakan tutorial berjudul "Edge AI in Action: Mastering On-Device Inference". Mereka memilih LibreYOLOXs sebagai model contoh untuk menjalankan deteksi objek pada chip edge.

## Yang mereka bangun

Tutorial tersebut membahas pipeline edge secara menyeluruh, dari awal hingga akhir. Mulai dengan model LibreYOLOXs. Ekspor ke ONNX. Lalu kompilasi ONNX tersebut untuk dua perangkat keras yang sangat berbeda: akselerator Hailo-8L dan Qualcomm Snapdragon.

## Real-time di Hailo-8L

Mereka mengompilasi ONNX menjadi HEF dengan Hailo Dataflow Compiler, lalu menjalankannya di Raspberry Pi 5 dengan Hailo-8L AI HAT.

![LibreYOLOXs berjalan di Raspberry Pi 5 dengan Hailo-8L, mendeteksi orang dan tas tangan di jalan yang ramai](/articles/libreyolo-at-cvpr-2026/live-detection-hailo.png)

Waktu pemrosesannya 19.0 ms per frame, setara 52.6 FPS, di Raspberry Pi.

## Real-time di Snapdragon

Di sisi Qualcomm, mereka mengkuantisasi LibreYOLOXs ke INT8 dan menjalankannya melalui tumpukan SNPE, QAIRT, dan AI Hub, dengan benchmark pada QCS6490.

<img src="/articles/libreyolo-at-cvpr-2026/live-detection-qualcomm.png" alt="LibreYOLOXs mendeteksi TV, laptop, keyboard, mouse, dan ponsel secara langsung di aplikasi EdgeVision AI pada ponsel Snapdragon" style="display:block;margin:1.5rem auto;max-width:360px;width:100%" />

Hasilnya 6.69 ms pada HTP/DSP.

## Terima kasih

Terima kasih banyak kepada Sai Narsi Reddy Donthi Reddy, Fabricio Batista Narcizo, Elizabete Munzlingera, dan Shan Ahmed Shaffi yang telah menampilkan proyek ini.

- Tutorial dan slide: [Edge AI dalam Aksi: Menguasai Inferensi di Perangkat](https://www.fabricionarcizo.com/cvpr2026-edge-ai-in-action/)
- Model LibreYOLOXs yang dikuantisasi untuk Qualcomm: [di Hugging Face](https://huggingface.co/fabricionarcizo/LibreYOLOXs)

## Coba

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
model.export(format="onnx")   # then compile for your edge target
```

LibreYOLO berlisensi MIT, berjalan di Linux, Mac, dan Windows, serta berfungsi di GPU, Apple Silicon, dan CPU biasa tanpa perubahan kode. Satu API mendukung YOLOX, RF-DETR, D-FINE, DEIM, YOLO-NAS, segmentasi, pose, kedalaman, dan lainnya.

Beri bintang di GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Dokumentasi: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
