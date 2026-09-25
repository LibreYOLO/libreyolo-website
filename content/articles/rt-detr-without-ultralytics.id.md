---
title: "RT-DETR tanpa Ultralytics: v1, v2, dan v4 dengan Apache-2.0"
description: "Jalankan dan lakukan fine-tuning RT-DETR, RT-DETRv2, dan RT-DETRv4 dengan bobot Apache-2.0 dalam library MIT: satu API untuk predict, train, val, dan export."
date: 2026-09-24
author: Xuban
tags: [LibreYOLO, rt-detr, rtdetr, rt-detrv4, object-detection, license, tutorial]
faq:
  - q: "Apakah RT-DETR gratis untuk penggunaan komersial?"
    a: "Ya, jika Anda menggunakan implementasi permisif. Kode asli RT-DETR dan RT-DETRv2 (lyuwenyu/RT-DETR), serta RT-DETRv4 (RT-DETRs/RT-DETRv4), berlisensi Apache-2.0, dan LibreYOLO menjalankan ketiga versi dengan bobot Apache-2.0 dalam library MIT. Implementasi RT-DETR di dalam paket ultralytics berlisensi AGPL-3.0, dengan Enterprise License berbayar sebagai alternatif untuk penggunaan closed source. Ini informasi umum, bukan nasihat hukum."
  - q: "Apa lisensi RT-DETRv4?"
    a: "Apache-2.0. Berkas LICENSE di github.com/RT-DETRs/RT-DETRv4 menggunakan Apache-2.0. RT-DETRv4 menggunakan teacher DINOv3 hanya selama pelatihan; bobot student yang dirilis tidak memuat parameter DINOv3, sehingga lisensi DINOv3 dari Meta tidak berlaku untuk bobot tersebut."
  - q: "Bisakah saya melakukan fine-tuning RT-DETR tanpa Ultralytics?"
    a: "Ya. LibreYOLO melakukan fine-tuning checkpoint deteksi RT-DETR, RT-DETRv2, dan RT-DETRv4 dengan model.train(), atau perintah libreyolo train di CLI, dimulai dari bobot COCO yang dipublikasikan. Model bounding box berorientasi pada RT-DETRv2 hanya mendukung inferensi."
  - q: "Bobot RT-DETR apa saja yang disertakan LibreYOLO?"
    a: "RT-DETR dalam r18, r34, r50, r50m, r101, l, dan x; RT-DETRv2 dalam r18, r34, r50, r50m, dan r101; RT-DETRv4 dalam s, m, l, dan x, semuanya untuk deteksi COCO pada 640 px. RT-DETRv2 juga memiliki model bounding box berorientasi n, s, m, l, dan x yang dilatih pada DOTA v1.0 pada 1024 px. Setiap berkas menggunakan Apache-2.0."
---

RT-DETR adalah transformer deteksi real-time dari Baidu. Model ini mendekode sekumpulan query tetap, bukan grid padat, sehingga tidak ada langkah NMS yang perlu disetel. Saat mencarinya, salah satu implementasi pertama yang ditemukan adalah implementasi di dalam paket Python `ultralytics`. Hal itu menimbulkan pertanyaan lisensi yang tidak pernah menjadi masalah pada modelnya sendiri.

## Lisensi RT-DETR bergantung pada repositorinya

Lisensi berlaku untuk repositori, bukan model. RT-DETR memiliki beberapa repositori, dan lisensinya berbeda-beda:

| Repositori | Cakupan | Lisensi |
|:---|:---|:---|
| [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | RT-DETR dan RT-DETRv2, PyTorch dan Paddle | Apache-2.0 |
| [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | RT-DETRv4 | Apache-2.0 |
| [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | RT-DETR di dalam paket `ultralytics` | AGPL-3.0 atau Enterprise License |
| [LibreYOLO](https://github.com/LibreYOLO/libreyolo) | RT-DETR, RT-DETRv2, dan RT-DETRv4 | Kode MIT, bobot Apache-2.0 |

Implementasi Ultralytics adalah pilihan yang solid. [Halaman RT-DETR](https://docs.ultralytics.com/models/rtdetr/) mencantumkan `rtdetr-l.pt` dan `rtdetr-x.pt` pretrained, dengan dukungan untuk pelatihan, validasi, inferensi, dan ekspor. Implementasi ini disertakan dalam paket berlisensi AGPL-3.0, dan Ultralytics menjual Enterprise License bagi tim yang tidak ingin membuka seluruh kode sumber proyeknya. Jika keduanya tidak sesuai, arsitekturnya tersedia dengan lisensi Apache-2.0 dari para pembuatnya sendiri. [Penjelasan lisensi YOLO](/articles/yolo-licenses-explained) membahas pola yang sama di seluruh keluarga YOLO, dan [panduan lisensi komersial](/articles/yolo-commercial-license) menjelaskan makna AGPL-3.0 bagi produk closed source.

## Jalankan RT-DETR dengan LibreYOLO

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO, SAMPLE_IMAGE

model = LibreYOLO("LibreRTDETRr18.pt")  # downloads from Hugging Face on first use
result = model(SAMPLE_IMAGE, save=True)

for box in result.boxes:
    print(box.cls, box.conf, box.xyxy)
```

Perintah yang sama dapat dijalankan dari CLI:

```bash
libreyolo predict model=LibreRTDETRr18.pt source=image.jpg save=True
```

`conf` dan `max_det` memfilter hasil dekode top-k atas query dan kelas. `iou` diterima tetapi tidak digunakan karena tidak ada NMS. Objek `Results` yang dikembalikan sama dengan yang dihasilkan setiap model LibreYOLO, sehingga mengganti detector hanya memerlukan perubahan satu baris.

## RT-DETR, RT-DETRv2, dan RT-DETRv4 dalam satu loader

Versi tercantum dalam nama berkas, dan `LibreYOLO()` memilih jalur berdasarkan checkpoint, sehingga ketiganya dimuat dengan cara yang sama:

* **RT-DETR:** `LibreRTDETR` dalam r18, r34, r50, r50m, r101 (backbone ResNet), serta l dan x (HGNetv2).
* **RT-DETRv2:** `LibreRTDETRv2` dalam r18, r34, r50, r50m, dan r101. Arsitekturnya tetap sama dengan versi 1, tetapi cara attention deformable mengambil sampel berubah.
* **RT-DETRv4:** `LibreRTDETRv4` dalam s, m, l, dan x. Model ini menggunakan kembali arsitektur D-FINE, dan bobotnya berasal dari distilasi teacher DINOv3 ke student HGNetv2.

Semua bobot deteksi dilatih pada COCO dan berjalan pada 640 px. Sebagai referensi, README upstream melaporkan 46.5 AP untuk RT-DETR-R18 dan 53.0 AP untuk RT-DETR-L pada COCO, serta 49.8 AP untuk RT-DETRv4-S hingga 57.0 AP untuk RT-DETRv4-X. [Halaman dokumentasi RT-DETR](/docs/models/rt-detr) memuat tabel benchmark LibreYOLO sendiri untuk setiap checkpoint.

RT-DETRv2 juga mendukung bounding box berorientasi. `LibreRTDETRv2n-obb.pt` hingga `LibreRTDETRv2x-obb.pt` adalah checkpoint resmi DOTA v1.0, dengan 15 kelas objek untuk citra udara beresolusi 1024 px, yang dikonversi ke format LibreYOLO:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRv2n-obb.pt")
result = model("aerial.png", save=True)
print(result.obb.xywhr)  # (N, 5): cx, cy, w, h, radians
```

[Deteksi berorientasi](/docs/tasks/oriented-detection) menjelaskan format label dan metriknya.

## Lakukan fine-tuning RT-DETR dengan data sendiri

Pelatihan dimulai dari checkpoint yang dipublikasikan:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRr18.pt")
model.train(data="coco128.yaml", epochs=50, batch=4, lr0=1e-4)
```

Arahkan `data` ke YAML dataset sendiri untuk menjalankan pelatihan sungguhan. Setiap versi memiliki nilai default learning rate sendiri, dan versi 1 serta 2 menetapkan learning rate backbone sebesar seperduapuluh dari `lr0`, mengikuti resep aslinya. Untuk multi-GPU, gunakan `device=0,1` di CLI. Untuk fine-tuning adapter [LoRA](/docs/train/lora), gunakan `lora=True` setelah memasang extra `lora`. Model berorientasi pada RT-DETRv2 hanya mendukung inferensi. Lihat [pelatihan](/docs/train) untuk informasi tentang dataset, augmentasi, dan logger.

## Validasi dan ekspor

```python
metrics = model.val(data="coco128.yaml")
print(metrics["metrics/mAP50-95"])

path = model.export(format="onnx")  # needs: pip install "libreyolo[onnx]"
```

`val()` mengembalikan dictionary biasa. Format ekspor yang tersedia mencakup ONNX, TorchScript, ExecuTorch, TensorRT, dan OpenVINO; matriks ekspor di [halaman model](/docs/models/rt-detr) menunjukkan format yang sudah divalidasi untuk setiap task. Berkas `.onnx` atau `.engine` hasil ekspor dapat dimuat kembali melalui `LibreYOLO()` dan mengembalikan `Results` yang sama. Lihat [ekspor](/docs/export) untuk panduan tiap format.

## Kapan repositori asli tetap menjadi pilihan yang tepat

Jika Anda perlu mereproduksi hasil paper dengan konfigurasi persis dari para pembuatnya, ingin menggunakan versi Paddle, atau ingin menjalankan sendiri distilasi teacher DINOv3 RT-DETRv4, gunakan repositori upstream. LibreYOLO melakukan fine-tuning pada student v4 yang dirilis, tetapi tidak menjalankan teacher. Jika proyek Anda sudah menggunakan AGPL-3.0 atau tercakup dalam Ultralytics Enterprise License, tidak ada alasan terkait lisensi untuk beralih.

## Catatan tentang lisensi

Kode LibreYOLO menggunakan MIT. Setiap berkas bobot RT-DETR menggunakan Apache-2.0, dan setiap repositori bobot Hugging Face menyertakan LICENSE dan NOTICE dari upstream, yang menurut Apache-2.0 harus tetap disertakan saat bobot didistribusikan ulang. Bobot yang Anda latih dengan data sendiri adalah milik Anda. RT-DETRv4 hanya menggunakan DINOv3 sebagai teacher selama pelatihan, dan student yang dirilis tidak memuat parameter DINOv3. Ini informasi umum, bukan nasihat hukum. Periksa berkas LICENSE terbaru sebelum merilis produk.

## Coba

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTDETRv4s.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO berlisensi MIT, berjalan di Linux, Mac, dan Windows, serta berfungsi di GPU, Apple Silicon, dan CPU biasa tanpa perubahan kode.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Dokumentasi RT-DETR](/docs/models/rt-detr)
