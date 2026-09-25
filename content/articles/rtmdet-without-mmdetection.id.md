---
title: Cara Menjalankan RTMDet Tanpa mmdetection
description: "RTMDet adalah salah satu detector akurat tercepat yang tersedia. Instalasi resminya rusak pada semua versi PyTorch yang dirilis dalam dua tahun terakhir. Berikut alternatifnya."
date: 2026-06-27
author: Xuban
tags: [LibreYOLO, rtmdet, object-detection, tutorial]
faq:
  - q: "Mengapa mmcv gagal dipasang dengan PyTorch versi terbaru?"
    a: "Wheel mmcv terbaru, versi 2.2.0 dari April 2024, hanya menyediakan binary siap pakai hingga torch 2.4 / CUDA 12.1. Pada PyTorch yang lebih baru, mmcv harus mengompilasi ops C++/CUDA dari source. Proses ini memerlukan 10 hingga 30 menit, CUDA toolkit yang cocok, nvcc, dan compiler C++ yang kompatibel. Dari sinilah muncul error seperti modul mmcv._ext yang hilang."
  - q: "Bisakah saya menjalankan RTMDet tanpa memasang mmdetection?"
    a: "Bisa. LibreYOLO memuat bobot RTMDet yang dikonversi dari checkpoint OpenMMLab upstream, dan inferensinya setara bit demi bit dengan mmdetection pada checkpoint yang sama. Muat LibreRTMDets.pt berdasarkan nama, lalu file akan diunduh otomatis. Kelima ukuran, dari Tiny hingga X, semuanya berjalan pada 640 px."
  - q: "Apakah RTMDet gratis untuk penggunaan komersial?"
    a: "Ya. MMDetection dan RTMDet menggunakan Apache 2.0, bobot hasil konversi memiliki ketentuan Apache 2.0 yang sama, dan kode LibreYOLO berlisensi MIT. Tidak ada batasan komersial."
  - q: "Kapan mmdetection masih menjadi pilihan yang tepat?"
    a: "Jika Anda perlu melatih atau melakukan fine-tuning RTMDet dengan pipeline augmentasi MMDetection lengkap, atau sudah banyak menggunakan ekosistem OpenMMLab dan mmcv berfungsi dengan baik. Untuk inferensi dan deployment, LibreYOLO adalah pilihan yang lebih baik."
---

RTMDet adalah detector real-time dari OpenMMLab yang mencapai akurasi COCO tinggi sambil tetap cukup cepat untuk deployment. Arsitekturnya rapi. Instalasinya tidak.

Menjalankan RTMDet dari source resmi berarti merangkai stack OpenMMLab yang terdiri dari empat lapisan:

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

Rentang versi yang kompatibel sempit. mmdet 3.3.0 mensyaratkan `mmcv < 2.2.0`, tetapi `mim install mmcv>=2.0.0` akan memasang 2.2.0 begitu saja, yang kemudian gagal saat assertion runtime. Anda harus menentukan versinya sendiri.

Lalu ada masalah yang lebih besar. Wheel mmcv terbaru adalah versi 2.2.0, dirilis pada April 2024 dan tidak pernah diperbarui sejak itu. Wheel tersebut hanya menyediakan binary siap pakai hingga **torch 2.4 / CUDA 12.1**. Instalasi `pip install torch` yang baru sekarang memberi Anda PyTorch 2.12. Kesenjangan versi ini membuat mmcv harus mengompilasi ops C++/CUDA dari source: prosesnya memerlukan 10 hingga 30 menit, CUDA toolkit yang cocok, `nvcc`, dan compiler C++ yang kompatibel. Di sinilah berbagai error yang terdokumentasi muncul:

* `ModuleNotFoundError: No module named 'mmcv._ext'` -- ops hasil kompilasi tidak terbentuk atau tidak cocok,

* `nvcc fatal: Unsupported gpu architecture 'compute_86'` pada semua kartu RTX 30-series,

* `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'` -- stack ini memaksa Anda kembali ke Python 3.8,

* segmentation fault saat impor karena versi GCC tidak cocok.

FAQ OpenMMLab sendiri menyarankan pemasangan mmcv dengan pip, bukan mim, untuk menghindari jebakan versi. Halaman Get Started mereka menyarankan penggunaan mim. Keduanya dokumen resmi. Isinya saling bertentangan.

Setelah stack berhasil dijalankan, inferensi tetap mengharuskan Anda memilih file konfigurasi yang namanya mengodekan resep pelatihan, lalu mengunduh checkpoint secara terpisah dengan nama berisi hash, dan menghubungkannya melalui registry yang perlu dipelajari terlebih dahulu.

LibreYOLO menggantikan semua itu:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDets.pt")  # auto-downloads on first run
results = model("image.jpg", save=True)
```

Tanpa `mim`, tanpa matriks versi empat paket, tanpa kompilasi dari source, tanpa file konfigurasi, tanpa perlu mencari checkpoint berdasarkan hash, dan tanpa harus menggunakan Python 3.8. Bobot dikonversi dari checkpoint OpenMMLab upstream dan inferensinya setara bit demi bit dengan mmdetection pada checkpoint yang sama.

Tersedia lima ukuran: Tiny, Small, Medium, Large, dan X. Semuanya berjalan pada 640 px.

```python
print(results[0].boxes.xyxy)  # xyxy coordinates
print(results[0].boxes.conf)  # confidence scores
```

## Kapan pilihan aslinya masih tepat

Jika Anda perlu melatih atau melakukan fine-tuning RTMDet dengan pipeline augmentasi MMDetection lengkap, atau sudah banyak menggunakan ekosistem OpenMMLab dan mmcv berfungsi dengan baik, tetap gunakan stack tersebut. LibreYOLO adalah pilihan yang lebih baik untuk inferensi dan deployment.

## Catatan tentang lisensi

MMDetection dan RTMDet menggunakan Apache 2.0. Kode LibreYOLO berlisensi MIT. Bobot memiliki ketentuan Apache 2.0 yang sama dari upstream. Tidak ada batasan komersial.

## Coba

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDetl.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

LibreYOLO berlisensi MIT, berjalan di Linux, Mac, dan Windows, serta berfungsi di GPU, Apple Silicon, dan CPU biasa tanpa perubahan kode. Satu API mencakup RTMDet, RT-DETR, RF-DETR, D-FINE, YOLOX, YOLO-NAS, segmentasi, pose, kedalaman, dan lainnya.

Beri bintang di GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Dokumentasi: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
