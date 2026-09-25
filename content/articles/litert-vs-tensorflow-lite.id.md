---
title: "LiteRT vs TensorFlow Lite: Apa Itu LiteRT dan Apakah TensorFlow Lite Sudah Dihentikan?"
description: "Google mengganti nama TensorFlow Lite menjadi LiteRT pada September 2024. Runtime dan berkas .tflite tetap sama, tidak ada yang rusak. Berikut alasan perubahan nama, apa yang berubah, dan cara mengekspor model LibreYOLO ke format ini."
date: 2026-07-10
author: Xuban
tags: [LibreYOLO, litert, tensorflow-lite, tflite, export, edge-ai, tutorial]
faq:
  - q: "Apakah TensorFlow Lite sudah dihentikan?"
    a: "Nama TensorFlow Lite sedang dipensiunkan, bukan teknologinya. Runtime tetap tersedia sebagai LiteRT dengan API yang sama, dan model serta kode TFLite yang sudah ada tetap berfungsi. Pengembangan baru, seperti API CompiledModel dan akselerasi NPU, dirilis dengan nama LiteRT."
  - q: "Apakah saya perlu mengekspor ulang model untuk LiteRT?"
    a: "Tidak. Berkas .tflite yang diekspor untuk TensorFlow Lite adalah model LiteRT. Tidak ada perubahan pada berkasnya, jadi tidak ada yang perlu diekspor ulang atau dimigrasikan."
---

**LiteRT adalah TensorFlow Lite dengan nama baru. Ini bukan runtime baru dan bukan format berkas baru. Model tetap berupa berkas `.tflite`, dan semua berkas `.tflite` yang berjalan di TensorFlow Lite tetap berjalan di LiteRT tanpa perubahan.**

Jika Anda menelusuri "LiteRT vs TensorFlow Lite", "apakah TensorFlow Lite sudah dihentikan", atau "apa itu LiteRT", paragraf itu jawabannya. Bagian selanjutnya menjelaskan secara singkat alasan perubahan nama dan cara mengekspor model ke format ini dengan LibreYOLO.

## Mengapa Google mengganti namanya

TensorFlow Lite diluncurkan pada 2017 sebagai cara untuk menjalankan model TensorFlow di ponsel dan board tertanam. Selama bertahun-tahun, cakupannya berkembang jauh melampaui itu: Google membangun jalur konversi dari PyTorch, JAX, dan Keras, sehingga pada 2024 sebagian besar model yang berjalan di dalamnya bahkan tidak pernah menggunakan TensorFlow.

Pada saat itu, namanya benar-benar menyesatkan. Pengguna PyTorch melewatkannya karena namanya terdengar seperti tool khusus TensorFlow. Jadi, pada September 2024 Google [mengganti namanya menjadi LiteRT](https://developers.googleblog.com/tensorflow-lite-is-now-litert/), singkatan dari "Lite Runtime". Hanya itu ceritanya. Tim dan kode tetap sama, namanya kini netral terhadap framework.

## Apa yang sebenarnya berubah

Perubahannya sangat sedikit, dan tidak ada yang merusak apa pun:

* Kodenya pindah ke lokasi baru: [github.com/google-ai-edge/LiteRT](https://github.com/google-ai-edge/litert).
* Dokumentasinya pindah ke [ai.google.dev/edge/litert](https://ai.google.dev/edge/litert).
* Paket Python untuk menjalankan model kini bernama `ai-edge-litert`. Paket lama `tflite-runtime` sudah usang; paket baru memiliki class `Interpreter` yang sama.
* Sejak perubahan nama, fitur baru dirilis dengan nama LiteRT: API `CompiledModel` yang lebih sederhana dan akselerasi GPU/NPU yang dinyatakan Google [siap untuk produksi pada Januari 2026](https://developers.googleblog.com/litert-the-universal-framework-for-on-device-ai/).

API Interpreter klasik juga tetap berfungsi seperti sebelumnya. Ada satu ekstensi yang benar-benar baru, `.litertlm`, tetapi format pengemasan ini hanya untuk LLM di perangkat. Untuk model computer vision, format ini tidak ada. Jadi, saat satu dokumentasi menyebut "TFLite" dan dokumentasi lain menyebut "LiteRT", keduanya membicarakan berkas yang sama.

## Mengekspor model LibreYOLO ke LiteRT

LibreYOLO menambahkan jalur ekspor TFLite/LiteRT pada v1.3.0. Fitur ini memerlukan Python 3.12+ dan satu paket tambahan:

```bash
pip install "libreyolo[tflite]"   # Python 3.12+
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9s.pt")  # auto-downloads on first run
model.export(format="tflite")        # writes a .tflite file
```

Atau, gunakan CLI:

```bash
libreyolo export model=LibreYOLO9s.pt format=tflite
```

Jalur yang sudah divalidasi saat ini adalah deteksi YOLO9 serta deteksi, segmentasi, dan pose RF-DETR. Semuanya masih ditandai eksperimental. Matriks dukungan lengkap tersedia di [dokumentasi](/docs/v1.3.0).

Untuk menjalankan berkas yang diekspor, gunakan paket `ai-edge-litert` dari Google di perangkat target:

```bash
pip install ai-edge-litert
```

```python
import numpy as np
from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

image = np.zeros((1, 640, 640, 3), dtype=np.float32)  # NHWC, your preprocessed image here
interpreter.set_tensor(inp["index"], image)
interpreter.invoke()
predictions = interpreter.get_tensor(out["index"])
```

Perhatikan bahwa input menggunakan NHWC (kanal di akhir): konversi dari ONNX mengubah tata letaknya, dan ini normal untuk format tersebut.

## Perubahan yang kami lakukan di LibreYOLO

Tidak banyak, karena memang tidak banyak yang perlu diubah. Dokumentasi kini memberi label format "TFLite (LiteRT)" agar kedua nama mudah ditemukan, dan API tetap menggunakan `format="tflite"` sebagai nama formatnya.

## FAQ

**Apakah TensorFlow Lite sudah dihentikan?**
Nama TensorFlow Lite sedang dipensiunkan, bukan teknologinya. Runtime tetap tersedia sebagai LiteRT dengan API yang sama, dan model serta kode TFLite yang sudah ada tetap berfungsi. Pengembangan baru, seperti API CompiledModel dan akselerasi NPU, dirilis dengan nama LiteRT.

**Apakah saya perlu mengekspor ulang model untuk LiteRT?**
Tidak. Berkas `.tflite` yang diekspor untuk TensorFlow Lite adalah model LiteRT. Tidak ada perubahan pada berkasnya, jadi tidak ada yang perlu diekspor ulang atau dimigrasikan.
