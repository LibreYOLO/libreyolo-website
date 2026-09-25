---
title: "YOLO-NAS Masih Dipelihara dan Bisa Digunakan dengan LibreYOLO"
description: "Jalankan, latih, validasi, dan ekspor YOLO-NAS dengan LibreYOLO. Rilis terakhir SuperGradients terbit pada April 2024. Kami berencana melatih bobot baru dari nol agar tidak bergantung pada lisensi Deci."
date: 2026-08-20
author: Xuban
tags: [LibreYOLO, yolo-nas, yolo-nas-maintained, object-detection]
faq:
  - q: "Apakah YOLO-NAS sudah ditinggalkan?"
    a: "SuperGradients terakhir merilis versi 3.7.1 pada April 2024. LibreYOLO mendukung prediksi, pelatihan, validasi, dan ekspor YOLO-NAS dengan PyTorch versi saat ini."
  - q: "Apakah Ultralytics memelihara YOLO-NAS?"
    a: "Ultralytics mendukung inferensi, validasi, dan ekspor. Ultralytics tidak mendukung pelatihan."
  - q: "Apakah bobot pretrained YOLO-NAS boleh digunakan secara komersial?"
    a: "Bobot pretrained dari Deci tetap tunduk pada ketentuan nonkomersial upstream, apa pun library yang memuatnya. Pelatihan dari inisialisasi acak menghindari penggunaan checkpoint tersebut. Kami juga berencana menerbitkan bobot COCO yang dilatih dari nol."
---

YOLO-NAS masih merupakan model deteksi yang berguna. Repositori asalnya, [SuperGradients](https://github.com/Deci-AI/super-gradients), terakhir merilis versi 3.7.1 pada April 2024. LibreYOLO memelihara dukungan prediksi, pelatihan, validasi, dan ekspor untuk model ini.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO mendukung ukuran S, M, dan L untuk deteksi, serta pose. Checkpoint diunduh dari CDN publik Deci dengan nama seperti `LibreYOLONASs.pt`. LibreYOLO tidak meng-host satu pun checkpoint tersebut, dan ketentuan nonkomersial Deci tetap berlaku. SuperGradients membatasi versi PyTorch pada rentang 1.9 hingga 1.13, sedangkan LibreYOLO memerlukan versi 2.4 atau lebih baru. Ekspor deteksi mendukung ONNX, TorchScript, OpenVINO, NCNN, TFLite, Paddle, MNN, ExecuTorch, dan Core AI. CoreML tidak didukung.

Untuk melatih tanpa checkpoint Deci:

```python
from libreyolo import LibreYOLONAS

model = LibreYOLONAS(None, size="s")
model.train(data="my-dataset.yaml", imgsz=640, batch=16)
```

Kami juga berencana melatih YOLO-NAS dari nol dengan COCO dan menerbitkan bobotnya.

Instal dengan `pip install libreyolo`. [Dokumentasi YOLO-NAS](https://www.libreyolo.com/docs/models/yolo-nas) memuat detail model. Baca juga artikel [Alternatif SuperGradients](/articles/supergradients-alternative) yang lebih lengkap.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Dokumentasi](https://www.libreyolo.com/docs)
