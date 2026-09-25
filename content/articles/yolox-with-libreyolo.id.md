---
title: "Alternatif YOLOX di 2026: Jalankan YOLOX dengan LibreYOLO"
description: "Rilis YOLOX terbaru berasal dari 2022. Jalankan, latih, validasi, dan ekspor YOLOX melalui API LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, yolox-alternative, object-detection, tutorial]
faq:
  - q: "Apakah YOLOX sudah ditinggalkan?"
    a: "Repositori resminya belum diarsipkan, tetapi rilis terbarunya adalah YOLOX 0.3.0 dari April 2022. Lebih tepat menyebutnya tidak aktif daripada secara resmi ditinggalkan."
  - q: "Bisakah LibreYOLO melatih YOLOX?"
    a: "Ya. LibreYOLO mendukung prediksi, pelatihan, validasi, dan ekspor YOLOX."
---

YOLOX masih merupakan detektor berlisensi Apache-2.0 yang berguna. [Repositori resminya](https://github.com/Megvii-BaseDetection/YOLOX) belum diarsipkan, tetapi rilis terbarunya, 0.3.0, bertanggal April 2022. Masalah pemeliharaannya memang nyata. Modelnya masih berfungsi.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLOXs.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO mendukung enam ukuran YOLOX, dari Nano hingga X, dengan prediksi, pelatihan, validasi, dan ekspor melalui satu API. Gunakan proyek aslinya jika Anda memerlukan alur kerja konfigurasi `Exp` yang persis sama.

Mulai dengan `pip install libreyolo`. [Dokumentasi YOLOX](https://www.libreyolo.com/docs/models/yolox) membahas seluruh API. Ada juga catatan singkat bahwa [YOLO-NAS masih dipelihara](/articles/yolo-nas-with-libreyolo).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Dokumentasi](https://www.libreyolo.com/docs)
