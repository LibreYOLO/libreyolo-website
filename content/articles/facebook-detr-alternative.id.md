---
title: "Alternatif Facebook DETR 2026: Jalankan DETR dengan LibreYOLO"
description: "Repositori Facebook Research DETR yang asli telah diarsipkan. Jalankan empat varian resmi DETR melalui LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, facebook-detr-alternative, object-detection, transformer]
faq:
  - q: "Apakah Facebook DETR sudah ditinggalkan?"
    a: "Repositori facebookresearch/detr yang asli diarsipkan pada March 12, 2024."
  - q: "Bisakah LibreYOLO melatih DETR yang asli?"
    a: "Tidak. LibreYOLO mendukung prediksi, validasi, dan ekspor DETR, tetapi tidak mendukung resep pelatihannya selama 500 epoch."
---

Repositori [Facebook Research DETR](https://github.com/facebookresearch/detr) yang asli diarsipkan pada March 12, 2024. LibreYOLO tetap membuat detektor asli ini dapat digunakan melalui API yang sama dengan keluarga model lainnya.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreDETRr50.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

Empat varian ResNet-50 dan ResNet-101 tersedia untuk prediksi, validasi, serta ekspor ONNX dan TorchScript. Pelatihan belum diimplementasikan. LibreYOLO juga menggunakan input tetap berukuran 800 kali 800, bukan mereproduksi pipeline evaluasi asli yang mempertahankan rasio aspek.

Coba setelah menjalankan `pip install libreyolo`. [Dokumentasi DETR](https://www.libreyolo.com/docs/models/detr) mencantumkan keempat checkpoint. Untuk perbandingan dengan CNN, lihat [alternatif EfficientDet](/articles/efficientdet-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Dokumentasi](https://www.libreyolo.com/docs)
