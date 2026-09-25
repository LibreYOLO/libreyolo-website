---
title: "Alternatif EfficientDet pada 2026: Jalankan D0-D4 dengan LibreYOLO"
description: "Jalankan prediksi, validasi, dan ekspor EfficientDet D0-D4 melalui API deteksi objek LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, efficientdet-alternative, object-detection, tutorial]
faq:
  - q: "Apakah EfficientDet sudah ditinggalkan?"
    a: "Repositori PyTorch rwightman tidak secara resmi diarsipkan. Lebih tepat menyebutnya sepi aktivitas daripada ditinggalkan."
  - q: "Bisakah LibreYOLO melatih EfficientDet?"
    a: "Tidak. LibreYOLO mendukung inferensi, validasi, dan ekspor EfficientDet, tetapi tidak mendukung pelatihan."
---

Repositori [EfficientDet PyTorch](https://github.com/rwightman/efficientdet-pytorch) yang banyak digunakan tidak diarsipkan, jadi keliru jika menyebutnya sudah ditinggalkan. Repositori itu sepi aktivitas. LibreYOLO menyediakan jalur deployment lain.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreEfficientDetd0.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO mendukung EfficientDet D0 hingga D4 untuk prediksi dan validasi. Ekspor ONNX, TorchScript, OpenVINO, dan TensorRT telah tercakup dalam pengujian kesetaraan. Pelatihan belum diimplementasikan, dan setiap ukuran mempertahankan resolusi input native yang tetap.

Pasang dengan `pip install libreyolo`. Lihat [dokumentasi EfficientDet](https://www.libreyolo.com/docs/models/efficientdet) untuk setiap ukuran, atau bandingkan dengan [alternatif Facebook DETR](/articles/facebook-detr-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Dokumentasi](https://www.libreyolo.com/docs)
