---
title: "Alternatif MiDaS pada 2026: Estimasi Kedalaman dengan LibreYOLO"
description: "Repositori resmi MiDaS telah diarsipkan. Jalankan model kedalaman Small dan DPT-Large melalui LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, midas-alternative, depth-estimation, tutorial]
faq:
  - q: "Apakah MiDaS sudah ditinggalkan?"
    a: "Repositori resmi isl-org/MiDaS diarsipkan pada 25 Agustus 2025."
  - q: "Apakah MiDaS menghasilkan kedalaman metrik?"
    a: "Tidak. MiDaS menghasilkan kedalaman invers relatif tanpa satuan metrik atau skala tetap antar gambar."
---

[Repositori resmi MiDaS](https://github.com/isl-org/MiDaS) diarsipkan pada 25 Agustus 2025. LibreYOLO menyediakan cara yang terus dipelihara untuk menjalankan checkpoint Small dan DPT-Large.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreMiDaSl-depth.pt")
result = model("image.jpg", save=True)
print(result.depth_map.data.shape)
```

Prediksi, validasi zero-shot, dan ekspor dengan resolusi tetap didukung. Pelatihan tidak didukung. Outputnya berupa kedalaman invers relatif: nilai yang lebih tinggi menunjukkan objek lebih dekat, tetapi tidak memiliki satuan metrik atau skala tetap antar gambar.

Instalasi dasar sudah cukup: `pip install libreyolo`. Lanjutkan dengan [dokumentasi MiDaS](https://www.libreyolo.com/docs/models/midas) atau [alternatif Real-ESRGAN](/articles/real-esrgan-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Dokumentasi](https://www.libreyolo.com/docs)
