---
title: "Alternatif Real-ESRGAN 2026: Upscale Gambar dengan LibreYOLO"
description: "Jalankan upscaling gambar Real-ESRGAN 2x dan 4x melalui LibreYOLO, termasuk inferensi berbasis tile untuk gambar berukuran besar."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, real-esrgan-alternative, super-resolution, tutorial]
faq:
  - q: "Apakah Real-ESRGAN sudah ditinggalkan?"
    a: "Repositorinya belum diarsipkan, tetapi rilis GitHub terbarunya adalah versi 0.3.0 dari September 2022."
  - q: "Bisakah LibreYOLO melatih Real-ESRGAN?"
    a: "Tidak. LibreYOLO mendukung prediksi, validasi, dan ekspor Real-ESRGAN, bukan pelatihan."
---

[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) belum secara resmi diarsipkan, tetapi rilis GitHub terbarunya, versi 0.3.0, berasal dari September 2022. LibreYOLO menjalankan checkpoint 2x dan 4x-nya melalui API prediksi biasa.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRealESRGANx4-restore.pt")
result = model("large-photo.jpg", tile=512, save=True)
print(result.restored.array.shape)
```

LibreYOLO mendukung checkpoint 2x, 4x, dan fast 4x, dengan prediksi berbasis tile, validasi PSNR/SSIM, dan ekspor. Pelatihan belum diimplementasikan. Penggabungan hasil dengan kekuatan denoise dari upstream juga belum tersedia dalam port ini.

Jalankan `pip install libreyolo` untuk mencobanya. Opsi pemulihan lengkap tersedia di [dokumentasi Real-ESRGAN](https://www.libreyolo.com/docs/models/real-esrgan). Untuk pembahasan lebih lanjut, lihat [alternatif MiDaS](/articles/midas-alternative).

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Dokumentasi](https://www.libreyolo.com/docs)
