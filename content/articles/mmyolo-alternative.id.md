---
title: "Alternatif MMYOLO pada 2026: Jalankan RTMDet dengan LibreYOLO"
description: "Rilis terbaru MMYOLO berasal dari 2023. Jalankan detector yang didukung seperti RTMDet melalui API langsung LibreYOLO."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, mmyolo-alternative, rtmdet, object-detection]
faq:
  - q: "Apakah MMYOLO sudah ditinggalkan?"
    a: "MMYOLO tidak secara resmi diarsipkan, tetapi rilis GitHub terbarunya adalah versi 0.6.0 dari Agustus 2023."
  - q: "Bisakah LibreYOLO memuat checkpoint MMYOLO apa pun?"
    a: "Tidak. LibreYOLO mendukung family model tertentu dan tidak menjanjikan pemuatan langsung untuk checkpoint MMYOLO sembarang."
---

[MMYOLO](https://github.com/open-mmlab/mmyolo) belum diarsipkan, tetapi rilis terbarunya, versi 0.6.0, berasal dari Agustus 2023. Jika hanya memerlukan detector yang didukung seperti RTMDet, Anda mungkin tidak memerlukan seluruh stack OpenMMLab.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRTMDets.pt")
result = model("image.jpg", save=True)
print(result.boxes.xyxy)
```

LibreYOLO bukan pengganti MMYOLO sepenuhnya. LibreYOLO tidak mereproduksi setiap konfigurasi, recipe, atau checkpoint. Namun, LibreYOLO menyediakan API langsung untuk family model yang didukung seperti RTMDet dan YOLOX, untuk prediksi, pelatihan, validasi, dan ekspor.

Mulai dengan `pip install libreyolo`. [Dokumentasi RTMDet](https://www.libreyolo.com/docs/models/rtmdet) mencantumkan operasi yang didukung. [Panduan singkat RTMDet](/articles/rtmdet-without-mmdetection) membahas migrasinya.

[GitHub](https://github.com/LibreYOLO/libreyolo) | [Dokumentasi](https://www.libreyolo.com/docs)
