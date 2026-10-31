---
title: SAM 2
families:
  - sam2
seo_title: 'SAM 2: segmentasi gambar dengan prompt di LibreYOLO'
description: >-
  Gunakan SAM 2 di LibreYOLO untuk segmentasi dengan prompt titik dan kotak.
  Instal dan prediksi dengan checkpoint tiny, small, base-plus, dan large
  berlisensi Apache-2.0.
lead: >-
  SAM 2 memperluas SAM dengan arsitektur memori streaming yang dibuat untuk
  video, serta mengubah klik titik atau kotak menjadi mask objek. LibreYOLO
  mendukung jalur segmentasi gambarnya melalui factory LibreSAM khusus yang
  terpisah dari factory detektor LibreYOLO().
keywords:
  - SAM 2
  - Segment Anything
  - promptable segmentation
  - segmentasi interaktif
  - point prompt
  - box prompt
  - Meta AI
  - Hiera
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompt titik dan kotak
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # Alias ukuran: "sam2-tiny", "sam2-small", "sam2-base-plus",

        # "sam2-large" (juga bentuk pendek
        "sam2-t"/"sam2-s"/"sam2-bp"/"sam2-l").

        model = LibreSAM("sam2-large")


        # Prompt titik: [x, y] dalam koordinat piksel, label 1 = foreground.

        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])

        print(result.masks.xy)      # poligon per mask

        print(result.boxes.xyxy)    # kotak rapat yang diturunkan dari mask


        # Prompt kotak sebagai pengganti titik.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])


        # Tanpa prompt akan menyegmentasikan seluruh gambar (generator mask
        otomatis

        # yang disederhanakan, bukan versi referensi lengkap).

        result = model.predict(SAMPLE_IMAGE)
    - label: 'Enkode sekali, gunakan banyak prompt'
      language: python
      code: >
        from libreyolo import LibreSAM2, SAMPLE_IMAGE


        # Kelas khusus family menerima ukuran tanpa prefiks "sam2-".

        model = LibreSAM2("large")


        # Encoder gambar adalah bagian yang mahal. set_image() menjalankannya
        sekali;

        # setiap pemanggilan predict() setelahnya menggunakan kembali embedding
        dari cache.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[640, 420], labels=[1])

        b = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
source_hash: 2a3090d7ecd533b0
---

## Instalasi

SAM 2 memerlukan komponen tambahan `sam`, yang memasang `transformers` dan `timm`.

```bash
pip install "libreyolo[sam]"
```

## Prediksi

`LibreSAM(...)` (atau `LibreSAM2(...)` yang khusus untuk family ini) adalah entry point
terpisah dari `LibreYOLO(...)`: hasilnya merupakan segmenter yang dapat diberi prompt,
bukan detektor, karena forward pass di sini tidak bermakna tanpa prompt spasial. Tidak
ada perintah CLI `libreyolo predict` untuk family ini; gunakan API Python. Hanya
segmentasi gambar yang didukung; pelacakan dengan memori video SAM 2 berada di luar cakupan.

<code-tabs name="predict" />

Prompt titik menerima `[x, y]` untuk satu objek, `[[x, y], ...]` untuk beberapa objek,
atau array numpy; `labels` menandai setiap titik sebagai `1` (foreground) atau `0`
(background), dengan default semuanya foreground. Prompt kotak menerima
`[x1, y1, x2, y2]` atau daftar kotak, satu mask per kotak. Menghilangkan kedua prompt
akan menyegmentasikan seluruh gambar dengan memberi prompt pada grid padat dan mempertahankan
mask yang meyakinkan serta tidak tumpang tindih; mode "segment everything" ini disederhanakan
dibandingkan generator mask otomatis referensi dan dapat kurang menyegmentasikan adegan padat,
sehingga prompt titik atau kotak yang nyata menjadi jalur presisi. `conf` memfilter
berdasarkan kualitas mask yang diprediksi (IoU), bukan confidence deteksi: berikan `0.0`
untuk mempertahankan setiap kandidat. `multimask=True` mengembalikan ketiga mask ambiguitas
keseluruhan-lawan-bagian milik SAM per prompt, bukan hanya mask terbaik. `device=`
memindahkan model dan, jika sesi `set_image()` aktif, embedding dari cache. Setiap mask
memiliki id kelas `0` dengan nama `"object"`, karena mask berbasis prompt tidak memiliki
kumpulan kelas tetap. `train()`, `val()`, `export()`, dan `track()` semuanya memunculkan
`NotImplementedError` untuk family ini: LibreYOLO hanya mendukung inferensi gambar di sini.
Lihat [prediksi](/docs/predict) untuk jenis sumber.

## Varian

Empat ukuran backbone Hiera: tiny, small, base-plus, dan large, semuanya pada resolusi
input yang sama. Belum ada benchmark akurasi atau latensi yang dipublikasikan untuk family
ini, sehingga pemilihan ukuran menukar bobot encoder dengan kualitas mask secara langsung:
tiny paling cepat untuk enkode, sedangkan large paling berat.

## Checkpoint

Setiap berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
