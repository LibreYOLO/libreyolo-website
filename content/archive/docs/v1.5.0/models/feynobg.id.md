---
title: FeyNobg
families:
  - feynobg
seo_title: 'FeyNobg: penghapusan latar belakang di LibreYOLO'
description: >-
  Gunakan FeyNobg di LibreYOLO untuk penghapusan latar belakang dan alpha
  matting, varian BiRefNet yang diperdalam dari Feyn Inc. Instal, prediksi, dan
  validasi.
lead: >-
  Model penghapus latar belakang dari Feyn Inc. yang memperdalam arsitektur
  BiRefNet dan melatihnya kembali. LibreYOLO menyediakan inferensi dan validasi
  untuk task matte FeyNobg.
keywords:
  - FeyNobg
  - hapus background gambar
  - background removal Python
  - dichotomous image segmentation
  - alpha matte
  - image matting
  - cutout
  - nobg
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFeyNobgl-matte.pt")
        result = model(SAMPLE_IMAGE, save=True)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFeyNobgl-matte.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Cutout
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFeyNobgl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # RGBA (H, W, 4) uint8: RGB sumber ditambah matte sebagai channel alpha.
        rgba = result.cutout()
        result.save("subject.png")
  val:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreFeyNobgl-matte.pt")


        # Direktori yang berisi images/ dan direktori matte yang dideteksi
        otomatis

        # (mattes/, matte/, gt/, masks/, mask/, atau alpha/) juga dapat
        digunakan sebagai

        # pengganti YAML dataset.

        metrics = model.val(data="my-matte-dataset/")


        print(metrics["metrics/MAE"])

        print(metrics["metrics/Smeasure"])
source_hash: 45de3b578d7ebbf2
---

## Instalasi

FeyNobg tidak memerlukan extra opsional. Semua yang diimpornya tersedia dalam
instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Checkpoint diunduh dari organisasi LibreYOLO di Hugging Face saat pertama kali
digunakan dan disimpan dalam cache lokal seperti family lain, meskipun belum
tercantum dalam tabel Checkpoint di halaman ini.

<code-tabs name="predict" />

Hasil matte tidak memuat box. `result.matte` adalah array padat float32 `(H, W)`
dalam rentang `[0, 1]`, dengan 1 berarti sepenuhnya foreground dan 0 sepenuhnya
background. Tidak seperti mask biner, matte lunak mempertahankan detail tepi
anti-alias seperti rambut dan bulu. `result.cutout()` mengomposisikan gambar
sumber dengan channel alpha tersebut menjadi array RGBA, sedangkan
`result.save(path)` (atau `save=True` pada panggilan prediksi) langsung
menulisnya ke PNG berlatar transparan. Model berjalan pada kanvas native tetap
1024x1024. Resolusi lain tidak didukung karena tabel posisi relatif backbone
Swin terikat pada resolusi tersebut, dan ketidakcocokan akan menginterpolasinya
dengan buruk alih-alih memunculkan error. Lihat [prediksi](/docs/predict) untuk
sumber, streaming, dan penanganan hasil.

## Varian

Ada satu ukuran yang dipublikasikan, `l`, dengan backbone tingkat Swin-L.
FeyNobg mengambil arsitektur BiRefNet dan memperdalam tahap Swin ketiganya dari
18 menjadi 24 blok sebelum melatihnya kembali. Karena itu, port LibreYOLO
menggunakan kembali forward path, preprocessing, dan kontrak output satu logit
milik BiRefNet. Prediksi, validasi, dan penanganan checkpoint berperilaku sama
seperti family `birefnet`.

## Validasi

`val()` melaporkan dua metrik pada folder pasangan gambar/matte. Keduanya dalam
rentang `[0, 1]` dan tidak bergantung pada resolusi: MAE, yaitu mean absolute
error terhadap alpha ground truth (lebih rendah lebih baik), dan S-measure
(Fan dkk., ICCV 2017), yaitu kemiripan struktural yang menilai pelestarian bentuk
dan lubang subjek yang terlewat oleh MAE piksel saja (lebih tinggi lebih baik).
Validasi menjalankan `predict` milik model, sehingga menggunakan preprocessing
yang tepat untuk family ini.

<code-tabs name="val" />

Validasi hanya untuk inferensi. Library `nobg` upstream menyediakan kode
pelatihan Apache-2.0. Fine-tuning saat ini berarti melatih di sana dan
mengonversi hasilnya dengan skrip konversi LibreYOLO, bukan memanggil `train()`
pada family ini, yang akan memunculkan error alih-alih menjalankan trainer
parsial.

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
