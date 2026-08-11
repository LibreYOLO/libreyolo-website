---
title: Real-ESRGAN
families:
  - realesrgan
seo_title: 'Real-ESRGAN: super-resolution gambar di LibreYOLO'
description: >-
  Gunakan Real-ESRGAN di LibreYOLO untuk super-resolution gambar praktis pada
  4x, 2x, dan tier 4x cepat. Instal, prediksi, validasi, dan ekspor.
lead: >-
  Upscaler blind super-resolution praktis yang dilatih pada degradasi sintetis,
  bukan hanya downscaling bikubik. LibreYOLO menyediakan inferensi dan validasi
  untuk checkpoint 4x, 2x, dan 4x cepatnya.
keywords:
  - Real-ESRGAN
  - RRDBNet
  - SRVGGNetCompact
  - super resolution gambar
  - restorasi gambar
  - blind super-resolution
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")
        result = model(SAMPLE_IMAGE, save=True)

        restored = result.restored
        print(restored.array.shape, restored.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRealESRGANx4-restore.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Tile untuk gambar besar
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRealESRGANx4-restore.pt")


        # tile membagi forward pass menjadi tile yang saling tumpang tindih dan
        memadukan

        # sambungannya kembali; tile_pad adalah halo yang ditambahkan di sekitar
        setiap tile

        # sebelum dipotong kembali. Keduanya adalah argumen keyword khusus
        Python,

        # bukan flag CLI.

        result = model("large-photo.jpg", tile=512, tile_pad=10, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRealESRGANx4-restore.pt")
        metrics = model.val(data="my-restore-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: >
        libreyolo val model=LibreRealESRGANx4-restore.pt
        data=my-restore-dataset.yaml
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRealESRGANx4-restore.pt")


        # imgsz memakai patch internal kecil secara default jika dihilangkan,
        bukan

        # resolusi kerja Anda, jadi berikan ukuran yang benar-benar digunakan
        deployment

        # sebagai input model.

        model.export(format="onnx", imgsz=512)

        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRealESRGANx4-restore.pt format=onnx
        imgsz=512
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan sufiks berkas, sehingga artefak hasil
        ekspor

        # dimuat seperti checkpoint lain dan mengembalikan objek Results yang
        sama.

        model = LibreYOLO("LibreRealESRGANx4-restore.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.restored.array.shape)
source_hash: f0efb4f65d38e22d
---

## Instalasi

Real-ESRGAN tidak memerlukan komponen tambahan opsional. Semua impornya tersedia dalam
instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam cache lokal.

<code-tabs name="predict" />

Hasil restorasi tidak memiliki kotak; `result.restored` adalah gambar RGB uint8
padat `(H, W, 3)` pada canvas yang ukurannya `Results.restore_scale` kali input di
setiap dimensi. `save=True` menulis gambar tersebut secara langsung, bukan plot
beranotasi. Input dikonversi ke RGB dan channel alfa akan dibuang. Sumber yang
lebih besar daripada kapasitas memori dapat dibagi dengan `tile` dan `tile_pad`,
yang memadukan kembali sambungan tile dalam output. Lihat [prediksi](/docs/predict)
untuk sumber, streaming, dan penanganan hasil.

## Varian

Tiga checkpoint yang dinamai berdasarkan faktor upscale. `x4` adalah RRDBNet
(`RealESRGAN_x4plus`), dengan 23 residual-in-residual dense block, dan menjadi default
berkualitas tinggi pada 4x. `x2` memakai arsitektur RRDBNet yang sama pada 2x. `x4t`
adalah SRVGGNetCompact (`realesr-general-x4v3`), generator yang lebih kecil dan cepat,
dibuat untuk video serta penggunaan berlatensi lebih rendah pada 4x. Model upstream
serbaguna juga menyediakan jaringan kekuatan denoise berpasangan yang dipadukan saat
inferensi; kontrol kekuatan tersebut bukan bagian dari port ini, yang menjalankan
generator dasar `x4t`.

## Validasi

`val()` mengukur PSNR dan SSIM antara output restorasi dan gambar target bersih.
Keduanya dihitung dalam RGB pada canvas asli tanpa pemotongan batas dan tanpa pengubahan
ukuran. SSIM memakai jendela Gaussian 11x11 dengan sigma 1.5, lalu dirata-ratakan pada
ketiga channel warna.

<code-tabs name="val" />

Argumen dataset adalah YAML yang memasangkan direktori gambar input terdegradasi dengan
direktori gambar target bersih beresolusi sama; lihat [format dataset](/docs/reference/dataset-formats)
untuk key yang tepat.

## Ekspor

<export-matrix />

Artefak hasil ekspor dimuat kembali melalui `LibreYOLO()` berdasarkan sufiks berkasnya,
sehingga berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan mengembalikan
`Results` yang sama. [Ekspor](/docs/export) mencantumkan argumen yang diterima setiap
format beserta komponen tambahan yang disediakan beberapa format.

<code-tabs name="export" />

## Checkpoint

Setiap berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
