---
title: SwinIR
families:
  - swinir
seo_title: 'SwinIR: jalankan super-resolution gambar 4x di LibreYOLO'
description: >-
  Gunakan SwinIR di LibreYOLO untuk super-resolution gambar 4x. Instal,
  prediksi, validasi, dan ekspor checkpoint lightweight, medium, dan large.
lead: >-
  Jaringan Swin Transformer untuk restorasi gambar. LibreYOLO menyediakan
  inferensi dan validasi untuk checkpoint super-resolution 4x: generator
  lightweight resmi, real-world medium, dan real-world large.
keywords:
  - SwinIR
  - Swin Transformer
  - super resolution gambar
  - restorasi gambar
  - residual Swin Transformer block
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSwinIRm-restore.pt")
        result = model(SAMPLE_IMAGE, save=True)

        restored = result.restored
        print(restored.array.shape, restored.array.dtype)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSwinIRm-restore.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Tile untuk gambar besar
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreSwinIRl-restore.pt")


        # tile membagi forward pass menjadi tile yang saling tumpang tindih dan
        memadukan

        # sambungannya kembali; tile_pad adalah halo yang ditambahkan di sekitar
        setiap tile

        # sebelum dipotong kembali. Keduanya adalah argumen keyword khusus
        Python,

        # bukan flag CLI.

        result = model("large-photo.jpg", tile=512, tile_pad=16, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSwinIRm-restore.pt")
        metrics = model.val(data="my-restore-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSwinIRm-restore.pt data=my-restore-dataset.yaml
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreSwinIRm-restore.pt")


        # imgsz memakai patch internal kecil secara default jika dihilangkan,
        bukan

        # resolusi kerja Anda, jadi berikan ukuran yang benar-benar digunakan
        deployment

        # sebagai input model.

        model.export(format="onnx", imgsz=512)

        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSwinIRm-restore.pt format=onnx imgsz=512
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan sufiks berkas, sehingga artefak hasil
        ekspor

        # dimuat seperti checkpoint lain dan mengembalikan objek Results yang
        sama.

        model = LibreYOLO("LibreSwinIRm-restore.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.restored.array.shape)
source_hash: 87fc3d5524480eec
---

## Instalasi

SwinIR tidak memerlukan komponen tambahan opsional. Semua impornya tersedia dalam instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam cache lokal.

<code-tabs name="predict" />

Hasil restorasi tidak memiliki kotak; `result.restored` adalah gambar RGB uint8 padat
`(H, W, 3)` pada canvas 4x input di setiap dimensi. `save=True` menulis gambar tersebut
secara langsung, bukan plot beranotasi. Input diberi padding hingga kelipatan 8, bukan diubah
ukurannya, sehingga prediksi berjalan pada resolusi foto sendiri; sumber yang lebih besar
daripada kapasitas memori dapat dibagi dengan `tile` dan `tile_pad`, yang memadukan kembali
sambungan tile dalam output. Lihat [prediksi](/docs/predict) untuk sumber, streaming, dan
penanganan hasil.

## Varian

Tiga ukuran, semuanya dengan upscale tetap 4x. `s` adalah generator lightweight resmi,
dengan empat tahap residual Swin Transformer block (RSTB) dan upsampling
pixel-shuffle-direct. `m` dan `l` adalah generator real-world medium dan large, dengan
enam dan sembilan tahap RSTB serta upsampler nearest-neighbor-plus-convolution yang dibuat
untuk degradasi dunia nyata, bukan hanya downscaling bikubik.

## Validasi

`val()` mengukur PSNR dan SSIM antara output restorasi dan gambar target bersih. Keduanya
dihitung dalam RGB pada canvas asli tanpa pemotongan batas dan tanpa pengubahan ukuran.
SSIM memakai jendela Gaussian 11x11 dengan sigma 1.5, lalu dirata-ratakan pada ketiga
channel warna.

<code-tabs name="val" />

Argumen dataset adalah YAML yang memasangkan direktori gambar input terdegradasi dengan
direktori gambar target bersih beresolusi sama; lihat [format dataset](/docs/reference/dataset-formats)
untuk key yang tepat.

## Ekspor

<export-matrix />

Artefak hasil ekspor dimuat kembali melalui `LibreYOLO()` berdasarkan sufiks berkasnya,
sehingga berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan mengembalikan
`Results` yang sama. ExecuTorch dan setiap format yang ditandai diblokir dalam matriks
tidak tersedia untuk family ini; ONNX, TorchScript, TensorRT, OpenVINO, dan TFLite tersedia.
[Ekspor](/docs/export) mencantumkan argumen yang diterima setiap format beserta komponen
tambahan yang disediakan beberapa format.

<code-tabs name="export" />

## Checkpoint

Setiap berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
