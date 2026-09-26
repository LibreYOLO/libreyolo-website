---
title: Depth Anything 3
families:
  - depth_anything3
seo_title: 'Depth Anything 3: prediksi kedalaman monokuler di LibreYOLO'
description: >-
  Gunakan Depth Anything 3 di LibreYOLO untuk estimasi kedalaman monokuler.
  Pasang, prediksi, validasi dan ekspor checkpoint DA3MONO-LARGE, Apache-2.0.
lead: >-
  Depth Anything 3 adalah transformer DINOv2 biasa yang dilatih untuk
  memprediksi kedalaman dan geometri kamera dari satu tampilan atau lebih tanpa
  spesialisasi arsitektur. LibreYOLO melakukan porting checkpoint
  DA3MONO-LARGE-nya untuk task depth: prediksi dan validasi zero-shot, tanpa
  jalur pelatihan.
keywords:
  - Depth Anything 3
  - DA3
  - estimasi kedalaman monokuler
  - monocular depth estimation
  - DINOv2
  - depth map python
  - depth estimation python
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDepthAnything3l-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Membaca peta kedalaman
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDepthAnything3l-depth.pt")

        result = model(SAMPLE_IMAGE)


        depth = result.depth_map    # DepthMap: dense (H, W), makin tinggi =
        makin dekat

        raw = depth.data                # tensor, tanpa satuan metrik atau skala
        lintas gambar

        normalized = depth.normalized() # diskalakan ulang ke [0, 1] untuk
        visualisasi
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDepthAnything3l-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnything3l-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDepthAnything3l-depth.pt format=onnx

        libreyolo export model=LibreDepthAnything3l-depth.pt format=tensorrt
        half=True
    - label: Memakai berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory memilih loader dari suffix berkas, jadi artefak hasil ekspor

        # dimuat seperti checkpoint biasa dan mengembalikan objek Results yang
        sama.

        model = LibreYOLO("LibreDepthAnything3l-depth.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.depth_map.data.shape)
source_hash: 0ac96180165c4891
---

## Instalasi

Depth Anything 3 tidak memerlukan extra opsional. Semua yang diimpornya sudah
tersedia di instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali dipakai lalu disimpan di
cache lokal.

<code-tabs name="predict" />

`result.depth_map` berisi peta inverse-depth relatif yang dense: nilai yang
lebih tinggi berarti lebih dekat ke kamera, dan nilainya tidak punya satuan
metrik maupun skala lintas gambar. Checkpoint upstream menghasilkan kedalaman
relatif positif; wrapper jaringan LibreYOLO membalikkannya dan mereproduksi
penanganan langit resmi sehingga keluarannya mengikuti kontrak kedalaman
bersama milik LibreYOLO. `save=True` menulis visualisasi peta itu dalam
colormap ke disk; `Results.plot()` tidak mencakup family ini, karena metode
tersebut hanya didefinisikan untuk surface normal dan edge. Lihat
[prediksi](/docs/predict) untuk source, streaming dan penanganan hasil.

## Varian

Satu ukuran, `l`, pada resolusi input tetap. DA3 upstream juga memublikasikan
checkpoint any-view Small dan Base, satu checkpoint metric-depth, serta
checkpoint Nested dan Giant; LibreYOLO tidak menyediakan satu pun di antaranya.
Kedalaman metrik membutuhkan kontrak publik yang berbeda dari task
relative-inverse-depth LibreYOLO, dan checkpoint any-view serta Nested
membutuhkan API kamera multi-gambar yang tidak ditawarkan LibreYOLO. Checkpoint
any-view Large dan Giant juga berlisensi CC-BY-NC-4.0 dan tidak dirujuk oleh
jalur unduhan LibreYOLO mana pun.

Pelatihan tidak tersedia untuk family ini. `LibreDepthAnything3.train()` selalu
memunculkan `NotImplementedError`; lakukan pelatihan di upstream lalu konversi
checkpoint DA3MONO-LARGE yang kompatibel dengan
`weights/convert_depth_anything3_weights.py`.

## Validasi

`val()` menjalankan validator kedalaman bersama: setiap prediksi diselaraskan
dengan ground truth-nya memakai scale dan shift least-squares per gambar, lalu
metrik relative-depth zero-shot standar dilaporkan, yaitu AbsRel, RMSE dan tiga
ambang batas delta.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Ekspor untuk family ini dibatasi pada lima format: ONNX, TorchScript,
ExecuTorch, TensorRT dan OpenVINO. Meminta format lain akan memunculkan
`NotImplementedError`, bukan mencoba konversi yang belum divalidasi. Artefak
hasil ekspor dimuat kembali lewat `LibreYOLO()` berdasarkan suffix berkasnya,
jadi berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan
mengembalikan `Results` yang sama, dengan `depth_map` menggantikan box.

<code-tabs name="export" />

## Checkpoint

Semua berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
