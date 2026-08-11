---
title: Depth Anything V2
families:
  - depth_anything
seo_title: 'Depth Anything V2: prediksi dan validasi kedalaman monokuler'
description: >-
  Gunakan Depth Anything V2 di LibreYOLO untuk estimasi kedalaman monokuler.
  Pasang, prediksi dan validasi; Small berlisensi Apache-2.0, Base dan Large
  CC-BY-NC-4.0.
lead: >-
  Depth Anything V2 adalah encoder DINOv2 yang dipadukan dengan decoder DPT dan
  memprediksi peta inverse-depth relatif yang padat dari satu gambar. LibreYOLO
  mendukungnya untuk task depth: prediksi dan validasi zero-shot, tanpa jalur
  pelatihan.
keywords:
  - Depth Anything V2
  - estimasi kedalaman monokuler
  - monocular depth estimation
  - depth map python
  - cara membuat depth map dari foto
  - DPT
  - DINOv2
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDepthAnythingV2s-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Membaca peta kedalaman
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")

        result = model(SAMPLE_IMAGE)


        depth = result.depth_map    # DepthMap: padat (H, W), makin tinggi =
        makin dekat

        raw = depth.data                # tensor, tanpa satuan metrik atau skala
        antargambar

        normalized = depth.normalized() # diskalakan ulang ke [0, 1] untuk
        visualisasi
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDepthAnythingV2s-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDepthAnythingV2s-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDepthAnythingV2s-depth.pt format=onnx

        libreyolo export model=LibreDepthAnythingV2s-depth.pt format=tensorrt
        half=True
    - label: Memakai berkas hasil ekspor
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory memilih rute berdasarkan sufiks berkas, jadi artefak hasil
        # ekspor dimuat seperti checkpoint biasa dan mengembalikan objek Results
        # yang sama.
        model = LibreYOLO("LibreDepthAnythingV2s-depth.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.depth_map.data.shape)
source_hash: e1043aba1b70b65c
---

## Instalasi

Depth Anything V2 tidak membutuhkan extra opsional. Semua yang diimpornya sudah ada di instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali dipakai dan disimpan di cache lokal.

<code-tabs name="predict" />

`result.depth_map` membawa peta inverse-depth relatif yang padat: nilai yang
lebih tinggi berarti lebih dekat ke kamera, dan nilainya tidak punya satuan
metrik maupun skala antargambar. `save=True` menulis visualisasi peta itu dengan
colormap ke disk; `Results.plot()` tidak mencakup family ini, karena metode itu
hanya didefinisikan untuk surface normal dan tepi. Resolusi masukan harus habis
dibagi 14, yaitu grid patch DINOv2 yang menjadi dasar head DPT; LibreYOLO
memeriksanya sebelum menjalankan model dan memunculkan error bila tidak sesuai.
Lihat [prediksi](/docs/predict) untuk sumber, streaming dan penanganan hasil.

## Varian

Empat ukuran encoder, s/b/l/g, yang sesuai dengan ViT-S/B/L/G. Tabel checkpoint
di bawah hanya memuat s, b dan l; tidak ada checkpoint Giant yang
dipublikasikan. Keempatnya memakai resolusi masukan yang sama, jadi memilih
ukuran berarti menukar kapasitas encoder, bukan ukuran gambar. Lisensi juga jadi
pertimbangan: checkpoint Small berlisensi Apache-2.0, sedangkan Base dan Large
CC-BY-NC-4.0, lihat Lisensi di bawah.

Pelatihan dan fine-tuning tidak disediakan untuk family ini. `LibreDepthAnythingV2.train()`
selalu memunculkan `NotImplementedError`; sebagai gantinya, konversikan
checkpoint upstream yang kompatibel dengan
`weights/convert_depth_anything_v2_weights.py`.

## Validasi

`val()` menjalankan validator depth bersama: setiap prediksi diselaraskan dengan
ground truth-nya memakai skala dan pergeseran least-squares per gambar, lalu
metrik standar kedalaman relatif zero-shot dilaporkan, yaitu AbsRel, RMSE dan
tiga ambang batas delta.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Artefak hasil ekspor dimuat kembali lewat `LibreYOLO()` berdasarkan sufiks
berkasnya, jadi berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan
mengembalikan `Results` yang sama, dengan `depth_map` menggantikan box.
[Ekspor](/docs/export) memuat daftar argumen yang diterima setiap format.

<code-tabs name="export" />

## Checkpoint

Semua berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
