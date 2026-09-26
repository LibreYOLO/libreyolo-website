---
title: ZipDepth
families:
  - zipdepth
seo_title: 'ZipDepth: kedalaman monokular ringan di LibreYOLO'
description: >-
  Gunakan ZipDepth di LibreYOLO untuk estimasi kedalaman monokular ringan.
  Instal, prediksi, validasi, dan ekspor dua checkpoint berlisensi MIT.
lead: >-
  ZipDepth adalah CNN reparameterizable ringkas yang didistilasi dari Depth
  Anything V2 Large dan memprediksi peta inverse-depth relatif yang padat.
  LibreYOLO mendukungnya untuk task kedalaman: prediksi dan validasi zero-shot,
  tanpa jalur pelatihan.
keywords:
  - ZipDepth
  - estimasi kedalaman monokular
  - edge depth model
  - kedalaman relatif
  - depth map
  - reparameterizable CNN
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        result = model(SAMPLE_IMAGE, save=True)

        depth = result.depth_map
        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreZipDepthb-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Checkpoint NPU/edge
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Encoder sama, dengan head upsampling tanpa unfold untuk compiler yang
        tidak

        # mendukung gather/unfold. Output secara visual setara dengan checkpoint
        b.

        model = LibreYOLO("LibreZipDepthbnpu-depth.pt")

        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreZipDepthb-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreZipDepthb-depth.pt")
        model.export(format="onnx")
        model.export(format="ncnn")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreZipDepthb-depth.pt format=onnx
        libreyolo export model=LibreZipDepthbnpu-depth.pt format=ncnn
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan sufiks berkas, sehingga artefak hasil
        ekspor

        # dimuat seperti checkpoint lain dan mengembalikan objek Results yang
        sama.

        model = LibreYOLO("LibreZipDepthb-depth.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.depth_map.data.shape)
source_hash: 891eaa1a42795a4c
---

## Instalasi

ZipDepth tidak memerlukan komponen tambahan opsional. Semua impornya tersedia dalam instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam cache lokal.

<code-tabs name="predict" />

`result.depth_map` menyimpan peta inverse-depth relatif yang padat: nilai lebih tinggi berarti
lebih dekat ke kamera, dan nilainya tidak memiliki satuan metrik atau skala lintas gambar.
`save=True` menulis visualisasi dengan colormap dari peta tersebut ke disk; `Results.plot()`
tidak mencakup family ini karena didefinisikan hanya untuk surface normal dan tepi. Lihat
[prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Varian

Dua checkpoint dengan kapasitas encoder sama, yang hanya berbeda pada head upsampling hasil
pelatihan. `b` memakai convex upsampling dan berjalan pada GPU atau CPU. `bnpu` menggantinya
dengan decoder tanpa unfold untuk compiler NPU dan edge yang tidak mendukung gather/unfold;
outputnya didokumentasikan setara secara visual dengan `b`. Pilih `bnpu` jika target ekspor
adalah runtime terbatas, atau `b` untuk lainnya.

Kedua checkpoint didistilasi dari pseudo-label Depth Anything V2 Large, sehingga family ini
adalah tier ringkas berorientasi edge untuk task kedalaman LibreYOLO, berdampingan dengan
encoder Depth Anything V2 yang lebih besar.

Pelatihan tidak ditawarkan untuk family ini. `LibreZipDepth.train()` selalu memunculkan
`NotImplementedError`: resep upstream mendistilasi pseudo-label pada kumpulan gambar besar
yang tidak dapat direproduksi sebagai proses pelatihan LibreYOLO. Lakukan pelatihan melalui
upstream di [fabiotosi92/ZipDepth](https://github.com/fabiotosi92/ZipDepth) dan konversi
hasilnya dengan `weights/convert_zipdepth_weights.py`.

## Validasi

`val()` menjalankan validator kedalaman bersama: metode ini menyelaraskan setiap prediksi
dengan ground truth-nya memakai scale dan shift least-squares per gambar, lalu melaporkan
metrik kedalaman relatif zero-shot standar, yaitu AbsRel, RMSE, dan tiga ambang delta.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Ekspor mengikuti kontrak padat beresolusi tetap: ukuran gambar sumber diregangkan ke canvas
hasil ekspor, lalu peta kedalaman yang dikembalikan diubah kembali ke canvas asli. Artefak
hasil ekspor dimuat kembali melalui `LibreYOLO()` berdasarkan sufiks berkasnya, sehingga
berkas `.onnx` atau `.ncnn` berperilaku seperti checkpoint dan mengembalikan `Results` yang
sama, dengan `depth_map` sebagai pengganti kotak.

<code-tabs name="export" />

## Checkpoint

Setiap berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
