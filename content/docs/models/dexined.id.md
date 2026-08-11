---
title: DexiNed
families:
  - dexined
seo_title: 'DexiNed: deteksi tepi, bawa checkpoint sendiri'
description: >-
  Gunakan DexiNed di LibreYOLO untuk prediksi peta probabilitas tepi yang padat.
  Konversikan checkpoint berlisensi, lalu prediksi, validasi dan ekspor.
lead: >-
  DexiNed (Dense Extreme Inception Network) adalah jaringan konvolusional yang
  memprediksi peta probabilitas tepi padat dari satu gambar RGB. LibreYOLO
  membungkus arsitekturnya khusus untuk deteksi tepi; tidak ada checkpoint yang
  disertakan bersama library.
keywords:
  - DexiNed
  - Dense Extreme Inception Network
  - edge detection
  - deteksi tepi gambar
  - edge detection python
  - BIPED
  - dense prediction
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")

        result = model(SAMPLE_IMAGE, save=True)


        edges = result.edges

        print(edges.array.shape)        # (H, W) float32 di [0, 1]

        print(edges.binary(0.5).sum())  # jumlah piksel tepi setelah ambang
        batas
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=weights/LibreDexiNedb-edge.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=352)

        print(metrics["metrics/ODS"])   # F-measure pada optimal dataset scale
        print(metrics["metrics/OIS"])   # F-measure pada optimal image scale
    - label: CLI
      language: bash
      code: >
        libreyolo val model=weights/LibreDexiNedb-edge.pt data=my-dataset.yaml
        imgsz=352
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        model.export(format="onnx", imgsz=352)
        model.export(format="tensorrt", imgsz=352, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=weights/LibreDexiNedb-edge.pt format=onnx
        imgsz=352

        libreyolo export model=weights/LibreDexiNedb-edge.pt format=tensorrt
        imgsz=352 half=True
    - label: Memakai berkas hasil ekspor
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.edges.array.shape)
source_hash: 342597fde3c4ba65
---

## Instalasi

DexiNed tidak membutuhkan extra opsional. Semua yang diimpornya sudah ada di
instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

LibreYOLO tidak menyertakan checkpoint DexiNed. Bobot yang dirilis resmi
dilatih pada BIPED, yang ketentuan dataset publiknya membatasi penggunaan
untuk tujuan non-komersial, jadi LibreYOLO tidak menyediakan mirror-nya.
Konversikan checkpoint yang Anda punya lisensinya dengan
`weights/convert_dexined_weights.py`, yang memeriksa key tensor terhadap
arsitektur runtime sebelum menulis berkas yang bisa dimuat LibreYOLO secara
langsung:

```bash
python weights/convert_dexined_weights.py upstream.pth weights/LibreDexiNedb-edge.pt --verify
```

<code-tabs name="predict" />

`result.edges` memuat hasilnya: array float32 `(H, W)` dengan nilai di
`[0, 1]`, dan `.binary(threshold)` mengembalikan mask tepi boolean. Tidak ada
box di sini, jadi `conf`, `iou` dan `max_det` tidak berpengaruh. Lihat
[prediksi](/docs/predict) untuk sumber, streaming dan penanganan hasil.

## Varian

DexiNed hadir dalam satu ukuran di LibreYOLO. Harness benchmark LibreYOLO
belum pernah mengukur family ini, jadi belum ada angka yang dipublikasikan
sebagai pembandingnya.

## Validasi

`val()` melaporkan F-measure ODS dan OIS bergaya BSDS terhadap dataset tepi
berpasangan: gambar berdampingan dengan peta tepi bernama dasar sama, plus mask
validitas opsional agar piksel padding tidak pernah ikut dihitung. `imgsz` harus
habis dibagi stride downsample jaringan, dan LibreYOLO memunculkan error yang
jelas bila tidak.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Ekspor tepi memakai kontrak runtime beresolusi tetap dengan batch 1: `dynamic`
dan `batch` selain 1 ditolak, dan graph hasil ekspor mengeluarkan satu peta
probabilitas yang sudah digabung. Artefak hasil ekspor dimuat kembali lewat
`LibreYOLO()` berdasarkan sufiks berkasnya, jadi berkas `.onnx` berperilaku
seperti checkpoint dan mengembalikan `Results` yang sama.

<code-tabs name="export" />

## Lisensi

<provenance-box>

LibreYOLO tidak mempublikasikan checkpoint DexiNed. Tidak ada berkas yang
disediakan sebagai mirror di bawah organisasi LibreYOLO; konversikan sendiri
checkpoint yang Anda punya lisensinya dengan
`weights/convert_dexined_weights.py`.

</provenance-box>

## Sitasi

<citation-block />
