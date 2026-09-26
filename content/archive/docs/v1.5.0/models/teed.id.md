---
title: TEED
families:
  - teed
seo_title: 'TEED: deteksi tepi, gunakan checkpoint sendiri'
description: >-
  Gunakan TEED di LibreYOLO untuk prediksi probabilitas tepi padat. Konversi
  checkpoint yang berlisensi, lalu prediksi, validasi, dan ekspor.
lead: >-
  TEED (Tiny and Efficient Edge Detector) adalah jaringan konvolusional kecil
  yang memprediksi peta probabilitas tepi padat dari satu gambar RGB. LibreYOLO
  membungkus arsitekturnya hanya untuk deteksi tepi; tidak ada checkpoint yang
  disertakan bersama library.
keywords:
  - TEED
  - Tiny and Efficient Edge Detector
  - deteksi tepi
  - BIPED
  - dense prediction
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("weights/LibreTEEDt-edge.pt")

        result = model(SAMPLE_IMAGE, save=True)


        edges = result.edges

        print(edges.array.shape)        # (H, W) float32 dalam [0, 1]

        print(edges.binary(0.5).sum())  # jumlah piksel tepi setelah penerapan
        ambang
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=weights/LibreTEEDt-edge.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreTEEDt-edge.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=352)

        print(metrics["metrics/ODS"])   # F-measure optimal pada skala dataset
        print(metrics["metrics/OIS"])   # F-measure optimal pada skala gambar
    - label: CLI
      language: bash
      code: >
        libreyolo val model=weights/LibreTEEDt-edge.pt data=my-dataset.yaml
        imgsz=352
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreTEEDt-edge.pt")
        model.export(format="onnx", imgsz=352)
        model.export(format="tensorrt", imgsz=352, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=weights/LibreTEEDt-edge.pt format=onnx imgsz=352

        libreyolo export model=weights/LibreTEEDt-edge.pt format=tensorrt
        imgsz=352 half=True
    - label: Gunakan berkas hasil ekspor
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreTEEDt-edge.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.edges.array.shape)
source_hash: c7203b254e460258
---

## Instalasi

TEED tidak memerlukan komponen tambahan opsional. Semua impornya tersedia dalam instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

LibreYOLO tidak menyediakan checkpoint TEED. Bobot yang dirilis secara resmi dilatih pada
BIPED, yang ketentuan dataset-nya membatasi penggunaan hanya untuk tujuan nonkomersial,
sehingga LibreYOLO tidak mencerminkannya. Konversi checkpoint yang berhak digunakan dengan
`weights/convert_teed_weights.py`, yang memeriksa key tensor terhadap arsitektur runtime
sebelum menulis berkas yang dapat dimuat LibreYOLO secara langsung:

```bash
python weights/convert_teed_weights.py upstream.pth weights/LibreTEEDt-edge.pt --verify
```

<code-tabs name="predict" />

`result.edges` menyimpan hasil: array float32 `(H, W)` dalam `[0, 1]`, dengan
`.binary(threshold)` mengembalikan mask tepi boolean. Tidak ada kotak, sehingga `conf`,
`iou`, dan `max_det` tidak berpengaruh. Lihat [prediksi](/docs/predict) untuk sumber,
streaming, dan penanganan hasil.

## Varian

TEED menyediakan satu ukuran di LibreYOLO. Harness benchmark LibreYOLO belum mengukur
family ini, sehingga tidak ada angka terbitan yang dapat dipakai untuk perbandingan.

## Validasi

`val()` melaporkan F-measure ODS dan OIS bergaya BSDS terhadap dataset tepi berpasangan:
gambar di samping peta tepi dengan stem sama, serta mask validitas opsional agar piksel
padding tidak pernah dihitung. `imgsz` harus habis dibagi stride downsample jaringan, dan
LibreYOLO memunculkan error yang jelas jika tidak.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Ekspor tepi memakai kontrak runtime resolusi tetap dengan batch 1: `dynamic` dan `batch`
selain 1 ditolak, sedangkan graph hasil ekspor mengeluarkan satu peta probabilitas gabungan.
Artefak hasil ekspor dimuat kembali melalui `LibreYOLO()` berdasarkan sufiks berkasnya,
sehingga berkas `.onnx` berperilaku seperti checkpoint dan mengembalikan `Results` yang sama.

<code-tabs name="export" />

## Lisensi

<provenance-box>

LibreYOLO tidak menerbitkan checkpoint TEED. Tidak ada yang dicerminkan di bawah organisasi
LibreYOLO; sebagai gantinya, konversi checkpoint yang Anda miliki lisensinya dengan
`weights/convert_teed_weights.py`.

</provenance-box>

## Sitasi

<citation-block />
