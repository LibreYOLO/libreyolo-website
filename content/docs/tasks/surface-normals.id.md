---
title: Surface normal
seo_title: Estimasi surface normal di LibreYOLO
description: >-
  Prediksi field surface-normal padat dari satu gambar di LibreYOLO. Baca
  konvensi frame kamera, validasi angular error, dan ekspor model.
lead: >-
  Estimasi surface-normal memprediksi arah yang dihadapi setiap permukaan
  terlihat. LibreYOLO menyediakannya sebagai task normal, yang mengembalikan
  field padat vektor satuan pada canvas gambar asli.
keywords:
  - estimasi surface normal python
  - normal map dari gambar
  - geometri monokular
  - metrik angular error
  - prediksi normal padat
last_verified: 1.5.0
snippets:
  predict:
    - label: Prediksi field normal
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreMoGe2s-normal.pt")

        result = model(SAMPLE_IMAGE, save=True)


        normals = result.normal_map

        print(normals.data.shape)      # vektor satuan float32 (H, W, 3)

        normals.assert_normalized()    # error jika ada piksel yang panjangnya
        bukan satu
    - label: Baca satu piksel
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreMoGe2s-normal.pt")

        result = model(SAMPLE_IMAGE)


        # Frame kamera OpenCV: +x ke kanan, +y ke bawah, +z masuk ke scene.
        Permukaan

        # yang menghadap kamera terbaca mendekati (0, 0, -1).

        field = result.normals.data

        h, w = field.shape[:2]

        print(field[h // 2, w // 2])
    - label: Simpan visualisasi
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE)

        # plot() merender field; tersedia untuk hasil normal dan edge.
        result.plot().save("normals.png")
  val:
    - label: Validasi dan baca key metrik
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=518)

        print(metrics["metrics/mean_angular_error"])     # derajat
        print(metrics["metrics/median_angular_error"])   # derajat
        print(metrics["metrics/within_11_25"])           # persen piksel
        print(metrics["metrics/within_22_5"], metrics["metrics/within_30"])
  export:
    - label: Ekspor
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        model.export(format="onnx", imgsz=518)
    - label: Jalankan file hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory mengarahkan berdasarkan suffix file, sehingga artefak hasil
        ekspor dimuat

        # seperti checkpoint dan mengembalikan objek Results yang sama.

        model = LibreYOLO("LibreMoGe2s-normal.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.normal_map.data.shape)
source_hash: d26d26d894b436ff
---

## Definisi

Task `normal` memprediksi vektor satuan tiga komponen per piksel dari satu gambar
RGB, yaitu arah yang dihadapi permukaan pada piksel. Berbeda dari depth, output
tidak memiliki skala bebas, sehingga dua prediksi dapat langsung dibandingkan
tanpa alignment.

Prediksi mengisi `result.normal_map`, payload `NormalMap` berisi array float32
`(H, W, 3)` pada canvas gambar asli, yang juga dapat diakses sebagai
`result.normals`. Vektor menggunakan frame kamera OpenCV LibreYOLO, dengan `+x`
ke kanan, `+y` ke bawah, dan `+z` masuk ke scene, serta menghadap kamera,
sehingga permukaan fronto-parallel terbaca `(0, 0, -1)`. `.assert_normalized()`
memeriksa setiap piksel terbatas dan panjangnya satu dalam toleransi.
`result.boxes` tetap kosong, sehingga `conf`, `iou`, dan `max_det` tidak
berpengaruh, serta `Results.plot()` mencakup task ini.

## Model

Dua family melayani `normal`.

[MoGe-2](/docs/models/moge-2) adalah model khusus: model geometri monokular
satu-forward dalam tiga ukuran encoder. LibreYOLO tidak menyalin checkpoint ini
ke organisasinya; pemuatan mengunduh ukuran cocok dari repository resmi pada
revisi yang di-pin dan memverifikasinya terhadap SHA-256 tercatat.

[LibreMODUS](/docs/models/libremodus) menghasilkan normal sebagai salah satu
target model any-to-any dan dapat menerima depth map, bukan gambar RGB, sebagai
input. Model memerlukan ekstra `modus` dan akun Hugging Face terautentikasi,
serta tidak menawarkan `val()` maupun `export()`, sehingga tidak termasuk bagian
validasi dan ekspor di bawah.

## Predict

Bobot MoGe-2 diunduh saat penggunaan pertama dan di-cache secara lokal.

<code-tabs name="predict" />

`imgsz` harus dapat dibagi patch size encoder ViT, yang diperiksa LibreYOLO
sebelum run dimulai. Prediksi list gambar menjalankan satu forward pass per
gambar; task ini tidak memiliki jalur cepat batch bertumpuk. Lihat
[prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Format dataset

Validasi normal memasangkan setiap gambar dengan PNG 16-bit tiga channel dengan
stem dan resolusi sama, ditambah validity mask opsional.

```text
dataset/
  data.yaml
  images/
    val/room.jpg
  normals/
    val/room.png
  masks/
    val/room.png
```

```yaml
path: dataset
train: images/train
val: images/val
normals_dir: normals
masks_dir: masks
nc: 1
names: {0: normal}
```

PNG target harus tepat berupa `uint16` tiga channel yang disimpan sebagai RGB.
Decode menggunakan `n = png / 65535 * 2 - 1`, lalu setiap vektor dinormalisasi
ulang, dan vektor hasil decode menggunakan frame kamera OpenCV yang sama dengan
prediksi. Piksel mask valid jika bukan nol; tanpa file mask, setiap vektor
hasil decode terbatas dan bukan nol dianggap valid. Piksel target tidak valid
dan padding disimpan sebagai `(0, 0, 0)` dan tidak pernah berkontribusi pada
metrik. Lihat [format dataset](/docs/reference/dataset-formats) untuk kontrak
lengkap.

## Train

Kedua family normal tidak memiliki implementasi pelatihan: `train()` memunculkan
`NotImplementedError`. Halaman MoGe-2 menunjuk ke checkpoint resmi yang di-pin
untuk predict, validate, dan export.

## Validate

`val()` mengukur sudut antara setiap vektor prediksi dan ground truth pada
piksel yang ditandai valid oleh dataset.

<code-tabs name="val" />

`metrics/mean_angular_error` dan `metrics/median_angular_error` adalah sudut
dalam derajat, dan nilai lebih rendah lebih baik. `metrics/within_11_25`,
`metrics/within_22_5`, dan `metrics/within_30` adalah persentase piksel valid
dengan angular error dalam 11.25, 22.5, dan 30 derajat, sehingga nilai lebih
tinggi lebih baik. Ketiganya berupa persentase, bukan fraksi. `fitness` adalah
`metrics/within_11_25` dibagi 100, sehingga pemilihan checkpoint terbaik
menggunakan skala `[0, 1]` yang sama dengan task lain.

## Export

Model normal hasil ekspor dimuat kembali melalui `LibreYOLO()` berdasarkan
suffix file, sehingga `.onnx` berperilaku seperti checkpoint dan mengembalikan
`Results` yang sama.

<code-tabs name="export" />

Ekspor normal menggunakan kontrak runtime resolusi tetap, batch 1: `dynamic`
dan `batch` selain 1 ditolak, serta `imgsz` harus dapat dibagi patch size
encoder. Cakupan per format tersedia pada [halaman MoGe-2](/docs/models/moge-2)
dan [matriks ekspor lengkap](/docs/reference/export-matrix).
[Ekspor](/docs/export) mencantumkan argumen setiap format.
