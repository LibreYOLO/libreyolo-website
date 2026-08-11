---
title: MoGe-2
families:
  - moge2
seo_title: 'MoGe-2: prediksi, validasi, dan ekspor surface normal'
description: >-
  Gunakan MoGe-2 di LibreYOLO untuk prediksi surface normal padat. Instal,
  prediksi, validasi, dan ekspor checkpoint resmi ViT-S, ViT-B, dan ViT-L.
lead: >-
  MoGe-2 adalah model geometri monokular satu forward pass yang memprediksi
  field surface normal padat dari satu gambar RGB. LibreYOLO hanya mendukungnya
  untuk estimasi normal melalui checkpoint resmi ViT-S, ViT-B, dan ViT-L.
keywords:
  - MoGe-2
  - MoGe 2
  - estimasi surface normal
  - monocular geometry
  - normal map
  - dense prediction
  - DINOv2
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        result = model(SAMPLE_IMAGE, save=True)

        normal = result.normal_map
        print(normal.array.shape)   # unit vector float32 (H, W, 3)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMoGe2s-normal.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=518)

        print(metrics["metrics/mean_angular_error"])   # derajat
        print(metrics["metrics/median_angular_error"])
        print(metrics["metrics/within_11_25"])          # persentase piksel
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMoGe2s-normal.pt data=my-dataset.yaml imgsz=518
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMoGe2s-normal.pt")
        model.export(format="onnx", imgsz=518)
        model.export(format="tensorrt", imgsz=518, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreMoGe2s-normal.pt format=onnx imgsz=518

        libreyolo export model=LibreMoGe2s-normal.pt format=tensorrt imgsz=518
        half=True
    - label: Gunakan berkas hasil ekspor
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreMoGe2s-normal.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.normal_map.array.shape)
source_hash: ddfacf6b7e9729f6
---

## Instalasi

MoGe-2 tidak memerlukan extra opsional. Semua yang diimpornya tersedia dalam
instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh otomatis saat pertama kali digunakan. LibreYOLO mengambil ukuran
yang cocok langsung dari checkpoint resmi dan menyimpannya dalam cache lokal.

<code-tabs name="predict" />

MoGe-2 mengembalikan field padat, bukan sekumpulan deteksi, sehingga
`result.boxes` kosong dan `conf`, `iou`, serta `max_det` tidak berpengaruh.
`result.normal_map` memuat hasilnya, yaitu array unit vector `(H, W, 3)` dalam
frame kamera OpenCV. `+x` mengarah ke kanan, `+y` ke bawah, `+z` masuk ke dalam
adegan, dan permukaan yang menghadap kamera terbaca sebagai `(0, 0, -1)`.
Memprediksi daftar gambar menjalankan satu forward pass per gambar. Family ini
tidak memiliki jalur cepat batch bertumpuk. Lihat [prediksi](/docs/predict)
untuk sumber, streaming, dan penanganan hasil.

## Varian

Ada tiga ukuran encoder yang disediakan sebagai checkpoint terpisah: ViT-S,
ViT-B, dan ViT-L, semuanya pada resolusi input yang sama. Harness benchmark
LibreYOLO belum mengukur family ini, sehingga belum ada angka akurasi yang
dipublikasikan sebagai pembanding. Pilih ukuran sesuai anggaran komputasi Anda.

## Validasi

`val()` mengukur error sudut terhadap dataset normal map berpasangan: gambar
berdampingan dengan PNG normal 16-bit bernama stem sama, dengan mask validitas
opsional agar piksel padding dan tidak valid tidak pernah dihitung. Fungsi ini
mengembalikan error sudut mean dan median dalam derajat, serta persentase piksel
dalam batas 11.25, 22.5, dan 30 derajat.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Ekspor normal memakai kontrak runtime resolusi tetap dengan batch 1. `dynamic`
dan `batch` selain 1 ditolak, serta `imgsz` harus habis dibagi ukuran patch
encoder ViT, yang diperiksa LibreYOLO sebelum run dimulai. Artefak hasil ekspor
dapat dimuat kembali melalui `LibreYOLO()` berdasarkan akhiran berkasnya,
sehingga berkas `.onnx` berperilaku seperti checkpoint dan mengembalikan
`Results` yang sama.

<code-tabs name="export" />

## Lisensi

<provenance-box>

LibreYOLO tidak menyalin checkpoint ini ke organisasinya sendiri.
`LibreYOLO("LibreMoGe2s-normal.pt")` mengunduh ukuran yang cocok langsung dari
repository Hugging Face resmi pada revisi yang dipatok dan memverifikasi berkas
terhadap checksum SHA-256 yang direkam sebelum digunakan.

</provenance-box>

## Sitasi

<citation-block />
