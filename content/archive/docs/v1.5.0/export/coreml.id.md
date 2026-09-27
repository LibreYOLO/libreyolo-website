---
title: Core ML
seo_title: Ekspor ke Core ML dari LibreYOLO
description: >-
  Mengekspor detektor LibreYOLO ke .mlpackage Core ML: kontrak input ImageType,
  FP16, compute unit, NMS tertanam, dan empat family yang didukung.
lead: >-
  Core ML adalah format model on-device milik Apple. LibreYOLO melakukan tracing
  pada detektor di balik wrapper preprocessing per family, sehingga graph hasil
  konversi selalu menerima input gambar RGB yang kanonis, lalu menulis
  .mlpackage ML Program dengan metadata model terlampir.
keywords:
  - export yolo ke coreml
  - mlpackage
  - coremltools
  - ct.ImageType
  - apple neural engine
  - compute_units
  - coreml nms pipeline
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="coreml")
    mono: true
  - label: Menulis
    value: Satu bundle .mlpackage (sebuah direktori) dalam format ML Program
  - label: Tambahan
    value: 'pip install "libreyolo[coreml]"'
    mono: true
  - label: Dimuat kembali
    value: LibreYOLO("weights/LibreYOLO9t.mlpackage") on macOS
    mono: true
  - label: Bentuk
    value: Tetap. Input berupa ct.ImageType dengan bentuk terkunci.
  - label: Presisi
    value: 'FP32, FP16 (half=True). Tidak ada INT8.'
  - label: Family
    value: 'Hanya deteksi, untuk yolox, yolo9, rtdetr dan rfdetr'
verification: >-
  Dibaca dari libreyolo/export/coreml.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/coreml.py dan pyproject.toml
  di branch dev.
snippets:
  install:
    - label: Instalasi
      language: bash
      code: |
        pip install "libreyolo[coreml]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Menulis bundle weights/LibreYOLO9t.mlpackage
        path = model.export(format="coreml")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreml
    - label: Argumen
      language: python
      code: >
        model.export(
            format="coreml",
            imgsz=640,
            batch=1,
            half=False,           # True mengonversi dengan presisi komputasi FLOAT16
            compute_units="all",  # all | cpu_and_gpu | cpu_and_ne | cpu_only
            output_path=None,     # None menulis weights/<stem>.mlpackage
        )


        # dynamic diterima, tetapi input berupa ct.ImageType dengan bentuk
        tetap,

        # dan metadata yang tertanam tetap mencatat dynamic=False.
  nms:
    - label: Menanamkan layer NMS milik Apple
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Hanya deteksi YOLOX dan YOLO9, batch 1.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="coreml",
            nms=True,
            conf=0.25,
            iou=0.45,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreml --nms \
          --conf 0.25 --iou 0.45
  run:
    - label: 'Lewat LibreYOLO, di macOS'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO(
            "weights/LibreYOLO9t.mlpackage",
            compute_units="all",   # atau cpu_and_ne untuk mengunci Neural Engine
        )
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Langsung dengan coremltools
      language: python
      code: >
        import coremltools as ct

        from PIL import Image


        mlmodel = ct.models.MLModel("weights/LibreYOLO9t.mlpackage")

        print(mlmodel.user_defined_metadata["model_family"])

        print(mlmodel.user_defined_metadata["names"])


        # Input berupa gambar bernama "image" dengan ukuran ekspor yang tetap.

        image = Image.open(SAMPLE_IMAGE).convert("RGB").resize((640, 640))

        out = mlmodel.predict({"image": image})

        print({name: value.shape for name, value in out.items()})


        # Letterboxing dan postprocessing menjadi tanggung jawab Anda di jalur
        ini.
  support:
    - label: Memeriksa satu family dan task sebelum mengekspor
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 09c5394e3837eca2
---

## Instalasi

<code-tabs name="install" />

Prediksi membutuhkan macOS. `LibreYOLO()` menolak `.mlpackage` di platform lain
dengan pesan yang menyebutkan platform saat ini, dan matriks dukungan mencatat
kombinasi ini sebagai tersedia dengan alasan bahwa paritas runtime membutuhkan runner macOS.

## Ekspor

<code-tabs name="export" />

Bundle ditulis ke `weights/` memakai stem checkpoint, dengan `_fp16` ditambahkan
saat `half=True`. Sebuah `.mlpackage` adalah direktori, jadi salin seluruh isinya.

Setiap family melalui tracing di balik wrapper preprocessing, sehingga graph hasil
konversi menerima satu input kanonis: RGB, `scale=1/255`, tanpa bias, dideklarasikan
sebagai `ct.ImageType`. Wrapper itu menyerap konvensi milik masing-masing family,
yaitu BGR dalam rentang 0 sampai 255 untuk YOLOX, mean dan standar deviasi ImageNet
untuk RF-DETR, serta identitas untuk YOLO9 dan RT-DETR. Karena itu konsumen Core ML
cukup memberikan gambar biasa, bukan tensor khusus per family.

Konversi menargetkan ML Program dengan deployment target minimum iOS 15.
`compute_units` disimpan pada model hasil konversi dan dapat ditimpa lagi saat
artefak dimuat.

Metadata model masuk ke `user_defined_metadata` sebagai string, dan dari sanalah
backend membaca family, task, nama kelas, ukuran input dan skema pose.

### NMS tertanam

<code-tabs name="nms" />

`nms=True` membungkus model dalam pipeline Core ML yang diakhiri layer
`NonMaximumSuppression` milik Apple. Hasilnya punya dua output: `confidence`,
berbentuk `N` kali jumlah kelas, dan `coordinates`, berbentuk `N` kali 4 sebagai
`xywh` ternormalisasi.

Ini hanya berlaku untuk deteksi YOLOX dan YOLO9, dan mensyaratkan batch 1. Family
bergaya DETR ditolak berdasarkan nama, karena set prediction mengambil top-k atas
query dan kelas tanpa langkah IoU sehingga tidak bisa memakai layer tersebut.
`max_det` juga tidak diekspos di sini; kalau batas jumlah deteksi penting, pakai
[NMS tertanam ONNX](/docs/export/onnx).

## Menjalankan artefak

<code-tabs name="run" />

`LibreYOLO()` mengenali direktori berakhiran `.mlpackage` dan mengembalikan objek
`Results` yang sama seperti checkpoint. `compute_units` adalah satu-satunya argumen
yang diteruskan factory untuk format ini, dan nilainya bisa `all`, `cpu_and_gpu`,
`cpu_and_ne` dan `cpu_only`. Argumen `device` diabaikan, karena Core ML mengarahkan
pekerjaan lewat compute unit.

Snippet kedua adalah jalur runtime langsung. Letterboxing, decoding, NMS dan
penyesuaian skala koordinat menjadi tanggung jawab Anda di sana, dan nama kelas
tersimpan di `user_defined_metadata`.

## Batasan

Empat family, hanya deteksi: `yolox`, `yolo9`, `rtdetr` dan `rfdetr`. Selain itu
ditolak saat preflight, karena wrapper preprocessing yang sadar family itulah yang
membuat kontrak input gambar tetap benar, dan family di luar daftar itu akan
dikonversi dengan normalisasi yang salah. Pesan error menyebut ONNX dan TorchScript
sebagai alternatif.

Bentuk input dikunci mati oleh `ct.ImageType`, jadi `dynamic=True` tidak mengubah
apa pun dan metadata mencatat `dynamic=False`. Ekspor bundle kedua untuk resolusi
kedua.

`half=True` mengonversi dengan presisi komputasi FP16. Tidak ada jalur INT8 dari
exporter ini.

Untuk grid family dan task lengkap, lihat
[matriks ekspor](/docs/reference/export-matrix). Untuk format on-device Apple yang
lebih baru, lihat [Core AI](/docs/export/coreai). Untuk satu kombinasi:

<code-tabs name="support" />
