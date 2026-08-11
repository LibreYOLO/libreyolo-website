---
title: MNN
seo_title: Ekspor ke MNN dari LibreYOLO
description: >-
  Ekspor detektor LibreYOLO ke MNN lewat ONNX dan mnnconvert: bentuk NCHW tetap,
  FP32 di CPU, dan sidecar metadata yang dituntut kontrak runtime-nya.
lead: >-
  MNN adalah engine inferensi ringan buatan Alibaba. LibreYOLO mengekspor graph
  ONNX statis, mengonversinya dengan tool mnnconvert yang dibawa paket MNN, lalu
  menulis sidecar JSON yang mencatat nama input dan output, bentuk input yang
  tetap serta nama kelasnya.
keywords:
  - export yolo ke mnn
  - mnnconvert
  - mnn inference
  - inferensi yolo di android
  - deteksi objek mobile
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="mnn")
    mono: true
  - label: Menulis
    value: Satu berkas .mnn plus sidecar metadata .mnn.json
  - label: Extra
    value: 'pip install "libreyolo[mnn]"'
    mono: true
  - label: Dimuat kembali
    value: LibreYOLO("weights/LibreYOLO9t.mnn")
    mono: true
  - label: Bentuk
    value: NCHW tetap. dynamic=True ditolak.
  - label: Presisi
    value: 'Hanya FP32, hanya CPU.'
  - label: Task
    value: Hanya deteksi di versi ini
verification: >-
  Dibaca dari libreyolo/export/mnn.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/mnn.py dan pyproject.toml di
  branch dev.
snippets:
  install:
    - label: Instalasi
      language: bash
      code: >
        # Extra ini sudah mencakup libreyolo[onnx]: MNN mengonversi dari
        perantara ONNX.

        pip install "libreyolo[mnn]"
    - label: Memastikan converter ada di PATH
      language: bash
      code: |
        mnnconvert --version
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Menulis weights/LibreYOLO9t.mnn dan weights/LibreYOLO9t.mnn.json
        path = model.export(format="mnn", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format mnn --imgsz 640
    - label: Argumen
      language: python
      code: |
        model.export(
            format="mnn",
            imgsz=640,        # int, atau (tinggi, lebar)
            batch=1,          # ditanam ke dalam artefak
            simplify=True,    # onnxsim atas perantara ONNX
            output_path=None, # None menulis weights/<stem>.mnn
            verbose=False,    # True menyalurkan log mnnconvert
        )

        # dynamic=True memunculkan ValueError. half=True dan int8=True ditolak.
  run:
    - label: Lewat LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.mnn")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: MNN langsung
      language: python
      code: >
        import json


        import MNN

        import numpy as np


        meta = json.load(open("weights/LibreYOLO9t.mnn.json"))

        print(meta["mnn_input_names"], meta["mnn_output_names"],
        meta["mnn_input_shape"])


        runtime = MNN.nn.create_runtime_manager(
            ({"backend": 0, "precision": 1, "numThread": 4},)
        )

        module = MNN.nn.load_module_from_file(
            "weights/LibreYOLO9t.mnn",
            meta["mnn_input_names"],
            meta["mnn_output_names"],
            runtime_manager=runtime,
            dynamic=False,
            shape_mutable=False,
        )


        blob = np.zeros(meta["mnn_input_shape"], dtype=np.float32)

        input_var = MNN.expr.const(
            blob, list(blob.shape), MNN.expr.NCHW, MNN.expr.float
        )

        outputs = module.forward([input_var])

        for out in outputs:
            print(np.array(MNN.expr.convert(out, MNN.expr.NCHW).read()).shape)

        # Preprocessing dan postprocessing menjadi tanggung jawab Anda di jalur
        ini.
  support:
    - label: Memeriksa satu family dan task sebelum mengekspor
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 68fad34d07aea149
---

## Instalasi

<code-tabs name="install" />

Extra ini sudah mencakup `libreyolo[onnx]`, karena konversinya berjalan lewat
perantara ONNX. Extra ini juga membawa executable `mnnconvert`, yang dicari
exporter pertama-tama di sebelah interpreter Python yang aktif, lalu di `PATH`.
Converter yang tidak ditemukan memunculkan `ImportError` yang menyebutkan
perintah instalasinya, bukan gagal di tengah konversi.

## Ekspor

<code-tabs name="export" />

Sebelum menyerahkan graph, exporter membaca kontrak input ONNX dan menolak apa
pun yang tidak bisa diungkapkannya: input gambar lebih dari satu, atau bentuk
input dengan dimensi simbolik. MNN pada versi ini menuntut bentuk NCHW yang
sepenuhnya tetap, dan `batch` ditanam ke dalam artefak, bukan dinegosiasikan
saat pemuatan.

Sidecar itu bukan catatan tambahan yang opsional. `weights/LibreYOLO9t.mnn.json`
mencatat nama input dan output, bentuk input yang tetap, batch, nama kelas,
versi MNN yang dipakai, dan backend yang menjadi target artefaknya, dan runtime
memvalidasi setiap field tersebut saat pemuatan.

Di Windows, MNN 3.6.1 kadang menyelesaikan konversi lalu berhenti saat proses
ditutup dengan access violation atau status fail-fast. Exporter mengenali exit
code spesifik itu dan menganggap konversinya berhasil bila berkas keluarannya
ada.

## Menjalankan artefak

<code-tabs name="run" />

`LibreYOLO()` melakukan dispatch berdasarkan sufiks `.mnn` dan mengembalikan
objek `Results` yang sama seperti checkpoint. Pemuatannya memang sengaja ketat:
sidecar harus menyatakan `format=mnn`, `mnn_backend=cpu`, `dynamic=false`,
`precision=fp32`, sebuah ukuran, task deteksi, bentuk NCHW tetap yang positif
dan cocok dengan ukuran gambar yang tercatat, serta nama kelas yang mencakup
setiap indeks dari 0 sampai `nc - 1`. Ketidakcocokan apa pun akan memunculkan
error, bukan menebak.

Prediksi dengan `imgsz` yang berbeda dari saat artefak dibangun juga memunculkan
error, dan `device` diabaikan disertai peringatan, karena hasil ekspor MNN di
sini berjalan di CPU.

Snippet kedua adalah jalur runtime langsung. Preprocessing, decoding, NMS dan
penskalaan ulang koordinat menjadi tanggung jawab Anda di sana, dan nama input
serta output diambil dari sidecar karena module loader MNN memintanya secara
eksplisit.

## Batasan

Hanya deteksi. Backend menolak task lain saat pemuatan, dan sisi ekspornya
sejalan: di luar kombinasi yang tercatat, preflight memunculkan error "MNN v1
has no implemented runtime contract for this family and task."

FP32, CPU, bentuk tetap. `dynamic=True` memunculkan `ValueError`, dan
`half=True` serta `int8=True` ditolak saat validasi.

Family deteksi yang tervalidasi adalah YOLO9, YOLO9-E2E, YOLO9-P2, RF-DETR, EC,
RT-DETR, RT-DETRv2, RT-DETRv4, D-FINE, DEIM dan YOLO-NAS, masing-masing dicakup
oleh konversi, pemuatan ulang artefak yang baru, eksekusi MNN di CPU,
pemeriksaan metadata dan kesetaraan deteksi pasca-NMS terhadap model PyTorch.
DEIMv2 berhasil dikonversi, dimuat ulang, dijalankan dan mempertahankan deteksi
pasca-NMS, tetapi jalur perantara ONNX-nya belum sepenuhnya setara pada skor di
tingkat query, sehingga statusnya dicatat sebagai tersedia, bukan tervalidasi.

Untuk grid family dan task selengkapnya, lihat
[matriks ekspor](/docs/reference/export-matrix). Untuk satu kombinasi:

<code-tabs name="support" />
