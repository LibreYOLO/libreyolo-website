---
title: TFLite
seo_title: Ekspor ke TFLite (LiteRT) dari LibreYOLO
description: >-
  Mengekspor model LibreYOLO ke FlatBuffer .tflite lewat onnx2tf: bentuk statis,
  hanya FP32, input NHWC, dan family yang terkonversi bersih.
lead: >-
  TFLite adalah format FlatBuffer yang dijalankan LiteRT di target mobile dan
  embedded. LibreYOLO mengekspor graph ONNX statis, mengonversinya dengan
  onnx2tf dalam mode flatbuffer-direct, lalu menulis metadata model di samping
  artefaknya sebagai sidecar JSON.
keywords:
  - export yolo ke tflite
  - litert
  - onnx2tf
  - ai-edge-litert
  - tflite flatbuffer
  - input nhwc tflite
  - inference edge device
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="tflite")
    mono: true
  - label: Menulis
    value: Satu berkas .tflite plus sidecar metadata .tflite.json
  - label: Tambahan
    value: 'pip install "libreyolo[tflite]"'
    mono: true
  - label: Dimuat kembali
    value: LibreYOLO("weights/LibreYOLO9t.tflite")
    mono: true
  - label: Bentuk
    value: Hanya statis. dynamic=True ditolak.
  - label: Presisi
    value: Hanya FP32. half=True dan int8=True ditolak.
  - label: Membutuhkan
    value: >-
      Python 3.12 atau lebih baru, karena onnx2tf 2.4.x tidak menerbitkan wheel
      yang lebih lama
verification: >-
  Dibaca dari libreyolo/export/tflite.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/tflite.py dan pyproject.toml
  di branch dev.
snippets:
  install:
    - label: Instalasi
      language: bash
      code: >
        # LiteRT adalah nama Google saat ini untuk TensorFlow Lite. Kedua extra

        # memasang toolchain yang sama dan menghasilkan output .tflite yang
        sama.

        pip install "libreyolo[tflite]"
    - label: Periksa dulu versi Python
      language: bash
      code: |
        python -c "import sys; print(sys.version_info >= (3, 12))"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Menulis weights/LibreYOLO9t.tflite dan weights/LibreYOLO9t.tflite.json
        path = model.export(format="tflite", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format tflite --imgsz 640

        # "litert" diterima sebagai alias dan mengarah ke exporter yang sama.
        libreyolo export --model LibreYOLO9t.pt --format litert --imgsz 640
    - label: Argumen
      language: python
      code: |
        model.export(
            format="tflite",
            imgsz=640,        # int, atau (tinggi, lebar)
            batch=1,
            simplify=True,    # onnxsim atas ONNX perantara
            output_path=None, # None menulis weights/<stem>.tflite
            verbose=False,    # True menampilkan log onnx2tf
        )

        # dynamic=True memunculkan ValueError: converter butuh bentuk statis.
        # half=True dan int8=True ditolak sebelum tracing.
  run:
    - label: Lewat LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.tflite")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: LiteRT langsung
      language: python
      code: >
        import json


        import numpy as np

        from ai_edge_litert.interpreter import Interpreter


        interpreter = Interpreter(model_path="weights/LibreYOLO9t.tflite")

        interpreter.allocate_tensors()

        detail = interpreter.get_input_details()[0]

        print(detail["shape"], detail["dtype"])   # NHWC, bukan NCHW


        interpreter.set_tensor(detail["index"], np.zeros(detail["shape"],
        np.float32))

        interpreter.invoke()

        for output in interpreter.get_output_details():
            print(output["name"], interpreter.get_tensor(output["index"]).shape)

        # Nama kelas, task dan ukuran input tersimpan di sidecar.

        meta = json.load(open("weights/LibreYOLO9t.tflite.json"))

        print(meta["model_family"], meta["task"], meta["names"])


        # Preprocessing, transpose NCHW ke NHWC dan postprocessing jadi milik
        Anda.
  support:
    - label: Memeriksa satu family dan task sebelum mengekspor
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: fa2deaa0ef6d9978
---

## Instalasi

<code-tabs name="install" />

Extra ini menarik `onnx2tf` untuk konversinya dan `ai-edge-litert` untuk
menjalankan hasilnya, keduanya di balik marker Python 3.12. Pada interpreter yang
lebih lama, ekspor memunculkan `ImportError` yang menyebutkan syarat versi, bukan
gagal di dalam converter.

`libreyolo[litert]` memasang hal yang persis sama. String format `litert` adalah
alias untuk `tflite`, dan berkas keluarannya tetap `.tflite`.

## Ekspor

<code-tabs name="export" />

Family dan task diperiksa sebelum hal lain terjadi, sehingga kombinasi yang tidak
didukung langsung gagal dengan error converter atau runtime spesifik yang membuatnya
tidak masuk daftar, bukan pesan generik. Konversinya sendiri adalah panggilan
subprocess ke `onnx2tf` dalam mode `flatbuffer_direct` atas ONNX perantara yang statis.

Metadata ditulis sebagai sidecar. `weights/LibreYOLO9t.tflite.json` memuat family,
task, nama kelas, ukuran input dan skema pose; FlatBuffer sendiri tidak punya field
metadata LibreYOLO, jadi kedua berkas itu harus selalu berpasangan.

## Menjalankan artefaknya

<code-tabs name="run" />

`LibreYOLO()` melakukan dispatch berdasarkan sufiks `.tflite` dan mengembalikan
objek `Results` yang sama seperti dari checkpoint. Backend membaca sidecar, melakukan
transpose blob NCHW menjadi NHWC ketika interpreter meminta input channels-last,
menerapkan skala kuantisasi dan zero point milik interpreter bila ada, lalu
melakukan transpose output kembali ke layout yang diharapkan postprocessing LibreYOLO.

Snippet kedua adalah jalur runtime langsung. Preprocessing, transpose layout,
decoding, NMS dan penskalaan ulang koordinat semuanya menjadi milik Anda di sana,
dan detail layout adalah bagian yang paling mudah terlewat: onnx2tf menghasilkan
input channels-last, jadi blob berbentuk `(1, 3, 640, 640)` tidak akan terikat.

## Batasan

Hanya bentuk statis. `dynamic=True` memunculkan `ValueError` sebelum tracing, dan
kanvas ekspor terkunci pada nilai yang dihasilkan `imgsz`.

Hanya FP32. `half=True` dan `int8=True` sama-sama ditolak saat validasi, sehingga
deployment terkuantisasi belum bisa dicapai dari exporter ini hari ini.

Cakupan di sini lebih sempit dibanding format graph, dan ditentukan oleh pengukuran,
bukan oleh family. Kombinasi yang sudah tervalidasi mencakup deteksi YOLO9, YOLOX
dan YOLO-NAS, segmentasi semantik PIDNet, empat family klasifikasi CNN, embedding
DINOv2 dan SigLIP2, klasifikasi SigLIP2, edge TEED dan DexiNed, serta restorasi
Real-ESRGAN dan SwinIR. SwinIR punya satu catatan tambahan: paritas hanya berlaku
bila dimensi sumber persis sama dengan kanvas ekspor, dan sumber yang lebih kecil
diberi padding sampai ukuran kanvas sebelum transformer berjalan, yang bisa berbeda
dari inferensi native berukuran variabel.

Entri yang diblokir menyebutkan kegagalan persisnya, dan itu layak dibaca sebelum
mencoba workaround. Beberapa contoh: deteksi RF-DETR terkonversi pada kanvas 384
native-nya tetapi LiteRT tidak bisa mengalokasikannya karena `STRIDED_SLICE`
menerima input di atas rank 5-D yang didukungnya; PicoDet ditolak karena sebuah
`RESHAPE` memetakan 19.200 elemen input menjadi 9.600 elemen output; D-FINE membuat
converter crash pada penanganan bentuk `GatherElements`; RTMDet berhasil diekspor
dan dimuat ulang dengan paritas raw yang utuh, tetapi box publik turun ke 0.911 IoU
dengan pergeseran koordinat 29.9 px.

Untuk grid family dan task lengkap, lihat
[matriks ekspor](/docs/reference/export-matrix). Untuk satu kombinasi, termasuk
string alasan di balik sebuah blokir:

<code-tabs name="support" />
