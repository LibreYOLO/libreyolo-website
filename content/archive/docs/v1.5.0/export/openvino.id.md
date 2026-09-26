---
title: OpenVINO
seo_title: Ekspor ke OpenVINO IR dari LibreYOLO
description: >-
  Mengonversi model LibreYOLO ke OpenVINO IR: pasangan model.xml dan model.bin,
  kompresi bobot FP16, INT8 lewat NNCF, serta inferensi di CPU, GPU atau NPU.
lead: >-
  OpenVINO IR adalah format runtime milik Intel, sebuah graph model.xml di
  samping blob bobot model.bin. LibreYOLO mengekspor perantara ONNX,
  mengonversinya dengan ov.convert_model, lalu menulis metadata.yaml ke
  direktori yang sama.
keywords:
  - export yolo ke openvino
  - openvino ir
  - model.xml model.bin
  - ov.convert_model
  - nncf int8 quantization
  - openvino npu
  - compress_to_fp16
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="openvino")
    mono: true
  - label: Menulis
    value: 'Sebuah direktori berisi model.xml, model.bin dan metadata.yaml'
  - label: Tambahan
    value: 'pip install "libreyolo[onnx,openvino]"'
    mono: true
  - label: Dimuat kembali
    value: LibreYOLO("weights/LibreYOLO9t_openvino")
    mono: true
  - label: Bentuk
    value: 'Mengikuti perantara ONNX: batch dinamis saat dynamic=True'
  - label: Presisi
    value: >-
      FP32, kompresi bobot FP16 (half=True), INT8 lewat NNCF (int8=True dengan
      data=)
verification: >-
  Dibaca dari libreyolo/export/openvino.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/openvino.py dan pyproject.toml
  di branch dev.
snippets:
  install:
    - label: Instalasi
      language: bash
      code: |
        # IR dikonversi dari perantara ONNX, jadi kedua extra dibutuhkan.
        pip install "libreyolo[onnx,openvino]"
    - label: INT8 juga membutuhkan NNCF
      language: bash
      code: |
        pip install nncf
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Menulis direktori weights/LibreYOLO9t_openvino
        path = model.export(format="openvino")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format openvino
    - label: Argumen
      language: python
      code: |
        model.export(
            format="openvino",
            imgsz=640,
            batch=1,
            dynamic=False,    # True mempertahankan sumbu batch dinamis sepanjang IR
            half=False,       # True menyimpan bobot FP16
            int8=False,       # True menjalankan kuantisasi pascapelatihan NNCF
            data=None,        # wajib saat int8=True
            output_path=None, # None menulis weights/<stem>_openvino
        )
  int8:
    - label: INT8 dengan data kalibrasi
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="openvino",
            int8=True,
            data="coco128.yaml",   # wajib: format ini tidak punya nilai bawaan
            fraction=1.0,
        )
  run:
    - label: Lewat LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_openvino")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Memilih perangkat
      language: python
      code: |
        from libreyolo import LibreYOLO

        # "auto" dan "cpu" dipetakan ke CPU, "gpu" dan "cuda" dipetakan ke GPU,
        # nilai lain diteruskan dalam huruf kapital, misalnya "npu" -> NPU.
        model = LibreYOLO("weights/LibreYOLO9t_openvino", device="gpu")
    - label: OpenVINO langsung
      language: python
      code: >
        import numpy as np

        import openvino as ov

        import yaml


        core = ov.Core()

        print(core.available_devices)


        compiled = core.compile_model("weights/LibreYOLO9t_openvino/model.xml",
        "CPU")

        outputs = compiled(np.zeros((1, 3, 640, 640), dtype=np.float32))

        print([tensor.shape for tensor in outputs.values()])


        # Nama kelas, task dan ukuran input tersimpan di metadata.yaml di
        samping IR.

        meta =
        yaml.safe_load(open("weights/LibreYOLO9t_openvino/metadata.yaml"))

        print(meta["model_family"], meta["task"], meta["names"])


        # Preprocessing dan postprocessing menjadi tanggung jawab Anda di jalur
        ini.
  support:
    - label: Memeriksa satu family dan task sebelum mengekspor
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 519816615e3aca3c
---

## Instalasi

<code-tabs name="install" />

Konversi melewati perantara ONNX, jadi extra `onnx` termasuk kebutuhan wajib, bukan
pelengkap opsional. NNCF dipasang terpisah dan hanya diperlukan untuk `int8=True`.

## Ekspor

<code-tabs name="export" />

Artefaknya berupa direktori, bukan berkas. `weights/LibreYOLO9t_openvino` berisi
`model.xml`, `model.bin` dan `metadata.yaml`, dan `_fp16` disisipkan sebelum suffix
saat `half=True`. Pindahkan atau salin seluruh direktori; ketiga berkas itu adalah
satu artefak.

`half=True` menyetel `compress_to_fp16` saat penyimpanan. Itu kompresi bobot di dalam
IR, bukan perubahan presisi inferensi yang dipilih perangkat saat runtime.

### INT8

<code-tabs name="int8" />

`int8=True` menjalankan kuantisasi pascapelatihan NNCF di atas loader kalibrasi
LibreYOLO dengan preset mixed, dan `data` wajib diisi: format ini tidak punya
fallback delapan gambar. NNCF yang tidak terpasang memunculkan `ImportError` yang
menyebutkan perintah instalasinya.

## Menjalankan artefak

<code-tabs name="run" />

`LibreYOLO()` mengenali direktori mana pun yang berisi `model.xml` dan mengembalikan
objek `Results` yang sama seperti checkpoint, dengan membaca nama kelas, task, ukuran
input dan skema pose dari `metadata.yaml`.

String device dipetakan, bukan diteruskan apa adanya. `auto` dan `cpu` sama-sama
dikompilasi untuk CPU, `gpu` dan `cuda` sama-sama dikompilasi untuk GPU, dan nilai
lain diubah menjadi huruf kapital lalu diserahkan ke OpenVINO, dan begitulah target
NPU dicapai.

Snippet ketiga ditujukan untuk pembaca yang tidak memasang LibreYOLO. Preprocessing,
decoding, NMS dan penyesuaian skala koordinat menjadi tanggung jawab Anda di sana,
dan nama kelas hanya ada di `metadata.yaml`.

## Batasan

IR tanpa `metadata.yaml` tetap bisa dimuat, tetapi backend lalu jatuh ke 80 kelas dan
task deteksi, yang salah untuk task lain. Jaga direktori tetap utuh.

Diblokir sebelum tracing: segmentasi YOLO9, segmentasi RTMDet-Ins, deteksi SSD,
Faster R-CNN dan RetinaNet, serta matting BiRefNet atau FeyNobg, karena OpenVINO
2026.2 tidak bisa menurunkan operasi ONNX standar `DeformConv-19` pada decoder matte
yang dipakai bersama.

Kalau sebuah kombinasi tidak divalidasi dan juga tidak diblokir, jalur converter
tersedia dan proyek belum mencatat paritas runtime OpenVINO untuknya. Beberapa
kombinasi divalidasi dengan konteks eksplisit yang dilampirkan, misalnya segmentasi
semantik DeepLabV3 pada input tetap 520 kali 520 di OpenVINO 2026.2 dengan presisi
inferensi default CPU, dan gaze L2CS pada crop wajah tetap 448 kali 448.
`libreyolo formats` mencetak konteks itu untuk tiap kombinasi.

Untuk grid family dan task lengkap, lihat
[matriks ekspor](/docs/reference/export-matrix). Untuk satu kombinasi:

<code-tabs name="support" />
