---
title: TensorRT
seo_title: Ekspor ke TensorRT dari LibreYOLO
description: >-
  Membangun engine TensorRT dari model LibreYOLO: perantara ONNX, build FP16 dan
  INT8, profil batch dinamis, dan batas portabilitas engine.
lead: >-
  TensorRT mengompilasi graph menjadi engine yang disetel untuk satu GPU.
  LibreYOLO mengekspor perantara ONNX lebih dulu, membacanya dengan ONNX parser
  milik TensorRT, membangun engine, lalu menulis metadata model di sebelahnya
  sebagai sidecar JSON.
keywords:
  - export yolo ke tensorrt
  - tensorrt engine
  - trt fp16
  - tensorrt int8 calibration
  - optimization profile
  - dynamic batch tensorrt
  - hardware compatibility level
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="tensorrt")
    mono: true
  - label: Menulis
    value: Satu berkas .engine plus satu sidecar metadata .engine.json
  - label: Tambahan
    value: 'pip install "libreyolo[onnx,tensorrt]"'
    mono: true
  - label: Dimuat kembali
    value: LibreYOLO("weights/LibreYOLO9t.engine")
    mono: true
  - label: Bentuk
    value: >-
      Statis secara default; dynamic=True menambahkan optimization profile pada
      sumbu batch
  - label: Presisi
    value: 'FP32, FP16 (half=True), INT8 (int8=True dengan data=)'
  - label: Membutuhkan
    value: >-
      GPU NVIDIA saat build dan saat dijalankan. Engine tidak bisa berpindah
      antar arsitektur GPU.
verification: >-
  Dibaca dari libreyolo/export/tensorrt.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/tensorrt.py dan pyproject.toml
  di branch dev.
snippets:
  install:
    - label: Instalasi
      language: bash
      code: |
        # Engine dibangun dari perantara ONNX, jadi kedua extra itu dibutuhkan.
        pip install "libreyolo[onnx,tensorrt]"
    - label: Memastikan toolchain sebelum build
      language: bash
      code: >
        python -c "import tensorrt, torch; print(tensorrt.__version__,
        torch.cuda.is_available())"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # Menulis weights/LibreYOLO9t_fp16.engine dan
        weights/LibreYOLO9t_fp16.engine.json

        path = model.export(format="tensorrt", half=True)

        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format tensorrt --half
    - label: Argumen
      language: python
      code: |
        model.export(
            format="tensorrt",
            imgsz=640,
            batch=1,
            half=False,
            int8=False,
            data=None,                      # wajib saat int8=True
            dynamic=False,
            workspace=4.0,                  # GiB memori scratch saat build
            min_batch=1,                    # batas profil dinamis
            opt_batch=1,
            max_batch=8,
            hardware_compatibility="none",  # atau "ampere_plus"
            gpu_device=0,                   # perangkat build di host multi-GPU
            verbose=False,
        )
  dynamic:
    - label: Engine batch dinamis
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Perantara ONNX perlu sumbu batch dinamis supaya profilnya
        # punya sesuatu untuk diikat.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            dynamic=True,
            min_batch=1,
            opt_batch=4,
            max_batch=8,
            half=True,
        )
  int8:
    - label: INT8 dengan data kalibrasi
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="tensorrt",
            int8=True,
            data="coco128.yaml",   # wajib: tidak ada nilai bawaan untuk format ini
            fraction=1.0,
        )
  run:
    - label: Lewat LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_fp16.engine")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Langsung dengan TensorRT
      language: python
      code: >
        import json


        import tensorrt as trt


        path = "weights/LibreYOLO9t_fp16.engine"

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

        with open(path, "rb") as handle:
            engine = runtime.deserialize_cuda_engine(handle.read())

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            print(engine.get_tensor_mode(name), name, engine.get_tensor_shape(name))

        # Nama kelas, task dan ukuran input ada di sidecar, bukan di engine.

        # Alokasi buffer, praproses dan postproses menjadi tanggung jawab Anda
        di sini.

        print(json.load(open(path + ".json"))["names"])
  support:
    - label: Memeriksa satu family dan task sebelum build
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cb90fc98ab735233
---

## Instalasi

Baik proses build maupun proses menjalankannya membutuhkan GPU NVIDIA dengan
stack CUDA yang berfungsi. Tidak ada fallback CPU untuk format ini.

<code-tabs name="install" />

Extra `tensorrt` mengunci `tensorrt-cu12` dan `pycuda`, dan marker-nya membuang
keduanya di macOS. Di Jetson, jangan pakai extra itu: ia mengunci build CUDA 12
di atas platform CUDA 13. Pakai TensorRT yang dipasang JetPack, seperti
dijelaskan di [NVIDIA Jetson](/docs/export/jetson).

## Ekspor

<code-tabs name="export" />

Ekspornya berjalan dalam dua langkah. Langkah pertama menulis perantara ONNX ke
path sementara, langkah kedua membacanya dan membangun engine, lalu perantara itu
dihapus setelahnya. `workspace` adalah memori scratch saat build dalam GiB; nilai
yang lebih besar memberi builder kesempatan mencoba lebih banyak kernel dan tidak
memengaruhi memori inferensi.

Sidecar metadata ditulis di sebelah engine sebagai `<engine>.json` dan mencatat
presisi yang benar-benar terwujud saat build. Kalau GPU tidak punya FP16 cepat
atau INT8 cepat, builder memberi peringatan lalu mundur ke presisi lain, dan
sidecar melaporkan presisi yang keluar, bukan presisi yang diminta.

Di bawah FP16, backbone ViT di dalam graph akan terdeteksi dan lapisan float-nya
dikunci ke FP32. Backbone bergaya DINOv2 mengalami overflow di FP16 dan
menghasilkan NaN, jadi build menyetel `OBEY_PRECISION_CONSTRAINTS` dan melaporkan
`FP16 (FP32 ViT backbone)`. Pass ini tidak melakukan apa pun pada backbone CNN.

### Batch dinamis

<code-tabs name="dynamic" />

`dynamic=True` menambahkan satu optimization profile yang membentang dari
`min_batch` sampai `max_batch`, dioptimalkan di `opt_batch`, dan mencatat ketiga
nilai itu di sidecar. Profil hanya ditambahkan kalau perantara ONNX memang membawa
dimensi batch dinamis; kalau tidak, build mencatat di log bahwa ia memakai
optimasi statis lalu melanjutkan.

### INT8

<code-tabs name="int8" />

INT8 memakai entropy calibrator milik TensorRT di atas loader kalibrasi LibreYOLO,
dan `data` wajib diisi: format ini tidak punya fallback delapan gambar. Kalibrasi
membutuhkan `cuda-python` atau `pycuda` untuk buffer perangkat. Cache kalibrasi
diberi kunci dari hash byte ONNX, sehingga skala dari satu model tidak pernah
dipakai ulang untuk model lain yang kebetulan menulis ke path keluaran yang sama.

`half=True` dan `int8=True` bersamaan akan memunculkan peringatan dan tetap
membangun INT8, yang menyimpan fallback FP16 untuk lapisan yang tidak bisa
dikuantisasi TensorRT.

## Menjalankan artefak

<code-tabs name="run" />

`LibreYOLO()` memilih jalurnya dari sufiks `.engine`, membaca sidecar untuk nama
kelas, task dan skema pose, lalu mengembalikan objek `Results` yang sama seperti
checkpoint. Fungsi ini langsung melempar error kalau tidak ada perangkat CUDA.

Snippet kedua adalah jalur runtime langsung. Alokasi buffer host dan perangkat,
praproses, decoding, NMS dan penyesuaian skala koordinat semuanya menjadi
tanggung jawab Anda, dan engine itu sendiri tidak membawa nama kelas, jadi
sidecar harus ikut dibawa bersamanya.

## Batasan

Engine yang sudah diserialisasi terikat pada arsitektur GPU, stack driver dan
versi TensorRT yang membangunnya. Engine yang dibangun di workstation tidak akan
bisa dimuat di arsitektur lain, dan karena itulah langkah build dijalankan di
mesin deployment. `hardware_compatibility="ampere_plus"` menukar sebagian performa
dengan portabilitas di Ampere dan yang lebih baru. Nilai
`"same_compute_capability"` dipetakan ke `NONE` dan memunculkan peringatan: engine
hanya dioptimalkan untuk GPU saat ini, dan ekspornya menyatakan hal itu alih-alih
mengklaim portabilitas yang tidak ia terapkan.

Hanya sumbu batch yang diprofilkan. Build dengan dimensi spasial dinamis bukan
bagian dari kontrak ini, dan karena itu FCOS diblokir: ia butuh tinggi dan lebar
padding yang dinamis untuk mempertahankan transformasi aspek 800 kali 1333
miliknya.

Diblokir sebelum tracing: segmentasi YOLO9, segmentasi RTMDet-Ins, deteksi SSD,
Faster R-CNN dan RetinaNet, serta matting BiRefNet atau FeyNobg, di mana TensorRT
10.16 sampai pada node ONNX `DeformConv` yang dipakai bersama dan tidak bisa
membacanya karena `ModulatedDeformConv2d` tidak ada di registry plugin.

Kalau sebuah kombinasi tidak divalidasi dan juga tidak diblokir, jalur
konverternya tersedia dan proyek ini belum mencatat paritas runtime TensorRT
untuknya. Itu pernyataan soal bukti, bukan soal apakah proses build berhasil.

Untuk grid family dan task lengkap, lihat
[matriks ekspor](/docs/reference/export-matrix). Untuk satu kombinasi:

<code-tabs name="support" />
