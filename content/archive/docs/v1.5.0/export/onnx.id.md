---
title: ONNX
seo_title: Ekspor ke ONNX dari LibreYOLO
description: >-
  Mengekspor model LibreYOLO ke ONNX: opset yang dipilih LibreYOLO per family,
  dynamic axes, NMS tertanam, INT8, dan cara graph dimuat kembali.
lead: >-
  ONNX adalah format graph yang portabel. LibreYOLO melakukan tracing pada model
  dengan torch.onnx.export, secara opsional menyederhanakan graph, lalu menulis
  family, task, nama kelas dan ukuran input ke metadata berkas itu sendiri
  sehingga backend LibreYOLO mana pun bisa membangun ulang postprocessing.
keywords:
  - export yolo ke onnx
  - onnxruntime python
  - torch.onnx.export
  - onnx opset
  - dynamic axes
  - embedded nms onnx
  - onnx int8 qdq
  - onnx metadata_props
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="onnx")
    mono: true
  - label: Menulis
    value: 'Satu berkas .onnx, metadata tertanam di dalam graph'
  - label: Tambahan
    value: 'pip install "libreyolo[onnx]"'
    mono: true
  - label: Dimuat kembali
    value: LibreYOLO("weights/LibreYOLO9t.onnx")
    mono: true
  - label: Bentuk
    value: Batch dinamis secara default di Python; pengecualian per task ada di bawah
  - label: Presisi
    value: 'FP32, FP16 (half=True), INT8 (int8=True, deteksi YOLO9)'
verification: >-
  Dibaca dari libreyolo/export/onnx.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/onnx.py dan
  libreyolo/cli/commands/export.py di branch dev.
snippets:
  install:
    - label: Instalasi
      language: bash
      code: |
        pip install "libreyolo[onnx]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Menulis weights/LibreYOLO9t.onnx
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format onnx
    - label: Argumen
      language: python
      code: |
        model.export(
            format="onnx",
            imgsz=640,        # int, atau (tinggi, lebar)
            batch=1,
            dynamic=True,     # default Python; di CLI defaultnya False
            simplify=True,    # menjalankan onnxsim pada graph
            opset=None,       # None memilih 13, atau 17 untuk family bergaya DETR
            half=False,       # bobot dan aktivasi FP16
            int8=False,       # QDQ INT8, hanya deteksi YOLO9
            data=None,        # data.yaml kalibrasi, hanya untuk INT8
            device=None,      # device tracing; None memakai device model
            output_path=None, # None menulis weights/<stem>.onnx
        )
  nms:
    - label: Menanamkan NMS di dalam graph
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Hanya deteksi YOLO9, batch 1. dynamic dipaksa menjadi False.
        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            nms=True,
            conf=0.25,
            iou=0.45,
            max_det=300,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format onnx --nms \
          --conf 0.25 --iou 0.45 --max-det 300
  int8:
    - label: INT8 dengan data kalibrasi
      language: python
      code: |
        from libreyolo import LibreYOLO

        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            int8=True,
            data="coco128.yaml",   # beberapa ratus gambar yang representatif
            fraction=1.0,
        )
  run:
    - label: Lewat LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.onnx")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Langsung dengan ONNX Runtime
      language: python
      code: >
        import numpy as np

        import onnx

        import onnxruntime as ort


        session = ort.InferenceSession(
            "weights/LibreYOLO9t.onnx",
            providers=["CPUExecutionProvider"],
        )


        # Preprocessing dan postprocessing menjadi tanggung jawab Anda di jalur
        ini.

        batch = np.zeros((1, 3, 640, 640), dtype=np.float32)

        outputs = session.run(None, {session.get_inputs()[0].name: batch})

        print([out.shape for out in outputs])


        # Graph membawa family, task, nama kelas dan ukuran input.

        meta = {p.key: p.value for p in
        onnx.load("weights/LibreYOLO9t.onnx").metadata_props}

        print(meta["model_family"], meta["task"], meta["imgsz"])
  support:
    - label: Memeriksa satu family dan task sebelum mengekspor
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cee78250fc7189a3
---

## Instalasi

<code-tabs name="install" />

Paket tambahan ini menarik `onnx`, `onnxsim` dan `onnxruntime`. `onnx` saja sudah
cukup untuk menulis berkasnya; `onnxsim` menjalankan tahap penyederhanaan dan
`onnxruntime` menjalankan artefak sekaligus melakukan kalibrasi INT8.

## Ekspor

<code-tabs name="export" />

Tanpa `output_path`, berkasnya mendarat di `weights/` memakai stem checkpoint,
dengan `_fp16` atau `_int8` ditambahkan saat presisi itu diminta.

`dynamic` bernilai `True` secara default di Python dan `False` di CLI. Saat aktif,
sumbu batch menjadi simbolis dan beberapa task melonggar lebih jauh: segmentasi
semantik juga membuka tinggi dan lebar mask, restorasi Real-ESRGAN membuka sumbu
spasial, dan detektor dua tahap menjaga tinggi dan lebar sumber tetap dinamis
karena resize dilakukan di dalam graph.

`opset` dipilih per family kalau tidak disebutkan. Family bergaya DETR (`detr`,
`deformable_detr`, `dinodetr`, `dfine`, `deim`, `deimv2`, `ec`, `lwdetr`,
`rfdetr`, `rtdetr`, `rtdetrv2`, `rtdetrv4`) ditambah `deit`, `midas` dan `moge2`
memakai opset 17, karena di sanalah `aten::scaled_dot_product` diturunkan. Sisanya
memakai 13. Matting selalu dinaikkan ke 19, karena decoder BiRefNet membutuhkan
operator `DeformConv`, yang didefinisikan ONNX mulai opset 19.

`simplify=True` menjalankan `onnxsim` dan mempertahankan graph asli kalau tahap itu
gagal, jadi kesalahan penyederhanaan hanya menjadi peringatan, bukan kegagalan
ekspor. Di macOS arm64 dengan `onnx` 1.22 atau lebih baru dan `onnxsim` 0.6.5 atau
lebih lama, tahap ini dilewati sepenuhnya, karena pasangan versi itu bisa
menghentikan paksa proses Python.

### NMS tertanam

<code-tabs name="nms" />

`nms=True` hanya untuk deteksi YOLO9 dan mensyaratkan batch 1; memintanya bersama
`dynamic=True` memunculkan peringatan di log dan mematikan dynamic. Graph lalu
punya dua output: `output`, berbentuk `(batch, max_det, 6)`, dan `raw`, yaitu
tensor detektor mentah sebelum decoding yang dipakai backend LibreYOLO sendiri
agar postprocessing tetap identik dengan jalur PyTorch.

### DeepStream

`deepstream=True` adalah opsi khusus ONNX. Opsi ini mengekspor graph dalam layout
yang diharapkan parser NVIDIA DeepStream dan menulis dua berkas pendamping di
sebelahnya, `config_infer_primary_<stem>.txt` dan `<stem>_labels.txt`, sehingga
artefak langsung masuk ke sebuah pipeline tanpa konfigurasi yang ditulis tangan.

Opsi ini saling eksklusif dengan `nms=True`, dan meminta keduanya memunculkan
`ValueError`: DeepStream menjalankan supresi di tahap clustering miliknya sendiri.
Memberikannya ke format selain ONNX juga memunculkan error. Lihat
[DeepStream](/docs/export/deepstream) untuk grid family dan task yang didukung
serta cara membangun parsernya.

### INT8

<code-tabs name="int8" />

`int8=True` menjalankan kuantisasi statis ONNX Runtime dan menulis graph QDQ dengan
input dan output float32. Hanya node `Conv` dan `Gemm` yang dikuantisasi.
Membiarkan decoding pada head deteksi tetap di float32 adalah keputusan yang
disengaja: konkatenasi itu mencampur koordinat box berskala piksel dengan skor
kelas pada rentang 0 sampai 1, dan satu skala aktivasi per tensor yang didominasi
besaran box akan menekan semua skor menjadi nol.

Untuk saat ini flag ini hanya berlaku pada deteksi YOLO9, dan kombinasi lain
memunculkan `NotImplementedError` di tahap preflight. Menghilangkan `data` membuat
kalibrasi jatuh ke `coco8.yaml` disertai peringatan; delapan gambar bukan set
kalibrasi yang representatif. Model yang sudah dikuantisasi di PyTorch menempuh
jalur berbeda, yang dijelaskan di [Kuantisasi](/docs/export/quantization).

## Menjalankan artefak

<code-tabs name="run" />

`LibreYOLO()` mengenali akhiran `.onnx` dan mengembalikan objek `Results` yang sama
seperti checkpoint `.pt`, karena nama kelas, task, ukuran input dan skema pose
sudah ditulis ke `metadata_props` milik graph saat ekspor. Dengan `device="auto"`,
session memakai `CUDAExecutionProvider` kalau ONNX Runtime melaporkannya, dan jatuh
ke CPU kalau tidak.

Snippet kedua ditujukan untuk pembaca yang tidak memasang LibreYOLO. Preprocessing,
decoding, NMS dan penyesuaian skala koordinat semuanya menjadi tanggung jawab Anda
di jalur itu; blok metadata tetap ada di sana untuk dibaca.

## Batasan

Nama tensor output ditetapkan per task, dan itulah yang harus dicocokkan konsumen
yang tidak membaca metadata:

| Task | Nama output |
|---|---|
| Deteksi, head grid dan anchor | `output` |
| Deteksi, bergaya DETR | `pred_logits`, `pred_boxes` |
| Deteksi, RF-DETR | `dets`, `labels` |
| Klasifikasi | `output` |
| Segmentasi semantik | `semantic_logits` |
| Kedalaman | `depth` |
| Normal permukaan | `normal` |
| Tepi | `edges` |
| Restorasi | `restored` |
| Matting | `matte` |
| Gaze | `yaw_logits`, `pitch_logits` |

RF-DETR juga satu-satunya family yang tensor inputnya bernama `input`, bukan
`images`.

Beberapa task membawa kontrak runtime beresolusi tetap di versi ini. Kedalaman,
normal permukaan dan tepi menolak `batch != 1` dan memaksa `dynamic=False`. Matting
memaksa ukuran persegi 1024 bawaannya, karena tabel posisi relatif Swin milik
BiRefNet terikat pada resolusi tersebut. Restorasi memaksa kanvas tetap untuk semua
family kecuali Real-ESRGAN, yang generatornya sepenuhnya konvolusional.

`imgsz` berbentuk persegi panjang bisa dipakai family YOLO9, HRNet, NAFNet dan
Real-ESRGAN. Family dengan kontrak persegi tetap (`clip`, `deformable_detr`,
`detr`, `dinodetr`, `dfine`, `deim`, `deimv2`, `ec`, `lwdetr`, `moge2`, `rtdetr`,
`rtdetrv2`, `rtdetrv4`, `rfdetr`, `siglip2`, `ssd`) menolaknya langsung.

Dua kombinasi ditolak sebelum tracing: segmentasi YOLO9, karena YOLO9 hanya untuk
deteksi di LibreYOLO, dan segmentasi RTMDet-Ins, yang decoding mask kernel
dinamisnya tidak punya kontrak runtime hasil ekspor.

Untuk grid family dan task selengkapnya, lihat
[matriks ekspor](/docs/reference/export-matrix). Untuk satu kombinasi, tanyakan
langsung ke library-nya:

<code-tabs name="support" />
