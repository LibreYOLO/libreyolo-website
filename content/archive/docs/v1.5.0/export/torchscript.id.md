---
title: TorchScript
seo_title: Ekspor ke TorchScript dari LibreYOLO
description: >-
  Mengekspor model LibreYOLO ke TorchScript: arsip .torchscript hasil tracing
  dengan metadata LibreYOLO di dalamnya, bisa dimuat dari Python atau libtorch.
lead: >-
  TorchScript adalah format serialized-graph milik PyTorch sendiri. LibreYOLO
  melakukan tracing pada model dengan torch.jit.trace dan menyimpan hasilnya
  bersama berkas extra libreyolo_metadata.json, sehingga arsipnya membawa
  family, task, nama kelas dan ukuran input.
keywords:
  - export yolo ke torchscript
  - torch.jit.trace
  - torch.jit.load
  - libtorch deployment
  - torchscript metadata
  - extra_files
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="torchscript")
    mono: true
  - label: Menulis
    value: Satu arsip .torchscript dengan berkas extra libreyolo_metadata.json
  - label: Tambahan
    value: Tidak ada. TorchScript sudah termasuk dalam PyTorch.
  - label: Dimuat kembali
    value: LibreYOLO("weights/LibreYOLO9t.torchscript")
    mono: true
  - label: Bentuk
    value: Tetap. Graph dihasilkan dari tracing pada satu bentuk input.
  - label: Presisi
    value: 'FP32, FP16 (half=True). Tidak ada INT8.'
verification: >-
  Dibaca dari libreyolo/export/torchscript.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py dan libreyolo/backends/torchscript.py di branch
  dev.
snippets:
  install:
    - label: Instalasi
      language: bash
      code: |
        pip install libreyolo
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Menulis weights/LibreYOLO9t.torchscript
        path = model.export(format="torchscript")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format torchscript
    - label: Argumen
      language: python
      code: >
        model.export(
            format="torchscript",
            imgsz=640,        # int, atau (height, width)
            batch=1,
            half=False,       # bobot dan aktivasi FP16
            device=None,      # None melakukan tracing di CPU untuk format ini
            output_path=None, # None menulis weights/<stem>.torchscript
        )


        # dynamic diterima, tetapi arsipnya selalu berupa trace dengan bentuk
        tetap,

        # dan metadata yang tertanam tetap mencatat dynamic=False.
  run:
    - label: Lewat LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.torchscript")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Langsung dengan PyTorch
      language: python
      code: >
        import json


        import torch


        extra_files = {"libreyolo_metadata.json": ""}

        module = torch.jit.load(
            "weights/LibreYOLO9t.torchscript",
            map_location="cpu",
            _extra_files=extra_files,
        )

        module.eval()


        metadata = json.loads(extra_files["libreyolo_metadata.json"])

        print(metadata["model_family"], metadata["task"], metadata["imgsz"])


        # Preprocessing dan postprocessing jadi tanggung jawab Anda di jalur
        ini.

        with torch.no_grad():
            out = module(torch.zeros(1, 3, 640, 640))
        print(out.shape if torch.is_tensor(out) else [t.shape for t in out])
  support:
    - label: Memeriksa satu family dan task sebelum mengekspor
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 286a082969ccd604
---

## Instalasi

<code-tabs name="install" />

TorchScript tidak butuh apa pun di luar instalasi dasar, karena `torch.jit` sudah
termasuk dalam PyTorch. Ini satu-satunya target ekspor tanpa dependensi opsional
dan tanpa converter eksternal, sehingga berguna sebagai pemeriksaan pertama
ketika toolchain yang lebih panjang gagal.

## Ekspor

<code-tabs name="export" />

Tracing berjalan di CPU kecuali ada device yang disebutkan, dan arsipnya ditulis
ke `weights/` memakai nama dasar checkpoint ketika `output_path` tidak diisi.

Pemeriksaan retrace yang biasanya dilakukan `torch.jit.trace` dimatikan. Beberapa
wrapper ekspor menyimpan cache anchor yang bergantung pada bentuk saat forward
pass pertama, sehingga trace kedua melihat jalur Python yang berbeda meskipun
graph bentuk tetap yang terekam sudah benar. Sebagai gantinya, uji paritas
memvalidasi modul yang tersimpan secara langsung.

Metadata tidak ditaruh di berkas sidecar. `torch.jit.save` menyimpan
`libreyolo_metadata.json` di dalam arsip, dan `torch.jit.load` mengembalikannya
lewat `_extra_files`.

## Menjalankan artefak

<code-tabs name="run" />

`LibreYOLO()` melakukan routing berdasarkan sufiks `.torchscript` dan
mengembalikan objek `Results` yang sama seperti checkpoint asalnya. Dengan
`device="auto"`, modulnya dipetakan ke CUDA bila tersedia, lalu MPS, lalu CPU.

Snippet kedua adalah jalur untuk pembaca yang tidak memasang LibreYOLO, dan untuk
deployment C++ lewat libtorch, tempat arsip yang sama dimuat dengan
`torch::jit::load`. Di sana, preprocessing, decoding, NMS dan penskalaan ulang
koordinat menjadi tanggung jawab Anda. Berkas extra metadata tetap bisa dibaca,
dan hanya di situlah nama kelas tersimpan.

## Batasan

Graph-nya adalah trace pada satu bentuk input. `dynamic=True` diterima demi
simetri antarmuka tetapi tidak mengubah apa pun, dan metadata yang tertanam
melaporkan `dynamic=False` supaya backend tidak pernah mengasumsikan sumbu yang
tidak bisa dipakainya. Ekspor arsip kedua untuk resolusi kedua.

`half=True` melakukan cast pada model dan input trace ke FP16. Tidak ada jalur
INT8: `int8=True` memunculkan `NotImplementedError` saat validasi.

`imgsz` persegi panjang bekerja untuk family YOLO9, HRNet, NAFNet dan
Real-ESRGAN, dan ditolak untuk family dengan kontrak persegi tetap.

Lima kombinasi ditolak sebelum tracing. Segmentasi YOLO9, karena YOLO9 hanya
untuk deteksi di LibreYOLO. Segmentasi RTMDet-Ins, yang decode mask dengan kernel
dinamisnya tidak punya kontrak runtime hasil ekspor. Deteksi SSD, Faster R-CNN
dan RetinaNet, yang graph berpanjang variabel atau ber-anchor dinamisnya hanya
punya bukti paritas melalui kontrak ONNX Runtime.

Untuk grid family dan task lengkap, lihat
[matriks ekspor](/docs/reference/export-matrix). Untuk satu kombinasi:

<code-tabs name="support" />
