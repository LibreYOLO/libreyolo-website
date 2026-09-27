---
title: ExecuTorch
seo_title: Ekspor ke ExecuTorch dari LibreYOLO
description: >-
  Ekspor model LibreYOLO ke program ExecuTorch .pte dengan delegasi XNNPACK:
  bentuk tetap, batch 1, FP32, dan sidecar metadata yang dibutuhkannya.
lead: >-
  ExecuTorch menjalankan program PyTorch di target edge. LibreYOLO menangkap
  model dengan torch.export dalam mode strict, menurunkannya ke XNNPACK, dan
  menyimpan program .pte bersama sidecar metadata JSON sebagai satu kesatuan.
keywords:
  - export yolo ke executorch
  - program .pte
  - xnnpack partitioner
  - torch.export strict
  - executorch runtime
  - inference pytorch di perangkat edge
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="executorch")
    mono: true
  - label: Menulis
    value: Satu program .pte plus satu sidecar metadata .pte.json
  - label: Extra
    value: 'pip install "libreyolo[executorch]"'
    mono: true
  - label: Dimuat kembali
    value: LibreYOLO("weights/LibreYOLO9t.pte")
    mono: true
  - label: Bentuk
    value: Tetap. dynamic=True dan batch != 1 ditolak.
  - label: Presisi
    value: Hanya FP32. half=True dan int8=True ditolak.
  - label: Delegate
    value: 'XNNPACK, CPU. delegate=''xnnpack'' adalah satu-satunya nilai yang diterima.'
verification: >-
  Dibaca dari libreyolo/export/executorch.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/executorch.py dan
  pyproject.toml di branch dev.
snippets:
  install:
    - label: Instalasi
      language: bash
      code: |
        # Sengaja tidak dimasukkan ke libreyolo[all]: ExecuTorch membatasi versi
        # Torch yang bisa dipasangkan dengannya.
        pip install "libreyolo[executorch]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Menulis weights/LibreYOLO9t.pte dan weights/LibreYOLO9t.pte.json
        path = model.export(format="executorch", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format executorch --imgsz 640
    - label: Argumen
      language: python
      code: |
        model.export(
            format="executorch",
            imgsz=640,             # int, atau (tinggi, lebar)
            batch=1,               # nilai lain memunculkan ValueError
            dynamic=False,         # True memunculkan ValueError
            delegate="xnnpack",    # satu-satunya nilai yang diterima
            device="cpu",          # device lain memunculkan ValueError
            output_path=None,      # None menulis weights/<stem>.pte
        )
  run:
    - label: Lewat LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t.pte")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Langsung di runtime ExecuTorch
      language: python
      code: >
        import json

        from pathlib import Path


        import torch

        from executorch.runtime import Runtime


        runtime = Runtime.get()

        print(runtime.backend_registry.is_available("XnnpackBackend"))


        program =
        runtime.load_program(Path("weights/LibreYOLO9t.pte").read_bytes())

        method = program.load_method("forward")


        # Praproses dan pascaproses menjadi tanggung jawab Anda di jalur ini.

        outputs = method.execute((torch.zeros(1, 3, 640, 640),))

        print([tensor.shape for tensor in outputs])


        meta = json.load(open("weights/LibreYOLO9t.pte.json"))

        print(meta["model_family"], meta["task"], meta["executorch_delegate"])
  support:
    - label: Memeriksa satu family dan task sebelum mengekspor
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: c2c354a76ee33157
---

## Instalasi

<code-tabs name="install" />

Extra ini sengaja berada di luar `libreyolo[all]`, karena ExecuTorch mengunci versi
Torch yang bisa dipakainya dan memasangnya akan menyeret seluruh environment ke
pasangan versi itu. Pasang di environment yang memang siap dibatasi.

Di Windows, tahap lowering memanggil executable `flatc` yang disertakan ExecuTorch.
Jika executable itu tidak ada di `PATH`, ekspor memunculkan `RuntimeError` yang
menyebutkannya, dan solusinya adalah menjalankan dari Visual Studio 2022 Developer
PowerShell.

## Ekspor

<code-tabs name="export" />

Penangkapan model memakai `torch.export.export(..., strict=True)`, yaitu capture
grafik sungguhan dengan guard, bukan trace hasil rekaman. Pembacaan skalar di host
dan alur kontrol yang bergantung pada data ditolak, bukan ikut dibekukan diam-diam,
sehingga beberapa family gagal di sini padahal lolos trace di tempat lain; alasannya
dicatat per kombinasi di matriks dukungan.

Lowering menjalankan `to_edge_transform_and_lower` dengan partitioner XNNPACK. Jika
hasilnya tidak memuat satu pun partisi delegate, ekspor memunculkan error alih-alih
melabeli program yang hanya memakai portable kernel sebagai XNNPACK.

Program dan sidecar disimpan sebagai satu kesatuan. Keduanya disiapkan lebih dulu,
keduanya ditukar masuk bersamaan, dan kegagalan mengembalikan keadaan ke isi
sebelumnya, sehingga pasangan yang setengah jadi tidak pernah sampai ke disk.

## Menjalankan artefaknya

<code-tabs name="run" />

`LibreYOLO()` mengenali akhiran `.pte` dan mengembalikan objek `Results` yang sama
seperti checkpoint. Sidecar wajib ada saat pemuatan: tanpa `<program>.pte.json`
backend memunculkan `FileNotFoundError`, karena program itu sendiri tidak membawa
nama kelas, task, maupun ukuran input. Backend juga memeriksa bahwa runtime yang
terpasang menyediakan `XnnpackBackend` sebelum memuat, dan membaca program dari byte
alih-alih memetakan berkasnya, sehingga tidak menahan file lock Windows selama
backend hidup.

Snippet kedua adalah jalur runtime langsung. Praproses, decoding, NMS dan
penskalaan ulang koordinat menjadi tanggung jawab Anda di sana.

## Batasan

Batch 1, bentuk tetap, FP32, CPU. `batch != 1` dan `dynamic=True` sama-sama
memunculkan `ValueError` sebelum ekspor mengubah apa pun, `half=True` dan
`int8=True` ditolak saat validasi, dan device selain CPU tidak diterima.

`delegate` menerima `"xnnpack"` dan tidak ada nilai lain di versi ini.

Ekspor klasifikasi membawa dua key metadata tambahan, `crop_pct` dan
`interpolation`, agar runtime bisa mereproduksi kebijakan resize dan center-crop
milik family tersebut.

Entri yang diblokir menyebutkan kegagalan konkretnya, bukan sekadar kategorinya.
Deteksi dan segmentasi D-FINE sampai pada pembacaan `ContextVar` yang tidak didukung
di deformable attention saat strict capture, dan memaksa jalur grid-sample manual
memang berhasil diserialisasi tetapi kemudian gagal saat dijalankan karena urutan
dimensi tensor terdelegasi tidak valid. DEIM dan DEIMv2 lolos capture, lowering dan
serialisasi, lalu gagal saat eksekusi. Segmentasi semantik EoMT gagal pada ekspresi
simbolik yang bergantung pada data di jalur mask. Matting BiRefNet lolos capture
pada 1024 kali 1024 tetapi tidak punya varian out untuk
`torchvision::deform_conv2d`. Restorasi SwinIR berhasil dimuat ulang lalu gagal di
`aten::alias_copy.out` karena urutan dimensi tidak cocok.

Untuk grid family dan task lengkap, lihat
[matriks ekspor](/docs/reference/export-matrix). Untuk satu kombinasi:

<code-tabs name="support" />
