---
title: RKNN
seo_title: Ekspor ke RKNN untuk NPU Rockchip
description: >-
  Mengompilasi detektor LibreYOLO menjadi artefak .rknn Rockchip: SDK vendor
  yang dipasang sendiri, empat varian RK3588 yang tervalidasi, dan paritas di
  simulator.
lead: >-
  RKNN adalah format NPU terkompilasi milik Rockchip. LibreYOLO mengekspor ONNX
  perantara opset-19, mengompilasinya dengan SDK RKNN Toolkit2, dan dapat
  membandingkan graf hasil kompilasi dengan ONNX Runtime di simulator host
  Toolkit2 tanpa perlu board.
keywords:
  - export yolo ke rknn
  - npu rockchip rk3588
  - rknn-toolkit2
  - cara compile model yolo untuk rockchip
  - rknn simulator parity
  - inferensi yolo orange pi
last_verified: 1.5.0
meta:
  - label: Flag
    value: 'export(format="rknn", name="rk3588")'
    mono: true
  - label: Menulis
    value: >-
      Satu berkas .rknn, satu sidecar .rknn.metadata.json, dan satu laporan
      .rknn.parity.json saat verify=True
  - label: Ekstra
    value: Tidak ada di PyPI. rknn-toolkit2 adalah SDK vendor yang dipasang sendiri.
  - label: Dimuat kembali
    value: >-
      Tidak lewat LibreYOLO. Artefaknya berjalan di board dengan runtime milik
      Rockchip.
  - label: Bentuk
    value: 'Persegi tetap, batch 1, opset 19. Ketiganya diwajibkan.'
  - label: Presisi
    value: Build floating milik vendor. half=True dan int8=True ditolak.
  - label: Cakupan
    value: >-
      Empat varian deteksi di RK3588: YOLO9-t, YOLO9-E2E-t, PicoDet-s dan
      YOLO-NAS-s
verification: >-
  Dibaca dari libreyolo/export/rknn.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py dan docs/rknn.md di branch dev. Angka paritas
  hasil pengukuran berasal dari catatan validasi bertanggal 2026-08-04 di
  docs/rknn.md.
snippets:
  install:
    - label: Sisi LibreYOLO
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 'SDK vendor, dipasang sendiri'
      language: bash
      code: >
        # rknn-toolkit2 adalah SDK Rockchip dengan lisensi terpisah. LibreYOLO

        # tidak membundelnya maupun memasangnya. Hanya untuk Linux x86_64; di

        # Windows pakai WSL2 atau container Linux.

        #

        # Toolkit2 2.3.2 membutuhkan setuptools<81 dan gagal pada ONNX 1.19 atau

        # lebih baru, yang menghapus onnx.mapping padahal compiler-nya masih

        # mengimpornya.

        pip install "setuptools==80.9.0" "onnx==1.18.0"


        # Lalu pasang wheel rknn-toolkit2 yang sesuai dari repository wheel
        milik

        # Rockchip sendiri, dan pastikan wheel itu bisa diimpor:

        python -c "import rknn.api; print('rknn-toolkit2 ready')"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # Menulis weights/LibreYOLO9t.rknn dan
        weights/LibreYOLO9t.rknn.metadata.json

        path = model.export(format="rknn", name="rk3588", imgsz=640,
        verify=True)

        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format rknn --name rk3588 \
          --imgsz 640 --verify
    - label: Argumen
      language: python
      code: |
        model.export(
            format="rknn",
            name="rk3588",     # target platform; target= dan target_platform= juga bisa
            imgsz=640,         # harus cocok dengan kanvas yang tercatat untuk varian itu
            batch=1,           # nilai lain memunculkan NotImplementedError
            dynamic=False,     # True memunculkan ValueError
            opset=19,          # nilai lain memunculkan NotImplementedError
            verify=False,      # True menjalankan simulator PC dan menjadikan paritas gerbang
        )
  parity:
    - label: Paritas tanpa board terhadap artefak ONNX yang sudah ada
      language: python
      code: |
        import numpy as np
        from libreyolo.export import verify_rknn_simulator_parity

        input_tensor = np.random.default_rng(0).standard_normal(
            (1, 3, 640, 640), dtype=np.float32
        )
        metrics = verify_rknn_simulator_parity(
            "weights/LibreYOLO9t.onnx",
            input_tensor,
            target_platform="rk3588",
            rtol=1e-3,
            atol=1e-4,
            raise_on_failure=False,
        )
        print(metrics)
  support:
    - label: Memeriksa satu family dan task sebelum mengompilasi
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: c659713cc3c8cc9e
---

## Instalasi

Kompilasi membutuhkan RKNN Toolkit2 milik Rockchip, yang didistribusikan sebagai
SDK vendor di bawah lisensi Rockchip sendiri dan bukan dependensi LibreYOLO. Tidak
ada ekstra `libreyolo[rknn]`, dan tidak ada bagian dari format ini yang terpasang
lewat satu baris perintah.

<code-tabs name="install" />

Board tidak diperlukan untuk mengompilasi maupun memeriksa paritas numerik. Board
RK3588 diperlukan untuk pengukuran latensi, daya dan termal, dan tidak satu pun
dari pengukuran itu yang sudah dicatat.

## Ekspor

<code-tabs name="export" />

Permintaan divalidasi terhadap daftar varian model yang persis sebelum apa pun
dikompilasi, dan kanvasnya juga divalidasi: memberikan `imgsz` selain nilai yang
tercatat untuk varian tersebut akan memunculkan error alih-alih diam-diam
mengompilasi sesuatu yang belum diuji. LibreYOLO menulis ONNX perantara opset-19,
mengompilasinya, secara opsional menyimulasikannya, lalu menghapus berkas
perantara itu setelahnya.

Metadata ditulis sebagai sidecar bernama `<model>.rknn.metadata.json`, karena
format RKNN tidak punya field metadata yang portabel.

`verify=True` menjalankan simulator PC Toolkit2 di dalam sesi yang sama dengan
sesi yang mengompilasi artefaknya, membandingkan setiap output dengan ONNX Runtime
pada input yang sama, lalu menulis `<model>.rknn.parity.json` berisi metrik error
per output. Gerbangnya adalah cosine similarity minimal 0.9999 dan RMSE
ternormalisasi maksimal 0.02, yang diterapkan pada output mana pun yang belum
dekat secara elementwise; build floating milik vendor menurunkan tensor internal ke
half precision, sehingga `allclose` yang ketat tidak terpenuhi bahkan ketika box
hasil decoding sudah stabil. Run yang gagal menulis
`<model>.rknn.failed.parity.json`, membuang kandidatnya, dan membiarkan hasil
ekspor sukses sebelumnya di path itu tetap utuh.

Untuk membandingkan artefak ONNX yang sudah ada, tanpa mengekspor ulang:

<code-tabs name="parity" />

Simulator Toolkit2 menjalankan graf in-memory yang dihasilkan `load_onnx` dan
`build`. Simulator itu tidak bisa memuat ulang berkas `.rknn` yang spesifik untuk
satu target tanpa board, dan itulah sebabnya `verify=True` melakukan kompilasi,
ekspor dan simulasi dalam satu sesi.

## Menjalankan artefak

Tidak ada entri RKNN di `libreyolo/backends`, jadi `LibreYOLO()` tidak memuat
berkas `.rknn`. Artefak hasil kompilasi diterapkan ke board dan dijalankan oleh
runtime milik Rockchip sendiri, dan di sana preprocessing, decoding, NMS serta
penskalaan ulang koordinat menjadi tanggung jawab aplikasi.

`<model>.rknn.metadata.json` membawa nama kelas, ukuran input, task dan target
platform, yaitu apa yang dibutuhkan sebuah aplikasi untuk mereproduksi
postprocessing LibreYOLO. Kirimkan berkas itu bersama model hasil kompilasi.

Untuk pemeriksaan di sisi host yang tidak membutuhkan board, simpan artefak ONNX
dengan bentuk tetap yang sama lalu bandingkan di simulator, seperti di atas.

## Batasan

Empat kombinasi bisa dikompilasi, dan semuanya adalah varian model, bukan family:

| Varian | Task | Kanvas | Target |
|---|---|---:|---|
| YOLO9-t | detect | 640 | RK3588 |
| YOLO9-E2E-t | detect | 640 | RK3588 |
| PicoDet-s | detect | 320 | RK3588 |
| YOLO-NAS-s | detect | 640 | RK3588 |

Selain itu semuanya ditolak sebelum kompilasi, dengan pesan bahwa RKNN pada versi
ini terbatas pada varian deteksi yang persis diuji di simulator. Hasil yang hanya
sampai tahap kompilasi untuk model lain memang ada, tetapi sengaja tidak disajikan
sebagai dukungan: pada run pengukuran yang sama, RF-DETR menyisakan dua node
`GridSample` di decoder tanpa lowering, sedangkan D-FINE, RT-DETR, RT-DETRv2,
RT-DETRv4, DEIM, DEIMv2 dan EC berhasil dikompilasi dan disimulasikan tetapi
dengan output hasil decoding yang salah secara signifikan.

Batch 1, bentuk statis, opset 19. `half=True` ditolak, karena RKNN tidak
mengekspos kontrak `half` milik LibreYOLO, dan `int8=True` ditolak sampai ada
kalibrasi yang representatif beserta hasil akurasi per task.

Target Rockchip lainnya ditolak: `rk3588` adalah satu-satunya platform yang
tervalidasi.

Untuk grid family dan task selengkapnya, lihat
[matriks ekspor](/docs/reference/export-matrix). Untuk satu kombinasi:

<code-tabs name="support" />
