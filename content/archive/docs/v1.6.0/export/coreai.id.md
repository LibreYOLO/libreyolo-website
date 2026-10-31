---
title: Core AI
seo_title: Ekspor ke Apple Core AI dari LibreYOLO
description: >-
  Mengekspor model LibreYOLO menjadi aset .aimodel Apple Core AI: khusus macOS,
  kanvas tetap, FP32, dan kontrak urutan output bernama yang wajib dipatuhi
  konsumen.
lead: >-
  Core AI adalah stack inferensi on-device milik Apple. LibreYOLO menangkap
  model dengan torch.export, menurunkannya lewat konverter Core AI, lalu menulis
  aset .aimodel yang membawa metadata model dan nama output hasil ekspor.
keywords:
  - libreyolo core ai export
  - aimodel
  - coreai-torch
  - cara export model yolo ke apple core ai
  - apple on-device inference
  - coreai_output_names
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="coreai")
    mono: true
  - label: Menghasilkan
    value: Satu aset .aimodel dengan metadata terlampir
  - label: Ekstra
    value: 'pip install "libreyolo[coreai]"'
    mono: true
  - label: Dimuat kembali
    value: Tidak lewat LibreYOLO. Konsumen memakai runtime Core AI secara langsung.
  - label: Bentuk
    value: Kanvas tetap. dynamic=True memunculkan NotImplementedError.
  - label: Presisi
    value: Hanya FP32. half=True dan int8=True ditolak.
  - label: Membutuhkan
    value: >-
      macOS. Toolchain ini tidak bisa mengonversi maupun berjalan di tempat
      lain, dan coreai-torch mengunci torch ke 2.11.x.
verification: >-
  Dibaca dari libreyolo/export/coreai.py, libreyolo/export/coreai_compat.py,
  libreyolo/export/exporter.py, libreyolo/export/support.py dan pyproject.toml
  di branch dev.
snippets:
  install:
    - label: 'Instalasi, di macOS'
      language: bash
      code: |
        # Sengaja dikecualikan dari semua ekstra agregat: coreai-torch mengunci
        # torch ke 2.11.x dan akan menyeret seluruh environment ke versi itu.
        pip install "libreyolo[coreai]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Menulis weights/LibreYOLO9t.aimodel
        path = model.export(format="coreai", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format coreai --imgsz 640
    - label: Argumen
      language: python
      code: |
        model.export(
            format="coreai",
            imgsz=640,        # int, atau (height, width); ini kanvas saat dijalankan
            batch=1,
            output_path=None, # None menulis weights/<stem>.aimodel
        )

        # dynamic=True memunculkan NotImplementedError.
        # half=True dan int8=True ditolak saat validasi.
  outputs:
    - label: Baca urutan output sebelum menyambungkan konsumen
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.export(format="coreai", imgsz=640)

        # Metadata aset mencatat nama output hasil ekspor, dalam urutan graf,
        # di bawah "coreai_output_names". Petakan dictionary yang dikembalikan
        # Core AI berdasarkan nama memakai daftar itu; jangan pernah
        # memasangkannya secara posisional dengan tuple eager.
  support:
    - label: Periksa satu family dan task sebelum mengekspor
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: a35bfeafac6d6966
---

## Instalasi

Format ini hanya untuk macOS. Dependensi `coreai-torch` membawa penanda
`sys_platform == 'darwin'`, dan toolchain ini tidak bisa mengonversi maupun
berjalan di tempat lain.

<code-tabs name="install" />

Ekstra ini berada di luar semua ekstra agregat, termasuk `libreyolo[all]`, karena
`coreai-torch` mengunci torch ke seri 2.11. Pasang di environment yang memang
bersedia dibatasi pada pasangan versi tersebut.

## Ekspor

<code-tabs name="export" />

Penangkapan graf memakai `torch.export`, sebuah graph capture sungguhan dengan
guard, bukan satu trace hasil rekaman. Ini lebih ketat daripada jalur Core ML:
pembacaan skalar di host dan alur kontrol yang bergantung pada data ditolak, bukan
diam-diam dibekukan ke dalam graf, dan itulah sebabnya beberapa family diblokir di
sini dengan catatan kegagalan capture.

Tiga langkah persiapan berjalan di dalam scope yang mengembalikan model aktif milik
pemanggil, baik ekspor berhasil maupun gagal. Pada family turunan Darknet, batch
normalization inferensinya dilipat secara persis ke konvolusi sebelumnya, karena
Core AI 0.4.1 tidak mempertahankan formula epsilon-setelah-akar-kuadrat milik
Darknet. Pada family berbasis grid dan anchor, anchor-nya dibekukan untuk kanvas
tetap. Pada RF-DETR, position embedding-nya dibangun ulang untuk kanvas yang
diminta dengan menjalankan ulang jalur baking milik model itu sendiri, karena
konverter tidak punya lowering untuk `aten._upsample_bicubic2d_aa`.

Proses lowering menyisipkan dekomposisi referensi PyTorch untuk
`aten.grid_sampler_2d` ke dalam tabel dekomposisi, karena konverter Core AI tidak
punya lowering untuk sampler deformable-attention yang dipakai family DETR.

Aset mendeklarasikan OS minimum v27, satu-satunya nilai yang ditawarkan toolchain.
Yang dibatasi adalah deployment, bukan konversi: konversi dan eksekusi di sisi
Python tetap berjalan di macOS yang lebih lama lewat runtime di dalam wheel, tetapi
angka numeriknya berbeda antar versi OS, sehingga paritas yang dicatat diukur di
macOS 27.

## Menjalankan artefak

Tidak ada entri Core AI di `libreyolo/backends`, jadi `LibreYOLO()` tidak memuat
berkas `.aimodel`. Konsumen memakai runtime Core AI secara langsung, dan
preprocessing, decoding, NMS serta penskalaan ulang koordinat menjadi tanggung
jawab mereka. Baris yang tervalidasi di matriks dukungan adalah klaim bahwa graf
hasil ekspor menghitung angka yang sama dengan referensi, bukan klaim bahwa
`predict` akan menjalankannya.

Satu hal yang tidak bisa diturunkan ulang oleh konsumen adalah urutan output:

<code-tabs name="outputs" />

Core AI mengembalikan dictionary bernama yang urutan key-nya tidak cocok dengan
urutan tuple pada forward eager, dan juga tidak bisa ditebak. Nama hasil ekspor
ditulis ke metadata aset sebagai `coreai_output_names` justru karena alasan ini.
Petakan berdasarkan nama.

## Batasan

Kanvas tetap, FP32, batch sesuai saat diekspor. `dynamic=True` memunculkan
`NotImplementedError`, dan `half=True` serta `int8=True` ditolak saat validasi.

Cakupan di sisi konversi cukup luas. Kombinasi yang tervalidasi mencakup family
YOLO9, YOLOX, YOLO7, empat detektor era Darknet, YOLO-NAS, PicoDet, RTMDet,
RT-DETR, RT-DETRv2, RT-DETRv4, D-FINE, DEIM, DEIMv2, EC dan deteksi RF-DETR; empat
family klasifikasi CNN plus CLIP dan SigLIP2 dengan kelas dibekukan; Depth Anything
V2 dan ZipDepth; restorasi NAFNet dan Real-ESRGAN; segmentasi semantik PIDNet dan
LingBotVision; serta deteksi titik FOMO. Masing-masing membawa konteks tercatatnya
sendiri, yang dicetak oleh `libreyolo formats`.

Yang diblokir, dengan alasan tercatat per kombinasi:

| Kombinasi | Alasan |
|---|---|
| Segmentasi semantik EoMT | Capture ketat gagal dengan `GuardOnDataDependentSymNode`: ada bagian di jalur mask yang membaca nilai dari tensor lalu bercabang berdasarkan nilai itu |
| Segmentasi semantik SegFormer | Jalur capture-nya belum dinilai, dan bobot yang dipublikasikan bersifat non-komersial terlepas dari formatnya |
| Gaze L2CS | Model ini sendiri hanya mendukung ONNX, TorchScript, ExecuTorch, TensorRT dan OpenVINO, dan itu keputusan di sisi model |
| Estimasi kedalaman Depth Anything 3 | Family ini menolak ekspor untuk semua format |

RF-DETR punya satu catatan yang sebaiknya dibaca sebelum membandingkan artefak.
Paritasnya dicatat terhadap graf yang disiapkan oleh eksportir Core AI itu sendiri,
bukan terhadap ONNX, dan pada kanvas 640 artefak ONNX RF-DETR tidak sepakat dengan
graf hasil persiapan itu. Rebake pada jalur Core AI mempertahankan resize
berantialias yang dilakukan model eager, sedangkan jalur ONNX mematikan
antialiasing. Karena itu ONNX bukan referensi yang sah untuk family tersebut pada
kanvas non-native.

Untuk format Apple yang lebih lama, lihat [Core ML](/docs/export/coreml). Untuk
grid family dan task lengkap, lihat [matriks ekspor](/docs/reference/export-matrix).
Untuk satu kombinasi:

<code-tabs name="support" />
