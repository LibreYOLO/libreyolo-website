---
title: Kernel
seo_title: Registry kernel LibreYOLO dan kernel Hub
description: >-
  Cara LibreYOLO memilih implementasi terakselerasi: registry kernel di bawah
  libreyolo/kernels, kernel MS-deform-attn Hugging Face Hub opsional, dan switch
  fused attention.
lead: >-
  Setiap operasi terakselerasi di LibreYOLO memiliki default portabel dan
  terkadang varian lebih cepat yang didaftarkan di atasnya. Pemilihan dilakukan
  saat runtime berdasarkan predicate, dependency opsional yang tidak ada
  menghasilkan fallback, bukan error, dan graph hasil ekspor selalu menggunakan
  jalur portabel.
keywords:
  - kernel libreyolo
  - LIBREYOLO_KERNELS
  - LIBREYOLO_HUB_KERNELS
  - ekstra hub-kernels
  - kernel ms_deform_attn
  - set_fused_attention
  - kernel triton libreyolo
last_verified: 1.5.0
verification: >-
  API registry dibaca dari libreyolo/kernels/__init__.py pada v1.5.0, API
  attention dari libreyolo/kernels/attention/__init__.py dan sdpa.py, provider
  Hub dari libreyolo/kernels/attention/ms_deform_attn.py termasuk revisi yang
  di-pin dan predicate kelayakannya. Layout direktori dicantumkan dari
  libreyolo/kernels/. Definisi ekstra dari pyproject.toml. Catatan perilaku dan
  angka benchmark dari docs/kernels.md. Riwayat gating v1.4.0 dari commit
  penghubungan slot RF-DETR dan entri CHANGELOG 1.5.0.
meta:
  - label: Package
    value: libreyolo.kernels
    mono: true
  - label: Ekstra untuk ikut serta
    value: 'libreyolo[hub-kernels]'
    mono: true
  - label: Paksa referensi
    value: LIBREYOLO_KERNELS=off
    mono: true
snippets:
  usage:
    - label: Lihat yang dipilih
      language: python
      code: |
        import libreyolo.kernels as kernels

        # Slot operasi ke nama implementasi yang dipilih, atau "unavailable".
        print(kernels.active())
    - label: Paksa jalur referensi
      language: bash
      code: |
        # off dan reference memiliki arti yang sama, serta melewati
        # import provider terakselerasi sepenuhnya.
        LIBREYOLO_KERNELS=off python train.py
    - label: Nonaktifkan kernel Hub tanpa menghapus instalasi
      language: bash
      code: |
        LIBREYOLO_HUB_KERNELS=0 python predict.py
    - label: Alihkan family ke fused attention
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.kernels.attention import set_fused_attention

        model = LibreYOLO("LibreSwinIRs.pt")

        # Mengembalikan jumlah modul attention yang beralih.
        print(set_fused_attention(model))
    - label: Daftarkan implementasi sendiri
      language: python
      code: |
        import libreyolo.kernels as kernels

        kernels.register(
            "fake_quant_fp8",
            my_impl,
            name="mybackend",
            predicate=my_check,
        )
source_hash: 23d504e88b7959f8
---

## Registry

`libreyolo/kernels/` adalah registry runtime kecil untuk implementasi pluggable.
Slot operasi adalah nama seperti `fake_quant_fp8` atau `ms_deform_attn`.
Pemanggil meminta sebuah slot dari registry dan memperoleh implementasi
terdaftar pertama yang lolos predicate, dengan pendaftaran terbaru sebagai
pemenang, lalu kembali ke implementasi referensi jika tidak ada yang berlaku.

Struktur ini memastikan dependency opsional tidak pernah menjadi persyaratan
keras. Mesin tanpa Triton, CUDA, atau package `kernels` menjalankan kode yang
sama dan menghasilkan angka yang sama, hanya lebih lambat.

| Fungsi | Tujuan |
|---|---|
| `active()` | Slot operasi ke nama implementasi terpilih, atau `"unavailable"` |
| `resolve(op)` | Callable yang akan dijalankan, atau `None` |
| `register(op, impl, *, name, predicate=None)` | Menambahkan implementasi, yang terbaru lebih dulu |
| `unregister(op, name)` | Menghapus satu implementasi |
| `clear_cache()` | Menghapus hasil resolve yang di-memoize |

<code-tabs name="usage" />

Predicate yang memunculkan error ditangkap dan menghasilkan peringatan, tidak
pernah diteruskan. Dengan demikian, implementasi pihak ketiga yang rusak kembali
ke jalur portabel alih-alih merusak prediksi.

### Layout

Tree diatur berdasarkan tujuan terlebih dahulu dan backend kemudian, sehingga
slot ditemukan berdasarkan yang dihitung, bukan library yang kebetulan
mengimplementasikannya saat ini.

| Direktori | Isi |
|---|---|
| `kernels/quant/simulate/` | Kernel fake-quantization Triton, dengan backward straight-through, pada device apa pun. Digunakan oleh QAT dan simulasi kuantisasi pascapelatihan |
| `kernels/quant/execute/` | Jalur presisi nyata khusus model final, tanpa backward: GEMM tensor-core FP8, prologue dan epilogue Triton yang di-fuse, serta kernel unpack bobot terkemas |
| `kernels/attention/` | Operasi attention bersama lintas family: slot `ms_deform_attn` dan kebijakan fused-SDPA |

Batas antara `simulate` dan `execute` ditentukan oleh apakah model telah
difinalisasi, bukan apakah sedang dilatih atau di-deploy. Implementasi referensi
tetap berada di `libreyolo/quant/`, yang mendefinisikan arti angka;
`kernels/` hanya mempercepatnya. Packing bobot tidak memiliki varian karena
merupakan kontrak checkpoint.

Slot GEMM dan attention tidak memiliki implementasi referensi. Pemanggil harus
memeriksa apakah `resolve()` mengembalikan nilai dan mempertahankan jalur
portabelnya sendiri. Karena itu, graph ONNX, TensorRT, dan `torch.export` selalu
memuat matematika portabel.

### Override pemilihan

`LIBREYOLO_KERNELS=off` atau `=reference` memaksa implementasi referensi dan
menghentikan import provider terakselerasi sepenuhnya. Nilai lain membatasi
pemilihan ke implementasi yang terdaftar dengan nama tersebut.
`LIBREYOLO_QUANT_KERNELS` dipatuhi sebagai alias lama dari saat registry berada
di bawah `libreyolo/quant/`, dan hanya dibaca ketika `LIBREYOLO_KERNELS` tidak
ditetapkan. Keduanya tercantum bersama variabel lain pada
[pengaturan](/docs/reference/settings).

## Kernel Hub

Kernel CUDA terkompilasi yang dipublikasikan pada Hugging Face Hub dimuat saat
runtime melalui package `kernels` opsional. Tidak ada yang di-vendor ke
LibreYOLO; artefak diambil dan di-cache oleh package tersebut, dan setiap
provider menetapkan revisi commit yang telah diaudit. Karena itu, perubahan pin
memerlukan run paritas GPU sebelum diterapkan.

Menginstal ekstra berarti ikut serta:

```bash
pip install "libreyolo[hub-kernels]"
```

Tanpa package tersebut, tidak ada yang berubah dan tidak ada permintaan network.
`LIBREYOLO_HUB_KERNELS=0` menonaktifkan pengambilan tanpa menghapus instalasi.
Kernel yang gagal dimuat atau dijalankan menonaktifkan diri selama sisa process
dan kembali ke fallback dengan satu peringatan.

Saat ini satu slot didukung Hub: `ms_deform_attn`, forward dan backward
multi-scale deformable attention terkompilasi dari Deformable DETR, berdasarkan
Apache 2.0. Slot ini dihubungkan ke seluruh lineage deformable: RF-DETR,
Deformable DETR, DINO-DETR, LW-DETR, Grounding DINO, RT-DETR, RT-DETRv2,
D-FINE, RT-DETRv4, DEIM, DEIMv2, EC, dan OV-DEIM. Karena backward juga
dikompilasi, pelatihan dan prediksi sama-sama memperoleh manfaat.

Kelayakannya sengaja dibatasi. Input harus menggunakan CUDA dan float32, serta
eksekusi harus eager: provider menolak ketika `torch.jit.is_tracing()`,
`torch.compiler.is_compiling()`, `torch.compiler.is_exporting()`, dan
`torch.onnx.is_in_onnx_export()` aktif. Dua layout input juga kembali ke jalur
portabel, yaitu jumlah titik per level yang berbeda antarlevel dan sampling
indeks bilangan bulat diskret. Varian pose EC tidak dihubungkan.

### Kernel ini baru dapat dijangkau

Baca bagian ini sebelum menginstal ekstra pada project yang sudah ada.

Pada v1.4.0, slot diperiksa dari dalam helper di balik kondisi yang mengharuskan
pasangan spatial-shape tidak ada. RF-DETR selalu meneruskan pasangan tersebut
melalui decoder, sehingga kondisi tidak pernah terpenuhi dan kernel tidak pernah
dijalankan dalam eager forward. Pemeriksaan dipindahkan pada v1.5.0, dan kini
kernel benar-benar berjalan.

Konsekuensi praktisnya adalah upgrade ke v1.5.0 sekaligus instalasi
`libreyolo[hub-kernels]` pada CUDA membuat RF-DETR dan lineage-nya menggunakan
forward dari binary terkompilasi untuk pertama kalinya. Prediksi dan metrik dapat
bergeser dalam toleransi float. Instalasi standar tanpa ekstra tidak terpengaruh.
Saat membandingkan metrik lintas upgrade, pertahankan ekstra yang sama atau
tetapkan `LIBREYOLO_HUB_KERNELS=0` pada kedua sisi.

## Fused attention

Fused scaled dot-product attention tidak memerlukan dependency opsional, hanya
PyTorch standar, sehingga diatur oleh kebijakan, bukan ketersediaan. Dua aturan
berlaku.

Pertama, graph capture tidak pernah menggunakannya. Setiap call site yang
diganti mempertahankan persamaan operasi primitif di balik pemeriksaan ekspor.
Hal ini mencakup ekspor ONNX, yang opset default-nya tidak memiliki symbolic
SDPA, serta `torch.jit.trace`, yang digunakan TorchScript, CoreML, dan NCNN.
Capture Dynamo sengaja berada di luar gate karena `torch.compile` menurunkan
SDPA lebih baik daripada matematika manual, sedangkan Core AI dan ExecuTorch
mendekomposisi SDPA ke core ATen sendiri.

Kedua, standar paritas untuk menjadikannya default adalah sama persis per byte.
Family yang lolos menggunakan SDPA secara default: SegFormer, Depth Anything
dan MoGe-2, BERT, Grounding DINO, SwinIR, dan PP-OCR. Family yang tidak lolos
mempertahankan matematika manual dan menyediakan flag `fused_attn`, yang diubah
oleh `set_fused_attention(model)`: Swin, backbone Swin milik DINO-DETR,
BiRefNet dan FeyNobg, OWLv2, LW-DETR, SigLIP 2, ZipDepth, dan MobileSAM. ViT dan
DeiT memiliki flag yang sama, tetapi default-nya aktif mengikuti upstream,
sehingga pemanggilan yang sama dengan `enabled=False` menonaktifkannya.

Fitur ini layak digunakan jika berlaku. Pada RTX 5070 Ti dengan autocast fp16,
Swin window attention turun dari 1.278 ms ke 0.721 ms, peningkatan 1.77x, dan
OWLv2 vision attention dari 6.483 ms ke 1.735 ms, peningkatan 3.74x.

## Hardware

| Platform | Perilaku |
|---|---|
| CPU dan MPS | Setiap predicate CUDA dan Triton gagal, sehingga semuanya menjalankan referensi |
| NVIDIA CUDA | Kernel Triton serta kernel Hub dan GEMM yang memenuhi syarat aktif |
| AMD ROCm | Triton dapat aktif karena wheel ROCm menyertakan backend AMD Triton, tetapi paritas hanya diuji pada NVIDIA di CI |

## Menambahkan implementasi

Panggil `register()` dengan nama dan predicate. Kernel terkompilasi di luar tree
dapat disediakan sebagai package `libreyolo_kernels` terpisah yang mendaftarkan
diri saat import, sehingga backend privat sepenuhnya berada di luar tree
LibreYOLO.

Paritas adalah gate bagi semua implementasi dalam tree: forward harus cocok
persis dengan referensi, dan gradien harus berada dalam 1e-6 dari estimator
straight-through, pada kumpulan bentuk yang dibawa test suite.

Pemilihan kernel berinteraksi dengan [CUDA graph](/docs/reference/cuda-graphs):
matriks paritas inferensi dijalankan tanpa package `kernels` terinstal, sehingga
keamanan capture dengan kernel terkompilasi aktif tidak dicakup olehnya.
