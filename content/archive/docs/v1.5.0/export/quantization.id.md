---
title: Kuantisasi
seo_title: Mengkuantisasi model LibreYOLO di PyTorch
description: >-
  API kuantisasi PyTorch milik LibreYOLO: sembilan resep, data kalibrasi yang
  dipisahkan dari data pelatihan, QAT dan QAD, serta dua artefak deployment.
lead: >-
  Kuantisasi di LibreYOLO berjalan sepenuhnya di PyTorch: model.quantize()
  menukar modul Conv2d dan Linear sebuah model dengan padanan terkuantisasi lalu
  mengkalibrasinya. Hasilnya tetap memenuhi kontrak predict, val, train dan save
  yang biasa, sehingga model terkuantisasi dinilai oleh validator yang sama
  dengan model float.
keywords:
  - libreyolo quantization
  - kuantisasi model int8
  - quantization aware training
  - qat qad
  - nvfp4 mxfp4
  - fp8 e4m3
  - dataset kalibrasi int8
  - export yolo ke onnx int8
last_verified: 1.5.0
meta:
  - label: Panggilan
    value: 'model.quantize(recipe="int8", calib="coco128.yaml")'
    mono: true
  - label: Perintah
    value: libreyolo quantize --model M.pt --recipe int8 --calib coco128.yaml
    mono: true
  - label: Tambahan
    value: Tidak ada. Kuantisasi berjalan di PyTorch.
  - label: Family
    value: 'yolo9, rfdetr, birefnet, feynobg'
  - label: Resep
    value: 'fp16, bf16, fp8, int8, w4a16, w4a8, nvfp4, mxfp4, int2'
    mono: true
  - label: Artefak deployment
    value: >-
      export(format="pt") untuk checkpoint terpaket, export(format="onnx") untuk
      graph QDQ INT8
    mono: true
verification: >-
  Dibaca dari libreyolo/quant/api.py, libreyolo/models/base/model.py,
  libreyolo/cli/commands/quantize.py dan docs/quantization.md di branch dev.
  Angka ukuran checkpoint adalah nilai terukur yang dicatat di
  docs/quantization.md.
snippets:
  quantize:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # Pertukaran struktur plus kalibrasi. calib adalah kumpulan gambar kecil
        TANPA LABEL,

        # dibaca hanya secara maju untuk menurunkan rentang aktivasi dan
        skalanya.

        qmodel = model.quantize(recipe="int8", calib="coco128.yaml",
        samples=128)


        print(qmodel.quant_info())

        qmodel.val(data="coco8.yaml")          # validator yang sama dengan
        model float

        qmodel.save("LibreYOLO9s-int8.pt")     # checkpoint membawa manifest
        quant
    - label: CLI
      language: bash
      code: >
        libreyolo quantize --model LibreYOLO9s.pt --recipe int8 --calib
        coco128.yaml
    - label: Argumen
      language: python
      code: |
        model.quantize(
            recipe="int8",
            calib="coco128.yaml",      # path data.yaml atau nama bawaan; None melewati kalibrasi
            samples=128,               # jumlah maksimum gambar kalibrasi
            batch=8,                   # ukuran batch kalibrasi
            algorithm="auto",          # auto dan minmax sama; percentile adalah alternatifnya
            keep_high_precision=None,  # None memakai kebijakan family
            verbose=True,
        )
  reload:
    - label: Checkpoint terkuantisasi dimuat kembali tetap terkuantisasi
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Manifest quant membangun ulang struktur terkuantisasi dan skalanya
        # sebelum bobot dimuat.
        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")
        print(qmodel.quant_info())
  train:
    - label: QAT hanyalah train() biasa pada model terkuantisasi
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        # Ini finetune, bukan pelatihan dari nol: pakai learning rate finetune.
        qmodel.train(data="coco8.yaml", epochs=5, lr0=1e-4)
    - label: QAD menambahkan argumen distilasi yang sudah ada
      language: python
      code: |
        qmodel.train(
            data="coco8.yaml",
            epochs=5,
            lr0=1e-4,
            distill_model="LibreYOLO9m.pt",
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train --model LibreYOLO9s-int8.pt --data coco8.yaml --epochs 5
        --lr0 1e-4
  export:
    - label: Checkpoint PyTorch terpaket
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        # Menulis LibreYOLO9s-int8-final.pt: bobot dan skala low-bit terpaket,
        # master fp32 dibuang, sisa yang tidak terkuantisasi dikonversi ke fp16.
        qmodel.export(format="pt")

        # remainder="fp32" menjaga tensor yang tidak terkuantisasi tetap persis.
        qmodel.export(format="pt", remainder="fp32")
    - label: QDQ INT8 ONNX
      language: python
      code: |
        from libreyolo import LibreYOLO

        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        # Pasangan QuantizeLinear/DequantizeLinear di dalam graph yang membawa
        # skala hasil kalibrasi atau hasil QAT milik model itu sendiri.
        qmodel.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9s-int8.pt --format onnx
  dequantize:
    - label: 'Kembali ke float, dengan mempertahankan bobot hasil QAT'
      language: python
      code: >
        from libreyolo import LibreYOLO


        qmodel = LibreYOLO("LibreYOLO9s-int8.pt")

        qmodel.dequantize()


        # Semua exporter float kini berlaku, pada presisi apa pun yang
        didukungnya.

        qmodel.export(format="tensorrt", half=True)
source_hash: 4ffb06b87cad017e
---

## Instalasi

Kuantisasi tidak memerlukan extra apa pun. Pertukaran modul, tahap kalibrasi dan
aritmetika tersimulasi semuanya berjalan di PyTorch, jadi `pip install libreyolo`
sudah menjadi seluruh syaratnya. Artefak deployment membutuhkan apa pun yang
diminta formatnya sendiri, yang untuk jalur ONNX berarti `libreyolo[onnx]`.

## Mengkuantisasi

<code-tabs name="quantize" />

`quantize()` mengubah model yang dimuat secara in-place lalu mengembalikannya.
Tidak ada gradien yang terlibat: pertukaran modul memasang modul terkuantisasi dan
tahap kalibrasi hanya berjalan maju.

Checkpoint yang dihasilkan adalah checkpoint LibreYOLO biasa dengan manifest
`quant` terlampir, sehingga dimuat kembali dengan struktur dan skalanya tetap utuh:

<code-tabs name="reload" />

Checkpoint trainer yang ditulis selama pelatihan QAT juga membawa manifest
tersebut, yang berarti `best.pt` dari pelatihan semacam itu sudah merupakan
checkpoint terkuantisasi.

## Resep

Ada empat family yang didukung: `yolo9`, `rfdetr`, `birefnet` dan `feynobg`.

| Resep | Fungsinya | Family | Kalibrasi |
|---|---|---|---|
| `fp16` | Konversi ke half precision dengan kontrak input dan output float32. Hanya inferensi. | keempatnya | tidak ada |
| `bf16` | Konversi ke bfloat16, yang mempertahankan rentang eksponen float32. Solusi ketika fp16 overflow pada model bergaya DETR. Hanya inferensi. | keempatnya | tidak ada |
| `fp8` | Bobot dan aktivasi E4M3 pada `Conv2d` dan `Linear`: skala bobot per-channel, skala aktivasi per-tensor hasil kalibrasi. | keempatnya | wajib |
| `int8` | W8A8 pada `Conv2d` dan `Linear`: bobot simetris per-channel, aktivasi affine per-tensor. | keempatnya | wajib, atau `calib=None` untuk bobot saja |
| `w4a16` | Bobot INT4 simetris berkelompok, grup 128 sepanjang `in_features`, aktivasi float, pada `Linear`. | rfdetr, birefnet, feynobg | tidak perlu |
| `w4a8` | Bobot INT4 berkelompok plus aktivasi INT8 hasil kalibrasi, pada `Linear`. | rfdetr, birefnet, feynobg | wajib |
| `nvfp4` | NVFP4 W4A4 pada `Linear`: elemen E2M1, blok 16 elemen, skala blok FP8 E4M3, skala tensor FP32. Penskalaan aktivasi dinamis. | rfdetr, birefnet, feynobg | tidak perlu |
| `mxfp4` | MXFP4 OCP pada `Linear`: elemen E2M1, blok 32 elemen, skala blok E8M0 berpangkat dua. Penskalaan aktivasi dinamis. | rfdetr, birefnet, feynobg | tidak perlu |
| `int2` | Hanya untuk riset: bobot 2-bit berkelompok, grup 64, plus aktivasi INT8, pada `Linear`. Hasil post-training saja tidak bisa dipakai, jadi QAT atau QAD wajib dilakukan. | rfdetr | wajib |

Resep di bawah 8 bit menyasar `nn.Linear` dan sengaja ditolak untuk `yolo9`:
akselerasi itu hanya berlaku pada GEMM di perangkat keras saat ini, jadi konvolusi
tetap berada pada presisi yang lebih tinggi. YOLO9 memakai `int8` atau `fp8`.
`int2` ditolak untuk `birefnet` dan `feynobg` karena kedua family itu hanya untuk
inferensi, sehingga pemulihan lewat QAT yang menjadi sandaran resep tersebut tidak
tersedia di sana.

Nilai bawaan per family mempertahankan lapisan pertama dan head tetap float, dan
konvolusi DFL pada YOLO9 tidak pernah dikuantisasi: operator itu adalah ekspektasi
integral yang tetap. Timpa dengan `keep_high_precision=("head.",)` bila ada alasan
untuk melakukannya.

## Data kalibrasi bukan data pelatihan

`calib=` menerima beberapa ratus gambar, tidak membaca label, dan hanya berjalan
maju untuk memperkirakan rentang aktivasi. `data=` pada `train()` dan `val()`
adalah dataset berlabel yang dipakai untuk gradien dan metrik. Keduanya argumen
berbeda dengan tujuan berbeda, dan nilai bawaan untuk `calib` adalah
`coco128.yaml`.

`algorithm="minmax"` menyimpan nilai ekstrem absolut yang terlihat di seluruh batch
kalibrasi dan itulah yang dipilih oleh `"auto"`. `"percentile"` memakai rata-rata
persentil 0.1 dan 99.9 per batch; pengukuran menunjukkan algoritma ini meruntuhkan
akurasi family DETR, karena outlier aktivasi pada transformer bersifat menentukan.
Yang benar-benar mengatasi sensitivitas INT8 pada model kecil adalah kalibrasi
dengan jumlah batch yang cukup: dengan nilai bawaan `coco128`, YOLO9-t berada dalam
jarak sekitar satu poin mAP dari skor float-nya. Algoritma yang dipilih dicatat di
manifest checkpoint.

## Memulihkan akurasi

<code-tabs name="train" />

Modul terkuantisasi menyimpan bobot master fp32 dan menerapkan fake quantization
dengan straight-through estimator, sehingga gradien sampai ke master dan trainer
yang sudah ada bekerja tanpa perubahan: EMA, AMP, melanjutkan dari checkpoint dan
argumen distilasi semuanya bisa dipadukan.

QAT adalah finetune dari model yang sudah dilatih. Pakai learning rate finetune,
bukan nilai bawaan untuk pelatihan dari nol, atau pelatihan singkat akan merusak
bobot pretrained terlepas dari kuantisasinya. Ketersediaan QAD mengikuti dukungan
distilasi per family, yang saat ini berarti `yolo9` dan `rfdetr`.

Model yang dikuantisasi dengan `fp16` dan `bf16` hanya untuk inferensi, dan trainer
menolaknya sambil mengarahkan ke `amp=True`.

## Ekspor

<code-tabs name="export" />

`format="pt"` mengkristalkan model. Bobot dan skala low-bit terpaket menggantikan
master, dan sisa yang tidak terkuantisasi dikonversi ke fp16 kecuali
`remainder="fp32"` diberikan. Invarian pemaketannya: pembongkaran menghasilkan
ulang simulasi bit demi bit pada perangkat tempat finalisasi dilakukan, jadi berkas
hasil finalisasi mencetak skor yang persis sama dengan yang sudah divalidasi. Hasil
pengukuran: YOLO9-s int8 turun dari 29.5 MB ke 9.6 MB, RF-DETR-n nvfp4 dari 122 MB
ke 26 MB. Memuat berkas seperti itu memberi model yang siap inferensi, dan
memanggil `train()` padanya akan merekonstruksi master dari bobot terpaket secara
otomatis.

`format="onnx"` berlaku untuk model `int8` dan menghasilkan graph QDQ yang membawa
skala hasil kalibrasi atau hasil QAT milik model itu sendiri, yang dijalankan ONNX
Runtime dan TensorRT dengan kernel INT8 sungguhan. Ini jalur yang berbeda dari
[`export(format="onnx", int8=True)`](/docs/export/onnx) pada model float, tempat
ONNX Runtime menurunkan skalanya sendiri.

Resep konversi presisi sama sekali tidak memerlukan exporter terkuantisasi:

<code-tabs name="dequantize" />

## Batasan

Aritmetika terkuantisasi dieksekusi dalam simulasi, yaitu fake quantization yang
dihitung di pulau float32 bahkan di bawah AMP. Simulasinya benar secara numerik,
jadi skor `val()` di perangkat mana pun merupakan klaim nyata tentang aritmetika
terkuantisasi. Itu bukan klaim tentang kecepatan.

Ada dua pengecualian yang dieksekusi secara native. `fp16` dan `bf16` hanyalah
konversi presisi biasa. Modul `fp8` yang sudah difinalisasi menjalankan GEMM-nya
langsung pada bobot E4M3 terpaket melalui `torch._scaled_mm` di perangkat keras
kelas Ada, Hopper dan Blackwell, memakai skala aktivasi hasil kalibrasi yang sama
dengan simulasinya; menyetel `LIBREYOLO_KERNELS=off` mengembalikan jalur simulasi
yang persis sama di mana pun.

Cakupan deployment lebih sempit daripada daftar resepnya. Hanya `int8` yang punya
bentuk ONNX yang bisa diterapkan di sini; `fp8` dan resep linear di bawah 8 bit
dieksekusi di PyTorch dan dikristalkan lewat `format="pt"`. Meminta ekspor ONNX
dari resep tersebut akan memunculkan error berisi instruksi itu, begitu pula
meminta format non-ONNX dari model `int8`: bangun engine hilir dari graph QDQ saja.

Mengekspor model `int8` yang aktivasinya tidak pernah dikalibrasi akan mencatat
peringatan dan menghasilkan graph yang hanya membawa kuantisasi bobot.
