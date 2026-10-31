---
title: PicoDet
families:
  - picodet
seo_title: 'PicoDet di LibreYOLO: prediksi, latih, dan ekspor'
description: >-
  Jalankan PicoDet di LibreYOLO untuk deteksi objek seluler. Instal, prediksi,
  latih, validasi, dan ekspor di bawah Apache-2.0.
lead: >-
  PicoDet adalah detektor satu tahap yang dibuat untuk CPU seluler dan edge:
  backbone ESNet, neck CSP-PAN, dan head Generalized Focal Loss bersama.
  LibreYOLO mendukungnya untuk deteksi.
keywords:
  - PicoDet
  - PP-PicoDet
  - deteksi objek
  - object detection mobile
  - deteksi objek edge
  - ESNet
  - Generalized Focal Loss
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePICODETs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibrePICODETs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePICODETs.pt data=my-dataset.yaml
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=300, batch=16, lr0=0.01,
        )
    - label: CLI
      language: bash
      code: >
        # imgsz perlu ditetapkan: nilai default CLI adalah 640, sedangkan

        # resolusi native checkpoint s adalah 320.

        libreyolo train model=LibrePICODETs.pt data=my-dataset.yaml imgsz=320
        epochs=300 batch=16 lr0=0.01
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePICODETs.pt")
        model.export(format="onnx", imgsz=320)
        model.export(format="tensorrt", imgsz=320, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibrePICODETs.pt format=onnx imgsz=320

        libreyolo export model=LibrePICODETs.pt format=tensorrt imgsz=320
        half=True
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan sufiks berkas, sehingga artefak hasil
        ekspor

        # dimuat seperti checkpoint lain dan mengembalikan objek Results yang
        sama.

        model = LibreYOLO("LibrePICODETs.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 947aa47214abc4c0
---

## Instalasi

PicoDet tidak memerlukan komponen tambahan selain paket dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam cache lokal.

<code-tabs name="predict" />

Objek `Results` yang dikembalikan sama dengan yang dikembalikan setiap family, jadi
mengganti detektor hanya memerlukan perubahan satu baris. `conf` menetapkan ambang batas
skor keyakinan dan `iou` menetapkan ambang NMS. Lihat [prediksi](/docs/predict) untuk sumber,
streaming, dan penanganan hasil.

## Varian

Tiga ukuran, masing-masing dengan resolusi input tetapnya sendiri: `s` paling kecil dan
`l` paling besar. Resolusi meningkat bersama ukuran, sehingga checkpoint yang lebih besar
juga lebih mahal untuk dijalankan per gambar, selain memiliki lebih banyak parameter.

<benchmark-table task="detect" />

<va-embed />

## Pelatihan

<code-tabs name="train" />

Komponen loss dan assigner mengikuti resep upstream: VFL, DFL,
GIoU, dan SimOTA, dengan pembobotan kualitas klasifikasi serta target VFL
IoU dinamis. Inferensi setara hingga tingkat bit dengan upstream pada checkpoint yang sama.

Hal yang belum diperiksa, sesuai docstring `train()` sendiri: konvergensi
dataset penuh, perilaku multi-GPU, dan augmentasi apa pun selain pembalikan horizontal.
Checkpoint `s` pada resolusi native 320 juga belum secara konsisten melewati batas bawah
akurasi LibreYOLO pada fixture 30 gambar dan dua kelas yang dipakai library untuk menguji
fine-tuning kecil. Ukuran tersebut lebih cocok pada skala COCO penuh.

`train()` juga menerima argumen `pretrained`, tetapi nilainya tidak pernah dibaca
di dalam metode: pelatihan selalu berlanjut dari bobot yang dipakai saat model
dibuat, sehingga `pretrained=False` tidak menginisialisasi ulang jaringan.
Biarkan `imgsz` tidak ditetapkan di Python agar nilainya memakai resolusi native
checkpoint yang dimuat, yaitu 320 untuk `s`, 416 untuk `m`, dan 640 untuk `l`. CLI selalu
mengirim `imgsz` dengan nilai default 640, jadi tetapkan nilainya di sana agar sesuai checkpoint.

Jika pengaturan lain dibiarkan, pelatih berjalan selama 300 epoch dengan SGD pada `lr0=0.01`,
momentum 0.9, weight decay 4e-5, serta warmup 1 epoch pada jadwal kosinus.
Pembalikan horizontal adalah satu-satunya augmentasi yang diterapkan.

Lihat [pelatihan](/docs/train) untuk dataset, augmentasi, multi-GPU, dan logger.

## Validasi

`val()` mengembalikan dictionary dengan key `metrics/` yang mencakup presisi, recall,
mAP 50, dan mAP 50-95, yang diukur terhadap dataset apa pun dalam format yang digunakan untuk pelatihan.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Artefak hasil ekspor dimuat kembali melalui `LibreYOLO()` berdasarkan sufiks berkasnya,
sehingga berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan mengembalikan
`Results` yang sama. Menjalankan graph pada runtime mandiri tanpa memasang LibreYOLO
juga didukung, tetapi prapemrosesan dan pascapemrosesannya harus ditulis sendiri.

<code-tabs name="export" />

## Checkpoint

Setiap berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box>

Port LibreYOLO mengikuti Bo396543018/Picodet_Pytorch, implementasi ulang PyTorch
dari PP-PicoDet asli milik PaddleDetection, dengan mmcv dihapus dan setiap aktivasi
dicocokkan secara tepat agar checkpoint PaddlePaddle yang dikonversi melalui pipeline Bo
dapat dimuat tanpa penyimpangan numerik. Kedua sumber memakai ketentuan Apache-2.0
yang sama dengan para penulis makalah.

</provenance-box>

## Sitasi

<citation-block />
