---
title: RTMDet
families:
  - rtmdet
seo_title: 'RTMDet di LibreYOLO: prediksi, latih, dan ekspor'
description: >-
  Jalankan RTMDet di LibreYOLO untuk deteksi objek dan segmentasi instance
  RTMDet-Ins. Instal, prediksi, latih, validasi, dan ekspor di bawah Apache-2.0.
lead: >-
  RTMDet adalah detektor satu tahap yang memprediksi dari satu prior berbasis
  titik per lokasi grid, tanpa anchor, melalui head yang konvolusinya dipakai
  bersama pada berbagai level feature. LibreYOLO mendukungnya untuk deteksi dan
  segmentasi instance RTMDet-Ins.
keywords:
  - RTMDet
  - deteksi objek
  - segmentasi instance
  - RTMDet-Ins
  - anchor-free detection
  - mmdetection
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTMDets.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRTMDets.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Segmentasi instance
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Sufiks -seg dalam nama berkas memilih head mask RTMDet-Ins,
        # sehingga argumen task tidak diperlukan di sini.
        model = LibreYOLO("LibreRTMDets-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRTMDets.pt data=my-dataset.yaml
    - label: Segmentasi instance
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # mask
        print(metrics["metrics/mAP50-95(B)"])   # kotak
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=300, imgsz=640, batch=16, lr0=0.004,
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreRTMDets.pt data=my-dataset.yaml imgsz=640
        epochs=300 batch=16 lr0=0.004
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTMDets.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRTMDets.pt format=onnx imgsz=640

        libreyolo export model=LibreRTMDets.pt format=tensorrt imgsz=640
        half=True
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan sufiks berkas, sehingga artefak hasil
        ekspor

        # dimuat seperti checkpoint lain dan mengembalikan objek Results yang
        sama.

        model = LibreYOLO("LibreRTMDets.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 2f5033bdc1c3c931
---

## Instalasi

RTMDet tidak memerlukan komponen tambahan selain paket dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam cache lokal.

<code-tabs name="predict" />

Objek `Results` yang dikembalikan sama dengan yang dikembalikan setiap family, jadi mengganti
detektor hanya memerlukan perubahan satu baris. Nama berkas `-seg` menentukan task
RTMDet-Ins dengan sendirinya, kemudian `result.masks` menyimpan mask instance di samping
kotak. `conf` menetapkan ambang batas confidence dan `iou` menetapkan ambang NMS.
Lihat [prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Varian

Lima ukuran, dari `t` hingga `x`, memakai satu arsitektur pada resolusi input yang sama.
Family ini tidak memiliki tabel benchmark di sini: bandingkan ukuran melalui ukuran berkas
checkpoint pada tabel di bawah.

## Pelatihan

<code-tabs name="train" />

Deteksi dilatih melalui `train()`. Komponen QualityFocalLoss, GIoU, dan
DynamicSoftLabelAssigner di-porting dari mmdetection upstream. Forward pass dan ekspor
ONNX setara hingga tingkat bit dengannya, sedangkan pascapemrosesan cocok dengan output
mmdet dalam selisih 0.001 mAP pada subset val2017.

Hal yang belum diperiksa, sesuai docstring `train()` sendiri: konvergensi fine-tuning
dataset kecil, kesetaraan makalah dari nol, perilaku multi-GPU, throughput Mosaic dan MixUp
dengan cache, peralihan pipeline dua tahap upstream yang ketat, serta override weight decay
per parameter yang membuat decay nol pada parameter norm dan bias.

RTMDet-Ins tidak memiliki jalur pelatihan. Memanggil `train()` pada checkpoint `-seg`,
atau dengan `task="segment"`, akan memunculkan `NotImplementedError`; segmentasi instance
hanya mendukung inferensi dan validasi.

`train()` juga menerima argumen `pretrained`, tetapi nilainya tidak pernah dibaca di dalam
metode: pelatihan selalu berlanjut dari bobot yang dipakai saat model dibuat, sehingga
`pretrained=False` tidak menginisialisasi ulang jaringan.

Jika pengaturan lain dibiarkan, pelatih berjalan selama 300 epoch dengan AdamW pada
`lr0=0.004` dan `weight_decay=0.05`, warmup 1 epoch pada jadwal kosinus, serta Mosaic dan
MixUp yang dinonaktifkan selama 20 epoch terakhir.

Lihat [pelatihan](/docs/train) untuk dataset, augmentasi, multi-GPU, dan logger.

## Validasi

`val()` mengembalikan dictionary dengan key `metrics/` yang mencakup presisi, recall,
mAP 50, dan mAP 50-95, yang diukur terhadap dataset apa pun dalam format yang digunakan
untuk pelatihan.

<code-tabs name="val" />

Untuk checkpoint `-seg`, key biasa `metrics/mAP50-95` menyimpan skor mask. Proses yang
sama juga melaporkan kotak pada `(B)` dan mask pada `(M)`, sehingga keduanya tersedia
dari satu tahap.

## Ekspor

<export-matrix />

Deteksi dapat diekspor ke sebagian besar format; segmentasi instance saat ini tidak dapat
diekspor ke format mana pun. Matriks di atas mencerminkan perbedaan itu. Artefak deteksi
hasil ekspor dimuat kembali melalui `LibreYOLO()` berdasarkan sufiks berkasnya, sehingga
berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan mengembalikan `Results`
yang sama. Menjalankan graph pada runtime mandiri tanpa memasang LibreYOLO juga didukung,
tetapi prapemrosesan dan pascapemrosesannya harus ditulis sendiri.

<code-tabs name="export" />

## Checkpoint

Setiap berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
