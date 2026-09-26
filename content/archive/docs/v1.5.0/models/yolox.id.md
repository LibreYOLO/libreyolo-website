---
title: YOLOX
families:
  - yolox
seo_title: 'YOLOX: prediksi, latih, dan ekspor di bawah Apache-2.0'
description: >-
  Gunakan YOLOX di LibreYOLO untuk deteksi objek: instal, prediksi, latih,
  validasi, dan ekspor di bawah Apache-2.0.
lead: >-
  YOLOX adalah detektor satu tahap anchor-free dengan head klasifikasi-regresi
  terpisah, yang dilatih dengan assignment label SimOTA. LibreYOLO mendukungnya
  untuk deteksi.
keywords:
  - YOLOX
  - deteksi objek
  - anchor-free detection
  - decoupled head
  - SimOTA
  - real-time object detection
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLOXs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLOXs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLOXs.pt")

        model.train(data="my-dataset.yaml", epochs=300, imgsz=640, batch=16,
        lr0=0.01)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLOXs.pt data=my-dataset.yaml \
          epochs=300 imgsz=640 batch=16 lr0=0.01
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLOXs.pt data=my-dataset.yaml
    - label: Dibandingkan dengan COCO
      language: bash
      code: |
        # YAML COCO bawaan menyertakan skrip unduhan tertanam, sehingga
        # memerlukan izin eksplisit kecuali dataset sudah tersedia secara lokal.
        libreyolo val model=LibreYOLOXn.pt data=coco.yaml imgsz=416 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLOXs.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreYOLOXs.pt format=onnx imgsz=640

        libreyolo export model=LibreYOLOXs.pt format=tensorrt imgsz=640
        half=True
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan sufiks berkas, sehingga artefak hasil
        ekspor

        # dimuat seperti checkpoint lain dan mengembalikan objek Results yang
        sama.

        model = LibreYOLO("LibreYOLOXs.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: f5ab735a29f85a95
---

## Instalasi

YOLOX tidak memerlukan komponen tambahan selain paket dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam cache lokal.

<code-tabs name="predict" />

Objek `Results` yang dikembalikan sama dengan yang dikembalikan setiap family, jadi mengganti
detektor hanya memerlukan perubahan satu baris. `conf` menetapkan ambang batas confidence
dan `iou` menetapkan ambang NMS yang diterapkan pada ketiga skala prediksi terpisah. Lihat
[prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Varian

Enam ukuran memakai backbone CSP dan neck PAFPN yang sama. Dua ukuran terkecil, `n` dan `t`,
berjalan pada resolusi input tetap yang lebih kecil daripada empat ukuran lainnya; tabel
benchmark di bawah mencantumkan angka tepat untuk setiap ukuran.

<benchmark-table task="detect" />

<va-embed />

## Pelatihan

<code-tabs name="train" />

Jika dibiarkan, pelatih berjalan selama 300 epoch pada `lr0=0.01` dengan momentum SGD 0.9,
warmup 5 epoch, serta augmentasi mosaic dan mixup yang dinonaktifkan selama 15 epoch terakhir.
`train()` juga menerima argumen `pretrained`, tetapi nilainya tidak pernah dibaca di dalam
metode: pelatihan selalu berlanjut dari bobot yang dipakai saat model dibuat, sehingga
`pretrained=False` tidak menginisialisasi ulang jaringan.

`imgsz` memakai nilai tetap dalam konfigurasi pelatihan dasar secara default, bukan resolusi
native checkpoint yang dimuat. Ini secara khusus memengaruhi checkpoint `n` dan `t`:
melanjutkan pelatihan salah satunya tanpa menetapkan `imgsz` secara eksplisit akan beralih
ke nilai default yang lebih besar, bukan ukuran lebih kecil tempat model dipublikasikan.

Lihat [pelatihan](/docs/train) untuk dataset, augmentasi, multi-GPU, dan logger.

## Validasi

`val()` mengembalikan dictionary dengan key `metrics/` yang mencakup presisi, recall,
mAP 50, dan mAP 50-95, yang diukur terhadap dataset apa pun dalam format yang digunakan
untuk pelatihan.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Artefak hasil ekspor dimuat kembali melalui `LibreYOLO()` berdasarkan sufiks berkasnya,
sehingga berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan mengembalikan
`Results` yang sama. Menjalankan graph pada runtime mandiri tanpa memasang LibreYOLO juga
didukung, tetapi prapemrosesan dan pascapemrosesannya harus ditulis sendiri. Ekspor CoreML
dapat menanam NMS ke dalam graph dengan `nms=True`; YOLOX dan YOLOv9 adalah dua family
yang saat ini menerima flag tersebut.

<code-tabs name="export" />

## Checkpoint

Setiap berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
