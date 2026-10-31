---
title: YOLOv7
families:
  - yolo7
seo_title: 'YOLOv7 di LibreYOLO: prediksi, latih, dan ekspor di bawah MIT'
description: >-
  Jalankan YOLOv7 di LibreYOLO untuk deteksi objek: instal, prediksi, latih,
  validasi, dan ekspor, dengan kode serta bobot berlisensi MIT.
lead: >-
  YOLOv7 adalah detektor satu tahap berbasis anchor yang head-nya menambahkan
  offset pengetahuan implisit yang dipelajari sebelum konvolusi akhir. LibreYOLO
  mendukung satu ukuran yang dipublikasikan untuk deteksi.
keywords:
  - YOLOv7
  - deteksi objek
  - anchor-based detection
  - implicit knowledge
  - ImplicitA
  - real-time object detection
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO7b.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreYOLO7b.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO7b.pt")

        model.train(data="my-dataset.yaml", epochs=300, imgsz=640, batch=16,
        lr0=0.01)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO7b.pt data=my-dataset.yaml \
          epochs=300 imgsz=640 batch=16 lr0=0.01
    - label: Warm start dari model baru
      language: python
      code: >
        from libreyolo import LibreYOLO7


        # pretrained=True selalu memuat checkpoint LibreYOLO7b.pt yang
        dipublikasikan,

        # terlepas dari bobot yang digunakan untuk membuat instance ini. Membuat
        kelas

        # secara langsung, bukan melalui LibreYOLO(), dimulai tanpa bobot yang
        dimuat.

        model = LibreYOLO7(None, size="b")

        model.train(data="my-dataset.yaml", epochs=300, pretrained=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO7b.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO7b.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreYOLO7b.pt format=onnx imgsz=640

        libreyolo export model=LibreYOLO7b.pt format=tensorrt imgsz=640
        half=True
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan sufiks berkas, sehingga artefak hasil
        ekspor

        # dimuat seperti checkpoint lain dan mengembalikan objek Results yang
        sama.

        model = LibreYOLO("LibreYOLO7b.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 361e81de5614a571
---

## Instalasi

YOLOv7 tidak memerlukan komponen tambahan selain paket dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam cache lokal.

<code-tabs name="predict" />

Objek `Results` yang dikembalikan sama dengan yang dikembalikan setiap family, jadi mengganti
detektor hanya memerlukan perubahan satu baris. `conf` menetapkan ambang batas confidence
dan `iou` menetapkan ambang NMS yang diterapkan setelah head berbasis anchor didekode.
Lihat [prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Varian

LibreYOLO menyediakan satu ukuran, `b`. Upstream menerbitkan satu model YOLOv7, sehingga
tidak ada ukuran lain untuk dipilih.

## Pelatihan

<code-tabs name="train" />

`pretrained` dibaca, berbeda dari argumen bernama sama yang tidak berpengaruh pada beberapa
family lain di sini: berikan `True` untuk melakukan warm start dari checkpoint
`LibreYOLO7b.pt` yang dipublikasikan (diunduh otomatis), atau path maupun nama untuk yang
lain. Checkpoint terbitan tersebut memakai COCO 80 kelas, sehingga memintanya pada model
yang sudah dibangun ulang untuk jumlah kelas berbeda akan membangun ulang model ke 80 kelas,
memuatnya, lalu mentransfer setiap tensor dengan bentuk yang cocok ke jumlah head target
setelah jumlah kelas dataset dibaca. `resume=True` tidak dapat digabungkan dengan
`pretrained`. Jika dibiarkan pada default `None`, pelatihan berlanjut dari bobot yang
digunakan saat model dibuat, atau dari inisialisasi acak jika tidak ada yang dimuat.

Jika pengaturan lain dibiarkan, pelatih berjalan selama 300 epoch pada `lr0=0.01` dengan
momentum SGD 0.937, warmup 3 epoch, serta assignment SimOTA dan tahap akhir tanpa augmentasi
selama 15 epoch yang sama seperti YOLOX, disesuaikan untuk head berbasis anchor. Satu
perbedaannya: YOLOX menambahkan penyempurnaan regresi kotak L1 selama epoch akhir tersebut,
sedangkan v7 melewatinya karena loss SimOTA v7 tidak memiliki cabang L1 raw-offset untuk
disempurnakan.

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
didukung, tetapi prapemrosesan dan pascapemrosesannya harus ditulis sendiri.

<code-tabs name="export" />

## Checkpoint

Setiap berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
