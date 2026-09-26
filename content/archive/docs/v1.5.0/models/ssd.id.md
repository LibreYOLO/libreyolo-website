---
title: SSD
families:
  - ssd
seo_title: 'SSD (SSD300): deteksi objek di LibreYOLO'
description: >-
  Jalankan SSD300 di LibreYOLO: detektor VGG16 single-shot untuk prediksi,
  validasi, dan ekspor ONNX di bawah BSD-3-Clause. Tidak ada jalur pelatihan.
lead: >-
  SSD (Single Shot MultiBox Detector) memprediksi setiap kotak dan skor kelas
  dari grid padat default box dalam satu forward pass, tanpa tahap region
  proposal terpisah. LibreYOLO menyediakan checkpoint SSD300 berbasis VGG16
  sebagai detektor khusus inferensi.
keywords:
  - SSD
  - SSD300
  - Single Shot MultiBox Detector
  - deteksi objek
  - VGG16
  - anchor-based detector
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSSD300.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSSD300.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSSD300.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSSD300.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreSSD300.pt")


        # imgsz sengaja dihilangkan di sini: SSD300 di-trace pada canvas native

        # checkpoint, dan nilai lain akan memunculkan error sebelum ekspor
        dimulai.

        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreSSD300.pt format=onnx
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Factory merutekan berdasarkan sufiks berkas, sehingga artefak hasil
        ekspor

        # dimuat seperti checkpoint lain dan mengembalikan objek Results yang
        sama.

        model = LibreYOLO("LibreSSD300.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 3b3f9ea72291c4fa
---

## Instalasi

SSD tidak memerlukan komponen tambahan opsional. Semua impornya tersedia dalam instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam cache lokal.

<code-tabs name="predict" />

Objek `Results` yang dikembalikan sama dengan yang dikembalikan setiap family, jadi mengganti
detektor hanya memerlukan perubahan satu baris. SSD mendekode grid default box dengan skor
per kelas, lalu menjalankan non-maximum suppression, sehingga `conf`, `iou`, dan `max_det`
semuanya benar-benar berpengaruh di sini, tidak seperti detektor berbasis query dalam library
ini. Lihat [prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Varian

SSD menyediakan satu checkpoint: jaringan SSD300 berbasis VGG16 pada canvas native tetap.
Tidak ada pilihan ukuran atau skala dalam family ini; prediksi, validasi, dan ekspor semuanya
memakai satu graph tersebut.

Berkas bobotnya adalah `LibreSSD300.pt`, yaitu prefiks family yang diikuti satu-satunya key
ukuran, `"300"`. Kelas di baliknya adalah `LibreSSD`, sehingga konstruksi langsungnya adalah
`LibreSSD(size="300")`, bukan kelas yang dinamai berdasarkan berkas.

## Validasi

`val()` mengembalikan dictionary dengan key `metrics/` yang mencakup presisi, recall,
mAP 50, dan mAP 50-95, yang diukur terhadap dataset apa pun dalam format yang digunakan
untuk pelatihan.

<code-tabs name="val" />

## Ekspor

<export-matrix />

SSD hanya dapat diekspor ke ONNX; semua format lain saat ini diblokir untuk family ini.
Ekspor selalu memakai canvas native checkpoint, dan graph mengekspos head mentah SSD yang
dipaketkan, bukan output non-maximum-suppression gabungan, sehingga `nms=True` tidak diterima
saat ekspor. Backend LibreYOLO sendiri menjalankan tahap dekode dan suppression setelah
graph dimuat kembali.

<code-tabs name="export" />

## Checkpoint

Setiap berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box>

Kode SSD300 LibreYOLO tidak di-porting dari rilis Caffe milik penulis makalah; kode tersebut
berasal dari implementasi SSD300 BSD-3-Clause milik torchvision, dan itulah repositori yang
ditautkan di atas sebagai sumber upstream. Bobot backbone VGG16 dapat ditelusuri lebih jauh
ke VGGNet reduced yang sepenuhnya konvolusional dari Oxford, dirilis di bawah CC BY 4.0 oleh
Karen Simonyan dan Andrew Zisserman.

</provenance-box>

## Sitasi

<citation-block />
