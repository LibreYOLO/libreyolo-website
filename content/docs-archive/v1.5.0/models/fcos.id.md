---
title: FCOS
families:
  - fcos
seo_title: 'FCOS di LibreYOLO: prediksi, validasi, dan ekspor'
description: >-
  Jalankan FCOS di LibreYOLO untuk deteksi objek anchor-free. Instal, prediksi,
  validasi, dan ekspor port torchvision ResNet-50/FPN berlisensi BSD-3-Clause.
lead: >-
  FCOS mendeteksi objek per piksel alih-alih mengandalkan sekumpulan anchor box
  yang telah ditentukan, dengan memprediksi box dan skor centerness di setiap
  lokasi pada feature map. LibreYOLO menyediakan port implementasi torchvision
  untuk deteksi.
keywords:
  - FCOS
  - anchor-free detection
  - deteksi objek Python
  - one-stage detector
  - torchvision
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreFCOSr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreFCOSr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCOSr50.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreFCOSr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreFCOSr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="torchscript", imgsz=800)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreFCOSr50.pt format=onnx imgsz=800
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan akhiran berkas, sehingga artefak hasil
        ekspor dimuat

        # seperti checkpoint lain dan mengembalikan objek Results yang sama.

        model = LibreYOLO("LibreFCOSr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 60bd7b8dfd903a8c
---

## Instalasi

FCOS tidak memerlukan extra opsional. Semua yang diimpornya tersedia dalam
instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam
cache lokal.

<code-tabs name="predict" />

Objek `Results` yang dikembalikan sama seperti yang dikembalikan setiap family,
sehingga mengganti detector hanya memerlukan perubahan satu baris. Memanggil
model tanpa argumen ambang batas menerapkan nilai default FCOS yang
dipublikasikan, yaitu `conf=0.2`, `iou=0.6`, dan `max_det=100`. Teruskan salah
satu dari ketiganya untuk mengganti nilai tersebut. FCOS mempertahankan langkah
NMS akhir pada prediksi per pikselnya. Lihat [prediksi](/docs/predict) untuk
sumber, streaming, dan penanganan hasil.

## Varian

Ada satu ukuran: ResNet-50 dengan feature pyramid, satu-satunya varian yang
dikenali family ini.

## Validasi

`val()` mengembalikan dictionary berisi key `metrics/` yang mencakup presisi,
recall, mAP 50, dan mAP 50-95, yang diukur terhadap dataset apa pun dalam format
yang digunakan saat pelatihan.

<code-tabs name="val" />

## Ekspor

<export-matrix />

FCOS dapat diekspor ke ONNX, TorchScript, dan OpenVINO. FCOS mempertahankan
rasio aspek sumber sebelum graph berjalan, sehingga LibreYOLO memaksa
`dynamic=True` untuk jalur ONNX dan OpenVINO apa pun nilai yang diteruskan agar
graph tetap valid untuk bentuk input yang diberi padding. Berkas `.onnx` hasil
ekspor dapat dimuat kembali melalui `LibreYOLO()` berdasarkan akhiran berkasnya
dan mengembalikan `Results` yang sama.

<code-tabs name="export" />

## Checkpoint

Semua berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
