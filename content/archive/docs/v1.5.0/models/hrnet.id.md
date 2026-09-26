---
title: HRNet
families:
  - hrnet
seo_title: 'HRNet: estimasi pose top-down di LibreYOLO'
description: >-
  Gunakan HRNet di LibreYOLO untuk estimasi pose top-down COCO-17. Instal,
  prediksi, validasi, dan ekspor checkpoint W32 dan W48 berlisensi MIT.
lead: >-
  HRNet adalah jaringan konvolusional yang mempertahankan stream fitur
  beresolusi tinggi melalui fusi multi-scale berulang, alih-alih memulihkan
  resolusi setelah downsampling. LibreYOLO membungkus varian pose top-down resmi
  untuk inferensi dan validasi.
keywords:
  - HRNet
  - estimasi pose manusia
  - top-down pose
  - keypoint COCO-17
  - high-resolution network
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Tidak ada sumber orang yang diberikan. HRNet otomatis dipasangkan
        dengan detector

        # LibreYOLO9t yang ringan dan mencatat pilihan itu satu kali.

        model = LibreYOLO("LibreHRNetw32-pose.pt")

        result = model(SAMPLE_IMAGE, save=True)


        print(result.keypoints.xy)

        print(result.boxes.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreHRNetw32-pose.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Sumber orang
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreHRNetw32-pose.pt")


        # Lewati deteksi sepenuhnya: perlakukan seluruh gambar sebagai satu
        orang.

        result = model(SAMPLE_IMAGE, cropped=True)


        # Atau berikan HRNet box dari detector yang sudah dijalankan.

        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])


        # Atau pasangkan dengan detector LibreYOLO tertentu sebagai pengganti

        # LibreYOLO9t default.

        result = model(SAMPLE_IMAGE, person_detector="rfdetr")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHRNetw32-pose.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreHRNetw32-pose.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreHRNetw32-pose.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreHRNetw32-pose.pt format=onnx
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # Graph hasil ekspor hanyalah head heatmap berkanvas tetap. Inputnya
        adalah

        # satu batch crop orang yang sudah dipotong dan dinormalisasi, lalu

        # mengembalikan heatmap mentah. Deteksi orang, geometri crop, decoding

        # heatmap, dan suppression OKS tidak termasuk dalam graph ini.
        Menjalankannya

        # di luar LibreYOLO berarti harus mengimplementasikan sendiri langkah
        decode itu.

        session = ort.InferenceSession("LibreHRNetw32-pose.onnx")

        name = session.get_inputs()[0].name

        heatmaps = session.run(
            None, {name: np.zeros((1, 3, 256, 192), dtype=np.float32)}
        )[0]
source_hash: 5a5540fd54ee6f23
---

## Instalasi

HRNet tidak memerlukan extra selain paket dasar.

```bash
pip install libreyolo
```

Detector orang default-nya, checkpoint LibreYOLO9t yang ringan, diunduh
otomatis saat HRNet pertama kali dipasangkan dengannya.

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam
cache lokal.

<code-tabs name="predict" />

HRNet adalah estimator pose top-down. Model ini memerlukan box orang sebelum
head pose dapat berjalan, sehingga setiap panggilan menentukan satu sumber box.
Jika dibiarkan, HRNet memasangkan dirinya dengan detector LibreYOLO9t saat
pertama kali digunakan dan mencatat pilihan tersebut. `cropped=True` melewati
deteksi dan memperlakukan seluruh gambar sebagai satu orang. `person_boxes`
menerima box dari detector yang sudah dijalankan. `person_detector` menerima
`"auto"`, `"rfdetr"`, model deteksi LibreYOLO apa pun, atau callable biasa.
`flip_test=True` juga menjalankan model pada crop yang dibalik secara horizontal
dan merata-ratakan kedua heatmap, yaitu augmentasi waktu pengujian milik HRNet.
`augment=True` generik tidak didefinisikan di sini. Sumber multi-gambar berjalan
secara berurutan. Detector HRNet dan jumlah orang per gambar yang bervariasi
tidak mendukung prediksi bertumpuk. Lihat [prediksi](/docs/predict) untuk sumber,
streaming, dan penanganan hasil.

## Varian

Ada dua ukuran, `w32` dan `w48`, yang keduanya memprediksi set keypoint COCO-17
standar dari crop orang beresolusi tetap. `w48` adalah backbone yang lebih lebar.

Model zoo upstream melaporkan akurasi pose untuk setiap ukuran dengan detector
orangnya sendiri, pengaturan flip testing sendiri, dan protokol evaluasi COCO
resmi. Pasangan default LibreYOLO menggunakan detector lain, sehingga run
validasi di sini mengukur kombinasi tersebut, bukan kombinasi upstream.
Mencocokkan angka upstream memerlukan box orang, skor detector, dan pengaturan
flip yang sama dengan evaluasi aslinya.

## Validasi

`val()` menjalankan keypoint OKS-AP bergaya COCO dan menerima `data.yaml`
YOLO-pose atau JSON keypoint COCO beserta direktori gambar. Backend metrik
default adalah faster-coco-eval, dengan `pycocotools` digunakan otomatis bila
faster-coco-eval tidak terpasang. `faster_coco_eval=False` memaksa jalur
`pycocotools`.

<code-tabs name="val" />

Validasi menjalankan `predict()` milik HRNet secara internal, sehingga memakai
detector orang yang digunakan saat model dibuat atau dipanggil. Buat model
dengan `person_detector=` yang eksplisit agar sumber tersebut tetap sama pada
setiap run, alih-alih membiarkan setiap panggilan menentukan ulang nilai
default.

## Ekspor

<export-matrix />

Kontrak ekspor HRNet hanya mencakup ONNX, TorchScript, OpenVINO, dan TensorRT.
Format lain memunculkan error sebelum tracing dimulai. Setiap ekspor hanyalah
head heatmap berkanvas tetap, batch satu, dan FP32, yang menerima crop orang dan
mengembalikan heatmap mentah. Geometri crop affine sebelumnya serta decoding
heatmap, pemulihan flip, dan suppression OKS setelahnya tetap berada dalam
Python. Karena itu, pipeline lengkap dari gambar ke keypoint tetap memerlukan
LibreYOLO di sisi lain.

<code-tabs name="export" />

## Checkpoint

Semua berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
