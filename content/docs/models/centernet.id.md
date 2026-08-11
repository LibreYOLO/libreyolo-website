---
title: CenterNet
families:
  - centernet
seo_title: 'CenterNet: deteksi objek di LibreYOLO'
description: >-
  Jalankan CenterNet (Objects as Points) di LibreYOLO dengan backbone ResDCN-18
  dan DLA-34. Prediksi, validasi dan ekspor ke ONNX di bawah lisensi MIT. Tanpa
  jalur pelatihan.
lead: >-
  CenterNet memodelkan objek sebagai titik pusat bounding box-nya dan menghitung
  semua properti lainnya lewat regresi dari puncak heatmap, sehingga tidak butuh
  anchor maupun langkah non-maximum-suppression. LibreYOLO menyediakannya
  sebagai detektor khusus inferensi.
keywords:
  - CenterNet
  - Objects as Points
  - keypoint detection
  - anchor-free detector
  - deteksi objek anchor free
  - ResDCN-18
  - DLA-34
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCenterNetresdcn18.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreCenterNetresdcn18.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: DLA-34
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCenterNetdla34.pt")
        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCenterNetresdcn18.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreCenterNetresdcn18.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreCenterNetresdcn18.pt")


        # Ekspor ONNX butuh opset 16 atau lebih baru: tahap upsampling

        # deformable-convolution diturunkan ke GridSample, yang diperkenalkan
        opset 16.

        model.export(format="onnx", opset=18)

        model.export(format="tensorrt")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreCenterNetresdcn18.pt format=onnx opset=18
    - label: Memakai berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Factory-nya memilih berdasarkan sufiks berkas, jadi artefak hasil
        ekspor

        # dimuat seperti checkpoint biasa dan mengembalikan objek Results yang
        sama.

        model = LibreYOLO("LibreCenterNetresdcn18.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 20aaef83cc95590d
---

## Instalasi

CenterNet tidak memerlukan extra opsional. Semua yang diimpornya sudah tersedia
di instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali dipakai lalu disimpan di
cache lokal.

<code-tabs name="predict" />

Objek `Results` yang dikembalikan sama dengan yang dikembalikan setiap family,
jadi mengganti detektor dengan yang lain hanya butuh perubahan satu baris.
`conf` dan `max_det` menyaring puncak heatmap yang sudah diurutkan; `iou` tetap
diterima demi keseragaman API tetapi tidak berpengaruh, karena decode top-k
puncak pada CenterNet tidak memerlukan langkah supresi berbasis IoU antar box.
Lihat [prediksi](/docs/predict) untuk source, streaming dan penanganan hasil.

## Varian

Dua backbone. `resdcn18` memadukan trunk ResNet-18 dengan upsampling
deformable-convolution; `dla34` memadukan trunk DLA-34 dengan upsampling
iterative deep-aggregation. Keduanya memberi masukan ke tiga head dense yang
sama (heatmap, lebar/tinggi, offset) dan kanvas input yang sama.

## Validasi

`val()` mengembalikan dictionary berisi key `metrics/` yang mencakup presisi,
recall, mAP 50 dan mAP 50-95, diukur terhadap dataset apa pun dalam format yang
dipakai saat pelatihan.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Ekspor ONNX membutuhkan opset 16 atau lebih baru: tahap upsampling
deformable-convolution pada kedua backbone diturunkan menjadi operator ONNX
`GridSample`, yang diperkenalkan opset 16. Meminta opset yang lebih lama akan
memunculkan error sebelum tracing dimulai.

<code-tabs name="export" />

## Checkpoint

Semua berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box>

Graf ResDCN-18 juga mengkredit human-pose-estimation.pytorch milik Microsoft
yang berlisensi MIT, dan graf DLA-34 mengkredit implementasi DLA berlisensi
BSD-3-Clause milik Fisher Yu. LibreYOLO tidak menyertakan ekstensi DCNv2 asli
yang dipakai proyek upstream; eksekusi native memakai `deform_conv2d`
BSD-3-Clause dari torchvision, dan implementasi portabel khusus ekspor ditulis
terpisah untuk LibreYOLO.

</provenance-box>

## Sitasi

<citation-block />
