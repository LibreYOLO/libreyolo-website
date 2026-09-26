---
title: EfficientDet
families:
  - efficientdet
seo_title: 'EfficientDet: deteksi objek di LibreYOLO'
description: >-
  Jalankan EfficientDet D0-D4 di LibreYOLO: detektor BiFPN untuk prediksi,
  validasi dan ekspor ke ONNX, TensorRT dan OpenVINO di bawah lisensi
  Apache-2.0.
lead: >-
  EfficientDet memadukan backbone EfficientNet dengan bi-directional feature
  pyramid network (BiFPN) yang diulang, lalu menskalakan kedalaman, lebar dan
  resolusi secara bersamaan pada lima ukuran. LibreYOLO menyediakannya sebagai
  detektor khusus inferensi.
keywords:
  - EfficientDet
  - BiFPN
  - EfficientNet
  - object detection
  - deteksi objek python
  - compound scaling
  - deteksi objek efisien
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEfficientDetd0.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreEfficientDetd0.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientDetd0.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEfficientDetd0.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEfficientDetd0.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreEfficientDetd0.pt format=onnx
        libreyolo export model=LibreEfficientDetd0.pt format=tensorrt half=True
    - label: Memakai berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Factory-nya memilih berdasarkan sufiks berkas, jadi artefak hasil
        ekspor

        # dimuat seperti checkpoint biasa dan mengembalikan objek Results yang
        sama.

        model = LibreYOLO("LibreEfficientDetd0.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 12c61fb0035437ce
---

## Instalasi

EfficientDet tidak memerlukan extra opsional. Semua yang diimpornya sudah
tersedia di instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali dipakai lalu disimpan di
cache lokal.

<code-tabs name="predict" />

Objek `Results` yang dikembalikan sama dengan yang dikembalikan setiap family,
jadi mengganti detektor dengan yang lain hanya butuh perubahan satu baris.
EfficientDet melakukan decode kandidat berbasis anchor lalu menjalankan
non-maximum suppression per kelas, sehingga `conf`, `iou` dan `max_det`
benar-benar berpengaruh di sini. Lihat [prediksi](/docs/predict) untuk source,
streaming dan penanganan hasil.

## Varian

Lima ukuran, D0 sampai D4. Setiap tingkat memadukan backbone EfficientNet yang
lebih besar dengan BiFPN yang lebih dalam dan lebih lebar serta head prediksi
yang lebih dalam, sehingga jumlah parameter dan beban komputasi tumbuh
bersamaan, mengikuti aturan compound scaling pada papernya.

## Validasi

`val()` mengembalikan dictionary berisi key `metrics/` yang mencakup presisi,
recall, mAP 50 dan mAP 50-95, diukur terhadap dataset apa pun dalam format yang
dipakai saat pelatihan.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Artefak hasil ekspor dimuat kembali lewat `LibreYOLO()` berdasarkan sufiks
berkasnya, jadi berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan
mengembalikan `Results` yang sama.

<code-tabs name="export" />

## Checkpoint

Semua berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box>

Checkpoint D0-D4 milik LibreYOLO dikonversi melalui proyek
rwightman/efficientdet-pytorch yang berlisensi Apache-2.0, dan proyek itu
sendiri mencerminkan bobot resmi hasil pelatihan TensorFlow dari google/automl
tanpa mengubah tensor yang sudah dipelajari. Tidak ada kode dari proyek
zylo117/Yet-Another-EfficientDet-Pytorch yang berlisensi LGPL yang dirujuk
maupun dipakai.

</provenance-box>
