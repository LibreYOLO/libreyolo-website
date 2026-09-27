---
title: Deformable DETR
families:
  - deformable_detr
seo_title: 'Deformable DETR: prediksi dan ekspor, Apache-2.0'
description: >-
  Jalankan Deformable DETR di LibreYOLO untuk deteksi objek. Instalasi,
  prediksi, validasi dan ekspor lima ukuran sparse-attention, semuanya
  berlisensi Apache-2.0.
lead: >-
  Deformable DETR mengganti cross-attention padat milik DETR dengan sampling
  multi-skala yang sparse di sekitar setiap titik referensi, dan itulah yang
  membuat detektor transformer praktis untuk dilatih. LibreYOLO menyediakan lima
  ukuran untuk deteksi, hanya untuk inferensi.
keywords:
  - Deformable DETR
  - detection transformer
  - sparse attention
  - deteksi objek python
  - object detection python
  - SenseTime
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDeformableDETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDeformableDETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeformableDETRr50.pt")

        # val() mengembalikan dict biasa, bukan objek
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDeformableDETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDeformableDETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDeformableDETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDeformableDETRr50.pt format=tensorrt
        imgsz=800 half=True
    - label: Memakai berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory-nya memilih berdasarkan sufiks berkas, jadi artefak hasil
        ekspor

        # dimuat seperti checkpoint biasa dan mengembalikan objek Results yang
        sama.

        model = LibreYOLO("LibreDeformableDETRr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 35225efc54b5ef91
---

## Instalasi

Deformable DETR tidak memerlukan extra opsional. Semua yang diimpornya sudah
tersedia di instalasi dasar, memakai inti multi-scale deformable attention
berbasis PyTorch murni.

```bash
pip install libreyolo
```

Memasang `libreyolo[hub-kernels]` bersifat opsional. Begitu paket `kernels`
tersedia, LibreYOLO mengambil kernel multi-scale deformable attention yang sudah
dikompilasi dari Hugging Face Hub saat runtime dan memakainya sebagai pengganti
inti PyTorch murni; `LIBREYOLO_HUB_KERNELS=0` mematikannya kembali.

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali dipakai lalu disimpan di
cache lokal.

<code-tabs name="predict" />

Objek `Results` yang dikembalikan sama dengan yang dikembalikan setiap family,
jadi mengganti detektor dengan yang lain hanya butuh perubahan satu baris.
`conf` dan `max_det` menyaring seleksi query; `iou` tetap diterima demi
keseragaman API tetapi tidak berpengaruh, karena decoder-nya adalah set
predictor tanpa langkah NMS. Lihat [prediksi](/docs/predict) untuk sumber,
streaming dan penanganan hasil.

Deformable DETR hanya tersedia untuk inferensi di LibreYOLO. Upstream melatihnya
dengan Hungarian matching dan focal classification loss; resep itu tidak
diimplementasikan di sini, jadi `train()` memunculkan `NotImplementedError`.

## Varian

Lima checkpoint mencakup seluruh konfigurasi yang dirilis, semuanya pada
resolusi input yang sama. `r50ss` membatasi attention pada satu skala feature
saja; `r50ssdc5` menambahkan tahap backbone C5 berdilatasi di atasnya. `r50`
adalah konfigurasi multi-skala default, dengan sampling di empat level feature
map. `r50refine` menambahkan penghalusan bounding box iteratif di seluruh
lapisan decoder, dan `r50twostage` menghasilkan region proposal awalnya dari
keluaran encoder, bukan dari query yang dipelajari.

## Validasi

`val()` mengembalikan dictionary berisi key `metrics/` yang mencakup presisi,
recall, mAP 50 dan mAP 50-95, diukur terhadap dataset apa pun dalam format yang
dipakai saat pelatihan.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Artefak hasil ekspor dimuat kembali lewat `LibreYOLO()` berdasarkan sufiks
berkasnya, jadi berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan
mengembalikan `Results` yang sama. [Ekspor](/docs/export) memuat daftar argumen
yang diterima setiap format.

<code-tabs name="export" />

## Checkpoint

Semua berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
