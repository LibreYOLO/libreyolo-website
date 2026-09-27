---
title: MiDaS
families:
  - midas
seo_title: 'MiDaS: estimasi kedalaman monokular di LibreYOLO'
description: >-
  Gunakan MiDaS di LibreYOLO untuk estimasi kedalaman monokular. Instal,
  prediksi, validasi, dan ekspor dua varian berlisensi MIT yang diunduh dari
  isl-org.
lead: >-
  MiDaS adalah estimasi kedalaman relatif monokular yang dilatih dengan loss
  invarian skala dan pergeseran pada campuran dataset. Rangkaian pekerjaan ini
  membentuk protokol transfer kedalaman zero-shot yang digunakan kembali oleh
  family berikutnya. LibreYOLO mendukungnya untuk task kedalaman: prediksi dan
  validasi zero-shot, tanpa jalur pelatihan.
keywords:
  - MiDaS
  - estimasi kedalaman monokular
  - DPT
  - relative depth
  - depth map
  - zero-shot depth
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Belum tersedia di disk: LibreYOLO mengunduhnya dari rilis GitHub resmi

        # isl-org/MiDaS dan memeriksanya terhadap SHA-256 yang dipatok sebelum
        digunakan.

        model = LibreYOLO("LibreMiDaSl-depth.pt")

        result = model(SAMPLE_IMAGE, save=True)


        depth = result.depth_map

        print(depth.min, depth.max, depth.mean)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreMiDaSl-depth.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Varian kecil
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Encoder EfficientNet-Lite3, lebih kecil dan cepat daripada ukuran
        DPT-Large l.

        model = LibreYOLO("LibreMiDaSs-depth.pt")

        result = model(SAMPLE_IMAGE, save=True)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMiDaSl-depth.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/abs_rel"])
        print(metrics["metrics/rmse"])
        print(metrics["metrics/delta1"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreMiDaSl-depth.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreMiDaSl-depth.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreMiDaSl-depth.pt format=onnx
        libreyolo export model=LibreMiDaSl-depth.pt format=tensorrt half=True
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan akhiran berkas, sehingga artefak hasil
        ekspor dimuat

        # seperti checkpoint lain dan mengembalikan objek Results yang sama.

        model = LibreYOLO("LibreMiDaSl-depth.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.depth_map.data.shape)
source_hash: ce2fbf3ae43e9be4
---

## Instalasi

MiDaS tidak memerlukan extra opsional. Semua yang diimpornya tersedia dalam
instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

MiDaS adalah satu-satunya family kedalaman yang tidak dipublikasikan ulang
LibreYOLO di organisasi Hugging Face miliknya. Meminta checkpoint berdasarkan
nama berkas LibreYOLO akan mengunduh aset resmi yang cocok langsung dari rilis
GitHub `isl-org/MiDaS`, memeriksanya terhadap SHA-256 yang dipatok, lalu
membungkusnya dengan metadata checkpoint LibreYOLO sebelum penggunaan pertama.
Run berikutnya menggunakan kembali berkas lokal dalam cache. Lihat Lisensi
untuk alasannya.

<code-tabs name="predict" />

`result.depth_map` memuat peta inverse-depth relatif yang padat. Nilai lebih
tinggi berarti lebih dekat ke kamera, dan nilainya tidak memiliki satuan metrik
atau skala lintas gambar. `save=True` menulis visualisasi berpeta warna dari peta
tersebut ke disk. `Results.plot()` tidak mencakup family ini karena hanya
didefinisikan untuk surface normal dan edge. Lihat [prediksi](/docs/predict)
untuk sumber, streaming, dan penanganan hasil.

## Varian

Ada dua varian dengan encoder berbeda, bukan sekadar skala berbeda dari encoder
yang sama. `s` adalah MiDaS v2.1 Small dengan encoder EfficientNet-Lite3. `l`
adalah DPT-Large dengan encoder ViT-L/16 dan decoder DPT yang diperkenalkan MiDaS
untuk prediksi padat. Preprocessing-nya juga berbeda. `s` memakai resize aspek
upper-bound dengan normalisasi mean/std ImageNet, sedangkan `l` memakai resize
aspek minimal dengan mean dan std 0.5. Pilih `s` untuk CNN yang lebih ringan,
atau `l` untuk akurasi decoder transformer.

Pelatihan tidak tersedia untuk family ini. `LibreMiDaS.train()` selalu
memunculkan `NotImplementedError`.

## Validasi

`val()` menjalankan validator kedalaman bersama. Setiap prediksi disejajarkan
dengan ground truth melalui skala dan pergeseran least-squares per gambar, lalu
validator melaporkan metrik kedalaman relatif zero-shot standar: AbsRel, RMSE,
dan tiga ambang batas delta.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Artefak hasil ekspor dapat dimuat kembali melalui `LibreYOLO()` berdasarkan
akhiran berkasnya, sehingga berkas `.onnx` atau `.engine` berperilaku seperti
checkpoint dan mengembalikan `Results` yang sama, dengan `depth_map` sebagai
pengganti box.

<code-tabs name="export" />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
