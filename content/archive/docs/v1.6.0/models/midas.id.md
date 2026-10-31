---
title: MiDaS
families:
  - midas
seo_title: 'MiDaS: estimasi kedalaman monokular di LibreYOLO'
description: Jalankan inferensi kedalaman relatif MiDaS di LibreYOLO. Checkpoint s dan l memakai
  mirror LibreYOLO dengan izin MIT dari penerbit.
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
last_verified: 1.6.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Mengunduh checkpoint dari mirror saat pertama kali dipakai.

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
source_hash: 64537aa3ad3a6f8f
---

## Instalasi

MiDaS memerlukan extra `midas` untuk encoder timm-nya.

```bash
pip install "libreyolo[midas]"
```

## Prediksi

Checkpoint s dan l diunduh dari mirror LibreYOLO dengan izin MIT dari penerbit dan disimpan dalam cache lokal.

<code-tabs name="predict" />

`result.depth_map` berisi peta kedalaman invers relatif yang rapat: nilai lebih besar berarti lebih dekat ke kamera, tanpa satuan metrik atau skala yang berlaku antar gambar. `save=True` menulis visualisasi peta berwarna ke disk; `Results.plot()` merender peta kedalaman. Lihat [prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

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
