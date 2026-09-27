---
title: API open-vocabulary
seo_title: 'API LibreOpenVocab: alias dan argumen'
description: >-
  Factory LibreOpenVocab, empat family dan semua aliasnya, set_classes, default
  conf per family, serta aturan text_threshold dan iou.
lead: >-
  LibreOpenVocab adalah factory untuk detektor yang dikondisikan oleh teks. List
  kelas merupakan prompt, bukan head tetap, sehingga vocabulary ditetapkan
  dengan set_classes dan model mengembalikan Results deteksi biasa berdasarkan
  vocabulary tersebut.
keywords:
  - LibreOpenVocab
  - deteksi open vocabulary
  - Grounding DINO
  - OWLv2
  - OMDet-Turbo
  - OV-DEIM
  - set_classes
last_verified: 1.5.0
verification: >-
  Alias dibaca dari libreyolo/models/openvocab/__init__.py; repository, ukuran,
  dan ambang batas dari grounding_dino.py, owlv2.py, omdet_turbo.py, dan
  ov_deim.py; aturan pemanggilan dari libreyolo/models/openvocab/base.py,
  semuanya pada v1.5.0. Tujuan desain dari
  docs/adr/0008-open-vocab-detector-contract.md.
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[openvocab]'
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-tiny")
        model.set_classes(["person", "skateboard", "handrail"])

        result = model.predict(SAMPLE_IMAGE)
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            print(result.names[int(cls)], box.tolist())
source_hash: 64e4c641c6f8cde0
---

## Instalasi

Tier ini memerlukan ekstra `openvocab`.

<code-tabs name="install" />

## Factory

```python
LibreOpenVocab(model: str = "grounding-dino-tiny", **kwargs) -> LibreOpenVocabDetector
```

`model` adalah alias, bukan path. Underscore diubah menjadi tanda hubung sebelum
lookup, sehingga nama dengan kualifikasi family yang dicetak oleh inventaris
CLI, seperti `omdet_turbo-t` dan `grounding_dino-t`, dimuat sebagaimana
diberikan. Alias yang tidak dikenal memunculkan `ValueError` yang mencantumkan
semua alias yang diketahui.

Constructor menerima `size`, `nb_classes=80`, `names=None`, `device="auto"`,
`task=None`, dan `text_threshold=None`. Memberikan `names` sama dengan memanggil
`set_classes` segera setelah pemuatan. Memberikan `text_threshold` kepada family
yang tidak mendukungnya memunculkan `TypeError`.

<code-tabs name="usage" />

## Family dan alias

| Family | Alias | Ukuran | Bobot |
|---|---|---|---|
| Grounding DINO | `grounding-dino`, `groundingdino`, `grounding-dino-tiny`, `groundingdino-tiny`, `grounding-dino-t`, `groundingdino-t`, `grounding-dino-base`, `groundingdino-base`, `grounding-dino-b`, `groundingdino-b` | `t`, `b` | `LibreYOLO/LibreGroundingDINOt`, `LibreYOLO/LibreGroundingDINOb` |
| OWLv2 | `owlv2`, `owl-v2`, `owlv2-base`, `owl-v2-base`, `owlv2-b16`, `owl-v2-b16`, `owlv2-large`, `owl-v2-large`, `owlv2-l14`, `owl-v2-l14` | `b16`, `l14` | `LibreYOLO/LibreOWLv2b16`, `LibreYOLO/LibreOWLv2l14` |
| OMDet-Turbo | `omdet-turbo`, `omdet`, `omdetturbo`, `omdet-turbo-tiny`, `omdet-turbo-swin-tiny`, `omdet-turbo-t` | `t` | `LibreYOLO/LibreOMDetTurbot` |
| OV-DEIM | `ov-deim`, `ovdeim`, `ov-deim-s`, `ovdeim-s`, `ov-deim-m`, `ovdeim-m`, `ov-deim-l`, `ovdeim-l` | `s`, `m`, `l` | `LibreYOLO/LibreOVDEIMs`, `LibreYOLO/LibreOVDEIMm`, `LibreYOLO/LibreOVDEIMl` |

Alias default adalah `grounding-dino-tiny`.

`LibreGroundingDINO`, `LibreOWLv2`, dan `LibreOMDetTurbo` diekspor pada level
package dan dapat dibuat secara langsung dengan `size=`. OV-DEIM dapat diakses
melalui alias factory di atas.

## set_classes

```python
model.set_classes(classes: list[str]) -> LibreOpenVocabDetector
```

Menetapkan vocabulary untuk setiap pemanggilan `predict()` berikutnya, serta
mengembalikan model agar pemanggilan dapat dirangkai. List tidak boleh kosong,
hanya boleh memuat string, dan elemennya harus unik saat dibandingkan tanpa
memedulikan kapitalisasi; label kosong ditolak. Memberikan string langsung
memunculkan `TypeError` karena string tersebut akan diuraikan menjadi kelas satu
karakter.

Setelah pemanggilan, `model.names` memetakan `0..N-1` ke label dalam urutan yang
diberikan, dan `model.nb_classes` bernilai `N`.

## Argumen pemanggilan

Tier ini menggunakan kembali antarmuka predict standar dengan tiga perbedaan.

Default `conf` menggunakan nilai milik family, bukan nilai bersama 0.25:

| Family | Default conf | Suppression |
|---|---|---|
| Grounding DINO | 0.25 | |
| OWLv2 | 0.1 | |
| OMDet-Turbo | 0.3 | Postprocessing sendiri, ambang batas 0.5, mematuhi `iou=` |
| OV-DEIM | 0.25 | Pencocokan one-to-one dengan pemilihan top-K, tanpa suppression |

`iou=` hanya bermakna untuk family yang menjalankan suppression. OMDet-Turbo
menerima ambang batas sebagai argumen dan memiliki default 0.5 ketika `iou=`
tidak ditetapkan. Tiga family lainnya tidak melakukan suppression, sehingga
memberikan `iou=` akan memunculkan peringatan dan diabaikan.

`text_threshold=` hanya untuk Grounding DINO, dengan default 0.25. Nilai ini
dapat diberikan saat konstruksi sebagai nilai persisten atau per pemanggilan.
Nilai per pemanggilan tidak dapat digabungkan dengan `stream=True` karena hasil
stream dibuat secara lazy; tetapkan pada constructor sebagai gantinya. Semua
family lain memunculkan `TypeError` jika menerimanya.

`imgsz=` memunculkan `ValueError`: pipeline preprocessing mengendalikan
pengubahan ukuran untuk tier ini. `augment=True` juga memunculkan error karena
augmentasi saat pengujian berada di luar cakupan. Ukuran input dicatat per
family hanya sebagai referensi: Grounding DINO 800, OWLv2 960 dan 1008,
OMDet-Turbo 640, OV-DEIM 640.

## Tidak didukung

`train()`, `val()`, `track()`, dan `export()` semuanya memunculkan
`NotImplementedError`. Lakukan fine-tuning di upstream dan muat bobot yang
dihasilkan; jalankan `predict()` per frame sebagai pengganti tracking. Validasi
akan memerlukan validator khusus karena validator deteksi bersama memanggil model
dengan tensor gambar, sedangkan tier ini memerlukan input yang dikondisikan oleh
teks.
