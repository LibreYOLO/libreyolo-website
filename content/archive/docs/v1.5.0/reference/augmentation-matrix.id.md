---
title: Matriks augmentasi
seo_title: Augmentasi yang dipatuhi setiap family LibreYOLO
description: >-
  Dukungan knob augmentasi per family: enam belas knob TrainConfig, tiga status,
  enam arketipe pipeline, dan knob yang diam-diam diabaikan suatu family.
lead: >-
  Menetapkan knob augmentasi tidak menjamin nilainya mencapai pipeline. Halaman
  ini mencatat cara setiap family yang dapat dilatih memperlakukan setiap knob
  pada TrainConfig, menggunakan tabel deklaratif bawaan library sebagai
  satu-satunya sumber kebenaran.
keywords:
  - augmentasi libreyolo
  - mosaic_prob
  - mixup_prob
  - hsv_prob
  - no_aug_epochs
  - matriks dukungan augmentasi
  - knob TrainConfig
last_verified: 1.5.0
verification: >-
  List knob, status, arketipe, deviasi per family, dan fungsi helper dibaca dari
  libreyolo/data/augment/spec.py pada v1.5.0. Tabel tersebut dikaitkan ke
  pipeline nyata oleh tests/unit/test_augment_spec.py.
snippets:
  usage:
    - label: Tanyakan langsung kepada spec
      language: python
      code: |
        from libreyolo.data.augment.spec import (
            AUG_KNOBS,
            aug_support,
            ignored_aug_params,
            uses_mosaic_gating,
        )

        print(sorted(AUG_KNOBS))

        table = aug_support("yolo9")
        print(table["mixup_prob"].status, table["mixup_prob"].note)

        print(sorted(ignored_aug_params("dfine")))
        print(uses_mosaic_gating("yolo9"), uses_mosaic_gating("yolonas"))
source_hash: d2e1b9f5c81072e1
---

## Knob

Berikut adalah nama field `TrainConfig`, bukan ejaan CLI. CLI memetakan aliasnya
sendiri ke field tersebut, sehingga `--mosaic` menetapkan `mosaic_prob`.

| Knob | Arti |
|---|---|
| `mosaic_prob` | Probabilitas membangun sampel mosaic 4 gambar |
| `mixup_prob` | Probabilitas memadukan sampel kedua |
| `hsv_prob` | Probabilitas jitter warna HSV |
| `flip_prob` | Probabilitas pembalikan horizontal |
| `degrees` | Rentang rotasi acak untuk affine warp, dalam derajat |
| `translate` | Fraksi translasi acak untuk affine warp |
| `mosaic_scale` | Rentang skala acak untuk affine warp |
| `mixup_scale` | Rentang skala jitter yang diterapkan pada gambar pasangan MixUp |
| `shear` | Rentang shear acak untuk affine warp, dalam derajat |
| `perspective` | Magnitudo projective warp untuk affine warp |
| `flipud` | Probabilitas pembalikan vertikal |
| `no_aug_epochs` | Epoch terakhir yang dilatih dengan augmentasi kuat dinonaktifkan |
| `auto_augment` | Kebijakan AutoAugment classification: randaugment, autoaugment, atau augmix |
| `erasing` | Probabilitas RandomErasing classification |
| `mixup` | Probabilitas batch-MixUp classification, dengan soft label |
| `cutmix` | Probabilitas batch-CutMix classification, dengan soft label |

Empat knob terakhir adalah paket classification. Family deteksi mengabaikannya.
`mixup` adalah knob khusus API: `--mixup` pada CLI merupakan alias untuk
`mixup_prob` milik deteksi.

<code-tabs name="usage" />

## Tiga status

| Status | Arti |
|---|---|
| `used` | Knob mencapai pipeline pelatihan family dan mengubah sampel |
| `gated_by_mosaic` | Knob hanya berlaku pada sampel yang melewati cabang mosaic, sehingga tidak pernah aktif ketika `mosaic_prob == 0` |
| `ignored` | Knob tidak pernah mencapai pipeline; menetapkannya tidak berpengaruh |

`ignored` adalah status yang perlu diperiksa sebelum run karena tidak ada proses
yang gagal. CLI memperingatkan jika parameter pelatihan yang ditetapkan secara
eksplisit diabaikan oleh family terpilih, dan trainer memperingatkan ketika
`mixup_prob > 0` tidak dapat aktif karena family membatasi MixUp pada mosaic
sementara `mosaic_prob` bernilai nol.

## Arketipe pipeline

Setiap family yang dicakup mengikuti salah satu dari enam pipeline, dengan
beberapa deviasi per family yang tercantum di bawah.

| Knob | Gaya YOLOX | YOLO-NAS | Gaya DETR | Classification | Semantic | Restore |
|---|---|---|---|---|---|---|
| `mosaic_prob` | used | ignored | ignored | ignored | ignored | ignored |
| `mixup_prob` | gated | used | ignored | ignored | ignored | ignored |
| `hsv_prob` | used | used | ignored | ignored | ignored | ignored |
| `flip_prob` | used | used | used | ignored | ignored | ignored |
| `degrees` | gated | used | ignored | ignored | ignored | ignored |
| `translate` | gated | used | ignored | ignored | ignored | ignored |
| `mosaic_scale` | gated | used | ignored | ignored | ignored | ignored |
| `mixup_scale` | gated | used | ignored | ignored | ignored | ignored |
| `shear` | gated | used | ignored | ignored | ignored | ignored |
| `perspective` | gated | used | ignored | ignored | ignored | ignored |
| `flipud` | used | used | ignored | ignored | ignored | ignored |
| `no_aug_epochs` | used | used | used | used | used | used |
| `auto_augment` | ignored | ignored | ignored | used | ignored | ignored |
| `erasing` | ignored | ignored | ignored | used | ignored | ignored |
| `mixup` | ignored | ignored | ignored | used | ignored | ignored |
| `cutmix` | ignored | ignored | ignored | used | ignored | ignored |

Dalam pipeline bergaya YOLOX, preprocessing per sampel menerapkan jitter HSV dan
flip, sedangkan affine warp dan MixUp hanya berjalan dalam cabang mosaic.
YOLO-NAS menjalankan affine per sampel yang selalu aktif, mengabaikan mosaic,
dan menerapkan MixUp secara independen dengan menggunakan kembali
`mosaic_scale` sebagai rentang skala affine.

Pipeline bergaya DETR adalah transform pass-through tanpa mosaic. Distorsi
fotometrik, zoom-out, dan IoU-crop merupakan konstanta resep, bukan knob yang
dapat dikonfigurasi. Karena itu, `hsv_prob` dan knob geometri tidak pernah
mencapainya. Pipeline classification menggunakan transform ImageFolder dengan
flip horizontal tetap sebesar 0.5, bukan `flip_prob`. Jitter skala semantic dan
HSV berasal dari atribut kelas family, bukan knob konfigurasi, sedangkan flip
restoration merupakan operasi input-dan-target berpasangan dengan probabilitas
tetap 0.5.

`no_aug_epochs` dipatuhi di semua tempat, meskipun yang dinonaktifkan berbeda:
mosaic dan MixUp untuk gaya YOLOX, affine dan MixUp untuk YOLO-NAS, augmentasi
fotometrik dan crop kuat beserta bagian akhir learning rate untuk gaya DETR,
serta bagian akhir scheduler untuk yang lain.

## Family menurut arketipe

| Arketipe | Family |
|---|---|
| Gaya YOLOX | `yolox`, `yolo7`, `yolo9`, `yolo9_e2e`, `yolo9_p2`, `rtmdet`, `picodet`, `rtdetr`, `rtdetrv2`, `fomo` |
| YOLO-NAS | `yolonas` |
| Gaya DETR | `dfine`, `domedetr`, `deim`, `deimv2`, `rtdetrv4`, `rfdetr`, `ec`, `dinov2` |
| Classification | `resnet`, `convnext`, `mobilenetv4`, `efficientnetv2` |
| Semantic | `segformer` |
| Restore | `nafnet` |

Dua puluh lima family dicakup. Family di luar list ini mengembalikan kumpulan
ignored kosong, sehingga tidak ada peringatan yang dikeluarkan.

## Deviasi

| Family | Perbedaan dari arketipe |
|---|---|
| `rtmdet` | `flipud` diabaikan: transform-nya tidak memiliki flip vertikal |
| `picodet` | `flipud` diabaikan |
| `rtdetr` | `flipud` diabaikan |
| `rtdetrv2` | `flipud` diabaikan |
| `fomo` | `perspective` dan `flipud` diabaikan |
| `ec` | `hsv_prob`, `degrees`, dan `translate` digunakan, hanya untuk `task="pose"`; detect dan segment menggunakan resep fotometrik tetap |
| `dinov2` | Paket classification digunakan, hanya untuk `task="classify"` |

`ec` dan `dinov2` adalah family multi-task, sehingga sebuah knob ditandai
ignored hanya ketika semua task family yang dapat dilatih mengabaikannya. Hal
ini mencegah peringatan CLI keliru untuk satu task meski benar bagi task lain.

Dome-DETR mewarisi transform D-FINE tanpa perubahan. Satu hal yang tidak dapat
diterimanya adalah pelatihan multi-scale, yang dinonaktifkan konfigurasinya,
bukan spec augmentasinya.

## Knob khusus family

Beberapa family memiliki knob augmentasi pada subclass `TrainConfig` sendiri,
bukan pada base. CLI tidak menyediakannya; tetapkan melalui API Python.

| Family | Knob | Arti |
|---|---|---|
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste` | Probabilitas augmentasi instance copy-paste, khusus `task="segment"` |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste_mode` | Sumber copy-paste: `flip` mencerminkan sampel yang sama, `mixup` menggunakan sampel kedua |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `rot90` | Probabilitas rotasi acak 90 derajat |
| `rfdetr` | `copy_paste` | Probabilitas copy-paste untuk `task="segment"`, hanya mode `flip` |
| `rfdetr` | `copy_paste_mode` | Mode sumber copy-paste untuk `task="segment"` |
| `rfdetr` | `crop_resize_prob` | Probabilitas crop-resize acak dalam pipeline native |
| `dfine` | `crop_resize_prob` | Probabilitas crop-resize acak, `task="segment"` |
| `ec` | `crop_resize_prob` | Probabilitas crop-resize acak, `task="segment"` |
| `ec`, `yolonas` | `brightness_contrast_prob` | Probabilitas jitter kecerahan dan kontras, `task="pose"` |
| `ec`, `yolonas` | `affine_prob` | Probabilitas affine yang memperhitungkan keypoint, `task="pose"` |

`rot90` berlaku untuk detect dan OBB pada `yolo9`.

## Melakukan query pada spec

| Helper | Mengembalikan |
|---|---|
| `aug_support(family)` | Tabel knob-ke-`Support`, atau `None` untuk family tidak dikenal |
| `ignored_aug_params(family)` | Kumpulan nama knob yang diabaikan family; kosong untuk family tidak dikenal |
| `uses_mosaic_gating(family)` | Apakah MixUp family hanya aktif pada sampel mosaic |
| `display_name(family)` | Nama family yang ditampilkan kepada pengguna dalam peringatan |
| `mixup_gating_warning(family, mosaic_prob, mixup_prob)` | Teks peringatan ketika MixUp tidak mungkin aktif, atau `None` |

`Support` adalah named tuple berisi `status` dan `note`, dengan catatan yang
menjelaskan alasan sebuah knob diabaikan atau dibatasi untuk family tersebut.

## Gate mosaic

Untuk family bergaya YOLOX, `mixup_prob=0.5` dengan `mosaic_prob=0`
menonaktifkan MixUp sepenuhnya karena MixUp hanya berlaku pada sampel mosaic.
Kombinasi ini mudah terjadi ketika mosaic dinonaktifkan menjelang akhir
pelatihan. Trainer mencatat peringatan yang menyebutkan family, dan
`mixup_gating_warning` adalah fungsi murni di baliknya.
