---
title: Matriks augmentasi
seo_title: Augmentasi yang dipatuhi setiap family LibreYOLO
description: >-
  Setiap knob augmentasi beserta rentang dan ejaan CLI, status setiap knob pada
  masing-masing dari 31 family yang dapat dilatih, dan nilai default bawaan
  setiap family.
lead: >-
  Menetapkan knob augmentasi tidak menjamin knob tersebut mencapai pipeline.
  Halaman ini mencatat cara setiap family yang dapat dilatih memperlakukan
  setiap knob pada TrainConfig, serta nilai awal setiap family, yang dihasilkan
  dari library pada rilis yang didokumentasikan halaman ini.
keywords:
  - augmentasi libreyolo
  - parameter augmentasi
  - default augmentasi
  - mosaic_prob
  - mixup_prob
  - hsv_prob
  - mosaic_scale
  - no_aug_epochs
  - matriks dukungan augmentasi
  - knob TrainConfig
last_verified: 1.8.0
verification: >-
  Dihasilkan oleh scripts/build-augmentation/collect.py dari LibreYOLO 1.8.0:
  daftar knob, status, dan catatan dari libreyolo/data/augment/spec.py, yang
  dikunci ke pipeline sebenarnya oleh test library, nilai default dari subclass
  TrainConfig setiap family, dan ejaan CLI dari perintah libreyolo train.
  Rentang dan satu-satunya titik tempat spec dan sebuah pipeline tidak sesuai
  diperiksa terhadap library oleh skrip yang sama.
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
    - label: Default suatu family
      language: python
      code: |
        from libreyolo.training.config import YOLO9Config

        config = YOLO9Config()
        print(config.mosaic_scale, config.degrees, config.mixup_prob)
source_hash: a88efd1537732e6c
---

## Knob

Berikut adalah nama field `TrainConfig`. `train()` dan CLI juga menerima ejaan
pendek di kolom CLI, sehingga `mosaic` menetapkan `mosaic_prob`. Di CLI, `mixup`
diarahkan berdasarkan task: pencampuran batch klasifikasi untuk pengklasifikasi,
`mixup_prob` untuk detektor. `perspective` tidak memiliki flag CLI dan
ditetapkan dari Python.

<aug-knobs />

Default dasar adalah nilai pada `TrainConfig` itu sendiri. Sebagian besar family
mengganti sebagian di antaranya, dan [Default per family](#defaults-by-family)
memberikan nilai yang benar-benar menjadi titik awal sebuah run. Untuk ilustrasi
fungsi setiap knob, lihat [Augmentasi](/docs/train/augmentations).

Rentang dalam tanda kurung adalah interval, sehingga `[0, 1)` mencakup 0 dan
tidak mencakup 1. Knob klasifikasi diperiksa saat pipeline klasifikasi dibangun
dan memunculkan `ValueError` di luar rentangnya. Knob deteksi tidak diperiksa
oleh `TrainConfig`, sehingga pelatihan dengan nilai di luar rentang tetap
berjalan tanpa peringatan. Pengecualiannya adalah skala hasil pengambilan acak
sebesar 0 atau kurang dan `perspective` bernilai negatif, yang memunculkan
`ValueError` saat sebuah sampel melewati warp.

<code-tabs name="usage" />

## Tiga status

| Status | Arti |
|---|---|
| `used` | Knob mencapai pipeline pelatihan family dan mengubah sampel |
| `gated_by_mosaic` | Knob hanya berlaku pada sampel yang mengambil cabang mosaic, sehingga dengan `mosaic_prob == 0` knob tersebut tidak pernah aktif |
| `ignored` | Knob tidak pernah mencapai pipeline; menetapkannya tidak berpengaruh apa pun |

`ignored` adalah status yang perlu diperiksa sebelum run, karena tidak ada yang
gagal. CLI memperingatkan jika parameter pelatihan yang ditetapkan secara
eksplisit termasuk yang diabaikan oleh family terpilih, dan trainer
memperingatkan jika `mixup_prob > 0` tidak dapat aktif karena family tersebut
membatasi MixUp pada mosaic dan `mosaic_prob` bernilai nol.

## Family mana yang mematuhi knob mana

<aug-matrix />

Setiap family yang dicakup mengikuti salah satu dari enam pipeline. Pada
pipeline bergaya YOLOX, prapemrosesan per sampel menerapkan jitter HSV dan
pembalikan, sedangkan affine warp dan MixUp hanya berjalan di dalam cabang
mosaic. Sebaliknya, YOLO-NAS menjalankan affine per sampel yang selalu aktif,
mengabaikan mosaic, dan menerapkan MixUp secara independen, dengan memakai ulang
`mosaic_scale` sebagai rentang skala affine.

Pipeline gaya DETR adalah transform pass-through tanpa mosaic. Distorsi
fotometrik, zoom-out, dan IoU-crop pada pipeline ini adalah konstanta resep,
bukan knob yang dapat dikonfigurasi, itulah sebabnya `hsv_prob` dan knob
geometri tidak pernah mencapainya. Klasifikasi memakai `flip_prob` untuk
pembalikan horizontal dan `flipud` untuk pembalikan vertikal. Pada segmentasi
semantik, jitter skala dan HSV berasal dari atribut kelas family, bukan dari
knob konfigurasi, dan pembalikan pada pemulihan gambar adalah operasi berpasangan
pada input dan target dengan probabilitas tetap 0.5.

`no_aug_epochs` dipatuhi di mana-mana, meskipun yang dimatikannya berbeda: mosaic
dan MixUp untuk gaya YOLOX, affine dan MixUp untuk YOLO-NAS, augmentasi
fotometrik kuat dan crop beserta bagian akhir learning rate untuk gaya DETR,
serta auto-augmentation, erasing, MixUp, dan CutMix untuk klasifikasi. Crop dan
pembalikan pada klasifikasi tetap aktif.

Jika sebuah family berbeda dari family lain yang memakai pipeline yang sama,
alasannya adalah salah satu dari berikut:

| Family | Perbedaan |
|---|---|
| `rtmdet`, `picodet`, `rtdetr`, `rtdetrv2` | `flipud` diabaikan: transform pada family ini tidak memiliki pembalikan vertikal |
| `fomo` | `perspective` dan `flipud` diabaikan |
| `ppyoloe` | `flipud` diabaikan: resep sumbernya memakai rotasi acak 90 derajat sebagai gantinya |
| `yolo9`, `yolo9_e2e`, `yolo9_p2`, `yolo7`, `rtdetr`, `rtdetrv2` | `mixup_scale` diabaikan: MixUp pada family ini memadukan frame utuh dengan rasio acak tanpa menskalakan ulang frame mana pun |
| `gtr` | `mosaic_prob` dan `mixup_prob` digunakan, selama `mosaic_epochs` epoch pertama; `degrees`, `translate`, dan `mosaic_scale` membentuk affine yang mengikuti mosaic |
| `rfdetr` | `hsv_prob` diabaikan: pipeline native tidak memiliki jitter HSV |
| `ec` | `hsv_prob`, `degrees`, dan `translate` digunakan, hanya untuk `task="pose"`; detect dan segment memakai resep fotometrik tetap |
| `dinov2` | Paket klasifikasi digunakan, hanya untuk `task="classify"` |

`ec` dan `dinov2` adalah family multi-task, sehingga sebuah knob ditandai
diabaikan hanya jika setiap task yang dapat dilatih pada family tersebut
mengabaikannya. Dengan begitu, peringatan CLI tidak pernah salah untuk satu task
sementara benar untuk task lain.

Dome-DETR mewarisi transform D-FINE tanpa perubahan. Satu-satunya hal yang tidak
dapat diterimanya adalah pelatihan multi-scale, yang dinonaktifkan oleh
konfigurasinya, bukan oleh spec augmentasi.

Tiga puluh satu family dicakup. Untuk family di luar daftar ini,
`ignored_aug_params` mengembalikan himpunan kosong, sehingga tidak ada peringatan
yang dikeluarkan untuknya.

## Default per family

Nilai awal pada konfigurasi setiap family deteksi. Sel kosong berarti family
tersebut mengabaikan knob itu, sehingga nilai default knob tersebut tidak
mengubah apa pun.

<aug-defaults />

Dua pola lebih penting daripada nilai tunggal mana pun. RT-DETR, RT-DETRv2,
PicoDet, dan FOMO dimulai dengan `mosaic_prob=0`, sehingga pada family tersebut
knob yang ditandai dibatasi oleh mosaic di matriks tidak berpengaruh apa pun
sampai mosaic diaktifkan. YOLOv9 dimulai dengan rotasi, shear, dan MixUp dalam
keadaan mati serta dengan `mosaic_scale=(0.5, 1.5)`, sedangkan default dasar
adalah `(0.1, 2.0)`.

EdgeCrafter membaca `hsv_prob`, `degrees`, dan `translate` hanya untuk
`task="pose"`, sehingga barisnya menampilkan default pose untuk ketiga knob
tersebut.

Family klasifikasi dimulai dari default dasar: `flip_prob=0.5`, `flipud=0.0`,
`scale=(0.5, 1.0)`, `crop_pct` milik family itu sendiri, serta `auto_augment`,
`erasing`, `mixup`, dan `cutmix` dalam keadaan nonaktif.

## Knob khusus family

Sebagian family membawa knob augmentasi pada subclass `TrainConfig` miliknya
sendiri, bukan pada class dasar. CLI tidak menyediakan knob ini; tetapkan
melalui API Python.

| Family | Knob | Arti |
|---|---|---|
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste` | Probabilitas augmentasi instance copy-paste, hanya `task="segment"` |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `copy_paste_mode` | Sumber copy-paste: `flip` mencerminkan sampel yang sama, `mixup` memakai sampel kedua |
| `yolo9`, `yolo9_e2e`, `yolo9_p2` | `rot90` | Probabilitas rotasi acak 90 derajat |
| `rfdetr` | `copy_paste` | Probabilitas copy-paste untuk `task="segment"`, hanya mode `flip` |
| `rfdetr` | `copy_paste_mode` | Mode sumber copy-paste untuk `task="segment"` |
| `rfdetr` | `crop_resize_prob` | Probabilitas random crop-resize dalam pipeline native |
| `dfine` | `crop_resize_prob` | Probabilitas random crop-resize, `task="segment"` |
| `ec` | `crop_resize_prob` | Probabilitas random crop-resize, `task="segment"` |
| `ec`, `yolonas` | `brightness_contrast_prob` | Probabilitas jitter kecerahan dan kontras, `task="pose"` |
| `ec`, `yolonas` | `affine_prob` | Probabilitas affine yang memperhitungkan keypoint, `task="pose"` |

`rot90` berlaku untuk detect dan OBB pada `yolo9`.

## Melakukan query pada spec

| Helper | Mengembalikan |
|---|---|
| `aug_support(family)` | Tabel knob-ke-`Support`, atau `None` untuk family yang tidak dikenal |
| `ignored_aug_params(family)` | Himpunan nama knob yang diabaikan family; kosong untuk family yang tidak dikenal |
| `uses_mosaic_gating(family)` | Apakah MixUp pada family hanya aktif pada sampel mosaic |
| `display_name(family)` | Nama family yang ditampilkan kepada pengguna, dipakai dalam peringatan |
| `mixup_gating_warning(family, mosaic_prob, mixup_prob)` | Teks peringatan ketika MixUp tidak pernah dapat aktif, selain itu `None` |

`Support` adalah named tuple berisi `status` dan `note`, dengan note yang
menjelaskan alasan sebuah knob diabaikan atau dibatasi oleh mosaic untuk family
tersebut. Di 1.8.0, spec masih mencantumkan `mixup_scale` sebagai dibatasi oleh
mosaic untuk family yang dilatih melalui wrapper mosaic YOLOv9, sehingga
`aug_support` melaporkannya sebagai dibatasi di sana, meskipun matriks di atas,
yang diperiksa terhadap pipeline, menandainya sebagai diabaikan.

## Pembatasan oleh mosaic

Untuk family bergaya YOLOX, `mixup_prob=0.5` dengan `mosaic_prob=0` menonaktifkan
MixUp sepenuhnya, karena MixUp hanya berlaku pada sampel mosaic. Kombinasi ini
mudah terjadi saat mosaic dimatikan menjelang akhir pelatihan. Trainer mencatat
peringatan yang menyebutkan nama family, dan `mixup_gating_warning` adalah fungsi
murni di baliknya.
