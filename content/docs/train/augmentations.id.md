---
title: Augmentasi
seo_title: Augmentasi pelatihan di LibreYOLO
description: >-
  Pengaturan augmentasi pada TrainConfig, empat bentuk pipeline di baliknya, dan
  tabel per family yang menunjukkan pengaturan mana yang digunakan, dibatasi,
  atau diabaikan.
lead: >-
  Augmentasi dikonfigurasi melalui pengaturan pada TrainConfig, tetapi setiap
  family model menjalankan pipeline pelatihannya sendiri, dan pipeline tanpa
  cabang mosaic mengabaikan mosaic_prob, bukan mengaproksimasinya.
keywords:
  - augmentasi data YOLO
  - augmentasi mosaic
  - mixup
  - HSV jitter
  - random affine
  - augmentasi copy paste
  - randaugment
  - cutmix
  - no_aug_epochs
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            mosaic_prob=1.0,
            mixup_prob=0.15,
            hsv_prob=1.0,
            flip_prob=0.5,
            no_aug_epochs=15,
        )
    - label: CLI
      language: bash
      code: |
        # CLI menulis mosaic_prob sebagai mosaic dan mixup_prob sebagai mixup.
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 mosaic=1.0 mixup=0.15 hsv_prob=1.0 \
          flip_prob=0.5 no_aug_epochs=15
  support:
    - label: Baca tabel dukungan suatu family
      language: python
      code: |
        from libreyolo.data.augment.spec import AUG_KNOBS, aug_support

        for knob, description in AUG_KNOBS.items():
            support = aug_support("yolo9")[knob]
            print(f"{knob:16} {support.status:16} {support.note or description}")
    - label: Hanya yang diabaikan
      language: python
      code: |
        from libreyolo.data.augment.spec import ignored_aug_params

        print(sorted(ignored_aug_params("rfdetr")))
  classify:
    - label: Paket classification
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreConvNeXtt-cls.pt")
        model.train(
            data="my-classification-dataset",
            epochs=50,
            auto_augment="randaugment",
            erasing=0.25,
            mixup=0.2,
            cutmix=0.2,
        )
source_hash: 47461cd13aab580c
---

## Mengatur pengaturan

Pengaturan augmentasi adalah argumen `train()` biasa.

<code-tabs name="train" />

Dua pengaturan memiliki ejaan CLI yang lebih pendek: `mosaic` dipetakan ke
`mosaic_prob` dan `mixup` ke `mixup_prob`. Pengaturan lain dieja sama di keduanya.

## Tiga status, bukan dua

Pengaruh suatu pengaturan bergantung pada family. Library menyimpan tabel
deklaratif dengan tiga kemungkinan status.

`used` berarti pengaturan mencapai pipeline dan mengubah sampel. `ignored`
berarti pengaturan tidak pernah mencapai pipeline. `gated_by_mosaic` berarti
hanya berlaku pada sampel yang masuk cabang mosaic, sehingga tidak pernah aktif
jika `mosaic_prob=0`.

Status ketiga sering mengejutkan. Pada pipeline bergaya YOLOX, affine warp
berjalan pada kanvas mosaic dan MixUp mencampur sampel mosaic, sehingga
`mosaic_prob=0` sekaligus menonaktifkan `degrees`, `translate`, `shear`,
`perspective`, `mosaic_scale`, `mixup_prob`, dan `mixup_scale`. Trainer mencatat
warning khusus untuk MixUp:

```text
mixup_prob=0.15 has no effect for YOLOv9: mixup only applies to mosaic samples
and mosaic_prob=0. Set mosaic_prob > 0 to enable mixup.
```

CLI juga memperingatkan tentang pengaturan yang diabaikan, hanya yang diketik:

```text
Warning: RF-DETR ignores these parameters: degrees, mosaic
```

## Empat bentuk pipeline

Family terbagi dalam empat pipeline pelatihan yang menentukan hampir semua jawaban.

Pipeline mosaic bergaya YOLOX menerapkan HSV jitter dan flip per sampel, lalu
menjalankan affine serta MixUp dalam cabang mosaic. Pipeline ini mencakup YOLOX,
YOLOv7, YOLOv9 beserta varian E2E dan P2, RTMDet, PicoDet, RT-DETR, RT-DETRv2,
dan FOMO.

Pipeline pass-through bergaya DETR tidak memiliki mosaic atau affine warp.
Photometric distortion, zoom-out, dan IoU crop adalah konstanta resep, sehingga
hanya `flip_prob` serta `no_aug_epochs` yang aktif. Pipeline ini mencakup D-FINE,
Dome-DETR, DEIM, DEIMv2, RT-DETRv4, EC, dan RF-DETR dengan satu perubahan.

Pipeline ImageFolder classification mengabaikan semua pengaturan deteksi.
Horizontal flip-nya ditetapkan ke 0,5 dan tidak dipengaruhi `flip_prob`. Pipeline
ini memiliki paket pengaturan sendiri yang dijelaskan di bawah.

YOLO-NAS memiliki bentuk tersendiri: tanpa mosaic, affine per sampel yang selalu
aktif, dan MixUp yang diterapkan secara independen. Nilai `mosaic_scale` digunakan
kembali sebagai rentang skala affine.

SegFormer dan NAFNet menjalankan pipeline khusus task dengan randomisasi tetap.
Untuk SegFormer, pengaturan aktif adalah atribut class `semantic_scale_jitter`
dan `semantic_hsv_prob`, bukan `mosaic_scale` serta `hsv_prob`. Crop dan flip
NAFNet merupakan operasi input-target berpasangan dengan probabilitas tetap 0,5.

## Pengaturan yang dipatuhi setiap family

Tabel berikut adalah spesifikasi yang disertakan di
`libreyolo/data/augment/spec.py` dan diuji terhadap wiring pipeline sebenarnya.
Baca sumber itu, jangan menyimpulkan dari arsitektur.

<code-tabs name="support" />

Ringkasan per pipeline untuk pengaturan dasar:

| Pengaturan | Gaya YOLOX | YOLO-NAS | Gaya DETR | Classification |
|---|---|---|---|---|
| `mosaic_prob` | digunakan | diabaikan | diabaikan | diabaikan |
| `mixup_prob` | dibatasi mosaic | digunakan | diabaikan | diabaikan |
| `hsv_prob` | digunakan | digunakan | diabaikan | diabaikan |
| `flip_prob` | digunakan | digunakan | digunakan | diabaikan |
| `flipud` | digunakan | digunakan | diabaikan | diabaikan |
| `degrees` | dibatasi mosaic | digunakan | diabaikan | diabaikan |
| `translate` | dibatasi mosaic | digunakan | diabaikan | diabaikan |
| `shear` | dibatasi mosaic | digunakan | diabaikan | diabaikan |
| `perspective` | dibatasi mosaic | digunakan | diabaikan | diabaikan |
| `mosaic_scale` | dibatasi mosaic | digunakan | diabaikan | diabaikan |
| `mixup_scale` | dibatasi mosaic | digunakan | diabaikan | diabaikan |
| `no_aug_epochs` | digunakan | digunakan | digunakan | digunakan |

Pengecualian dalam kolom tersebut hanya mempersempit dukungan:

- RTMDet, PicoDet, RT-DETR, RT-DETRv2, dan FOMO tidak memiliki vertical flip,
  sehingga `flipud` diabaikan. Wrapper mosaic FOMO juga tidak memiliki perspective.
- Pipeline native RF-DETR tidak memiliki HSV jitter, sehingga `hsv_prob` diabaikan.
- EC mematuhi `hsv_prob`, `degrees`, dan `translate` hanya untuk `task="pose"`,
  yang transform-nya memahami keypoint. Jalur detect dan segment memakai resep tetap.
- DINOv2 mengikuti kolom gaya DETR untuk task detect dan semantic, lalu menambahkan
  paket classification untuk `task="classify"`.

`no_aug_epochs` digunakan di semua tempat, tetapi artinya berbeda. Pada pipeline
mosaic, pengaturan ini menonaktifkan mosaic dan MixUp untuk epoch terakhir. Pada
pipeline bergaya DETR, pengaturan ini menghentikan augmentasi photometric,
zoom-out, dan crop serta membentuk ujung schedule. Pada pipeline classification
dan semantic, pengaturan ini hanya membentuk ujung schedule.

## Paket classification

Empat pengaturan mengendalikan pipeline classification dan tidak memengaruhi
yang lain. Family deteksi mengabaikan keempatnya.

<code-tabs name="classify" />

`auto_augment` menerima `"randaugment"`, `"autoaugment"`, `"augmix"`, atau
`None`. `erasing` adalah probabilitas RandomErasing. `mixup` dan `cutmix` adalah
probabilitas per batch yang menghasilkan soft label; paling banyak satu berjalan
per batch, dengan MixUp lebih dahulu, sehingga jumlah keduanya maksimal 1.

Keempatnya default nonaktif, sehingga pelatihan classification tidak berubah
kecuali diminta.

Ada benturan nama: pada CLI, `mixup` adalah alias untuk `mixup_prob` deteksi.
Field `mixup` classification tidak memiliki ejaan CLI dan hanya dapat dicapai
melalui `model.train(mixup=...)` di Python.

## Pengaturan khusus family

Sebagian pengaturan berada pada subclass konfigurasi family, sehingga hanya ada
untuk family tersebut dan tidak memiliki flag CLI.

| Family | Pengaturan | Efek |
|---|---|---|
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste` | Probabilitas augmentasi instance copy-paste, hanya `task="segment"` |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste_mode` | `"flip"` memakai ulang sampel yang dicerminkan, `"mixup"` mengambil sampel kedua |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `rot90` | Probabilitas rotasi acak 90 derajat |
| YOLOv9 | `max_labels` | Batas ground truth per gambar dalam transform pelatihan, default 100 |
| RF-DETR | `copy_paste`, `copy_paste_mode` | Copy-paste untuk `task="segment"`, hanya mode `"flip"` |
| RF-DETR, D-FINE, EC | `crop_resize_prob` | Probabilitas random crop-resize |
| EC, YOLO-NAS | `brightness_contrast_prob`, `affine_prob` | Jitter jalur pose dan probabilitas affine yang memahami keypoint |

`max_labels` dapat diam-diam menghilangkan data. Box setelah batas dibuang tanpa
error, sehingga imagery padat seperti foto aerial perlu menaikkan nilainya.

Mosaic dan MixUp dinonaktifkan untuk pelatihan oriented-box tanpa memandang
pengaturan karena augmentasi yang memahami sudut box berotasi belum tersedia.

## Terkait

- [Hyperparameter](/docs/train/hyperparameters) untuk `no_aug_epochs` sebagai
  argumen schedule dan bagian lain `train()`.
- [Dataset](/docs/train/datasets) untuk format label yang digunakan transform ini.
