---
title: Augmentasi
seo_title: 'Augmentasi pelatihan di LibreYOLO, beserta contohnya'
description: >-
  Fungsi setiap knob augmentasi LibreYOLO, diperlihatkan pada gambar nyata oleh
  kode pelatihan library itu sendiri, beserta rentang, nilai default, dan family
  yang membacanya.
lead: >-
  Augmentasi pelatihan adalah transformasi acak yang diterapkan pada setiap
  sampel pelatihan. Augmentasi dikonfigurasi melalui knob pada TrainConfig, dan
  setiap family model menjalankan pipeline sendiri, sehingga sebuah knob hanya
  mengubah sampel jika pipeline tersebut membacanya.
keywords:
  - augmentasi data yolo
  - augmentasi mosaic
  - mixup
  - augmentasi hsv
  - random affine
  - random erasing
  - deteksi objek kecil
  - contoh augmentasi
  - augmentasi copy paste
  - randaugment
  - cutmix
  - no_aug_epochs
last_verified: 1.8.0
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
  small:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Nilai awal YOLOX adalah mosaic_scale=(0.1, 2.0), yang dapat
        memperkecil

        # objek hingga sepersepuluh ukurannya. (0.5, 1.5) berhenti di
        setengahnya.

        model = LibreYOLO("LibreYOLOXs.pt")

        model.train(
            data="my-dataset.yaml",
            epochs=100,
            mosaic_scale=(0.5, 1.5),
        )
    - label: CLI
      language: bash
      code: >
        # Di CLI, sebuah pasangan nilai ditulis sebagai dua angka yang
        dipisahkan koma.

        libreyolo train model=LibreYOLOXs.pt data=my-dataset.yaml \
          epochs=100 mosaic_scale=0.5,1.5
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
    - label: Paket klasifikasi
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
source_hash: 72d22a38d647de0a
---

## Mengatur knob

Knob augmentasi adalah argumen `train()` biasa.

<code-tabs name="train" />

Tiga di antaranya memiliki ejaan yang lebih pendek, yang diterima oleh `train()`
maupun CLI: `mosaic` dipetakan ke `mosaic_prob`, `fliplr` ke `flip_prob`, dan
pada deteksi `mixup` dipetakan ke `mixup_prob`. Memberikan ejaan pendek beserta
field yang dipetakannya dengan nilai berbeda akan memunculkan error.
`perspective` tidak memiliki flag CLI di 1.8.0, sehingga nilainya ditetapkan
dari Python.

Knob klasifikasi divalidasi saat pipeline klasifikasi dibangun, dan CLI juga
memeriksa `flip_prob`, `flipud`, `cutmix`, dan `erasing`. `TrainConfig` tidak
memeriksa rentang knob deteksi, sehingga `degrees=100` melatih dengan rotasi
hingga 100 derajat. [Matriks
augmentasi](/docs/reference/augmentation-matrix) mencantumkan rentang setiap
knob.

## Tiga status, bukan dua

Apakah sebuah knob berpengaruh bergantung pada family. Library menyimpan tabel
deklaratif untuk hal itu, dan setiap entrinya memiliki salah satu dari tiga
status.

`used` berarti knob mencapai pipeline dan mengubah sampel. `ignored` berarti
knob tidak pernah mencapai pipeline, sehingga menetapkannya tidak berpengaruh
apa pun. `gated_by_mosaic` berarti knob hanya berlaku pada sampel yang mengambil
cabang mosaic, sehingga dengan `mosaic_prob=0` knob tersebut tidak pernah aktif
meskipun sudah tersambung ke pipeline.

Status ketiga inilah yang sering mengejutkan. Pada pipeline bergaya YOLOX,
affine warp berjalan pada kanvas mosaic dan MixUp mencampur sampel mosaic,
sehingga `mosaic_prob=0` diam-diam menonaktifkan `degrees`, `translate`,
`shear`, `perspective`, `mosaic_scale`, `mixup_prob`, dan `mixup_scale`
sekaligus. Trainer mencatat peringatan khusus untuk kasus MixUp:

```text
mixup_prob=0.15 has no effect for YOLOv9: mixup only applies to mosaic samples
and mosaic_prob=0. Set mosaic_prob > 0 to enable mixup.
```

CLI juga memperingatkan tentang knob yang diabaikan, dan hanya mencantumkan knob
yang benar-benar diketik:

```text
Warning: RF-DETR ignores these parameters: degrees, mosaic
```

Jika dipanggil dari Python, `train()` tidak memperingatkan tentang knob yang diabaikan.
[Matriks augmentasi](/docs/reference/augmentation-matrix) memuat status setiap
knob untuk setiap family dan nilai awal yang dipakai setiap family. Tabel
tersebut adalah spec yang disertakan di `libreyolo/data/augment/spec.py`, yang
diperiksa terhadap pipeline oleh test milik library itu sendiri, dan dapat
dibaca langsung:

<code-tabs name="support" />

## Apa yang dilakukan setiap knob

Ilustrasi di bawah ini dihasilkan oleh transform pelatihan milik LibreYOLO 1.8.0
sendiri, yang dijalankan pada gambar contoh melalui dataset berisi satu gambar,
sehingga setiap tile mosaic dan setiap pasangan MixUp adalah foto yang sama.
Kotak yang ditampilkan adalah orang yang dideteksi LibreYOLO9t pada gambar
tersebut. Di bawah setiap panel tercantum pengaturannya dan ukuran kotak
terbesar, dalam piksel pada resolusi pelatihan 640 piksel. Tanpa augmentasi,
orang tersebut berukuran 291 × 217 px.

Beberapa knob geometri berupa rentang: `degrees=10` berarti sudut yang diambil
secara acak di mana saja dari -10 hingga +10 derajat. Panel knob tersebut
mematok hasil pengambilan acak pada nilai yang tertera di keterangan, sehingga
panel menunjukkan efek terkuat yang dimungkinkan oleh suatu pengaturan, bukan
sampel yang tipikal. Ilustrasi deteksi berasal dari pipeline YOLOv9. YOLOv7
dilatih melalui pipeline tersebut tanpa perubahan, sedangkan RT-DETR dan
RT-DETRv2 memakai wrapper mosaic pipeline itu dengan transform per sampel milik
mereka sendiri. YOLOX, RTMDet, PicoDet, dan FOMO membangun mosaic dan warp
dengan cara yang sama di wrapper terpisah, yang MixUp-nya berbeda.

### mosaic_prob

Probabilitas bahwa sebuah sampel adalah mosaic: empat gambar yang ditempatkan
pada kanvas berukuran dua kali ukuran input, yang kemudian dipetakan oleh affine
warp ke ukuran input. Pada pipeline bergaya YOLOX, warp dan MixUp hanya berjalan
pada sampel mosaic, sehingga keduanya berhenti jika nilai ini 0. RT-DETR,
RT-DETRv2, PicoDet, dan FOMO memakai default 0 untuk knob ini.

<aug-figure name="mosaic" />

Panel kedua mematok skala warp pada 0.5×, yaitu skala saat seluruh kanvas muat.
Dua panel terakhir adalah pengambilan acak biasa dengan default YOLOv9, dan
keterangannya menyebutkan skala yang diperoleh masing-masing.

### mosaic_scale

Rentang asal pengambilan acak nilai zoom pada warp. Pada 1×, objek mempertahankan
ukurannya saat inferensi, dan pada 0.5× ukurannya menjadi setengahnya. Default
dasarnya adalah `(0.1, 2.0)`, yang dipertahankan oleh YOLOX, YOLOv7, RTMDet,
PicoDet, RT-DETR, RT-DETRv2, dan FOMO, sehingga objek pada family tersebut dapat
menyusut hingga sepersepuluh ukurannya. Default YOLOv9 adalah `(0.5, 1.5)`.
YOLO-NAS dan PP-YOLOE tidak memiliki mosaic dan memakai ulang rentang ini
sebagai skala affine warp per sampel milik mereka.

<aug-figure name="mosaic_scale" />

Di atas 1×, output hanya menampilkan bagian kiri atas kanvas, sehingga objek
terpotong di tepi atau keluar dari frame.

### degrees

Rentang rotasi: setiap sampel yang melewati warp berputar sebesar sudut yang
diambil secara acak dari `-degrees` hingga `+degrees`. Rotasi berporos pada
sudut kiri atas kanvas, sehingga sudut rotasi yang besar mengayunkan sebagian
besar gambar keluar dari frame. YOLOX dan YOLOv7 mempertahankan default dasar
10, sedangkan default YOLOv9 adalah 0.

<aug-figure name="degrees" />

### translate

Rentang pergeseran, sebagai fraksi sisi input: setiap sampel yang melewati warp
bergeser hingga `translate` kali 640 px pada setiap sumbu saat `imgsz=640`.
Default dasarnya adalah 0.1. Default YOLO-NAS dan PP-YOLOE adalah 0.25, sedangkan
default RTMDet, PicoDet, dan FOMO adalah 0.

<aug-figure name="translate" />

### shear

Rentang shear dalam derajat, yang diambil secara terpisah untuk setiap sumbu.
YOLOX dan YOLOv7 mempertahankan default dasar 2, dan setiap family lain yang
membacanya memakai default 0.

<aug-figure name="shear" />

### perspective

Projective warp di atas rotasi, skala, shear, dan translasi. Setiap sudut kanvas
bergeser ke dalam sejauh nilai acak hingga `min(perspective * 100, 0.2)` kali
setengah sisi kanvas, per sumbu, sehingga `perspective=0.001` menggeser sebuah
sudut paling jauh 5 persen dari sisi pada resolusi apa pun, dan setiap nilai
mulai dari `0.002` ke atas mencapai maksimum yang sama. Default `0` membuat warp
tetap affine, dan nilai negatif memunculkan `ValueError` saat sampel pertama
melewati warp. Semua panel memakai satu hasil pengambilan acak yang sama, yang
diskalakan oleh nilai tersebut. YOLOv9 serta varian E2E dan P2 miliknya, YOLOX,
YOLOv7, YOLO-NAS, PP-YOLOE, RTMDet, PicoDet, RT-DETR, dan RT-DETRv2
menerapkannya.

<aug-figure name="perspective" />

### hsv_prob

Probabilitas jitter warna HSV. Hanya probabilitasnya yang merupakan knob. Jitter
menggeser hue hingga 5 pada skala hue OpenCV 0 sampai 179, serta saturasi dan
value hingga 30 pada skala 0 sampai 255, dan masing-masing dari ketiga
pergeseran itu dilewati dengan probabilitas setengah. Setiap pipeline deteksi
yang membaca `hsv_prob` memakai besaran tersebut. Tiga panel terakhir adalah hasil
yang paling terlihat dari 119 pengambilan acak.

<aug-figure name="hsv" />

### flip_prob dan flipud

Probabilitas pembalikan horizontal dan vertikal. Nilai awal `flip_prob` adalah
0.5 di setiap family kecuali FOMO, yang memakai default 0. Nilai awal `flipud`
adalah 0. RTMDet, PicoDet, RT-DETR, RT-DETRv2, FOMO, PP-YOLOE, dan family
bergaya DETR tidak memiliki pembalikan vertikal dan mengabaikan `flipud`. Resep
PP-YOLOE memakai rotasi acak 90 derajat sebagai gantinya.

<aug-figure name="flip" />

### mixup_prob dan mixup_scale

MixUp memadukan sampel kedua ke dalam sampel saat ini dan mempertahankan kotak
dari keduanya. Pada pipeline YOLOv9, pasangannya adalah satu sampel utuh tanpa
mosaic, rasio pencampuran diambil secara seragam antara 0 dan 1, dan
`mixup_scale` tidak pernah dibaca, meskipun spec library mencantumkannya sebagai
dibatasi oleh mosaic. Pada wrapper YOLOX, yang dipakai oleh YOLOX, RTMDet,
PicoDet, dan FOMO, serta pada wrapper YOLO-NAS, pasangannya lebih dulu
diskalakan ulang dengan faktor yang diambil dari `mixup_scale`, lalu kedua
gambar dirata-ratakan dengan rasio tetap 0.5. Default dasar `mixup_prob` adalah
1.0, dan default YOLOv9 adalah 0.

<aug-figure name="mixup" />

Dua panel terakhir memadukan sampel kedua ke dalam mosaic berskala 0.5×.
Keterangan kedua panel itu menyebutkan rasio pencampuran λ, yaitu bobot sampel
mosaic.

### no_aug_epochs

`no_aug_epochs` berstatus `used` di mana-mana, tetapi maknanya tidak sama di
mana-mana. Pada pipeline mosaic, knob ini mematikan mosaic dan MixUp untuk epoch
terakhir. Pada pipeline bergaya DETR, knob ini menghentikan augmentasi
fotometrik, zoom-out, dan crop serta membentuk bagian akhir jadwal. Pada
pipeline klasifikasi, knob ini menonaktifkan auto-augmentation, erasing, MixUp,
dan CutMix, sementara crop dan pembalikan tetap berjalan.

## Resep gaya DETR

D-FINE, Dome-DETR, DEIM, DEIMv2, RT-DETRv4, dan TinyFormer dilatih melalui satu
transform torchvision tanpa mosaic dan tanpa affine warp, begitu pula EdgeCrafter
(EC) untuk deteksi. Tiga operasi kuat transform tersebut adalah konstanta resep,
bukan knob:

- `RandomPhotometricDistort` menerapkan setiap perubahan warna (kecerahan 0.875
  hingga 1.125, kontras dan saturasi 0.5 hingga 1.5, hue ±0.05, dan channel
  shuffle) dengan probabilitas 0.5.
- `RandomZoomOut`, dengan probabilitas 0.5, menempatkan gambar pada kanvas hitam
  yang sisinya hingga empat kali sisi gambar, sehingga objek dapat menyusut
  hingga seperempat ukurannya.
- `RandomIoUCrop`, yang diterapkan dengan probabilitas 0.8, melakukan crop ke
  area antara 0.3 dan 1.0 dari setiap sisi, yang dipilih agar setidaknya
  sebagian kotak tetap dipertahankan.

Setelah langkah-langkah ini, kotak yang berukuran 1 px atau kurang pada salah
satu sisinya dibuang. Ketiga operasi berhenti pada epoch penghentian family
tersebut, paling lambat saat `no_aug_epochs` epoch terakhir dimulai. Knob yang
dibaca pipeline ini hanya `flip_prob` dan `no_aug_epochs`.

<aug-figure name="detr_ops" />

Tiga sampel dari awal pelatihan D-FINE, dengan ketiga operasi aktif:

<aug-figure name="detr_samples" />

RF-DETR memiliki pipeline sendiri: pembalikan horizontal, lalu dengan
probabilitas `crop_resize_prob` pengubahan ukuran sisi yang lebih pendek menjadi
400, 500, atau 600 px dan crop persegi acak berukuran 384 hingga 600 px, dan
terakhir pengubahan ukuran langsung ke resolusi pelatihan persegi, yang
meregangkan gambar, bukan menerapkan letterbox. Default `crop_resize_prob`
adalah 0.5 dan nilainya ditetapkan dari Python.

<aug-figure name="rfdetr_crop" />

## Objek kecil

`erasing` tidak berlaku untuk deteksi. Random Erasing termasuk dalam paket
klasifikasi dan tidak ada pipeline deteksi yang memiliki langkah erasing,
sehingga detektor dilatih dengan cara yang sama berapa pun nilainya, dan tidak
ada yang perlu dimatikan. Berikut adalah langkah-langkah yang memperkecil atau
menghilangkan objek kecil:

- `mosaic_scale`. Pada pipeline mosaic, ukuran objek dikalikan dengan skala yang
  diambil secara acak, sehingga batas bawah menentukan ukuran terkecilnya.
  Dengan default dasar `(0.1, 2.0)`, objek 20 px dapat menjadi 2 px. Pipeline
  YOLOv9 kemudian membuang kotak berukuran 2 px atau kurang pada salah satu
  sisinya, dan transform terakhirnya, seperti milik YOLOX, membuang kotak
  berukuran 1 px atau kurang pada salah satu sisinya. Piksel objek tetap berada di gambar tanpa label.
  Batas bawah 0.5, seperti pada `(0.5, 1.5)` milik YOLOv9, menjaga setiap objek
  tetap berukuran setengah dari ukurannya saat inferensi atau lebih.
- `degrees`, `translate`, `shear`, dan `perspective` mendorong objek ke tepi,
  tempat kotaknya terpotong dan melewati filter ukuran yang sama.
- Pada resep gaya DETR, `RandomZoomOut` dapat memperkecil objek hingga
  seperempat ukurannya dan `RandomIoUCrop` dapat memotongnya keluar dari gambar.
  Keduanya bukan knob, dan keduanya berhenti pada epoch penghentian.
- Crop RF-DETR dapat memotong objek di tepi area crop. `crop_resize_prob=0`
  mematikannya.
- Resolusi pelatihan menskalakan setiap objek: letterbox mengubah ukuran gambar
  dengan faktor `imgsz` dibagi sisi gambar yang lebih panjang, sehingga pada
  `imgsz=1280` setiap objek memiliki lebar dan tinggi dua kali lipat
  dibandingkan pada 640. [YOLOv9](/docs/models/yolov9) juga memiliki varian P2
  dengan head stride-4 untuk objek kecil.

<code-tabs name="small" />

## Paket klasifikasi

Enam knob hanya bekerja pada pipeline klasifikasi. Family deteksi mengabaikan
keenamnya.

<code-tabs name="classify" />

`scale` adalah rentang luas area crop acak. `scale=0.5` berarti `(0.5, 1.0)`,
dan pasangan nilai eksplisit menetapkan kedua batasnya. Batas bawah yang kecil
dapat memotong objek keluar sepenuhnya, seperti pada panel ketiga.

<aug-figure name="cls_crop" />

`auto_augment` menerima `"randaugment"`, `"autoaugment"`, `"augmix"`, atau
`None`, dan menerapkan satu pengambilan acak dari kebijakan tersebut pada setiap
sampel.

<aug-figure name="auto_augment" />

`erasing` adalah probabilitas Random Erasing, dalam `[0, 1)`. Langkah ini
berjalan setelah normalisasi dan mengisi sebuah persegi panjang yang mencakup 2
hingga 33 persen gambar dengan nol, yang setelah normalisasi adalah warna
rata-rata ImageNet.

<aug-figure name="erasing" />

`mixup` dan `cutmix` adalah probabilitas per batch yang menghasilkan soft label.
Paling banyak satu yang berjalan per batch, dengan MixUp lebih dahulu, sehingga
keduanya bersifat aditif dan jumlahnya harus paling banyak 1. Pada panel, A dan
B adalah dua crop dari foto yang sama, dan setiap keterangan menyebutkan soft
label yang menjadi target pelatihan gambar campuran tersebut.

<aug-figure name="batch_mix" />

`auto_augment`, `erasing`, `mixup`, dan `cutmix` nonaktif secara default,
sehingga pelatihan klasifikasi tidak berubah kecuali diminta.
`train()` dan CLI mengarahkan `mixup` berdasarkan task: klasifikasi memakai
pencampuran batch, sedangkan deteksi memakai `mixup_prob`.

`crop_pct=None` mempertahankan rasio resize evaluasi milik family. Override
mengubah crop evaluasi pada tahap validasi yang berjalan selama pelatihan,
sedangkan ekspor tetap memakai prapemrosesan native family.

CLI menyediakan `auto_augment`, `erasing`, `cutmix`, `scale`, `crop_pct`,
`fliplr`, dan `flipud`. `mixup` pada klasifikasi berarti pencampuran batch,
dengan default 0.0. Pembalikan horizontal memiliki default 0.5 dan pembalikan
vertikal 0.0.

`no_aug_epochs` menonaktifkan auto-augmentation, erasing, MixUp, dan CutMix
untuk epoch terakhir, dengan tetap mempertahankan crop dan pembalikan. Secara
default, resep klasifikasi menetapkan bagian akhir ini ke 0.

## Knob khusus family

Sebagian knob berada pada subclass konfigurasi family, bukan pada class dasar,
sehingga knob tersebut hanya ada untuk family itu dan tidak memiliki flag CLI.

| Family | Knob | Efek |
|---|---|---|
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste` | Probabilitas augmentasi instance copy-paste, hanya `task="segment"` |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `copy_paste_mode` | `"flip"` memakai ulang sampel yang sama dalam keadaan dicerminkan, `"mixup"` mengambil sampel kedua |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `rot90` | Probabilitas rotasi acak 90 derajat |
| YOLOv9, YOLOv9-E2E, YOLOv9-P2 | `max_labels` | Batas ground truth per gambar dalam transform pelatihan, default 300 |
| RF-DETR | `copy_paste`, `copy_paste_mode` | Copy-paste untuk `task="segment"`, hanya mode `"flip"` |
| RF-DETR | `crop_resize_prob` | Probabilitas random crop-resize dalam pipeline native, default 0.5 |
| D-FINE, EdgeCrafter | `crop_resize_prob` | Probabilitas random crop-resize, hanya `task="segment"` |
| EdgeCrafter, YOLO-NAS | `brightness_contrast_prob`, `affine_prob` | Probabilitas jitter pada jalur pose dan probabilitas affine yang memperhitungkan keypoint |

`max_labels` adalah knob yang menghilangkan data. Kotak yang melebihi batas
dibuang dengan peringatan satu kali, bukan error, sehingga gambar padat seperti
fotografi udara memerlukan nilai yang dinaikkan. Memori penetapan label
mengikuti gambar terpadat dalam batch, bukan batas tersebut.

Mosaic dan MixUp dinonaktifkan untuk pelatihan bounding box berorientasi,
terlepas dari nilai knob, karena augmentasi yang memperhitungkan titik sudut
untuk kotak berotasi belum diimplementasikan.

## Terkait

- [Matriks augmentasi](/docs/reference/augmentation-matrix) untuk status setiap
  family pada setiap knob, rentangnya, dan default setiap family.
- [Hyperparameter](/docs/train/hyperparameters) untuk `no_aug_epochs` sebagai
  argumen jadwal dan bagian lain dari `train()`.
- [Dataset](/docs/train/datasets) untuk format label yang dipakai oleh transform
  ini.
