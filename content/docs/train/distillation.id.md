---
title: Distilasi pengetahuan
seo_title: Distilasi pengetahuan di LibreYOLO
description: >-
  Latih detector kecil menggunakan teacher yang lebih besar atau backbone DINOv2
  beku: loss MGD, CWD, dan feature-MSE, tap point, serta dukungan family.
lead: >-
  Distilasi menambahkan term loss kedua yang menarik feature map antara milik
  student menuju feature map teacher beku. LibreYOLO mengambil feature dengan
  forward hook, sehingga head dan loss milik teacher tidak pernah terlibat.
keywords:
  - knowledge distillation
  - masked generative distillation
  - channel wise distillation
  - feature distillation
  - teacher DINOv2
  - training teacher student
  - MGD loss
  - CWD loss
last_verified: 1.5.0
snippets:
  detector:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Checkpoint lebih besar dari family yang sama membimbing model kecil.
        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            distill_model="LibreYOLO9c.pt",
            distill_loss_type="mgd",
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 distill_model=LibreYOLO9c.pt distill_loss_type=mgd
  foundation:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # ViT self-supervised beku membimbing satu tahap backbone.
        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            distill_model="dinov2",
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 distill_model=dinov2
  tuned:
    - label: Tuning loss
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="my-dataset.yaml",
            distill_model="LibreYOLO9c.pt",
            distill_loss_type="cwd",
            dis=1.0,           # bobot distilasi global
            distill_tau=1.0,   # suhu softmax CWD
        )
source_hash: 7210031328f6826f
---

## Distilasi dari checkpoint lebih besar

Menetapkan `distill_model` mengaktifkan distilasi. Nilainya adalah checkpoint
teacher yang dimuat melalui factory yang sama seperti model lain.

<code-tabs name="detector" />

Teacher menjalankan forward di bawah `no_grad`, dan di bawah autocast jika AMP
aktif, sehingga model beku tidak membayar komputasi presisi penuh pada setiap
langkah. Forward hook menangkap feature map di tap point bernama, loss
membandingkannya dengan feature student, lalu hasilnya ditambahkan ke loss
pelatihan dan dilaporkan sebagai komponen bernama `distill`.

## Distilasi dari foundation backbone beku

ViT self-supervised dapat membimbing satu tahap backbone student. Feature teacher
berasal dari feature extractor-nya sendiri, bukan hook, dan loss menangani
ketidakcocokan antara grid patch dan stride konvolusional.

<code-tabs name="foundation" />

`distill_model` mengenali `dinov2`, yaitu DINOv2-base, serta `dinov2_vits14`,
`dinov2_vitb14`, `dinov2_vitl14`, `dinov2-small`, `dinov2-base`,
`dinov2-large`, dan semua hub id mentah yang diawali `facebook/dinov2`. Nilai
lain diperlakukan sebagai path checkpoint teacher.

Jalur ini menggunakan `feat_mse` tanpa memandang `distill_loss_type` dan
memerlukan instalasi `transformers`. Teacher yang dimuat dengan key bobot hilang
menghentikan proses, bukan melakukan distilasi dari backbone yang sebagian acak.

## Family yang didukung

Dukungan distilasi merupakan method pada model student, dan ada dua method.

`get_distill_config()` menyediakan tap point multiskala yang dibimbing teacher
detector. YOLOv9, YOLOX, dan RF-DETR mengimplementasikannya.

`get_backbone_distill_config()` menyediakan satu tahap backbone yang dibimbing
foundation teacher. YOLOv9 mengimplementasikannya dan menjadi satu-satunya family
yang demikian.

Family lain memunculkan error, bukan berlatih tanpa loss:

```text
LibreDFINE does not implement get_distill_config(). Distillation is not yet
supported for the 'dfine' family.
```

```text
Foundation-model distillation into the 'yolox' family is not supported yet
(no get_backbone_distill_config()).
```

## Tap point

Tap point ditetapkan per family dan per peran, sehingga teacher dan student tidak
harus memakai arsitektur yang sama; keduanya harus memiliki stride feature yang
cocok.

| Family | Peran | Tap point | Stride |
|---|---|---|---|
| YOLOv9 | teacher atau student | `neck.elan_up2`, `neck.elan_down1`, `neck.elan_down2` | 8, 16, 32 |
| YOLOv9 | foundation student | `backbone.elan3` | 16 |
| YOLOX | teacher atau student | `backbone.C3_p3`, `backbone.C3_n3`, `backbone.C3_n4` | 8, 16, 32 |
| RF-DETR | teacher atau student | `model.backbone.0.projector.stages.0` | diperiksa saat setup |

Stride yang tidak cocok memunculkan error sebelum pelatihan dimulai:

```text
Teacher and student must have matching strides. Teacher: [8, 16, 32],
Student: [16]
```

Pemeriksaan tersebut dilewati untuk foundation teacher, karena perbedaan grid
memang merupakan tujuan jalur itu.

## Tiga loss

`distill_loss_type` memilih feature loss untuk teacher detector. Foundation
teacher selalu menggunakan `feat_mse`.

`mgd`, masked generative distillation, menutupi sebagian posisi spasial student
dan melatih generator dua-konvolusi kecil untuk merekonstruksi feature map penuh
teacher dari bagian yang tersisa. `distill_mask_ratio` menentukan fraksi yang
ditutupi, dengan default 0,65.

`cwd`, channel-wise distillation, mengubah aktivasi spasial setiap channel
menjadi distribusi probabilitas dan meminimalkan KL divergence per channel.
`distill_tau` adalah suhu softmax, dengan default 1,0.

`feat_mse` menyelaraskan channel student terhadap teacher dengan konvolusi 1x1,
mengubah ukuran grid teacher secara bilinear agar cocok dengan student, lalu
mengambil mean squared error. `distill_normalize=True` lebih dahulu melakukan
normalisasi L2 pada kedua feature map di dimensi channel, sehingga pencocokan
hanya berdasarkan sudut dan tidak bergantung skala. Default-nya `False`.

`dis` adalah bobot global yang diterapkan di atasnya. Jika tidak ditetapkan,
setiap loss memakai default terbitannya sendiri: 2e-5 untuk MGD, 1,0 untuk CWD,
dan 1,0 untuk feature MSE. Nilai tersebut berbeda lima orde magnitudo, sehingga
bobot yang di-tuning untuk satu jenis loss tidak bermakna bagi jenis lain.

<code-tabs name="tuned" />

`distill_mask_ratio`, `distill_tau`, dan `distill_normalize` tidak memiliki flag
CLI. Ketiganya merupakan argumen Python atau key YAML `cfg=`. Distilasi RF-DETR
secara keseluruhan juga hanya tersedia di Python karena pemetaan argumen CLI-nya
tidak membawa key distilasi.

## Adapter, checkpoint, dan multi-GPU

Setiap loss membangun modul kecil yang dapat dilatih di luar student: adapter
channel 1x1 dan generator MGD. Modul tersebut mendapat group parameter optimizer
sendiri pada learning rate efektif proses.

Modul itu ditulis ke checkpoint dalam key `distiller` dan dipulihkan saat resume,
sehingga proses lanjutan tidak memulai projectornya dari keadaan dingin.

Di bawah DDP, adapter berada di luar student yang dibungkus sehingga reducer DDP
tidak pernah melihat gradiennya. Trainer melakukan all-reduce secara eksplisit
setiap langkah agar semua rank melatih adapter yang sama.

CUDA graph capture tidak tersedia pada proses distilasi. Memberikan
`cuda_graph=True` mencatat satu baris dan berlatih secara eager. Lihat
[Performa pelatihan](/docs/train/performance).

## Terkait

- [Pembekuan lapisan](/docs/train/layer-freezing) dan
  [fine-tuning LoRA](/docs/train/lora), yang keduanya dapat digabungkan dengan
  distilasi.
- [Hyperparameter](/docs/train/hyperparameters) untuk bagian lain `train()`.
