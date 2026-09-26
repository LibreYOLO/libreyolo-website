---
title: Hyperparameter
seo_title: Hyperparameter pelatihan di LibreYOLO
description: >-
  Argumen train() yang penting: epochs, batch, lr0, optimizer, EMA, autobatch,
  gradient accumulation dan resume, plus mengapa default berbeda per family.
lead: >-
  Setiap argumen pelatihan adalah kolom pada dataclass TrainConfig. Kelas dasar
  mendefinisikan kolom dan defaultnya; setiap model family membuat subclass darinya dan
  mengganti default yang diubah oleh resep yang dipublikasikannya.
keywords:
  - argumen train
  - learning rate
  - ukuran batch
  - autobatch
  - exponential moving average
  - gradient accumulation
  - melanjutkan pelatihan
  - early stopping patience
  - amp bfloat16
  - konfigurasi train yaml
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        results = model.train(
            data="my-dataset.yaml",
            epochs=100,
            batch=16,
            imgsz=640,
            lr0=0.01,
        )

        print(results["best_mAP50_95"])
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 batch=16 imgsz=640 lr0=0.01
  defaults:
    - label: Baca default yang telah diselesaikan dari family
      language: python
      code: |
        from dataclasses import fields

        from libreyolo import LibreYOLO9
        from libreyolo.training.config import TrainConfig

        family_cfg = LibreYOLO9.TRAIN_CONFIG()
        base_cfg = TrainConfig()

        for f in fields(family_cfg):
            family_value = getattr(family_cfg, f.name)
            base_value = getattr(base_cfg, f.name, None)
            if not hasattr(base_cfg, f.name) or family_value != base_value:
                print(f"{f.name}: {family_value}")
    - label: CLI
      language: bash
      code: |
        # Mencetak default train, val, dan predict, termasuk override family.
        libreyolo cfg
  autobatch:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # batch=-1 memeriksa memori GPU dan menyelesaikannya menjadi pangkat dua
        yang konkret.

        model.train(data="my-dataset.yaml", batch=-1, imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml batch=-1
  accumulate:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # 4 micro-batch dari 16 per langkah optimizer, batch efektif 64.
        model.train(data="my-dataset.yaml", batch=16, nbs=64)
  resume:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Memuat checkpoint dari jalannya yang terhenti, lalu minta untuk
        melanjutkan.

        model = LibreYOLO("runs/train/exp/weights/last.pt")

        model.train(data="my-dataset.yaml", epochs=100, resume=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=runs/train/exp/weights/last.pt \
          data=my-dataset.yaml epochs=100 resume=true
  cfg:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Kunci dalam yaml adalah nama kolom TrainConfig. Kwargs eksplisit
        menang.

        model = LibreYOLO("LibreYOLO9s.pt")

        model.train(data="my-dataset.yaml", cfg="my-recipe.yaml", epochs=50)
source_hash: d838d1abd45af40f
---

## Mengatur argumen

`train()` mengambil argumen kata kunci dan CLI mengambil nama yang sama dalam
format `key=value`.

<code-tabs name="train" />

Kedua jalur berakhir pada tempat yang sama. Kwargs diserahkan ke
`TrainConfig.from_kwargs()`, yang membangun dataclass konfigurasi family.

## Salah ketik tidak menyebabkan error

`from_kwargs()` menghapus setiap kunci yang bukan merupakan bidang pada konfigurasi dan mengeluarkan sebuah
Memberi nama `UserWarning`. Pelatihan kemudian dimulai dengan pengaturan default yang ada:

```python
# UserWarning: Kunci konfigurasi pelatihan tidak dikenal (diabaikan): ['learning_rate']
model.train(data="my-dataset.yaml", learning_rate=0.001)
```

Tidak ada yang gagal, proses selesai, dan learning rate tidak pernah menjadi apa yang pemanggil
diminta. Baca peringatan pada epoch pertama dari resep baru. CLI adalah
lebih ketat, karena memvalidasi nama flag sebelum konfigurasi dibuat, sehingga
CLI flag yang salah eja langsung ditolak.

## Default adalah per family

`TrainConfig` mendefinisikan bidang dan default dasar. Setiap family menurunkannya dan
menimpa perubahan resep yang dipublikasikannya, jadi tidak ada jawaban yang benar secara tunggal
ke "apa itu learning rate default".

Default dasar adalah `optimizer="sgd"`, `lr0=0.01`, `momentum=0.937`,
`weight_decay=5e-4`, `scheduler="yoloxwarmcos"`, `epochs=300`, `batch=16`,
`imgsz=640` dan `amp=True`. Tiga contoh seberapa jauh family bergerak dari itu:

| Bidang | Dasar | YOLOv9 | D-FINE | YOLO-NAS |
|---|---|---|---|---|
| `optimizer` | `sgd` | `sgd` | `adamw` | `adamw` |
| `lr0` | `0.01` | `0.01` | `2e-4` | `5e-4` |
| `weight_decay` | `5e-4` | `5e-4` | `1e-4` | `1e-5` |
| `scheduler` | `yoloxwarmcos` | `linear` | `flat_cosine` | `cos` |
| `epochs` | `300` | `300` | `132` | `300` |
| `amp` | `True` | `True` | `False` | `False` |

D-FINE dan DEIM dikirim bersama `amp=False` karena decoder D-FINE membatasi
aktivasi pada 65504, nilai float16 terbesar yang terbatas. YOLO-NAS dan FOMO juga
secara default mematikannya. `--amp` flag di CLI defaultnya ke `True` untuk setiap family, jadi itu
dihitung sebagai disediakan pengguna dan menimpa default family; biarkan saja kecuali
Anda bermaksud untuk mengubahnya.

Untuk membaca default nyata family daripada menebak:

<code-tabs name="defaults" />

## ukuran Batch

`batch` adalah batch global. Dalam pelatihan multi-GPU setiap peringkat memuat
`batch // world_size`, jadi nomor yang Anda berikan adalah jumlah gambar per
dilakukan terlepas dari berapa banyak GPU yang terlibat. Lihat
[Pelatihan Multi-GPU](/docs/train/multi-gpu).

`batch=-1` mengaktifkan autobatch. Trainer memeriksa model dalam mode pelatihan dengan
langkah mundur nyata pada pangkat dua, menyesuaikan garis ke kurva memori, dan memilih
pangkat dua terbesar yang secara ketat di bawah nilai yang diekstrapolasi yang muat dalam
60 persen dari total VRAM.

<code-tabs name="autobatch" />

Memeriksa dalam mode pelatihan dengan langkah mundur adalah inti: pemeriksaan dalam mode inferensi
melewatkan aktivasi yang dipertahankan dan tensor gradien, yang untuk CNN dalam
adalah beberapa kali jejak inferensi. RF-DETR menurunkan fraksi target menjadi
45 persen, karena backward sintetis probe masih meremehkan apa yang dibutuhkan oleh
kriteria probe dan lapisan decoder tambahan.

Autobatch adalah fitur CUDA. Pada CPU atau MPS ia mencatat satu baris dan mempertahankan
batch default.

## Akumulasi gradien

`nbs` menetapkan ukuran nominal, atau efektif, batch. Pelatih mengakumulasi
`round(nbs / batch)` mikro-batch per langkah optimizer.

<code-tabs name="accumulate" />

Dibiarkan sebagai `None`, default, akumulasi dimatikan dan pelatihan tidak berubah.

## Learning rate dan jadwal

`lr0` adalah learning rate awal dan `optimizer` menerima `sgd`, `adam` dan
`adamw`. `momentum` adalah momentum SGD atau beta1 Adam, `weight_decay` adalah L2
istilah, dan `nesterov` berlaku untuk SGD.

Jadwal dibentuk oleh `scheduler`, `warmup_epochs`, `warmup_lr_start` dan
`min_lr_ratio`. `no_aug_epochs` mengatur berapa banyak epoch akhir yang dijalankan tanpa kuat
augmentasi, dan beberapa jadwal menggunakannya untuk membentuk ekornya juga, jadi itu
bukan semata-mata kenop augmentasi. Apa yang dilakukan setiap family dengan setengah augmentasi
darinya ada di [Augmentations](/docs/train/augmentations).

Beberapa keluarga menambahkan tombol tingkat pembelajaran mereka sendiri. `backbone_lr_mult` menyesuaikan
Kelompok backbone melawan head, `clip_max_norm` menetapkan pemotongan gradien, dan
SegFormer menggunakan `head_lr_mult` untuk menjalankan decode head-nya sepuluh kali backbone
tingkat. Ini berada pada subclass konfigurasi family, bukan yang dasar.

## EMA

`ema=True` menyimpan rata-rata bergerak eksponensial dari bobot bersamaan dengan
bobot yang dilatih. Ini diaktifkan secara default di mana-mana kecuali FOMO.

`ema_decay` adalah target decay. Decay meningkat secara bertahap daripada dimulai pada
targetnya: nilai efektif pada update `n` adalah `ema_decay * (1 - exp(-n / tau))`
dengan `tau` default ke 2000, sehingga update awal mengikuti model lebih dekat dan
update akhir melicinkannya. Default Family berkisar dari `0.997` pada pose YOLO-NAS
hingga `0.9998` pada YOLOX dan `0.9999` pada YOLOv9 dan lini DETR.

Bobot EMA adalah yang divalidasi dan yang dibawa oleh `best.pt` dan `last.pt`.
Berat terlatih mentah juga disimpan, di bawah kunci `train_model`, jadi dapat dilanjutkan
berlanjut dari lintasan yang dilatih daripada dari rata-rata.

## Presisi

`amp=True` menjalankan forward pass di bawah CUDA autocast. `amp_dtype` memilih
`float16` (default) atau `bfloat16`; `fp16` dan `bf16` adalah ejaan yang diterima.

Float16 membutuhkan skala loss dinamis dan mendapatkan `GradScaler` langsung. Bfloat16 lebih lebar
jangkauan eksponen tidak, jadi skalernya dibuat tetapi dinonaktifkan, yang menjaga
jalur pengoptimal identik. Meminta bfloat16 pada perangkat CUDA tanpa
Dukungan bfloat16 meningkat saat pengaturan alih-alih menurun secara diam-diam.

## Keluaran, titik pemeriksaan, dan penghentian

Run ditulis ke `project/name`. `project` default ke `runs/train`
di mana-mana, tetapi `name` adalah salah satu override per-family: default dasarnya adalah
`exp`, sementara YOLOv9 menggunakan `yolo9_exp` dan D-FINE menggunakan `dfine_exp`. Dengan
`exist_ok=False`, default, direktori yang ada mendapatkan sufiks yang ditingkatkan
alih-alih ditimpa.

`save_period` menulis `weights/epoch_<N>.pt` tambahan setiap N epoch, di atas
`weights/last.pt` setelah setiap epoch dan `weights/best.pt` setiap kali metrik yang dilacak
membaik. `eval_interval` menentukan seberapa sering validasi dijalankan, dan `patience`
menghentikan run setelah selama itu epoch tanpa perbaikan, dengan `0` menonaktifkan
early stopping.

`cache` mempercepat epoch berulang dengan menyimpan gambar yang telah didekode di RAM (`True` atau
`"ram"`) atau sebagai berkas `.npy` di samping sumber (`"disk"`). Bacaan yang disimpan dalam cache adalah
identik byte dengan yang baru. Dengan pekerja dataloader, `"disk"` adalah yang lebih aman dari
keduanya.

## Ringkasan

`resume=True` melanjutkan perjalanan yang terhenti. checkpoint harus dimuat
pertama, karena resume membacanya dari model, bukan dari argumen terpisah.

<code-tabs name="resume" />

Resume memulihkan bobot yang dilatih, status optimizer, bobot EMA dan
perbarui jumlah, pelacakan metrik terbaik, skala `GradScaler`, dan PyTorch,
Status acak CUDA dan NumPy. Itu dimulai pada checkpoint epoch ditambah satu dan
mempercepat jadwal ke posisi tersebut.

Dua hal yang tidak akan dilakukannya. `resume=True` tidak dapat digabungkan dengan `pretrained`,
yang menimbulkan masalah. Dan ketika kunci metrik-terbaik checkpoint berbeda dari
jalannya saat ini, pelacakan metrik-terbaik direset ke nol dengan peringatan
alih-alih membandingkan nilai yang tidak berarti sama.

## Resep dalam sebuah berkas

`cfg=` memuat pemetaan YAML dari nama-nama bidang `TrainConfig` dan menggabungkannya
di bawah argumen kata kunci eksplisit, jadi sebuah kwarg selalu menang atas berkas.

<code-tabs name="cfg" />

`size` dan `num_classes` dihapus dari berkas, karena instansi model
sudah memilikinya. Tidak ada `--cfg` flag di CLI; jalur berkas adalah Python
argumen.

## Terkait

- [Dataset](/docs/train/datasets) untuk apa yang diterima `data=`.
- [Augmentations](/docs/train/augmentations) untuk kenop augmentasi dan yang mana
  keluarga menghormati mereka.
- [Pembekuan layer](/docs/train/layer-freezing) dan [LoRA](/docs/train/lora) untuk
  melatih sebagian dari bobot.
- [Validasi dan metrik](/docs/train/validation) untuk apa yang dilaporkan oleh jalannya.




