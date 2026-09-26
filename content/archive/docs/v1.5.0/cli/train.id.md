---
title: libreyolo train
seo_title: referensi perintah libreyolo train
description: >-
  Melatih model dari baris perintah: seluruh 59 argumen beserta nilai
  default-nya, cara default tiap family menimpanya, dan argumen mana saja yang
  diabaikan sebuah family.
lead: >-
  Melatih satu model pada satu dataset, lalu menulis checkpoint, metrik dan log
  ke sebuah direktori run. Setiap argumen di bawah punya nilai default dari
  definisi perintah, yang bisa diganti oleh config pelatihan milik family model
  itu sendiri.
keywords:
  - libreyolo train cli
  - perintah training libreyolo
  - training yolo lewat command line
  - argumen libreyolo train
  - cara training yolo dengan dataset sendiri
  - freeze layer yolo
last_verified: 1.5.0
meta:
  - label: Perintah
    value: libreyolo train
    mono: true
  - label: Wajib
    value: data
    mono: true
  - label: Keluaran
    value: 'Checkpoint, metrik dan log di dalam runs/train/exp'
snippets:
  examples:
    - label: Dasar
      language: bash
      code: >
        # coco8.yaml disertakan dalam paket dan mengunduh 8 gambarnya saat
        pertama dipakai.

        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml epochs=10 imgsz=640
        batch=8
    - label: Periksa dulu config hasil resolusi
      language: bash
      code: >
        # Mencetak apa yang akan dipakai run ini, termasuk default family, lalu
        keluar

        # tanpa melatih atau memuat data.

        libreyolo train model=LibreDFINEn.pt data=coco8.yaml epochs=10
        dry_run=true
    - label: Run bernama dengan resep eksplisit
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml \
          epochs=50 batch=8 optimizer=adamw lr0=0.001 weight_decay=0.0001 \
          patience=20 save_period=5 project=runs/train name=yolo9s-coco8 exist_ok=true
source_hash: 3aad4298310d3081
---

## Sinopsis

```bash
libreyolo train data=<dataset.yaml> [model=<name|path>] [key=value ...]
```

Argumen berbentuk pasangan `key=value`, dan bentuk POSIX juga berlaku, jadi
`epochs=50` dan `--epochs 50` adalah argumen yang sama. Boolean menerima `true`
dan `false`: `amp=false` menjadi `--no-amp` pada flag yang punya bentuk negatif.

## Argumen

### Model dan data

| Argumen | Default | Arti |
|---|---|---|
| `data` | | Path ke YAML dataset (format YOLO, mis. `coco8.yaml`). Wajib |
| `model` | `yolox-s` | Nama model atau path ke bobot |
| `task` | | Override task secara eksplisit: `detect`, `segment`, `semantic`, `pose`, `classify`, `gaze`, `obb`, `point`, `depth` |
| `pretrained` | `true` | Pakai bobot pretrained. `false` membangun arsitekturnya dan melatih dari nol |
| `allow_download_scripts` | `false` | Izinkan Python yang tertanam di blok download pada YAML dataset |

### Loop pelatihan

| Argumen | Default | Arti |
|---|---|---|
| `epochs` | `300` | Epoch pelatihan |
| `batch` | `16` | Ukuran batch per perangkat |
| `imgsz` | `640` | Ukuran gambar pelatihan: `640` (persegi) atau `480x640` (tinggi x lebar) |
| `device` | `auto` | Perangkat: `0`, `cpu`, `mps`, `auto` |
| `workers` | `4` | Worker dataloader |
| `cache` | `false` | Cache gambar untuk mempercepat pemuatan data: `ram`, `disk`, `true`, `false` |
| `seed` | `0` | Seed acak |
| `resume` | | Lanjutkan pelatihan: `true`, atau path ke sebuah checkpoint |
| `amp` | `true` | Automatic Mixed Precision |
| `amp_dtype` | `float16` | Dtype AMP CUDA: `float16` atau `bfloat16` |
| `cuda_graph` | `false` | Tangkap forward dan backward pelatihan ke dalam CUDA graph. Hanya GPU tunggal dan family yang didukung; sisanya berjalan eager |
| `lora` | `false` | Fine-tuning LoRA, untuk family transformer yang terdaftar di bagian Catatan |
| `freeze` | | Bekukan layer: jumlah berupa integer, daftar indeks, atau nama modul |

### Distilasi

| Argumen | Default | Arti |
|---|---|---|
| `distill_model` | | Teacher: sebuah checkpoint detektor, atau id foundation teacher seperti `dinov2` untuk distilasi fitur backbone |
| `dis` | | Bobot loss distilasi. Default yang dipublikasikan untuk tipe loss tersebut bila tidak diatur |
| `distill_loss_type` | `mgd` | Loss fitur untuk teacher detektor: `mgd`, `cwd`. Foundation teacher selalu memakai `feat_mse` |

### Optimizer

| Argumen | Default | Arti |
|---|---|---|
| `optimizer` | `sgd` | Optimizer: `sgd`, `adam`, `adamw` |
| `lr0` | `0.01` | Learning rate (laju pembelajaran) awal |
| `momentum` | `0.937` | Momentum SGD, sekaligus koefisien momen pertama untuk optimizer Adam |
| `weight_decay` | `0.0005` | Regularisasi L2 |
| `nesterov` | `true` | Momentum Nesterov |

### Scheduler

| Argumen | Default | Arti |
|---|---|---|
| `scheduler` | `yoloxwarmcos` | Tipe jadwal LR |
| `warmup_epochs` | `5` | Durasi warmup |
| `warmup_lr_start` | `0.0` | LR awal warmup |
| `min_lr_ratio` | `0.05` | Rasio LR minimum |
| `lr_drop` | `100` | Epoch penurunan LR bertahap untuk RF-DETR |

### Augmentasi

| Argumen | Default | Arti |
|---|---|---|
| `mosaic` | `1.0` | Probabilitas mosaic |
| `mixup` | `1.0` | Probabilitas mixup |
| `hsv_prob` | `1.0` | Probabilitas jitter HSV |
| `flip_prob` | `0.5` | Probabilitas flip horizontal |
| `degrees` | `10.0` | Rentang rotasi, plus dan minus, dalam derajat |
| `translate` | `0.1` | Rasio translasi |
| `shear` | `2.0` | Sudut shear |
| `mosaic_scale` | `(0.1,2.0)` | Rentang skala mosaic |
| `mixup_scale` | `(0.5,1.5)` | Rentang skala mixup |
| `no_aug_epochs` | `15` | Matikan augmentasi pada N epoch terakhir |

### EMA

| Argumen | Default | Arti |
|---|---|---|
| `ema` | `true` | Exponential Moving Average |
| `ema_decay` | `0.9998` | Faktor decay EMA |

### Validasi selama pelatihan

| Argumen | Default | Arti |
|---|---|---|
| `val` | `true` | Jalankan validasi selama pelatihan |
| `eval_interval` | `10` | Validasi setiap N epoch |
| `max_det` | `300` | Prediksi maksimum per gambar setelah NMS validasi |
| `eval_max_det` | | Batas evaluator COCO. Konvensi AP@100 pycocotools bila tidak diatur |
| `faster_coco_eval` | `true` | Pakai backend C++ faster-coco-eval untuk metrik COCO bila terpasang; kalau tidak, kembali ke pycocotools |
| `save_plots` | `false` | Simpan plot validasi akhir selama pelatihan |
| `patience` | `50` | Patience early stopping. `0` mematikannya |

### Keluaran

| Argumen | Default | Arti |
|---|---|---|
| `project` | `runs/train` | Root direktori keluaran |
| `name` | `exp` | Nama eksperimen |
| `exist_ok` | `false` | Pakai ulang direktori keluaran yang sudah ada |
| `save_period` | `10` | Simpan checkpoint setiap N epoch |
| `log_interval` | `10` | Catat loss setiap N batch |

### Flag agent

| Argumen | Default | Arti |
|---|---|---|
| `json` | `false` | Keluaran JSON ke stdout |
| `quiet` | `false` | Redam stderr |
| `dry_run` | `false` | Resolusi lalu cetak config tanpa menjalankannya |
| `help_json` | `false` | Cetak skema perintah sebagai JSON lalu keluar |

## Contoh

<code-tabs name="examples" />

## Catatan

### Default di atas tidak selalu jadi nilai yang dipakai

Setiap family model membawa config pelatihannya sendiri, dan di tempat config
itu berbeda dari config dasar, nilainya menggantikan default perintah untuk
setiap argumen yang tidak diatur secara eksplisit. Nilai yang diatur sendiri
selalu menang. `libreyolo cfg` mencetak default dasar berikut override
per family, dan itulah cara melihat apa yang benar-benar akan dipakai sebuah
family.

`imgsz` adalah argumen yang paling terdampak. Default perintahnya `640`, dan
itu bukan input native setiap checkpoint: ukuran deteksi RF-DETR yang
dipublikasikan adalah 384, 512, 576 dan 704, sedangkan checkpoint YOLOX `n` dan
`t` memakai 416. RF-DETR dan DEIMv2 ditangani dengan hanya meneruskan `imgsz`
ketika nilainya diatur secara eksplisit, sehingga di luar itu ukuran mereka
sendiri yang tetap berlaku. Family lain menerima nilai apa adanya dan dilatih
pada ukuran tersebut. FOMO yang paling ketat: setiap ukuran hanya menerima
input native-nya (96, 192 dan 224), jadi run FOMO perlu `imgsz` yang cocok atau
prosesnya berhenti dengan error. RF-DETR juga menuntut nilainya habis dibagi
ukuran patch dikali jumlah window, dan melaporkan dua ukuran legal terdekat
bila tidak.

### Argumen yang diabaikan sebuah family

Tidak setiap family membaca setiap argumen, dan argumen augmentasi adalah
tempat hal itu paling terlihat. RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETRv4 dan
DINOv2 dilatih lewat pipeline pass-through tanpa mosaic, tanpa mixup dan tanpa
affine warp, jadi `mosaic`, `mixup`, `hsv_prob`, `degrees`, `translate`,
`shear`, `mosaic_scale` dan `mixup_scale` tidak mengenai apa pun di sana. EC
memakai pipeline yang sama tapi tetap membaca `hsv_prob`, `degrees` dan
`translate` ketika task-nya pose. Family klasifikasi, SegFormer dan NAFNet
mengabaikan seluruh kumpulan itu berikut `flip_prob`, karena flip mereka
berjalan pada probabilitas tetap, bukan probabilitas yang bisa diatur. YOLO-NAS
hanya mengabaikan `mosaic`, sebab ia justru melakukan augmentasi dengan affine
per sampel yang selalu aktif. RF-DETR mengabaikan tiga argumen lagi di luar
daftar itu: `optimizer`, `momentum` dan `nesterov`.

Mengatur salah satunya bukan error. Run akan mencatat satu baris ke stderr yang
menyebut family dan argumen yang akan diabaikan, lalu melatih, dan baris itu
adalah daftar otoritatif untuk versi yang terpasang. Baris itu juga satu-satunya
sinyal, jadi run terskrip dengan `quiet=true` ikut meredam peringatan tersebut
bersama semua keluaran stderr lainnya.

`val=false` adalah kasus serupa. Nilai itu menetapkan `eval_interval` ke `0`
untuk sebagian besar family; RF-DETR tidak bisa mematikan validasi lewat cara
itu dan mencatat bahwa permintaan tersebut diabaikan.

### Perilaku lain yang perlu diketahui

`lora=true` diterima oleh RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETR v1, v2 dan v4,
EC serta ConvNeXt. Family lain akan keluar dengan `config_unsupported`
alih-alih melatih tanpa LoRA.

`pretrained=false` yang digabung dengan `resume` ditolak pada family yang
mendukung pelatihan dari nol, karena keduanya meminta hal yang berlawanan.

`mosaic` dan `mixup` adalah penulisan versi baris perintah dari field config
`mosaic_prob` dan `mixup_prob`. Pada family yang mixup-nya hanya berlaku untuk
sampel mosaic, `mixup` di atas nol dengan `mosaic` bernilai nol tidak pernah
aktif, dan run akan memberi tahu hal itu.

`dry_run=true` meresolusi referensi model, menerapkan default family, dan
mencetak config yang akan dipakai untuk melatih. Perintah ini tidak memuat
dataset, jadi ini cara murah untuk memastikan sebuah argumen sampai pada nilai
yang diharapkan.

stdout membawa objek hasil akhir; progres dan peringatan dikirim ke stderr.
Kode keluar adalah `0` bila berhasil, `2` untuk error penggunaan atau
konfigurasi, `3` bila dataset tidak ditemukan atau tidak terbaca, `4` bila
model tidak bisa dimuat, dan `1` untuk kegagalan runtime lainnya.

Terkait: [`libreyolo doctor`](/docs/cli/doctor) untuk memeriksa dataset sebelum
memutuskan menjalankan pelatihan, [`libreyolo monitor`](/docs/cli/monitor)
untuk memantau run di browser, [`libreyolo val`](/docs/cli/val) untuk mengukur
hasilnya.
