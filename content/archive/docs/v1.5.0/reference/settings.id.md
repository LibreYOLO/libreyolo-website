---
title: Pengaturan
seo_title: Variabel environment dan direktori LibreYOLO
description: >-
  Setiap variabel environment yang dibaca LibreYOLO, direktori yang ditulis,
  token yang diperlukan, dan toggle yang mengubah jalur kode yang dijalankan.
lead: >-
  LibreYOLO tidak memiliki file konfigurasi. Perilaku yang bukan argumen fungsi
  dikendalikan oleh variabel environment dan sejumlah kecil direktori
  konvensional, yang semuanya tercantum di sini.
keywords:
  - LIBREYOLO_DATASETS_DIR
  - LIBREYOLO_KERNELS
  - LIBREYOLO_FASTER_COCO_EVAL
  - HF_TOKEN
  - direktori bobot libreyolo
  - cache libreyolo
last_verified: 1.5.0
verification: >-
  Variabel ditemukan dengan mencari os.environ dan os.getenv dalam
  libreyolo/**/*.py pada v1.5.0; semantik dibaca di setiap lokasi penggunaan.
  Konvensi direktori dibaca dari libreyolo/data/utils.py,
  libreyolo/utils/download.py, libreyolo/export/exporter.py,
  libreyolo/models/base/model.py, dan libreyolo/models/sam3dbody/mhr_body.py.
snippets:
  usage:
    - label: Arahkan root dataset ke lokasi lain
      language: bash
      code: |
        export LIBREYOLO_DATASETS_DIR=/data/datasets
        python -c "from libreyolo.data import DATASETS_DIR; print(DATASETS_DIR)"
    - label: Baca nilai yang telah di-resolve dari Python
      language: python
      code: |
        from libreyolo.data import DATASETS_DIR

        # Default-nya ~/datasets; LIBREYOLO_DATASETS_DIR menimpanya saat import.
        print(DATASETS_DIR)
source_hash: 462f1288582225ce
---

## Variabel environment

| Variabel | Default | Efek |
|---|---|---|
| `LIBREYOLO_DATASETS_DIR` | `~/datasets` | Root dataset. Dibaca satu kali saat import ke `libreyolo.data.DATASETS_DIR` |
| `LIBREYOLO_FASTER_COCO_EVAL` | tidak ditetapkan | Menimpa flag validasi `faster_coco_eval`. `1`, `true`, `yes`, atau `on` memaksa backend lebih cepat aktif, nilai lain memaksanya nonaktif, sedangkan nilai yang tidak ditetapkan mengikuti flag konfigurasi |
| `LIBREYOLO_KERNELS` | tidak ditetapkan | Pemilihan kernel. `off` atau `reference` memaksa implementasi referensi; nilai lain hanya memilih implementasi yang terdaftar dengan nama tersebut |
| `LIBREYOLO_QUANT_KERNELS` | tidak ditetapkan | Alias lama untuk `LIBREYOLO_KERNELS`, hanya dibaca ketika variabel tersebut tidak ditetapkan |
| `LIBREYOLO_HUB_KERNELS` | tidak ditetapkan | `0`, `false`, `off`, atau `no` menonaktifkan pemuatan kernel Hugging Face Hub. Nilai lain, termasuk tidak ditetapkan, membiarkannya aktif |
| `LIBREYOLO_MHR_PATH` | `~/.cache/libreyolo/mhr/mhr_model.pt` | Lokasi model tubuh MHR yang digunakan task `mesh` |
| `LIBRELABEL_ENABLE_LOCATE` | tidak ditetapkan | Harus tepat bernilai `1`, `true`, `yes`, atau `on` agar asisten LocateAnything tersedia dalam alat labeling. Nilai lain membuatnya tetap nonaktif |
| `SAM_3D_BODY_PATH` | tidak ditetapkan | Path ke package SAM 3D Body untuk family mesh, ketika tidak diberikan kepada constructor |
| `HF_TOKEN` | tidak ditetapkan | Token akses Hugging Face, digunakan untuk repository gated |

<code-tabs name="usage" />

`LIBREYOLO_DATASETS_DIR` dibaca saat import, sehingga menetapkannya setelah
`libreyolo.data` diimpor tidak berpengaruh terhadap `DATASETS_DIR`.

Kernel Hub menggunakan keikutsertaan dua bagian. Pengambilan saat runtime hanya
terjadi ketika package opsional `kernels` diinstal, sehingga menginstal
`libreyolo[hub-kernels]` berarti ikut serta dan `LIBREYOLO_HUB_KERNELS=0` berarti
keluar. Instalasi tanpa ekstra tersebut tidak terpengaruh oleh keduanya.

Pemilihan kernel juga menghentikan import lebih awal: ketika
`LIBREYOLO_KERNELS` memaksa `off` atau `reference`, provider terakselerasi di
dalam source tree sama sekali tidak diimpor. Registry yang dikendalikan ketiga
variabel ini didokumentasikan pada [kernel](/docs/reference/kernels).

## Variabel yang ditetapkan library

Variabel berikut ditulis, bukan dibaca, sehingga menetapkannya secara manual
bukan jalur yang didukung.

| Variabel | Ditetapkan oleh |
|---|---|
| `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` | Helper spawn DDP, satu nilai per worker process |
| `CUDA_VISIBLE_DEVICES` | Dipersempit sementara selama setup terdistribusi, lalu dipulihkan |
| `PYTORCH_ENABLE_MPS_FALLBACK` | Ditetapkan ke `1` oleh trainer EC dengan `setdefault`, sehingga nilai yang sudah ada menang |
| `MOMENTUM_ENABLED` | Ditetapkan dengan `setdefault` oleh loader family mesh |

`LOCAL_RANK` juga berfungsi sebagai sinyal mode terdistribusi: keberadaannya di
environment adalah cara kode pelatihan mendeteksi bahwa kode berjalan di bawah
DDP.

## Variabel logger

Logger pelatihan opsional menggunakan nilai default environment untuk nama
project.

| Variabel | Default | Digunakan oleh |
|---|---|---|
| `WANDB_PROJECT` | `libreyolo` | Logger Weights and Biases, ketika project tidak diberikan |
| `COMET_PROJECT_NAME` | `libreyolo` | Logger Comet, ketika project tidak diberikan |

Autentikasi untuk layanan tersebut mengikuti alat masing-masing, bukan
LibreYOLO.

## Token

`HF_TOKEN` adalah token akses Hugging Face. Ketika tidak ditetapkan, token dibaca
dari `~/.cache/huggingface/token`, yaitu lokasi yang ditulis oleh login CLI
Hugging Face. Kedua jalur dapat digunakan.

Token hanya diperlukan untuk repository gated. SAM 3 adalah contoh bawaan:
bobotnya diunduh dari repository gated dengan lisensi khusus, sehingga
persyaratan harus diterima pada halaman repository dan sesi harus diautentikasi.

## Direktori

| Path | Isi |
|---|---|
| `weights/` | Checkpoint yang diunduh, snapshot Hugging Face yang diunduh, dan artefak hasil ekspor |
| `~/datasets` | Root dataset, kecuali `LIBREYOLO_DATASETS_DIR` menentukan lokasi lain |
| `~/.cache/huggingface/token` | Token Hugging Face, ketika tidak berada di `HF_TOKEN` |
| `~/.cache/libreyolo/mhr/mhr_model.pt` | Model tubuh MHR, kecuali `LIBREYOLO_MHR_PATH` menentukan lokasi lain |
| `runs/track/` | Output default untuk `model.track(save=True)` |

`weights/` bersifat relatif terhadap direktori kerja. Nama file tanpa path
di-resolve melalui direktori tersebut, sehingga `LibreYOLO("LibreYOLO9t.pt")`
mencari `weights/LibreYOLO9t.pt` dan mengunduhnya ke sana jika tidak ada.
`model.export()` menulis ke direktori yang sama ketika `output_path` tidak
diberikan. Tier yang berdekatan mengunduh snapshot multi-file ke
`weights/<Prefix><size>/`.

## Perilaku pengunduhan

Pengunduhan bobot dicoba ulang tiga kali dengan backoff, dilanjutkan dari file
parsial, dan dilindungi oleh lock file agar dua process tidak mengambil
checkpoint yang sama secara bersamaan. Family yang mengambil dari host pihak
ketiga dapat menetapkan checksum dan berhenti jika terjadi ketidakcocokan.

Beberapa pengunduhan mencetak pemberitahuan lisensi sebelum dimulai.
Pemberitahuan tersebut merupakan bagian dari jalur pengunduhan dan tidak dapat
disembunyikan melalui konfigurasi.

## Backend validasi

`model.val()` menerima `faster_coco_eval=True` secara default dan kembali ke
pycocotools ketika package tersebut tidak diinstal, dengan satu kali peringatan.
Menetapkan `LIBREYOLO_FASTER_COCO_EVAL` menimpa flag per pemanggilan. Inilah yang
sebaiknya digunakan oleh benchmark harness yang tidak dapat mengubah konfigurasi
per run. Backend yang benar-benar dijalankan dilaporkan pada
`model.last_eval_backend`.

## Script pengunduhan dataset

YAML dataset dapat memiliki field `download` yang berisi Python. Kode tersebut
tidak dieksekusi kecuali `allow_download_scripts=True` diberikan pada
pemanggilan yang membacanya, yaitu argumen fungsi pada `val()` dan `export()`,
bukan variabel environment.
