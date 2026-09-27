---
title: libreyolo predict
seo_title: referensi perintah libreyolo predict
description: >-
  Menjalankan inferensi dari baris perintah: setiap argumen, nilai bawaannya
  yang dibaca dari definisi CLI, dan flag yang mengubah apa yang muncul di
  stdout.
lead: >-
  Menjalankan model yang sudah dimuat pada satu source lalu mencetak
  prediksinya. Source dapat berupa gambar, direktori, video, URL atau live
  stream; modelnya dapat berupa checkpoint atau artefak hasil ekspor.
keywords:
  - libreyolo predict cli
  - inference yolo command line
  - cara prediksi yolo di terminal
  - argumen libreyolo predict
  - output json deteksi objek
last_verified: 1.5.0
meta:
  - label: Perintah
    value: libreyolo predict
    mono: true
  - label: Wajib
    value: source
    mono: true
  - label: Output
    value: >-
      Prediksi di stdout. Dengan save=true, berkas beranotasi di
      runs/detect/predict
snippets:
  examples:
    - label: Dasar
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Menyimpan gambar beranotasi
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt save=true \
          project=runs/detect name=parkour exist_ok=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 'Kelas terfilter, JSON di stdout'
      language: bash
      code: >
        # kelas 0 adalah person pada daftar kelas COCO yang dibawa checkpoint
        ini.

        libreyolo predict model=LibreYOLO9s.pt classes="[0]" conf=0.4 max_det=50
        \
          json=true quiet=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: 7e46c7ed7dd9e6c4
---

## Sinopsis

```bash
libreyolo predict source=<path|url|index> [model=<name|path>] [key=value ...]
```

Argumen berupa pasangan `key=value`. Perintah yang sama juga menerima bentuk
POSIX, sehingga `conf=0.4` dan `--conf 0.4` dapat saling menggantikan, dan
boolean yang ditulis `save=true` menjadi `--save`. Nama yang memakai garis bawah
menerima kedua ejaan: `max_det=50` dan `--max-det 50` menuju opsi yang sama.

`libreyolo detect predict ...` juga diterima dan berperilaku sama persis; kata
task-nya dibuang sebelum parsing.

## Argumen

| Argumen | Default | Arti |
|---|---|---|
| `source` | | Path gambar, direktori, atau URL. Wajib |
| `model` | `yolox-s` | Nama atau path model |
| `conf` | `0.25` | Ambang batas skor keyakinan (confidence) |
| `iou` | `0.45` | Ambang IoU untuk NMS |
| `imgsz` | | Ukuran gambar masukan: `640` (persegi) atau `480x640` (HxW). Ukuran masukan milik model sendiri bila tidak diisi |
| `classes` | | Filter berdasarkan ID kelas, misalnya `[0,2,5]`. Integer tunggal juga diterima |
| `max_det` | `300` | Deteksi maksimum per gambar |
| `half` | `false` | Inferensi FP16 (hanya CUDA, model harus mendukungnya) |
| `save` | `false` | Menyimpan gambar beranotasi |
| `batch` | `1` | Jumlah gambar per forward pass untuk source berupa direktori. Di atas 1 menjalankan inferensi batch sesungguhnya pada model yang mendukungnya |
| `stream` | `false` | Mengeluarkan hasil secara bertahap. Otomatis aktif untuk webcam dan live stream |
| `stream_buffer` | `false` | Menyangga setiap frame live, bukan hanya menyimpan yang terbaru |
| `vid_stride` | `1` | Memproses setiap frame video atau frame live ke-N |
| `show` | `false` | Menampilkan hasil video dan live; `q` menghentikannya |
| `tiling` | `false` | Inferensi berbasis tile untuk gambar besar |
| `overlap_ratio` | `0.2` | Rasio tumpang tindih antartile |
| `output_path` | | Path output eksplisit. Bila tidak diisi, `project/name` saat `save=true` |
| `color_format` | `auto` | Warna masukan: `auto`, `rgb`, `bgr` |
| `output_file_format` | | Format output: `jpg`, `png`, `webp` |
| `device` | `auto` | Perangkat: `0`, `cpu`, `mps`, `auto` |
| `face_detector` | | Model detektor wajah (path atau nama CLI). Wajib untuk model gaze |
| `gallery` | | Galeri wajah `.npz` dari `libreyolo enroll` sebagai acuan identifikasi wajah. Hanya untuk model face-embedding |
| `gallery_threshold` | `0.4` | Ambang cosine untuk kecocokan identitas dari galeri |
| `project` | `runs/detect` | Direktori akar output |
| `name` | `predict` | Nama eksperimen |
| `exist_ok` | `false` | Memakai ulang direktori output yang sudah ada |
| `json` | `false` | Output JSON ke stdout |
| `quiet` | `false` | Menyembunyikan keluaran stderr |
| `verbose` | `false` | Keluaran stderr yang detail |
| `help_json` | `false` | Mencetak skema perintah sebagai JSON lalu keluar |

## Contoh

<code-tabs name="examples" />

## Catatan

Artefak hasil ekspor dimuat dengan cara yang sama seperti checkpoint, jadi
`model=weights/LibreYOLO9s.onnx` dan `model=weights/LibreYOLO9s.engine` adalah
nilai yang sah untuk `model`. Tiga opsi ditolak, bukan diabaikan, pada runtime
tersebut: `tiling`, `overlap_ratio` dan `output_file_format` keluar dengan
`config_unsupported` bila backend runtime tidak dapat memenuhinya.

`half` justru sebaliknya. Runtime hasil ekspor menerimanya dan berjalan di FP16;
inferensi PyTorch native mencatat bahwa flag itu diabaikan lalu melanjutkan di
FP32.

Model gaze bekerja dua tahap dan tidak punya detektor sendiri, sehingga
`face_detector` wajib diisi untuknya. `gallery` hanya berlaku untuk model yang
task-nya `embed`; memberikannya ke model lain akan keluar dengan
`config_unsupported`.

stdout hanya membawa hasil dan tidak lebih; progres, peringatan dan error
dikirim ke stderr. `json=true` mencetak satu objek JSON per pemanggilan, atau
satu per frame saat streaming, masing-masing memuat `schema_version`.
`quiet=true` membungkam stderr. Keduanya sekaligus memberi pembaca mesin aliran
stdout yang bersih.

Exit code-nya `0` bila berhasil, `2` untuk kesalahan pemakaian atau konfigurasi,
`3` bila source tidak ditemukan, `4` bila model tidak dapat dimuat, dan `1`
untuk kegagalan runtime lainnya.

`help_json=true` mencetak parameter, tipe, nilai bawaan dan flag milik perintah
ini sebagai JSON tanpa menjalankan apa pun, dan itulah cara andal untuk membaca
ulang tabel di atas dari versi yang terpasang.

Terkait: [`libreyolo val`](/docs/cli/val) untuk metrik terukur pada sebuah
dataset, [`libreyolo export`](/docs/cli/export) untuk menghasilkan artefak
runtime yang disebut di atas.
