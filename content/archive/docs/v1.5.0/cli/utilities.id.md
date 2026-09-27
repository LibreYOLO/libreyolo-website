---
title: utilitas libreyolo
seo_title: referensi perintah utilitas CLI libreyolo
description: >-
  Perintah kecil LibreYOLO: version, checks, models, formats, cfg, info,
  metadata, enroll dan compare, masing-masing dengan argumen dan nilai
  bawaannya.
lead: >-
  Sembilan perintah yang melaporkan atau memeriksa, bukan menghitung. Semuanya
  mencetak fakta tentang lingkungan, inventaris model dan format, nilai bawaan
  yang berlaku, detail checkpoint, serta membangun dan menelusuri galeri wajah.
keywords:
  - libreyolo version
  - libreyolo checks
  - cara melihat daftar model libreyolo
  - format export libreyolo
  - cek metadata checkpoint yolo
  - galeri wajah libreyolo enroll
last_verified: 1.5.0
meta:
  - label: Perintah
    value: 'version, checks, models, formats, cfg, info, metadata, enroll, compare'
    mono: true
  - label: Keluaran
    value: >-
      stdout, dalam bentuk teks atau dengan json=true sebagai satu objek yang
      membawa schema_version
snippets:
  examples:
    - label: Lingkungan
      language: bash
      code: |
        libreyolo version
        libreyolo checks
    - label: Apa yang tersedia
      language: bash
      code: |
        libreyolo models
        libreyolo formats family=yolo9 task=detect
    - label: Memeriksa checkpoint
      language: bash
      code: |
        libreyolo info model=LibreYOLO9s.pt
        libreyolo metadata path=weights/LibreYOLO9s.pt
source_hash: 7b5b53c46df00c06
---

## Sinopsis

```bash
libreyolo <command> [key=value ...]
```

Argumen berupa pasangan `key=value`, dan bentuk POSIX juga bisa dipakai, jadi
`model=x` dan `--model x` adalah argumen yang sama. Semua perintah di halaman
ini menulis hasilnya ke stdout dan menerima `json=true` serta `quiet=true`.

Perintah root punya satu flag miliknya sendiri, `libreyolo --version`, yang
mencetak string versi lalu keluar. Keluarannya lebih ringkas daripada perintah
`version` di bawah.

## version

Mencetak versi LibreYOLO beserta versi Python, torch dan CUDA yang dipakainya.

```bash
libreyolo version
```

| Argumen | Bawaan | Arti |
|---|---|---|
| `json` | `false` | Keluaran JSON ke stdout |
| `quiet` | `false` | Menyembunyikan stderr |

## checks

Mencetak informasi lingkungan secara lebih rinci: Python, torch, CUDA, cuDNN,
setiap GPU yang terdeteksi lengkap dengan nama dan memorinya, serta versi
terpasang dari tiap paket opsional yang dipakai jalur ekspor.

```bash
libreyolo checks
```

| Argumen | Bawaan | Arti |
|---|---|---|
| `json` | `false` | Keluaran JSON ke stdout |
| `quiet` | `false` | Menyembunyikan stderr |

Daftar paketnya mencakup `onnx`, `onnxruntime`, `tensorrt`, `openvino`,
`paddlepaddle`, `x2paddle`, `mnn`, `ncnn`, `onnx2tf`, `ai-edge-litert`,
`transformers` dan `scipy`. Paket yang belum terpasang tetap dilaporkan sebagai
belum terpasang, bukan dihilangkan dari daftar, sehingga ekspor yang gagal bisa
dilacak ke dependensi yang hilang hanya dari satu perintah ini.

## models

Menampilkan setiap family model beserta task, ukuran, nama CLI yang mengarah ke
checkpoint-nya, dan resolusi input tiap ukuran.

```bash
libreyolo models
```

| Argumen | Bawaan | Arti |
|---|---|---|
| `json` | `false` | Keluaran JSON ke stdout |
| `quiet` | `false` | Menyembunyikan stderr |

Family yang dependensi opsionalnya belum terpasang tetap didaftar sebagai tidak
tersedia, disertai baris `pip install` yang akan membuatnya tersedia. Nama CLI
adalah bentuk singkat yang diterima `model=`: `yolox-s` mengarah ke
`LibreYOLOXs.pt`, dan task selain deteksi membawa akhiran task-nya.

## formats

Menampilkan format ekspor yang bisa dihasilkan lingkungan terpasang, lengkap
dengan ekstensi berkas tiap format dan apakah format itu mendukung FP16 dan
INT8.

```bash
libreyolo formats [family=<family>] [task=<task>]
```

| Argumen | Bawaan | Arti |
|---|---|---|
| `family` | | Menampilkan tier untuk satu family model. `model=` diterima sebagai opsi yang sama |
| `task` | | Task model kanonis. Task bawaan family bila tidak diisi |
| `json` | `false` | Keluaran JSON ke stdout |
| `quiet` | `false` | Menyembunyikan stderr |

Tanpa `family`, keluarannya hanya inventaris format. Dengan `family`, tiap
format mendapat tier dukungan untuk family dan task tersebut, alasan di balik
tier itu, dan batasan apa pun yang melekat padanya. Family yang tidak dikenal,
atau task yang tidak didukung family tersebut, dihitung sebagai kesalahan
penggunaan.

Alias format muncul di sebelah nama kanonisnya: `engine` untuk `tensorrt`,
`litert` untuk `tflite`.

## cfg

Mencetak konfigurasi bawaan yang berlaku: nilai bawaan pelatihan, nilai bawaan
validasi, nilai bawaan prediksi, dan override per family.

```bash
libreyolo cfg
```

| Argumen | Bawaan | Arti |
|---|---|---|
| `json` | `false` | Keluaran JSON ke stdout |
| `quiet` | `false` | Menyembunyikan stderr |

Nilainya dibaca dari dataclass konfigurasi, bukan dari salinan, jadi inilah
acuan resmi untuk apa yang akan dipakai sebuah proses pelatihan ketika Anda
tidak memberikan argumen. `family_overrides` adalah bagian yang menjawab kenapa
sebuah family dilatih dengan setelan yang tidak Anda minta. Lihat
[`libreyolo train`](/docs/cli/train) untuk cara override itu diterapkan.

## info

Memuat model di CPU dan melaporkan family, ukuran, jumlah parameter, kelas,
serta tier ekspor untuk tiap format.

```bash
libreyolo info model=<name|path>
```

| Argumen | Bawaan | Arti |
|---|---|---|
| `model` | | Nama model atau path ke bobot. Wajib |
| `detailed` | `false` | Sertakan detail per parameter |
| `json` | `false` | Keluaran JSON ke stdout |
| `quiet` | `false` | Menyembunyikan stderr |

## metadata

Membaca metadata checkpoint tanpa membangun model, lalu memvalidasinya terhadap
skema checkpoint LibreYOLO.

```bash
libreyolo metadata path=<checkpoint.pt>
```

| Argumen | Bawaan | Arti |
|---|---|---|
| `path` | | Path ke checkpoint `.pt`. Wajib |
| `json` | `false` | Keluaran JSON ke stdout |
| `quiet` | `false` | Menyembunyikan stderr |

Entri besar yang memuat tensor diringkas, bukan dicetak utuh, sehingga
keluarannya tetap terbaca pada checkpoint pelatihan yang lengkap. Checkpoint
yang tidak ada akan keluar dengan `checkpoint_not_found`, dan checkpoint yang
metadatanya gagal divalidasi akan mencetak error lalu keluar dengan kode `1`.

## enroll

Membangun galeri wajah dari struktur satu folder per orang, agar prediksi
berikutnya bisa menyebutkan nama wajah yang ditemukannya.

```bash
libreyolo enroll model=<embedder> source=<people-dir> gallery=<gallery.npz>
```

| Argumen | Bawaan | Arti |
|---|---|---|
| `model` | | Model face embedding, path atau nama. Wajib |
| `source` | | Struktur satu folder per orang, `source/<identity>/*.jpg`. Wajib |
| `gallery` | | Berkas galeri keluaran `.npz`. Ditambahkan langsung ke berkas itu bila sudah ada. Wajib |
| `face_detector` | | Detektor wajah: berkas YuNet `.onnx` atau detektor LibreYOLO. Detektor bawaan family bila tidak diisi |
| `device` | `auto` | Perangkat: `0`, `cpu`, `mps`, `auto` |
| `json` | `false` | Keluaran JSON ke stdout |
| `quiet` | `false` | Menyembunyikan stderr |

```bash
# people/ berisi satu folder per identitas; nama folder menjadi identitasnya.
libreyolo enroll model=librefacerec-l.onnx source=people/ gallery=people.npz
```

Nama subfolder adalah identitasnya. Gambar referensi yang wajahnya tidak
terdeteksi akan dilewati dengan satu baris pesan di stderr dan sisanya tetap
diproses; source tanpa subfolder identitas, atau yang sama sekali tidak
menemukan wajah, dihitung sebagai error.

Berikan berkas hasilnya ke
[`libreyolo predict`](/docs/cli/predict) sebagai `gallery=people.npz` supaya
setiap deteksi membawa identitas dan skor kecocokan.

## compare

Melaporkan kemiripan kosinus antara dua gambar wajah dan apakah nilainya
melewati ambang batas identitas yang sama.

```bash
libreyolo compare model=<embedder> source=<a.jpg> source2=<b.jpg>
```

| Argumen | Bawaan | Arti |
|---|---|---|
| `model` | | Model face embedding, path atau nama. Wajib |
| `source` | | Gambar pertama. Wajib |
| `source2` | | Gambar kedua sebagai pembanding. Wajib |
| `face_detector` | | Detektor wajah: berkas YuNet `.onnx` atau detektor LibreYOLO |
| `threshold` | `0.4` | Ambang batas kemiripan kosinus untuk keputusan identitas yang sama |
| `device` | `auto` | Perangkat: `0`, `cpu`, `mps`, `auto` |
| `json` | `false` | Keluaran JSON ke stdout |
| `quiet` | `false` | Menyembunyikan stderr |

```bash
libreyolo compare model=librefacerec-l.onnx source=a.jpg source2=b.jpg
```

`libreyolo verify` terdaftar sebagai nama kedua untuk perintah ini dan menerima
argumen yang sama.

Baik `compare` maupun `enroll` membutuhkan model yang task-nya adalah face
embedding. Model lain akan keluar dengan `config_unsupported`. Path gambar lokal
maupun URL `http` atau `https` sama-sama diterima sebagai source.

## Contoh

<code-tabs name="examples" />

## Catatan

stdout membawa hasilnya; progres dan peringatan dikirim ke stderr. `json=true`
mencetak satu objek berisi `schema_version`, dan bentuk itulah yang sebaiknya
dibaca dari sebuah skrip. Keluaran teks adalah bentuk bawaannya dan ditujukan
untuk dibaca manusia.

Kode keluar mengikuti peta yang sama dengan bagian CLI lainnya: `0` bila
berhasil, `2` untuk kesalahan penggunaan atau konfigurasi, `3` bila source tidak
ditemukan, `4` bila model atau checkpoint tidak bisa dimuat, dan `1` untuk
kegagalan runtime lainnya.

Terkait: [`libreyolo doctor`](/docs/cli/doctor), perintah pemeriksaan di sisi
dataset, dan [`libreyolo profile`](/docs/cli/profile), perintah untuk sisi
performa.
