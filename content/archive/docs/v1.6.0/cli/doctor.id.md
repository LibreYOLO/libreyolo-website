---
title: libreyolo doctor
seo_title: referensi perintah libreyolo doctor
description: >-
  Memeriksa dataset deteksi sebelum pelatihan: argumen beserta nilai bawaannya,
  family pemeriksaan yang bisa dilewati atau dipilih, dan exit code yang bisa
  dipakai CI sebagai gerbang.
lead: >-
  Menjalankan serangkaian pemeriksaan kesehatan pada dataset deteksi dan
  melaporkan apa saja yang akan mengganggu proses pelatihan: berkas yang hilang,
  label rusak, gambar korup, kebocoran antar split dan ketimpangan kelas.
keywords:
  - libreyolo doctor cli
  - cek kesehatan dataset yolo
  - validasi dataset object detection
  - cek data leakage antar split
  - libreyolo doctor strict
last_verified: 1.5.0
meta:
  - label: Perintah
    value: libreyolo doctor
    mono: true
  - label: Wajib
    value: data
    mono: true
  - label: Output
    value: Laporan temuan di stdout. Keluar dengan kode 1 bila ditemukan error
snippets:
  examples:
    - label: Dasar
      language: bash
      code: >
        # download=true membuat coco8.yaml bawaan mengunduh gambarnya bila belum
        ada.

        libreyolo doctor coco8.yaml download=true
    - label: 'Pass cepat, tanpa decoding gambar'
      language: bash
      code: |
        libreyolo doctor coco8.yaml download=true fast=true
    - label: Gerbang CI pada pemeriksaan terpilih
      language: bash
      code: |
        libreyolo doctor coco8.yaml download=true strict=true json=true \
          only=labels,files,config
source_hash: 79e0ef471d567ea3
---

## Sinopsis

```bash
libreyolo doctor <data.yaml> [key=value ...]
```

Dataset bersifat posisional, dan `data=<path>` diterima sebagai alternatif.
Memberikan keduanya dengan nilai berbeda akan keluar dengan `config_conflict`.
Selebihnya berupa pasangan `key=value`, dan bentuk POSIX juga berfungsi,
sehingga `imgsz=1024` dan `--imgsz 1024` adalah argumen yang sama.

## Argumen

| Argumen | Default | Arti |
|---|---|---|
| `data` | | Posisional. YAML dataset dalam format deteksi YOLO, mis. `coco8.yaml`. Wajib |
| `imgsz` | `640` | Ukuran gambar pelatihan yang dipakai untuk pemeriksaan berbasis piksel seperti objek sangat kecil |
| `fast` | `false` | Melewati decoding gambar, sehingga pemeriksaan korupsi, duplikat dan kebocoran ikut hilang |
| `skip` | | Id pemeriksaan atau family yang dilewati, dipisah koma, mis. `images,labels.tiny_object` |
| `only` | | Id pemeriksaan atau family yang dijalankan secara eksklusif, dipisah koma |
| `strict` | `false` | Peringatan ikut membuat exit code gagal, untuk gerbang CI |
| `download` | `false` | Mengizinkan pengunduhan dataset berbasis URL bila belum ada. Tidak pernah skrip |
| `json` | `false` | Output JSON ke stdout |
| `quiet` | `false` | Menekan stderr |
| `help_json` | `false` | Mencetak skema perintah sebagai JSON lalu keluar |

### Family pemeriksaan

`skip` dan `only` menerima id pemeriksaan lengkap maupun awalan family, sehingga
`images` memilih semua pemeriksaan `images.*`.

| Family | Cakupan |
|---|---|
| `config` | YAML dataset itu sendiri: `names` yang hilang, `nc` terhadap `names`, split yang hilang, `path` yang tidak bisa diselesaikan, nama kelas duplikat |
| `files` | Pasangan gambar dan label: label yang hilang, gambar yang hilang, label yatim, ekstensi yang tidak didukung, benturan huruf besar kecil |
| `labels` | Isi label: sintaks, baris poligon, id kelas di luar rentang, koordinat di luar rentang, box degenerat, objek sangat kecil, box raksasa, rasio aspek ekstrem, box duplikat, gambar yang terlalu padat, berkas identik |
| `images` | Data piksel: berkas korup, orientasi EXIF, mode warna tidak lazim, dimensi sangat kecil atau ekstrem, gambar seragam, duplikat persis dan nyaris persis |
| `splits` | Kebocoran antar split, persis maupun nyaris persis |
| `balance` | Distribusi kelas: kelas tanpa instance atau dengan sedikit instance, ketimpangan, cakupan split, rasio background, kemiringan antar split |

## Contoh

<code-tabs name="examples" />

## Catatan

### Exit code

`0` bila tidak ada error yang ditemukan, `1` bila ada temuan yang berupa error.
Dengan `strict=true`, peringatan ikut menaikkan exit code menjadi `1`, dan
itulah setelan yang dibutuhkan gerbang CI.

Masalah pemakaian punya kode sendiri: `2` untuk id pemeriksaan atau family yang
tidak dikenal di `skip` atau `only`, `3` bila dataset tidak ditemukan, dan `3`
bila bentuk dataset bukan deteksi.

### Seleksi diselesaikan sebelum pemindaian

`skip` dan `only` diselesaikan terhadap registry pemeriksaan sebelum apa pun
dibaca dari disk, sehingga salah ketik langsung gagal, bukan setelah satu pass
gambar yang panjang. Selektor yang tidak cocok dengan apa pun dianggap error,
dan pesannya menampilkan daftar family yang dikenal.

Bila kombinasi `skip`, `only` dan `fast` menyisakan nol pemeriksaan untuk
dijalankan, itu juga dianggap error, bukan lolos diam-diam.

### Pengunduhan

Dataset tidak diambil kecuali `download=true`, dan yang dilakukan hanyalah
pengunduhan lewat URL. Skrip unduhan Python yang tertanam di dalam YAML dataset
tidak pernah dijalankan oleh perintah ini, apa pun nilai flag yang diberikan.

### Cakupan

Pemeriksaan ini ditulis untuk dataset deteksi. Dataset yang labelnya berbentuk
pose, segmentasi atau bounding box berorientasi akan dikenali dan ditolak dengan
`data_invalid`, bukan dinilai dengan aturan yang keliru.

### Output

Laporan untuk dibaca manusia dikirim ke stdout, dan `json=true` menggantinya
dengan objek terstruktur yang memuat hitungan ringkasan, statistik dataset,
setiap temuan, dan daftar pemeriksaan yang dilewati.

Terkait: [`libreyolo train`](/docs/cli/train), proses yang seharusnya didahului
oleh perintah ini.
