---
title: libreyolo label
seo_title: Referensi perintah libreyolo label
description: >-
  Menjalankan alat anotasi bounding box lokal: argumen beserta nilai bawaannya,
  switch AI assist, dan apa yang terbuka saat mengikat ke antarmuka jaringan.
lead: >-
  Menjalankan alat web lokal untuk menggambar dan menyunting bounding box. Alat
  ini menulis berkas label format native LibreYOLO, jadi dataset yang dianotasi
  di sini langsung bisa dilatih tanpa langkah konversi.
keywords:
  - libreyolo label cli
  - tool anotasi bounding box
  - aplikasi labeling yolo
  - cara labeling dataset yolo
  - auto label cli
  - libreyolo label share
last_verified: 1.5.0
meta:
  - label: Perintah
    value: libreyolo label
    mono: true
  - label: Output
    value: URL server di stdout; label ditulis sebagai labels/*.txt di samping gambar
snippets:
  examples:
    - label: Dasar
      language: bash
      code: |
        # Membuka halaman utama proyek; pilih atau buat dataset di browser.
        libreyolo label
    - label: 'Manual saja, port tetap'
      language: bash
      code: |
        libreyolo label no_assist=true port=9200 no_browser=true
    - label: Mengajak rekan tim bergabung
      language: bash
      code: |
        libreyolo label share=true
source_hash: bddad245877793b1
---

## Sinopsis

```bash
libreyolo label [data=<dataset.yaml|folder>] [key=value ...]
```

Argumen berupa pasangan `key=value`, dan bentuk POSIX juga berlaku, sehingga
`port=9200` dan `--port 9200` adalah argumen yang sama.

## Argumen

| Argumen | Default | Arti |
|---|---|---|
| `data` | | Dataset YAML atau folder yang langsung dibuka. Mulai dari halaman utama proyek bila tidak diisi |
| `host` | `127.0.0.1` | Host atau antarmuka yang diikat |
| `port` | `8000` | Port yang diikat. Naik ke port bebas berikutnya bila sudah terpakai |
| `device` | `auto` | Perangkat untuk auto-label AI: `0`, `cpu`, `mps`, `auto` |
| `no_assist` | `false` | Menonaktifkan auto-label AI, menyisakan pelabelan manual |
| `no_browser` | `false` | Jangan buka browser otomatis |
| `share` | `false` | Mengikat `0.0.0.0` agar rekan tim di jaringan Anda bisa bergabung |
| `json` | `false` | Output JSON ke stdout |
| `quiet` | `false` | Menyembunyikan stderr |
| `verbose` | `false` | Output stderr yang rinci |

## Contoh

<code-tabs name="examples" />

## Catatan

### Apa yang ditulisnya

Bounding box disimpan sebagai berkas `labels/*.txt` format native LibreYOLO,
yaitu format yang dibaca `libreyolo train`, jadi tidak ada yang perlu dikonversi
setelahnya. Versi ini hanya menangani bounding box. Perubahan tersimpan saat
berpindah antar gambar.

### Membuka dataset

Tanpa `data`, alat ini mulai dari halaman utama proyek dan dataset dipilih atau
dibuat dari browser. Memberikan `data=path/to/data.yaml` langsung membuka
dataset tersebut, dan baris awal melaporkan jumlah gambar, jumlah kelas, serta
apakah dataset bisa ditulisi. Dataset yang hanya bisa dibaca tetap terbuka dan
menjelaskan kenapa isinya tidak bisa ditulisi.

### Berbagi akses, dan fungsi `host`

`share=true` mengikat alamat wildcard, sehingga mesin lain di jaringan Anda bisa
menjangkau alat ini, sementara aksi administratif, yaitu mengganti atau
menghapus proyek dan memulai komputasi, tetap tinggal di mesin ini.

Mengatur `host` ke antarmuka tertentu berdampak lain dan kurang aman: host
menjadi tidak terbedakan dari klien jaringan, sehingga setiap klien mendapat hak
administratif. Perintah ini mencetak peringatan di stderr bila hal itu
dilakukan. Lebih baik pakai `share=true`.

### Port dan penghentian

Port yang sudah terpakai berpindah ke port berikutnya, sampai dua puluh port di
atas yang diminta. Bila kedua puluhnya gagal, perintah berhenti dengan
`io_error`. URL yang dicetak ke stdout memakai port yang benar-benar diikat.
Dengan `share=true`, hasilnya juga memuat `lan_url`, yaitu alamat yang perlu
dibuka rekan tim.

Perintah ini melayani di foreground sampai Ctrl+C.

Terkait: [`libreyolo doctor`](/docs/cli/doctor) untuk memeriksa dataset yang
sudah dilabeli sebelum pelatihan, dan [`libreyolo train`](/docs/cli/train) untuk
melatih model dengannya.
