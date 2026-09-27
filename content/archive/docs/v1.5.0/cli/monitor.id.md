---
title: libreyolo monitor
seo_title: Referensi perintah libreyolo monitor
description: >-
  Menyajikan dashboard langsung untuk run pelatihan: argumen beserta nilai
  bawaannya, apa yang dibaca server dari disk, dan bagaimana satu server
  melayani banyak run.
lead: >-
  Menyajikan dashboard web untuk run pelatihan dengan membaca artefak yang
  ditulis sebuah run ke disk. Server tidak pernah menempel ke proses pelatihan,
  sehingga run yang sedang berjalan, yang sudah selesai, maupun yang gagal
  sama-sama tampil.
keywords:
  - libreyolo monitor cli
  - dashboard training yolo
  - pantau proses training
  - libreyolo monitor port
  - monitoring training model
last_verified: 1.5.0
meta:
  - label: Perintah
    value: libreyolo monitor
    mono: true
  - label: Output
    value: 'URL server di stdout, lalu proses tetap berjalan di foreground'
snippets:
  examples:
    - label: Dasar
      language: bash
      code: |
        # Memantau runs/ dan menampilkan setiap run di dalamnya.
        libreyolo monitor
    - label: Root runs yang berbeda
      language: bash
      code: |
        libreyolo monitor experiments/
    - label: 'Satu run, port tetap, tanpa browser'
      language: bash
      code: |
        libreyolo monitor runs/train/exp port=9100 no_browser=true
source_hash: 4aa178141d451728
---

## Sinopsis

```bash
libreyolo monitor [<run-dir|runs-root>] [key=value ...]
```

Direktori bersifat posisional. Selebihnya berupa pasangan `key=value`, dan
bentuk POSIX juga berlaku, jadi `port=9100` dan `--port 9100` adalah argumen
yang sama.

## Argumen

| Argumen | Default | Arti |
|---|---|---|
| `run_dir` | `runs` | Posisional. Root runs yang dipantau, atau satu direktori run yang langsung dibuka. Bagaimanapun juga, setiap run di bawah root tetap terdaftar |
| `host` | `127.0.0.1` | Host atau interface tempat server mengikat |
| `port` | `8420` | Port yang diikat. Naik ke port bebas berikutnya bila sudah terpakai |
| `no_browser` | `false` | Jangan membuka browser secara otomatis |
| `json` | `false` | Output JSON ke stdout |
| `quiet` | `false` | Sembunyikan stderr |
| `verbose` | `false` | Output stderr yang lebih rinci |

## Contoh

<code-tabs name="examples" />

## Catatan

### Satu server, banyak run

Server memantau sebuah root runs, bukan satu run saja, dan mengalamati setiap
run lewat URL, sehingga beberapa run di satu mesin berbagi satu port. Buka URL
root untuk melihat indeksnya, atau satu tab untuk tiap run; parameter `?run=`
pada setiap URL menandai run mana yang dimaksud.

Mengarahkan perintah ke satu direktori run membuat server berakar di direktori
induknya, sehingga run lain yang setingkat tetap muncul di indeks, sekaligus
menautkan langsung ke run yang disebutkan.

### Apa yang dibaca

Dashboard disusun dari berkas yang ditulis `libreyolo train`: `status.json`,
`metrics.jsonl`, `train.log`, dan gambar milik run tersebut. Tidak ada yang
dibaca dari proses pelatihan itu sendiri, jadi run yang sudah selesai, atau
yang mati, tampil persis seperti run yang masih berjalan.

### Prasyarat dan port

Minimal satu run harus sudah ada. Tanpa argumen dan tanpa direktori `runs/`,
perintah berhenti dengan `source_not_found`; hal yang sama terjadi bila
direktori yang diberikan tidak memuat run.

Port yang sudah terpakai digeser ke port berikutnya, sampai dua puluh port
setelah yang diminta. Bila kedua puluh port itu gagal semua, perintah berhenti
dengan `io_error`. URL yang dicetak ke stdout memakai port yang benar-benar
diikat.

Perintah berjalan di foreground sampai Ctrl+C. `json=true` mencetak URL, root
yang dipantau, dan jumlah run yang ditemukan, sebagai satu objek dengan
`schema_version`.

Terkait: [`libreyolo train`](/docs/cli/train), yang argumen `project` dan
`name`-nya menentukan ke mana direktori run tersebut ditaruh.
