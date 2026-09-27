---
title: libreyolo ui
seo_title: referensi perintah libreyolo ui
description: >-
  Menjalankan UI web inferensi lokal: alamat bind, perilaku port, pemilihan
  perangkat, dan cara perintah ini berakhir.
lead: >-
  Menjalankan server web lokal yang menerima gambar hasil drag-and-drop atau
  tempel, menjalankan model pilihan pada gambar tersebut, dan menampilkan
  hasilnya di browser.
keywords:
  - libreyolo ui cli
  - web ui libreyolo
  - inference lokal di browser
  - drag and drop deteksi objek
  - port libreyolo ui
last_verified: 1.5.0
meta:
  - label: Perintah
    value: libreyolo ui
    mono: true
  - label: Keluaran
    value: 'Sebuah URL server di stdout, lalu proses tetap berjalan di foreground'
snippets:
  examples:
    - label: Dasar
      language: bash
      code: |
        libreyolo ui
    - label: 'Port tetap, tanpa browser'
      language: bash
      code: |
        libreyolo ui port=9000 no_browser=true
    - label: 'Di CPU, keluaran terbaca mesin'
      language: bash
      code: |
        libreyolo ui device=cpu json=true
source_hash: b0eebd33fd0f463b
---

## Sinopsis

```bash
libreyolo ui [key=value ...]
```

Argumen berbentuk pasangan `key=value`, dan bentuk POSIX juga berlaku, jadi
`port=9000` dan `--port 9000` adalah argumen yang sama.

## Argumen

| Argumen | Default | Arti |
|---|---|---|
| `host` | `127.0.0.1` | Host atau antarmuka untuk bind |
| `port` | `8000` | Port untuk bind. Naik ke port bebas berikutnya jika sudah dipakai |
| `device` | `auto` | Perangkat: `0`, `cpu`, `mps`, `auto` |
| `no_browser` | `false` | Jangan buka browser secara otomatis |
| `json` | `false` | Keluaran JSON ke stdout |
| `quiet` | `false` | Sembunyikan keluaran stderr |
| `verbose` | `false` | Keluaran stderr yang lebih rinci |

## Contoh

<code-tabs name="examples" />

## Catatan

Secara default server melakukan bind ke loopback, jadi UI hanya bisa diakses
dari mesin ini.

Jika port yang diminta sedang dipakai, perintah ini mencoba port berikutnya dan
terus naik sampai dua puluh port di atas permintaan. Kalau kedua puluh port itu
gagal semua, perintah berhenti dengan `io_error` beserta saran untuk memakai
port lain. URL yang dicetak di stdout memakai port yang benar-benar dipakai,
jadi baca URL tersebut, jangan berasumsi port yang diminta yang terpakai.

Kecuali `no_browser=true`, tab browser terbuka di URL itu tak lama setelah bind.

Perintah ini lalu melayani di foreground sampai Ctrl+C, yang mematikan server
dengan bersih. Tidak ada mode detached; pindahkan ke background lewat shell jika
ingin terminal kembali bebas.

`json=true` mencetak URL dan perangkat sebagai satu objek dengan
`schema_version` sebelum server mulai, dan begitulah sebuah script mengetahui
port yang dipakai.

Terkait: [`libreyolo label`](/docs/cli/label) untuk menggambar bounding box dan
menyimpan label, [`libreyolo monitor`](/docs/cli/monitor) untuk memantau proses
pelatihan. Keduanya server web lokal dengan perilaku port dan browser yang sama.
