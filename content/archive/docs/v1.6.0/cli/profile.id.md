---
title: libreyolo profile
seo_title: Referensi perintah libreyolo profile
description: >-
  Mengukur kecepatan pelatihan dan inferensi lalu membaca hasilnya: setiap
  subperintah profile, argumen dan nilai bawaannya, serta apa yang dilaporkan
  tiap sudut pandang.
lead: >-
  Grup perintah yang mengukur ke mana waktu habis dalam satu step pelatihan atau
  satu panggilan inferensi, menulis profil yang mandiri, lalu membaca profil itu
  kembali lewat beberapa sudut pandang.
keywords:
  - libreyolo profile cli
  - profiling training yolo
  - latensi inference yolo
  - profiling kernel gpu
  - libreyolo profile compare
last_verified: 1.5.0
meta:
  - label: Perintah
    value: libreyolo profile
    mono: true
  - label: Output
    value: profile.json and profile_trace.json under runs/profile
    mono: true
snippets:
  examples:
    - label: Mengukur inferensi
      language: bash
      code: |
        # Tanpa argumen source, gambar contoh bawaan yang dipakai.
        libreyolo profile infer --device cpu --warmup 5 --runs 20
    - label: Membaca kesimpulannya
      language: bash
      code: |
        libreyolo profile summary runs/profile/infer/profile.json
    - label: Membandingkan dua pengukuran
      language: bash
      code: >
        libreyolo profile infer --device cpu --warmup 5 --runs 20 --project
        runs/profile/a

        libreyolo profile infer --device cpu --warmup 5 --runs 20 --batch 4
        --project runs/profile/b


        libreyolo profile compare runs/profile/a/infer/profile.json \
          runs/profile/b/infer/profile.json
source_hash: b967e869fd9ba418
---

## Sinopsis

```bash
libreyolo profile <subcommand> [<positional>] [--flag value ...]
```

Grup ini tidak menerima argumen `key=value`. Subperintahnya memakai argumen
posisional dan flag POSIX, jadi bentuknya `--weights LibreYOLO9t.pt`, bukan
`weights=LibreYOLO9t.pt`. Menjalankan `libreyolo profile` tanpa subperintah akan
menampilkan daftarnya.

Dua subperintah melakukan pengukuran dan menulis profil; sisanya membaca profil.
`run` dan `infer` sama-sama menghasilkan `profile.json` mandiri yang sama, jadi
setiap subperintah pembaca bisa dipakai pada keduanya.

## profile run

Menjalankan pelatihan singkat yang diprofil dan menulis sebuah profil.

```bash
libreyolo profile run <data> [--flag value ...]
```

| Argumen | Default | Arti |
|---|---|---|
| `data` | | Posisional. YAML atau nama dataset, misalnya `coco128`. Wajib |
| `--weights` | `LibreYOLO9t.pt` | Berkas bobot atau nama model |
| `--size` | `t` | Varian ukuran model |
| `--batch` | `16` | Micro-batch. `-1` menyesuaikan otomatis sekitar 70% VRAM |
| `--imgsz` | `640` | Ukuran gambar pelatihan |
| `--workers` | `8` | Worker dataloader |
| `--amp` | `true` | Pakai jalur AMP milik family. `--no-amp` mematikannya |
| `--steps` | `20` | Step yang diprofil, artinya yang diukur |
| `--warmup` | `5` | Step warmup sebelum pengukuran |
| `--repeat` | `1` | Ulangi N kali untuk mendapat rata-rata dan simpangan baku |
| `--device` | `0` | Perangkat |
| `--project` | `runs/profile` | Root direktori output |
| `--json` | `false` | Output JSON ke stdout |

Jendela pengukurannya adalah `--warmup` ditambah `--steps` iterasi. Dataset yang
terlalu kecil untuk mengisinya tidak menghasilkan profil dan perintah keluar
dengan kode `3`, sambil menyebut tiga jalan keluarnya: dataset yang lebih besar,
step yang lebih sedikit, atau batch yang lebih kecil.

`--repeat` di atas 1 menulis `runs/profile/profile_repeat.json` teragregasi yang
metrik skalarnya dirata-ratakan lintas percobaan, sementara daftar kernelnya
diambil dari percobaan terakhir. Ini juga prasyarat untuk kesimpulan signifikansi
di `compare`: satu kali run tidak bisa memberikannya.

## profile infer

Memprofil jalur inferensi dan menulis sebuah profil.

```bash
libreyolo profile infer [<source>] [--flag value ...]
```

| Argumen | Default | Arti |
|---|---|---|
| `source` | | Posisional. Gambar atau direktori. Gambar contoh bawaan bila tidak diisi |
| `--weights` | `LibreYOLO9t.pt` | Berkas bobot atau nama model |
| `--size` | `t` | Varian ukuran model |
| `--batch` | `1` | Jumlah gambar per forward pass |
| `--imgsz` | `640` | Ukuran gambar masukan |
| `--half` | `false` | Forward dengan autocast, hanya CUDA. `--no-half` mematikannya |
| `--amp-dtype` | `float16` | dtype autocast CUDA: `float16` atau `bfloat16` |
| `--warmup` | `20` | Iterasi warmup sebelum pengukuran |
| `--runs` | `100` | Iterasi yang diukur |
| `--repeat` | `1` | Ulangi N kali untuk mendapat rata-rata dan simpangan baku |
| `--conf` | `0.25` | Ambang batas skor keyakinan (confidence), yang mengubah banyaknya kerja NMS |
| `--iou` | `0.45` | Ambang IoU untuk NMS |
| `--max-det` | `300` | Deteksi maksimum per gambar, yang mengubah banyaknya kerja NMS |
| `--device` | `0` | Perangkat |
| `--trace` | `true` | Hasilkan Chrome trace untuk menelusuri kernel dan op. `--no-trace` melewatinya |
| `--project` | `runs/profile` | Root direktori output |
| `--json` | `false` | Output JSON ke stdout |

Melaporkan latensi pada p50, p90 dan p99, throughput dalam gambar per detik,
serta pembagian tahap antara preprocess, forward dan postprocess. Ketiga argumen
ambang batas itu ada di sini karena semuanya menggeser angka postprocess.

## profile summary

```bash
libreyolo profile summary <trace> [--json]
```

| Argumen | Default | Arti |
|---|---|---|
| `trace` | | Posisional. Path ke sebuah `profile.json` atau `profile_trace.json`. Wajib |
| `--json` | `false` | Output JSON ke stdout |

Bacaan tingkat tingginya: waktu per step, throughput, utilisasi GPU, porsi Tensor
Core, puncak VRAM, overhead host, peluncuran kernel per step, kesimpulan
bottleneck beserta alasannya, komposisi kernel per kategori, dan kernel teratas
per step. Pada profil inferensi, ditampilkan juga persentil latensi dan pembagian
tahapnya.

Profil yang diambil saat terjadi VRAM thrash akan ditandai, karena utilisasi dan
throughput yang terukur di sana tidak bisa dipercaya.

## profile get

```bash
libreyolo profile get <trace> [<field>] [--json]
```

| Argumen | Default | Arti |
|---|---|---|
| `trace` | | Posisional. Path ke sebuah profil. Wajib |
| `field` | | Posisional. Nama metrik. Kosongkan untuk menampilkan daftar metrik yang tersedia |
| `--json` | `false` | Output JSON ke stdout |

Menampilkan satu metrik dan tidak lebih, untuk dipakai di loop skrip. Field yang
tidak dikenal keluar dengan kode `2` dan menunjuk ke bentuk daftarnya.

## profile phases

```bash
libreyolo profile phases <trace> [--json]
```

| Argumen | Default | Arti |
|---|---|---|
| `trace` | | Posisional. Path ke sebuah profil. Wajib |
| `--json` | `false` | Output JSON ke stdout |

Milidetik GPU, milidetik wall, jumlah kernel dan jumlah op per fase: forward,
backward, dataload, to_device, optimizer.

## profile kernels

```bash
libreyolo profile kernels <trace> [--flag value ...]
```

| Argumen | Default | Arti |
|---|---|---|
| `trace` | | Posisional. Path ke sebuah profil. Wajib |
| `--top` | `20` | Tampilkan N teratas berdasarkan waktu GPU |
| `--category` | | Saring berdasarkan potongan nama kategori: `gemm`, `layout`, `norm`, `elementwise` |
| `--grep` | | Saring dengan regular expression nama kernel |
| `--tensorcore` | `false` | Hanya kernel Tensor Core |
| `--sort` | `time` | `time`, `count` atau `name` |
| `--phase` | | Batasi ke satu fase: `forward`, `backward`, `dataload`, `to_device`, `optimizer` |
| `--json` | `false` | Output JSON ke stdout |

Dasar dari analisisnya: kernel GPU satu per satu dengan porsinya terhadap waktu
GPU, milidetik per step, jumlah pemanggilan per step dan kategorinya. `--phase`
yang tidak dikenal keluar dengan kode `2` dan menampilkan daftar fase yang
dimiliki profil itu.

## profile ops

```bash
libreyolo profile ops <trace> [--flag value ...]
```

| Argumen | Default | Arti |
|---|---|---|
| `trace` | | Posisional. Path ke sebuah profil. Wajib |
| `--top` | `20` | Tampilkan N teratas berdasarkan waktu CPU |
| `--phase` | | Batasi ke satu fase |
| `--json` | `false` | Output JSON ke stdout |

Sudut pandang framework, bukan sudut pandang perangkat: op `aten` dan autograd
yang diurutkan berdasarkan waktu CPU, dan di situlah biaya peluncuran dari host
terlihat.

## profile compare

```bash
libreyolo profile compare <before> <after> [--json]
```

| Argumen | Default | Arti |
|---|---|---|
| `before` | | Posisional. Profil baseline. Wajib |
| `after` | | Posisional. Profil baru. Wajib |
| `--json` | `false` | Output JSON ke stdout |

Membandingkan selisih throughput, milidetik per gambar, utilisasi GPU, overhead
host, peluncuran kernel per step dan kesimpulan bottleneck.

Penilaian signifikansi menuntut kedua sisi diukur dengan `--repeat` minimal 2.
Bila itu terpenuhi, sebuah selisih dihitung signifikan ketika melebihi dua kali
galat baku gabungan, dan outputnya menampilkan perbandingan yang dilakukannya.
Tanpa itu, barisnya menyatakan bahwa satu kali run tidak cukup untuk menopang
penilaian tersebut.

## profile what-if

```bash
libreyolo profile what-if <trace> [--flag value ...]
```

| Argumen | Default | Arti |
|---|---|---|
| `trace` | | Posisional. Path ke sebuah profil. Wajib |
| `--remove-category` | | Proyeksikan efek menghilangkan satu kategori kernel: `gemm`, `layout`, `norm`, `elementwise` |
| `--remove-launches` | | Proyeksikan efek menghilangkan N peluncuran kernel per step, misalnya hasil dari op fusion |
| `--json` | `false` | Output JSON ke stdout |

Memperkirakan apa yang akan didapat dari sebuah perubahan sebelum perubahan itu
ditulis. Salah satu dari dua opsi tersebut wajib diisi; bila tidak ada yang
diberikan, perintah keluar dengan kode `2`.

Proyeksinya mengikuti kesimpulan profil itu sendiri. Di bawah utilisasi GPU 80%,
penghematannya dimodelkan sebagai berkurangnya peluncuran dikali biaya host per
peluncuran yang terukur; di atasnya, sebagai berkurangnya kerja GPU. Hasilnya
membawa sebuah field peringatan, karena biaya per peluncuran hanyalah pendekatan
dan satu-satunya bukti adalah pengukuran kedua.

## Contoh

<code-tabs name="examples" />

## Catatan

Profiler mengukur dan melaporkan. Ia tidak mengubah apa pun: membaca
kesimpulannya, menyunting konfigurasi atau kodenya, menjalankan ulang, lalu
membandingkan, itulah siklus yang menjadi tujuan pembuatannya.

Nilai bawaan `--device` adalah `0`, yaitu perangkat CUDA 0. Memberikan
`--device cpu` akan mengukur di CPU dan menghasilkan profil yang tetap diterima
subperintah pembaca, tanpa detail kernel GPU.

Setiap subperintah mendukung `--json`, dan subperintah pembaca hanya mencetak ke
stdout, dan itulah yang membuat grup ini bisa dipakai dari skrip.

Kode keluar di sini adalah milik grup ini sendiri: `2` untuk berkas yang tidak
ada atau argumen yang tidak bisa diselesaikan, `3` ketika `run` tidak
menghasilkan profil, dan `1` ketika sebuah trace tidak bisa dianalisis.

Terkait: [`libreyolo train`](/docs/cli/train), yang argumennya biasanya menjadi
sasaran penyetelan saat sebuah profil pelatihan diambil.
