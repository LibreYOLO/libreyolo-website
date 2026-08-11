---
title: Impor bobot yang ada
seo_title: Memuat bobot upstream di LibreYOLO
description: >-
  Arahkan LibreYOLO ke checkpoint dari project upstream. Konversi otomatis
  membungkusnya ulang saat pemuatan dengan mempertahankan jumlah dan nama kelas.
lead: >-
  LibreYOLO melakukan port family model dari project upstream, sehingga
  checkpoint rilisnya hampir dapat langsung dimuat. Yang tidak ada adalah
  metadata. Konversi otomatis menyediakannya saat pemuatan.
keywords:
  - konversi bobot libreyolo
  - memuat checkpoint upstream
  - migrasi libreyolo
  - konversi pth ke libreyolo
  - autoconversion
last_verified: 1.5.0
meta:
  - label: Entry point
    value: LibreYOLO("path/to/upstream.pth")
    mono: true
  - label: Ditulis di samping sumber sebagai
    value: '<source>-<Prefix><size>[-task].pt'
    mono: true
  - label: Konverter berbasis script
    value: weights/ dalam repository
    mono: true
snippets:
  convert:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Ganti dengan path checkpoint yang sudah tersedia. Layout upstream yang

        # dikenali dikonversi saat pemuatan, ditulis di samping sumber,

        # lalu dimuat.

        model = LibreYOLO("path/to/upstream-checkpoint.pth")


        # Jumlah dan nama kelas berasal dari tensor serta metadata file sendiri,

        # sehingga hasil fine-tuning mempertahankan kumpulan labelnya, bukan
        COCO.

        print(model.family, model.size, model.task, model.nb_classes)

        print(model.names)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=path/to/upstream-checkpoint.pth \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Periksa hasil
      language: bash
      code: |
        # File hasil konversi memenuhi skema yang sama seperti file terbitan.
        libreyolo metadata path=path/to/upstream-checkpoint-LibreYOLO9t.pt
source_hash: bf9d7c7d168fd2c0
---

Halaman ini membahas checkpoint dari project lain. Jika memindahkan kode sendiri
dari LibreYOLO lama, lihat [upgrade ke 1.5.0](/docs/upgrade).

## Yang terjadi ketika file asing dimuat

`LibreYOLO()` terlebih dahulu memuat setiap file bobot melalui jalur restricted
weights-only. Jika hasil memiliki metadata LibreYOLO lengkap, file langsung
digunakan. Jika tidak, file dikirim ke konverter otomatis sebelum proses lain.
Jika pemuatan restricted gagal sepenuhnya, yang terjadi ketika checkpoint
memiliki objek pihak ketiga yang di-pickle, konverter otomatis dicoba dengan
loader yang menetralkan objek tersebut.

Konversi otomatis melakukan empat hal. Proses ini membuka dictionary tensor dari
layout yang digunakan project upstream. Lalu, setiap family terdaftar ditanya
apakah mengenali key tersebut, dengan remapping nama ketika penamaan upstream
berbeda dari port LibreYOLO. Pemenang dibungkus dalam checkpoint yang memenuhi
skema metadata v1.0, dengan ukuran, task, dan jumlah kelas dibaca dari tensor.
Terakhir, hasil ditulis di samping file sumber dan dimuat.

<code-tabs name="convert" />

Konversi tidak berlangsung diam-diam. File hasil konversi dicatat bersama
family, nama sumber, nama output, dan jumlah kelas hasil, sehingga log run
merekam yang dimuat secara tepat.

## Layout yang dibuka

Checkpoint upstream menempatkan bobot di sejumlah lokasi konvensional, dan
konverter mencobanya berurutan hingga menemukan tensor: block EMA di bawah
`ema.module` atau `ema` flat, `ema_state_dict` setelah prefix `module.` dihapus,
lalu `params_ema`, `params`, `ema_net`, `net`, `model`, `state_dict`, dan
akhirnya objek itu sendiri. Mencoba beberapa kandidat berarti block `ema` yang
hanya berisi counter tidak menutupi bobot asli di bawahnya.

Prefix wrapper juga dihapus: `module.` dari pelatihan terdistribusi,
`_orig_mod.` dari model terkompilasi, dan nesting `model.model.` yang ditambahkan
beberapa redistribusi.

## Yang dibaca dan sumbernya

Ukuran, task, dan jumlah kelas berasal dari tensor, bukan nama file. Karena itu,
checkpoint hasil fine-tuning dikonversi dengan jumlah kelas sendiri, bukan
default arsitektur. Nama kelas diambil dari metadata checkpoint jika ada, dari
block `args` atau `hyper_parameters` jika berada di sana, lalu dipangkas hingga
jumlah kelas terdeteksi agar hasil fine-tuning yang mempertahankan kumpulan label
dasar tidak membawa indeks yang sudah tidak dimiliki head.

Task padat ditangani secara eksplisit, bukan diberi label buatan. Checkpoint depth
mendapat satu kelas bernama `depth`, checkpoint restore satu kelas bernama
`image`. Checkpoint pose harus menghasilkan jumlah keypoint, baik dari tensor
maupun family; jika keduanya tidak menghasilkan nilai, konversi ditolak
alih-alih menulis file tidak lengkap.

RF-DETR memiliki recognizer sendiri karena deteksi ukuran memerlukan seluruh
checkpoint dan head-nya memiliki 91 output ketika LibreYOLO menggunakan
konvensi COCO 80 kelas. Checkpoint dinormalisasi menjadi 80 kelas ketika
memiliki tepat 80 nama, menyatakan jumlah kelas 80, menamai COCO sebagai
dataset, atau sama sekali tidak memiliki metadata kelas maupun dataset. Model
90 kelas asli yang diidentifikasi berdasarkan nama, jumlah eksplisit selain 80,
atau petunjuk dataset non-COCO dipertahankan sebagaimana adanya.

## Lokasi file hasil konversi

Output ditulis di samping sumber dengan nama berdasarkan sumber:

```text
<source stem>-<FilenamePrefix><size>[-<task suffix>].pt
```

Detektor YOLOv9 tiny yang disimpan sebagai `upstream-checkpoint.pth` menjadi
`upstream-checkpoint-LibreYOLO9t.pt`. Penamaan berdasarkan sumber berarti dua
hasil fine-tuning dari family dan ukuran sama dalam satu direktori tidak saling
menimpa dan tidak bertabrakan dengan checkpoint resmi. File ditulis ulang pada
setiap pemuatan agar tidak pernah tertinggal dari sumber. Jika direktori
read-only, file hasil konversi masuk ke direktori sementara privat baru dan log
menyatakan lokasinya.

Setelah itu, file menjadi checkpoint LibreYOLO biasa: dimuat melalui jalur
metadata, dan `libreyolo metadata` melaporkannya valid.

## Kasus yang memerlukan penanganan manual

Dua family berada di luar recognizer generik. Family gaze dikecualikan
sepenuhnya: hanya mendukung inferensi dan bobot rilisnya memiliki batasan
distribusi ulang. RF-DETR dikecualikan karena memiliki recognizer khusus yang
dijelaskan di atas, dan recognizer itulah yang menanganinya.

Checkpoint PIDNet upstream mentah ditolak, dengan error yang menunjuk ke
`weights/convert_pidnet_weights.py`. Script tersebut menulis metadata semantic
Cityscapes yang diperlukan checkpoint.

D-FINE dan DEIM berbagi key arsitektur yang sama, sehingga tensor saja tidak
dapat membedakannya. Ketika keduanya mengklaim file dan tidak ada sibling family
dengan marker pembeda, nama file menentukan: nama berbentuk
`dfine_hgnetv2_n_coco.pth` atau `deim_hgnetv2_n_coco.pth` menyelesaikannya,
sedangkan nama tanpa petunjuk ditolak dengan penjelasan, bukan ditebak. Membuat
`LibreDFINE` atau `LibreDEIM` secara langsung juga menyelesaikannya.

Jika beberapa family secara sah mengklaim satu file, subclass mengalahkan base
class yang disempurnakannya, dan urutan registry menentukan sisanya karena
urutan tersebut mengenkode spesifisitas pemeriksaan setiap family. Nama file
hanya diperiksa untuk tie D-FINE dan DEIM, sehingga nama file tidak pernah dapat
mempromosikan kecocokan luas di atas kecocokan tepat.

## Konverter berbasis script

Repository memiliki script konversi per family di bawah `weights/`, ditambah
helper bersama untuk plumbing berulang. Script menjadi jalur bagi file yang
ditolak runtime, untuk menghasilkan checkpoint sebelum waktu pemuatan, serta
bagi family yang metadata-nya harus diberikan, bukan disimpulkan dari tensor.

Script tersebut merupakan bagian repository, bukan package terinstal, sehingga
penggunaannya memerlukan clone:

```bash
git clone https://github.com/LibreYOLO/libreyolo.git
cd libreyolo
python weights/convert_pidnet_weights.py --help
```

Setiap script menulis checkpoint yang memenuhi skema v1.0, yaitu standar sama
yang dipenuhi konversi otomatis dan bobot terbitan. Lihat
[checkpoint dan bobot](/docs/weights) untuk isi skema tersebut.
