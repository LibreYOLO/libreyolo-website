---
title: Tier stabilitas
seo_title: Arti setiap tier dukungan LibreYOLO
description: >-
  Vocabulary tier yang digunakan LibreYOLO: tiga tier dukungan ekspor, empat
  tier API, enam kelompok cakupan, dan hal-hal yang tidak dijanjikan olehnya.
lead: >-
  LibreYOLO menggunakan kata tier untuk tiga hal terpisah: bukti di balik jalur
  ekspor, kontrak pemanggilan yang dijawab family model, dan kelompok cakupan
  tempat family terdaftar. Halaman ini mendefinisikan masing-masing dan
  menjelaskan hal yang tidak tersirat.
keywords:
  - tier dukungan libreyolo
  - validated available blocked
  - tier dukungan ekspor
  - kelompok cakupan libreyolo
  - g0 g1 g2 g3 g4
  - tier model
last_verified: 1.5.0
verification: >-
  Tier ekspor dari docs/adr/0011-export-support-tiers.md dan
  libreyolo/export/support.py; kelompok cakupan dan jumlah per family dari
  MODEL_GROUPS di libreyolo/models/registry.py; gate from-scratch dari
  libreyolo/models/base/model.py dan libreyolo/cli/commands/train.py; inventaris
  CLI dibaca dari libreyolo/models/inventory.py; tier API dari docstring package
  dan kontrak base.py di libreyolo/models/sam/, openvocab/, dan vlm/, semuanya
  pada v1.5.0. Label kelompok untuk pembaca (Flagship, Core, Supported,
  Inference only, Museum, Sibling tier) adalah vocabulary situs sendiri untuk
  kelompok yang sama, dari src/data/docs/registry.json.
snippets:
  usage:
    - label: Baca kedua klasifikasi untuk satu family
      language: python
      code: |
        from libreyolo.models.registry import GROUPS, group_of
        from libreyolo.export.support import get_support, validated_alternatives

        family = "yolo9"

        group = group_of(family)
        print(group, GROUPS[group])

        print(get_support(family, "detect", "onnx").tier)
        print(validated_alternatives(family, "detect"))
source_hash: de545894b0d125e4
---

## Tier dukungan ekspor

Tier ini menentukan apakah suatu pemanggilan berhasil. Tier berlaku pada triple
`(family, task, format)`, dan setiap kombinasi memiliki tepat satu tier.

| Tier | Arti | Yang terjadi pada `export()` |
|---|---|---|
| `validated` | Paritas numerik dicakup dalam CI atau run nightly yang terdokumentasi | Berjalan |
| `available` | Konversi telah diimplementasikan, tetapi bukti paritas runtime numerik belum dicatat | Berjalan |
| `blocked` | Tidak ada jalur yang didukung | Memunculkan `NotImplementedError` saat preflight beserta alasannya |

Validated dan available sama-sama berjalan tanpa konfirmasi atau peringatan
umum. Perbedaannya terletak pada bukti, bukan izin: entri validated didukung oleh
pengujian paritas dan memiliki release `since`, sedangkan entri available belum.
Sebagai contoh, konversi CoreML tanpa run prediksi macOS berstatus available,
bukan validated.

Kombinasi blocked gagal sebelum pemeriksaan dependency, pemuatan kalibrasi,
tracing, atau pembuatan artefak, sehingga tidak ada hasil parsial yang ditulis.

Setiap cell validated memiliki constraint yang menjelaskan konfigurasi sumber
angka paritas, biasanya canvas input tetap, batch 1, FP32, dan versi runtime
tertentu. Bacalah sebagai klaim mengenai konfigurasi tersebut, bukan mengenai
format secara umum. Aturan yang mengisi cell tanpa entri eksplisit tersedia di
halaman [matriks ekspor](/docs/reference/export-matrix).

<code-tabs name="usage" />

## Tier API

Tier ini menentukan bentuk suatu pemanggilan. Sebuah family berada tepat dalam
satu tier, yang dipilih berdasarkan kontrak pemanggilan, bukan arsitektur.

| Tier | Factory | Kontrak |
|---|---|---|
| Factory detektor | `LibreYOLO` | Satu forward tanpa prompt mengembalikan setiap objek yang ditemukan, dengan score terkalibrasi. Anggota mendaftarkan diri dengan mengenali checkpoint |
| Promptable segmentation | `LibreSAM` | Forward tidak bermakna tanpa prompt spasial atau konsep per gambar yang diberikan saat pemanggilan. Interaktif dan stateful: encode sekali, prompt berkali-kali |
| Deteksi open-vocabulary | `LibreOpenVocab` | Detektor diskriminatif yang dikondisikan oleh teks. List kelas adalah prompt yang ditetapkan oleh `set_classes` |
| Vision-language | `LibreVLM` | Model generatif yang dijalankan sebagai detektor. List kelas adalah prompt dan confidence merupakan placeholder |

Ketiga sibling tier sengaja tidak mendaftar ke factory detektor, sehingga
`LibreYOLO("some-alias")` tidak dapat menjangkaunya. Tier tersebut dimuat
berdasarkan alias ukuran dan mengunduh secara otomatis, bukan melalui sniffing
checkpoint.

Keempatnya mengembalikan `Results` yang sama, sehingga kode downstream tidak
berubah. Perbedaannya adalah metode mana yang bekerja: sibling tier memunculkan
`NotImplementedError` untuk `train()`, `val()`, dan `export()`, sementara tier
SAM dan open-vocabulary juga memunculkannya untuk `track()`. Setiap halaman tier
mencantumkan pengecualiannya sendiri.

## Kelompok cakupan

Klasifikasi ini menentukan family yang disertakan dalam pengujian lintas family,
dan paling mungkin ditemui pembaca pada halaman model. Setiap family terdaftar
masuk tepat ke satu kelompok, dan pengujian gagal ketika ada family terdaftar
yang tidak dimasukkan. `GROUPS` dalam `libreyolo/models/registry.py` adalah
sumber kolom Arti di bawah; `MODEL_GROUPS` dalam file yang sama menempatkan
setiap family, dan kolom Family menghitung penempatan tersebut secara langsung.
Kolom Label adalah nama lebih pendek yang digunakan situs untuk kelompok yang
sama pada header halaman model.

| Kelompok | Label | Family | Arti |
|---|---|---|---|
| `g0` | Flagship | 2 | Anchor flagship yang diwajibkan dalam cakupan fitur bersama |
| `g1` | Core | 10 | Kelompok cakupan detektor yang dapat dilatih |
| `g2` | Supported | 14 | Kelompok cakupan tambahan untuk family yang dapat dilatih |
| `g3` | Inference only | 35 | Family tanpa implementasi pelatihan |
| `g4` | Museum | 5 | Family historis dengan cakupan inferensi |
| `s` | Sibling tier | 21 | API sibling (SAM, open-vocab, VLM, zero-shot) yang dicakup secara terpisah |

Jumlahnya 87 family dalam enam kelompok. `g3` sendiri memuat lebih banyak family
daripada gabungan semua kelompok lain karena sebagian besar registry merupakan
lineage inference-only dan cakupan museum, bukan detektor yang aktif dilatih.

Bagi pembaca yang memilih model, kelompok menunjukkan tempat perhatian
engineering diharapkan, bukan tingkat akurasi family. `g0` dan `g1` adalah
tempat fitur baru dirancang dan diterapkan lebih dulu; `g2` dijaga tetap hijau
di CI, tetapi fitur diterapkan secara oportunistis, bukan dalam gelombang release
yang sama. `g3` menyatakan ketiadaan, bukan batas: predict, validate, dan export
jika didukung family tetap bekerja, sedangkan `train()` pada family `g3` atau
`g4` memunculkan `NotImplementedError` yang menyebutkan alasannya, bukan
melakukan proses parsial secara diam-diam. Family `s` sama sekali tidak berada
dalam trade-off ini karena dimuat melalui factory sendiri, bukan `LibreYOLO()`.
Lihat [konsep inti](/docs/concepts) untuk mengetahui posisi kelompok bersama
task, family, dan ukuran saat membaca nama file checkpoint.

Kelompok tidak memberikan atau membatasi kapabilitas yang terlihat pengguna.
Dukungan berasal dari API yang diimplementasikan family dan pemeriksaan
kapabilitas khusus format, bukan dari keanggotaan kelompok. Kelompok
mengklasifikasikan family, bukan task, sehingga run cakupan yang dibatasi task
menyebut task secara eksplisit, seperti "g1 detect".

Dua tempat membaca kelompok saat runtime, bukan hanya dalam pengujian.
`collect_model_inventory()` dalam `libreyolo/models/inventory.py` melampirkan
kelompok ke setiap entri yang dicetak inventaris CLI, dan `pretrained=False`
memicu jalur reinitialization from-scratch khusus hanya untuk family dalam `g0`
dan `g1`. Di luar kedua kelompok tersebut, pemeriksaan di
`libreyolo/models/base/model.py` dilewati sepenuhnya, sehingga
`pretrained=False` mencapai `train()` milik family sebagai keyword biasa.

## Pelatihan

Family dalam `g3` atau `g4` tidak memiliki implementasi pelatihan, dan
memanggil `train()` akan memunculkan error. Hal itu merupakan properti kode
family, bukan kelompoknya: kelompok mencatat fakta tersebut, bukan menyebabkannya.

Untuk family yang dapat dilatih, apakah suatu knob augmentasi mencapai pipeline
adalah pertanyaan terpisah dengan vocabulary tiga nilai sendiri, yaitu `used`,
`gated_by_mosaic`, dan `ignored`. Lihat
[matriks augmentasi](/docs/reference/augmentation-matrix).

## Hal yang tidak dijelaskan tier

Tier bukan klaim akurasi. Ekspor validated berarti artefak mereproduksi model
native dalam ambang batas yang ditetapkan; status tersebut tidak menjelaskan
kinerja model native pada dataset. Angka benchmark tersedia di halaman model.

Tier juga bukan pernyataan lisensi. Lisensi bobot berbeda-beda dalam satu family
dan repository yang menyediakan checkpoint tertentu adalah sumber otoritatif.
Keberadaan family dalam factory detektor tidak menjelaskan apakah bobot yang
dipublikasikan mengizinkan penggunaan komersial.
