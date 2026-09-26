---
title: Checkpoint dan bobot
seo_title: Checkpoint dan bobot LibreYOLO
description: >-
  Cara LibreYOLO menemukan, mengunduh, dan memverifikasi bobot model, lokasi
  hosting, cara berjalan tanpa jaringan, dan hal yang membuat checkpoint dimuat
  dengan aman.
lead: >-
  Checkpoint LibreYOLO adalah dictionary torch.save yang menyimpan state dict
  beserta metadata untuk mengidentifikasinya. Halaman ini membahas sumber berkas,
  lokasi penyimpanan, dan cara pemuatannya.
keywords:
  - bobot libreyolo
  - checkpoint libreyolo
  - download bobot libreyolo
  - libreyolo offline
  - libreyolo hugging face
  - metadata checkpoint
last_verified: 1.5.0
meta:
  - label: Lokasi hosting
    value: 'Satu repositori Hugging Face per checkpoint:'
    links:
      - label: huggingface.co/LibreYOLO
        href: 'https://huggingface.co/LibreYOLO'
  - label: Cache lokal
    value: bobot/ di bawah direktori kerja
    mono: true
  - label: Skema metadata
    value: v1.0
snippets:
  load:
    - label: Pengunduhan otomatis
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Nama berkas tanpa path diselesaikan ke weights/LibreYOLO9t.pt dan
        # diunduh ke sana jika belum tersedia.
        model = LibreYOLO("LibreYOLO9t.pt")
        print(model(SAMPLE_IMAGE).boxes)
    - label: Path eksplisit
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Path dengan komponen direktori digunakan persis sebagaimana ditulis
        dan

        # tidak pernah diambil dari jaringan.

        model = LibreYOLO("/opt/models/LibreYOLO9t.pt")

        print(model.family, model.size, model.task)
  inspect:
    - label: CLI
      language: bash
      code: |
        # Membaca metadata tanpa membuat model dan melaporkan
        # apakah berkas memenuhi skema.
        libreyolo metadata path=weights/LibreYOLO9t.pt
    - label: JSON
      language: bash
      code: |
        libreyolo metadata path=weights/LibreYOLO9t.pt --json
    - label: Python
      language: python
      code: >
        from libreyolo.utils.serialization import (
            load_untrusted_torch_file,
            validate_checkpoint_metadata,
        )


        loaded = load_untrusted_torch_file("weights/LibreYOLO9t.pt")


        # Mengembalikan list masalah. Kosong berarti berkas memenuhi v1.0.

        print(validate_checkpoint_metadata(loaded))

        print(loaded["model_family"], loaded["size"], loaded["task"],
        loaded["nc"])
source_hash: 210a12baa1417cfb
---

## Lokasi pencarian checkpoint

Referensi model tanpa komponen direktori, seperti `LibreYOLO9t.pt`, diselesaikan
terhadap `weights/` relatif terhadap direktori kerja saat ini. Jika
`weights/LibreYOLO9t.pt` tersedia, berkas tersebut digunakan; jika berkas dengan
nama yang sama tersedia dalam direktori kerja, berkas itu digunakan; jika tidak,
`weights/LibreYOLO9t.pt` menjadi target pengunduhan.

Referensi yang memuat direktori, absolut maupun relatif, diperlakukan secara
literal. Gunakan bentuk ini ketika bobot berada di lokasi pusat dan tidak boleh
ada pengambilan berkas.

<code-tabs name="load" />

## Pengunduhan otomatis

Jika path hasil resolve tidak ada, LibreYOLO melakukan parse nama berkas untuk
mendapatkan family, ukuran, dan task, lalu meminta URL unduhan dari family yang
cocok. Sebagian besar family membangunnya dari organisasi LibreYOLO di Hugging
Face, dengan satu repositori per checkpoint yang dinamai menurut berkas:

```text
https://huggingface.co/LibreYOLO/<name>/resolve/main/<name>.pt
```

Suffix varian dataset tetap menjadi bagian nama repositori, sehingga checkpoint
yang dilatih pada dataset selain default family diselesaikan ke repositori sendiri
dan tidak menimpa checkpoint default.

Transfer bersifat defensif karena berkas bobot terpotong akan gagal kemudian
dengan error yang tidak membantu. Pengunduhan disalurkan sebagai stream ke berkas `.part` dan
dipindahkan secara atomik hanya setelah lengkap, sehingga process yang terputus
tidak pernah meninggalkan checkpoint setengah tertulis pada path akhir. Transfer
yang terputus dilanjutkan dari offset byte dengan validator HTTP, lalu dimulai
dari nol jika server menyatakan objek berubah. Kegagalan dicoba ulang tiga kali
dengan exponential backoff. Process bersamaan yang menargetkan path sama memakai
lock berkas, sehingga dua run pelatihan yang dimulai bersamaan hanya mengunduh
sekali. Jika family mengambil dari host pihak ketiga, family dapat menetapkan
checksum dan menolak berkas yang tidak cocok.

Jika `HF_TOKEN` ditetapkan atau token disimpan dalam cache di
`~/.cache/huggingface/token`, token dilampirkan sebagai bearer token. Token hanya
dilampirkan ke URL `huggingface.co`, sehingga family yang mengunduh dari host
lain tidak pernah menerimanya.

Tidak setiap family mengunduh otomatis. Beberapa sengaja tidak mengembalikan URL
karena bobot terbitan tidak boleh didistribusikan ulang, lalu error menjelaskan
yang harus diberikan. Family lain mencetak pemberitahuan lisensi sebelum
transfer. Pemberitahuan tersebut adalah sinyal runtime bahwa ketentuan checkpoint
lebih sempit daripada kode dan layak dibaca.

## Organisasi Hugging Face

Bobot terbitan berada di
[huggingface.co/LibreYOLO](https://huggingface.co/LibreYOLO), satu repositori per
checkpoint. Setiap repositori memiliki lisensi dan lisensinya tidak seragam
dalam family: family dengan kode MIT dapat memiliki bobot yang bukan MIT.
Repositori adalah sumber otoritatif. Setiap halaman model mencantumkan checkpoint
terbitan dan lisensinya dalam bagian Checkpoint serta Licensing.

## Bekerja offline

Library tidak memerlukan akses jaringan setelah berkas tersedia secara lokal. Dua
pendekatan dapat digunakan:

Isi direktori `weights/` di sebelah lokasi job berjalan. Mengambil checkpoint
satu kali pada mesin terhubung lalu menyalin direktori sudah cukup; langkah
resolve di atas menemukannya dan tidak mengakses jaringan.

Atau, berikan path absolut ke lokasi bersama. Referensi dengan komponen direktori
digunakan sebagaimana diberikan, sehingga mount read-only berisi bobot terkurasi
merupakan setup valid. Jika process tidak dapat menulis di samping checkpoint
yang harus dikonversi, konversi kembali ke direktori sementara privat alih-alih
gagal.

Dataset mengikuti aturan terpisah: diselesaikan di bawah `~/datasets`, atau di
bawah direktori `LIBREYOLO_DATASETS_DIR` jika variabel tersebut ditetapkan.

## Keamanan pemuatan

Checkpoint adalah pickle, dan pickle dapat mengeksekusi kode arbitrer saat
dibuka. LibreYOLO memperlakukan setiap berkas bobot sebagai tidak tepercaya dan
memuatnya melalui jalur `weights_only=True` PyTorch, yang membatasi unpickler ke
tensor serta kumpulan kecil jenis aman. Aturan ini berlaku pada berkas yang
diberikan, bukan hanya berkas hasil unduhan LibreYOLO. Pada build PyTorch yang
terlalu lama untuk mendukung argumen tersebut, pemuatan ditolak alih-alih
dilakukan secara tidak aman.

Beberapa checkpoint pelatihan upstream menyematkan objek yang ditolak unpickler
restricted, seperti objek konfigurasi dari framework pelatihannya. Objek tersebut
adalah metadata yang tidak diperlukan LibreYOLO, sehingga selama konversi setiap
kelas yang diblokir diganti dengan pengganti inert yang memenuhi unpickler tanpa
menjalankan apa pun, dan hanya tensor yang bertahan dalam berkas hasil konversi.
Nama modul sensitif ditolak, bukan diberi stub, dan loop percobaan ulang dibatasi
agar berkas yang dirancang memperkenalkan rangkaian kelas terblokir tanpa akhir
gagal secara tertutup. Lihat [impor bobot yang ada](/docs/migrate) untuk jalur
selengkapnya.

## Metadata checkpoint

Checkpoint LibreYOLO adalah dictionary dengan kunci `model` yang menyimpan state
dict PyTorch. Sembilan kunci diwajibkan skema v1.0, dan bersama-sama memungkinkan
factory mengidentifikasi berkas tanpa parsing nama atau menebak bentuk tensor.

| Kunci | Arti |
|---|---|
| `model` | State dict PyTorch |
| `schema_version` | Versi kontrak metadata. v1.0 menggunakan string `1.0` |
| `libreyolo_version` | Versi LibreYOLO yang menghasilkan berkas |
| `model_family` | Identifier family terdaftar, seperti `yolo9` |
| `size` | Varian dalam family, seperti `t` atau `r18` |
| `task` | Satu nama task kanonis |
| `nc` | Jumlah kelas positif |
| `names` | Mapping indeks kelas ke label, mencakup `0` hingga `nc - 1` |
| `imgsz` | Resolusi input positif |

Task dengan struktur tambahan mencatatnya bersama kunci tersebut. Checkpoint pose
menambahkan `num_keypoints` dan `keypoint_dim`, serta dapat menambahkan sigma OKS
per keypoint. Checkpoint OCR menyematkan charset CTC lengkap agar berkas mandiri.
Checkpoint restore dapat mencatat jenis degradasi dan faktor upscale. Checkpoint
trainer menambahkan status resume seperti `epoch`, status optimizer, dan bobot
EMA; bobot inferensi terbitan seharusnya tidak memuatnya.

Berkas yang memenuhi sembilan kunci dimuat melalui jalur metadata. Berkas yang tidak
memenuhinya akan dikonversi jika family mengenali tata letak, atau dimuat melalui
jalur kompatibilitas dengan peringatan yang menyebutkan kekurangan.

## Memeriksa checkpoint

<code-tabs name="inspect" />

`libreyolo metadata` tidak pernah membuat model, sehingga dapat digunakan pada
berkas yang family-nya belum diinstal dan pada berkas yang belum diyakini.

