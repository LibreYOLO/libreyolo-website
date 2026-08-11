---
title: Checkpoint dan bobot
seo_title: Checkpoint dan bobot LibreYOLO
description: >-
  Cara LibreYOLO menemukan, mengunduh, dan memverifikasi bobot model, lokasi
  hosting, cara berjalan tanpa network, dan hal yang membuat checkpoint dimuat
  dengan aman.
lead: >-
  Checkpoint LibreYOLO adalah dictionary torch.save yang menyimpan state dict
  beserta metadata untuk mengidentifikasinya. Halaman ini membahas sumber file,
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
  - label: Di-host di
    value: 'Satu repository Hugging Face per checkpoint:'
    links:
      - label: huggingface.co/LibreYOLO
        href: 'https://huggingface.co/LibreYOLO'
  - label: Cache lokal
    value: weights/ di bawah direktori kerja
    mono: true
  - label: Skema metadata
    value: v1.0
snippets:
  load:
    - label: Pengunduhan otomatis
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Nama file tanpa path di-resolve ke weights/LibreYOLO9t.pt dan
        # diunduh ke sana jika belum tersedia.
        model = LibreYOLO("LibreYOLO9t.pt")
        print(model(SAMPLE_IMAGE).boxes)
    - label: Path eksplisit
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Path dengan komponen direktori digunakan persis sebagaimana ditulis
        dan

        # tidak pernah diambil dari network.

        model = LibreYOLO("/opt/models/LibreYOLO9t.pt")

        print(model.family, model.size, model.task)
  inspect:
    - label: CLI
      language: bash
      code: |
        # Membaca metadata tanpa membuat model dan melaporkan
        # apakah file memenuhi skema.
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


        # Mengembalikan list masalah. Kosong berarti file memenuhi v1.0.

        print(validate_checkpoint_metadata(loaded))

        print(loaded["model_family"], loaded["size"], loaded["task"],
        loaded["nc"])
source_hash: 210a12baa1417cfb
---

## Lokasi pencarian checkpoint

Referensi model tanpa komponen direktori, seperti `LibreYOLO9t.pt`, di-resolve
terhadap `weights/` relatif terhadap direktori kerja saat ini. Jika
`weights/LibreYOLO9t.pt` tersedia, file tersebut digunakan; jika file dengan
nama yang sama tersedia dalam direktori kerja, file itu digunakan; jika tidak,
`weights/LibreYOLO9t.pt` menjadi target pengunduhan.

Referensi yang memuat direktori, absolut maupun relatif, diperlakukan secara
literal. Gunakan bentuk ini ketika bobot berada di lokasi pusat dan tidak boleh
ada pengambilan file.

<code-tabs name="load" />

## Pengunduhan otomatis

Jika path hasil resolve tidak ada, LibreYOLO melakukan parse nama file untuk
mendapatkan family, ukuran, dan task, lalu meminta URL unduhan dari family yang
cocok. Sebagian besar family membangunnya dari organisasi LibreYOLO di Hugging
Face, dengan satu repository per checkpoint yang dinamai menurut file:

```text
https://huggingface.co/LibreYOLO/<name>/resolve/main/<name>.pt
```

Suffix varian dataset tetap menjadi bagian nama repository, sehingga checkpoint
yang dilatih pada dataset selain default family di-resolve ke repository sendiri
dan tidak menimpa checkpoint default.

Transfer bersifat defensif karena file bobot terpotong akan gagal kemudian
dengan error yang tidak membantu. Pengunduhan di-stream ke file `.part` dan
dipindahkan secara atomik hanya setelah lengkap, sehingga process yang terputus
tidak pernah meninggalkan checkpoint setengah tertulis pada path akhir. Transfer
yang terputus dilanjutkan dari offset byte dengan validator HTTP, lalu dimulai
dari nol jika server menyatakan objek berubah. Kegagalan dicoba ulang tiga kali
dengan exponential backoff. Process bersamaan yang menargetkan path sama memakai
lock file, sehingga dua run pelatihan yang dimulai bersamaan hanya mengunduh
sekali. Jika family mengambil dari host pihak ketiga, family dapat menetapkan
checksum dan menolak file yang tidak cocok.

Jika `HF_TOKEN` ditetapkan atau token di-cache di
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
[huggingface.co/LibreYOLO](https://huggingface.co/LibreYOLO), satu repository per
checkpoint. Setiap repository memiliki lisensi dan lisensinya tidak seragam
dalam family: family dengan kode MIT dapat memiliki bobot yang bukan MIT.
Repository adalah sumber otoritatif. Setiap halaman model mencantumkan checkpoint
terbitan dan lisensinya dalam bagian Checkpoints serta Licensing.

## Bekerja offline

Library tidak memerlukan akses network setelah file tersedia secara lokal. Dua
pendekatan dapat digunakan:

Isi direktori `weights/` di sebelah lokasi job berjalan. Mengambil checkpoint
satu kali pada mesin terhubung lalu menyalin direktori sudah cukup; langkah
resolve di atas menemukannya dan tidak mengakses network.

Atau, berikan path absolut ke lokasi bersama. Referensi dengan komponen direktori
digunakan sebagaimana diberikan, sehingga mount read-only berisi bobot terkurasi
merupakan setup valid. Jika process tidak dapat menulis di samping checkpoint
yang harus dikonversi, konversi kembali ke direktori sementara privat alih-alih
gagal.

Dataset mengikuti aturan terpisah: di-resolve di bawah `~/datasets`, atau di
bawah direktori `LIBREYOLO_DATASETS_DIR` jika variabel tersebut ditetapkan.

## Keamanan pemuatan

Checkpoint adalah pickle, dan pickle dapat mengeksekusi kode arbitrer saat
dibuka. LibreYOLO memperlakukan setiap file bobot sebagai tidak tepercaya dan
memuatnya melalui jalur `weights_only=True` PyTorch, yang membatasi unpickler ke
tensor serta kumpulan kecil jenis aman. Aturan ini berlaku pada file yang
diberikan, bukan hanya file hasil unduhan LibreYOLO. Pada build PyTorch yang
terlalu lama untuk mendukung argumen tersebut, pemuatan ditolak alih-alih
dilakukan secara tidak aman.

Beberapa checkpoint pelatihan upstream menyematkan objek yang ditolak unpickler
restricted, seperti objek konfigurasi dari framework pelatihannya. Objek tersebut
adalah metadata yang tidak diperlukan LibreYOLO, sehingga selama konversi setiap
kelas yang diblokir diganti dengan pengganti inert yang memenuhi unpickler tanpa
menjalankan apa pun, dan hanya tensor yang bertahan dalam file hasil konversi.
Nama modul sensitif ditolak, bukan diberi stub, dan loop percobaan ulang dibatasi
agar file yang dirancang memperkenalkan rangkaian kelas terblokir tanpa akhir
gagal secara tertutup. Lihat [impor bobot yang ada](/docs/migrate) untuk jalur
selengkapnya.

## Metadata checkpoint

Checkpoint LibreYOLO adalah dictionary dengan key `model` yang menyimpan state
dict PyTorch. Sembilan key diwajibkan skema v1.0, dan bersama-sama memungkinkan
factory mengidentifikasi file tanpa parsing nama atau menebak bentuk tensor.

| Key | Arti |
|---|---|
| `model` | State dict PyTorch |
| `schema_version` | Versi kontrak metadata. v1.0 menggunakan string `1.0` |
| `libreyolo_version` | Versi LibreYOLO yang menghasilkan file |
| `model_family` | Identifier family terdaftar, seperti `yolo9` |
| `size` | Varian dalam family, seperti `t` atau `r18` |
| `task` | Satu nama task kanonis |
| `nc` | Jumlah kelas positif |
| `names` | Mapping indeks kelas ke label, mencakup `0` hingga `nc - 1` |
| `imgsz` | Resolusi input positif |

Task dengan struktur tambahan mencatatnya bersama key tersebut. Checkpoint pose
menambahkan `num_keypoints` dan `keypoint_dim`, serta dapat menambahkan sigma OKS
per keypoint. Checkpoint OCR menyematkan charset CTC lengkap agar file mandiri.
Checkpoint restore dapat mencatat jenis degradasi dan faktor upscale. Checkpoint
trainer menambahkan status resume seperti `epoch`, status optimizer, dan bobot
EMA; bobot inferensi terbitan seharusnya tidak memuatnya.

File yang memenuhi sembilan key dimuat melalui jalur metadata. File yang tidak
memenuhinya akan dikonversi jika family mengenali layout, atau dimuat melalui
jalur kompatibilitas dengan peringatan yang menyebutkan kekurangan.

## Memeriksa checkpoint

<code-tabs name="inspect" />

`libreyolo metadata` tidak pernah membuat model, sehingga dapat digunakan pada
file yang family-nya belum diinstal dan pada file yang belum diyakini.
