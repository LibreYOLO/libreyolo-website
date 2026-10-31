---
title: Checkpoint upstream
seo_title: Memuat checkpoint upstream di LibreYOLO
description: >-
  Cara konversi otomatis mengubah checkpoint upstream yang dirilis menjadi
  checkpoint LibreYOLO v1.0: tata letak yang dibuka, family yang mengenali jenis
  tertentu, dan batas prosesnya.
lead: >-
  Family LibreYOLO diadaptasi dari project upstream yang checkpoint rilisnya hampir
  dapat dimuat, tetapi tidak memiliki metadata LibreYOLO. Konversi otomatis
  mengenali berkas tersebut, membungkusnya dalam skema v1.0, dan menulis hasil di
  samping sumber.
keywords:
  - autoconvert libreyolo
  - memuat checkpoint upstream
  - convert_upstream_state_dict
  - bobot upstream libreyolo
  - konversi checkpoint
last_verified: 1.5.0
verification: >-
  Perilaku dibaca dari libreyolo/models/autoconvert.py dan
  BaseModel.convert_upstream_state_dict; recognizer per family diperiksa dengan
  membaca override convert_upstream_state_dict setiap family, semuanya pada
  v1.5.0. Aturan COCO RF-DETR dari docs/checkpoint_schema.md.
snippets:
  usage:
    - label: Cukup berikan berkas kepada factory
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Berkas upstream yang dikenali dikonversi saat dimuat, dan checkpoint
        # hasil konversi ditulis di sampingnya.
        # model = LibreYOLO("yolov9-t-converted.pt")

        # Semua checkpoint LibreYOLO dimuat tanpa perubahan.
        model = LibreYOLO("LibreYOLO9t.pt")
        print(model.family, model.size, model.task, model.nb_classes)
source_hash: c6022771a2a207a1
---

## Yang terjadi saat pemuatan

Ketika `LibreYOLO()` menemukan berkas `.pt` yang belum menjadi checkpoint v1.0
lengkap, factory memanggil konverter otomatis yang:

1. membuka tensor dict dari tata letak upstream yang umum;
2. menanyakan kepada setiap family terdaftar apakah tata letak dikenali, dengan memetakan ulang kunci ketika penamaan upstream berbeda dari port native;
3. membungkus pemenang dalam checkpoint metadata v1.0 yang ketat, dengan membaca ukuran, task, dan jumlah kelas dari tensor itu sendiri agar checkpoint hasil fine-tuning dikonversi dengan benar;
4. menulisnya di samping sumber sebagai `<source>-<Prefix><size>[-task].pt` dan mengembalikan path tersebut agar factory memuatnya secara normal.

Pemanggil tidak perlu melakukan apa pun. Berkas yang tidak diklaim oleh family
mana pun tidak menghasilkan nilai, lalu factory melaporkan bahwa berkas tidak
dapat dimuat.

<code-tabs name="usage" />

## Tata letak yang dibuka

Tensor dict dicari dalam urutan preferensi berikut, dimulai dari EMA, dan setiap
kandidat dicoba hingga benar-benar memuat tensor. Karena itu, block EMA kosong
atau hanya berisi metadata tidak menutupi bobot valid di bawahnya.

| Kunci | Catatan |
|---|---|
| `ema.module` | Wrapper EMA umum |
| `ema` | Wrapper EMA flat lama yang menyimpan tensor secara langsung |
| `ema_state_dict` | Prefix `module.` pada entri dihapus |
| `params_ema` | |
| `params` | |
| `ema_net` | |
| `net` | |
| `model` | |
| `state_dict` | |
| Berkas itu sendiri | State dict biasa |

Setiap kandidat lalu dipersempit ke entri yang nilainya berupa tensor dan
dinormalisasi: prefix awal `module.` atau `_orig_mod.` dihapus, dan dict yang
semua kunci-nya diawali `model.model.` akan menghapus prefix tersebut.

## Family yang mengenali jenis tertentu

Pengenalan merupakan classmethod per family. Implementasi default mengklaim
tata letak yang kunci-nya sudah cocok dengan port native. Family yang penamaan kunci
upstream-nya berbeda melakukan override dengan remap, dan tidak mengembalikan
apa pun untuk tata letak yang tidak dikenal.

Family yang menyediakan recognizer dengan remapping: `centernet`, `deeplabv3`,
`deformable_detr`, `dexined`, `moge2`, `picodet`, `rtdetr`, `rtdetrv2`,
`rtdetrv4`, `rtmdet`, `segformer`, `swin`, `teed`, `yolo7`, `yolo9`,
`yolo9_e2e`, `yolo9_p2`.

Family yang menolak konversi otomatis sepenuhnya: `efficientdet`, `eomt`, dan
`pidnet` tidak mengembalikan apa pun dari recognizer, sehingga berkas upstream
mereka melewati script konversi. `l2cs` dikecualikan dari recognizer generik
karena hanya mendukung inferensi dengan bobot yang distribusi ulangnya dibatasi.

RF-DETR mempertahankan recognizer sendiri karena memerlukan seluruh checkpoint,
bukan hanya tensor dict, untuk mendeteksi ukuran dan memetakan ulang kelas COCO.
Recognizer ini hanya terdaftar ketika dependency opsionalnya diinstal.

Semua family terdaftar lain menggunakan default: family mengklaim berkas ketika
loader-nya sendiri sudah mengenali kunci tersebut.

## Family yang menang

Beberapa family dapat mengklaim berkas yang sama, sehingga proses resolve
mencerminkan aturan dispatch factory.

Klaim subclass mengalahkan base kelas. Urutan pendaftaran mengikuti pembuatan
kelas, sehingga family turunan mendaftar setelah base yang disempurnakannya, dan
marker positifnya tidak boleh kalah dari passthrough base yang lebih luas.

Selanjutnya, urutan registry menentukan karena mengenkode spesifisitas: klaim
paling awal adalah kecocokan paling spesifik.

Satu-satunya tie yang tidak dapat dipecahkan urutan registry adalah DEIM dan
D-FINE, yang kunci arsitekturnya identik. Hanya pada kasus tersebut nama berkas
menjadi sinyal penentu, dan berkas tanpa petunjuk nama ditolak alih-alih ditebak.
Nama berkas sengaja tidak diperiksa di tempat lain, sehingga klaim false-positive
yang luas tidak pernah dipromosikan di atas klaim lebih spesifik hanya karena
nama berkas.

## Pemuatan aman

Berkas upstream dimuat melalui unpickler bobot-only. Beberapa checkpoint
pelatihan upstream menyematkan objek library yang ditolak unpickler. Objek
tersebut merupakan metadata pelatihan, bukan bobot, sehingga setiap global yang
diblokir dicoba ulang dengan kelas pengganti inert yang memenuhi unpickler tanpa
mengeksekusi apa pun. Nama yang ditangkap hanya digunakan sebagai label string,
tidak pernah diimpor, dievaluasi, atau dipanggil.

Nama modul sensitif ditolak sepenuhnya dan tidak pernah diberi stub: `builtins`,
`os`, `sys`, `posix`, `nt`, dan `subprocess`. Loop percobaan ulang dibatasi 32
kali, sehingga berkas yang dirancang untuk memperkenalkan rangkaian global berbeda
tanpa batas akan gagal secara tertutup, bukan berputar terus. Hanya tensor yang
bertahan ke checkpoint hasil konversi.

## Lokasi berkas hasil konversi

Output ditulis di samping sumber dengan nama
`<source>-<Prefix><size>[-task].pt`. Berkas selalu ditulis ulang, bukan digunakan
kembali. Hal ini menjaga pemuatan berulang sumber yang sama tetap baru sekaligus
menghindari tabrakan dengan bobot resmi atau hasil fine-tuning lain dari family,
ukuran, dan task yang sama dalam direktori tersebut.

Jika direktori sumber bersifat read-only, konversi kembali ke direktori sementara
privat baru yang dibuat per pemanggilan, dan baris log menyebutkan path yang
digunakan. Hanya jika langkah ini juga gagal, hasil konversi dibuang dengan
peringatan.

## Checkpoint LibreYOLO yang sudah ada

Berkas yang memiliki marker khusus LibreYOLO, `libreyolo_version` atau
`model_family`, termasuk jalur pemuatan normal dan tidak dikonversi ulang.
Pelewatan hanya berlaku pada klaim passthrough, yaitu ketika kumpulan kunci tidak
berubah. Klaim yang konversinya mengubah kumpulan kunci membuktikan tata letak upstream
asing dan diterima meski berkas memiliki marker.

`schema_version` sengaja tidak diperlakukan sebagai marker karena alat pelatihan
dan ekspor lain menggunakan nama generik tersebut. Hal yang sama berlaku untuk
`names`, `nc`, `size`, `task`, atau `imgsz` karena hasil fine-tuning upstream
juga dapat memilikinya. Oleh sebab itu, hasil fine-tuning asing yang hanya
memiliki kunci `names` generik tidak ditandai, sehingga klaim dengan kunci native
dikonversi secara normal dan jumlah kelas diturunkan dari tensor head, bukan
keliru dimuat sebagai 80 kelas.

## Metadata yang dibaca konverter

Nama kelas diambil dari kunci top-level `names`, atau dari `class_names` di dalam
block `args` atau `hyper_parameters`. Map nama yang menggunakan label sebagai
kunci, bukan indeks kelas, tidak dapat digunakan dan diganti dengan default yang
dihasilkan. List nama yang lebih panjang daripada jumlah kelas yang terdeteksi
dipangkas karena indeks di luar rentang akan gagal pada validator ketat dan
diam-diam membatalkan konversi.

`args` upstream dibawa sebagai metadata biasa, dengan nilai selain string,
angka, boolean, list, atau dict dibuang agar tidak ada nilai tidak aman yang
mencapai berkas tersimpan.

## Normalisasi COCO RF-DETR

Checkpoint RF-DETR upstream menyediakan classification head dengan 91 output,
yaitu 90 kelas COCO ditambah background. Konversi otomatis menormalisasi
RF-DETR COCO ke konvensi COCO-80, dengan remap diterapkan pada postprocessing.

Checkpoint diperlakukan sebagai COCO ketika memuat tepat 80 nama, menyatakan
jumlah kelas 80, memiliki petunjuk dataset `coco`, atau sama sekali tidak
memiliki metadata kelas maupun dataset. Kasus terakhir penting: state dict
upstream biasa adalah checkpoint pretrained COCO kanonis dan merupakan satu-
satunya RF-DETR 91-output tanpa metadata yang didistribusikan.

RF-DETR kustom 90 kelas yang sesungguhnya dipertahankan sebagai 90 kelas. Model
ini diidentifikasi oleh list nama, jumlah kelas eksplisit selain 80, atau
petunjuk dataset non-COCO, sehingga fallback checkpoint biasa tidak aktif.
Placeholder kosong diabaikan saat menentukan keberadaan petunjuk dataset.

## Batasan

Konversi otomatis mengenali tata letak upstream yang dirilis. Proses ini tidak
menulis ulang arsitektur dan tidak membuat model yang belum diadaptasi dapat
dimuat. Ketika tidak ada family yang mengklaim berkas, jawabannya adalah script
konversi, bukan argumen factory: repositori menyediakan `weights/convert_*.py`
untuk family yang memerlukannya, termasuk EoMT, PIDNet, dan EfficientDet.

Konversi juga tidak menciptakan metadata yang tidak dapat dibaca. Ukuran, task,
dan jumlah kelas berasal dari tensor; nama berasal dari berkas jika tersedia dan
dihasilkan sebagai `class_i` jika tidak ada.

