---
title: Checkpoint upstream
seo_title: Memuat checkpoint upstream di LibreYOLO
description: >-
  Cara konversi otomatis mengubah checkpoint upstream yang dirilis menjadi
  checkpoint LibreYOLO v1.0: layout yang dibuka, family yang mengenali jenis
  tertentu, dan batas prosesnya.
lead: >-
  Family LibreYOLO di-port dari project upstream yang checkpoint rilisnya hampir
  dapat dimuat, tetapi tidak memiliki metadata LibreYOLO. Konversi otomatis
  mengenali file tersebut, membungkusnya dalam skema v1.0, dan menulis hasil di
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
    - label: Cukup berikan file kepada factory
      language: python
      code: |
        from libreyolo import LibreYOLO

        # File upstream yang dikenali dikonversi saat dimuat, dan checkpoint
        # hasil konversi ditulis di sampingnya.
        # model = LibreYOLO("yolov9-t-converted.pt")

        # Semua checkpoint LibreYOLO dimuat tanpa perubahan.
        model = LibreYOLO("LibreYOLO9t.pt")
        print(model.family, model.size, model.task, model.nb_classes)
source_hash: c6022771a2a207a1
---

## Yang terjadi saat pemuatan

Ketika `LibreYOLO()` menemukan file `.pt` yang belum menjadi checkpoint v1.0
lengkap, factory memanggil konverter otomatis yang:

1. membuka tensor dict dari layout upstream yang umum;
2. menanyakan kepada setiap family terdaftar apakah layout dikenali, dengan memetakan ulang key ketika penamaan upstream berbeda dari port native;
3. membungkus pemenang dalam checkpoint metadata v1.0 yang ketat, dengan membaca ukuran, task, dan jumlah kelas dari tensor itu sendiri agar checkpoint hasil fine-tuning dikonversi dengan benar;
4. menulisnya di samping sumber sebagai `<source>-<Prefix><size>[-task].pt` dan mengembalikan path tersebut agar factory memuatnya secara normal.

Pemanggil tidak perlu melakukan apa pun. File yang tidak diklaim oleh family
mana pun tidak menghasilkan nilai, lalu factory melaporkan bahwa file tidak
dapat dimuat.

<code-tabs name="usage" />

## Layout yang dibuka

Tensor dict dicari dalam urutan preferensi berikut, dimulai dari EMA, dan setiap
kandidat dicoba hingga benar-benar memuat tensor. Karena itu, block EMA kosong
atau hanya berisi metadata tidak menutupi bobot valid di bawahnya.

| Key | Catatan |
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
| File itu sendiri | State dict biasa |

Setiap kandidat lalu dipersempit ke entri yang nilainya berupa tensor dan
dinormalisasi: prefix awal `module.` atau `_orig_mod.` dihapus, dan dict yang
semua key-nya diawali `model.model.` akan menghapus prefix tersebut.

## Family yang mengenali jenis tertentu

Pengenalan merupakan classmethod per family. Implementasi default mengklaim
layout yang key-nya sudah cocok dengan port native. Family yang penamaan key
upstream-nya berbeda melakukan override dengan remap, dan tidak mengembalikan
apa pun untuk layout yang tidak dikenal.

Family yang menyediakan recognizer dengan remapping: `centernet`, `deeplabv3`,
`deformable_detr`, `dexined`, `moge2`, `picodet`, `rtdetr`, `rtdetrv2`,
`rtdetrv4`, `rtmdet`, `segformer`, `swin`, `teed`, `yolo7`, `yolo9`,
`yolo9_e2e`, `yolo9_p2`.

Family yang menolak konversi otomatis sepenuhnya: `efficientdet`, `eomt`, dan
`pidnet` tidak mengembalikan apa pun dari recognizer, sehingga file upstream
mereka melewati script konversi. `l2cs` dikecualikan dari recognizer generik
karena hanya mendukung inferensi dengan bobot yang distribusi ulangnya dibatasi.

RF-DETR mempertahankan recognizer sendiri karena memerlukan seluruh checkpoint,
bukan hanya tensor dict, untuk mendeteksi ukuran dan memetakan ulang kelas COCO.
Recognizer ini hanya terdaftar ketika dependency opsionalnya diinstal.

Semua family terdaftar lain menggunakan default: family mengklaim file ketika
loader-nya sendiri sudah mengenali key tersebut.

## Family yang menang

Beberapa family dapat mengklaim file yang sama, sehingga proses resolve
mencerminkan aturan dispatch factory.

Klaim subclass mengalahkan base class. Urutan pendaftaran mengikuti pembuatan
kelas, sehingga family turunan mendaftar setelah base yang disempurnakannya, dan
marker positifnya tidak boleh kalah dari passthrough base yang lebih luas.

Selanjutnya, urutan registry menentukan karena mengenkode spesifisitas: klaim
paling awal adalah kecocokan paling spesifik.

Satu-satunya tie yang tidak dapat dipecahkan urutan registry adalah DEIM dan
D-FINE, yang key arsitekturnya identik. Hanya pada kasus tersebut nama file
menjadi sinyal penentu, dan file tanpa petunjuk nama ditolak alih-alih ditebak.
Nama file sengaja tidak diperiksa di tempat lain, sehingga klaim false-positive
yang luas tidak pernah dipromosikan di atas klaim lebih spesifik hanya karena
nama file.

## Pemuatan aman

File upstream dimuat melalui unpickler weights-only. Beberapa checkpoint
pelatihan upstream menyematkan objek library yang ditolak unpickler. Objek
tersebut merupakan metadata pelatihan, bukan bobot, sehingga setiap global yang
diblokir dicoba ulang dengan kelas pengganti inert yang memenuhi unpickler tanpa
mengeksekusi apa pun. Nama yang ditangkap hanya digunakan sebagai label string,
tidak pernah diimpor, dievaluasi, atau dipanggil.

Nama modul sensitif ditolak sepenuhnya dan tidak pernah diberi stub: `builtins`,
`os`, `sys`, `posix`, `nt`, dan `subprocess`. Loop percobaan ulang dibatasi 32
kali, sehingga file yang dirancang untuk memperkenalkan rangkaian global berbeda
tanpa batas akan gagal secara tertutup, bukan berputar terus. Hanya tensor yang
bertahan ke checkpoint hasil konversi.

## Lokasi file hasil konversi

Output ditulis di samping sumber dengan nama
`<source>-<Prefix><size>[-task].pt`. File selalu ditulis ulang, bukan digunakan
kembali. Hal ini menjaga pemuatan berulang sumber yang sama tetap baru sekaligus
menghindari tabrakan dengan bobot resmi atau hasil fine-tuning lain dari family,
ukuran, dan task yang sama dalam direktori tersebut.

Jika direktori sumber bersifat read-only, konversi kembali ke direktori sementara
privat baru yang dibuat per pemanggilan, dan baris log menyebutkan path yang
digunakan. Hanya jika langkah ini juga gagal, hasil konversi dibuang dengan
peringatan.

## Checkpoint LibreYOLO yang sudah ada

File yang memiliki marker khusus LibreYOLO, `libreyolo_version` atau
`model_family`, termasuk jalur pemuatan normal dan tidak dikonversi ulang.
Pelewatan hanya berlaku pada klaim passthrough, yaitu ketika kumpulan key tidak
berubah. Klaim yang konversinya mengubah kumpulan key membuktikan layout upstream
asing dan diterima meski file memiliki marker.

`schema_version` sengaja tidak diperlakukan sebagai marker karena alat pelatihan
dan ekspor lain menggunakan nama generik tersebut. Hal yang sama berlaku untuk
`names`, `nc`, `size`, `task`, atau `imgsz` karena hasil fine-tuning upstream
juga dapat memilikinya. Oleh sebab itu, hasil fine-tuning asing yang hanya
memiliki key `names` generik tidak ditandai, sehingga klaim dengan key native
dikonversi secara normal dan jumlah kelas diturunkan dari tensor head, bukan
keliru dimuat sebagai 80 kelas.

## Metadata yang dibaca konverter

Nama kelas diambil dari key top-level `names`, atau dari `class_names` di dalam
block `args` atau `hyper_parameters`. Map nama yang menggunakan label sebagai
key, bukan indeks kelas, tidak dapat digunakan dan diganti dengan default yang
dihasilkan. List nama yang lebih panjang daripada jumlah kelas yang terdeteksi
dipangkas karena indeks di luar rentang akan gagal pada validator ketat dan
diam-diam membatalkan konversi.

`args` upstream dibawa sebagai metadata biasa, dengan nilai selain string,
angka, boolean, list, atau dict dibuang agar tidak ada nilai tidak aman yang
mencapai file tersimpan.

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

Konversi otomatis mengenali layout upstream yang dirilis. Proses ini tidak
menulis ulang arsitektur dan tidak membuat model yang belum di-port dapat
dimuat. Ketika tidak ada family yang mengklaim file, jawabannya adalah script
konversi, bukan argumen factory: repository menyediakan `weights/convert_*.py`
untuk family yang memerlukannya, termasuk EoMT, PIDNet, dan EfficientDet.

Konversi juga tidak menciptakan metadata yang tidak dapat dibaca. Ukuran, task,
dan jumlah kelas berasal dari tensor; nama berasal dari file jika tersedia dan
dihasilkan sebagai `class_i` jika tidak ada.
