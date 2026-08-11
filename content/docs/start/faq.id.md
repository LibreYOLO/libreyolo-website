---
title: FAQ
seo_title: FAQ LibreYOLO
description: >-
  Jawaban singkat untuk pertanyaan yang mencakup semua model LibreYOLO:
  hardware, lisensi, bobot, device, pelatihan, cakupan ekspor, dan CLI.
lead: >-
  Jawaban untuk pertanyaan yang tidak khusus pada satu family model. Hal yang
  khusus family tersedia pada halaman family tersebut.
keywords:
  - faq libreyolo
  - apakah libreyolo perlu gpu
  - lisensi libreyolo
  - lokasi bobot libreyolo
  - cli libreyolo
  - libreyolo offline
last_verified: 1.5.0
source_hash: a729b43a6642f2a0
---

## Model mana yang sebaiknya digunakan pertama kali?

YOLOv9 untuk detektor CNN dan RF-DETR untuk detektor transformer. Keduanya berada
dalam tier flagship, yang berarti fitur dirancang dan divalidasi pada GPU
terhadap keduanya sebelum yang lain. Lihat [YOLOv9](/docs/models/yolov9) dan
[RF-DETR](/docs/models/rf-detr), atau [semua model](/docs/models) untuk pilihan
lain.

## Apakah GPU diperlukan?

Tidak. Setiap model berjalan pada CPU, dan semua contoh dalam
[mulai cepat](/docs/quickstart) ditulis agar berjalan di sana. GPU mengubah durasi
pelatihan dan inferensi video, bukan apakah keduanya berfungsi.

## Bagaimana LibreYOLO memilih device?

Default-nya adalah `device="auto"`, yang menggunakan CUDA ketika PyTorch
melaporkannya tersedia, kemudian Metal Performance Shaders jika tersedia, dan
CPU jika tidak. Untuk menetapkannya, berikan `device` kepada model atau
`predict`, `train`, `val`, dan `export`. Nilai yang diterima adalah `"cpu"`,
`"cuda"`, `"cuda:0"`, `"mps"`, bilangan bulat seperti `0`, atau string digit;
dua yang terakhir dikembangkan menjadi `cuda:<n>`.

`libreyolo checks` mencetak build Torch, versi CUDA dan cuDNN, serta setiap GPU
yang terlihat. Jika perintah tersebut tidak menunjukkan CUDA, wheel PyTorch
adalah build CPU; [instalasi](/docs/install) menjelaskan cara menggantinya.

## Di mana bobot hasil unduhan disimpan?

Dalam `weights/` relatif terhadap direktori kerja. Referensi model tanpa
komponen direktori diselesaikan di sana dan diunduh saat penggunaan pertama;
referensi yang menyertakan direktori digunakan persis sebagaimana ditulis dan
tidak pernah diambil. Lihat [checkpoint dan bobot](/docs/weights).

## Dapatkah LibreYOLO berjalan tanpa akses jaringan?

Ya. Ambil checkpoint satu kali pada mesin terhubung, salin direktori `weights/`,
dan tidak ada akses jaringan lagi. Path read-only bersama juga berfungsi karena
referensi yang memuat direktori diperlakukan secara literal. Dataset diselesaikan
di bawah `~/datasets`, atau di bawah `LIBREYOLO_DATASETS_DIR`.

## Dapatkah LibreYOLO digunakan secara komersial?

Kode berlisensi MIT. Bobot pretrained merupakan pertanyaan terpisah: bobot dapat
mewarisi ketentuan dari project atau dataset asalnya, dan ketentuannya tidak
seragam bahkan dalam satu family. Lisensi pada repositori Hugging Face tertentu
adalah sumber otoritatif, dan setiap halaman model memiliki bagian lisensi yang
mereproduksinya. Jika bobot dibatasi, LibreYOLO mencetak batasan sebelum
pengunduhan dimulai.

## Dapatkah checkpoint dari project lain dimuat?

Biasanya bisa, dengan memberikan path-nya kepada `LibreYOLO()`. Tata letak upstream
yang dikenali dikonversi saat pemuatan dengan mempertahankan jumlah dan nama
kelasnya, lalu checkpoint LibreYOLO ditulis di samping sumber. [Impor bobot yang
ada](/docs/migrate) menjelaskan yang dikenali dan yang memerlukan script konversi.

## Mengapa train memunculkan NotImplementedError?

Karena family tersebut hanya mendukung inferensi, dan exception menyebutkan
alasannya. Prediksi, validasi, dan ekspor jika didukung tetap berfungsi; tidak
ada loop pelatihan untuk arsitektur tersebut dalam LibreYOLO. Tier dukungan pada
header halaman model memberi tahu hal ini sebelum dicoba. Lihat
[konsep inti](/docs/concepts).

## Apa yang dikembalikan val?

Dictionary biasa, bukan objek. Kunci deteksi mencakup `metrics/precision`,
`metrics/recall`, `metrics/mAP50`, dan `metrics/mAP50-95`. Task lain
mengembalikan kunci yang relevan, seperti `metrics/accuracy_top1` untuk
classification atau `metrics/PQ`, `metrics/SQ`, dan `metrics/RQ` untuk
panoptic segmentation.

## Bagaimana menjalankan folder, video, atau webcam?

Berikan sebagai sumber. Path berkas adalah satu gambar, direktori adalah setiap
gambar di dalamnya, path video adalah video, bilangan bulat adalah indeks
webcam, dan URL RTSP, RTMP, TCP, UDP, atau HLS adalah live stream. Berkas
`.streams` mencantumkan beberapa sumber sekaligus. Live sumber memerlukan
`stream=True`, yang menghasilkan satu `Results` per frame alih-alih membangun
list; flag yang sama layak digunakan untuk video panjang dan direktori besar.
Hanya URL halaman YouTube yang memerlukan ekstra `libreyolo[stream]`.

## Bagaimana hanya mempertahankan kelas tertentu?

Berikan `classes` kepada `predict` dengan indeks kelas yang diinginkan, misalnya
`classes=[0, 2]`. `conf` menetapkan ambang batas confidence dengan default
`0.25`, dan `max_det` membatasi deteksi per gambar dengan default `300`.

## Apakah CLI menggunakan flag atau pasangan kunci=value?

Kunci dan value yang disambung tanda sama dengan untuk setiap perintah:

```bash
libreyolo predict model=yolo9-t source=my-image.jpg save=True
libreyolo train model=yolo9-t data=coco8.yaml epochs=50 imgsz=640
```

`model` menerima path atau nama pendek berbentuk `family-size`, dengan suffix
task opsional, dan `libreyolo models` mencantumkan setiap nilai valid. Perintah
diagnostik dan inventaris juga menerima `--json`, yang mencetak data yang sama
sebagai objek machine-readable ke stdout.

## Dapatkah setiap model diekspor ke setiap format?

Tidak. Cakupan berlaku per family dan per task, bukan seragam, dan setiap format
memiliki ekstra yang harus diinstal. Setiap halaman model memiliki matriks
ekspor family; [bagian ekspor](/docs/export) menjelaskan formatnya.

## Apa perbedaan segment, semantic, dan panoptic?

Ketiganya merupakan task berbeda. `segment` menghasilkan satu mask per objek
terdeteksi. `semantic` memberi label kelas pada setiap piksel tanpa memisahkan
instance. `panoptic` memberi setiap piksel tepat satu label, menggabungkan thing
yang dapat dihitung dengan stuff amorf. Ground truth, kolom hasil, dan metriknya
berbeda, dan family mendukung task yang muncul dalam list task-nya.

## Bagaimana melatih dengan kelas sendiri?

Buat YAML dataset dengan `train`, `val`, dan `names`. Label berada di samping
gambar dalam tree `labels/` paralel, satu `.txt` per gambar, dengan koordinat
ternormalisasi. `nc` opsional dan harus cocok dengan `names` jika ada. Jalankan
`libreyolo doctor <data.yaml>` terlebih dahulu: perintah ini memeriksa masalah
dataset dan keluar nonzero ketika menemukan error, sehingga dapat digunakan
sebagai gate CI.

## Mengapa pemuatan menampilkan peringatan metadata?

Karena checkpoint tidak memiliki metadata v1.0 lengkap. Pemuatan berlanjut
melalui jalur kompatibilitas, dan peringatan menyebutkan kunci yang hilang.
Jalankan `libreyolo metadata path=<file>` untuk melihat isinya, dan lihat
[checkpoint dan bobot](/docs/weights) untuk persyaratan skema.

## Import berhenti berfungsi setelah upgrade. Apa yang berubah?

Dua nama kelas diubah agar konsisten: `LibreYOLORTDETR` menjadi `LibreRTDETR`
dan `LibreYOLORFDETR` menjadi `LibreRFDETR`. Nama lama tetap diselesaikan dan
menghasilkan `DeprecationWarning` yang menunjuk ke nama baru, sehingga kode lama
tetap berjalan selama diperbarui.



