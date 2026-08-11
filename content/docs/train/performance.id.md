---
title: Performa pelatihan
seo_title: 'Pelatihan lebih cepat: CUDA graph, AMP, profiler'
description: >-
  Percepat proses pelatihan: tangkap langkah ke CUDA graph, pilih dtype AMP, dan
  gunakan profiler bawaan untuk menemukan sumber penggunaan waktu.
lead: >-
  Tiga pengungkit mengubah kecepatan langkah pelatihan: mixed precision, CUDA
  graph capture untuk forward dan backward jaringan, serta tindakan yang
  disarankan profiler berdasarkan penghambat sebenarnya.
keywords:
  - CUDA graph training
  - mempercepat training
  - mixed precision training
  - training bfloat16
  - profiler PyTorch
  - dataloader bottleneck
  - kernel launch overhead
  - utilisasi GPU
last_verified: 1.5.0
snippets:
  profile:
    - label: Profiling lalu lanjutkan pelatihan
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Melakukan profiling pada jendela pendek langkah nyata, mencetak hasil,
        # lalu melanjutkan proses setelah hook dilepas.
        model.train(data="my-dataset.yaml", epochs=100, profile=True)
    - label: 'Hanya ukur, lalu berhenti'
      language: bash
      code: >
        # Menetapkan no_aug_epochs=0 dan menjalankan epoch secukupnya untuk
        mengisi jendela.

        libreyolo profile run coco128 --weights LibreYOLO9s.pt --size s
    - label: Periksa hasil lebih dalam
      language: bash
      code: |
        libreyolo profile summary runs/profile/prof/profile.json
        libreyolo profile phases runs/profile/prof/profile.json
        libreyolo profile kernels runs/profile/prof/profile.json --top 10
  graph:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", epochs=100, cuda_graph=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 cuda_graph=true
  amp:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", amp=True, amp_dtype="bfloat16")
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          amp_dtype=bfloat16
source_hash: ee5bb727065b6099
---

## Ukur sebelum mengubah apa pun

Tiga pengungkit di bawah mengatasi masalah berbeda. Menerapkan pengungkit yang
salah tidak mengubah apa pun. Profiler menunjukkan masalah yang terjadi.

<code-tabs name="profile" />

`profile=True` mengukur jendela langkah pelatihan nyata, default-nya lima langkah
dibuang lalu dua puluh diukur, mencetak laporan, menulis artefak, dan melanjutkan
pelatihan setelah hook dilepas. Tidak ada biaya saat nonaktif, dan pengaturan ini
diabaikan pada pelatihan distributed.

Laporan berakhir dengan salah satu dari empat hasil:

| Hasil | Arti | Pengungkit |
|---|---|---|
| `dataloader` | GPU menunggu data input | tambah `workers`, `cache="ram"` atau `"disk"`, augmentasi lebih ringan, batch lebih besar |
| `host / launch` | GPU menerima pekerjaan terlalu lambat, banyak kernel kecil | batch lebih besar, CUDA graph, kurangi sinkronisasi host per langkah |
| `compute` | GPU jenuh | AMP atau bfloat16, atau terima kondisi ini |
| `memory-pressure` | allocator terus bekerja, VRAM di batas | turunkan batch; angka utilisasi tidak andal |

Angka utilisasi adalah waktu kernel sibuk dibagi waktu langkah tanpa sinkronisasi.
Jendela sengaja dibagi: separuh pertama berjalan tanpa sinkronisasi tambahan agar
hasil mencerminkan overlap nyata, dan hanya separuh kedua mengapit tiap fase dengan
sinkronisasi untuk mengatribusikan waktu GPU. Sinkronisasi setiap fase memberi
kelonggaran pada worker dataloader dan menyembunyikan starvation, sehingga angka
komposisi tidak digunakan untuk memilih hasil.

Empat file dibuat di direktori proses: `timeline.html` yang dapat dibuka langsung
di browser, `profile_trace.json` untuk Perfetto atau Nsight,
`profile_summary.json`, dan `profile.json`, file mandiri yang dapat disalin serta
diberikan ke subperintah `libreyolo profile`.

Dua hal mengenai `profile run` perlu diketahui. Perintah ini menetapkan
`no_aug_epochs=0` karena profiler mengukur epoch 0, dan proses pendek dengan
default akan mengukur dataloader tanpa augmentasi yang lebih ringan. `--repeat N`
melaporkan mean dan standard deviation karena langkah yang dibatasi launch cukup
berisik untuk menyesatkan jika hanya diukur sekali; perintah menulis direktori
`prof_1`, `prof_2`, dan seterusnya, serta agregat `profile_repeat.json`.

## Mixed precision

`amp=True` adalah default untuk sebagian besar family dan menjalankan forward pass
di bawah CUDA autocast. `amp_dtype` memilih `float16` atau `bfloat16`.

<code-tabs name="amp" />

Float16 memerlukan dynamic loss scaling dan gradient scaler aktif; rentang
eksponen bfloat16 lebih lebar sehingga scaler-nya dinonaktifkan. Empat family
memiliki `amp=False`: D-FINE, DEIM, YOLO-NAS, dan FOMO, dengan pengaturan DEIM
diwariskan ke RT-DETRv4. D-FINE menyebut alasannya: decoder membatasi aktivasi
pada 65504, nilai float16 hingga terbesar.

Semantik argumen, termasuk perilaku permintaan bfloat16 pada hardware tanpa
dukungan bfloat16, tersedia di [Hyperparameter](/docs/train/hyperparameters).

## CUDA graph

`cuda_graph=True` menangkap forward dan backward pelatihan jaringan ke CUDA graph,
menghilangkan overhead peluncuran kernel per langkah.

<code-tabs name="graph" />

Flag selalu aman diberikan. Family, task, atau konfigurasi yang tidak dapat
ditangkap mencatat satu baris dan tetap berlatih secara eager.

Hanya jaringan yang ditangkap. Loss tetap eager karena loss deteksi memilih
dengan mask boolean, menjalankan Hungarian matching, dan bercabang berdasarkan
hasil assignment, yang tidak dapat direkam graph. Langkah optimizer, gradient
clipping, pembaruan EMA, dan schedule learning rate juga tetap eager.

Keuntungan dibatasi oleh bagian langkah yang merupakan jaringan. Pada RTX 5070 Ti,
640 px, batch 8: jaringan mencakup 84 persen langkah YOLOv9-t, 44 persen YOLOv7-b,
31 persen YOLOX-t, dan 26 persen RTMDet-t. Dua terakhir menghabiskan sebagian
besar langkah di label assigner, sehingga mendapat manfaat paling kecil.

### Besarnya manfaat

Kondisi semua angka berikut: RTX 5070 Ti, Windows, AMP, satu proses per cabang
dari state tersimpan yang sama, replay satu batch nyata tanpa dataloader, langkah
tercepat dari 24 setelah warm-up. Deteksi 640 px, classification 224 px. Ukuran
batch tercantum per baris.

| Family | Ukuran | Batch | Eager | Graph | Percepatan |
|---|---|---:|---:|---:|---:|
| FOMO | s | 16 | 7.0 ms | 1.9 ms | 3.63x |
| MobileNetV4 | s | 16 | 14.5 ms | 5.3 ms | 2.74x |
| EfficientNetV2 | b0 | 16 | 29.0 ms | 11.9 ms | 2.44x |
| YOLOv9 | t | 8 | 93.6 ms | 47.0 ms | 1.99x |
| NAFNet | s | 8 | 132.5 ms | 105.5 ms | 1.26x |
| PicoDet | s | 8 | 145.0 ms | 118.7 ms | 1.22x |
| D-FINE | n | 4 | 185.3 ms | 159.2 ms | 1.16x |
| RF-DETR | n | 4 | 276.3 ms | 239.8 ms | 1.15x |
| YOLOX | t | 8 | 102.2 ms | 90.5 ms | 1.13x |
| RTMDet | t | 8 | 149.7 ms | 136.2 ms | 1.10x |
| YOLOv7 | b | 4 | 102.5 ms | 98.0 ms | 1.05x |

Angka itu mengisolasi langkah GPU. Fine-tuning lengkap juga membayar dataloader
dan validasi. YOLOv9-t pada dataset deteksi 406 gambar, 20 epoch, batch 8, 640 px,
4 worker, di mesin yang sama: 428,4 detik eager dibanding 367,7 detik dengan
graph, peningkatan 1,16x, dengan mAP50-95 0,6394 pada keduanya.

Tiga hal mengubah angka tersebut. Batch kecil dibatasi launch dan batch besar
dibatasi compute, sehingga RT-DETR-r18 mendapat 1,19x pada batch 2 dan 1,04x pada
batch 8. Overhead launch tertinggi di Windows; peningkatan Linux sekitar sepertiga
hingga setengah tabel. Proses yang dibatasi dataloader tidak mengalami perubahan
wall-clock, sehingga profiler harus digunakan lebih dahulu.

Capture bekerja sama pada `amp=False`, tetapi kernel fp32 berjalan lebih lama,
sehingga langkah kurang dibatasi launch dan sebagian besar family mendapat
manfaat lebih kecil. Pada hardware yang sama, MobileNetV4-s batch 16 berubah dari
2,74x dengan AMP menjadi 3,61x pada fp32, YOLOv9-t batch 8 dari 1,99x menjadi
1,69x, dan RT-DETR-r18 batch 4 dari 1,12x menjadi 0,99x.

### Cakupan capture

| Task | Family |
|---|---|
| detect | yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, rfdetr, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, ec |
| classify | resnet, convnext, mobilenetv4, efficientnetv2 |
| semantic | segformer, lingbotvision |
| point | fomo |
| restore | nafnet |

Semua yang lain kembali ke eager dengan satu baris log: task lain pada family
tersebut, family yang tidak tercantum, proses distributed, dan distilasi.
Kegagalan capture saat runtime juga mengalihkan sisa proses ke eager.

Untuk detector encoder-decoder D-FINE, DEIM, DEIMv2, RT-DETR v1, v2, v4, dan EC,
hanya backbone serta encoder yang ditangkap. Decoder membaca ground truth untuk
membangun query contrastive-denoising, yang jumlahnya mengikuti jumlah ground
truth terbesar dalam batch, sehingga jumlah token berubah antarbatch.

### Bentuk

Graph hanya valid untuk bentuk input saat capture. Trainer menghitung bentuk batch
dan melakukan capture setelah satu bentuk berulang tiga kali. Batch dengan bentuk
lain berjalan eager, termasuk batch multiskala dan batch parsial terakhir epoch.

Ini jebakan bagi family DETR yang mengubah ukuran setiap batch secara default.
Dengan `multi_scale=True`, proses pendek mungkin tidak melihat satu bentuk cukup
sering untuk capture. Berikan `multi_scale=False` jika percepatan menjadi tujuan.

YOLOX mengubah komputasi region yang ditangkap di tengah proses dengan menyalakan
cabang regresi L1 saat mosaic ditutup pada `no_aug_epochs`. Trainer membatalkan
capture lalu menangkap ulang setelah bentuk baru stabil.

### Numerik dan memori

Sebagian besar family mereproduksi trajectory loss eager bit demi bit di bawah
AMP. FOMO dan LingBot-Vision berbeda pada bit terakhir float32 akibat urutan
penjumlahan. Detector deformable-attention D-FINE, DEIM, DEIMv2, RT-DETR,
RF-DETR, dan EC juga tidak mereproduksi proses eager sendiri karena backward
berakumulasi dengan atomics dan konvolusi TF32 memilih urutan reduksi per launch;
proses graph tetap dalam sebaran tersebut. RTMDet berbeda sekitar 3e-4 relatif
pada dua dari 139 gradien karena berbagi konvolusi head antartingkat pyramid.
SegFormer memiliki stochastic depth dalam region capture, sehingga graph replay
memakai random stream sendiri dan setara secara statistik, bukan identik; manager
mencatatnya sekali saat capture.

Pada `amp=False`, hasil bit-identical tidak tersedia pada hardware ini, dengan
atau tanpa capture. Dua proses eager YOLOv9-t dengan seed identik berbeda 36
persen relatif setelah 20 langkah dan YOLOX-t 2,6 persen karena cuDNN memilih
algoritma weight-gradient nondeterministik untuk bentuk konvolusi fp32 tertentu.

Graph yang ditangkap menahan buffer input, output, dan workspace statis, sehingga
puncak VRAM naik kira-kira satu set aktivasi tambahan. Pada family di atas,
alokasi puncak berubah antara -5 dan +19 persen. Biaya relatif terbesar pada
model classification kecil: ResNet-18 224 px, batch 16, naik dari 0,48 GB eager
menjadi 0,57 GB dengan graph. Jika melewati batas, turunkan batch atau nonaktifkan flag.

## Terkait

- [Hyperparameter](/docs/train/hyperparameters) untuk `batch`, `nbs`, `cache`,
  dan `workers`.
- [Pelatihan multi-GPU](/docs/train/multi-gpu), yang tidak mendukung CUDA graph
  maupun profiler.
- [CUDA graph](/docs/reference/cuda-graphs) untuk matriks dukungan inferensi dan
  pelatihan gabungan, pembagian seam, serta kontrak numerik.
