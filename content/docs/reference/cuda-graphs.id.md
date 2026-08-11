---
title: CUDA graph
seo_title: Matriks dukungan CUDA graph LibreYOLO
description: >-
  Family yang menangkap forward saat prediksi serta forward dan backward saat
  pelatihan, jaminan numeriknya, lokasi pemisahan capture, dan alasan family
  yang tidak didukung memunculkan error.
lead: >-
  CUDA graph merekam satu eksekusi dari urutan kernel tetap dan memutarnya ulang
  sebagai satu launch. LibreYOLO menangkap inferensi pada 39 family
  terverifikasi dan pelatihan pada 24 family, selalu per family, selalu setelah
  pemeriksaan paritas bitwise, dan tidak pernah sebagai fallback diam-diam.
keywords:
  - cuda graph libreyolo
  - cuda_graph=True
  - matriks dukungan cuda graph
  - pelatihan torch cuda graph
  - capture_error_mode thread_local
  - cuda graph identik bit
last_verified: 1.5.0
verification: >-
  List family inferensi diturunkan dari matriks CAPTURABLE dalam
  tests/e2e/test_cuda_graph_families.py pada v1.5.0. List family pelatihan,
  kelas paritas, dan timing dari docs/training_cuda_graphs.md. API dan
  NotImplementedError dari BaseModel._require_cuda_graph_support,
  cuda_graph_scope, dan capture_graph di libreyolo/models/base/model.py, dengan
  variabel kelas SUPPORTS_CUDA_GRAPH. Pemisahan seam dibaca dari override
  _get_graph_runner dalam family depth_anything3, birefnet, ppocr, sam, dan
  sensenova serta dari libreyolo/models/base/detr_cuda_graph.py.
  capture_error_mode dari libreyolo/models/base/cuda_graph.py dan
  libreyolo/training/cuda_graph.py. Fallback pelatihan dari
  libreyolo/training/trainer.py dan flag --cuda-graph dari
  libreyolo/cli/commands/train.py.
meta:
  - label: Family inferensi
    value: '39'
  - label: Family pelatihan
    value: '24'
  - label: Flag inferensi
    value: predict(cuda_graph=True)
    mono: true
  - label: Flag pelatihan
    value: train(cuda_graph=True)
    mono: true
snippets:
  usage:
    - label: Predict
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        # True menangkap saat penggunaan pertama per bentuk input.
        # "auto" menunggu hingga bentuk berulang sebelum membayar biaya capture.
        result = model(SAMPLE_IMAGE, cuda_graph=True)
    - label: Train
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.train(data="my-dataset.yaml", epochs=100, cuda_graph=True)
    - label: Pelatihan dari CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=my-dataset.yaml \
          epochs=100 --cuda-graph
source_hash: 67c46199939278f2
---

## Yang ditangkap

Graph merekam urutan kernel tetap dan alamat memori yang dibaca serta ditulis.
Graph tidak merekam nilai, bentuk, atau control flow. Replay menjadi satu launch
alih-alih ratusan, sehingga peningkatan terbesar terjadi pada network kecil dan
ukuran batch kecil, ketika sebuah langkah didominasi overhead launch, bukan
aritmetika.

Kedua entry point menangkap jumlah pekerjaan yang berbeda.

| | Di dalam graph | Eager |
|---|---|---|
| Inferensi | Forward network, `model._forward(x)` | Preprocessing, NMS, seluruh postprocessing |
| Pelatihan | Forward dan backward network | Loss, optimizer step, gradient clipping, EMA, schedule LR |

NMS maupun loss deteksi bukan kandidat. Keduanya memilih dengan boolean mask,
menjalankan Hungarian matching atau assigner, dan bercabang berdasarkan hasil,
yang merupakan hal persis yang tidak dapat direkam graph. Membiarkannya di luar
membuat capture aman, bukan batasan yang harus diakali.

<code-tabs name="usage" />

`cuda_graph` menerima tiga nilai saat prediksi. `False` adalah default. `True`
menangkap saat pertama kali setiap bentuk input terlihat. `"auto"` menunggu
hingga bentuk berulang, sehingga pekerjaan sekali jalan dan dengan bentuk
berubah-ubah tidak membayar capture yang tidak akan digunakan kembali.
`capture_graph(imgsz=None, batch=1, dtype=None)` memindahkan biaya dari
permintaan pertama, `graph_info()` melaporkan graph yang ditangkap dan jumlah
replay, sedangkan `release_graphs()` membebaskannya.

Saat pelatihan, flag berupa boolean biasa, `--cuda-graph` pada CLI. Lihat
[performa prediksi](/docs/predict/performance) dan
[performa pelatihan](/docs/train/performance) untuk kontrol di sekitarnya.

## Dukungan inferensi

Dukungan berlaku per family, dinyatakan melalui variabel kelas
`SUPPORTS_CUDA_GRAPH`, dan family baru ditandai setelah berhasil menangkap dan
melakukan replay yang identik secara bit terhadap dua input probe dari
distribusi berbeda. Matriks paritas bersama tersebut mencakup 39 family dalam
sembilan task.

| Task | Family |
|---|---|
| detect | yolo1, yolo2, yolo3, yolo4, yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, rfdetr, ec |
| segment | dfine, rtmdet, rfdetr, ec |
| pose | ec, yolonas, rfdetr |
| point | fomo |
| classify | resnet, convnext, mobilenetv4, efficientnetv2, clip, dinov2, siglip2 |
| semantic | eomt, dinov2, segformer, pidnet, lingbotvision |
| depth | depth_anything, depth_anything3, zipdepth |
| restore | nafnet, realesrgan, swinir |
| matte | birefnet |

Beberapa family muncul pada lebih dari satu task, sehingga matriks menjalankan
lebih banyak baris daripada jumlah family berbeda. Tiga family tambahan
melakukan capture melalui jalur khusus family dengan pengujian khusus sendiri,
bukan melalui matriks bersama, dan tidak termasuk dalam 39: PP-OCR, SAM, dan
SenseNova.

Verifikasi bersifat bitwise, bukan perkiraan. Versi awal protokol menilai paritas
berdasarkan magnitudo relatif dan secara keliru menurunkan tiga family sehat,
YOLOX, EfficientNetV2, dan YOLOv7, yang perbedaan eager-ke-graph-nya terukur
sekitar 1e-7 meski identik secara bit pada probe yang penting.

## Dukungan pelatihan

Capture pelatihan bertambah dari dua family menjadi 24 pada release ini, dalam
lima task.

| Task | Family |
|---|---|
| detect | yolo9, yolo9_p2, yolo9_e2e, yolox, yolo7, yolonas, picodet, rtmdet, rfdetr, dfine, deim, deimv2, rtdetr, rtdetrv2, rtdetrv4, ec |
| classify | resnet, convnext, mobilenetv4, efficientnetv2 |
| semantic | segformer, lingbotvision |
| point | fomo |
| restore | nafnet |

Selain yang tercantum, semuanya berlatih secara eager: task lain pada family
yang sama, family yang tidak ada dalam list, run terdistribusi, dan run
distillation. Capture juga dilewati selama bentuk masih baru karena jalur
pelatihan menunggu bentuk input berulang tiga kali sebelum menangkap. Artinya,
`multi_scale=True` dapat sama sekali tidak pernah melakukan capture.

## Dua jawaban berbeda untuk family yang tidak didukung

Jalur inferensi memunculkan error. `predict(cuda_graph=True)` pada family yang
belum ikut serta memunculkan `NotImplementedError` yang menyebutkan family,
bukan menjalankan eager dan membuat pengguna mengira memperoleh percepatan.
Alasannya, capture buruk tidak gagal dengan jelas: replay forward yang melakukan
operasi tidak dapat ditangkap akan diam-diam mengembalikan angka salah. Karena
itu, dukungan harus berupa pernyataan eksplisit per family, bukan percobaan
dengan fallback.

Jalur pelatihan mencatat log. `train(cuda_graph=True)` selalu aman diberikan,
dan family, task, atau konfigurasi yang tidak dapat ditangkap menulis satu baris
lalu berlatih eager tanpa perubahan. Capture yang gagal di tengah run juga
mengalihkan sisa run ke eager, bukan membatalkannya. Asimetri ini disengaja:
prediksi adalah pemanggilan yang dapat diperbaiki di call site, sedangkan run
pelatihan tidak seharusnya berhenti pada jam keenam akibat optimisasi opsional.

## Pemisahan seam

Beberapa family tidak dapat ditangkap secara utuh karena satu tahap benar-benar
melakukan operasi yang tidak dapat direkam graph. Alih-alih menghapus family,
capture dibagi pada seam yang terverifikasi: bagian yang dapat ditangkap
melakukan replay, sisanya berjalan eager, dan output gabungan sama dengan
menjalankan semuanya secara eager.

| Family | Ditangkap | Eager dan alasannya |
|---|---|---|
| Depth Anything 3 | Network | Tahap sky, yaitu pekerjaan yang terlihat host setelah forward |
| BiRefNet | Encoder, `forward_enc` | Decoder, dengan `deform_conv2d` yang menghasilkan nilai berbeda saat replay capture |
| PP-OCR | Tahap deteksi, `forward_det` | Pengenalan, karena lebar crop berbeda per baris |
| SAM | Image encoder | Jalur prompt, yang berjalan berkali-kali per encode |
| SenseNova | Vision tower | Generasi autoregresif dengan cache KV yang bertambah setiap langkah |
| Detektor encoder-decoder | Backbone dan encoder | Decoder dan Hungarian criterion |

Pemisahan BiRefNet perlu diperhatikan: kesalahan `deform_conv2d` saat capture
dapat direproduksi pada pemanggilan sederhana di luar model. Menggantinya dengan
padanan PyTorch murni ditolak karena juga akan menggeser prediksi eager, dan
angka eager adalah kontrak.

Kasus encoder-decoder mencakup D-FINE, DEIM, DEIMv2, RT-DETR, RT-DETRv2,
RT-DETRv4, dan EC. Decoder membangun query contrastive-denoising dari ground
truth, dan jumlah query berasal dari jumlah ground truth terbesar dalam batch,
sehingga jumlah token decoder berubah antarbatch. Itulah satu hal yang tidak
dapat ditoleransi graph. Backbone ditambah encoder menghabiskan sekitar
seperlima hingga seperempat langkah untuk family tersebut, sehingga berada di
bagian bawah tabel percepatan.

PP-OCR menangkap satu graph per bentuk input deteksi, dibatasi oleh batas cache
runner, dan mengembalikan hasil eager ketika tidak ada scope capture aktif.

## Numerik

Sebagian besar family identik secara bit, dan jika tidak, alasannya disebutkan
secara jelas. Pada langkah nol pelatihan, loss identik secara bit untuk semua 24
family dan tidak ada buffer BatchNorm yang berbeda; perbandingan gradienlah yang
memisahkan kategori.

| Kelas | Family | Arti |
|---|---|---|
| Exact | Sebagian besar dari 24 | Setiap gradien identik secara bit |
| 1 ULP | fomo, lingbotvision | Bit terakhir float32, sekitar 1e-7 relatif, akibat urutan penjumlahan berbeda |
| Eager noise | Lineage DETR | Perbedaan graph terhadap eager tidak lebih besar daripada perbedaan dua run eager satu sama lain |
| Float rounding | rtmdet | 137 dari 139 gradien identik secara bit, dua berbeda sekitar 3e-4 |
| Own RNG stream | segformer | Stochastic depth berada di dalam region yang ditangkap |

Kelas eager-noise harus dibaca dengan benar. Untuk family tersebut, dua run
eager dengan seed yang sama sudah berbeda, sehingga identik secara bit bukan
standar yang gagal dicapai run graph, melainkan standar yang tidak dicapai apa
pun. Hal ini berlaku lebih luas pada `amp=False`, ketika nondeterminisme relatif
3.2e-7 yang diukur dalam gradien bobot fp32 terakumulasi: dua run eager
YOLOv9-t dengan seed sama berbeda 36 persen setelah 20 langkah, dan
menonaktifkan TF32 tidak memperbaikinya.

## Pin memory

Capture berjalan dengan `capture_error_mode="thread_local"`. Dalam mode default
PyTorch `"global"`, thread pin-memory DataLoader yang menyiapkan batch berikutnya
memanggil `cudaHostAlloc`, yang membatalkan capture aktif sekaligus dirusak oleh
capture tersebut. Akibatnya, run berhenti saat mengambil batch berikutnya dengan
error dari dalam thread pin-memory. Pasangan kegagalan ini teramati dua kali
dalam campaign pelatihan nyata sebelum didiagnosis.

Mode thread-local hanya membatasi thread yang melakukan capture. Thread pin
tidak pernah menyentuh stream capture, sehingga tidak ada operasinya yang
termasuk graph. Pelatihan melangkah lebih jauh dan sementara mengganti subclass
`torch.cuda.CUDAGraph` yang memaksa mode tersebut karena
`make_graphed_callables` tidak menyediakan argumen untuknya. Penggantian
dilakukan di bawah lock agar dua capture bersamaan tidak meninggalkannya
terpasang.

## Nilai manfaatnya

Diukur pada RTX 5070 Ti dengan AMP, satu process per kelompok, melakukan replay
satu batch nyata agar dataloader tidak terlibat, menggunakan langkah tercepat
dari 24 langkah setelah warm-up. Deteksi pada 640 px, classification pada 224 px.

| Family | Batch | Percepatan |
|---|---:|---:|
| FOMO s | 16 | 3.63x |
| MobileNetV4 s | 16 | 2.74x |
| EfficientNetV2 b0 | 16 | 2.44x |
| YOLOv9-t | 8 | 1.99x |
| YOLOv9 e2e | 8 | 1.76x |
| YOLOv9 p2 | 8 | 1.49x |
| Selain itu | beragam | 1.04x hingga 1.26x |

Percepatan seluruh run lebih kecil karena graph tidak dapat mempercepat
dataloader atau validasi. Fine-tuning YOLOv9-t selama 20 epoch pada 406 gambar
turun dari 428.4 detik menjadi 367.7 detik, peningkatan end-to-end 1.16x, dengan
mAP50-95 identik sebesar 0.6394 pada kedua kelompok dan loss per epoch identik.

Batas atas ditentukan oleh porsi network dalam satu langkah. Pada hardware yang
sama di 640 px dan batch 8, porsinya 84 persen untuk YOLOv9-t, tetapi hanya 26
persen untuk RTMDet-t, yang menghabiskan sebagian besar langkah dalam label
assigner. Overhead launch tertinggi ada di Windows, sehingga peningkatan Linux
sekitar sepertiga hingga setengah tabel ini, dan run yang dibatasi dataloader
tidak mengalami perubahan wall-clock. Peak memory berubah antara 5 persen lebih
rendah dan 19 persen lebih tinggi.

## Catatan penting

Graph merekam alamat, bukan nilai, sehingga semua tindakan yang memindahkan
parameter akan menghapusnya. Mengubah device melalui `predict(device=...)`,
melakukan kuantisasi, dan membatalkan kuantisasi semuanya membatalkan graph yang
telah ditangkap.

Ukuran batch lebih berpengaruh daripada family: RT-DETR-r18 meningkat 1.19x pada
batch 2 dan 1.04x pada batch 8 karena batch besar dibatasi komputasi dan memiliki
lebih sedikit overhead launch yang dapat dihilangkan.

Suite paritas inferensi berjalan tanpa package `kernels` opsional terinstal,
sehingga keamanan capture dengan kernel Hub terkompilasi aktif tidak dicakup.
Tetapkan `LIBREYOLO_HUB_KERNELS=0` untuk mengeluarkannya saat mengisolasi masalah
capture. Lihat [kernel](/docs/reference/kernels).
