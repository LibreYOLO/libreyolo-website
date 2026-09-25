---
title: Alternatif Terbaik Ultralytics YOLO 2026
description: "Panduan praktis untuk alternatif open source terbaik bagi Ultralytics YOLO pada 2026: lisensi, cakupan task, dan batas deployment, serta posisi LibreYOLO sebagai library YOLO berlisensi MIT yang paling lengkap."
date: 2026-07-01
author: Xuban
tags: [LibreYOLO, ultralytics-alternative, object-detection, yolo, mit-license]
faq:
  - q: "Apakah Ultralytics YOLO gratis untuk penggunaan komersial?"
    a: "Hanya di bawah AGPL-3.0, yang mewajibkan Anda merilis source aplikasi yang dibangun di atasnya, termasuk layanan jaringan. Penggunaan komersial closed source memerlukan lisensi komersial berbayar dari Ultralytics. Alternatif permisif (MIT, Apache-2.0) menghindari kewajiban ini."
  - q: "Apa alternatif Ultralytics dengan lisensi paling sederhana?"
    a: "Untuk stack permisif yang dapat dikirim ke mana saja: LibreYOLO (kode MIT), RF-DETR (Apache-2.0), dan YOLOX (Apache-2.0). Semuanya menghindari copyleft AGPL."
  - q: "Apa yang menggantikan YOLO pada 2026?"
    a: "Detektor transformer real-time semakin banyak digunakan: RT-DETR, RF-DETR, serta lini D-FINE dan DEIM. Pada hardware yang mumpuni, akurasinya setara atau lebih baik daripada YOLO dengan latensi serupa. CNN bergaya YOLO masih unggul pada perangkat edge kecil, tempat transformer tersebut berjalan kurang baik dan belum memiliki jalur NCNN atau CPU yang matang."
  - q: "Bisakah model ini berjalan di Raspberry Pi atau NPU?"
    a: "Tergantung ekspornya. Detektor CNN seperti YOLOX dan RTMDet dapat diekspor ke NCNN dan berjalan di Pi, dan beberapa di antaranya (YOLOX) dapat menggunakan NPU Hailo. Detektor transformer seperti RF-DETR tidak bisa, model tersebut memerlukan GPU atau CPU yang kuat."
---

Pengungkapan: LibreYOLO adalah proyek kami, dan berada di urutan pertama dalam daftar ini. Semua tool lainnya di sini kami gunakan dalam pekerjaan nyata, dan kami mencatat keunggulannya dibanding LibreYOLO.

Sebagian besar tim meninggalkan Ultralytics karena salah satu dari tiga alasan berikut:

- **Lisensi.** Model YOLO dari Ultralytics berlisensi AGPL-3.0 (YOLOv5 sejak 2023, dan v8 ke atas). Polanya sudah akrab: Anda membangun produk, merilisnya, lalu tinjauan hukum atau pelanggan menyoroti model tersebut. Pilihannya menjadi membuka source seluruh aplikasi atau membeli lisensi komersial. Copyleft AGPL menjangkau kode Anda begitu Anda mendistribusikan atau menyediakan layanan. Ini alasan umum orang mencari alternatif.
- **Keterbukaan.** Satu vendor memiliki model, kode pelatihan, dan ketentuannya. Sebagian tim menginginkan bobot dan kode di bawah lisensi permisif yang bisa mereka kembangkan tanpa meminta izin.
- **Modelnya.** YOLO tidak selalu menjadi arsitektur terbaik untuk task tertentu. Detektor transformer real-time seperti RT-DETR, RF-DETR, dan D-FINE dapat lebih akurat pada latensi yang sama. Kekurangannya, model tersebut tersebar di berbagai repo riset, dan seperti yang akan Anda lihat, LibreYOLO menjalankan ketiganya melalui satu API. Jadi ini alasan untuk beralih library, bukan alasan untuk meninggalkan satu library demi satu model.

Setiap tool di bawah dinilai berdasarkan tiga hal: lisensi yang memungkinkan Anda merilis produk, apakah tool dapat dipasang dan dijalankan dengan PyTorch terkini, dan apakah tool dapat diekspor ke hardware yang benar-benar Anda gunakan.

Satu pola akan berulang saat Anda membaca: setiap alternatif bagus, tetapi merepotkan jika digunakan sendiri. YOLOX hampir tidak bisa dipasang, MMDetection terjebak dalam masalah pin versi wheel, keluarga RT-DETR adalah kode riset tanpa jalur dukungan, dan Detectron2 tidak berubah sejak 2021. LibreYOLO berada di urutan pertama dalam daftar ini karena sudah membungkus sebagian besar tool tersebut dalam satu API yang familier dan terawat, dengan lisensi MIT serta stack terkini. Posisinya bukan sekadar salah satu entri dalam daftar, melainkan lapisan di atasnya.

## Ringkasan singkat

| Tool | Lisensi | Task yang dicakup | Terbaik untuk | Ekspor edge |
| --- | --- | --- | --- | --- |
| **LibreYOLO** | MIT (kode) | Deteksi, segmentasi, pose, klasifikasi, kedalaman, gaze, tracking | Hub MIT untuk lebih dari 20 family model dalam satu API | ONNX, TensorRT, OpenVINO, NCNN, CoreML, TFLite |
| **RF-DETR** | Apache-2.0 (N/S/M/L) | Deteksi, segmentasi, pose (pratinjau) | Deteksi dan segmentasi untuk produksi | ONNX, TFLite, TensorRT; tanpa NCNN atau Hailo |
| **Lightly / LightlyTrain** | MIT / AGPL-3.0 | Pretraining, distilasi | Data tanpa label, distilasi DINOv2/v3 | n/a (bukan detektor) |
| **YOLOX** | Apache-2.0 | Deteksi | Deteksi real-time anchor-free | Upstream sulit dipasang, berjalan melalui LibreYOLO |
| **MMDetection / fork OneDL** | Apache-2.0 | Semua (riset) | Ragam arsitektur, mereproduksi paper | melalui ekspor |
| **Detectron2** | Apache-2.0 | Deteksi, segmentasi, keypoint | Mask R-CNN dan keluarga R-CNN | manual |
| **D-FINE / DEIM / DEIMv2** | Apache-2.0 | Deteksi | DETR real-time tercanggih | ONNX, TensorRT |
| **EdgeCrafter** | Apache-2.0 | Deteksi, segmentasi, pose | ViT edge ringkas hasil distilasi DINOv3 | tingkat riset |
| **RT-DETR (v1-v4)** | Apache-2.0 | Deteksi | Deteksi real-time terdepan | ONNX, TensorRT |

## 1. LibreYOLO

Lisensi: MIT (kode). Terbaik sebagai library YOLO berlisensi MIT yang paling lengkap dan alternatif Ultralytics yang bisa langsung menggantikan.

**LibreYOLO adalah library YOLO berlisensi MIT yang menjalankan setiap model melalui satu API.** Ini proyek kami, dan setelah membaca bagian lain dalam daftar ini, perannya akan jelas: LibreYOLO adalah hub untuk menjalankan alternatif tersebut. Hampir semua repo yang merepotkan di atas, YOLOX, RTMDet, RF-DETR, D-FINE, DEIM, lini RT-DETR, sudah dibungkus LibreYOLO dalam satu API yang familier dan terawat. Anda mendapat modelnya tanpa perlu menelusuri cara instalasi atau mengurus lisensinya.

Ada alasan yang lebih mendasar daripada sekadar kemudahan. YOLO bermula sebagai riset terbuka: Darknet karya Joseph Redmon, dari v1 sampai v3, gratis untuk digunakan dan dikembangkan siapa pun. Era AGPL membatasi akses tersebut. LibreYOLO hadir agar karya itu bisa diakses lagi seperti yang dimaksud para pembuatnya, dengan lisensi permisif, dikelola komunitas, dan tanpa mengirim data ke server lain. Ada dua hal yang perlu dibahas.

**Pertama: ini alternatif Ultralytics YOLO berlisensi MIT.** API yang sudah Anda kenal tetap tersedia, sehingga dataset berformat YOLO dan skrip Anda dapat dipindahkan dengan sedikit perubahan. Namun, lisensinya MIT, bukan AGPL: copyleft tidak menjangkau kode Anda, Anda tidak perlu membeli lisensi komersial, dan tidak ada telemetri yang mengirim data ke server seperti yang dilakukan Ultralytics secara default. Jika masalah lisensi yang membawa Anda ke sini, itulah intinya. YOLO9 dan RF-DETR adalah default yang paling banyak diuji dan siap produksi. Zoo model yang lebih luas masih baru dan ditandai dengan jelas sebagai eksperimental. Kami memilih untuk menyatakannya daripada mengklaim semuanya sudah teruji di berbagai kondisi.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")   # or LibreRFDETRl.pt, LibreDFINEl.pt, ...
results = model("image.jpg", save=True)
```

**Kedua: cakupannya jauh lebih luas daripada Ultralytics.** Satu API mencakup lebih dari 20 family model: YOLO berbasis CNN, DETR berbasis transformer, backbone ViT, SAM, dan lainnya, bukan hanya jajaran produk dari satu vendor. Cakupannya juga tidak terbatas pada detektor:

- **Task:** segmentasi instance dan semantik, pose, klasifikasi, serta bounding box berorientasi. Ada juga dua task yang sama sekali tidak tersedia di Ultralytics: estimasi kedalaman monokular (Depth Anything V2) dan estimasi gaze (L2CS).
- **Lapisan praktis:** tracking multi-objek dengan ID persisten (ByteTrack dan OC-SORT), inferensi video dengan subsampling frame dan pratinjau langsung, inferensi berbasis tile untuk citra drone dan satelit, UI browser seret dan lepas (`libreyolo ui`), serta perintah `doctor` yang memeriksa dataset sebelum Anda menjalankan pelatihan.
- **Pelatihan mendalam:** akumulasi gradien, pembekuan layer, melanjutkan pelatihan, augmentasi saat pengujian, fine-tuning LoRA/DoRA, multi-GPU, serta pencatatan TensorBoard/MLflow/Weights & Biases.
- **Ekspor:** tujuh format (ONNX, TorchScript, TensorRT, OpenVINO, NCNN, CoreML, TFLite) dengan kuantisasi INT8/FP16, NMS tertanam, dan metadata model tertanam.
- **Berjalan di mana saja:** menerima path, URL, PIL, NumPy, tensor, atau byte mentah, dan berjalan di CUDA, Apple Silicon (MPS), atau CPU biasa tanpa perubahan kode.

Saat repo detektor yang kuat ditinggalkan atau sulit dipasang, LibreYOLO tetap dapat menjalankan bobotnya, dengan stack terkini dan pemeliharaan aktif.

## 2. RF-DETR

Lisensi: Apache-2.0 (Nano, Small, Medium, Large). Terbaik untuk deteksi dan segmentasi produksi.

RF-DETR, dari Roboflow dan dipublikasikan di ICLR 2026, adalah model paling siap produksi dalam daftar ini. Rekayasanya baik, dikembangkan oleh tim yang merilis produk, dan menjadi pilihan awal yang kuat saat Anda memerlukan deteksi atau segmentasi yang berfungsi dan dapat diandalkan dalam sistem nyata. Paket `rfdetr` dan bobot Nano hingga Large berlisensi Apache-2.0. Bobot XL dan 2XL yang lebih besar menggunakan lisensi PML milik Roboflow.

## 3. Lightly

Terbaik untuk data tanpa label, pretraining self-supervised, dan distilasi backbone DINO.

Lightly bukan detektor. Tool ini membantu Anda memanfaatkan data yang belum diberi label, dan terdiri dari dua bagian dengan lisensi berbeda.

- **[`lightly`](https://github.com/lightly-ai/lightly) (LightlySSL), MIT.** Framework berisi komponen pembelajaran self-supervised: loss, head, augmentasi, memory bank, dan implementasi referensi lebih dari dua puluh metode (SimCLR, MoCo, BYOL, DINO, DINOv2, MAE, dan lainnya). Anda menulis loop pelatihan sendiri; Lightly menyediakan semua komponen untuk melakukan pretraining representasi dari nol dengan gambar tanpa label.
- **[LightlyTrain](https://github.com/lightly-ai/lightly-train), AGPL-3.0 (dengan opsi komersial dan komunitas gratis).** Opsi siap pakai, dan alur kerja yang biasanya dimaksud orang. Arahkan tool ini ke model foundation seperti DINOv2 atau DINOv3, lalu ke data tanpa label, dan tool akan mendistilasi backbone tersebut menjadi model lebih kecil yang dapat diterapkan (YOLO, RT-DETR, ViT, atau jaringan kustom) hanya dengan beberapa baris kode, tanpa label. Perhatikan lisensinya: AGPL-3.0 memiliki copyleft yang sama dengan alasan orang mencari alternatif Ultralytics. Baca ketentuannya jika Anda berencana merilis produk closed source, atau gunakan lisensi komersial.

Pilih Lightly saat hambatannya adalah tidak adanya label untuk backbone yang bagus. Lightly dapat digunakan bersama detektor dalam daftar ini, bukan sebagai pesaingnya. Model terbaru dalam daftar ini juga menunjukkan hal tersebut: RT-DETRv4 mengintegrasikan distilasi vision foundation model ke dalam pelatihan, sedangkan DEIMv2 menyertakan backbone DINOv3 langsung di dalam detektor.

## 4. YOLOX

Lisensi: Apache-2.0. Terbaik untuk deteksi real-time anchor-free, jika berhasil dipasang.

Repo upstream tidak diperbarui sejak rilis terakhirnya, v0.3.0 pada April 2022. Ada lebih dari 700 issue terbuka, dan repo ini sulit dipasang pada stack 2026. Tidak ada `pyproject.toml`, sementara `requirements.txt` mengunci `onnx-simplifier==0.4.10`, yang tidak menyediakan wheel bawaan untuk Python versi di atas 3.10. Pada interpreter modern, paket harus dibangun dari source sehingga memerlukan cmake dan toolchain C++. Versi NumPy juga tidak dikunci, sehingga berisiko menimbulkan konflik ABI NumPy 2.0 dengan pycocotools dan torch versi lama. Deteksi inti dapat berjalan setelah rantai dependensi dibereskan, tetapi upaya itu menjadi biaya tersendiri.

Modelnya sendiri masih bagus: detektor real-time anchor-free dari Megvii, dengan kode dan bobot berlisensi Apache-2.0, serta menjadi baseline yang masuk akal meski detektor lebih baru kini mengunggulinya. Menghidupkan kembali repo terbengkalai yang masih bagus adalah pekerjaan yang kini bisa ditangani coding agent dalam satu sore, jadi kondisi instalasinya tidak sesulit yang terlihat. LibreYOLO juga menjalankan bobot YOLOX langsung dengan stack terkini. Bagaimanapun caranya, arsitektur ini layak digunakan. Panduan lengkap kami ada [di sini](/articles/yolox-with-libreyolo).

## 5. Ekosistem MMDetection

Lisensi: sebagian besar Apache-2.0. Terbaik untuk ragam arsitektur dan mereproduksi paper.

Selama bertahun-tahun, MMDetection menjadi implementasi resmi bagi hampir setiap paper deteksi: Faster R-CNN, DINO, Grounding-DINO, RTMDet, Mask R-CNN, dan ratusan konfigurasi. Pada 2026, OpenMMLab menghentikan pengembangan lini mm-series. Rilis terakhir [MMDetection](https://github.com/open-mmlab/mmdetection) adalah v3.3.0 pada Januari 2024. Issue tidak dijawab, dan stack terjebak pada pin versi `mmcv`. Wheel bawaan tertinggal dari PyTorch dan CUDA terkini, sehingga instalasi baru biasanya gagal sampai semua paket diturunkan versinya. Kami membahas kendala itu [di sini](/articles/rtmdet-without-mmdetection).

Perusahaan Belanda, VBTI, mempertahankannya. [Fork OneDL](https://github.com/VBTI-development/onedl-mmdetection) mereka menerbitkan ulang seluruh stack dengan awalan `onedl-` (`onedl-mmdetection`, `onedl-mmcv`, `onedl-mmengine`, dan lainnya), yang dibangun ulang untuk PyTorch 2.x, CUDA, dan Python 3.10+ terkini. Dengan begitu, masalah pin versi teratasi. Lisensinya Apache-2.0 dan pemeliharaannya aktif, dengan v3.5.1 pada Mei 2026. Gunakan fork ini untuk mendapatkan cakupan MMDetection tanpa kerepotan instalasi. Timnya kecil, jadi berkontribusilah kembali jika Anda mengandalkannya.

Dua tool terkait:

- **[Detectron2](https://github.com/facebookresearch/detectron2)** (Meta, Apache-2.0) berada dalam mode pemeliharaan, tanpa rilis bertag sejak v0.6 pada 2021. Namun, tool ini andal dan tetap menjadi sumber Mask R-CNN serta keluarga R-CNN terbersih untuk segmentasi instance dan panoptik.
- **[TorchVision](https://github.com/pytorch/vision)** (BSD-3-Clause, dipelihara aktif) menyediakan Faster R-CNN, RetinaNet, FCOS, SSD, Mask R-CNN, dan Keypoint R-CNN. Tidak ada YOLO atau DETR modern, dan Anda harus menulis loop pelatihan sendiri, tetapi ini adalah baseline tanpa dependensi tambahan.

Satu catatan untuk penggunaan komersial: **MMYOLO**, hub implementasi ulang YOLO dari OpenMMLab, berlisensi GPL-3.0, bukan Apache-2.0, dan tidak diperbarui sejak Agustus 2023.

## 6. Keluarga RT-DETR

Lisensi: seluruhnya Apache-2.0. Terbaik untuk deteksi real-time tercanggih dengan kode tingkat riset.

Batas terdepan deteksi real-time saat ini terdiri dari kelompok detektor transformer terkait, bukan YOLO. Sebagian besar berasal dari RT-DETR, menggunakan banyak kode backbone dan encoder yang sama, serta hampir semuanya berlisensi Apache-2.0, sehingga mudah berpindah di antara model tersebut. Kebanyakan hanya mendukung deteksi. Satu pengecualian adalah EdgeCrafter, yang memperluas ide distilasi serupa ke segmentasi dan pose. Kekurangannya sama: ini repo riset dengan pelatihan berbasis file konfigurasi, pengalaman penggunaan yang berbeda dari Ultralytics, dan tanpa kanal dukungan. Modelnya kuat, dan ekspor ONNX serta TensorRT biasanya sudah didukung dengan baik.

- **[D-FINE](https://github.com/Peterande/D-FINE)** (USTC, ICLR 2025 Spotlight). Merumuskan ulang regresi bounding box sebagai penyempurnaan distribusi dengan self-distillation. Akurasi tinggi dibanding latensinya (D-FINE-X mencapai sekitar 55.8 AP pada sekitar 13 ms di T4), tersedia dalam ukuran N sampai X, serta terintegrasi dengan Hugging Face Transformers. Karena itu, model ini paling mudah dimuat dan menjalani fine-tuning di luar repo aslinya.
- **[DEIM dan DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)** (Intellindust AI Lab). DEIM (CVPR 2025) adalah resep pelatihan (pencocokan Dense O2O) yang dapat ditambahkan ke D-FINE atau RT-DETRv2 dan memangkas waktu pelatihan kira-kira separuh. DEIMv2 menambahkan fitur DINOv3 dan tersedia hingga kelas Atto dengan parameter kurang dari 1M untuk perangkat seluler. DEIMv2-S adalah model pertama dengan kurang dari 10M parameter yang melampaui 50 AP. Pemeliharaannya aktif dan diperbarui hingga 2026.
- **[EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter)** (Intellindust AI Lab, 2026). Karya terbaru lab yang sama, dan satu-satunya entri di sini yang melampaui deteksi: cakupannya meliputi deteksi (ECDet), segmentasi instance (ECSeg), dan pose (ECPose), masing-masing dalam ukuran S/M/L/X, semuanya berlisensi Apache-2.0. Model ini mengadaptasi ViT besar yang menjalani pretraining DINOv3 sebagai teacher khusus task, lalu mendistilasinya menjadi backbone student ringkas untuk perangkat edge. Angkanya kuat untuk ukuran ini: ECDet-S mencapai 51.7 AP dengan kurang dari 10M parameter hanya pada COCO, ECPose-X mencapai 74.8 AP (melampaui YOLO26Pose-X pada 71.6), dan segmentasi instance-nya bersaing dengan RF-DETR. Model ini sangat baru, jadi anggap sebagai rilis riset. Namun, model ringkas yang mencakup ketiga task sekaligus seperti ini jarang ditemukan.
- **[RT-DETR dan RT-DETRv2](https://github.com/lyuwenyu/RT-DETR)** (Baidu). Riset awal "DETR mengalahkan YOLO dalam deteksi real-time" (CVPR 2024) dan cikal bakal keluarga ini: tanpa NMS, end-to-end, serta ramah ekspor. v2 adalah pembaruan bag-of-freebies pada 2024, peningkatan langsung dengan latensi yang sama. Versi asli Paddle dan port PyTorch kanonis berbagi repo ini, dan modelnya tersedia di HF Transformers.
- **[RT-DETRv3](https://github.com/clxia12/RT-DETRv3)** (Baidu, WACV 2025 Oral) dan **[RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4)** (Peking University dan Tsinghua, akhir 2025). v3 hanya mendukung PaddlePaddle. v4 adalah versi terbaik saat ini di lini tersebut (X mendekati 57 AP), menggunakan PyTorch, dan bekerja dengan mendistilasi vision foundation model menjadi detektor ringan tanpa menambah biaya inferensi. Basis kodenya juga dapat mereproduksi D-FINE, DEIM, dan RT-DETRv2 melalui perubahan konfigurasi.
- **[LW-DETR](https://github.com/Atten4Vis/LW-DETR)** (Baidu). DETR berbasis ViT biasa yang diperkenalkan sebagai pengganti YOLO berbasis transformer, tersedia dari ukuran tiny hingga xlarge, dengan ekspor ONNX dan TensorRT. Model ini juga menjadi dasar OVLW-DETR untuk open-vocabulary. Aktivitas terbaru menurun, jadi anggap sebagai rilis riset yang stabil.

LibreYOLO sudah membungkus banyak model ini (D-FINE, DEIM, DEIMv2, RT-DETRv2, RT-DETRv4, RTMDet, EdgeCrafter) dalam satu API, dengan YOLO9 dan RF-DETR sebagai model yang paling banyak diuji. Repo sumber di atas tetap menjadi rujukan untuk implementasi dan checkpoint riset terbaru.

## FAQ

**Apakah Ultralytics YOLO gratis untuk penggunaan komersial?**
Hanya di bawah AGPL-3.0, yang mewajibkan Anda merilis source aplikasi yang dibangun di atasnya, termasuk layanan jaringan. Penggunaan komersial closed source memerlukan lisensi komersial berbayar dari Ultralytics. Alternatif permisif (MIT, Apache-2.0) menghindari kewajiban ini.

**Apa alternatif Ultralytics dengan lisensi paling sederhana?**
Untuk stack permisif yang dapat dikirim ke mana saja: LibreYOLO (kode MIT), RF-DETR (Apache-2.0), dan YOLOX (Apache-2.0). Semuanya menghindari copyleft AGPL.

**Apa yang menggantikan YOLO pada 2026?**
Detektor transformer real-time semakin banyak digunakan: RT-DETR, RF-DETR, serta lini D-FINE dan DEIM. Pada hardware yang mumpuni, akurasinya setara atau lebih baik daripada YOLO dengan latensi serupa. CNN bergaya YOLO masih unggul pada perangkat edge kecil, tempat transformer tersebut berjalan kurang baik dan belum memiliki jalur NCNN atau CPU yang matang.

**Bisakah model ini berjalan di Raspberry Pi atau NPU?**
Tergantung ekspornya. Detektor CNN seperti YOLOX dan RTMDet dapat diekspor ke NCNN dan berjalan di Pi, dan beberapa di antaranya (YOLOX) dapat menggunakan NPU Hailo. Detektor transformer seperti RF-DETR tidak bisa, model tersebut memerlukan GPU atau CPU yang kuat.

## Coba

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRl.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

LibreYOLO berlisensi MIT, berjalan di Linux, Mac, dan Windows, serta bekerja di GPU, Apple Silicon, dan CPU biasa tanpa perubahan kode. Satu API mencakup RF-DETR, D-FINE, DEIM, YOLOX, YOLO-NAS, RTMDet, RT-DETR, EdgeCrafter, dan lainnya, untuk deteksi, segmentasi, pose, klasifikasi, kedalaman, gaze, dan tracking.

Beri bintang di GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Dokumentasi: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
