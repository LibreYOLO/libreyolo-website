---
title: "Alternatif SuperGradients untuk Computer Vision Real-Time: Jalankan YOLO-NAS dengan LibreYOLO"
description: "SuperGradients berhenti merilis pembaruan setelah NVIDIA mengakuisisi Deci pada 2024. LibreYOLO adalah alternatif terpelihara yang menjalankan bobot YOLO-NAS yang sama: lakukan prediksi, pelatihan, dan ekspor melalui satu API berlisensi MIT."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, supergradients-alternative, yolo-nas, object-detection]
faq:
  - q: "Apakah SuperGradients masih dipelihara?"
    a: "Tidak. Rilis terakhirnya adalah 3.7.1 pada April 2024, tepat sebelum NVIDIA mengakuisisi Deci. Sejak itu, repo tersebut tidak merilis apa pun, issue tidak dijawab, dan situs dokumentasinya offline. Repo ini tidak resmi diarsipkan, tetapi sudah tidak ada yang mengurusnya."
  - q: "Apa alternatif terbaik untuk SuperGradients?"
    a: "Untuk menjalankan YOLO-NAS, LibreYOLO memuat bobot Deci yang sama melalui API berlisensi MIT yang terus dipelihara, serta menambahkan pelatihan, validasi, dan ekspor. Jika Anda bergantung pada resep pelatihan SuperGradients untuk arsitektur lain, repo lama itu masih bisa dijalankan setelah dependensinya dipatok versinya."
  - q: "Bisakah saya menggunakan YOLO-NAS secara komersial?"
    a: "Bobot pretrained Deci bersifat nonkomersial, apa pun library yang memuatnya. Arsitekturnya sendiri berlisensi permisif, sehingga melatih YOLO-NAS dari nol dengan data sendiri menghasilkan model yang tidak memiliki checkpoint Deci dalam riwayatnya."
  - q: "Apakah LibreYOLO menggantikan semua fungsi SuperGradients?"
    a: "Tidak semuanya. LibreYOLO belum mendukung ekspor YOLO-NAS ke CoreML, dan jalur TensorRT-nya untuk keluarga ini belum diuji. Resep pelatihan yang memperhitungkan kuantisasi dari SuperGradients juga tidak memiliki padanan langsung. Untuk kebutuhan itu, pertahankan repo lama."
---

SuperGradients adalah library pelatihan open source dari Deci dan tempat YOLO-NAS dikembangkan, salah satu detektor real-time paling akurat yang pernah dirilis. Pada Mei 2024, NVIDIA mengakuisisi Deci dan SuperGradients berhenti aktif. Rilis terakhirnya, 3.7.1, terbit pada 8 April 2024. Dokumentasi di supergradients.com offline. Sekitar 120 issue masih terbuka, dan issue berjudul ["Apakah proyek ini sudah mati?"](https://github.com/Deci-AI/super-gradients/issues/2062) tidak mendapat jawaban dari maintainer. Itu sendiri sudah menjadi jawaban.

Repo ini belum diarsipkan, jadi sekilas tampak masih aktif. Namun, dampak ditinggalkan langsung terasa saat mencoba memasangnya: `super-gradients` mematok `torchmetrics==0.8`, yang bertentangan dengan stack PyTorch saat ini, serta menarik hydra, omegaconf, boto3, dan tensorboard sebagai dependensi. Setiap bulan, versi yang dipatok semakin sulit diselaraskan dengan bagian lain dari environment Anda, dan tidak akan ada perbaikannya.

Semua ini tidak membuat YOLO-NAS menjadi model yang lebih buruk. Varian besar masih mencatat 52.2 mAP pada COCO dengan kecepatan real-time. Modelnya baik-baik saja, rumahnya yang sudah terbakar.

<iframe
  src="https://visionanalysis.org/embed/scatter?highlight=yolonas-s%2Cyolonas-m%2Cyolonas-l"
  width="100%"
  height="420"
  style="border:0;border-radius:12px;overflow:hidden"
  loading="lazy"
  title="YOLO-NAS accuracy vs parameters - visionanalysis.org">
</iframe>

## Jalankan bobot yang sama dengan LibreYOLO

LibreYOLO memuat checkpoint YOLO-NAS langsung dari CDN Deci, melalui API yang sama dengan yang digunakan untuk setiap keluarga model lainnya. Tidak perlu menulis konfigurasi hydra atau membungkus ulang prediksi:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")  # auto-downloads on first run
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

Semua varian deteksi S, M, dan L dapat digunakan, begitu juga pose: ganti dengan `LibreYOLONASs-pose.pt` untuk mendapatkan keypoint COCO. Karena setiap keluarga model mengembalikan objek `Results` yang sama, membandingkan YOLO-NAS dengan RF-DETR atau D-FINE pada data sendiri hanya perlu mengubah satu baris, alih-alih memakai basis kode kedua.

## Pelatihan dan ekspor, bukan hanya inferensi

SuperGradients sejak awal adalah library pelatihan, jadi alternatif yang sesungguhnya juga harus bisa melatih model. LibreYOLO melakukan fine-tuning YOLO-NAS pada dataset Anda dengan pemanggilan yang sama seperti model lainnya:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
```

Anda juga dapat melatih model dengan inisialisasi bobot acak, yang penting untuk lisensi (lihat penjelasan di bawah). Validasi mengembalikan metrik mAP berdasarkan dataset dalam format apa pun yang didukung, dan ekspor mendukung ONNX, TorchScript, OpenVINO, NCNN, dan TFLite. Cakupan format ekspornya lebih luas daripada yang pernah disediakan library aslinya untuk YOLO-NAS. Detail lengkap tersedia di [halaman dokumentasi YOLO-NAS](https://www.libreyolo.com/docs/models/yolo-nas), dan ada catatan singkat bahwa [YOLO-NAS masih dipelihara](/articles/yolo-nas-with-libreyolo).

## Saat repo lama masih menjadi pilihan yang tepat

Bagian apa adanya. Ada tiga kondisi ketika Anda sebaiknya tetap memasang SuperGradients:

**CoreML.** LibreYOLO belum mengekspor YOLO-NAS ke CoreML. Jika Anda merilis produk di perangkat Apple dan memerlukan `.mlpackage`, SuperGradients masih memiliki jalur yang berfungsi.

**TensorRT.** SuperGradients mendokumentasikan dan menguji alur TensorRT untuk YOLO-NAS, termasuk semua keanehan ukuran batch. Dukungan TensorRT LibreYOLO untuk keluarga ini belum diuji.

**Resep pelatihan yang memperhitungkan kuantisasi.** YOLO-NAS dirancang untuk INT8, dan SuperGradients menyediakan resep QAT yang membuatnya unggul dalam penggunaan tersebut. LibreYOLO mengekspor dengan kuantisasi INT8 dan FP16, tetapi tidak memiliki padanan untuk resep pelatihan itu.

Repo ini masih dapat dijalankan jika Anda mematok environment era 2024 di sekelilingnya. Namun, sebaiknya jangan membangun hal baru di atas fondasi yang tidak dipelihara siapa pun.

## Catatan tentang bobot

Bobot YOLO-NAS pretrained berasal dari Deci, dirilis dengan lisensi nonkomersial, dan lisensi tersebut tetap melekat pada bobot di library mana pun yang memuatnya. LibreYOLO tidak meng-host atau mencerminkan bobot tersebut. Unduhan berasal dari CDN publik Deci dan ketentuan Deci ditampilkan sebelum unduhan dimulai. Untuk riset dan penggunaan nonkomersial, bobot tersebut dapat digunakan melalui library mana pun.

Jika memerlukan YOLO-NAS untuk penggunaan komersial, kini ada cara yang jelas: arsitekturnya sendiri berlisensi permisif, sehingga pelatihan dari nol dengan data sendiri menghasilkan model yang tidak berasal dari checkpoint Deci. LibreYOLO mendukung cara tersebut melalui `LibreYOLONAS(None, size="s")`.

## Coba

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASl.pt")
results = model("image.jpg", save=True)
```

LibreYOLO berlisensi MIT, berjalan di Linux, Mac, dan Windows, serta berfungsi di GPU, Apple Silicon, dan CPU biasa tanpa perubahan kode. Satu API mendukung YOLO-NAS, RF-DETR, D-FINE, DEIM, YOLOX, RTMDet, dan banyak lagi, untuk deteksi, segmentasi, pose, klasifikasi, kedalaman, dan tracking.

Beri bintang di GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Dokumentasi: [libreyolo.com/docs](https://www.libreyolo.com/docs)
