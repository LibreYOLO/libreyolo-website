---
title: Cara Termudah Menjalankan Depth Anything V2 dengan LibreYOLO
description: Depth Anything V2 menghasilkan peta kedalaman monokular mutakhir. Berikut cara menjalankannya dengan LibreYOLO hanya dalam dua baris.
date: 2026-06-26
author: Xuban
tags: [LibreYOLO, depth-anything-v2, depth-estimation, tutorial]
faq:
  - q: "Apa cara termudah menjalankan Depth Anything V2?"
    a: "Dua baris dengan LibreYOLO: muat LibreDepthAnythingV2l-depth.pt berdasarkan namanya, lalu panggil predict dengan save=True. Bobot diunduh otomatis saat pertama kali digunakan, sedangkan normalisasi, colormap, dan penempatan perangkat ditangani untuk Anda, baik di CUDA, Apple Silicon, maupun CPU biasa."
  - q: "Bisakah Depth Anything V2 digunakan secara komersial?"
    a: "Hanya encoder Small yang menggunakan Apache 2.0. Base, Large, dan Giant menggunakan CC-BY-NC-4.0, jadi checkpoint terkuat hanya untuk penggunaan nonkomersial. Lisensi upstream tetap berlaku, apa pun library yang memuat bobotnya."
  - q: "Apakah LibreYOLO mendukung pelatihan Depth Anything V2?"
    a: "Tidak. Pelatihan tetap dilakukan di repositori asli; LibreYOLO mendukung inferensi, video, dan validasi untuk task kedalaman."
---

![Guggenheim Bilbao di samping peta kedalamannya](/articles/simplest-way-to-run-depth-anything-v2/guggenheim-bilbao-input-vs-depth.jpg)

Depth Anything V2 menghasilkan peta kedalaman monokular terbaik yang tersedia saat ini.

Jika terbiasa dengan YOLO, repositori resmi bukan pilihan yang paling mudah untuk mulai digunakan. Untuk menghasilkan satu peta kedalaman saja, ada beberapa langkah manual:

* unduh checkpoint yang tepat secara manual dan letakkan di folder yang sesuai,

* sesuaikan encoder dengan konfigurasinya (`encoder="vitl"`, `features=256`, `out_channels=...`),

* tulis sendiri langkah normalisasi dan pewarnaan,

* atur penempatan perangkat agar dapat berjalan di Mac atau CPU, bukan hanya CUDA,

* lalu pelajari API inferensi yang sangat berbeda dari bagian lain di stack Anda.

Tidak ada yang sulit. Langkah-langkah ini hanya menambah hambatan, dan semuanya bisa dilewati.

LibreYOLO memuat bobot Depth Anything V2 yang sama dan menyediakan task kedalaman melalui satu pemanggilan `predict()`. Sebutkan nama model, lalu bobot akan diunduh saat pertama kali digunakan:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreDepthAnythingV2l-depth.pt")  # auto-downloads on first run
model.predict("image.jpg", save=True)     # writes a colorized depth map to disk
```

Itu saja. Jika pernah menggunakan API YOLO standar, polanya akan terasa familier: muat model berdasarkan nama, panggil `predict`, lalu biarkan `save=True` menyimpan visualisasinya. Pemanggilannya sama persis dengan yang digunakan untuk deteksi objek, hanya saja hasilnya berisi peta kedalaman, bukan bounding box. Colormap, normalisasi, dan penempatan perangkat ditangani untuk Anda. Model berjalan dengan cara yang sama di Linux dengan CUDA, di Mac dengan Apple Silicon, atau di CPU biasa. Tidak perlu mengubah kode.

Dari sini, pemanggilan yang sama juga memungkinkan Anda mengambil nilai kedalaman mentah untuk digunakan secara langsung. Pelatihan Depth Anything V2 tetap dilakukan di repositori aslinya; LibreYOLO mendukung inferensi, video, dan validasi.

Pemanggilan satu baris yang sama, pada adegan yang sangat berbeda:

![Gambar sampel parkour di samping peta kedalamannya: para pelompat di depan tampak menonjol dari dinding beton di belakang.](/articles/simplest-way-to-run-depth-anything-v2/parkour-input-vs-depth.jpg)

![Pemandangan udara teluk La Concha di Donostia di samping peta kedalamannya: perahu dan garis pantai tampak dekat, sementara laut terbuka terlihat menjauh.](/articles/simplest-way-to-run-depth-anything-v2/donostia-bay-input-vs-depth.jpg)

![Halaman bertiang di Casa de Juntas de Gernika di samping peta kedalamannya, yang memperlihatkan arsitektur yang menjauh.](/articles/simplest-way-to-run-depth-anything-v2/gernika-casa-juntas-input-vs-depth.jpg)

![Kerumunan festival malam di Plaza de la Constitucion, Donostia, di samping peta kedalamannya. Barisan orang di dekat tampak menonjol dari fasad bercahaya di belakang.](/articles/simplest-way-to-run-depth-anything-v2/donostia-plaza-crowd-input-vs-depth.jpg)

Catatan tentang bobot: LibreYOLO menyediakan checkpoint Depth Anything V2 yang telah dikonversi dan mengunduhnya saat pertama kali digunakan, jadi tidak perlu mengunduhnya secara manual. Lisensi upstream tetap berlaku: encoder Small menggunakan Apache-2.0, sedangkan Base, Large, dan Giant menggunakan CC-BY-NC-4.0 (nonkomersial), sehingga checkpoint terkuat hanya untuk penggunaan nonkomersial. Ingin tetap offline atau mengonversi checkpoint sendiri? Skrip konversi satu kali tetap tersedia:

```bash
# optional: convert an official checkpoint yourself instead of auto-downloading
python weights/convert_depth_anything_v2_weights.py \
  depth_anything_v2_vitl.pth weights/LibreDepthAnythingV2l-depth.pt
```

## Coba

```bash
pip install libreyolo
```

LibreYOLO adalah library computer vision paling lengkap yang dapat dipasang dengan pip. Satu API familier mencakup daftar model mutakhir yang terus bertambah: deteksi objek (RF-DETR, D-FINE, DEIM), segmentasi, pose, bounding box berorientasi, dan kini kedalaman monokular dengan Depth Anything V2, serta pelatihan dan validasi untuk task yang mendukungnya. LibreYOLO berjalan di semua sistem operasi utama, dengan GPU maupun CPU, dan sepenuhnya berlisensi MIT, sehingga dapat digunakan untuk produk komersial tanpa syarat tambahan.

Beri bintang di GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Dokumentasi: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
