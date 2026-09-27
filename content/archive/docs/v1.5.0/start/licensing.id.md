---
title: Lisensi
seo_title: 'Lisensi LibreYOLO: kode dan bobot'
description: >-
  Kode LibreYOLO sendiri berlisensi MIT. Kode upstream yang disertakan dan
  checkpoint terbitan memiliki lisensi masing-masing, dan beberapa bersifat
  nonkomersial.
lead: >-
  LibreYOLO memuat tiga hal dengan lisensi terpisah: kode sendiri, kode upstream
  yang disertakan ke family model, dan checkpoint pretrained. Ketiganya sering
  kali tidak menggunakan lisensi yang sama.
keywords:
  - lisensi libreyolo
  - library computer vision mit
  - bobot model nonkomersial
  - lisensi checkpoint model
  - deteksi objek apache-2.0
last_verified: 1.5.0
source_hash: 83536fea4dc4eaec
---

## Kode LibreYOLO sendiri

Library berlisensi MIT. Lisensi ini mencakup API Python, CLI, trainer, validator,
exporter, loader dataset, dan script konversi di bawah `weights/`. Kode dapat
digunakan dalam produk komersial atau closed-sumber. Pertahankan baris copyright
dan teks lisensi bersama setiap salinan yang didistribusikan ulang, dan kewajiban
berakhir di sana.

Izin tersebut berhenti pada kode. Berkas
[`LICENSE`](https://github.com/LibreYOLO/libreyolo/blob/release/LICENSE)
menyatakannya dengan jelas:

> Lisensi tersebut beragam dan tidak semuanya permisif: beberapa bobot yang diterbitkan
> bersifat nonkomersial atau dibatasi dengan cara lain, dan Lisensi MIT ini tidak mencakup
> bobot tersebut. Memilih model berarti memilih lisensinya.

## Kode upstream per family

Sebagian besar family merupakan port penelitian terbitan dan beberapa
menyertakan sumber upstream secara langsung. Berkas yang disertakan mempertahankan
header copyright dan lisensi aslinya. MIT tidak menimpanya, dan LibreYOLO tidak
melisensikan ulang karya pihak lain. Apache-2.0 dan BSD-3-Clause adalah dua yang
paling sering muncul.

Apache-2.0 mencakup lineage DETR dan banyak karya transformer: DETR dari Meta AI
(FAIR), Deformable DETR dari SenseTime, LW-DETR dari Baidu, OV-DEIM oleh Leilei
Wang dan rekan penulis, implementasi SegFormer yang diadaptasi LibreYOLO dari
Hugging Face Transformers, PP-OCRv5 dari PaddlePaddle Authors, SwinIR dari
Computer Vision Lab di ETH Zurich, serta Depth Anything 3 dari ByteDance Seed.
Lisensi ini juga mencakup classifier turunan timm karya Ross Wightman dan
kontributor timm, termasuk ResNet, DeiT, EfficientNetV2, MobileNetV4, dan Swin,
yang nama modulnya mencerminkan timm agar tensor ImageNet dapat dimuat tanpa
perubahan.

BSD-3-Clause mencakup semua turunan torchvision: Faster R-CNN, Mask R-CNN,
FCOS, RetinaNet, SSD300, AlexNet, VGG, FCN, dan DeepLabv3.

MIT mencakup kelompok lebih kecil, termasuk NAFNet dari Megvii, CenterNet dari
Xingyi Zhou, dan YOLOv7 sebagaimana dirilis ulang oleh penulisnya, Kin-Yiu Wong
dan Hao-Tang Tsui, di MultimediaTechLab. Family YOLOv1 hingga YOLOv4
mereproduksi arsitektur dari project Darknet oleh Joseph Redmon dan, untuk
YOLOv4, Alexey Bochkovskiy. Darknet berada di domain publik, sehingga tidak
menimbulkan kewajiban.

Satu subtree bawaan tidak memakai lisensi open-sumber. Family DEIMv2 menyertakan
kode backbone DINOv3 dari Meta Platforms berdasarkan DINOv3 License Agreement,
lisensi non-OSI khusus. Distribusi ulang kode tersebut harus menyertakan salinan
perjanjian, dan perjanjian melarang penggunaan untuk aktivitas yang tunduk pada
ITAR, tujuan militer atau peperangan, industri nuklir, spionase, dan
pengembangan senjata. Ketentuan tersebut hanya mengikat subtree itu.

Dua berkas dalam repositori memuat gambaran lengkap.
[`NOTICE`](https://github.com/LibreYOLO/libreyolo/blob/release/NOTICE)
mencantumkan setiap subtree pihak ketiga beserta path, berkas lisensi, dan sumber
upstream.
[`THIRD_PARTY_NOTICES.txt`](https://github.com/LibreYOLO/libreyolo/blob/release/THIRD_PARTY_NOTICES.txt)
mencantumkan project upstream asal LibreYOLO dan mereproduksi setiap teks lisensi
secara lengkap.

## Bobot per checkpoint

Tidak ada berkas bobot pretrained yang disertakan dalam paket. Checkpoint
terbitan berada di Hugging Face di bawah [organisasi
LibreYOLO](https://huggingface.co/LibreYOLO), dan setiap repositori memiliki
`LICENSE` serta atribusi sendiri yang mencerminkan project asal bobot.

Repositori tersebut adalah sumber otoritatif untuk ketentuannya. Bukan halaman
ini, halaman model, atau ringkasan dalam sumber tree. Lihat
[checkpoint dan bobot](/docs/weights) untuk cara berkas dinamai dan lokasi
pengunduhannya.

Lisensi berbeda antar-family dan antar-berkas dalam satu family. Dua contoh kasus
kedua:

- Checkpoint COCO YOLO9 berlisensi MIT. `LibreYOLO9P2s-visdrone.pt`, yang dilatih pada VisDrone2019-DET, berlisensi CC BY-NC-SA 3.0 dan bersifat nonkomersial.
- Checkpoint deteksi RF-DETR berlisensi Apache-2.0. Checkpoint oriented-bounding box berlisensi CC BY 4.0 karena mendapat fine-tuning pada dataset Roboflow Universe yang berlisensi CC BY 4.0, dan bobot meneruskan kewajiban atribusi dataset tersebut.

Rentang antar-family lebih luas, dan beberapa checkpoint terbitan tidak dapat
digunakan dalam produk komersial:

- SegFormer menunjukkan pemisahan dua lapisan dengan paling jelas. Implementasinya adalah port Apache-2.0 dari kode Hugging Face Transformers. Checkpoint ADE20K terbitan dikonversi dari rilis NVIDIA berdasarkan NVIDIA Source Code License, yang mengizinkan distribusi ulang tetapi membatasi penggunaan pada penelitian atau evaluasi nonkomersial, serta meneruskan batasan tersebut ke karya turunan. Checkpoint itu tidak dicakup ketentuan permisif LibreYOLO.
- Checkpoint OV-DEIM berlisensi CC BY-NC 4.0, yang dikonfirmasi penulis upstream. Setiap prediksi juga memuat text tower MobileCLIP-B(LT) milik Apple, yang lisensinya membatasi penggunaan pada penelitian, lebih ketat daripada lisensi checkpoint.
- Kode SenseNova-Vision berlisensi Apache-2.0 dan bobotnya CC BY-NC 4.0. Loader mencetak pemberitahuan nonkomersial sebelum setiap pengunduhan otomatis.

Beberapa family sama sekali tidak memiliki checkpoint yang disediakan oleh LibreYOLO,
dan halamannya menyatakan demikian pada baris Bobot. SAM 3 bersifat gated di
Hugging Face berdasarkan SAM License khusus Meta dan diunduh langsung dari Meta.
Aset rilis MiDaS diambil dari URL resmi serta diverifikasi hash-nya, bukan
dihosting ulang. Dome-DETR ditautkan ke upstream karena metadata model card-nya
tidak menyatakan lisensi, sementara prosanya mengklaim Apache-2.0 sekaligus
membatasi penggunaan ke penelitian akademik, dan kedua pernyataan tidak cocok.
Arsitektur TEED dan DexiNed berlisensi MIT, tetapi checkpoint penulis dilatih
pada BIPED dengan ketentuan dataset nonkomersial, sehingga LibreYOLO tidak
menyertakan atau mengunduhnya otomatis.

Beberapa checkpoint torchvision tidak memiliki berkas lisensi sendiri. LibreYOLO
mencerminkannya dengan lisensi yang digunakan project penerbit, menyatakan pada
setiap model card bahwa dasarnya tersirat, bukan diberikan per checkpoint, dan
mengulangi peringatan torchvision bahwa ketentuan model pretrained dapat berasal
dari data pelatihan.

## Menemukan ketentuan untuk satu model

Halaman model memiliki baris **Licenses** pada header dalam bentuk
`Code X, weights Y`, yang menautkan ke bagian Licensing halaman. Bagian tersebut
mencantumkan karya asli dan penulisnya, lisensi upstream, sumber upstream,
lisensi kode LibreYOLO, bobot, serta interpretasi hal yang diizinkan.
Tabel Checkpoint pada halaman yang sama memiliki kolom **Lisensi bobot**,
satu baris per berkas terbitan, sehingga family dengan ketentuan campuran
menampilkannya per berkas.

Semuanya dirender dari data yang sama dengan yang diperiksa library, sehingga
halaman ini tidak mengulanginya sebagai tabel. Matriks lisensi yang diketik
manual akan salah dalam satu rilis, dan kesalahan di sini berbiaya besar.

Dalam sumber tree, padanannya adalah `NOTICE` untuk kode bawaan,
`THIRD_PARTY_NOTICES.txt` untuk project upstream dan teks lisensinya, serta
[`weights/LICENSE_NOTICE.txt`](https://github.com/LibreYOLO/libreyolo/blob/release/weights/LICENSE_NOTICE.txt)
untuk ringkasan per family dari checkpoint terbitan.

Kemudian, periksa repositori Hugging Face dari berkas persis yang akan diunduh.
Repositori tersebut otoritatif dan dapat berubah tanpa perubahan halaman
dokumentasi.

## Penggunaan komersial

Kode jarang menjadi masalah. MIT, Apache-2.0, dan BSD-3-Clause semuanya
mengizinkan penggunaan komersial dan closed-sumber. Masing-masing meminta teks
lisensi dan pemberitahuan atribusi dipertahankan bersama salinan yang
didistribusikan ulang, Apache-2.0 juga memberikan lisensi paten, dan tidak ada
yang memberi syarat pada kode aplikasi sendiri.

Checkpoint adalah tempat produk sering terhambat. Checkpoint nonkomersial tetap
nonkomersial meskipun kode di sekitarnya permisif, dan konversi berkas tidak
mengubah ketentuan yang berlaku. Hal ini dinyatakan langsung dalam
`weights/LICENSE_NOTICE.txt`. Artefak ONNX atau TensorRT yang dibangun dari
checkpoint terbatas mewarisi batasan tersebut.

Jika lisensi meneruskan batasan ke karya turunan, seperti NVIDIA Sumber Code
License, fine-tuning juga tidak menghindarinya. Melatih arsitektur yang sama dari
awal dengan data yang berhak digunakan dapat menghindarinya: kode bersifat
permisif, sehingga model hasil pelatihan sendiri menjadi milik pengguna dan
ketentuan checkpoint pretrained tidak pernah masuk. Halaman SegFormer
menjelaskannya untuk bobotnya; baca baris Interpretation pada halaman family
yang akan dikirim.

Tentukan lisensi saat memilih model, bukan saat akan merilis produk, dan baca
ketentuan berkas yang benar-benar diunduh karena family dengan satu checkpoint
permisif dapat memiliki checkpoint terbatas di sampingnya.

## Bukan nasihat hukum

Halaman ini menjelaskan lisensi yang terlibat. Penjelasan ini bukan nasihat
hukum dan tidak menciptakan jaminan apa pun. Jika jawabannya penting secara
komersial, baca lisensi sendiri dan minta nasihat hukum.



