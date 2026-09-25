---
title: "Lisensi YOLO untuk Penggunaan Komersial: Panduan YOLOv1 hingga YOLO26 (2026)"
description: "Peta lengkap lisensi YOLO tahun 2026: YOLOv1 hingga YOLOv13, YOLO26, YOLO-World dan YOLOE, repo asli dan implementasi ulang permisif, tautan paper dan GitHub, serta model yang dapat dikirim dalam produk komersial."
date: 2026-07-11
author: Xuban
tags: [LibreYOLO, yolo-license, agpl, gpl, apache, mit-license, yolov9, yolov10, yolo11, yolov12, yolov13, yolo26]
faq:
  - q: "Apakah YOLOv9 gratis untuk penggunaan komersial?"
    a: "Tergantung repo yang digunakan. Repo paper (WongKinYiu/yolov9) berlisensi GPL-3.0. Lab yang sama juga merilis implementasi YOLOv9 dan YOLOv7 terpisah berlisensi MIT di MultimediaTechLab/YOLO, yang ditulis setelah pengguna komersial meminta opsi permisif. Ini adalah penulisan ulang, bukan pemberian lisensi ulang untuk file GPL. Lisensi kodenya jelas; bobot pretrained tidak memiliki pernyataan lisensi tersendiri di kedua repo, jadi asal bobot perlu diperiksa sebagai persoalan terpisah."
  - q: "Versi YOLO mana yang gratis untuk penggunaan komersial closed source?"
    a: "Setidaknya melalui satu repo permisif: YOLOv1 hingga YOLOv4 (Darknet asli, domain publik), YOLOv7 dan YOLOv9 (penulisan ulang MIT oleh labnya), YOLOX dan PP-YOLOE (Apache-2.0), serta implementasi ulang MIT di LibreYOLO. YOLOv5, v6, v8, v10, YOLO11, v12, v13, YOLO26, YOLO-World dan YOLOE belum memiliki implementasi permisif per Juli 2026. Periksa sendiri file LICENSE terbaru sebelum mengandalkannya: lisensi dapat berubah, dan ini adalah informasi umum, bukan nasihat hukum."
  - q: "Apa lisensi YOLOv10?"
    a: "AGPL-3.0. YOLOv10 adalah rilis akademis dari Tsinghua University, tetapi kodenya dibangun di atas codebase Ultralytics, sehingga mewarisi AGPL-3.0. Tidak ada implementasi ulang permisif."
  - q: "Apa lisensi YOLOv12 dan YOLOv13?"
    a: "Keduanya AGPL-3.0, sebagaimana dikonfirmasi oleh file LICENSE masing-masing. Seperti YOLOv10, keduanya adalah paper akademis dengan kode referensi yang dibangun di atas repo Ultralytics, sehingga AGPL tetap berlaku terlepas dari siapa penulis papernya. Halaman pihak ketiga yang mencantumkan YOLOv13 sebagai Apache-2.0 itu keliru."
  - q: "Kapan Ultralytics mengubah lisensi YOLO menjadi AGPL-3.0?"
    a: "Pada 14 April 2023, bukan pada 2022 seperti yang banyak diulang. File LICENSE di ultralytics/yolov5 hanya pernah diubah tiga kali, dan commit AGPL (34cf749, PR #11359) bertanggal 2023-04-14. Rilis terakhir tahun 2022, YOLOv5 v7.0, masih menggunakan GPL-3.0. Repo ultralytics/ultralytics diberi lisensi ulang pada hari yang sama. Karena itu, YOLOv8 diluncurkan pada Januari 2023 dengan GPL-3.0: rilis PyPI 8.0.0 hingga 8.0.76 menyatakan GPL-3.0, sedangkan 8.0.80 (16 April 2023) adalah versi pertama yang menyatakan AGPL-3.0."
  - q: "Apakah YOLO26 gratis untuk penggunaan komersial?"
    a: "Tidak untuk produk closed source. YOLO26 dirilis dengan AGPL-3.0 dan Enterprise License berbayar sebagai alternatif, dengan ketentuan yang sama seperti YOLOv8 dan YOLO11."
  - q: "Benarkah YOLO Darknet asli berada di domain publik?"
    a: "Ya. LICENSE Darknet milik Joseph Redmon menyatakan 'Darknet is public domain. Do whatever you want with it.' Fork YOLOv4 milik AlexeyAB memuat teks domain publik yang sama. Namun, port PyTorch yang benar-benar digunakan orang memiliki lisensinya sendiri, mulai dari Apache-2.0, AGPL-3.0 hingga tanpa lisensi sama sekali."
  - q: "Bisakah bobot pretrained YOLO-NAS digunakan secara komersial?"
    a: "Tidak. Kode super-gradients berlisensi Apache-2.0, tetapi bobot resmi YOLO-NAS dirilis dengan lisensi terpisah yang menyatakan bahwa Anda 'may not use the Software for any commercial use, including in connection with any models used in a production environment.'"
  - q: "Apakah bobot jaringan saraf mewarisi lisensi kode pelatihannya?"
    a: "Belum ada kepastian. Ultralytics menyatakan AGPL-3.0 mencakup 'the training code and the models produced by that training code', dan sebagai pemegang hak cipta mereka menetapkan ketentuan untuk materi yang mereka terbitkan. Belum pernah diuji di pengadilan apakah bobot yang Anda latih sendiri merupakan karya turunan dari kode pelatihan. Perlakukan bobot AGPL yang diterbitkan sebagai materi yang terkena pembatasan lisensi, dan berhati-hatilah saat memuatnya ke implementasi ulang permisif."
  - q: "Mengapa implementasi berbeda dari YOLO yang sama bisa memiliki lisensi berbeda?"
    a: "Hak cipta melindungi ekspresi, bukan gagasan (17 U.S.C. 102(b)). Arsitektur yang dijelaskan dalam paper dapat diimplementasikan secara mandiri dan dilisensikan sesuka penulisnya. Yang tidak boleh dilakukan adalah menyalin file kode sumber GPL atau AGPL lalu memberi lisensi ulang. Perhatikan bahwa ini hanya argumen hak cipta: implementasi mandiri bukan pembelaan terhadap paten."
---

Setiap tahun jumlah YOLO bertambah, dan setiap tahun pertanyaan soal lisensi muncul lagi, biasanya lima menit sebelum keputusan produk dibuat. Bagian yang membingungkan adalah "YOLO" bukan satu proyek. Itu nama merek yang digunakan bersama oleh sekitar dua puluh keluarga model dari penulis berbeda. Ada satu detail yang luput dari kebanyakan panduan lisensi: **lisensi melekat pada repo, bukan pada model.** Versi YOLO yang sama dapat tersedia sekaligus di repo GPL dan repo MIT, bahkan dari penulis yang sama. YOLOv9 adalah contohnya, dan hal itu mengubah seluruh keputusan.

Panduan ini memetakan lanskap repo demi repo, dengan tautan ke setiap paper dan codebase, berdasarkan informasi hingga Juli 2026. Setiap lisensi di bawah ini telah diperiksa pada file LICENSE di repo terkait, karena cukup banyak informasi tentang lisensi YOLO yang beredar (termasuk di panduan dengan peringkat tertinggi) ternyata keliru.

Jika yang ingin diketahui hanya soal AGPL Ultralytics, kami membahasnya secara mendalam di [Apakah YOLO Gratis untuk Penggunaan Komersial?](/articles/yolo-commercial-license). Artikel ini adalah peta lengkapnya.

**Dua hal perlu diungkapkan sebelum mulai.** Pertama, kami mengelola LibreYOLO, library berlisensi MIT yang bersaing dengan beberapa proyek di bawah ini, dan bagian terakhir artikel ini membahasnya. Itu alasan untuk memeriksa pekerjaan kami, bukan langsung memercayainya. Karena itu, setiap klaim lisensi di sini ditautkan ke sumber utama. Kedua, artikel ini berisi informasi umum tentang teks lisensi dan pernyataan publik yang diterbitkan, berdasarkan informasi hingga tanggal di atas. Ini bukan nasihat hukum dan tidak menggantikan nasihat pengacara di yurisdiksi Anda. Lisensi dan sikap maintainer dapat berubah. Baca file LICENSE terbaru dari setiap repo sebelum mengambil keputusan untuk merilis produk.

## Tabel lisensi

"Jalur permisif" berarti: repo yang mengimplementasikan model ini dan dapat digunakan dalam produk komersial closed source tanpa membuka kode sumber aplikasi atau membayar lisensi.

| Model | Tahun | Kode asli | Jalur permisif | Paper |
| --- | --- | --- | --- | --- |
| YOLOv1 | 2015 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (domain publik) | Versi aslinya | [arXiv](https://arxiv.org/abs/1506.02640) |
| YOLOv2 | 2016 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (domain publik) | Versi aslinya | [arXiv](https://arxiv.org/abs/1612.08242) |
| YOLOv3 | 2018 | [pjreddie/darknet](https://github.com/pjreddie/darknet) (domain publik) | Versi asli, waspadai port PyTorch berlisensi AGPL | [arXiv](https://arxiv.org/abs/1804.02767) |
| YOLOv4 | 2020 | [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) (domain publik) | Versi asli; [pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) (Apache-2.0) | [arXiv](https://arxiv.org/abs/2004.10934) |
| YOLOv5 | 2020 | [ultralytics/yolov5](https://github.com/ultralytics/yolov5) (AGPL-3.0) | Tidak ada | tanpa paper |
| YOLOv6 | 2022 | [meituan/YOLOv6](https://github.com/meituan/YOLOv6) (GPL-3.0) | Tidak ada | [arXiv](https://arxiv.org/abs/2209.02976) |
| YOLOv7 | 2022 | [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7) (GPL-3.0) | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) (MIT, lab yang sama) | [arXiv](https://arxiv.org/abs/2207.02696) |
| YOLOv8 | 2023 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0; GPL-3.0 saat diluncurkan, lihat di bawah) | Tidak ada (port Keras dihentikan, lihat di bawah) | tanpa paper |
| YOLOv9 | 2024 | [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) (GPL-3.0) | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) (MIT, lab yang sama) | [arXiv](https://arxiv.org/abs/2402.13616) |
| YOLOv10 | 2024 | [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10) (AGPL-3.0) | Tidak ada | [arXiv](https://arxiv.org/abs/2405.14458) |
| YOLO11 | 2024 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) | Tidak ada | tanpa paper |
| YOLOv12 | 2025 | [sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12) (AGPL-3.0) | Tidak ada | [arXiv](https://arxiv.org/abs/2502.12524) |
| YOLOv13 | 2025 | [iMoonLab/yolov13](https://github.com/iMoonLab/yolov13) (AGPL-3.0) | Tidak ada | [arXiv](https://arxiv.org/abs/2506.17733) |
| YOLO26 | 2026 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0) | Tidak ada | [arXiv](https://arxiv.org/abs/2606.03748) |
| YOLOX | 2021 | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0) | Versi aslinya | [arXiv](https://arxiv.org/abs/2107.08430) |
| PP-YOLOE | 2022 | [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) (Apache-2.0) | Versi asli dari PaddleDetection (bukan PaddleYOLO) | [arXiv](https://arxiv.org/abs/2203.16250) |
| YOLO-NAS | 2023 | [Deci-AI/super-gradients](https://github.com/Deci-AI/super-gradients) (kode Apache-2.0, bobot nonkomersial) | Kodenya bisa, bobot resminya tidak | tanpa paper |
| YOLO-World | 2024 | [AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World) (GPL-3.0) | Tidak ada | [arXiv](https://arxiv.org/abs/2401.17270) |
| YOLOE | 2025 | [THU-MIG/yoloe](https://github.com/THU-MIG/yoloe) (AGPL-3.0) | Tidak ada | [arXiv](https://arxiv.org/abs/2503.07465) |
| MMYOLO | 2022 | [open-mmlab/mmyolo](https://github.com/open-mmlab/mmyolo) (GPL-3.0, tidak aktif sejak 2024) | Tidak ada | tanpa paper |

Baca tabel ini per kolom, dan satu hal segera terlihat: nomor versi tidak memberi tahu apa pun tentang lisensinya. Repo-lah yang menentukan.

## Model sama, dua lisensi: kasus YOLOv9

YOLOv9 adalah bukti paling jelas bahwa pertanyaan "apa lisensi YOLOvN?" keliru. Linimasa ini perlu dijelaskan dengan tepat karena inilah satu-satunya saat dalam sejarah YOLO ketika tekanan komunitas benar-benar mengubah hasilnya.

- **18 Februari 2024:** [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9) dipublikasikan bersamaan dengan [paper](https://arxiv.org/abs/2402.13616).
- **22 Februari 2024:** seorang pengguna membuka issue #10 dan bertanya tentang lisensinya. Chien-Yao Wang (WongKinYiu) menjawab "I think it should be GPL3", lalu menambahkan file GPL-3.0 empat hari kemudian.
- **26 Februari 2024:** pengguna lain membuka [issue #82, "An Apache/MIT rewrite"](https://github.com/WongKinYiu/yolov9/issues/82). Selama dua setengah minggu berikutnya, belasan pengguna komersial ikut menyampaikan pendapat.
- **14 Maret 2024:** Wang menulis di thread tersebut: *"Okay I create a new repo for mit rewrite... I think I can handle most of implementation of architectures and loss functions. And need someone to give great help about dataloader and ddp training."*

Repo itu, yang awalnya bernama `yolov9mit`, kini adalah **[MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO)**: berlisensi MIT, hak cipta "Kin-Yiu, Wong and Hao-Tang, Tsui", dan mendeskripsikan dirinya sebagai *"the official implementation of YOLOv7 and YOLOv9, YOLO-RD."* Sebagian besar penulisan ulang ini dikerjakan bukan oleh sukarelawan di thread tersebut, melainkan oleh Hao-Tang Tsui, rekan satu lab Wang, yang menulis sekitar 90 persen commit repo itu.

Jadi, YOLOv9 sekaligus "tidak dapat dirilis" dan "dapat dirilis", tergantung repo mana yang di-clone. Arsitekturnya sama, labnya sama, lisensinya dua macam.

**Sebelum mengandalkannya untuk produk, ada tiga catatan yang tidak disebutkan repo itu di halaman depannya:**

1. **Proyeknya masih berkembang.** Sebuah [issue terbuka](https://github.com/MultimediaTechLab/YOLO/issues/231) dari Februari 2026 melaporkan bahwa checkpoint yang dirilis tidak menghasilkan ulang angka COCO dari paper dan memiliki parameter jauh lebih banyak daripada yang diklaim paper. Pelatihan segmentasi dan keypoint [belum diimplementasikan](https://github.com/MultimediaTechLab/YOLO/issues/232). Tidak ada perubahan yang digabungkan ke main sejak tag v1.0 pada Desember 2025.
2. **Bobotnya tidak memiliki pernyataan lisensi terpisah.** Menurut konvensi, LICENSE MIT repo tersebut mencakup aset rilisnya, dan bukti teknis menunjukkan checkpoint dilatih oleh proyek ini, bukan disalin dari repo GPL: ukuran file dan jumlah parameternya berbeda dari checkpoint repo GPL. Namun, tidak ada dokumen yang menyatakan bahwa file .pt berlisensi MIT, dan [permintaan komunitas untuk audit formal terkait kontaminasi AGPL](https://github.com/MultimediaTechLab/YOLO/issues/51) masih terbuka. Ini adalah kekosongan dalam dokumentasi: tidak ada yang menyatakan "verified clean", dan tidak ada pula yang menuduh adanya masalah.
3. **Codebase-nya berbeda**, bukan pengganti langsung untuk skrip di repo GPL.

Semua itu tidak menjadikannya tak layak digunakan. Artinya, proyek ini perlu dievaluasi, bukan sekadar dicentang di daftar.

## Era 1: tahun-tahun domain publik (YOLOv1 hingga YOLOv4)

Joseph Redmon merilis [YOLOv1](https://arxiv.org/abs/1506.02640), [YOLOv2](https://arxiv.org/abs/1612.08242) (diterbitkan sebagai paper YOLO9000), dan [YOLOv3](https://arxiv.org/abs/1804.02767) dalam framework Darknet miliknya. File [LICENSE](https://github.com/pjreddie/darknet/blob/master/LICENSE)-nya terkenal hanya terdiri dari tiga baris:

> 0. Darknet is public domain.
> 1. Do whatever you want with it.
> 2. Stop emailing me about it!

Itu adalah dedikasi domain publik yang sesungguhnya. [YOLOv4](https://arxiv.org/abs/2004.10934), yang dikelola Alexey Bochkovskiy dalam sebuah [fork Darknet](https://github.com/AlexeyAB/darknet), memuat file yang *sama*, bukan Unlicense seperti yang disebutkan beberapa panduan lisensi. Classifier GitHub sendiri tidak dapat mengidentifikasinya dan menampilkan "Other". Ini perlu diketahui jika alat kepatuhan Anda membaca kolom itu: pemindai akan menandainya sebagai lisensi tak dikenal, bukan lisensi permisif yang bersih, walaupun teksnya sangat permisif.

Darknet sendiri juga belum sepenuhnya mati. Repo Redmon tidak aktif sejak 2022, tetapi README AlexeyAB kini menunjuk ke [hank-ai/darknet](https://github.com/hank-ai/darknet) sebagai penerus C/C++ yang direkomendasikan dan masih aktif dipelihara.

**Jebakannya ada pada port.** Pada 2026 hampir tak ada yang berlatih dengan Darknet; orang memilih implementasi ulang PyTorch, dan implementasi itu punya lisensinya sendiri:

- [ultralytics/yolov3](https://github.com/ultralytics/yolov3), port YOLOv3 terpopuler, berlisensi **AGPL-3.0**. Arsitektur domain publik didistribusikan ulang di bawah lisensi paling ketat yang umum di ekosistem ini. Lisensi yang Anda dapat sepenuhnya bergantung pada port yang Anda pasang dengan pip.
- [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) berlisensi **GPL-3.0**.
- [Tianxiaomo/pytorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4) berlisensi **Apache-2.0**, sehingga jalur permisif YOLOv4 tetap tersedia di PyTorch.
- [WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4), dari salah satu penulis YOLOv4, sama sekali tidak memiliki file LICENSE. Ini lebih buruk daripada copyleft: tanpa lisensi, tidak ada hak yang diberikan, dan semua hak tetap dilindungi secara default.

Periksa lisensi repo yang Anda clone, bukan nomor versi pada papernya.

## Era 2: GPL-3.0 (YOLOv6, v7, v9)

[YOLOv6](https://github.com/meituan/YOLOv6) (Meituan), [YOLOv7](https://github.com/WongKinYiu/yolov7), dan [YOLOv9](https://github.com/WongKinYiu/yolov9) dirilis dengan GPL-3.0.

GPL-3.0 adalah copyleft: jika Anda mendistribusikan versi yang dimodifikasi atau menggabungkan kode GPL dengan kode Anda menjadi satu program, Anda harus menyediakan source lengkap yang berkaitan dengan gabungan tersebut di bawah GPL-3.0. Ada dua batasan penting. **Agregasi semata** berarti program terpisah dan independen yang sekadar dikirim bersama tidak otomatis menjadi satu gabungan. Pemicu lainnya adalah **distribusi**: berbeda dari AGPL, jika gabungan tersebut tidak pernah Anda kirimkan, GPL biasa tidak mewajibkan pengungkapan. Karena itulah beberapa perusahaan menjalankan model GPL sepenuhnya di balik API mereka sendiri.

Jalur itu lebih sempit daripada kesannya, dan gagal dalam tiga situasi yang jarang direncanakan orang. Begitu model dipasang di perangkat pelanggan atau instalasi on-prem, itu berarti distribusi. Status "hanya internal" tidak berlaku lagi ketika salinan keluar dari badan hukum Anda: memberikan build kepada kontraktor, tim pengembangan alih daya, atau afiliasi berbadan hukum terpisah termasuk distribusi, meskipun menyalin file di dalam satu perusahaan bukan. Perlindungan itu hanya berlaku jika *semua* yang dapat dijangkau melalui API tersebut berlisensi GPL atau lebih permisif, karena satu komponen AGPL saja di mana pun dalam layanan itu akan membuat seluruhnya tunduk pada Section 13, apa pun bunyi file GPL-nya.

**YOLOv7 dan YOLOv9 punya jalur MIT alternatif yang dijelaskan di atas.** YOLOv6 tidak. Bukti terkuatnya datang dari Baidu: PaddleDetection berlisensi Apache-2.0, dan maintainer-nya sengaja memisahkan YOLOv5, YOLOv6, YOLOv7, dan YOLOv8 ke repo GPL-3.0 tersendiri, [PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO), dengan menyatakan bahwa kode model tersebut "will not be merged into PaddleDetection." Ketika perusahaan sebesar Baidu membuat batas agar YOLOv6 tidak masuk ke framework permisifnya, itu menunjukkan nilai lisensinya.

## Era 3: AGPL-3.0 (YOLOv5, v8, v10, 11, v12, v13, 26, YOLOE)

Ultralytics beralih ke AGPL-3.0 pada **14 April 2023**, bukan pada 2022 seperti yang sangat sering diulang, termasuk oleh panduan yang muncul di halaman pertama hasil pencarian untuk pertanyaan ini. Semua yang muncul setelahnya ([YOLOv8](https://github.com/ultralytics/ultralytics), YOLO11, dan [YOLO26](https://arxiv.org/abs/2606.03748)) menggunakan AGPL-3.0, dengan Enterprise License berbayar sebagai alternatif. [Halaman lisensi](https://ultralytics.com/license) mereka menyatakan bahwa tier Enterprise mencakup "YOLO26, earlier YOLO versions, and any future YOLO models", jadi kebijakan ini disengaja dan berlaku ke depan. Harga tidak dipublikasikan.

Tanggal tersebut dapat diverifikasi dan penting untuk diperiksa, karena versi cerita yang populer keliru dalam hal yang berdampak:

- File `LICENSE` di `ultralytics/yolov5` tercatat tepat tiga kali diubah. Perubahan ketiga adalah commit [`34cf749`](https://github.com/ultralytics/yolov5/commit/34cf749958d2dd3ed1205f6bb07e0f20f6e2372d), "Update LICENSE to AGPL-3.0 (#11359)," bertanggal **2023-04-14**, yang mengganti `GNU GENERAL PUBLIC LICENSE` dengan `GNU AFFERO GENERAL PUBLIC LICENSE` di 101 file.
- Rilis YOLOv5 terakhir pada 2022, **v7.0 (22 November 2022), masih menggunakan GPL-3.0**, begitu pula v6.2, v6.1, dan v6.0 sebelumnya. Ribuan fork yang berhenti menyinkronkan sebelum April 2023 juga demikian: pilih salah satunya, GitHub masih menyatakan GPL-3.0 hari ini.
- Pada hari yang sama, 41 menit kemudian, repo `ultralytics/ultralytics` [mengalami perubahan yang sama](https://github.com/ultralytics/ultralytics/commit/2c6fc0a4443b9cf805ef17b1cfdd71a98693b4d4) (PR #2031).

Dari sini muncul fakta yang hampir tidak diketahui orang: **YOLOv8 tidak diluncurkan dengan AGPL.** Model itu dirilis pada Januari 2023 dengan **GPL-3.0**, lalu lisensinya diubah tiga bulan kemudian. Catatan PyPI tegas: `ultralytics` versi 8.0.0 (10 Januari 2023) hingga 8.0.76 (13 April 2023) semuanya menyatakan `GPL-3.0`. Versi 8.0.80 yang diunggah pada 16 April 2023 adalah versi pertama yang menyatakan `AGPL-3.0`.

Ini bukan celah. Lisensi yang sudah diberikan tidak dapat dicabut secara retroaktif, jadi versi lama tetap tersedia berdasarkan ketentuan saat diterbitkan. Namun, versi itu sudah berusia tiga tahun, tidak dipelihara dan tidak mendapat patch. GPL-3.0 tetap copyleft (klausul jaringan ditukar dengan klausul distribusi, bukan berarti copyleft terhindarkan), dan posisi komersial Ultralytics lebih luas daripada teks lisensinya. Pelajaran yang berguna lebih terbatas dan umum: **lisensi melekat pada versi yang Anda terima, bukan nama proyeknya.** Pin dependency dan catat bunyi lisensi pada hari Anda mengambil salinannya, karena proyek dapat mengubahnya besok dan catatan kepatuhan Anda bergantung pada versi yang benar-benar Anda dapatkan.

AGPL-3.0 adalah GPL ditambah Section 13, yaitu klausul jaringan: *jika Anda memodifikasi Program*, pengguna yang berinteraksi dengan versi modifikasi melalui jaringan harus ditawari source-nya. Klausul ini menutup jalur SaaS yang masih terbuka pada GPL biasa.

Perhatikan batasan ini, karena sebagian besar artikel menghilangkannya. Section 0 AGPL mendefinisikan "modify" sebagai menyalin atau mengadaptasi karya *dengan cara yang memerlukan izin hak cipta*. Menjalankan kode pelatihan tanpa modifikasi pada dataset Anda sendiri berarti mengeksekusi program, bukan mengadaptasi source-nya. Ini sama seperti mengompilasi kode Anda dengan compiler GPL yang tidak dimodifikasi, yang tidak berarti Anda memodifikasi compiler tersebut. Jadi, pelatihan saja tidak otomatis menjadi pemicu, dan siapa pun yang mengatakan demikian telah melewatkan definisinya.

Yang membuat Anda masuk ke dalam cakupan lisensi adalah menggabungkan atau menyematkan paket itu ke codebase Anda sehingga keduanya dikirimkan atau dijalankan sebagai satu program. Menurut penafsiran FSF sendiri, menghubungkan karya GPL atau AGPL dengan modul lain menghasilkan karya gabungan; belum pernah diuji di pengadilan apakah hal itu mencakup impor Python. Perlu diketahui juga bahwa [ketentuan komersial](https://ultralytics.com/license) Ultralytics sendiri mewajibkan Enterprise License untuk platform SaaS, API, dan sistem cloud yang menggunakan YOLO di balik layar. Jadi, posisi vendor lebih luas daripada bunyi lisensi saja. Anggap integrasi ke produk yang Anda layani atau kirim sebagai risiko; jangan berasumsi bahwa pelatihan secara terpisah tidak menimbulkan kewajiban.

Khusus YOLO26, tanggalnya sering tercampur di berbagai sumber. Model ini **diperkenalkan sebagai preview pada September 2025** di YOLO Vision, **benar-benar dirilis pada Januari 2026** (rilis `ultralytics` 8.4.0, dengan catatan "Ultralytics YOLO26 has arrived"), dan **papernya terbit pada Juni 2026**. YOLO26 berada di repo utama `ultralytics/ultralytics`; `ultralytics/yolo26` hanya halaman pengantar.

**Bagian yang mengejutkan banyak orang:** YOLO akademis dari dua tahun terakhir juga berlisensi AGPL, dan bukan karena Ultralytics yang menulisnya.

- [YOLOv10](https://arxiv.org/abs/2405.14458) berasal dari Tsinghua University. README-nya menyatakan *"The code base is built with ultralytics."*
- [YOLOv12](https://arxiv.org/abs/2502.12524): *"The code is based on ultralytics."*
- [YOLOv13](https://arxiv.org/abs/2506.17733): *"The code is based on Ultralytics."*
- [YOLOE](https://arxiv.org/abs/2503.07465), model "see anything" dengan open-vocabulary dari kelompok YOLOv10, juga berlisensi AGPL-3.0 dan dibangun di atas Ultralytics.

Copyleft diwariskan: karya turunan dari kode AGPL harus disediakan dengan AGPL. Karena itu, begitu salah satu fork ini dikirim atau disajikan, lisensinya ikut berlaku. (Jika dimodifikasi lalu hanya digunakan sendiri, tanpa dibagikan, kewajiban tidak timbul. Karena itu, penggunaan untuk riset tidak terdampak.) Risetnya independen, tetapi lisensinya tidak. Sejak 2024, menerbitkan "YOLO berikutnya" sebagai turunan Ultralytics menjadi alur kerja default. Akibatnya, AGPL kini otomatis merambat ke versi yang lebih baru.

Jika menemukan halaman yang menyatakan YOLOv13 berlisensi Apache-2.0, informasi itu keliru. File LICENSE, GitHub API, dan model card semuanya menyatakan AGPL-3.0.

### Jalur permisif YOLOv8 yang sudah tidak tersedia

Banyak panduan yang masih beredar, termasuk draf lama artikel ini, menunjuk YOLOv8 KerasCV berlisensi Apache-2.0 sebagai jalan keluar dari AGPL YOLOv8. **Saat ini opsi itu tidak dapat diandalkan.** Berikut catatan publiknya:

- Dalam diskusi KerasCV [Clarifications regarding YOLOv8 licensing](https://github.com/keras-team/keras-cv/discussions/2032), seorang kontributor menulis bahwa "many parts are derived form ultralytics." Tidak ada maintainer yang menjawab thread tersebut, sehingga pertanyaan tentang seberapa independen implementasinya belum pernah dipastikan secara publik.
- Issue KerasHub [add YOLOV8](https://github.com/keras-team/keras-hub/issues/1760) ditutup pada Juli 2025 oleh maintainer Keras dengan komentar: "Hey all!! Because of license issues, we have decided to drop this." Dua orang kemudian bertanya apa dampaknya bagi pengguna komersial versi KerasCV yang sudah ada. Keduanya tidak mendapat jawaban.
- KerasCV kini diarsipkan, dan KerasHub, penerusnya yang masih dipelihara, **tidak menyertakan YOLOv8**.

Kami tidak tahu apakah kedua hal itu berkaitan, dan kami tidak menyatakan telah terjadi sesuatu yang tidak semestinya. Namun, ini bukan dasar yang dapat diandalkan untuk membangun produk.

## Pengecualian permisif: YOLOX, PP-YOLOE, YOLO-NAS

Tidak semua YOLO ikut dalam tren copyleft.

- **[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)** (Megvii, 2021) berlisensi Apache-2.0 tanpa pengecualian, baik untuk kode maupun bobot. Ini masih YOLO klasik paling aman untuk penggunaan komersial, meskipun repo tersebut tidak menerima commit sejak pertengahan 2025, dan [pertanyaan tentang menyertakan bobot pretrained dalam aplikasi komersial](https://github.com/Megvii-BaseDetection/YOLOX/issues/1865) masih terbuka sejak Maret 2026 tanpa jawaban. Teks lisensinya jelas. Hanya saja, saat ini tidak ada orang yang dapat menjawab pertanyaan tentangnya. Kami menulis tentang [menjalankan YOLOX dengan LibreYOLO](/articles/yolox-with-libreyolo).
- **[PP-YOLOE](https://arxiv.org/abs/2203.16250)** (Baidu) berlisensi Apache-2.0 **di dalam PaddleDetection**. Ambil dari sana, bukan dari PaddleYOLO, yang berisi direktori konfigurasi hampir identik dan berlisensi GPL-3.0. Model sama, perusahaan sama, dua repo, dua lisensi. Ini kebalikan dari situasi YOLOv9.
- **[YOLO-NAS](https://github.com/Deci-AI/super-gradients)** (Deci, 2023) sering mengecoh orang. Kodenya berlisensi Apache-2.0, tetapi bobot resminya tidak. [Lisensi terpisah](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md) menyatakan bahwa Anda *"may not use the Software for any commercial use, including in connection with any models used in a production environment."* Sejak NVIDIA mengakuisisi Deci pada 2024, kami tidak menemukan pekerjaan fitur baru di repo tersebut, dan URL bobot aslinya tidak lagi berfungsi (checkpoint kini berada di host lain). Ada saran untuk mengatasinya dengan melatih arsitekturnya dari nol agar tetap bersih di bawah Apache. Itu mungkin benar, tetapi belum dikonfirmasi Deci atau NVIDIA, dan sulit diselaraskan dengan teks lisensi yang mencakup "any components comprising the model." Informasi lebih lanjut tentang YOLO-NAS [di sini](/articles/yolo-nas-with-libreyolo).
- **[MMYOLO](https://github.com/open-mmlab/mmyolo)** kadang dicantumkan sebagai Apache-2.0 dalam panduan lama. Sebenarnya tidak: MMYOLO berlisensi GPL-3.0 (saudaranya, MMDetection, yang berlisensi Apache), dan tidak menerima commit sejak Juli 2024.
- **[YOLO-World](https://github.com/AILab-CVC/YOLO-World)** (Tencent) berlisensi **GPL-3.0**, bukan AGPL dan bukan permisif. Ini perlu ditegaskan karena YOLO open-vocabulary sering dianggap permisif karena asosiasinya dengan ekosistem bergaya CLIP yang digunakannya.

Kasus-kasus itu mencakup tiga kekeliruan lisensi yang paling umum: menganggap lisensi bobot sama dengan lisensi kode, mengira semua proyek dari satu organisasi memiliki lisensi yang sama, dan menyimpulkan lisensi suatu model dari namanya.

## Lisensi mencakup kode, bukan gagasan (tetapi waspadai paten)

Satu prinsip menjadi dasar hukum bagi semua jalur permisif di atas: **hak cipta melindungi ekspresi, bukan gagasan.** Ini bukan sekadar slogan, melainkan hukum yang berlaku secara tertulis di kedua sisi Atlantik.

Di AS, prinsip ini terdapat pada [17 U.S.C. 102(b)](https://www.law.cornell.edu/uscode/text/17/102), yang mengecualikan "idea, procedure, process, system, method of operation, concept, principle, or discovery" dari perlindungan hak cipta, sesuai prinsip dalam *Baker v. Selden* (1879). Di UE, yang hukumnya berlaku bagi kami dan mungkin juga bagi banyak pembaca, Pasal 1(2) [Software Directive (2009/24/EC)](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32009L0024) menyatakan bahwa "ideas and principles which underlie any element of a computer program" tidak dilindungi. Court of Justice menegaskan dalam *SAS Institute v World Programming* ([C-406/10](https://curia.europa.eu/juris/liste.jsf?num=C-406/10), 2012) bahwa fungsionalitas program, bahasa pemrogramannya, dan format datanya bukan ekspresi yang dilindungi. Hanya kodenya.

Arsitektur yang dijelaskan dalam paper arXiv adalah gagasan yang telah dipublikasikan. Siapa pun boleh mengimplementasikannya. Yang tidak boleh dilakukan adalah menyalin atau mengadaptasi file kode sumber GPL/AGPL milik orang lain lalu memberi lisensi ulang. Implementasi independen yang ditulis berdasarkan paper, bukan source, merupakan karya baru dan penulisnya dapat memilih lisensi.

Ada dua catatan penting yang kerap dilewatkan vendor:

- **Implementasi itu memang harus "independen".** Bentuk yang dapat dipertahankan adalah proses clean-room: penulis kode bekerja dari paper dan spesifikasi, bukan source copyleft. Membaca repo GPL dengan saksama lalu "menulis ulang" bukan hal yang sama, dan perbedaan itu penting jika suatu saat ada pemeriksaan.
- **Ini argumen hak cipta, bukan argumen paten.** Penemuan independen *bukan* pembelaan terhadap paten. Implementasi ulang clean-room menjawab persoalan hak cipta, tetapi tidak mengubah persoalan paten. Saat ini tidak ada paten yang banyak diklaim atas arsitektur YOLO, tetapi "belum ada yang menggugat" tidak sama dengan "tidak ada dasar gugatan."

## Bobot adalah lisensi kedua

Kesalahan termahal di bidang ini adalah mengaudit kode lalu melupakan checkpoint. Ada tiga lapisan yang masing-masing memiliki lisensi, dan lisensinya bisa berbeda:

**1. Kode.** Semua yang dibahas di atas.

**2. Bobot.** Kadang lisensinya dinyatakan secara eksplisit, seperti pada YOLO-NAS. Kadang dinyatakan sepihak: [halaman lisensi](https://ultralytics.com/license) Ultralytics menyatakan bahwa AGPL-3.0 "covers the training code and the models produced by that training code", dan memperluasnya ke model yang Anda latih sendiri dengan kode mereka. Sebagai pemegang hak cipta atas materi yang mereka terbitkan, mereka dapat menetapkan ketentuan untuk checkpoint mereka sendiri. Apakah bobot yang *Anda* hasilkan secara hukum merupakan karya turunan dari kode pelatihan masih menjadi pertanyaan terbuka: belum ada preseden hukum, dan pendapat itu bertentangan dengan aturan umum FSF sendiri bahwa output program tidak otomatis tercakup oleh hak cipta program tersebut. Namun, belum pasti bukan berarti aman. Dalam praktiknya, perlakukan bobot AGPL yang diterbitkan sebagai materi yang terkena pembatasan lisensi.

Risiko terbesar bagi siapa pun yang mengikuti saran artikel ini: **repo YOLOv7 dan YOLOv9 berlisensi GPL tidak menjelaskan apa pun tentang bobotnya.** Checkpoint-nya merupakan aset rilis di repo GPL-3.0 tanpa pemberian hak terpisah. Memuat file .pt itu ke implementasi ulang MIT menghasilkan kode yang bersih dan checkpoint yang statusnya belum dijelaskan siapa pun. Jika alasan Anda beralih adalah lisensi, gunakan bobot yang dilatih sendiri oleh proyek permisif tersebut atau latih bobot sendiri.

**3. Data pelatihan.** *Anotasi* COCO berlisensi CC BY 4.0, jadi labelnya tidak bermasalah. *Gambar* bukan milik COCO untuk dilisensikan: foto Flickr memiliki beragam ketentuan tersendiri, dan konsorsium COCO secara eksplisit tidak memilikinya. Sebagian besar industri menganggapnya berisiko rendah lalu melanjutkan pekerjaan, tetapi "COCO tidak masalah" hanya merujuk pada anotasinya. Di luar COCO, batasannya cepat menjadi lebih ketat: banyak dataset udara, medis, dan kendaraan hanya untuk penggunaan nonkomersial, dan lisensi kode model yang permisif tidak menghapus batasan tersebut.

Audit ketiga lapisan ini setiap kali.

## Catatan tentang MIT dan Apache-2.0

Kami merilis library berlisensi MIT, jadi akan mudah bagi kami untuk mengatakan bahwa MIT adalah lisensi terbaik. Perbandingan yang jujur lebih menarik.

MIT tidak berarti tanpa kewajiban: Anda harus mempertahankan pemberitahuan hak cipta dan teks lisensi dalam salinan yang didistribusikan. Itu persyaratan nyata, meskipun ringan. Apache-2.0 juga memiliki sesuatu yang tidak dimiliki MIT: **pemberian hak paten secara tegas** dari setiap kontributor, serta klausul pembalasan yang mengakhiri pemberian hak bagi siapa pun yang menggugat terkait karya tersebut. MIT tidak menyebut paten. Bagi perusahaan yang paling khawatir terhadap paparan paten, Apache-2.0 bisa dibilang *lebih aman* daripada MIT. Banyak ekosistem deteksi permisif (YOLOX, RT-DETR, D-FINE, DEIM) menggunakan Apache-2.0 karena alasan itu.

Keunggulan MIT adalah kesederhanaan dan kompatibilitas. Kedua lisensi tidak akan pernah memaksa Anda menerbitkan source, dan itulah aspek yang dibahas artikel ini. Jika ada yang mengatakan bahwa pilihan pentingnya adalah MIT atau Apache, mereka sedang menjual sesuatu. Pilihan pentingnya adalah permisif atau copyleft.

## Jadi, YOLO mana yang dapat digunakan?

- **Produk komersial closed source:** YOLOv7 atau YOLOv9 melalui [repo MIT milik lab](https://github.com/MultimediaTechLab/YOLO), YOLOX atau PP-YOLOE (dari PaddleDetection) di antara model asli, atau framework MIT yang dipelihara seperti LibreYOLO jika Anda tidak ingin merakit dan mengaudit semuanya sendiri. Hindari GPL atau AGPL kecuali Anda akan membuka seluruh aplikasi atau membeli Enterprise License.
- **Proyek open source:** apa pun, selama lisensinya kompatibel. Jika proyek Anda menggunakan AGPL, lini Ultralytics cocok secara alami, dan riset terbaru (YOLOv13, YOLO26, YOLOE) pertama kali muncul di sana.
- **Riset dan benchmarking:** lisensi hampir tidak membatasi. Gunakan apa pun yang digunakan paper pembanding.
- **Rencana "nanti saja dipikirkan":** tidak akan berhasil. Membongkar dependency AGPL setelah produk selesai dibuat berarti melatih ulang, memvalidasi ulang, dan mengekspor ulang semuanya, termasuk bobot. Pilih repo, dan karena itu lisensinya, terlebih dahulu.

Untuk membandingkan framework, bukan lisensinya, lihat [Alternatif Ultralytics Terbaik pada 2026](/articles/best-ultralytics-alternatives).

## Opsi MIT: yang disertakan LibreYOLO

Pengungkapan: LibreYOLO adalah proyek kami, dan lisensinya menjadi alasan artikel ini ditulis.

**[LibreYOLO](https://github.com/LibreYOLO/libreyolo) adalah satu codebase berlisensi MIT** yang mencakup detektor di bawah ini melalui satu API, dengan pelatihan serta ekspor ONNX, TensorRT, OpenVINO, dan NCNN yang sudah tersedia. Tanpa copyleft, tanpa klausul jaringan, tanpa tier enterprise.

Berikut daftar lengkap model deteksi objek beserta upstream dan lisensinya, agar asal setiap jalur permisif terlihat jelas:

| Di LibreYOLO | Model | Upstream | Lisensi upstream |
| --- | --- | --- | --- |
| `LibreYOLO2`, `LibreYOLO3`, `LibreYOLO4` | YOLO era Darknet, dengan PyTorch | [pjreddie/darknet](https://github.com/pjreddie/darknet), [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) | Domain publik |
| `LibreYOLO7` | YOLOv7 | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) | MIT (penulisan ulang milik lab, bukan repo GPL) |
| `LibreYOLO9`, `LibreYOLO9E2E` | YOLOv9, serta varian end-to-end tanpa NMS | [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) | MIT (penulisan ulang milik lab, bukan repo GPL) |
| `LibreYOLO9P2` | YOLOv9 dengan head stride-4 untuk objek kecil | Asli LibreYOLO | MIT |
| `LibreYOLOX` | YOLOX | [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | Apache-2.0 |
| `LibreYOLONAS` | YOLO-NAS, deteksi dan pose | [Deci-AI/super-gradients](https://github.com/Deci-AI/super-gradients) | Kode Apache-2.0. Bobot upstream yang dibatasi **tidak** didistribusikan ulang |
| `LibreRTDETR`, `LibreRTDETRv2` | RT-DETR dan v2 | [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | Apache-2.0 |
| `LibreRTDETRv4` | RT-DETRv4 | [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | Apache-2.0 |
| `LibreRFDETR` | RF-DETR, deteksi, segmentasi, pose, OBB | [roboflow/rf-detr](https://github.com/roboflow/rf-detr) | Apache-2.0 untuk N/S/M/L. XL/2XL menggunakan Platform Model License Roboflow, jadi kami hanya menyertakan ukuran berlisensi Apache |
| `LibreDFINE` | D-FINE | [Peterande/D-FINE](https://github.com/Peterande/D-FINE) | Apache-2.0 |
| `LibreDEIM` | DEIM | [Intellindust-AI-Lab/DEIM](https://github.com/Intellindust-AI-Lab/DEIM) | Apache-2.0 |
| `LibreDEIMv2` | DEIMv2 | [Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) | Kode Apache-2.0. Ukuran yang lebih besar menggunakan backbone DINOv3 Meta dengan lisensi non-OSI milik Meta. Redistribusi diizinkan, tetapi **melarang penggunaan untuk militer, nuklir, spionase, dan senjata**. Ukuran tersebut diterbitkan sebagai `other`, bukan Apache. Baca lisensinya sebelum merilis apa pun untuk penggunaan ganda |
| `LibreRTMDet` | RTMDet ([tanpa MMDetection](/articles/rtmdet-without-mmdetection)) | [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) | Apache-2.0 (saudara yang berlisensi Apache, bukan mmyolo berlisensi GPL-3.0) |
| `LibrePICODET` | PP-PicoDet | Arsitektur dari [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), checkpoint dari port ulang [Picodet_Pytorch](https://github.com/Bo396543018/Picodet_Pytorch) | Apache-2.0 (keduanya) |
| `LibreEC` | EdgeCrafter, deteksi, pose, segmentasi | [Intellindust-AI-Lab/EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter) | Apache-2.0 |
| `LibreFOMO` | Detektor centroid untuk perangkat keras kelas mikrokontroler | Asli LibreYOLO | MIT |
| `LibreGroundingDINO` | Grounding DINO, open-vocabulary (inferensi) | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | Apache-2.0 |
| `LibreOWLv2` | OWLv2, open-vocabulary (inferensi) | Google, melalui `transformers` | Apache-2.0 |

Ada dua hal yang sengaja disampaikan dengan jujur di kolom itu.

**Tidak ada source GPL atau AGPL yang disalin ke codebase.** Semua keluarga model di atas dibangun dari upstream domain publik, MIT, atau Apache-2.0. Itulah aturan yang menjaga framework tetap MIT. Jika suatu arsitektur tersedia dalam versi permisif dan copyleft, kami memakai versi permisifnya: YOLOv9 dan YOLOv7 kami berasal dari repo MIT milik lab tersebut, bukan repo GPL aslinya; RTMDet kami berasal dari MMDetection berlisensi Apache, bukan MMYOLO berlisensi GPL. YOLO-World tidak ada dalam daftar, meskipun sering diminta, karena model itu berlisensi GPL-3.0 dan tidak ada sumber permisif untuk mengambilnya.

**Permisif tidak sama dengan tanpa batasan, dan kami lebih suka menjelaskannya sekarang daripada membiarkan Anda menemukannya saat audit.** Ukuran besar DEIMv2 mewarisi ketentuan DINOv3 Meta, termasuk larangan penggunaan untuk militer, nuklir, spionase, dan senjata. Ini batasan nyata jika Anda bekerja dekat dengan penggunaan ganda. Sejumlah kecil checkpoint pratinjau riset dilatih dengan dataset nonkomersial (DOTA, VisDrone) dan diberi label `cc-by-nc` di Hugging Face, alih-alih diam-diam dirilis seolah bebas digunakan. Peringatan yang sama untuk bobot pihak lain berlaku juga bagi bobot kami: checkpoint YOLO9 kami dikonversi dari checkpoint repo MIT milik lab tersebut, sehingga mewarisi statusnya, yang seperti dijelaskan di atas adalah "dinyatakan, belum diaudit secara formal." Lisensi MIT mencakup kode yang kami tulis. Setiap checkpoint memiliki ketentuan berdasarkan data dan sumber pelatihannya, yang dipublikasikan untuk masing-masing bobot.

Selain deteksi, API yang sama juga mencakup segmentasi instance dan semantik, pose, bounding box berorientasi, klasifikasi (MobileNetV4, ConvNeXt, EfficientNetV2, ResNet, CLIP), estimasi kedalaman monokular, restorasi gambar, dan estimasi arah pandang.

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
```

Alur kerja YOLO yang sama, tanpa pekerjaan rumah soal lisensi.

Beri bintang di GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Dokumentasi: [libreyolo.com/docs](https://libreyolo.com/docs)

---

*Merek dagang dan afiliasi: YOLO, YOLOv8, YOLO11, YOLO26, dan Ultralytics adalah merek dagang pemiliknya masing-masing, termasuk Ultralytics Inc. LibreYOLO adalah proyek independen dan tidak terafiliasi dengan, disponsori oleh, atau didukung oleh Ultralytics maupun organisasi lain yang disebut dalam artikel ini. Nama produk dan repo hanya digunakan untuk mengidentifikasi dan membandingkan perangkat lunak yang dibahas.*

*Artikel ini memuat informasi umum, bukan nasihat hukum, berdasarkan informasi hingga 11 Juli 2026, dan ditulis oleh salah satu vendor dalam perbandingan ini. Periksa lisensi terbaru sebelum merilis produk.*
