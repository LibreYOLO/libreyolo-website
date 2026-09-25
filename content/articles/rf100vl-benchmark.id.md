---
title: "Benchmark RF100-VL: Seberapa Baik Model LibreYOLO Melakukan Generalisasi?"
description: Hasil RF100-VL yang mengukur generalisasi YOLOv9, YOLOX, YOLO-NAS, EdgeCrafter, dan RF-DETR di 100 dataset dunia nyata setelah fine-tuning.
date: 2026-09-05
author: Xuban
layout: paper
tags: [LibreYOLO, RF100-VL, benchmark, object-detection, roboflow]
---

RF100-VL mengukur seberapa baik sebuah detektor beradaptasi di luar COCO. Setiap model menjalani fine-tuning secara terpisah pada 100 dataset yang sangat beragam, termasuk inframerah, sinar-X, mikroskopi, olahraga, gambar spektrum radio, objek kecil, dan video game. Kami menguji 17 konfigurasi LibreYOLO: **total 1700 proses fine-tuning.**

<div>
<rf100vl-explorer></rf100vl-explorer>
</div>

## Hasil

<div>
<rf100vl-results-chart></rf100vl-results-chart>
</div>

RF-DETR-L memimpin perbandingan ini dengan skor **61.76**. M menyusul dengan 61.13, lalu S dengan 60.41. Keempat hasil RF-DETR mendekati angka yang dipublikasikan [Roboflow](https://rfdetr.roboflow.com/latest/learn/benchmarks/), yang menjadi validasi berguna bagi pelatihan RF-DETR di LibreYOLO.

Ukuran model tidak selalu memprediksi hasil. YOLO-NAS-S dan M pada dasarnya imbang, masing-masing 58.00 dan 57.99. Skor EdgeCrafter-M adalah 57.93, lebih tinggi daripada S dengan 55.99 dan L dengan 56.11. Kami berencana menyelidiki regresi ini. Rangkaian pengujian ini bukan eksperimen penskalaan terkontrol: resolusi, bobot pretrained, presisi, dan resepnya beragam.

RF-DETR-S dengan backbone LoRA mendapat skor 57.19, dibandingkan 60.41 untuk fine-tuning penuh. Fine-tuning penuh unggul pada 95 dataset. Median waktu yang tercatat hanya tujuh menit lebih lama, sedangkan total waktu pekerjaan hampir sama.

Baris YOLOv9 berasal dari sebelum perbaikan pelatihan penting diterapkan. Hasil M yang tidak wajar membantu menemukan PGI yang hilang, konvensi letterbox yang berbeda, dan batas pelatihan 100 label. Angka arsip tersebut menggambarkan kode yang dijalankan. Angka itu bukan penilaian atas arsitektur YOLOv9.

Pilih model di bawah untuk memeriksa skornya pada seluruh 100 dataset.

<div>
<rf100vl-results-detail></rf100vl-results-detail>
</div>

## Metodologi

Untuk setiap dataset, kami melatih model pada `train`, memilih checkpoint AP50:95 validasi terbaik, lalu mengevaluasinya sekali pada `test`. Skor utama adalah rata-rata tanpa pembobotan dari 100 hasil pengujian tersebut.

| Pengaturan | Aturan kampanye |
|---|---|
| Pelatihan | 100 epoch; tanpa early stopping |
| Batch efektif | 16 |
| Seed | 0; satu proses yang selesai per dataset |
| Checkpoint | AP50:95 validasi terbaik dengan bobot EMA |
| Skor pengujian | pycocotools dengan `maxDets=500` |
| Tuning | Satu resep tetap per keluarga model; tanpa tuning per dataset |

Setiap keluarga model menggunakan resep pelatihannya sendiri:

| Keluarga model | Resolusi | Presisi | Ringkasan augmentasi |
|---|---:|---:|---|
| YOLOv9 T/S/M | 640 | FP32 | Mosaic, HSV, flip; augmentasi kuat dinonaktifkan selama 15 epoch terakhir |
| YOLOX Nano/Tiny | 416 | FP32 | Mosaic, mixup, HSV, affine, flip; bagian akhir 15 epoch |
| YOLOX S/M | 640 | FP32 | Resep yang sama untuk keluarga model ini, dengan learning rate khusus ukuran |
| YOLO-NAS S/M | 640 | FP32 | Mixup, HSV, dan flip; mosaic dinonaktifkan |
| EdgeCrafter S/M/L | 640 | FP16 | Flip; default keluarga model LibreYOLO |
| RF-DETR N/S/M/L dan S LoRA | 384/512/576/704; LoRA pada 512 | BF16 | Multi-scale, crop/resize, dan flip |

Ini adalah hasil dengan satu seed. Perbedaan kecil belum terbukti sebagai peningkatan. RF-DETR M dan L menggunakan akumulasi gradien agar muat dalam memori yang tersedia, dan beberapa dataset padat memerlukan batch fisik yang lebih kecil. Resep RF-DETR publik juga dimulai dari checkpoint COCO, sedangkan Roboflow menggunakan checkpoint Objects365 privat untuk tabel historisnya.

Waktu pelatihan berasal dari catatan kampanye, bukan benchmark kecepatan terkontrol. Pekerjaan terkadang berbagi GPU, dicoba ulang, atau dilanjutkan. Kami tidak menjalankan protokol latensi satu artefak T4 opsional. YOLO-NAS menggunakan [bobot pretrained nonkomersial](https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md) milik Deci, dengan lisensi terpisah.

## Peningkatan pada beban kerja ini

Uji kecil dapat membuktikan bahwa pelatihan dimulai. Uji tersebut tidak mereproduksi fine-tuning berkelanjutan selama berminggu-minggu pada banyak arsitektur dan dataset. Pada skala ini, jalur yang lambat, tekanan memori, dan masalah kebenaran hasil sulit diabaikan.

- **Pemuatan gambar dan validasi.** Kami menyimpan hasil resize deterministik sebelum augmentasi, menggunakan kembali worker dan buffer validator, serta merekam forward validasi dalam CUDA graph jika didukung. [PR #677](https://github.com/LibreYOLO/libreyolo/pull/677), [PR #682](https://github.com/LibreYOLO/libreyolo/pull/682)
- **Graf CUDA untuk pelatihan.** Dukungan capture berkembang dari YOLOv9 dan RF-DETR menjadi 24 keluarga model. Kami juga memperbaiki race pin-memory DataLoader dan invalidasi graph saat transisi pelatihan. Pengujian YOLOv9-T turun dari 428.4 menjadi 367.7 detik dengan AP yang dilaporkan tetap sama. [PR #671](https://github.com/LibreYOLO/libreyolo/pull/671), [PR #681](https://github.com/LibreYOLO/libreyolo/pull/681), [PR #716](https://github.com/LibreYOLO/libreyolo/pull/716)
- **Skoring COCO.** Evaluasi pada dataset padat terkadang memakan waktu lebih lama daripada pelatihan. Integrasi faster-coco-eval memberi skor pada prediksi tersimpan dari seluruh 100 split pengujian dalam 8.4 detik, bukan 131.4 detik. Dari 1,400 nilai metrik, 1,381 identik hingga bit; perbedaan terbesar adalah 2.22e-16 dan AP utama tidak berubah. [PR #708](https://github.com/LibreYOLO/libreyolo/pull/708)
- **RF-DETR dan jalur pelatihan bersama.** Pencocokan L1 yang lebih cepat, sinkronisasi perangkat yang lebih sedikit, AdamW terpadu, dan memori matcher yang dibatasi mengurangi waktu satu langkah RF-DETR-S dari 266 menjadi 234 ms dalam pengujian yang tercatat. Penyelidikan yang sama juga meningkatkan lima matcher lain, logging YOLOv9, dan penggunaan ulang tensor EdgeCrafter. [PR #761](https://github.com/LibreYOLO/libreyolo/pull/761), [PR #762](https://github.com/LibreYOLO/libreyolo/pull/762), [PR #765](https://github.com/LibreYOLO/libreyolo/pull/765)
- **Attention.** Kami mengarahkan attention yang memenuhi syarat ke PyTorch SDPA, mengintegrasikan implementasi CUDA berlisensi Apache-2.0, dan memperbaiki eksekusi mixed precision. Kami juga menulis kernel deformable-attention Triton di dalam repositori untuk inferensi. Dalam pengujian yang didokumentasikan, kernel ini sekitar empat kali lebih cepat secara terpisah dan sekitar 7% lebih cepat dari awal hingga akhir untuk RF-DETR-N. [PR #712](https://github.com/LibreYOLO/libreyolo/pull/712), [PR #713](https://github.com/LibreYOLO/libreyolo/pull/713), [PR #760](https://github.com/LibreYOLO/libreyolo/pull/760), [PR #784](https://github.com/LibreYOLO/libreyolo/pull/784), [PR #790](https://github.com/LibreYOLO/libreyolo/pull/790)
- **Kebenaran pelatihan.** Kami memperbaiki rekonstruksi BatchNorm YOLOX dan memulihkan PGI YOLOv9, metadata letterbox, kapasitas label, serta warmup momentum. Benchmark menyediakan checkpoint dan pola kegagalan yang mengungkap kedua masalah tersebut. [PR #700](https://github.com/LibreYOLO/libreyolo/pull/700), [PR #796](https://github.com/LibreYOLO/libreyolo/pull/796)

Angka-angka tersebut berasal dari pengujian terpisah pada beban kerja yang berbeda. Angka itu tidak boleh dikalikan untuk menghasilkan satu klaim peningkatan kecepatan. LibreYOLO kini lebih cepat dan lebih andal untuk beban kerja pelatihan nyata dibandingkan sebelum kampanye ini.

## Harness dan artefak

[Harness pelatihan](https://github.com/LibreYOLO/vision-analysis-benchmark/tree/rf100vl-harness) publik menjadwalkan dataset di berbagai GPU, melanjutkan pekerjaan yang terhenti, menangani pemulihan OOM, serta mencatat resep, versi dataset, log, checkpoint, dan prediksi.

[Skill run-rf100vl-benchmark](https://github.com/LibreYOLO/libreyolo/blob/release/skills/run-rf100vl-benchmark/SKILL.md) memberikan protokol dan petunjuk operasi Vast.ai kepada agen coding AI. [Arsip hasil](https://huggingface.co/datasets/LibreYOLO/rf100-vl-results/tree/main) memuat setiap proses yang dipublikasikan.

## Terima kasih, Roboflow

Joseph Nelson menawarkan dukungan GPU setelah saya menulis bahwa saya ingin membuat benchmark di luar COCO, tetapi tidak mampu membiayai prosesnya. Matvei Popov membagikan konfigurasi referensi, menjelaskan pengaturan evaluasi, dan mengikuti hasilnya saat tersedia.

Dukungan itu memungkinkan kami memvalidasi pelatihan LibreYOLO pada 17 konfigurasi, menerbitkan bukti yang dapat digunakan orang saat memilih model, merilis harness, dan menguji library sampai titik lemahnya terlihat jelas.

Terima kasih kepada Joseph, Matvei, dan tim Roboflow atas daya komputasi, waktu, dan panduannya.
