---
title: Matriks ekspor lengkap
seo_title: Matriks dukungan ekspor LibreYOLO dan aturannya
description: >-
  Cara LibreYOLO menentukan apakah kombinasi family, task, dan format dapat
  diekspor: dua belas format, tiga tier, aturan fallback, dan ambang batas
  paritas.
lead: >-
  Dukungan ekspor merupakan lookup pada triple (family, task, format). Halaman
  ini menjelaskan bentuk matriks, aturan yang mengisi cell tanpa entri
  eksplisit, dan cara melakukan query untuk kombinasi yang diperlukan.
keywords:
  - dukungan ekspor libreyolo
  - matriks ekspor
  - onnx tensorrt openvino tflite
  - perintah formats libreyolo
  - ambang batas paritas ekspor
  - NotImplementedError ekspor
last_verified: 1.5.0
verification: >-
  Format, tier, urutan fallback, block task dan family, serta block NCNN dibaca
  dari libreyolo/export/support.py; alias dan argumen bersama dari
  libreyolo/export/exporter.py; definisi tier dari
  docs/adr/0011-export-support-tiers.md; ambang batas paritas dari
  docs/export_support.md, semuanya pada v1.5.0. Cell per kombinasi tidak disalin
  di sini; lakukan query dengan snippet di bawah.
snippets:
  usage:
    - label: Lakukan query pada matriks tanpa model
      language: python
      code: |
        from libreyolo.export.support import (
            EXPORT_FORMATS,
            get_support,
            validated_alternatives,
        )

        print(EXPORT_FORMATS)

        entry = get_support("yolo9", "detect", "onnx")
        print(entry.tier, entry.since)
        print(entry.constraint)

        print(validated_alternatives("yolo9", "detect"))
    - label: CLI
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
        libreyolo formats --family yolo9 --task detect --json
  export:
    - label: Ekspor dan baca penolakan
      language: python
      code: >
        from libreyolo import LibreYOLO

        from libreyolo.export.support import get_support


        model = LibreYOLO("LibreYOLO9t.pt")

        print(model.export(format="onnx"))


        # Periksa sebelum memanggil: kombinasi blocked memunculkan error saat
        preflight

        # dan pesan memuat alasan ini.

        blocked = get_support("domedetr", "detect", "onnx")

        print(blocked.tier)

        print(blocked.reason)
source_hash: 83de3289634888c6
---

## Bentuk matriks

Matriks menggunakan key `(family, task, format)`. Key family adalah nama
kanonis dari registry model, key task berasal dari `libreyolo.tasks.TASKS`, dan
terdapat dua belas format:

`onnx`, `torchscript`, `executorch`, `tensorrt`, `openvino`, `paddle`, `mnn`,
`rknn`, `ncnn`, `tflite`, `coreml`, `coreai`.

`model.export(format=...)` juga menerima dua alias: `engine` untuk `tensorrt`,
dan `litert` untuk `tflite`, yang merupakan nama terkini TensorFlow Lite.
Format dan suffix `.tflite` tidak berubah.

<code-tabs name="usage" />

Karena sebuah cell merupakan fungsi dari tiga key, grid lengkap berukuran besar
dan berubah pada setiap release. Grid dihasilkan, bukan ditulis manual, dan
berada di `docs/export_support.md` dalam repository library. Lakukan query pada
matriks dari Python atau CLI alih-alih membaca salinannya.

## Tiga tier

| Tier | Arti |
|---|---|
| `validated` | Paritas numerik dicakup dalam CI atau run nightly yang terdokumentasi |
| `available` | Konversi telah diimplementasikan, tetapi bukti paritas runtime numerik belum dicatat |
| `blocked` | Preflight memunculkan `NotImplementedError` beserta alasan sebelum tracing |

Kombinasi validated dan available sama-sama berjalan tanpa konfirmasi atau
peringatan umum. Bukti dan constraint yang tercatat tetap terlihat dalam
dokumentasi yang dihasilkan. Kombinasi blocked gagal sebelum pemeriksaan
dependency, pemuatan kalibrasi, tracing, atau pembuatan artefak.

Penambahan entri validated memerlukan pengujian paritas dan field `since`.

`SupportEntry` memiliki empat field: `tier`, string `reason`, release `since`,
dan string `constraint`. Constraint adalah bagian yang penting saat integrasi:
tanda centang hanya berlaku pada kondisi yang disebutkan, biasanya canvas input
tetap, batch 1, FP32, dan versi runtime tertentu.

## Cara sebuah cell ditentukan

`get_support(family, task, fmt)` melakukan resolve dalam urutan berikut. Aturan
pertama yang cocok akan digunakan.

1. Task yang tidak dikenal atau format di luar dua belas format mengembalikan `blocked`.
2. Entri eksplisit `(family, task, format)` dikembalikan sebagaimana tercatat.
3. Block seluruh family mengembalikan `blocked` dengan alasan family tersebut.
4. Block seluruh task mengembalikan `blocked` dengan alasan task tersebut.
5. Untuk `ncnn`, family pada list block NCNN mengembalikan `blocked`.
6. `mnn` mengembalikan `blocked`: tidak ada kontrak runtime untuk family dan task ini.
7. `rknn` mengembalikan `blocked`. RKNN dalam versi ini terbatas pada varian deteksi yang tepat diuji dalam simulator: YOLO9-t, YOLO9-E2E-t, YOLO-NAS-s, dan PicoDet-s pada RK3588.
8. `tensorrt` dan `openvino` mengembalikan `available`: jalur converter tersedia, tetapi paritas runtime belum dicatat untuk family dan task tersebut.
9. `tflite`, `paddle`, `coreai`, dan `coreml` mengembalikan `blocked`, masing-masing dengan alasannya sendiri.
10. Selain itu mengembalikan `available`: konversi telah diimplementasikan, tetapi paritas runtime numerik belum dicatat.

Asimetri pada langkah 8 sampai 10 disengaja. TensorRT dan OpenVINO melakukan
konversi generik dari ONNX, sehingga kombinasi yang tidak tercantum layak
dicoba. TFLite, Paddle, Core AI, dan CoreML masing-masing memerlukan jalur per
family, sehingga kombinasi yang tidak tercantum ditolak, bukan diundang untuk
dicoba.

## Task yang diblokir

Task berikut diblokir untuk setiap family tanpa entri eksplisit.

| Task | Alasan |
|---|---|
| `ocr` | Dua network dengan cropping dinamis per region tidak sesuai dengan kontrak ekspor satu graph |
| `point` | Family belum dihubungkan ke kontrak heatmap titik bersama dan peak-decoding backend |
| `semantic` | Family belum dihubungkan ke kontrak dense-logits bersama dan argmax backend |
| `mesh` | Output graph body-mesh, metadata, dan kontrak runtime belum didefinisikan |
| `normal` | Family belum dihubungkan ke kontrak dense unit-normal canvas tetap dan renormalization backend |
| `panoptic` | Ekspor panoptic tidak memiliki kontrak runtime backend |
| `gaze` | Family belum dihubungkan ke kontrak logits two-head bersama dan expectation-decoding backend |

Entri eksplisit menimpa aturan ini, yang memungkinkan family semantic yang telah
terhubung untuk tetap diekspor.

## Family yang diblokir

| Family | Diblokir untuk |
|---|---|
| `depth_anything3` | Setiap format; graph depth-nya tidak termasuk dalam kontrak runtime hasil ekspor |
| `domedetr` | Setiap format. PAQI menetapkan jumlah query per gambar, sehingga graph hasil tracing hanya valid untuk gambar sumber tracing. Gunakan D-FINE untuk DETR yang dapat diekspor |
| `eomt` | Ekspor instance dan panoptic, yang tidak memiliki parsing runtime |
| `l2cs` | Semua selain ONNX, TorchScript, ExecuTorch, TensorRT, dan OpenVINO |
| `hrnet` | Semua selain ONNX, TorchScript, OpenVINO, dan TensorRT |
| `sam`, `sam2`, `sam3`, `edgetam`, `mobilesam` | Setiap format; ekspor promptable model berada di luar cakupan kontrak runtime v1 |
| `grounding_dino`, `owlv2`, `omdet_turbo`, `ov_deim` | Setiap format; ekspor runtime open-vocabulary berada di luar cakupan v1 |
| `florence2`, `kosmos2`, `lfm2vl`, `internvl3`, `qwen3vl`, `smolvlm2`, `locateanything` | Setiap format; ekspor VLM generatif berada di luar cakupan v1 |

PicoSAM3 adalah pengecualian dalam tier promptable: model ini mengekspor network
ROI mentah 96 piksel ke ONNX.

## Diblokir untuk NCNN

Decoder bergaya DETR memerlukan operasi sampling yang tidak diimplementasikan
NCNN, sehingga family berikut diblokir untuk `ncnn` kecuali ada entri eksplisit
yang menyatakan sebaliknya: Deformable DETR, DETR, DINO-DETR, D-FINE, LW-DETR,
DEIM, DEIMv2, RT-DETR, RT-DETRv2, RT-DETRv4, RF-DETR, dan EC. Penolakan menyebut
ONNX, OpenVINO, TorchScript, dan TensorRT sebagai alternatif.

## Ambang batas paritas

Cell validated berarti artefak hasil ekspor mereproduksi model native dalam
batas berikut:

| Kelompok task | Ambang batas |
|---|---|
| Deteksi dan OBB | IoU box yang cocok di atas 0.95, MAE score di bawah 0.01 |
| Segmentation dan panoptic | IoU mask di atas 0.95 |
| Pose | L2 keypoint di bawah 2 piksel pada resolusi native |
| Classification | Cosine logits di atas 0.999 dan kelas top-1 sama |
| Depth dan restoration | PSNR di atas 40 dB terhadap output native |
| Surface normal | Mean angular error di bawah 0.1 derajat |
| Point | Lokasi peak sama dalam satu cell output |

Baris query DETR merupakan kumpulan tanpa urutan, sehingga paritas family DETR
menyelaraskan baris query sebagai kumpulan, bukan berdasarkan posisi.

## Melakukan ekspor

<code-tabs name="export" />

Kombinasi blocked memunculkan `NotImplementedError` saat preflight dan pesannya
memuat alasan yang tercatat. `validated_alternatives(family, task)`
mengembalikan format yang tervalidasi untuk pasangan tersebut, yang berguna
untuk dicetak bersama penolakan.

Argumen yang digunakan bersama oleh setiap exporter tercantum pada
[halaman API model](/docs/reference/model-api). Argumen khusus format berada
pada halaman setiap format.

## Membaca constraint

Cell validated adalah klaim mengenai satu konfigurasi terukur, bukan format
secara umum. String constraint seperti `FP32, batch 1, fixed 520x520 input`
berarti paritas dicatat pada bentuk dan presisi tersebut. Ekspor dengan resolusi
atau ukuran batch lain tetap menghasilkan artefak, tetapi bukan konfigurasi
yang menjadi sumber angka tersebut.
