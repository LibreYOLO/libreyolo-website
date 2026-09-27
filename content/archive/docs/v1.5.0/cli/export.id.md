---
title: libreyolo export
seo_title: referensi perintah libreyolo export
description: >-
  Mengekspor checkpoint ke format deployment: setiap argumen beserta nilai
  bawaannya, ke mana artefaknya ditulis, dan kombinasi yang ditolak perintah
  ini.
lead: >-
  Mengubah satu checkpoint menjadi satu format deployment dan menulis artefaknya
  di weights/. Format menentukan argumen mana di bawah ini yang berlaku.
keywords:
  - libreyolo export cli
  - libreyolo export command
  - export yolo ke onnx
  - cara export yolo tensorrt
  - argumen libreyolo export
last_verified: 1.5.0
meta:
  - label: Perintah
    value: libreyolo export
    mono: true
  - label: Wajib
    value: model
    mono: true
  - label: Keluaran
    value: 'weights/<checkpoint-stem>[_fp16|_int8]<format-suffix>'
    mono: true
snippets:
  examples:
    - label: Dasar
      language: bash
      code: |
        # Menulis weights/LibreYOLO9s.onnx
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640
    - label: NMS di dalam graph
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx \
          nms=true conf=0.25 iou=0.45 max_det=300
    - label: Menjalankan artefaknya
      language: bash
      code: >
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640


        # Factory merutekan berdasarkan sufiks berkas, jadi hasil ekspor dimuat
        seperti checkpoint.

        libreyolo predict model=weights/LibreYOLO9s.onnx \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: ef2ca20af3814109
---

## Sinopsis

```bash
libreyolo export model=<name|path> [format=<format>] [key=value ...]
```

Argumen berupa pasangan `key=value`, dan bentuk POSIX juga berlaku, jadi
`format=onnx` dan `--format onnx` adalah argumen yang sama.

## Argumen

| Argumen | Bawaan | Arti |
|---|---|---|
| `model` | | Bobot model `.pt`. Wajib |
| `format` | `onnx` | Format ekspor: `onnx`, `torchscript`, `executorch`, `tensorrt`, `openvino`, `paddle`, `mnn`, `rknn`, `ncnn`, `tflite`, `coreml`, `coreai` |
| `name` | | Platform target RKNN, saat ini hanya `rk3588`. Ditolak bila dipakai dengan format lain |
| `imgsz` | | Ukuran gambar masukan: `640` atau `480x640` (HxW). `480,640` juga diterima. Memakai ukuran milik model sendiri bila tidak diatur |
| `batch` | `1` | Ukuran batch ekspor |
| `half` | `false` | Presisi FP16 |
| `int8` | `false` | Kuantisasi INT8 |
| `dynamic` | `false` | Bentuk masukan dinamis (ONNX) |
| `simplify` | `true` | Penyederhanaan graph ONNX |
| `nms` | `false` | Menyematkan NMS di dalam model. Hanya ONNX dan CoreML |
| `conf` | `0.25` | Ambang batas skor keyakinan untuk NMS yang disematkan |
| `iou` | `0.45` | Ambang IoU untuk NMS yang disematkan |
| `max_det` | `300` | Deteksi maksimum untuk NMS ONNX yang disematkan |
| `opset` | | Versi opset ONNX. Dipilih otomatis bila tidak diatur |
| `data` | | Data kalibrasi untuk INT8 |
| `fraction` | `1.0` | Bagian data kalibrasi yang dipakai |
| `device` | `auto` | Perangkat untuk tracing |
| `allow_download_scripts` | `false` | Mengizinkan Python yang tertanam di blok download pada YAML dataset |
| `json` | `false` | Keluaran JSON ke stdout |
| `quiet` | `false` | Menyembunyikan stderr |
| `verbose` | `false` | Log ekspor yang detail |
| `verify` | `false` | Menjalankan simulator PC RKNN Toolkit2 dan membandingkan hasilnya dengan ONNX Runtime. Hanya RKNN |
| `help_json` | `false` | Menampilkan skema perintah sebagai JSON lalu keluar |

`engine` adalah alias untuk `tensorrt` dan `litert` alias untuk `tflite`.
Keduanya diselesaikan menjadi nama kanonis sebelum apa pun ditulis, jadi keluaran
JSON dan baris log selalu melaporkan `tensorrt` atau `tflite`.

## Contoh

<code-tabs name="examples" />

## Catatan

### Ke mana berkasnya ditulis

Perintah ini tidak menerima path keluaran. Artefak ditulis ke `weights/`, dinamai
sesuai stem checkpoint sumber ditambah sufiks format, dengan sisipan `_fp16` atau
`_int8` bila salah satu presisi itu diminta. `LibreYOLO9s.pt` yang diekspor ke
ONNX pada FP16 menjadi `weights/LibreYOLO9s_fp16.onnx`. Hasil JSON memuat
`output_path` yang sudah diselesaikan, ukuran berkas dalam MB, dan bentuk masukan
sebagai `[batch, 3, height, width]`.

### Kombinasi yang ditolak

`nms=true` diterima untuk ONNX dan CoreML, dan ditolak untuk semua format lain
dengan `nms_unsupported_format`. Pada ONNX, flag ini mematikan `dynamic`, karena
graph yang disematkan terkunci pada batch 1, dan hal itu disebutkan di stderr.
Pada CoreML, flag ini menerima `conf` dan `iou` tetapi tidak `max_det`, jadi
`max_det` non-bawaan bersama `format=coreml nms=true` akan keluar dengan
`config_unsupported`.

`half=true` bersamaan dengan `int8=true` bukan error. INT8 menang, `half`
diabaikan, dan peringatan dikirim ke stderr.

`name` dan `verify` saat ini adalah opsi RKNN. Memberikan salah satunya dengan
format lain akan keluar dengan `config_unsupported`, bukan diabaikan.

### Format apa saja yang didukung sebuah family

Dukungan bersifat per family dan per task, bukan global. `libreyolo formats
family=<family> task=<task>` mencetak tier setiap format untuk kombinasi
tersebut, beserta alasannya dan batasan yang menyertainya. Lihat
[`libreyolo formats`](/docs/cli/utilities) untuk argumennya.

Sebagian format memerlukan instalasi opsional dan sebagian lagi memerlukan
toolchain. Dependensi Python yang hilang membuat perintah keluar dengan
`export_dep_missing`; presisi yang tidak bisa dihasilkan format tersebut keluar
dengan `format_precision_unsupported`.

### Menjalankan hasil ekspor

Artefak hasil ekspor dimuat lewat model factory yang sama dengan checkpoint,
berdasarkan sufiks berkas, jadi `libreyolo predict model=weights/LibreYOLO9s.onnx`
berjalan tanpa konversi tambahan. Tiga opsi prediksi menjadi pengecualian dan
ditolak pada backend runtime: `tiling`, `overlap_ratio` dan `output_file_format`.

Dua target deployment punya halaman sendiri:
[NVIDIA DeepStream](/docs/export/deepstream) dan
[NVIDIA Jetson](/docs/export/jetson).

### Keluaran dan kode keluar

stdout membawa hasilnya; progres dikirim ke stderr. Kode keluarnya `0` bila
berhasil, `2` untuk kesalahan penggunaan atau konfigurasi, `4` bila model tidak
bisa dimuat, `5` untuk format yang tidak dikenal, dependensi ekspor yang hilang,
presisi yang tidak didukung atau permintaan NMS tersemat yang ditolak, dan `1`
untuk kegagalan runtime lainnya.

Terkait: [`libreyolo quantize`](/docs/cli/quantize), yang tetap berada di PyTorch
dan menulis checkpoint alih-alih artefak deployment.
