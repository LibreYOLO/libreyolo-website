---
title: LibreMODUS
families:
  - libremodus
seo_title: 'LibreMODUS di LibreYOLO: analisis gambar any-to-any'
description: >-
  Gunakan LibreMODUS di LibreYOLO untuk kedalaman, normal, edge, dan deteksi,
  lalu komposisikan semuanya dengan any2any(). Hanya inferensi. Bobot dimuat
  dari EPFL-VILAB.
lead: >-
  LibreMODUS adalah integrasi khusus inferensi dari checkpoint MODUS 14B-A7B,
  model any-to-any yang mengubah satu input turunan gambar menjadi turunan lain:
  RGB masuk, kedalaman keluar; kedalaman masuk, normal keluar; salah satunya
  ditambah frasa, box keluar. LibreYOLO mendukung empat task melalui API
  prediksi standar dan set yang lebih luas melalui any2any().
keywords:
  - LibreMODUS
  - MODUS
  - any-to-any
  - estimasi kedalaman
  - surface normals
  - deteksi tepi
  - referring detection
  - EPFL VILAB
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreMODUS

        model = LibreMODUS(size="14b-a7b", task="normal")
        result = model.predict("room.jpg")
        normals = result.normal_map.data

        model.set_task("edge")
        result = model.predict("room.jpg")
        edges = result.edges.data

        # Tanpa vocabulary khusus, detect mendekode token label checkpoint
        # menjadi ID kelas COCO-80 yang berurutan.
        model.set_task("detect")
        result = model.predict("street.jpg")
        print(result.boxes.xyxy)
    - label: Grounding frasa
      language: python
      code: >
        from libreyolo import LibreMODUS


        model = LibreMODUS(task="detect")

        # set_classes() mengalihkan deteksi ke grounding frasa: setiap frasa

        # berjalan secara independen dan dikembalikan melalui kontrak Boxes yang
        sama.

        model.set_classes(["red bus", "cyclist"])

        result = model.predict("street.jpg", conf=0.2)

        print(result.boxes.xyxy, result.boxes.cls)
    - label: any2any()
      language: python
      code: >
        from libreyolo import LibreMODUS


        model = LibreMODUS()


        # Satu hingga tiga input turunan gambar (rgb, depth, normal,
        canny/edge),

        # ditambah teks bantu opsional, dikomposisikan menuju satu target.

        result = model.any2any(
            inputs={"rgb": "room.jpg"},
            target="normal",
            steps=10,
            cfg=2.0,
            seed=0,
        )

        normals = result.normal_map.data


        # Grounding melalui any2any() memerlukan input teks yang menyebutkan
        frasa.

        result = model.any2any(
            {"rgb": "street.jpg", "text": "red bus"},
            target="grounding",
        )

        print(result.boxes.xyxy)
source_hash: 7386886d4c36ea9a
---

## Instalasi

LibreMODUS memerlukan extra tersendiri, yang memasang `accelerate` untuk dispatch
model besar yang diperlukan checkpoint ini.

```bash
pip install "libreyolo[modus]"
```

LibreYOLO tidak mendistribusikan ulang atau mencerminkan bobot MODUS. Secara
default, memuat model `LibreMODUS` akan mengunduh berkas yang diperlukan
langsung dari `EPFL-VILAB/MODUS` pada revisi Hugging Face yang dipatok. Unduhan
baru selalu memerlukan akun Hugging Face terautentikasi milik pengguna, bahkan
jika gate hosting upstream sementara terbuka. Tinjau dan terima ketentuan
upstream, lalu lakukan autentikasi:

```bash
hf auth login
```

```python
from libreyolo import LibreMODUS

model = LibreMODUS(token="hf_...")
```

Untuk menghindari permintaan jaringan, arahkan ke snapshot yang sudah tersedia:

```python
model = LibreMODUS(checkpoint_path="/models/MODUS")
```

Direktori tersebut harus memuat `model.safetensors`, `ae.safetensors`,
`llm_config.json`, `vit_config.json`, `tokenizer_config.json`, `vocab.json`, dan
`merges.txt`. Lihat Lisensi di bawah untuk ketentuan penggunaan checkpoint.

## Prediksi

<code-tabs name="predict" />

API task standar mencakup empat task yang masing-masing dipetakan ke satu target
MODUS: `depth` ke kedalaman relatif (`result.depth_map`), `normal` ke surface
normal (`result.normal_map`), `edge` ke edge bergaya Canny (`result.edges`), dan
`detect` ke box COCO-80 (`result.boxes`), kecuali `set_classes()` mengalihkannya
ke grounding frasa. `set_task()` beralih di antaranya pada model yang sama.
Resep yang dirilis menggunakan sepuluh langkah flow sampling dengan panduan
teks 4.0 dan panduan gambar 2.0. Ganti nilainya dengan `inference_steps=`,
`inference_cfg=`, dan `inference_image_cfg=` saat konstruksi.

`any2any()` menjangkau permukaan analisis publik yang lebih luas: satu hingga
tiga input turunan gambar (`rgb`, `depth`, `normal`, `canny`/`edge`), ditambah
teks bantu opsional, yang dikomposisikan menuju salah satu dari kedalaman,
normal, edge, edge turunan SAM, deteksi COCO, atau grounding frasa. Semua input
turunan gambar harus menjelaskan kanvas selaras yang sama. LibreMODUS menolak
lebar dan tinggi yang tidak cocok alih-alih mengubah ukurannya secara terpisah.
`chain=(...)` menghasilkan target perantara dan memasukkannya kembali ke konteks
yang sama dalam anggaran pelatihan tiga kondisi milik checkpoint. `verify=N`
(N >= 2) menghasilkan N kandidat dan mempertahankan kandidat dengan skor
tertinggi pada pemeriksaan self-consistency terbatas, yang tersedia sebagai
`result.verification_score`.

`dtype="bf16"` (default) cocok dengan presisi checkpoint yang dirilis.
`dtype="fp8"` menyimpan bobot linear trunk decoder yang memenuhi syarat sebagai
E4M3 dengan skala per channel output, mengonversinya satu kali ke cache lokal di
`~/.cache/libreyolo/modus/fp8`, dan mendekuantisasi ke dtype input untuk setiap
perkalian matriks. Dengan demikian, opsi ini menukar penggunaan memori, bukan
akurasi pada tingkat aktivasi.

`train()`, `val()`, dan `export()` semuanya memunculkan error. LibreMODUS hanya
untuk inferensi, validasi dataset tidak disediakan, dan tidak ada jalur ekspor
ONNX, TensorRT, atau TFLite. `predict()` berbentuk batch dan augmentasi waktu
pengujian juga tidak didukung. Setiap panggilan menangani satu gambar.

## Lisensi

<provenance-box>

LibreYOLO tidak menghosting atau mencerminkan checkpoint MODUS di mana pun,
termasuk dalam organisasi Hugging Face miliknya sendiri. Memuat checkpoint
selalu mengambil revisi yang dipatok langsung dari EPFL-VILAB/MODUS atau membaca
snapshot yang sudah tersimpan di disk pada `checkpoint_path`.

</provenance-box>

## Sitasi

<citation-block />
