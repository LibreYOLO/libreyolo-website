---
title: SenseNova-Vision
families:
  - sensenovavision
seo_title: 'SenseNova-Vision di LibreYOLO: 7 task, satu checkpoint'
description: >-
  Gunakan SenseNova-Vision di LibreYOLO untuk deteksi, segmentasi, panoptik,
  pose, titik, kedalaman, dan OCR dari satu checkpoint generatif berbasis
  prompt.
lead: >-
  SenseNova-Vision adalah model multimodal terpadu yang mengubah task vision
  menjadi generasi berbasis prompt pada decoder bersama: kotak, titik, keypoint,
  dan kata OCR keluar sebagai teks bertag, sedangkan peta kedalaman, mask, dan
  panoptik keluar sebagai gambar yang dirender decoder. LibreYOLO memuatnya
  melalui LibreVLM dan mendukung tujuh task dari satu checkpoint 7B.
keywords:
  - SenseNova-Vision
  - SenseTime
  - unified multimodal model
  - Bagel
  - prompted detection
  - dense perception
  - referring segmentation
  - segmentasi panoptik
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="detect")
        model.set_classes(["bird", "boat"])
        result = model.predict("image.jpg")
        print(result.boxes.xyxy)

        # set_task() mengganti task pada model yang sama dan sudah dimuat.
        model.set_task("depth")
        result = model.predict("image.jpg")
        depth = result.depth_map.data
    - label: Segmentasi referring dan panoptik
      language: python
      code: >
        from libreyolo import LibreVLM


        model = LibreVLM("sensenova-vision", task="segment")

        # Segmentasi bersifat referring: memerlukan frasa target, bukan daftar
        kelas.

        model.set_classes(["the person furthest to the right"])

        result = model.predict("street.jpg")

        mask = result.masks.data[0]


        model.set_task("panoptic")

        # Tanpa kosakata khusus, panoptik kembali ke kategori panoptik COCO

        # yang digunakan untuk fine-tuning checkpoint.

        result = model.predict("street.jpg")

        segment_map = result.panoptic.data

        for segment in result.panoptic.segments_info:
            print(segment)
    - label: 'Titik, pose, dan OCR'
      language: python
      code: |
        from libreyolo import LibreVLM

        model = LibreVLM("sensenova-vision", task="point")
        model.set_classes(["screw"])
        result = model.predict("board.jpg")
        print(result.points.xy)

        # Tanpa kosakata yang ditetapkan, pose kembali ke "person".
        model.set_task("pose")
        result = model.predict("gym.jpg")
        print(result.boxes.xyxy, result.keypoints.data.shape)

        model.set_task("ocr")
        result = model.predict("sign.jpg")
        print(result.ocr.texts)
source_hash: 8749277e1910baa4
---

## Instalasi

SenseNova-Vision memerlukan komponen tambahannya sendiri, yang memasang `accelerate` untuk
dispatch model besar yang dibutuhkan checkpoint ini dan, pada platform selain macOS,
`bitsandbytes` untuk pemuatan 4-bit.

```bash
pip install "libreyolo[sensenova]"
```

Checkpoint dicerminkan di Hugging Face dalam organisasi LibreYOLO sendiri dan diunduh
secara otomatis saat pertama kali digunakan; lisensinya CC BY-NC 4.0, hanya untuk penggunaan
nonkomersial, dan loader mencetak pemberitahuan tersebut sebelum setiap pengunduhan otomatis.
Lihat Lisensi di bawah.

## Prediksi

<code-tabs name="predict" />

Setiap prediksi adalah dekode difusi melalui backbone Bagel-MoT bersama, sehingga ini
merupakan model kapabilitas, bukan model real-time: latensi per gambar akan jauh lebih tinggi
daripada detektor atau segmenter khusus. `dtype="auto"` (default) memuat bf16 pada GPU dengan
memori cukup dan kembali ke kuantisasi NF4 4-bit di tempat lain, yang memerlukan
`bitsandbytes`; berikan `dtype="bf16"` untuk memaksa presisi penuh pada GPU yang cukup besar.
`noise_seed=42` saat konstruksi memberikan seed pada sampler difusi agar output padat dapat
direproduksi; berikan `noise_seed=None` untuk menonaktifkan seed.

Ketujuh task memakai satu checkpoint yang sudah dimuat: `set_task()` berpindah di antaranya
tanpa memuat ulang. `set_classes()` menetapkan kosakata aktif; deteksi, titik, pose, dan
panoptik menerima daftar kelas, sedangkan segmentasi bersifat referring dan memerlukan frasa
yang tepat untuk diisolasi. Setiap task mengembalikan objek `Results` standar dengan payload
berbeda yang terisi: `boxes` untuk detect, `points` untuk point, `boxes` dan `keypoints`
untuk pose, `ocr` untuk OCR, `depth_map` untuk depth, `masks` untuk segment, serta
`panoptic` (dengan `segments_info`) untuk panoptic. Lihat [prediksi](/docs/predict) untuk
sumber, streaming, dan penanganan hasil.

## Checkpoint

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
