---
title: LibreFaceRec
families:
  - facerec
seo_title: 'LibreFaceRec: pengenalan dan verifikasi wajah'
description: >-
  Gunakan LibreFaceRec di LibreYOLO untuk deteksi wajah, embedding, dan
  verifikasi. Instal dan prediksi. Bobot embedding berlisensi Apache-2.0.
lead: >-
  LibreFaceRec adalah task embedding wajah LibreYOLO: detector wajah menemukan
  dan menyelaraskan wajah, lalu head pengenalan menghasilkan embedding identitas
  ternormalisasi L2 untuk verifikasi atau pencarian.
keywords:
  - LibreFaceRec
  - pengenalan wajah Python
  - face embedding
  - verifikasi wajah
  - ArcFace
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Nama librefacerec-* diarahkan ke family ini terlepas dari akhiran
        berkas

        # dan diunduh dari organisasi LibreYOLO di Hugging Face saat pertama
        kali

        # digunakan, bersama dengan detector wajah default.

        model = LibreYOLO("librefacerec-l.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.embeddings.data.shape)   # (N, D), dinormalisasi L2
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=face.jpg
    - label: Verifikasi
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        # Membandingkan wajah paling menonjol dalam setiap gambar melalui cosine
        # similarity embedding ternormalisasi L2.
        result = model.verify("person_a.jpg", "person_b.jpg", threshold=0.4)
        print(result["similarity"], result["same_person"])
    - label: Pencarian galeri
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        query = model("query.jpg").embeddings          # wajah dalam gambar ini
        gallery = model.embed(["a.jpg", "b.jpg", "c.jpg"])   # (N_total, D)

        # Cosine similarity (query_faces, N_total).
        scores = query.similarity(gallery)
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")
        model.export(format="onnx")
source_hash: f1a345bb96e32f12
---

## Instalasi

Head pengenalan LibreFaceRec berjalan melalui `onnxruntime`, yang tidak termasuk
dalam instalasi dasar.

```bash
pip install "libreyolo[onnx]"
```

## Prediksi

<code-tabs name="predict" />

Deteksi dan pengenalan adalah dua graph ONNX terpisah di balik satu panggilan.
Detector wajah menemukan serta menyelaraskan setiap wajah ke crop kanonis, lalu
head pengenalan mengembalikan satu embedding ternormalisasi L2 per wajah. Jika
dibiarkan, `predict()` mengunduh dan memasangkan detector default bawaan secara
otomatis. `face_detector` menerima callable, model deteksi LibreYOLO, atau
instance `FaceDetector`. `face_boxes` sepenuhnya melewati deteksi dengan memakai
box yang sudah tersedia. `result.embeddings` memuat satu baris per wajah yang
terdeteksi dan disejajarkan dengan `result.boxes`. Metode `.similarity()`-nya
menghitung cosine similarity terhadap embedding lain atau seluruh galeri dalam
satu panggilan. Untuk membandingkan dua gambar secara langsung, bukan dua
embedding yang sudah dihitung, `model.verify(image_a, image_b)` menjalankan
deteksi dan embedding pada keduanya, lalu membandingkan wajah dengan confidence
tertinggi. Model pengenalan ONNX lain yang mengikuti konvensi ArcFace (crop
selaras sebagai input, embedding `(N, D)` sebagai output) dapat digunakan
sebagai pengganti dengan meneruskan path berkasnya alih-alih nama
`librefacerec-*`. Lihat [prediksi](/docs/predict) untuk sumber, streaming, dan
penanganan hasil.

## Ekspor

<export-matrix />

LibreFaceRec sudah membungkus graph ONNX yang telah diekspor. Ekspor ulang ke
format lain belum diimplementasikan.

## Lisensi

<provenance-box>

Detector wajah default bawaan merupakan artefak kedua dengan lisensi kedua,
yaitu YuNet milik OpenCV Zoo, berlisensi MIT, hak cipta Shiqi Yu. Tidak ada kode
arsitektur yang di-port dari kedua proyek. Kedua graph digunakan secara opak
melalui `onnxruntime`, sehingga wrapper LibreYOLO sendiri tidak memuat kode pihak
ketiga dan seluruhnya berlisensi MIT.

</provenance-box>

## Sitasi

<citation-block />
