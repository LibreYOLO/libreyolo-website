---
title: Pengenalan wajah
seo_title: Pengenalan wajah di LibreYOLO
description: >-
  Deteksi, buat embedding, dan identifikasi wajah di LibreYOLO. Daftarkan
  gallery, bandingkan dua gambar, dan lakukan pencocokan dengan cosine
  similarity dari Python atau CLI.
lead: >-
  Pengenalan wajah adalah task embed yang diterapkan pada wajah. Detektor
  melokalisasi dan menyelaraskan setiap wajah, head pengenalan mengembalikan
  vektor ternormalisasi L2 per wajah, dan identitas ditentukan oleh cosine
  similarity terhadap referensi terdaftar, bukan list kelas tetap.
keywords:
  - pengenalan wajah python
  - face embedding
  - verifikasi wajah
  - face gallery
  - arcface onnx
  - task embed libreyolo
  - cosine similarity wajah
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Nama librefacerec-* diarahkan ke family face-embedding tanpa memedulikan
        # suffix berkas, lalu diunduh dari organisasi Hugging Face LibreYOLO saat
        # penggunaan pertama bersama detektor wajah default.
        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)             # bounding box wajah (N, 4)
        print(result.embeddings.data.shape)  # (N, D), satu baris per wajah
        print(result.embeddings.dim)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=photo.jpg
    - label: Bandingkan dua gambar
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        # Menjalankan deteksi dan embedding pada kedua gambar lalu membandingkan
        # wajah dengan confidence tertinggi. Cosine similarity berada dalam [-1, 1].
        outcome = model.verify("person_a.jpg", "person_b.jpg", threshold=0.4)
        print(outcome["similarity"], outcome["same_person"])
    - label: Daftarkan gallery dan identifikasi
      language: python
      code: |
        from libreyolo import Gallery, LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        gallery = Gallery(model)
        gallery.enroll("ada", ["people/ada/1.jpg", "people/ada/2.jpg"])
        gallery.enroll("grace", "people/grace/1.jpg")
        gallery.save("faces.npz")

        result = model("group_photo.jpg", gallery=gallery, threshold=0.4)
        for name, score in result.identities.data:
            print(name, score)   # name bernilai None di bawah ambang batas
    - label: Daftarkan dan identifikasi dari CLI
      language: bash
      code: |
        libreyolo enroll model=librefacerec-l.onnx source=people/ gallery=faces.npz
        libreyolo predict model=librefacerec-l.onnx source=group_photo.jpg gallery=faces.npz
    - label: Gunakan bounding box wajah sendiri
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("librefacerec-l.onnx")

        # face_boxes melewati deteksi; face_detector menerima callable, model
        # deteksi LibreYOLO, atau instance FaceDetector.
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])
        print(result.embeddings.data.shape)
source_hash: d7dfcb6f812ebb2d
---

## Definisi

Pengenalan wajah mengembalikan vektor per wajah, bukan label. Prediksi
menjalankan dua tahap: detektor wajah melokalisasi setiap wajah dan lima
landmark-nya, crop ditransformasi ke alignment kanonis 112x112, lalu head pengenalan
menghasilkan embedding ternormalisasi L2.

`result.embeddings` adalah payload `Embeddings` berbentuk `(N, D)`, dengan baris
selaras terhadap `result.boxes`, sehingga baris `i` menjelaskan wajah dalam bounding box
`i`. Karena setiap baris adalah vektor satuan, cosine similarity merupakan dot
product, dan `embeddings.similarity()` menghitungnya terhadap `Embeddings` lain
atau seluruh matriks dalam satu pemanggilan.

Pemberian nama wajah merupakan langkah terpisah. `Gallery` menyimpan vektor
referensi bernama; memberikan `gallery=` kepada `predict()` melampirkan
`result.identities`, dengan baris selaras terhadap embedding serta nama dan
skor cosine terbaik per wajah. Wajah di bawah ambang batas kecocokan tetap
memiliki nama `None`, dan nama terdekat di bawah ambang batas tidak pernah
digunakan sebagai pengganti.

Kunci task kanonis library adalah `embed`. `face-recognition`,
`facial-recognition`, `reid`, dan `face` semuanya dinormalisasi ke sana, sehingga
`task="face-recognition"` dan `task="embed"` memilih hal sama. Wajah adalah
bentuk region dari task yang lebih luas; [embedding](/docs/tasks/embeddings)
menjelaskan bentuk seluruh gambar dan teks, API `Embeddings`, `Identities`, serta
`Gallery` bersama, dan model yang menghasilkan vektor tanpa deteksi.

## Model

[LibreFaceRec](/docs/models/librefacerec) adalah family untuk task ini. Model
terdiri dari dua artefak ONNX di balik satu pemanggilan: `librefacerec-l.onnx`,
head pengenalan iResNet100 yang menghasilkan embedding 512-d, dan
`librefacerec-det.onnx`, detektor wajah default dengan lima landmark dari
OpenCV zoo. Keduanya diunduh dari organisasi Hugging Face LibreYOLO saat
penggunaan pertama. Berkas ONNX lain dengan konvensi ArcFace (input 112x112
selaras, output `(N, D)`) dapat mengganti head pengenalan dengan memberikan
path-nya.

Kunci task `embed` lebih luas daripada wajah. [CLIP](/docs/models/clip),
[SigLIP2](/docs/models/siglip2), dan [DINOv2](/docs/models/dinov2) juga
mendukung `task="embed"` dan mengembalikan satu vektor seluruh gambar, yaitu
gambar retrieval, bukan identitas wajah. Ketiganya berbagi API `Gallery` dan
`Embeddings`, sehingga workflow pendaftaran dan pencocokan tetap berlaku, tetapi
tidak mendeteksi atau menyelaraskan wajah.

Head pengenalan berjalan melalui `onnxruntime`, yang tidak disertakan instalasi
dasar:

```bash
pip install "libreyolo[onnx]"
```

## Prediksi

<code-tabs name="predict" />

Jika dibiarkan, `predict()` mengunduh dan memasangkan detektor default.
`face_detector` menimpanya dengan callable, model deteksi LibreYOLO, atau
instance `FaceDetector`, serta dapat ditetapkan pada constructor atau per
pemanggilan. `face_boxes` melewati deteksi menggunakan bounding box yang sudah ada. Pada
CLI, `face_detector=` menerima path `.onnx` detektor wajah atau nama detektor
LibreYOLO.

`model.verify(image_a, image_b)` adalah shortcut dua gambar: metode ini membuat
embedding wajah dengan confidence tertinggi dari masing-masing dan mengembalikan
`{"similarity", "same_person", "threshold"}`. `model.embed(sources)`
mengembalikan setiap baris wajah lintas satu atau lebih gambar yang ditumpuk
menjadi tensor `(N_total, D)`. Lihat [prediksi](/docs/predict) untuk sumber,
streaming, dan penanganan hasil.

## Format dataset

Pendaftaran membaca satu folder per identitas. Nama folder menjadi identitas,
dan setiap gambar di dalamnya menambahkan referensi untuk nama tersebut:

```text
people/
  ada/
    1.jpg
    2.jpg
  grace/
    1.jpg
```

`libreyolo enroll` menelusuri tree tersebut dan menulis gallery `.npz`. Berkas
gallery yang sudah ada diperluas di tempat, bukan diganti, sehingga identitas
dapat ditambahkan bertahap. Gallery terikat pada bobot pembuatnya berdasarkan
dimensi embedding dan fingerprint berkas; pencocokan dengan model berbeda
memunculkan error, bukan membandingkan ruang vektor yang tidak kompatibel.

Secara default, setiap gambar sumber menyumbang satu baris referensi, yaitu
wajah dengan confidence tertinggi, sehingga portrait dengan orang lain hanya
mendaftarkan subjek. Berikan `select="all"` kepada `Gallery.enroll` untuk
menyimpan setiap baris.

## Pelatihan

Tidak ada family dalam task ini yang berlatih di LibreYOLO.
`LibreFaceEmbedder.train()` memunculkan error: latih head pengenalan di upstream,
ekspor ke ONNX dalam konvensi ArcFace, lalu muat berkas berdasarkan path.

## Validasi

Tidak ada validator dataset untuk task ini dan `val()` memunculkan error.
Akurasi verifikasi diukur pada pasangan gambar berlabel dengan `model.verify()`,
melakukan sweep `threshold` untuk memilih titik operasi. Akurasi identifikasi
diukur dengan mendaftarkan gallery lalu membaca `result.identities.name` dan
`result.identities.score` pada gambar held-out, dengan nama `None` dihitung
sebagai penolakan.

## Ekspor

Head pengenalan sudah berupa graph ONNX, sehingga tidak ada yang dikonversi:
`LibreFaceEmbedder.export()` memunculkan error. Deploy berkas `.onnx` secara
langsung, atau arahkan LibreYOLO ke berkas tersebut agar family menangani deteksi,
alignment, dan normalisasi.


