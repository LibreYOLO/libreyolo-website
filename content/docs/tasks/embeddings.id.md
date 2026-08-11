---
title: Embeddings
seo_title: Embedding gambar dan wilayah di LibreYOLO
description: >-
  Embed task menghasilkan vektor float32 yang dinormalisasi L2 untuk seluruh
  gambar, untuk setiap wilayah yang terdeteksi, atau untuk teks. Daftarkan
  galeri, cocokkan dengan kesamaan kosinus, dan cari dari Python atau CLI.
lead: >-
  Satu task mencakup setiap vektor yang dihasilkan LibreYOLO. embed
  mengembalikan baris float32 dengan panjang satuan yang produk titiknya adalah
  skor kesamaan, apakah baris menggambarkan seluruh gambar, satu wajah yang
  terdeteksi, atau satu baris teks, dan Galeri yang sama cocok untuk semuanya.
keywords:
  - embedding gambar python
  - L2 dinormalisasi embedding
  - pencarian kesamaan kosinus
  - libreyolo embed task
  - pengambilan gambar
  - pendaftaran galeri
  - embedding clip
  - embedding dinov2
  - embedding reid
last_verified: 1.5.0
verification: >-
  Task kunci dan alias dibaca dari libreyolo/tasks.py. Payload hasil dari kelas
  Embeddings dan Identities di libreyolo/utils/results.py. API Gallery dari
  libreyolo/utils/gallery.py. embed dan _postprocess_embeddings dari
  libreyolo/models/base/model.py. Keluarga yang didukung ditemukan dengan
  mencari libreyolo/models/**/model.py untuk embed di SUPPORTED_TASKS. Permukaan
  CLI dari libreyolo/cli/__init__.py, libreyolo/cli/commands/special.py dan
  libreyolo/cli/commands/predict.py. Niat desain dari
  docs/adr/0015-embed-generalization.md.
meta:
  - label: kunci Task
    value: embed
    mono: true
  - label: Aliases
    value: 'pengenalan wajah, reid, wajah'
    mono: true
  - label: Payload hasil
    value: 'Embeddings, Identities'
    mono: true
  - label: Tipe data Baris
    value: 'float32, panjang satuan'
snippets:
  predict:
    - label: Seluruh gambar
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # CLIP secara default untuk mengklasifikasikan, jadi mintalah vektor
        secara eksplisit.

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        result = model(SAMPLE_IMAGE)


        print(result.embeddings.data.shape)  # (1, 512), satu baris per gambar

        print(result.boxes)                  # None: tidak ada yang
        terlokalisasi
    - label: Per wilayah
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("librefacerec-l.onnx")
        result = model(SAMPLE_IMAGE)

        # Baris i menjelaskan wilayah di kotak i.
        print(result.boxes.xyxy.shape)       # (N, 4)
        print(result.embeddings.data.shape)  # (N, 512)
    - label: Banyak gambar sekaligus
      language: python
      code: |
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")

        # Setiap baris dari setiap hasil, digabungkan menjadi satu tensor.
        vectors = model.embed(["a.jpg", "b.jpg", "c.jpg"])
        print(vectors.shape)  # (3, 384)
    - label: Text
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")


        # Teks adalah metode, bukan sumber prediksi. Sebuah string yang
        diberikan ke

        # model(...) tetap merupakan jalur atau URL.

        text = model.embed_text(["a photo of a cat", "a photo of a dog"])

        print(text.shape)  # (2, 512)
  similarity:
    - label: Bandingkan dua set baris
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        query = model.embed("query.jpg")          # (1, 512)
        pool = model.embed(["a.jpg", "b.jpg"])    # (2, 512)

        # Baris memiliki panjang satu, jadi kesamaan kosinus adalah produk dot.
        scores = model("query.jpg").embeddings.similarity(pool)
        print(scores.shape)  # (1, 2)
    - label: Gambar terhadap teks
      language: python
      code: |
        import torch

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        image = model.embed("photo.jpg")                       # (1, 512)
        text = model.embed_text(["a cat", "a dog", "a car"])   # (3, 512)

        print(torch.matmul(image, text.T))
  gallery:
    - label: Daftar dan identifikasi
      language: python
      code: |
        from libreyolo import Gallery, LibreYOLO

        model = LibreYOLO("librefacerec-l.onnx")

        gallery = Gallery(model)
        gallery.enroll("ada", ["people/ada/1.jpg", "people/ada/2.jpg"])
        gallery.enroll("grace", "people/grace/1.jpg")
        gallery.save("refs.npz")

        result = model("group.jpg", gallery=gallery, threshold=0.4)
        for name, score in result.identities.data:
            print(name, score)   # nama adalah None di bawah ambang
    - label: Pencarian Top-k
      language: python
      code: |
        from libreyolo import Gallery
        from libreyolo.models.dinov2.model import LibreDINOv2

        model = LibreDINOv2(size="s", task="embed")
        gallery = Gallery.load("refs.npz", model=model)

        result = model("query.jpg")
        matches = gallery.match(result.embeddings, top_k=5, threshold=0.4)
        print(matches[0])   # [(nama, skor), ...] untuk baris pertama
    - label: Daftarkan vektor yang sudah Anda miliki
      language: python
      code: |
        from libreyolo import Gallery

        gallery = Gallery()
        gallery.enroll_embedding("ada", vector)  # dinormalisasi saat masuk
        print(gallery.identities, gallery.dim, len(gallery))
  cli:
    - label: Daftarkan pohon folder
      language: bash
      code: >
        # source/<identity>/*.jpg. Galeri yang ada diperluas di tempat.

        libreyolo enroll model=librefacerec-l.onnx source=people/
        gallery=refs.npz
    - label: Identifikasi sambil memprediksi
      language: bash
      code: |
        libreyolo predict model=librefacerec-l.onnx source=group.jpg \
          gallery=refs.npz gallery_threshold=0.45
    - label: Bandingkan dua gambar
      language: bash
      code: >
        libreyolo compare model=librefacerec-l.onnx \
          source=a.jpg source2=b.jpg threshold=0.4

        # verify adalah perintah yang sama dengan nama kedua.

        libreyolo verify model=librefacerec-l.onnx source=a.jpg source2=b.jpg
        --json
source_hash: ffbaad5599035bc7
---

## Definisi

`embed` mengubah gambar, wilayah dari gambar, atau string menjadi baris float32 dari
lebar tetap yang panjangnya satu. Karena setiap baris adalah vektor satuan, membandingkan
dua di antaranya adalah produk titik, dan membandingkan dua set dari mereka adalah satu matriks
perkalian. Tidak ada yang lain di task yang spesifik model: pengambilan,
deteksi duplikat, identifikasi ulang, dan pengenalan wajah semuanya sama
aritmatika di atas baris yang berbeda.

Vektor adalah keluaran. Tidak ada daftar kelas, jadi nama ditambahkan kemudian oleh
membandingkan dengan referensi yang Anda berikan daripada oleh apa pun yang dimiliki jaringan
dilatih untuk memprediksi.

### Tiga bentuk

| Bentuk | `Results.embeddings` | `Results.boxes` | Diproduksi oleh |
|---|---|---|---|
| Gambar utuh | `(1, D)` | `None` | Mengirim gambar ke family gambar utuh |
| Wilayah | `(N, D)` | `(N, 4)`, sejajar baris | Keluarga yang melakukan lokalisasi terlebih dahulu, seperti pengenalan wajah |
| Teks | sama sekali bukan `Results` | | `model.embed_text(texts)`, mengembalikan `(M, D)` |

Hasil gambar utuh tetap dua dimensi bahkan untuk satu gambar. `(D,)` bukanlah
bentuk pengembalian yang diizinkan, sehingga pengguna tidak perlu melakukan kasus khusus untuk baris tunggal.
Teks mengembalikan tensor biasa daripada `Results`, karena string
bukan sumber gambar: memberikan satu ke `model(...)` tetap berarti path atau URL,
dan perpustakaan tidak pernah menebak bahwa string adalah prosa.

Kunci task kanonik adalah `embed`. `embedding`, `embeddings`,
`face-recognition`, `facial-recognition`, `recognition`, `face`, `faceid` dan
`reid` semuanya dinormalisasi menjadi itu, sehingga `task="reid"` dan `task="embed"` memilih hal yang sama persis.


## Model

Empat keluarga melayani task, dan mereka terbagi dengan jelas berdasarkan apakah mereka memproses lokal terlebih dahulu
apa pun terlebih dahulu.

| Family | Bentuk | Dimensi | Juga mendukung |
|---|---|---|---|
| [LibreFaceRec](/docs/models/librefacerec) | Wilayah, satu baris per wajah yang terdeteksi | 512 | Tidak ada; `embed` adalah satu-satunya task |
| [CLIP](/docs/models/clip) | Gambar lengkap, dengan menara teks yang dipasangkan | 512 untuk `b32` dan `b16`, 768 untuk `l14` | `classify`, yang tetap default-nya |
| [SigLIP 2](/docs/models/siglip2) | Gambar lengkap, dengan menara teks yang dipasangkan | 768 untuk `b16`, 1152 untuk `so400m` | `classify`, yang tetap default-nya |
| [DINOv2](/docs/models/dinov2) | Gambar lengkap, hanya gambar | 384 | `semantic`, `classify` |

CLIP dan SigLIP 2 tetap menggunakan `classify` sebagai task default mereka, sehingga `task="embed"` memiliki
yang perlu ditanyakan. `-cls` checkpoint yang ada adalah artefak
dua-menara yang dibagikan; tidak ada `-embed` checkpoint duplikat yang diterbitkan untuk bobot yang identik.

`embed_text` hanya ada pada CLIP dan SigLIP 2, dua keluarga dengan menara teks
. DINOv2 tidak memiliki sama sekali. DINOv2 embedding melewati kepala semantik dan klasifikasi
dan membaca token CLS final yang ternormalisasi pada 224 piksel;
`n`, `s`, `m` dan `l` varian semuanya menggunakan encoder DINOv2-S yang sama, sehingga keempat
mengembalikan `D = 384`.

Backbone yang hanya untuk klasifikasi yang ditambahkan dalam rilis ini, [ViT](/docs/models/vit),
[Swin](/docs/models/swin) dan [DeiT](/docs/models/deit), nyatakan `classify` saja
dan jangan menyajikan task ini.

<code-tabs name="predict" />

`model.embed(source, **kwargs)` adalah pintasan batch: itu menjalankan `predict` dan
menggabungkan setiap baris dari setiap hasil menjadi satu `(N_total, D)` CPU float32
tensor, meningkatkan jika baris memiliki dimensi campuran. Sebuah family tanpa `embed` di
tugas yang didukungnya meningkatkan `NotImplementedError`.

## Muatan hasil

`result.embeddings` adalah muatan `Embeddings`. `data`-nya selalu `(N, D)`
float32, sudah dinormalisasi L2 oleh jalur inferensi, dan bukan dua dimensi
input menimbulkan kesalahan daripada dibentuk ulang secara diam-diam.

| Anggota | Makna |
|---|---|
| `.data` | Matriks `(N, D)` |
| `.dim` | `D` |
| `.normalized` | Baris yang sama, dinormalisasi ulang secara defensif |
| `.similarity(other)` | `(N, M)` terhadap set lain, atau `(N,)` terhadap satu vektor `(D,)` |
| `.verify(i, j, threshold=0.4)` | Apakah baris `i` dan `j` adalah subjek yang sama |

`result.identities` adalah payload `Identities`, hanya hadir ketika sebuah galeri dilewatkan.
Ini adalah wadah biasa, bukan tensor, sehingga memindahkan `Results` antar
perangkat tidak mengubahnya.

| Anggota | Makna |
|---|---|
| `.name` | Daftar nama, `None` di mana tidak ada yang melewati ambang |
| `.score` | `(N,)` float32 skor kosinus terbaik, tetap disimpan bahkan ketika nama adalah `None` |
| `.data` | Daftar tuple `(name, score)` |

<code-tabs name="similarity" />

Vektor secara default tidak termasuk dalam `summary()` dan `to_json()`, karena sebuah float 512
baris sekitar dua kilobita per subjek. Setiap baris melaporkan `embedding_dim`
sebagai gantinya, ditambah `identity` dan `identity_score` ketika sebuah galeri digunakan. Lulus
`summary(embeddings=True)` untuk menyertakan angka-angka.

## Galeri

`Gallery` adalah sekumpulan baris referensi yang diberi nama. Ini menyimpan setiap referensi
secara terpisah daripada meratanya, jadi sebuah nama dinilai berdasarkan yang terbaik tunggalnya
mencocokkan referensi, dan menambahkan foto yang buruk tidak dapat menyeret pusat identitas
sekitar.

<code-tabs name="gallery" />

`Gallery(model)` menempel pada bobot yang akan menghasilkan vektornya.
`enroll(name, sources, select="best")` menjalankan prediksi pada setiap sumber dan menyimpan
baris dengan kepercayaan tertinggi per hasil; `select="all"` tetap menyimpan setiap baris,
yang merupakan apa yang Anda inginkan ketika gambar referensi secara sah berisi beberapa
subjek. `enroll_embedding(name, vector)` melewatkan inferensi dan mengambil vektor
secara langsung, menormalkannya dan menolak baris yang semuanya nol.

`FaceGallery` adalah alias permanen dari kelas yang sama, dan arsip yang ditulis oleh
rilis sebelumnya yang hanya berfokus pada wajah masih dapat dimuat.

### Pencocokan dan ambang batas

Pencocokan adalah perkalian matriks padat terhadap setiap referensi yang disimpan,
dikurangi menjadi satu skor per nama dengan mengambil nilai maksimum. Tidak ada indeks perkiraan,
yang menjaga angka tetap tepat dan memberikan batas praktis pada ukuran galeri
.

Dua titik masuk berbeda dalam apa yang mereka lakukan di bawah ambang batas. `match()` mengembalikan
`[(name, score), ...]` per baris dengan semuanya yang berada di bawah ambang dibuang, jadi sebuah
baris tanpa kecocokan adalah daftar kosong. `identify()` mengembalikan payload `Identities`
yang selalu mempertahankan skor terbaik dan mengatur namanya menjadi `None` ketika berada di bawah
ambang. Tidak satu pun pernah menggantikan nama terdekat di bawah ambang.

Ambang batas default adalah `0.4` secara menyeluruh. Ini adalah nilai kosinus, bukan
probabilitas, dan titik operasi yang tepat adalah sifat dari data Anda dan Anda
toleransi terhadap kecocokan palsu, jadi lakukan penyaringan pada pasangan berlabel daripada langsung menerima
default. `libreyolo enroll` dan argumen prediksi `gallery=` menggunakan
nomor yang sama.

### Ketekunan

`save(path)` menulis `.npz` terkompresi yang memegang vektor, nama-nama, dan sebuah
blok metadata yang membawa versi format, dimensi embedding dan sebuah
sidik jari dari bobot yang menghasilkan baris. `Gallery.load(path,
model=...)` memeriksa keduanya sebelum membandingkan apa pun, jadi mengarahkan galeri pada sebuah
model yang berbeda meningkatkan alih-alih diam-diam memberi skor vektor dari dua yang tidak terkait
ruang terhadap satu sama lain. Menyimpan galeri kosong ditolak.

## Baris perintah

| Perintah | Tujuan |
|---|---|
| `libreyolo enroll` | Menelusuri pohon folder per identitas dan menulis atau memperluas galeri `.npz` |
| `libreyolo compare` | Menyematkan subjek utama dalam dua gambar dan melaporkan kesamaan kosinus |
| `libreyolo verify` | Perintah yang sama dengan nama kedua |
| `libreyolo predict gallery=...` | Menyertakan identitas ke dalam jalannya prediksi biasa |

<code-tabs name="cli" />

Setiap perintah LibreYOLO menerima baik `key=value` maupun `--key value`, jadi
`gallery=refs.npz` dan `--gallery refs.npz` adalah argumen yang sama.

`enroll` mengambil `model`, `source` dan `gallery`, plus opsional `face-detector`,
`device`, `--json` dan `--quiet`. Ini membaca satu folder per identitas, di mana
nama folder adalah identitas dan setiap gambar di dalamnya menyumbang referensi:

```text
people/
  ada/
    1.jpg
    2.jpg
  grace/
    1.jpg
```

Gambar yang tidak menghasilkan apa-apa dilewati dengan sebuah garis pada stderr daripada
menghentikan jalannya, dan ringkasan melaporkan berapa banyak referensi yang disimpan untuk
setiap nama. File galeri yang ada diperluas di tempat, sehingga identitas dapat
ditambahkan seiring waktu.

`compare` dan `verify` adalah satu fungsi yang didaftarkan dua kali. Mereka mengambil `model`,
`source`, `source2` dan opsional `threshold`, dan mencetak kesamaan kosinus,
keputusan sama atau berbeda dan ambang yang menghasilkan keputusan tersebut.
`--json` mencetak tiga bidang yang sama sebagai sebuah objek.

Pada `predict`, `gallery` menunjuk pada `.npz` dan `gallery_threshold` yang tersimpan
menimpa default `0.4`. Mengirimkan galeri ke model yang task-nya tidak
`embed` adalah sebuah kesalahan daripada operasi tanpa efek, dan sebuah file galeri yang hilang
menyarankan perintah `libreyolo enroll` yang akan membuatnya.

## Wajah

Pengenalan wajah adalah bentuk wilayah dari task ini, dan itu adalah satu-satunya yang dikirim
implementasi dari bentuk itu. Ini menambahkan tahap deteksi dan penyelarasan di depan
dari embedding head, ditambah dengan metode `verify()`, membawa kotak sendiri
argumen, angka akurasi yang dipublikasikan dan panduan kalibrasi untuk ambang batas.
Semua itu hidup di [pengenalan wajah](/docs/tasks/face-recognition), yang adalah
panduan langkah demi langkah yang harus diikuti ketika subjeknya adalah wajah. Segala sesuatu di halaman ini
berlaku untuk itu tanpa perubahan.

## Latih, validasi, dan ekspor

Tidak ada dalam task yang dilatih di dalam LibreYOLO. Wajah embedding head adalah
artefak ONNX yang `train()`, `val()`, dan `export()` semuanya menaikkan; latih head
hulu dan muat file melalui jalur. CLIP, SigLIP 2, dan DINOv2 melatih dan mengekspor
melalui tugas klasifikasi dan segmentasi mereka, bukan melalui `embed`.

Tidak ada validator pengambilan. Ukur akurasi verifikasi pada pasangan yang diberi label
dengan menyapu `threshold`, dan akurasi identifikasi dengan mendaftarkan galeri dan
membaca `identities.name` dan `identities.score` pada gambar yang disimpan, menghitung
nama `None` sebagai penolakan.

