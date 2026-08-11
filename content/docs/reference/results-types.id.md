---
title: Jenis Results
seo_title: Referensi objek Results LibreYOLO
description: >-
  Setiap payload yang dapat dibawa objek Results LibreYOLO, satu slot per bentuk
  task: bounding box, mask, keypoint, probs, obb, depth, ocr, embedding, dan sepuluh
  lainnya.
lead: >-
  Results adalah satu-satunya jenis nilai kembalian per gambar dari setiap model
  LibreYOLO. Objek ini memiliki delapan belas slot payload opsional, satu per
  bentuk task, dan hanya mengisi slot yang dihasilkan model.
keywords:
  - objek results libreyolo
  - Results.boxes
  - Results.masks
  - Results.probs
  - Results.depth_map
  - Results.summary
  - results libreyolo ke json
last_verified: 1.5.0
verification: >-
  Nama slot, bentuk, properti, dan default dibaca dari
  libreyolo/utils/results.py pada v1.5.0. Semantik dikutip dari docstring kelas
  payload.
snippets:
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE)

        print(result.orig_shape, result.path)
        print(result.boxes.xyxy)
        print(result.boxes.conf)
        print(result.names[int(result.boxes.cls[0])])
  convert:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE)

        # Setiap payload berpindah bersama-sama.
        result = result.cpu().numpy()

        # Baris sebagai dict biasa, lalu sebagai JSON.
        print(result.summary()[:1])
        print(result.to_json())
source_hash: 16f654364ae6448a
---

## Objek Results

Satu `Results` menjelaskan satu gambar. Sumber satu gambar mengembalikan satu
objek, sumber list atau direktori mengembalikan list, dan `stream=True`
mengembalikan generator yang menghasilkan objek tersebut.

| Atribut | Jenis | Arti |
|---|---|---|
| `orig_shape` | `(int, int)` | Tinggi dan lebar gambar asli |
| `path` | `str` | Path sumber ketika input berasal dari disk |
| `names` | `dict[int, str]` | Indeks kelas ke nama kelas |
| `speed` | `dict[str, float]` | Milidetik per tahap |
| `track_id` | tensor | Id track ketika hasil berasal dari `track()` |
| `frame_idx` | `int` | Indeks frame untuk sumber video dan stream |
| `restore_scale` | `int` | Faktor upscale output-ke-input dari hasil restore; `1` di tempat lain |

<code-tabs name="usage" />

## Slot payload

Setiap slot bernilai `None` kecuali model menghasilkannya. Slot yang diisi family
ditentukan oleh task-nya.

| Slot | Kelas | Task |
|---|---|---|
| `boxes` | `Boxes` | detect |
| `masks` | `Masks` | segment |
| `keypoints` | `Keypoints` | pose |
| `probs` | `Probs` | classify |
| `obb` | `OBB` | obb |
| `gaze` | `Gaze` | gaze |
| `points` | `Points` | point |
| `semantic_mask` | `SemanticMask` | semantic |
| `panoptic` | `PanopticSegmentation` | panoptic |
| `depth_map` | `DepthMap` | depth |
| `normal_map` | `NormalMap` | normal |
| `edges` | `EdgeMap` | edge |
| `restored` | `RestoredImage` | restore |
| `matte` | `Matte` | matte |
| `ocr` | `OCRRegions` | ocr |
| `embeddings` | `Embeddings` | embed |
| `identities` | `Identities` | embed, dengan gallery |
| `meshes` | `Meshes` | mesh |

`result.normals` adalah alias baca-tulis untuk `result.normal_map`.

Lebih dari satu slot dapat ditetapkan sekaligus. Model segmentation mengisi
`boxes` dan `masks`; model gaze mengisi `boxes` dengan bounding box wajah dan `gaze`
dengan sudut; model mesh mengisi `boxes` dengan bounding box orang dan `meshes` yang
barisnya diselaraskan dengannya.

## Bounding box

Bounding box deteksi untuk satu gambar.

| Anggota | Mengembalikan |
|---|---|
| `xyxy` | Koordinat sudut dalam piksel gambar asli |
| `xywh` | Pusat dan ukuran dalam piksel |
| `xyxyn` | Sudut yang dinormalisasi ke `[0, 1]` |
| `xywhn` | Pusat dan ukuran yang dinormalisasi ke `[0, 1]` |
| `conf` | Confidence per bounding box |
| `cls` | Indeks kelas per bounding box |
| `id` | Id track per bounding box, atau `None` |
| `is_track` | `True` ketika terdapat id track |
| `data` | Tensor terkemas |

`with_id(id)` dan `with_orig_shape(orig_shape)` mengembalikan `Boxes` baru
dengan kolom tersebut diganti.

## Masks

Instance mask untuk satu gambar. `data` adalah tensor mask; `xy` mengembalikan
kontur per instance dalam piksel dan `xyn` mengembalikannya dalam bentuk
ternormalisasi.

## Keypoints

Keypoint pose yang barisnya diselaraskan dengan `boxes`. `xy` adalah pasangan
koordinat per keypoint dan `xyn` pasangan ternormalisasi. `conf` adalah channel
ketiga ketika data memilikinya, atau `None`. `has_visible` adalah array boolean,
true ketika `conf > 0`, dan seluruhnya true jika tidak ada channel confidence.

## Points

Lokalisasi titik untuk satu gambar. `data` berbentuk `(N, 4)` dengan baris
`x, y, class, confidence`. Koordinat berupa piksel absolut; `xy`, `cls`, dan
`conf` memisahkan kolom, sedangkan `xyn` menormalisasi koordinat.

## Probs

Skor classification. `top1` adalah indeks pemenang, `top5` adalah lima indeks
terbaik, dan `top1conf` serta `top5conf` adalah skor-nya.

## OBB

Oriented bounding box. `data` menyimpan 7 atau 8 nilai per baris: `xywhr`, id track
opsional, lalu confidence dan kelas.

| Anggota | Mengembalikan |
|---|---|
| `xywhr` | Pusat, ukuran, dan rotasi dalam radian |
| `xyxyxyxy` | Empat sudut dalam piksel |
| `xyxyxyxyn` | Empat sudut ternormalisasi |
| `xyxy` | Hull yang sejajar sumbu dalam piksel |
| `conf`, `cls`, `id`, `is_track` | Seperti pada `Boxes` |

## Gaze

Sudut arah pandang per wajah dalam radian, berbentuk `(N, 2)`, dengan baris
yang diselaraskan terhadap bounding box wajah dalam `boxes`. Kolom 0 adalah pitch dan
kolom 1 adalah yaw menurut konvensi L2CS: yaw positif memutar pandangan ke arah
kiri subjek dan pitch positif memutarnya ke bawah. `pitch_deg` dan `yaw_deg`
mengonversi ke derajat, sedangkan `direction_3d` mengembalikan vektor arah
satuan.

## SemanticMask

Map semantic padat, berbentuk `(H, W)` yang berisi id kelas bilangan bulat pada
canvas gambar asli. `255` adalah nilai ignore dan tidak pernah dihitung sebagai
kelas (`SemanticMask.IGNORE_INDEX`). `classes` mencantumkan id kelas yang ada,
dan `class_mask(class_id)` mengembalikan mask boolean untuk satu kelas.

## PanopticSegmentation

Setiap piksel mendapatkan tepat satu segmen yang tidak tumpang tindih, dengan
region stuff dan instance thing disatukan. `data` adalah map id segmen bilangan
bulat `(H, W)`; id segmen `0` tidak berlabel
(`PanopticSegmentation.IGNORE_INDEX`). `segments_info` adalah list dict, satu
per segmen, masing-masing setidaknya memuat `{"id": int, "category_id": int}`,
dengan `id` cocok dengan nilai dalam map dan `category_id` mengindeks `names`.
`segment_ids` mencantumkan id yang ada dan `segment_mask(segment_id)`
mengembalikan mask boolean satu segmen.

Thing-versus-stuff merupakan properti kategori, bukan segmen. Payload dapat
melakukan denormalization properti tersebut ke setiap segmen sebagai
`"isthing": bool`, dan jika dilakukan, nilainya harus sesuai dengan map level
kategori.

## DepthMap

Map inverse-depth relatif padat, berbentuk `(H, W)` float pada canvas gambar
asli. Nilai lebih tinggi berarti lebih dekat ke kamera. Nilainya relatif, bukan
meter metrik. `min`, `max`, dan `mean` dihitung pada nilai terbatas, sedangkan
`normalized()` melakukan rescale map ke `[0, 1]`.

## NormalMap

Kolom surface-normal padat, float32 `(H, W, 3)` pada canvas gambar asli, dalam
frame kamera OpenCV: `+x` ke kanan, `+y` ke bawah, dan `+z` masuk ke scene.
Normal menghadap kamera, sehingga permukaan fronto-parallel adalah
`(0, 0, -1)`. Setiap piksel berupa vektor satuan.
`assert_normalized(atol=1e-4)` memeriksa invariant tersebut.

## EdgeMap

Map probabilitas edge padat, float32 `(H, W)` pada canvas gambar asli, dengan
`0` berarti bukan edge dan `1` berarti edge. Map kontinu dipertahankan agar
ambang batas tetap menjadi pilihan pemanggil: `binary(threshold=0.5)`
menerapkannya, dan `array` mengembalikan tampilan numpy.

## RestoredImage

Gambar RGB hasil restorasi, uint8 `(H, W, 3)`. Untuk super-resolution, canvas
berukuran `Results.restore_scale` kali input. `array` mengembalikan tampilan
numpy dan `save(path)` menulis gambar.

## Matte

Matte opasitas lunak, float32 `(H, W)` dalam `[0, 1]` pada canvas gambar asli.
`1` berarti foreground penuh dan `0` berarti background penuh. Matte lunak
mencakup mask penghapusan latar belakang keras yang menggunakan ambang batas
0.5, sekaligus mempertahankan edge anti-alias yang dibuang mask biner. `array`
mengembalikan tampilan numpy.

Pada hasil matte, `Results.cutout(image=None)` mengembalikan array uint8 RGBA
`(H, W, 4)` yang channel keempatnya adalah matte, dan
`Results.save(path, image=None)` menulis cutout sebagai PNG berlatar belakang
transparan. Keduanya mengambil RGB dari `image` jika diberikan, atau memuat
ulang dari `Results.path`.

## OCRRegions

Teks terlokalisasi beserta transkrip. `data` adalah poligon float `(N, 4, 2)`
dalam piksel gambar asli, berurutan kiri atas, kanan atas, kanan bawah, kiri
bawah, dan region berada dalam urutan baca, atas ke bawah lalu kiri ke kanan.
`texts` adalah list N transkrip. `conf` adalah skor pengenalan per region dan
`det_conf` adalah skor deteksi, keduanya berbentuk `(N,)`.

Quad deteksi merupakan poligon sebenarnya, sehingga tidak mengisi
`Results.boxes`. `xyxy` memberikan hull yang sejajar sumbu.

## Embedding

Vektor ternormalisasi L2 dari task `embed`, selalu berbentuk `(N, D)`. Hasil
seluruh gambar memuat satu baris tanpa bounding box; embedding region memiliki baris yang
diselaraskan dengan `boxes`. Karena setiap baris dinormalisasi, cosine similarity
merupakan dot product.

| Anggota | Mengembalikan |
|---|---|
| `dim` | `D` |
| `normalized` | Baris yang dinormalisasi ulang |
| `similarity(other)` | Cosine similarity pairwise terhadap `Embeddings` atau tensor lain |
| `verify(i, j, threshold=0.4)` | `True` ketika baris `i` dan `j` cocok |

## Identities

Kecocokan gallery bernama, dengan baris yang diselaraskan terhadap `embeddings`.
Dihasilkan ketika `Gallery` diberikan kepada prediksi `embed`. `name` adalah
list dengan entri bernilai `None` di bawah ambang batas kecocokan, dan nama
terdekat yang berada di bawah ambang batas tidak pernah ditebak. `score` adalah
array skor kecocokan dan `data` memasangkan keduanya.

## Meshes

Body mesh manusia parametrik, dengan baris yang diselaraskan terhadap bounding box orang
dalam `boxes`. Semuanya berada dalam frame kamera gambar asli. `transl` bersifat
metrik dalam meter dengan `+z` menjauhi kamera; `vertices` dan `joints3d`
bersifat metrik dan sudah menyertakan `transl`; `joints2d` berada dalam piksel
pada canvas gambar asli, bukan crop yang dilihat jaringan. Tidak ada kolom yang
memuat frame dunia atau gravitasi.

Tata letak parameter berbeda antar body model, sehingga tidak ada bentuk yang
ditetapkan langsung. `body_model` menyebut parameterization dan jumlahnya dibaca dari
tensor: `num_vertices`, `num_joints`, `num_betas`, dan `has_vertices`. `params`
mengembalikan dict parameter, sedangkan `save_obj(path, index=0)` menulis satu
mesh. Kolom-nya adalah `global_orient`, `body_pose`, `betas`, `transl`,
`vertices`, `faces`, `joints3d`, `joints2d`, `conf`, `focal_length`, dan
`extras`.

Untuk `body_model="mhr"`, rotasi berupa sudut Euler dalam radian, bukan
axis-angle, `body_pose` merupakan vektor parameter per joint yang flat, bukan
satu triplet per joint, dan `betas` adalah koefisien identity blendshape. Skala
skeleton, pose tangan, dan ekspresi wajah berada dalam `extras`.

## Konversi dan pemilihan

Setiap payload memiliki `to(*args, **kwargs)`, `cpu()`, `cuda()`, dan `numpy()`,
dan memanggil salah satunya pada `Results` menerapkannya ke setiap slot yang
terisi sekaligus.

<code-tabs name="convert" />

`result[idx]` memilih baris lintas payload yang selaras baris. `len(result)`
adalah jumlah deteksi, atau jumlah titik jika tidak ada bounding box. `result.update(...)`
mengembalikan salinan dengan slot bernama diganti; metode ini menerima setiap
slot ditambah `track_id` dan `restore_scale`.

## summary dan to_json

`summary(normalize=False, decimals=5, embeddings=False)` mengembalikan list dict
biasa, satu baris per deteksi, segmen, titik, atau region bergantung pada slot
yang ditetapkan. `to_json(**kwargs)` meneruskan argumennya ke `summary` dan
mengembalikan string JSON.

`plot()` merender hasil normal atau edge padat dalam visualisasi kanonis;
metode ini memunculkan error untuk jenis hasil lain. Gambar beranotasi untuk
task lain berasal dari `predict(save=True)`.



