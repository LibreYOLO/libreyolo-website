---
title: Bekerja dengan hasil
seo_title: Objek Results LibreYOLO
description: >-
  Satu objek Results per gambar, dengan satu slot untuk setiap jenis payload:
  box, mask, keypoint, probabilitas, depth, panoptic, OCR, dan lainnya.
  Plotting, penyimpanan, dan JSON.
lead: >-
  Setiap prediksi mengembalikan satu objek Results per gambar. Objek ini
  memiliki satu slot bernama untuk setiap jenis payload, semuanya kosong kecuali
  yang dihasilkan model, dengan slot yang sama pada artefak hasil ekspor.
keywords:
  - objek results yolo python
  - results.boxes xyxy
  - hasil prediksi ke json
  - simpan gambar beranotasi
  - segmentation mask python
  - hasil keypoint
  - hasil depth map
  - ringkasan results
  - hasil onnx sama
last_verified: 1.5.0
verification: >-
  Kelas payload, slot, semantik pemindahan, summary(), to_json(), plot(),
  save(), dan cutout() dibaca dari libreyolo/utils/results.py. Perilaku anotasi
  dan penulisan ke disk dari InferenceRunner._save_annotated_image di
  libreyolo/models/base/inference.py dan resolve_save_path di
  libreyolo/utils/general.py. Dispatch suffix dari LibreYOLO() di
  libreyolo/models/__init__.py.
snippets:
  basic:
    - label: Box
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        print(result.orig_shape)   # (tinggi, lebar) gambar sumber
        print(result.path)         # path sumber, None untuk input dalam memori

        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            print(result.names[int(cls)], round(float(conf), 3), xyxy)
    - label: Koordinat ternormalisasi
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy[:1])    # piksel, x1 y1 x2 y2
        print(result.boxes.xywh[:1])    # piksel, pusat x, pusat y, w, h
        print(result.boxes.xyxyn[:1])   # box yang sama dibagi lebar dan tinggi
        print(result.boxes.xywhn[:1])
    - label: NumPy dan device
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        # Masing-masing mengembalikan Results baru; objek asli tidak berubah.
        as_numpy = result.numpy()
        on_cpu = result.cpu()

        print(type(as_numpy.boxes.xyxy).__name__)
        print(type(on_cpu.boxes.xyxy).__name__)
  json:
    - label: summary dan to_json
      language: python
      code: |
        import json

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        rows = result.summary()
        print(json.dumps(rows[:2], indent=2))

        # Konten yang sama sebagai string, dengan keyword argument yang sama.
        print(result.to_json(normalize=True, decimals=3)[:200])
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt --json \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  saving:
    - label: Gambar beranotasi
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt")


        # save=True menggambar payload dan menulisnya di bawah
        runs/detect/predict*.

        result = model(SAMPLE_IMAGE, save=True)

        print(result.saved_path)
  exported:
    - label: Instal ekstra ekspor
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: Results yang sama dari artefak hasil ekspor
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        path = model.export(format="onnx")   # mengembalikan path yang ditulis

        # LibreYOLO() melakukan dispatch berdasarkan suffix file.
        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)

        print(type(result).__name__, len(result.boxes))
source_hash: 548dbc9c7f5552ec
---

## Satu objek, satu slot per payload

Prediksi pada satu gambar mengembalikan satu `Results`. Objek ini memiliki
delapan belas slot payload, dan model hanya mengisi slot yang dihasilkan oleh
task-nya. Semua slot lain bernilai `None`, jadi membaca `result.masks` pada
detektor menghasilkan `None`, bukan error.

| Slot | Kelas | Bentuk | Dihasilkan oleh |
|---|---|---|---|
| `boxes` | `Boxes` | `(N, 4)` ditambah score dan kelas | Deteksi, dan semua task yang melakukan lokalisasi terlebih dahulu |
| `masks` | `Masks` | `(N, H, W)` | Instance segmentation |
| `keypoints` | `Keypoints` | `(N, K, 2)` atau `(N, K, 3)` | Pose |
| `probs` | `Probs` | `(C,)` | Classification |
| `obb` | `OBB` | `(N, 7)` atau `(N, 8)` | Oriented box |
| `gaze` | `Gaze` | `(N, 2)` pitch dan yaw dalam radian | Estimasi arah pandang |
| `points` | `Points` | `(N, 4)` sebagai x, y, kelas, confidence | Lokalisasi titik |
| `semantic_mask` | `SemanticMask` | `(H, W)` id kelas | Semantic segmentation |
| `panoptic` | `PanopticSegmentation` | `(H, W)` id segmen ditambah `segments_info` | Panoptic segmentation |
| `depth_map` | `DepthMap` | `(H, W)` float | Estimasi kedalaman |
| `normal_map` | `NormalMap` | `(H, W, 3)` vektor satuan | Surface normal |
| `edges` | `EdgeMap` | `(H, W)` float dalam `[0, 1]` | Deteksi tepi |
| `restored` | `RestoredImage` | `(H, W, 3)` uint8 RGB | Restorasi dan super-resolution |
| `matte` | `Matte` | `(H, W)` float dalam `[0, 1]` | Alpha matting dan penghapusan latar belakang |
| `ocr` | `OCRRegions` | `(N, 4, 2)` poligon ditambah transkrip | Deteksi dan pengenalan teks |
| `embeddings` | `Embeddings` | `(N, D)` baris ternormalisasi L2 | Task `embed` |
| `identities` | `Identities` | N nama dan score | Task `embed` dengan gallery |
| `meshes` | `Meshes` | Parameter tubuh dan vertex opsional | Rekonstruksi body mesh |

Di sampingnya terdapat field yang dimiliki setiap hasil: `orig_shape` sebagai
`(height, width)`, `path` (path sumber, atau `None` untuk input dalam memori),
`names` yang memetakan id kelas ke nama kelas, `frame_idx` untuk video dan frame
live, `track_id` saat tracking, serta `restore_scale`, yaitu faktor upscale
bilangan bulat dari hasil restorasi.

`result.normals` adalah alias untuk `result.normal_map`.

`result.speed` ada pada setiap hasil, tetapi hanya diisi oleh
[ensemble](/docs/predict/ensembling), dengan key `member_0`, `member_1`, dan
`fusion` dalam milidetik. Untuk satu model, field ini tetap berupa dict kosong.

## Box

<code-tabs name="basic" />

`Boxes` menyimpan koordinat dan score sebagai array terpisah, bukan sebagai satu
tensor yang dikemas.

| Atribut | Isi |
|---|---|
| `xyxy` | `(N, 4)` piksel absolut, x1 y1 x2 y2 |
| `xywh` | `(N, 4)` piksel absolut, pusat x, pusat y, lebar, tinggi |
| `xyxyn`, `xywhn` | Nilai yang sama dibagi lebar dan tinggi gambar |
| `conf` | `(N,)` confidence |
| `cls` | `(N,)` id kelas, sebagai float |
| `id` | `(N,)` id track, atau `None` |
| `is_track` | Apakah `id` telah ditetapkan |
| `data` | Semua yang digabungkan: box, id opsional, conf, cls |

`cls` adalah array float, jadi gunakan sebagai `result.names[int(cls)]`.

`xyxyn` dan `xywhn` memerlukan `orig_shape`, yang diisi oleh `Results`.

## Payload padat

Payload yang mencakup seluruh gambar berperilaku berbeda dari payload per
instance, dan perbedaan ini penting saat melakukan slicing.

`SemanticMask` menyimpan id kelas `(H, W)` pada canvas asli, dengan `255`
dicadangkan sebagai nilai ignore yang tidak pernah dihitung sebagai kelas.
`classes` mencantumkan id yang ada dan tidak menyertakannya;
`class_mask(id)` mengembalikan boolean `(H, W)`.

`PanopticSegmentation` menyimpan id segmen `(H, W)`, dengan `0` sebagai id void,
serta list dict `segments_info` yang setidaknya memuat `id` dan `category_id`.
`segment_ids` mencantumkan id yang ada, sedangkan `segment_mask(id)` memilih
salah satunya.

`DepthMap` menyimpan inverse depth relatif `(H, W)`: nilai lebih tinggi berarti
lebih dekat, dan nilainya bukan meter metrik. Objek ini menyediakan `min`, `max`,
dan `mean` untuk nilai terbatas, serta `normalized()` yang melakukan rescaling
ke `[0, 1]`.

`NormalMap` menyimpan vektor satuan `(H, W, 3)` dalam frame kamera OpenCV, dengan
`+x` ke kanan, `+y` ke bawah, dan `+z` masuk ke scene. Karena itu, permukaan yang
menghadap kamera adalah `(0, 0, -1)`. `assert_normalized()` memeriksa bahwa
setiap piksel memiliki nilai terbatas dan panjang satuan.

`EdgeMap` menyimpan float32 `(H, W)` dalam `[0, 1]`. Map kontinu dipertahankan
tanpa diterapkan ambang batas, sehingga `binary(threshold=0.5)` menjadi tempat
untuk memilih cutoff.

`Matte` menyimpan float32 `(H, W)` dalam `[0, 1]`, dengan `1` berarti foreground
sepenuhnya. `array` mengembalikannya sebagai float32 yang dibatasi.

`RestoredImage` menyimpan RGB uint8 `(H, W, 3)`, dengan `array` untuk ndarray
mentah dan `save(path)` untuk menulisnya.

`Probs` menyimpan satu vektor probabilitas untuk gambar. `top1` dan `top5` adalah
indeks kelas, sedangkan `top1conf` dan `top5conf` adalah score yang sesuai.

`Embeddings` menyimpan baris `(N, D)` yang sudah dinormalisasi L2, sehingga
cosine similarity merupakan dot product. `similarity(other)` mengembalikan
`(N, M)` terhadap gallery atau `(N,)` terhadap satu vektor, dan
`verify(i, j, threshold=0.4)` membandingkan dua baris.

`OCRRegions` menyimpan poligon `(N, 4, 2)` dalam urutan baca, dengan sudut
berurutan kiri atas, kanan atas, kanan bawah, kiri bawah. Transkrip berada di
`texts`, score pengenalan di `conf`, dan score deteksi di `det_conf`. Karena
poligon ini benar-benar berotasi, field `boxes` tidak diisi; `ocr.xyxy`
memberikan hull yang sejajar sumbu ketika diperlukan rectangle.

## Slicing dan pemindahan

`result[i]` mengembalikan `Results` baru yang menyimpan satu instance. Payload
per instance di-slice; payload seluruh gambar diteruskan tanpa perubahan,
sehingga slicing hasil classification tidak dapat memangkas vektor
probabilitasnya menjadi satu kelas, dan slicing hasil depth tidak dapat merusak
layout `(H, W)`.

`len(result)` menghitung instance: box, point, embedding, region OCR, atau mesh.
Setiap payload padat seluruh gambar dihitung sebagai `1`. Hasil tanpa isi
bernilai `0`.

`to()`, `cpu()`, `cuda()`, dan `numpy()` masing-masing mengembalikan `Results`
baru dengan setiap slot yang terisi telah dikonversi. Metode tersebut tidak
mengubah objek asli.

`update()` adalah satu-satunya metode yang melakukan mutasi di tempat, mengganti
slot bernama, lalu mengembalikan objek yang sama.

## JSON

<code-tabs name="json" />

`summary()` mengembalikan list dict biasa, dan `to_json()` meneruskan list
tersebut melalui `json.dumps`. Keduanya menerima tiga argumen yang sama:
`normalize=False` mengubah koordinat menjadi `[0, 1]`, `decimals=5` menetapkan
pembulatan, dan `embeddings=False` mengatur apakah vektor embedding disertakan.

Bentuk baris mengikuti payload. Baris deteksi memuat `name`, `class`,
`confidence`, dan dict `box`, serta mendapatkan `segments` ketika mask tersedia,
`obb` dan `corners` untuk oriented box, sudut `gaze` dalam radian dan derajat,
`track_id` saat tracking, dan parameter `mesh` ketika mesh tersedia.

Jika tidak ada box, satu payload menentukan barisnya: OCR menghasilkan satu
baris per region beserta `text`-nya, point menghasilkan satu baris per point,
panoptic satu baris per segmen dengan `pixel_count` dan `pixel_fraction`,
semantic satu baris per kelas yang ada, dan classification lima kelas teratas.
Depth, normal, edge, restoration, dan matting masing-masing menghasilkan satu
baris ringkasan yang menjelaskan map, bukan pikselnya.

Dua payload sengaja disingkat. Vektor embedding hanya dilaporkan sebagai
`embedding_dim` karena satu baris 512-float berukuran sekitar 2 KB per wajah;
berikan `embeddings=True` untuk menyertakan nilainya. Vertex mesh tidak pernah
disertakan karena jumlahnya mencapai puluhan ribu koordinat per orang. Baca
`result.meshes.vertices` atau panggil `result.meshes.save_obj(path)` untuk
geometri.

## Menggambar dan menyimpan

<code-tabs name="saving" />

`predict(save=True)` adalah jalur yang memberi anotasi dan menulis hasil. Metode
ini memilih rutin menggambar berdasarkan slot yang terisi, sehingga hasil
semantic ditulis sebagai mask berwarna, hasil depth sebagai visualisasi depth,
hasil panoptic beserta segmennya, matte sebagai PNG RGBA dengan latar belakang
transparan, dan hasil detektor sebagai box dengan mask di bawahnya. Path yang
ditulis dilampirkan ke hasil sebagai `result.saved_path`.

`Results.plot()` lebih sempit daripada yang ditunjukkan namanya. Metode ini
hanya didefinisikan untuk normal map dan edge map, serta memunculkan
`NotImplementedError` untuk yang lain. Gunakan `save=True` untuk task lain.

`Results.save(path)` juga memiliki cakupan sempit: metode ini menulis hasil matte
sebagai cutout PNG RGBA dengan latar belakang transparan dan memunculkan
`NotImplementedError` untuk yang lain. `Results.cutout()` mengembalikan array
RGBA yang sama tanpa menulisnya. Keduanya memerlukan gambar sumber, yang diambil
dari `result.path` atau diberikan sebagai `image=`.

Dua payload memiliki writer sendiri: `result.restored.save(path)` untuk gambar
yang direstorasi, dan `result.meshes.save_obj(path, index=0)` untuk mesh.

Untuk lokasi file serta perilaku `output_path` dan `output_file_format`, lihat
[Sumber prediksi](/docs/predict/sources).

## Artefak hasil ekspor mengembalikan objek yang sama

<code-tabs name="exported" />

`LibreYOLO()` melakukan dispatch berdasarkan suffix file, sehingga artefak hasil
ekspor dimuat melalui pemanggilan yang sama seperti checkpoint `.pt` dan
mengembalikan `Results` yang sama. File `.onnx`, `.engine`, `.pte`, dan `.mnn`
dikenali berdasarkan suffix, demikian pula direktori OpenVINO, Paddle, dan ncnn
serta URL model Triton. Kode yang membaca `result.boxes.xyxy` tidak berubah saat
model ditukar dengan build hasil ekspornya. Lihat [Ekspor](/docs/export) untuk
daftar lengkap format.

Menggunakan API milik runtime secara langsung berarti preprocessing,
postprocessing, dan nama kelas harus ditangani sendiri.
