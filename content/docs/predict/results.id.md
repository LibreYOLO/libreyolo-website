---
title: Bekerja dengan hasil
seo_title: Objek Results LibreYOLO
description: >-
  Satu objek Results per gambar, dengan satu slot per jenis payload: kotak,
  mask, keypoint, probabilitas, kedalaman, panoptik, OCR, dan lainnya. Plot,
  penyimpanan, dan JSON.
lead: >-
  Setiap prediksi mengembalikan satu objek Results per gambar. Objek ini
  memiliki satu slot bernama untuk setiap jenis payload, semuanya kosong kecuali
  yang dihasilkan model, dengan slot yang sama pada artefak hasil ekspor.
keywords:
  - yolo results object python
  - results.boxes xyxy
  - results ke json
  - simpan gambar anotasi
  - segmentation mask python
  - hasil keypoint
  - hasil depth map
  - results summary
  - onnx hasil sama
last_verified: 1.5.0
verification: >-
  Kelas payload, slot, semantik pemindahan, summary(), to_json(), plot(),
  save(), dan cutout() dibaca dari libreyolo/utils/results.py. Perilaku anotasi
  dan penulisan disk berasal dari InferenceRunner._save_annotated_image dalam
  libreyolo/models/base/inference.py serta resolve_save_path dalam
  libreyolo/utils/general.py. Dispatch sufiks berasal dari LibreYOLO() dalam
  libreyolo/models/__init__.py.
snippets:
  basic:
    - label: Kotak
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        print(result.orig_shape)   # (height, width) gambar sumber
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
        print(result.boxes.xyxyn[:1])   # kotak sama dibagi lebar dan tinggi
        print(result.boxes.xywhn[:1])
    - label: NumPy dan perangkat
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

        # Konten sama sebagai string, dengan argumen keyword yang sama.
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
    - label: Pasang komponen tambahan ekspor
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: Results yang sama dari artefak hasil ekspor
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        path = model.export(format="onnx")   # mengembalikan path yang ditulis

        # LibreYOLO() melakukan dispatch berdasarkan sufiks berkas.
        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)

        print(type(result).__name__, len(result.boxes))
source_hash: 548dbc9c7f5552ec
---

## Satu objek, satu slot per payload

Prediksi pada satu gambar mengembalikan satu `Results`. Objek ini memiliki delapan belas slot
payload, dan model hanya mengisi slot yang dihasilkan task-nya. Semua slot lain bernilai
`None`, sehingga membaca `result.masks` pada detektor menghasilkan `None`, bukan error.

| Slot | Kelas | Bentuk | Dihasilkan oleh |
|---|---|---|---|
| `boxes` | `Boxes` | `(N, 4)` ditambah skor dan kelas | Deteksi, dan task apa pun yang melakukan lokalisasi lebih dulu |
| `masks` | `Masks` | `(N, H, W)` | Segmentasi instance |
| `keypoints` | `Keypoints` | `(N, K, 2)` atau `(N, K, 3)` | Pose |
| `probs` | `Probs` | `(C,)` | Klasifikasi |
| `obb` | `OBB` | `(N, 7)` atau `(N, 8)` | Kotak berorientasi |
| `gaze` | `Gaze` | `(N, 2)` pitch dan yaw dalam radian | Estimasi arah pandang |
| `points` | `Points` | `(N, 4)` sebagai x, y, kelas, confidence | Lokalisasi titik |
| `semantic_mask` | `SemanticMask` | `(H, W)` id kelas | Segmentasi semantik |
| `panoptic` | `PanopticSegmentation` | `(H, W)` id segmen ditambah `segments_info` | Segmentasi panoptik |
| `depth_map` | `DepthMap` | `(H, W)` float | Estimasi kedalaman |
| `normal_map` | `NormalMap` | `(H, W, 3)` vektor satuan | Surface normal |
| `edges` | `EdgeMap` | `(H, W)` float dalam `[0, 1]` | Deteksi tepi |
| `restored` | `RestoredImage` | `(H, W, 3)` RGB uint8 | Restorasi dan super-resolution |
| `matte` | `Matte` | `(H, W)` float dalam `[0, 1]` | Alpha matting dan penghapusan latar belakang |
| `ocr` | `OCRRegions` | `(N, 4, 2)` poligon ditambah transkrip | Deteksi dan pengenalan teks |
| `embeddings` | `Embeddings` | `(N, D)` baris ternormalisasi L2 | Task `embed` |
| `identities` | `Identities` | N nama dan skor | Task `embed` dengan galeri |
| `meshes` | `Meshes` | Parameter tubuh dan vertices opsional | Pemulihan mesh tubuh |

Di sampingnya terdapat field yang dimiliki setiap hasil: `orig_shape` sebagai
`(height, width)`, `path` (path sumber atau `None` untuk input dalam memori), `names` yang
memetakan id kelas ke nama kelas, `frame_idx` untuk frame video dan live, `track_id` saat
pelacakan, serta `restore_scale`, yaitu faktor upscale bilangan bulat dari hasil restorasi.

`result.normals` adalah alias untuk `result.normal_map`.

`result.speed` ada pada setiap hasil, tetapi hanya diisi oleh
[ensemble](/docs/predict/ensembling), dengan key `member_0`, `member_1`, dan `fusion` dalam
milidetik. Untuk satu model, nilainya tetap berupa dict kosong.

## Kotak

<code-tabs name="basic" />

`Boxes` menyimpan koordinat dan skor sebagai array terpisah, bukan satu tensor terpaket.

| Atribut | Isi |
|---|---|
| `xyxy` | `(N, 4)` piksel absolut, x1 y1 x2 y2 |
| `xywh` | `(N, 4)` piksel absolut, pusat x, pusat y, lebar, tinggi |
| `xyxyn`, `xywhn` | Nilai yang sama dibagi lebar dan tinggi gambar |
| `conf` | `(N,)` confidence |
| `cls` | `(N,)` id kelas sebagai float |
| `id` | `(N,)` id track, atau `None` |
| `is_track` | Apakah `id` ditetapkan |
| `data` | Semuanya digabungkan: kotak, id opsional, conf, cls |

`cls` adalah array float, jadi gunakan sebagai `result.names[int(cls)]`.

`xyxyn` dan `xywhn` memerlukan `orig_shape`, yang diisi oleh `Results`.

## Payload padat

Payload yang mencakup seluruh gambar berperilaku berbeda dari payload per instance, dan hal
ini penting saat melakukan slicing.

`SemanticMask` menyimpan id kelas `(H, W)` pada canvas asli, dengan `255` sebagai nilai ignore
yang tidak pernah dihitung sebagai kelas. `classes` mencantumkan id yang ada dan mengecualikan
nilai tersebut; `class_mask(id)` mengembalikan boolean `(H, W)`.

`PanopticSegmentation` menyimpan id segmen `(H, W)`, dengan `0` sebagai id void, serta daftar
dict `segments_info` yang setidaknya memuat `id` dan `category_id`. `segment_ids`
mencantumkan id yang ada, sedangkan `segment_mask(id)` memilih satu segmen.

`DepthMap` menyimpan inverse depth relatif `(H, W)`: nilai lebih tinggi berarti lebih dekat,
dan nilainya bukan meter metrik. Objek ini mengekspos `min`, `max`, dan `mean` pada nilai
finite, sedangkan `normalized()` menskalakan ulang ke `[0, 1]`.

`NormalMap` menyimpan vektor satuan `(H, W, 3)` dalam frame kamera OpenCV, dengan `+x` ke
kanan, `+y` ke bawah, dan `+z` masuk ke adegan, sehingga permukaan yang menghadap kamera
adalah `(0, 0, -1)`. `assert_normalized()` memeriksa bahwa setiap piksel finite dan panjangnya satu.

`EdgeMap` menyimpan float32 `(H, W)` dalam `[0, 1]`. Peta kontinu dipertahankan, bukan diberi
ambang, sehingga `binary(threshold=0.5)` adalah tempat memilih batas.

`Matte` menyimpan float32 `(H, W)` dalam `[0, 1]`, dengan `1` berarti foreground penuh.
`array` mengembalikan nilai yang sudah dipotong sebagai float32.

`RestoredImage` menyimpan RGB uint8 `(H, W, 3)`, dengan `array` untuk ndarray mentah dan
`save(path)` untuk menulisnya.

`Probs` menyimpan satu vektor probabilitas untuk gambar. `top1` dan `top5` adalah indeks
kelas, sedangkan `top1conf` dan `top5conf` adalah skor yang sesuai.

`Embeddings` menyimpan baris `(N, D)` yang sudah dinormalisasi L2, sehingga cosine similarity
adalah dot product. `similarity(other)` mengembalikan `(N, M)` terhadap galeri atau `(N,)`
terhadap satu vektor, dan `verify(i, j, threshold=0.4)` membandingkan dua baris.

`OCRRegions` menyimpan poligon `(N, 4, 2)` dalam urutan baca, dengan sudut kiri atas, kanan
atas, kanan bawah, kiri bawah. Transkrip ada di `texts`, skor pengenalan di `conf`, dan skor
deteksi di `det_conf`. Karena ini poligon berotasi asli, payload tersebut tidak mengisi
`boxes`; `ocr.xyxy` memberikan hull axis-aligned jika memerlukan persegi panjang.

## Slicing dan pemindahan

`result[i]` mengembalikan `Results` baru yang memuat satu instance. Payload per instance
di-slice; payload seluruh gambar diteruskan tanpa perubahan, sehingga slicing hasil
klasifikasi tidak dapat memotong vektor probabilitasnya menjadi satu kelas dan slicing hasil
kedalaman tidak dapat merusak tata letak `(H, W)`.

`len(result)` menghitung instance: kotak, titik, embedding, wilayah OCR, atau mesh. Payload
padat seluruh gambar dihitung sebagai `1`. Hasil tanpa isi bernilai `0`.

`to()`, `cpu()`, `cuda()`, dan `numpy()` masing-masing mengembalikan `Results` baru dengan
setiap slot terisi dikonversi. Semuanya tidak mengubah objek asli.

`update()` adalah satu-satunya metode yang melakukan mutasi in-place, mengganti slot bernama,
dan mengembalikan objek yang sama.

## JSON

<code-tabs name="json" />

`summary()` mengembalikan daftar dict biasa, sedangkan `to_json()` meneruskan daftar tersebut
melalui `json.dumps`. Keduanya menerima tiga argumen yang sama: `normalize=False` mengubah
koordinat ke `[0, 1]`, `decimals=5` menetapkan pembulatan, dan `embeddings=False` mengontrol
apakah vektor embedding disertakan.

Bentuk baris mengikuti payload. Baris deteksi memuat `name`, `class`, `confidence`, dan dict
`box`, lalu mendapatkan `segments` saat mask ada, `obb` dan `corners` untuk kotak berorientasi,
sudut `gaze` dalam radian dan derajat, `track_id` saat pelacakan, serta parameter `mesh` saat
mesh ada.

Jika tidak ada kotak, satu payload menentukan baris: OCR mengeluarkan satu baris per wilayah
dengan `text`, titik satu baris per titik, panoptik satu baris per segmen dengan `pixel_count`
dan `pixel_fraction`, semantik satu baris per kelas yang ada, serta klasifikasi lima kelas
teratas. Kedalaman, normal, tepi, restorasi, dan matting masing-masing mengeluarkan satu baris
ringkasan yang menggambarkan peta, bukan pikselnya.

Dua payload sengaja disingkat. Vektor embedding hanya dilaporkan sebagai `embedding_dim`
karena baris 512 float berukuran sekitar 2 KB per wajah; berikan `embeddings=True` untuk
menyertakan nilainya. Vertices mesh tidak pernah disertakan karena jumlahnya puluhan ribu
koordinat per orang. Baca `result.meshes.vertices` atau panggil
`result.meshes.save_obj(path)` untuk geometri.

## Menggambar dan menyimpan

<code-tabs name="saving" />

`predict(save=True)` adalah jalur untuk memberi anotasi dan menulis. Metode ini memilih
rutinitas gambar berdasarkan slot yang terisi, sehingga hasil semantik ditulis sebagai mask
berwarna, hasil kedalaman sebagai visualisasi kedalaman, hasil panoptik dengan segmennya,
matte sebagai PNG RGBA berlatar transparan, dan detektor sebagai kotak dengan mask di bawahnya.
Path yang ditulis dilampirkan ke hasil sebagai `result.saved_path`.

`Results.plot()` lebih sempit daripada namanya. Metode ini hanya didefinisikan untuk peta
normal dan peta tepi, serta memunculkan `NotImplementedError` untuk lainnya. Gunakan
`save=True` untuk task lain.

`Results.save(path)` juga sempit: metode ini menulis hasil matte sebagai cutout PNG RGBA
berlatar transparan dan memunculkan `NotImplementedError` untuk lainnya. `Results.cutout()`
mengembalikan array RGBA yang sama tanpa menulis. Keduanya memerlukan gambar sumber, yang
diambil dari `result.path` atau diberikan sebagai `image=`.

Dua payload memiliki writer sendiri: `result.restored.save(path)` untuk gambar hasil restorasi,
dan `result.meshes.save_obj(path, index=0)` untuk mesh.

Untuk lokasi berkas serta perilaku `output_path` dan `output_file_format`, lihat
[Sumber prediksi](/docs/predict/sources).

## Artefak hasil ekspor mengembalikan objek yang sama

<code-tabs name="exported" />

`LibreYOLO()` melakukan dispatch berdasarkan sufiks berkas, sehingga artefak hasil ekspor
dimuat melalui pemanggilan yang sama seperti checkpoint `.pt` dan mengembalikan `Results`
yang sama. Berkas `.onnx`, `.engine`, `.pte`, dan `.mnn` dikenali berdasarkan sufiks,
demikian juga direktori OpenVINO, Paddle, dan ncnn serta URL model Triton. Kode yang membaca
`result.boxes.xyxy` tidak berubah saat model diganti dengan build hasil ekspor. Lihat
[Ekspor](/docs/export) untuk seluruh format.

Menggunakan API runtime sendiri berarti harus menangani prapemrosesan, pascapemrosesan, dan
nama kelas sendiri.
