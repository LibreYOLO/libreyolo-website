---
title: Estimasi pose
seo_title: Estimasi pose di LibreYOLO
description: >-
  Prediksi keypoint per instance di LibreYOLO: family yang melayani task, format
  label, serta pemanggilan prediksi, pelatihan, validasi, dan ekspor.
lead: >-
  Estimasi pose melokalisasi setiap instance dan mengembalikan kumpulan keypoint
  bernama dalam urutan tetap, sehingga output memuat struktur internal objek,
  bukan hanya luas cakupannya. Kunci task-nya adalah pose.
keywords:
  - estimasi pose Python
  - deteksi keypoint
  - model human pose
  - COCO keypoints
  - OKS mAP
  - melatih model pose
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Akhiran -pose pada nama berkas memilih keypoint head, sehingga argumen
        # task tidak diperlukan.
        model = LibreYOLO("LibreECs-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy.shape)   # (N, K, 2) koordinat piksel
        print(result.boxes.xyxy.shape)     # (N, 4), N instance yang sama
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreECs-pose.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Hanya keypoint terlihat
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreECs-pose.pt")(SAMPLE_IMAGE)
        kpts = result.keypoints

        # .has_visible diturunkan dari kolom keypoint ketiga, dan seluruhnya
        # true jika checkpoint hanya memprediksi (x, y).
        for person, visible in zip(kpts.xy, kpts.has_visible):
            print(person[visible])
    - label: Gunakan top-down
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # HRNet bersifat top-down: model memotong setiap orang lebih dahulu. Tanpa
        # sumber orang, model memasangkan dirinya dengan detector LibreYOLO9t dan mencatat pilihannya.
        model = LibreYOLO("LibreHRNetw32-pose.pt")
        result = model(SAMPLE_IMAGE)

        print(result.keypoints.xy.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # coco8-pose.yaml memuat skrip unduhan tertanam, sehingga memerlukan
        # izin eksplisit kecuali datanya sudah tersedia secara lokal.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(
            data="coco8-pose.yaml",
            epochs=50,
            imgsz=640,
            batch=4,
            allow_download_scripts=True,
        )
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreECs-pose.pt data=coco8-pose.yaml \
          epochs=50 imgsz=640 batch=4 allow_download_scripts=True
    - label: Dataset Anda sendiri
      language: python
      code: |
        from libreyolo import LibreYOLO

        # data.yaml harus mendeklarasikan kpt_shape, dan baris label harus
        # memuat tepat 5 + K * D kolom.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(data="my-pose-dataset.yaml", epochs=50, imgsz=640, batch=8)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")

        # val() mengembalikan dict biasa, bukan objek.
        metrics = model.val(data="coco8-pose.yaml", allow_download_scripts=True)

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"], metrics["metrics/keypoints_mAP75"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreECs-pose.pt data=coco8-pose.yaml \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreECs-pose.pt format=onnx imgsz=640
    - label: Gunakan berkas hasil ekspor
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory merutekan berdasarkan akhiran berkas, sehingga artefak hasil
        # ekspor dimuat seperti checkpoint dan mengembalikan objek Results yang sama.
        model = LibreYOLO("LibreECs-pose.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.keypoints.xy)
source_hash: 9de01d1f615bdf33
---

## Definisi

Estimasi pose mengembalikan struktur, bukan hanya luas cakupan. Setiap instance
tetap memperoleh bounding box, kelas, dan skor, serta `K` keypoint dalam urutan tetap,
sehingga indeks 5 berarti bagian tubuh yang sama pada setiap instance dan gambar.
Kumpulan label mendefinisikan urutan tersebut; tidak ada bagian output yang
mengidentifikasi keypoint berdasarkan nama.

`pose` adalah kunci task kanonis, dan akhiran `-pose` pada nama berkas checkpoint
memilihnya, sehingga `task=` tidak diperlukan saat memuat bobot terbitan.

`predict()` mengisi `result.keypoints` bersama `result.boxes`. `.data` berbentuk
`(N, K, 2)` atau `(N, K, 3)` dan sejajar per baris dengan bounding box, sehingga instance
`i` dalam satu payload sama dengan instance `i` pada payload lainnya. `.xy`
mengambil koordinat piksel dan `.xyn` menormalisasinya berdasarkan ukuran gambar
asli. `.conf` adalah kolom ketiga jika checkpoint memprediksinya dan `None` jika
tidak, sedangkan `.has_visible` merupakan mask boolean yang diturunkan darinya,
seluruhnya true jika kolom ketiga tidak ada.

Dua arsitektur mencapai output ini. Model one-stage memprediksi bounding box dan keypoint
dalam satu pass. Model top-down menjalankan detector terlebih dahulu, memotong
setiap instance, lalu meregresi keypoint di dalam crop, sehingga akurasinya
bergantung pada detector di depannya.

## Model

Tiga family dapat berlatih dan memprediksi:
[RF-DETR](/docs/models/rf-detr), [EdgeCrafter](/docs/models/edgecrafter), dan
[YOLO-NAS](/docs/models/yolo-nas), seluruhnya one-stage. RF-DETR memerlukan extra
sendiri, `pip install "libreyolo[rfdetr]"`. RF-DETR dan EdgeCrafter menyertakan
checkpoint pose terbitan dan keduanya mendapat fine-tuning pada dataset satu kelas yang
hanya berisi orang; keypoint head EdgeCrafter ditetapkan saat konstruksi dan
menolak dataset dengan jumlah berbeda, sedangkan RF-DETR menginisialisasi ulang
head-nya. YOLO-NAS mengambil bobot dari CDN milik Deci.AI berdasarkan lisensi
nonkomersial, dan LibreYOLO tidak menerbitkannya; pose head-nya juga dibangun
ulang untuk jumlah keypoint baru, dan hanya family ini dari ketiganya yang jumlah
kelas-nya tidak ditetapkan ke satu. Karena itu, family ini cocok untuk skeleton
multi-kelas atau nonmanusia, seperti pose hewan.

[HRNet](/docs/models/hrnet) adalah pilihan top-down. Model ini memprediksi,
memvalidasi, dan mengekspor, sedangkan `train()`-nya memunculkan
`NotImplementedError`. Jika sumber orang tidak diberikan, model otomatis
memasangkan dirinya dengan detector LibreYOLO9t; `cropped=True` memperlakukan
seluruh gambar sebagai satu instance, `person_boxes=` menerima bounding box yang sudah
tersedia, dan `person_detector=` menamai detector lain.

[SenseNova-Vision](/docs/models/sensenova-vision) juga menghasilkan keypoint.
Model generatif berbasis prompt ini memiliki factory sendiri, `LibreVLM`, dan
extra sendiri; tanpa vocabulary, `set_task("pose")` kembali ke kategori orang.
Bobotnya nonkomersial dan latensi per gambar jauh lebih tinggi daripada pose head
khusus karena setiap prediksi merupakan diffusion decode.

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam
cache lokal.

<code-tabs name="predict" />

Jumlah dan urutan keypoint merupakan properti checkpoint, bukan library, sehingga
model yang dilatih pada skeleton berbeda mengembalikan `K` dan arti per indeks
yang berbeda. Isi kolom keypoint ketiga juga merupakan properti checkpoint:
EdgeCrafter menulis konstanta di sana, bukan skor per titik, dan tidak memiliki
bounding box head sama sekali, sehingga setiap pose bounding box-nya adalah batas cakupan keypoint
instance itu sendiri. Lihat [prediksi](/docs/predict) untuk sumber, streaming, dan
penanganan hasil.

## Format dataset

Tata letaknya sama dengan deteksi: satu berkas label `.txt` per gambar, yang
ditemukan dengan mengganti `images` menjadi `labels` pada path gambar dan
mengubah ekstensinya.

```text
dataset/
  data.yaml
  images/
    train/000001.jpg
    val/000101.jpg
  labels/
    train/000001.txt
    val/000101.txt
```

Satu baris adalah baris deteksi dengan keypoint yang ditambahkan:

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

Jumlah kolom tepat `5 + K * D`, dengan `D` sebagai nilai kedua `kpt_shape`.
Koordinat bounding box dan keypoint berupa float ternormalisasi relatif terhadap lebar dan
tinggi gambar asli. Visibilitas `v`, yang hanya tersedia jika `D` adalah 3,
bernilai `0`, `1`, atau `2`.

YAML menambahkan dua kunci ke kontrak bersama:

```yaml
path: dataset
train: images/train
val: images/val
kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
names:
  0: person
```

`kpt_shape` wajib dan berbentuk `[K, 2]` atau `[K, 3]`. `flip_idx` bersifat
opsional dan merupakan permutasi `0..K-1` yang memberikan indeks setiap keypoint
setelah horizontal flip, sehingga pergelangan kiri tetap menjadi pergelangan
kiri. Jika dihilangkan, augmentasi horizontal flip dinonaktifkan untuk keypoint,
bukan diterapkan dengan urutan indeks yang salah.

## Pelatihan

<code-tabs name="train" />

Pelatihan dilanjutkan dari checkpoint `-pose` terbitan yang sudah memuat keypoint
head; task dibaca dari checkpoint yang dimuat, bukan flag saat pelatihan, sehingga
checkpoint deteksi tidak menjadi proses pose hanya dengan memintanya. `kpt_shape`
dalam YAML harus persis cocok dengan head untuk EdgeCrafter karena head-nya
ditetapkan saat konstruksi, sedangkan RF-DETR dan YOLO-NAS mengubah ukuran head
untuk jumlah berbeda. Lihat [pelatihan](/docs/train) untuk dataset, augmentasi,
multi-GPU, dan logger.

## Validasi

`val()` mengembalikan dictionary biasa berisi kunci `metrics/`. Penilaian
menggunakan evaluasi keypoint COCO berdasarkan Object Keypoint Similarity, yang
memberi bobot error jarak setiap keypoint menurut skala instance dan toleransi
per keypoint, sehingga berperan seperti IoU untuk bounding box. Evaluasi ini memerlukan
`pycocotools`, yang termasuk dalam instalasi dasar.

<code-tabs name="val" />

`metrics/keypoints_mAP50-95` adalah angka utama, presisi rata-rata yang
dirata-ratakan pada ambang batas OKS 0,50 hingga 0,95, dan digunakan pelatihan
untuk memilih epoch terbaik. `metrics/keypoints_mAP50` serta
`metrics/keypoints_mAP75` adalah versi satu ambang batas, sedangkan
`metrics/keypoints_mAP_M` dan `metrics/keypoints_mAP_L` membagi rata-rata menurut
luas instance, medium dan large; evaluasi keypoint COCO tidak mendefinisikan
bucket small. Angka average recall pasangannya adalah
`metrics/keypoints_AR50-95`, `metrics/keypoints_AR50`, `metrics/keypoints_AR75`,
`metrics/keypoints_AR_M`, dan `metrics/keypoints_AR_L`. Setiap kunci pada task ini
diawali `keypoints_`, sehingga kunci `mAP` bounding box dari detector tidak muncul.

## Ekspor

<code-tabs name="export" />

Artefak hasil ekspor dimuat kembali melalui `LibreYOLO()` berdasarkan akhiran
filenya, sehingga berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan
mengembalikan `Results` yang sama. Cakupan format berbeda per family; matriks pada
setiap halaman model dibuat dari kumpulan tervalidasi, bukan diketik manual.
Lihat [ekspor dan deployment](/docs/export) untuk format, extra, dan batasannya.


