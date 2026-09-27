---
title: YOLOv9
families:
  - yolo9
seo_title: 'YOLOv9: prediksi, latih, dan ekspor di bawah MIT'
description: >-
  Jalankan YOLOv9 di LibreYOLO, termasuk head end-to-end tanpa NMS dan head
  objek kecil stride-4. Instal, prediksi, latih, validasi, dan ekspor.
lead: >-
  Detektor konvolusional satu tahap: satu tahap menilai grid kotak yang padat
  dan NMS membuang duplikat. LibreYOLO menyediakan tiga variannya, salah satunya
  tidak memiliki tahap NMS.
keywords:
  - YOLOv9
  - YOLO9
  - deteksi objek
  - NMS-free detection
  - end-to-end detection
  - small object detection
  - programmable gradient information
  - GELAN
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Tanpa NMS
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Pemanggilan sama, checkpoint berbeda. Head end-to-end mengembalikan
        prediksi

        # dengan skor teratasnya sendiri, sehingga NMS tidak berjalan dan iou
        diabaikan.

        model = LibreYOLO("LibreYOLO9E2Es.pt")

        result = model(SAMPLE_IMAGE, conf=0.25, max_det=300)


        print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=100 imgsz=640 batch=16
    - label: Objek kecil
      language: python
      code: >
        from libreyolo import LibreYOLO9P2


        # Varian stride-4 tidak memiliki checkpoint COCO sendiri, jadi sebutkan

        # checkpoint deteksi dasar: backbone dan neck dimuat tanpa perubahan,

        # sedangkan tower head stride-4 dimulai dari inisialisasi acak.

        model = LibreYOLO9P2(None, size="s")

        model.train(data="my-dataset.yaml", epochs=100,
        pretrained="LibreYOLO9s.pt")
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=my-dataset.yaml
    - label: Dibandingkan dengan COCO
      language: bash
      code: |
        # YAML COCO bawaan menyertakan skrip unduhan tertanam, sehingga
        # memerlukan izin eksplisit kecuali dataset sudah tersedia secara lokal.
        libreyolo val model=LibreYOLO9c.pt data=coco.yaml imgsz=640 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx imgsz=640
    - label: Dengan NMS dalam graph
      language: bash
      code: |
        libreyolo export model=LibreYOLO9s.pt format=onnx nms=True \
          conf=0.25 iou=0.45 max_det=300
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan sufiks berkas, sehingga artefak hasil
        ekspor

        # dimuat seperti checkpoint lain dan mengembalikan objek Results yang
        sama.

        model = LibreYOLO("LibreYOLO9s.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: eaa6023a4a0b9e71
---

## Instalasi

YOLOv9 tidak memerlukan komponen tambahan selain paket dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam cache lokal.

<code-tabs name="predict" />

Objek `Results` yang dikembalikan sama dengan yang dikembalikan setiap family, jadi mengganti
detektor hanya memerlukan perubahan satu baris. Pada model dasar dan stride-4, `conf`
menetapkan ambang batas confidence dan `iou` menetapkan ambang NMS. Model end-to-end tidak
menjalankan NMS dan mengabaikan `iou`, sehingga `conf` dan `max_det` menentukan bentuk
outputnya. Lihat [prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Varian

Tiga varian memakai satu backbone. Ketiganya hanya mendeteksi dan menerima argumen yang sama.

Model dasar memprediksi pada tiga skala feature dan menghapus kotak duplikat dengan NMS.

Model end-to-end mempertahankan head tersebut dan menambahkan cabang pencocokan one-to-one
di sampingnya. Inferensi hanya membaca cabang one-to-one dan mengambil prediksi dengan skor
teratas, sehingga NMS tidak berjalan. Pilih model ini jika runtime deployment tidak memiliki
operator NMS.

Model stride-4 mengekspos satu level lebih atas pada backbone, memperpanjang neck ke level
tersebut, dan memprediksi pada empat skala, bukan tiga. Skala tambahan ditujukan untuk objek
yang hanya mencakup sedikit piksel; satu checkpoint yang dipublikasikan untuknya dilatih pada
gambar udara. Checkpoint deteksi dasar dapat ditransfer ke model ini: backbone dan neck dimuat
tanpa perubahan, ketiga tower head pretrained bergeser satu slot, dan tower stride-4 dimulai
dari inisialisasi acak.

<benchmark-table task="detect" />

<va-embed />

## Pelatihan

<code-tabs name="train" />

`pretrained` menentukan titik awal proses. Berikan `True` untuk memuat checkpoint terbitan
bagi model dan ukuran yang sama, atau nama maupun path untuk yang lain. Tensor dengan bentuk
yang tidak cocok dilewati, bukan ditolak, dan proses mencatat jumlah tensor yang dimuat,
sehingga checkpoint yang dilatih pada jumlah kelas berbeda tetap dapat menjadi titik awal.

Model stride-4 tidak memiliki checkpoint COCO terbitan sendiri, sehingga `True` diselesaikan
menjadi berkas yang tidak ada dan pengunduhan gagal. Sebutkan checkpoint deteksi dasar sebagai
gantinya.

Lihat [pelatihan](/docs/train) untuk dataset, augmentasi, multi-GPU, dan logger.

## Validasi

`val()` mengembalikan dictionary dengan key `metrics/` yang mencakup presisi, recall,
mAP 50, dan mAP 50-95, yang diukur terhadap dataset apa pun dalam format yang digunakan
untuk pelatihan.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Tanda centang berlaku untuk ketiga varian: jika kemampuannya berbeda, matriks mencantumkan
dukungan terlemah dari ketiganya.

Artefak hasil ekspor dimuat kembali melalui `LibreYOLO()` berdasarkan sufiks berkasnya,
sehingga berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan mengembalikan
`Results` yang sama. Menjalankan graph pada runtime mandiri tanpa memasang LibreYOLO juga
didukung, tetapi prapemrosesan dan pascapemrosesannya harus ditulis sendiri.

Untuk model deteksi dasar, setengah pascapemrosesan tersebut dapat dipindahkan ke graph.
`nms=True` pada ekspor ONNX menempatkan suppression di dalam model, dan output pertama
menjadi tensor tetap `(1, max_det, 6)` dengan baris `x1, y1, x2, y2, score, class`, yang
diberi padding nol setelah jumlah deteksi. Graph tersebut memakai batch 1 dan tidak memiliki
sumbu dinamis. Model end-to-end dan stride-4 tidak menerima flag itu.

Setiap format memasang komponen tambahan berbeda dan menerima beberapa argumen sendiri.
Keduanya dijelaskan pada halaman format tersebut.

<code-tabs name="export" />

## Checkpoint

Setiap berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box>

Satu checkpoint di sini tidak berlisensi MIT. Model stride-4 yang dilatih pada
VisDrone2019-DET mewarisi ketentuan CC BY-NC-SA 3.0 dataset tersebut: hanya untuk penggunaan
nonkomersial, share-alike pada semua turunannya, dan berada di luar lisensi permisif yang
dipakai family lainnya. Model ini memprediksi kelas udara VisDrone, bukan kelas COCO.
Library mencetak semua informasi ini sebelum mengunduh berkas.

</provenance-box>

## Sitasi

<citation-block />
