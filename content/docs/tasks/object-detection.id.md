---
title: Deteksi objek di LibreYOLO
seo_title: ''
description: >-
  Mendeteksi objek sebagai kotak yang sejajar dengan sumbu di LibreYOLO:
  keluarga yang melayani task, format label, dan panggilan predict, train,
  validate, dan export.
lead: >-
  Deteksi objek menemukan setiap contoh objek dalam gambar dan mengembalikan
  persegi panjang yang sejajar dengan sumbu, label kelas, dan skor untuk
  masing-masing. Kunci task adalah detect.
keywords:
  - deteksi objek python
  - mendeteksi objek dalam gambar
  - deteksi bounding box
  - perpustakaan deteksi objek MIT
  - alternatif YOLO
  - melatih pendeteksi objek
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(result.names[int(box.cls)], float(box.conf), box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9t.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: 'family lain, panggilan yang sama'
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Pabrik merutekan di checkpoint, dan setiap pendeteksi mengembalikan

        # objek Results yang sama, sehingga mengganti family hanya butuh satu
        baris perubahan.

        model = LibreYOLO("LibreDFINEn.pt")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy.shape)
    - label: Video dan streaming
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # Sumber apa pun yang diterima perpustakaan: file, folder, URL, indeks
        webcam,

        # aliran RTSP, atau daftar .streams.

        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # coco128.yaml mengunduh sampel 128 gambar pada penggunaan pertama.
        Arahkan data

        # ke YAML dataset Anda sendiri untuk menjalankan yang sesungguhnya.

        model.train(data="coco128.yaml", epochs=50, imgsz=640, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=coco128.yaml \
          epochs=50 imgsz=640 batch=8
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreYOLO9t.pt data=coco128.yaml \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # val() mengembalikan dict biasa, bukan objek.
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"], metrics["metrics/mAP75"])
        print(metrics["metrics/AR100"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9t.pt data=coco128.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLO9t.pt format=onnx imgsz=640
    - label: Gunakan file yang diekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Pabrik memproses berdasarkan akhiran file, jadi artefak yang diekspor
        dimuat

        # seperti checkpoint dan mengembalikan objek Results yang sama.

        model = LibreYOLO("LibreYOLO9t.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: c735b6e3de78dd2b
---

## Definisi

Deteksi objek menjawab di mana setiap objek berada dan apa itu. Satu gambar masuk,
satu baris per instance keluar: empat angka untuk persegi panjang, indeks kelas dan sebuah
. Tidak ada informasi tentang bentuk piksel, orientasi, atau bagian yang disertakan, yang menjadi
pembeda dari [instance segmentation](/docs/tasks/instance-segmentation),
[kotak berorientasi](/docs/tasks/oriented-detection) dan
[pose](/docs/tasks/pose-estimation).

`detect` adalah kunci task kanonik dan default: sebuah checkpoint yang nama filenya
tidak memiliki akhiran task dimuat sebagai detektor.

`predict()` mengisi `result.boxes`. `.xyxy` memberikan sudut piksel pada
kanvas gambar asli, `.conf` skor, dan `.cls` indeks kelas ke dalam
`result.names`. `.xywh`, `.xyxyn` dan `.xywhn` adalah tampilan turunan dari
baris yang sama, dan `.id` membawa id pelacak setelah pelacak dipasang. Iterasi
sebuah objek `Boxes` menghasilkan irisan satu baris, jadi `box.cls`, `box.conf` dan
`box.xyxy` semuanya berfungsi per deteksi.

## Model

Dua belas keluarga baik melatih maupun memprediksi: [YOLOv9](/docs/models/yolov9),
[RF-DETR](/docs/models/rf-detr), [EdgeCrafter](/docs/models/edgecrafter),
[RT-DETR](/docs/models/rt-detr), [D-FINE](/docs/models/d-fine),
[DEIM](/docs/models/deim), [Dome-DETR](/docs/models/dome-detr),
[YOLO-NAS](/docs/models/yolo-nas),
[YOLOX](/docs/models/yolox), [YOLOv7](/docs/models/yolov7),
[RTMDet](/docs/models/rtmdet) dan [PicoDet](/docs/models/picodet). YOLOv9 dan
RF-DETR adalah dua keluarga unggulan, dan fitur-fitur mendarat pada mereka terlebih dahulu. RF-DETR
membutuhkan tambahan sendiri, `pip install "libreyolo[rfdetr]"`; sisanya berjalan pada
paket dasar.

Sebelas lagi memprediksi, memvalidasi dan mengekspor, tetapi `train()` mereka naik
`NotImplementedError`: [LW-DETR](/docs/models/lw-detr),
[DETR](/docs/models/detr), [DETR](/docs/models/deformable-detr) yang dapat berubah bentuk,
[DINO-DETR](/docs/models/dino-detr), [Faster R-CNN](/docs/models/faster-rcnn),
[Mask R-CNN](/docs/models/mask-rcnn), [FCOS](/docs/models/fcos),
[RetinaNet](/docs/models/retinanet), [SSD](/docs/models/ssd),
[CenterNet](/docs/models/centernet) dan
[EfficientDet](/docs/models/efficientdet).

Garis keturunan Darknet, [YOLOv1](/docs/models/yolov1),
[YOLOv2](/docs/models/yolov2), [YOLOv3](/docs/models/yolov3) dan
[YOLOv4](/docs/models/yolov4), disimpan sebagai pameran beku: prediksi, validasi
dan ekspor pekerjaan, pelatihan tidak.

Sebuah kelompok terpisah mengambil daftar kelasnya saat runtime daripada dari
checkpoint, jadi mendeteksi nama yang belum pernah terlihat dalam pelatihan:
[Grounding DINO](/docs/models/grounding-dino), [OWLv2](/docs/models/owlv2),
[OMDet-Turbo](/docs/models/omdet-turbo) dan [OV-DEIM](/docs/models/ov-deim),
ditambah keluarga vision-language
[Florence-2](/docs/models/florence-2), [Kosmos-2](/docs/models/kosmos-2),
[Qwen3-VL](/docs/models/qwen3-vl), [SmolVLM2](/docs/models/smolvlm2),
[InternVL3](/docs/models/internvl3), [LFM2-VL](/docs/models/lfm2-vl),
[LocateAnything](/docs/models/locate-anything),
[SenseNova-Vision](/docs/models/sensenova-vision) dan
[LibreMODUS](/docs/models/libremodus). Ini dimuat melalui pabrikannya sendiri dan
tambahan; setiap halaman model membawa panggilan yang tepat.

## Prediksi

Bobot diunduh dari Hugging Face saat penggunaan pertama dan disimpan secara lokal.

<code-tabs name="predict" />

`conf` mengatur ambang kepercayaan dan `max_det` membatasi jumlah baris.
`iou` adalah ambang NMS, sehingga hanya berpengaruh pada family yang menjalankan NMS;
RF-DETR dan YOLOv9 head ujung-ke-ujung mendekodekan satu set prediksi tetap dan
abaikan itu. Lihat [prediction](/docs/predict) untuk sumber, streaming, dan hasil
penanganan.

## format Dataset

Satu file label `.txt` per gambar, ditemukan dengan menukar `images` dengan `labels` di
jalur gambar dan mengubah ekstensi.

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

Setiap baris terdiri dari tepat lima kolom, sebuah indeks kelas diikuti oleh sebuah nilai yang dinormalisasi
kotak-tengah-dan-ukuran:

```text
<class_id> <cx> <cy> <w> <h>
```

Koordinat adalah bilangan desimal di `[0, 1]`, relatif terhadap lebar gambar asli dan
tinggi. `w` dan `h` harus positif. File label yang hilang atau kosong berarti
Gambar tidak memiliki objek. Baris tidak membawa kepercayaan dan tidak ada ID jalur.

YAML menamai pembagian dan kelas-kelasnya:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: person
  1: bicycle
```

`train` dan `val` mungkin adalah direktori gambar, file daftar-gambar `.txt`, atau daftar
dari keduanya. `nc` bersifat opsional dan harus cocok dengan `names` jika ada. Native COCO
JSON juga berfungsi: tambahkan pemetaan `annotations` dari nama split ke file JSON, dan
jalur yang terpisah kemudian memberikan akar gambar. Ketika `names` ada, itu menentukan
ID label, jadi nama kategori JSON harus cocok dengannya.

## Kereta

<code-tabs name="train" />

`epochs`, `imgsz`, `batch` dan `lr0` adalah argumen yang bergerak terlebih dahulu. `lr0` adalah
yang tidak terbawa antar keluarga: sebuah laju detektor konvolusional
tolerates akan menyimpang dari satu transformer, jadi ambil nilai dari halaman model
daripada dari contoh family lainnya. family juga dapat mengabaikan sebuah argumen
secara langsung, dan halamannya mencantumkan yang mana. Lihat [training](/docs/train) untuk set data,
augmentasi, multi-GPU, dan pencatat.

## Validasi

`val()` mengembalikan kamus biasa dari kunci `metrics/`, dihitung dengan COCO
evaluasi atas pembagian yang dinamai oleh `val` dalam YAML dataset.

<code-tabs name="val" />

`metrics/mAP50-95` adalah rata-rata presisi yang dihitung rata-rata atas ambang IoU 0,50
hingga 0,95, dan ini adalah angka utama. `metrics/mAP50` dan `metrics/mAP75` adalah
versi ambang tunggal. `metrics/mAP_small`, `metrics/mAP_medium` dan
`metrics/mAP_large` membagi rata-rata yang sama berdasarkan area objek, dan `metrics/AR1`,
`metrics/AR10`, `metrics/AR100`, `metrics/AR_small`, `metrics/AR_medium` dan
`metrics/AR_large` adalah angka rata-rata yang cocok-recall.
`metrics/AR_max_det` dan `metrics/max_det` mencatat batas deteksi yang digunakan oleh
.

Bacalah `metrics/precision` dan `metrics/recall` dengan teliti pada task ini. Mereka
disimpan untuk kompatibilitas mundur dan adalah alias, bukan titik operasi:
`metrics/precision` memegang nilai yang sama seperti `metrics/mAP50-95`, dan
`metrics/recall` memiliki nilai yang sama seperti `metrics/AR100`. Memplotnya sebagai
pasangan presisi-recall melaporkan satu angka dua kali. Empat kunci juga diulang di bawah
akhiran `(B)`, untuk kotak, sehingga sebuah kunci deteksi terbaca sama pada sebuah model yang
juga memprediksi mask: `metrics/mAP50-95(B)`, `metrics/mAP50(B)`,
`metrics/precision(B)` dan `metrics/recall(B)`.

## Ekspor

<code-tabs name="export" />

Sebuah artefak yang diekspor dimuat kembali melalui `LibreYOLO()` pada akhiran file-nya, jadi sebuah
File `.onnx` atau `.engine` berperilaku seperti checkpoint dan mengembalikan hal yang sama
`Results`. Cakupan format berbeda menurut family; matriks pada setiap halaman model adalah
dihasilkan dari set yang tervalidasi daripada diketik dengan tangan. Lihat
[ekspor dan deploy](/docs/export) untuk format, tambahannya, dan mereka]
batasan.

