---
title: Ambang batas dan pemfilteran
seo_title: 'conf, iou, dan max_det di LibreYOLO'
description: >-
  Fungsi sebenarnya conf, iou, max_det, dan classes saat prediksi, family yang
  mengabaikan iou karena tidak menjalankan NMS, serta alasan agnostic_nms tidak
  berpengaruh.
lead: >-
  Empat argumen menentukan prediksi yang bertahan: conf, iou, max_det, dan
  classes. Hanya dua yang berlaku pada setiap family karena set predictor
  mendekode kumpulan query tetap dan tidak pernah menjalankan NMS.
keywords:
  - yolo conf threshold
  - iou threshold nms
  - max_det
  - filter class detection python
  - agnostic nms
  - nms free detr
  - detection confidence threshold
  - class filtering inference
last_verified: 1.5.0
verification: >-
  Nilai default dikutip dari InferenceRunner.__call__ dalam
  libreyolo/models/base/inference.py. Perilaku NMS per family dibaca dari setiap
  modul dalam libreyolo/postprocess/ dan diperiksa silang terhadap
  _is_nms_free_family dalam libreyolo/backends/base.py. Pemfilteran kelas
  berasal dari InferenceRunner._apply_classes_filter dan _wrap_results. Status
  agnostic_nms berasal dari NOOP_PREDICT_KWARGS dalam
  libreyolo/utils/predict_args.py. Penanganan open-vocabulary berasal dari
  NMS_THRESHOLD dalam libreyolo/models/openvocab/base.py. Nilai default validasi
  berasal dari BaseModel.val.
snippets:
  basic:
    - label: Empat argumen
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        result = model(
            SAMPLE_IMAGE,
            conf=0.25,      # pertahankan prediksi pada atau di atas skor ini
            iou=0.45,       # ambang tumpang tindih NMS, jika NMS berjalan
            max_det=300,    # batas per gambar
            classes=None,   # atau daftar id kelas
        )
        print(len(result.boxes))
    - label: Menguji berbagai conf
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        for conf in (0.1, 0.25, 0.5, 0.75):
            result = model(SAMPLE_IMAGE, conf=conf)
            print(conf, len(result.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt conf=0.4 iou=0.5 max_det=100 \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  classes:
    - label: Memfilter ke kelas tertentu
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # Id kelas mengindeks model.names. Pada COCO, 0 adalah person.
        result = model(SAMPLE_IMAGE, classes=[0])

        print({result.names[int(c)] for c in result.boxes.cls.tolist()})
    - label: Menemukan id dari nama
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        result = model(SAMPLE_IMAGE)

        wanted = {"person", "backpack"}
        ids = [i for i, name in result.names.items() if name in wanted]
        print(ids)

        filtered = model(SAMPLE_IMAGE, classes=ids)
        print(len(filtered.boxes))
  nmsfree:
    - label: iou pada family tanpa NMS
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # RF-DETR mendekode kumpulan query tetap, sehingga iou tidak mengubah
        apa pun.

        model = LibreYOLO("LibreRFDETRs.pt")


        loose = model(SAMPLE_IMAGE, iou=0.9)

        tight = model(SAMPLE_IMAGE, iou=0.1)


        # Jumlah sama untuk keduanya. conf dan max_det adalah kontrol yang
        berfungsi.

        print(len(loose.boxes), len(tight.boxes))
source_hash: 0b978963c356027d
---

## Empat argumen

| Argumen | Default | Berlaku pada |
|---|---|---|
| `conf` | `0.25` | Setiap family |
| `iou` | `0.45` | Family yang menjalankan non-maximum suppression |
| `max_det` | `300` | Setiap family |
| `classes` | `None` | Setiap family |

<code-tabs name="basic" />

Dua bersifat universal dan dua lainnya tidak. Inilah hal terpenting yang perlu diketahui
sebelum menyetel apa pun.

Validasi sengaja memakai default berbeda: `val()` berjalan pada `conf=0.001` dan `iou=0.6`
karena average precision dihitung pada kurva precision-recall penuh dan batas 0.25 akan
memotongnya.

## conf

`conf` adalah skor yang menjadi batas pembuangan prediksi. Argumen ini berlaku pada setiap
family, termasuk yang tidak pernah menjalankan NMS, dan menjadi kontrol pertama saat deteksi
terlalu banyak atau terlalu sedikit.

Default `0.25` cocok untuk melihat gambar. Sistem downstream biasanya memerlukan nilai lebih
tinggi; pengukuran akurasi memerlukan nilai jauh lebih rendah.

## iou

`iou` adalah nilai tumpang tindih yang membuat non-maximum suppression membuang kotak dengan
skor lebih rendah dari dua kotak kelas sama. Nilai ini hanya bermakna jika family menjalankan
suppression.

Set predictor mendekode jumlah query tetap dan mengambil yang memiliki skor tertinggi.
Duplikat ditekan di dalam arsitektur saat pelatihan, bukan melalui tahap pascapemrosesan,
sehingga tidak ada ambang yang dapat diubah. Family berikut menerima `iou` demi kesetaraan API
dan mengabaikannya:

CenterNet, DEIM, DETR, Deformable DETR, D-FINE, DINO-DETR, EdgeCrafter, Faster R-CNN,
LW-DETR, Mask R-CNN, RF-DETR, RT-DETR, dan head YOLOv9 end-to-end. Varian yang dibuat di
atas decoder tersebut mewarisi perilakunya.

<code-tabs name="nmsfree" />

Sebagian besar menyatakannya dalam docstring pascapemrosesan, tetapi tidak ada peringatan saat
runtime, sehingga pengujian berbagai `iou` pada RF-DETR menghasilkan garis datar, bukan error.
Faster R-CNN dan Mask R-CNN sedikit berbeda: keduanya sudah menjalankan NMS di dalam model
pada ambang upstream tetap yang tidak dapat diubah oleh `iou`.

Family berikut menggunakannya: YOLOv1 hingga YOLOv4, YOLOv7, YOLOv9, YOLOX, YOLO-NAS,
RTMDet, PicoDet, EfficientDet, FCOS, RetinaNet, dan SSD.

Dua opsi saat prediksi membuat `iou` berpengaruh bahkan untuk set predictor karena keduanya
menggabungkan kotak setelah model selesai:

- `tiling=True` merekonsiliasi tile yang tumpang tindih dengan NMS per kelas pada `iou`
- `augment=True` menggabungkan tampilan yang dibalik dengan NMS per kelas pada `iou`

Keduanya dibahas dalam [Performa inferensi](/docs/predict/performance).

Detektor open-vocabulary memiliki aturan sendiri. Family yang prosesornya menjalankan NMS
mendeklarasikan ambang default sendiri dan menghormati `iou`, seperti OMDet-Turbo. Family
tanpa suppression, Grounding DINO, OWLv2, dan OV-DEIM, mengeluarkan peringatan jika `iou`
diberikan. Peringatan tersebut satu-satunya dari jenisnya dalam library.

## max_det

`max_det` membatasi jumlah prediksi yang dikembalikan untuk satu gambar. Argumen ini berlaku
di semua tempat melalui mekanisme berbeda: family NMS memangkas setelah suppression, sedangkan
set predictor memakainya sebagai ukuran pilihan top-k.

Beberapa family membatasi di bawah nilai yang diminta karena konfigurasi referensi upstream.
SSD membatasi pada 200, segmentasi instance RTMDet pada 100, dan FCOS pada batas deteksi per
gambarnya sendiri. Menaikkan `max_det` melewati angka tersebut tidak berpengaruh.

Satu tempat `max_det` diterapkan secara terpusat, bukan per family, adalah inferensi berbasis
tile, tempat daftar gabungan dipangkas setelah tile direkonsiliasi.

## Pemfilteran kelas

<code-tabs name="classes" />

`classes` menerima daftar id kelas dan hanya mempertahankan prediksi yang kelasnya ada dalam
daftar. Id mengindeks `result.names`, dan cara paling pasti memperoleh id adalah membaca
`names` dari hasil, bukan mengasumsikan urutan dataset.

Pemfilteran terjadi secara terpusat setelah pascapemrosesan setiap family, dalam satu funnel
yang dilalui setiap jalur prediksi. Ada dua konsekuensi penting. Fitur ini berfungsi pada
setiap family, termasuk yang tidak memiliki NMS. Fitur ini juga memfilter payload yang sejajar
dengan kotak, sehingga mask, keypoint, dan kotak berorientasi dipangkas bersamanya dan tidak
menjadi tidak cocok.

Pada command line, `classes` menerima bilangan bulat tunggal, daftar, atau string dipisahkan koma:

```bash
libreyolo predict model=LibreYOLO9s.pt classes=0 source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
libreyolo predict model=LibreYOLO9s.pt classes="[0,2,5]" source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
```

Pemfilteran bukan akurasi gratis. Model tetap menghabiskan kapasitas untuk memprediksi kelas
yang kemudian dibuang, dan `max_det` diterapkan oleh family sebelum filter, sehingga gambar
yang dipenuhi kelas tidak diinginkan dapat mencapai batas sebelum kelas target. Turunkan
`conf` atau naikkan `max_det` jika itu terjadi.

## agnostic_nms

`agnostic_nms` diterima dan tidak melakukan apa pun. Meneruskannya memunculkan peringatan bahwa
argumen tersebut no-op demi kompatibilitas command line, lalu argumen dibuang.

Tidak ada mode suppression class-agnostic. Setiap pemanggilan NMS dalam library sadar kelas,
sehingga dua kotak tumpang tindih dari kelas berbeda sama-sama bertahan pada nilai `iou` apa
pun. Jika ini bermasalah, filter lebih dulu dengan `classes`, atau lakukan suppression lintas
kelas sendiri pada `result.boxes`.

## Hal yang ditolak predict

Dua argumen memunculkan error, bukan peringatan: `visualize` dan `embed` sama-sama memunculkan
`NotImplementedError`. Untuk embedding, muat model dengan `task="embed"`, lalu panggil
`predict` atau `embed` seperti biasa.

Semua argumen tidak dikenal memunculkan `TypeError` yang menyebut opsi yang didukung, sehingga
kesalahan ketik langsung gagal, bukan diabaikan.

Argumen berikut diterima, diberi peringatan, lalu dibuang: `agnostic_nms`, `boxes`, `dnn`,
`half`, `line_width`, `retina_masks`, `show_conf`, `show_labels`, dan `verbose`.
