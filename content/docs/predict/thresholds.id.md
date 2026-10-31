---
title: Ambang batas dan pemfilteran
seo_title: 'conf, iou, dan max_det di LibreYOLO'
description: >-
  Fungsi conf, iou, max_det, classes, dan agnostic_nms saat prediksi, serta
  family yang mengabaikan iou karena tidak menjalankan NMS sendiri.
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
last_verified: 1.6.0
verification: >-
  Nilai default dikutip dari InferenceRunner.__call__ dalam
  libreyolo/models/base/inference.py. Perilaku NMS per family dibaca dari setiap
  modul dalam libreyolo/postprocess/ dan diperiksa silang terhadap
  _is_nms_free_family dalam libreyolo/backends/base.py. Pemfilteran kelas
  berasal dari InferenceRunner._apply_classes_filter dan _wrap_results.
  agnostic_nms berasal dari check_agnostic_nms dan postprocess_max_det dalam
  libreyolo/utils/predict_args.py serta langkah agnostic dalam
  InferenceRunner._wrap_results. Penanganan open-vocabulary berasal dari
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
  agnostic:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt")


        per_class = model(SAMPLE_IMAGE)

        agnostic = model(SAMPLE_IMAGE, agnostic_nms=True)


        # Kotak tumpang tindih dari kelas berbeda diciutkan menjadi satu dengan
        skor tertinggi.

        print(len(per_class.boxes), len(agnostic.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt agnostic_nms=true \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
source_hash: fa0e6cee02b3e336
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

Validasi deteksi menyediakan `metrics/best_conf` dan ambang per kelas dari F1 pada IoU 0.50. Gunakan sebagai titik awal terukur untuk ambang deployment; lihat [validasi](/docs/train/validation). Segmentasi tidak menyediakan kunci ini.

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

Tiga opsi saat prediksi membuat `iou` berpengaruh bahkan untuk set predictor karena ketiganya
bekerja pada kotak setelah model selesai:

- `tiling=True` merekonsiliasi tile yang tumpang tindih dengan NMS per kelas pada `iou`
- `augment=True` menggabungkan tampilan yang dibalik dengan NMS per kelas pada `iou`
- `agnostic_nms=True` menekan kotak yang tumpang tindih lintas kelas pada `iou`,
  dijelaskan [di bawah](#agnostic_nms)

Dua yang pertama dibahas dalam [Performa inferensi](/docs/predict/performance).

Detektor open-vocabulary memiliki aturan sendiri. Family yang prosesornya menjalankan NMS
mendeklarasikan ambang default sendiri dan menghormati `iou`, seperti OMDet-Turbo. Family
tanpa suppression, Grounding DINO, OWLv2, dan OV-DEIM, mengeluarkan peringatan jika `iou`
diberikan. Peringatan tersebut satu-satunya dari jenisnya dalam library.

## max_det

`max_det` membatasi jumlah prediksi yang dikembalikan untuk satu gambar. Argumen ini berlaku
di semua tempat melalui mekanisme berbeda: family NMS memangkas setelah suppression, sedangkan
set predictor memakainya sebagai ukuran pilihan top-k.

Beberapa family membatasi di bawah nilai yang diminta karena konfigurasi referensi upstream.
SSD membatasi pada 200, dan FCOS pada batas deteksi per gambarnya sendiri. Menaikkan `max_det` melewati angka tersebut tidak berpengaruh.

`max_det` diterapkan secara terpusat, bukan per family, dalam dua kasus: inferensi berbasis
tile, tempat daftar gabungan dipangkas setelah tile direkonsiliasi, dan pemanggilan dengan
`classes` atau `agnostic_nms=True`, tempat pemangkasan terjadi setelah filter atau suppression.

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
yang kemudian dibuang. Dengan `classes` diatur, family mempertahankan anggaran kandidat
sebesar 300 kotak, atau `max_det` bila lebih tinggi, dan pemangkasan ke `max_det` terjadi
setelah filter, sehingga `max_det` kecil tidak menyingkirkan kelas target. Backend hasil
ekspor memakai anggaran yang sama, termasuk engine TensorRT ber-batch. Gambar dengan lebih
dari 300 kotak berskor lebih tinggi dari kelas yang tidak diinginkan tetap dapat memenuhi
anggaran sebelum kelas target tercapai; naikkan `max_det` di atas 300 jika itu terjadi.

## agnostic_nms

NMS setiap family sadar kelas, sehingga secara default dua kotak tumpang tindih dari kelas
berbeda sama-sama bertahan pada nilai `iou` apa pun. `agnostic_nms=True` menambahkan satu
langkah class-agnostic: di antara kotak yang tumpang tindih di atas `iou`, hanya kotak dengan
skor tertinggi yang bertahan, apa pun kelasnya.

<code-tabs name="agnostic" />

Langkah ini berjalan pada deteksi akhir milik family, setelah filter `classes` dan sebelum
pemangkasan ke `max_det`. Karena itu perilakunya sama pada set predictor yang tidak menjalankan
NMS sendiri, pada backend hasil ekspor, serta dengan `augment=True` dan `tiling=True`. Mask dan
keypoint dibuang bersama kotaknya, dan kotak berorientasi dibandingkan dengan IoU terotasi.
Langkah ini mencakup task yang mengembalikan kotak: detect, segment, pose, dan obb.

Family mempertahankan anggaran kandidat sebesar 300 kotak, atau `max_det` bila lebih tinggi,
sebagai bahan kerja langkah ini. Dengan `max_det` di bawah 300, kandidat berskor lebih rendah
mengisi kembali slot yang dibebaskan suppression. Pada 300 atau lebih, termasuk nilai default,
slot itu dibiarkan kosong.

Tempat langkah ini tidak berlaku:

- Pada jalur predict standar, task tanpa kotak memunculkan `ValueError`.
- `LibreEnsemble` dan [HRNet](/docs/models/hrnet) memunculkan `NotImplementedError`.
- Model `LibreSAM` dan [detektor 3D](/docs/tasks/3d-object-detection), yaitu WildDet3D,
  3D-MOOD, FCOS3D, dan DetAny3D, tidak menerima argumen semacam itu dan memunculkan
  `TypeError`.
- [L2CS-Net](/docs/models/l2cs), [PP-OCRv5](/docs/models/pp-ocrv5),
  [SAM 3D Body](/docs/models/sam-3d-body), dan
  [LibreFaceRec](/docs/models/librefacerec) menjalankan jalur predict sendiri dan
  mengabaikannya.

`val(agnostic_nms=True)` menerapkan langkah yang sama pada validasi detect dan segment;
lihat [Validasi dan metrik](/docs/train/validation).

## Hal yang ditolak predict

Dua argumen memunculkan error, bukan peringatan: `visualize` dan `embed` sama-sama memunculkan
`NotImplementedError`. Untuk embedding, muat model dengan `task="embed"`, lalu panggil
`predict` atau `embed` seperti biasa. `agnostic_nms=True` memunculkan error di tempat
ia tidak berlaku, seperti yang tercantum di atas.

Semua argumen tidak dikenal memunculkan `TypeError` yang menyebut opsi yang didukung, sehingga
kesalahan ketik langsung gagal, bukan diabaikan.

Argumen berikut diterima, diberi peringatan, lalu dibuang: `boxes`, `dnn`, `half`,
`line_width`, `retina_masks`, `show_conf`, `show_labels`, dan `verbose`.
