---
title: Ambang batas dan pemfilteran
seo_title: 'conf, iou, dan max_det di LibreYOLO'
description: >-
  Apa yang sebenarnya dilakukan conf, iou, max_det, dan classes saat prediksi,
  family mana yang mengabaikan iou karena tidak menjalankan NMS, serta alasan
  agnostic_nms tidak berpengaruh.
lead: >-
  Empat argumen menentukan prediksi yang bertahan: conf, iou, max_det, dan
  classes. Hanya dua yang berlaku untuk setiap family karena set predictor
  mendekode set query tetap dan tidak pernah menjalankan NMS.
keywords:
  - ambang batas conf yolo
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
            iou=0.45,       # ambang batas overlap NMS, bila NMS berjalan
            max_det=300,    # batas per gambar
            classes=None,   # atau daftar ID kelas
        )
        print(len(result.boxes))
    - label: Menyapu conf
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
    - label: Filter ke kelas tertentu
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # ID kelas mengindeks model.names. Pada COCO, 0 adalah person.
        result = model(SAMPLE_IMAGE, classes=[0])

        print({result.names[int(c)] for c in result.boxes.cls.tolist()})
    - label: Temukan ID berdasarkan nama
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
    - label: iou pada family yang tidak menjalankan NMS
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # RF-DETR mendekode set query tetap, sehingga iou tidak mengubah apa pun
        di sini.

        model = LibreYOLO("LibreRFDETRs.pt")


        loose = model(SAMPLE_IMAGE, iou=0.9)

        tight = model(SAMPLE_IMAGE, iou=0.1)


        # Jumlah sama untuk keduanya. conf dan max_det adalah kontrol yang
        bekerja.

        print(len(loose.boxes), len(tight.boxes))
source_hash: 0b978963c356027d
---

## Empat argumen

| Argumen | Default | Berlaku untuk |
|---|---|---|
| `conf` | `0.25` | Setiap family |
| `iou` | `0.45` | Family yang menjalankan non-maximum suppression |
| `max_det` | `300` | Setiap family |
| `classes` | `None` | Setiap family |

<code-tabs name="basic" />

Dua argumen bersifat universal dan dua lainnya tidak. Ini adalah hal terpenting
yang perlu diketahui sebelum menyetel apa pun.

Validasi sengaja memakai nilai default berbeda. `val()` berjalan pada
`conf=0.001` dan `iou=0.6` karena average precision dihitung pada kurva
precision-recall penuh, dan cutoff 0.25 akan memotongnya.

## conf

`conf` adalah skor yang jika tidak tercapai akan membuat prediksi dibuang.
Argumen ini berlaku untuk setiap family, termasuk yang tidak pernah menjalankan
NMS, dan menjadi kontrol pertama yang perlu diubah jika deteksi terlalu banyak
atau terlalu sedikit.

Nilai default `0.25` cocok untuk melihat gambar. Sistem downstream biasanya
memerlukan nilai lebih tinggi, sedangkan pengukuran akurasi memerlukan nilai
jauh lebih rendah.

## iou

`iou` adalah overlap yang jika terlampaui membuat non-maximum suppression
menghapus box berskor lebih rendah dari dua box berkelas sama. Argumen ini hanya
bermakna bila family benar-benar menjalankan suppression.

Set predictor mendekode jumlah query tetap dan mengambil query dengan skor
tertinggi. Duplikat di-suppress di dalam arsitektur selama pelatihan, bukan
melalui langkah postprocessing, sehingga tidak ada ambang batas untuk diubah.
Family berikut menerima `iou` demi paritas API dan mengabaikannya:

CenterNet, DEIM, DETR, Deformable DETR, D-FINE, DINO-DETR, EdgeCrafter,
Faster R-CNN, LW-DETR, Mask R-CNN, RF-DETR, RT-DETR, dan head end-to-end YOLOv9.
Varian yang dibangun pada decoder tersebut mewarisi perilakunya.

<code-tabs name="nmsfree" />

Sebagian besar menyatakannya dalam docstring postprocessing, tetapi tidak ada
peringatan saat runtime. Karena itu, penyapuan `iou` pada RF-DETR menghasilkan
garis datar, bukan error. Faster R-CNN dan Mask R-CNN sedikit berbeda. Keduanya
sudah menjalankan NMS di dalam model pada ambang batas upstream tetap yang tidak
dapat diubah secara didukung melalui `iou`.

Family berikut menggunakannya: YOLOv1 hingga YOLOv4, YOLOv7, YOLOv9, YOLOX,
YOLO-NAS, RTMDet, PicoDet, EfficientDet, FCOS, RetinaNet, dan SSD.

Dua opsi waktu prediksi membuat `iou` relevan bahkan bagi set predictor karena
keduanya menggabungkan box setelah model selesai:

- `tiling=True` menyelaraskan tile yang tumpang tindih dengan NMS per kelas pada `iou`
- `augment=True` menggabungkan tampilan yang dibalik dengan NMS per kelas pada `iou`

Keduanya dibahas dalam [Performa inferensi](/docs/predict/performance).

Detector open-vocabulary memiliki aturan sendiri. Family yang processornya
menjalankan NMS mendeklarasikan ambang batas default sendiri dan menghormati
`iou`, seperti OMDet-Turbo. Family tanpa suppression, yaitu Grounding DINO,
OWLv2, dan OV-DEIM, mengeluarkan peringatan saat `iou` diteruskan. Peringatan
tersebut adalah satu-satunya yang sejenis dalam library.

## max_det

`max_det` membatasi jumlah prediksi yang dikembalikan untuk satu gambar. Argumen
ini berlaku di semua tempat, tetapi melalui mekanisme berbeda. Family NMS
memotong setelah suppression, sedangkan set predictor menggunakannya sebagai
ukuran pemilihan top-k.

Beberapa family membatasi di bawah nilai yang diminta karena konfigurasi
referensi upstream menetapkannya demikian. SSD membatasi pada 200, segmentasi
instance RTMDet pada 100, dan FCOS pada batas deteksi per gambarnya sendiri.
Menaikkan `max_det` melampaui angka tersebut tidak berpengaruh.

Satu-satunya tempat `max_det` diterapkan secara terpusat, bukan per family,
adalah inferensi berbasis tile, saat daftar gabungan dipotong setelah tile
diselaraskan.

## Pemfilteran kelas

<code-tabs name="classes" />

`classes` menerima daftar ID kelas dan hanya mempertahankan prediksi dengan
kelas yang termasuk di dalamnya. ID mengindeks `result.names`, dan cara paling
pasti untuk mendapatkannya adalah membaca `names` dari hasil, bukan mengasumsikan
urutan dataset.

Pemfilteran dilakukan secara terpusat setelah postprocessing setiap family,
dalam satu funnel yang dilalui setiap jalur prediksi. Hal ini memiliki dua
konsekuensi penting. Filter bekerja pada setiap family, termasuk yang tanpa
NMS. Filter juga memotong payload yang selaras dengan box, sehingga mask,
keypoint, dan box berorientasi dipotong bersamanya dan tidak menjadi tidak
selaras.

Pada command line, `classes` menerima satu integer, daftar, atau string yang
dipisahkan koma:

```bash
libreyolo predict model=LibreYOLO9s.pt classes=0 source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
libreyolo predict model=LibreYOLO9s.pt classes="[0,2,5]" source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
```

Pemfilteran tidak memberikan akurasi gratis. Model tetap menghabiskan anggaran
untuk memprediksi kelas yang kemudian dibuang, dan `max_det` diterapkan oleh
family sebelum filter. Karena itu, gambar yang penuh kelas tidak diinginkan
dapat mencapai batas sebelum kelas yang dicari tercapai. Turunkan `conf` atau
naikkan `max_det` jika itu terjadi.

## agnostic_nms

`agnostic_nms` diterima dan tidak melakukan apa pun. Meneruskannya memunculkan
peringatan bahwa argumen ini adalah no-op untuk kompatibilitas command line,
lalu argumen dibuang.

Tidak ada mode suppression class-agnostic. Setiap panggilan NMS dalam library
bersifat class-aware, sehingga dua box bertumpang tindih dari kelas berbeda
sama-sama bertahan pada nilai `iou` apa pun. Jika hal ini menjadi masalah,
filter lebih dulu dengan `classes` atau lakukan suppression lintas kelas sendiri
pada `result.boxes`.

## Hal yang ditolak predict

Dua argumen memunculkan error, bukan peringatan. `visualize` dan `embed`
keduanya memunculkan `NotImplementedError`. Untuk embedding, muat model dengan
`task="embed"`, lalu panggil `predict` atau `embed` seperti biasa.

Argumen yang tidak dikenal memunculkan `TypeError` yang menyebutkan opsi yang
didukung, sehingga salah ketik langsung gagal dan tidak diabaikan diam-diam.

Argumen berikut diterima, memunculkan peringatan, lalu dibuang: `agnostic_nms`,
`boxes`, `dnn`, `half`, `line_width`, `retina_masks`, `show_conf`,
`show_labels`, dan `verbose`.
