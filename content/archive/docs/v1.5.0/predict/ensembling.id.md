---
title: Ensemble detektor
seo_title: Ensemble detektor di LibreYOLO
description: >-
  Jalankan beberapa detektor pada satu gambar dan gabungkan kotaknya dengan
  weighted boxes fusion atau NMS, termasuk model dengan daftar kelas berbeda.
lead: >-
  LibreEnsemble menjalankan dua detektor atau lebih pada gambar terdekode yang
  sama dan menggabungkan kotaknya menjadi satu objek Results. Setiap anggota
  mempertahankan bobot, ambang batas, perangkat, dan daftar kelasnya sendiri.
keywords:
  - ensemble model deteksi objek
  - weighted boxes fusion
  - wbf python
  - gabungkan dua detektor
  - gabungkan bounding box
  - LibreEnsemble
  - ensemble detection python
  - min_votes
last_verified: 1.5.0
verification: >-
  Signature konstruktor dan pemanggilan, nilai default, error validasi,
  penyatuan ruang kelas, penghitungan suara, dan Results yang dikembalikan
  dibaca dari libreyolo/ensemble/model.py. Algoritme fusion dan argumennya
  berasal dari libreyolo/ops/fusion.py. Maksud desain berasal dari
  docs/adr/0004-model-ensembling.md. Pola penggunaan diperiksa silang terhadap
  tests/unit/test_ensemble.py dan tests/unit/test_ops_fusion.py.
snippets:
  basic:
    - label: 'Dua detektor, digabungkan'
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        # Anggota dapat berupa path checkpoint atau model yang sudah dimuat.
        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])

        result = ensemble(SAMPLE_IMAGE)
        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            print(result.names[int(cls)], round(float(conf), 3), xyxy)
    - label: Bobot dan persyaratan suara
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ensemble = LibreEnsemble(
            ["LibreYOLO9s.pt", "LibreRFDETRs.pt"],
            weights=[1.0, 1.3],   # sesuai konvensi, proporsional terhadap mAP validasi
            fusion="wbf",
            fusion_iou=0.55,
            min_votes=2,          # pertahankan hanya kotak yang ditemukan kedua anggota
        )

        result = ensemble(SAMPLE_IMAGE)
        print(len(result.boxes), "agreed detections")
    - label: Ambang batas per anggota
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])

        # Skalar berlaku untuk setiap anggota; daftar dibaca per anggota.
        result = ensemble(SAMPLE_IMAGE, conf=[0.3, 0.5], iou=0.5)
        print(len(result.boxes))
  external:
    - label: Menambahkan detektor yang tidak dimuat LibreYOLO
      language: python
      code: |
        from libreyolo import ExternalDetector, LibreEnsemble, SAMPLE_IMAGE

        def my_detector(pil_image):
            # Kembalikan (boxes, scores, labels): xyxy dalam piksel gambar asli.
            return ([[100.0, 100.0, 200.0, 300.0]], [0.9], [0])

        external = ExternalDetector(my_detector, names={0: "person"})

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", external])
        result = ensemble(SAMPLE_IMAGE)
        print(len(result.boxes))
  sources:
    - label: Sumber yang sama seperti pada satu model
      language: python
      code: |
        from libreyolo import LibreEnsemble

        ensemble = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])

        # Ganti clip.mp4 dengan berkas video di disk.
        for result in ensemble("clip.mp4", stream=True, vid_stride=2):
            print(result.frame_idx, len(result.boxes))
source_hash: 6dcd2f84ec6f3f65
---

## Apa itu ensemble

`LibreEnsemble` menerima dua detektor atau lebih, menjalankan masing-masing pada gambar yang
sama, lalu menggabungkan kotaknya menjadi satu `Results`. Ini adalah konstruksi saat prediksi:
tidak ada yang dilatih, dan setiap anggota tetap menjadi model independen yang dapat divalidasi
serta diekspor sendiri.

Deteksi adalah satu-satunya task yang didukung. Anggota dengan task lain akan memunculkan
`ValueError` saat konstruksi, disertai indeks anggota dan task-nya.

Kedua nama diimpor secara lazy, sehingga tidak memerlukan biaya sebelum digunakan:

```python
from libreyolo import LibreEnsemble, ExternalDetector
```

## Membuat ensemble

<code-tabs name="basic" />

```python
LibreEnsemble(
    members,
    *,
    weights=None,
    fusion="wbf",
    fusion_iou=0.55,
    min_votes=1,
)
```

`members` adalah urutan berisi dua anggota atau lebih. Entri `str` atau `Path` dimuat
melalui `LibreYOLO()`; entri lain harus callable dan mengekspos dict `names`. Kurang dari
dua akan memunculkan `ValueError`, sedangkan meneruskan string tunggal memunculkan `TypeError`
alih-alih mengiterasi karakternya.

`weights` memakai default `None`, yang berarti pembobotan seragam. Bobot yang diberikan harus
berjumlah satu per anggota dan semuanya positif, sehingga bobot nol memunculkan error, bukan
membuang anggota secara diam-diam. Konvensi yang didokumentasikan adalah menetapkannya secara
proporsional terhadap mAP validasi setiap anggota.

`fusion_iou` memakai default `0.55` dan menjadi IoU saat kotak dari anggota berbeda
dikelompokkan. Ini berbeda dari `iou` per pemanggilan, yang merupakan pengaturan NMS setiap
anggota sendiri.

`min_votes` memakai default `1`, yang berarti satu anggota dapat mempertahankan kotak.
Menaikkannya hanya mempertahankan cluster yang dikonfirmasi oleh sebanyak itu anggota berbeda.
Nilainya harus berupa bilangan bulat positif yang tidak melebihi jumlah anggota, dan dibatasi
per kelas ke jumlah anggota yang benar-benar mengenal kelas tersebut, sehingga kelas yang
hanya dilatih pada satu anggota tidak dihapus secara diam-diam.

## Metode fusion

Tiga nama diterima, dan callable juga dapat diberikan.

| `fusion` | Perilaku |
|---|---|
| `"wbf"` | Weighted boxes fusion, berurutan dan sesuai makalah [1]. Ini default |
| `"wbf_seeded"` | Weighted boxes fusion satu tahap; NMS yang sadar kelas memilih seed cluster |
| `"nms"` | Gabungkan kotak semua anggota, lalu jalankan NMS yang sadar kelas |

[1] Roman Solovyev, Weimin Wang, Tatiana Gabruseva, ["Weighted boxes fusion:
Ensembling boxes from different object detection models"](https://arxiv.org/abs/1910.13302),
arXiv:1910.13302.

Weighted boxes fusion merata-ratakan koordinat cluster dengan pembobotan confidence, sehingga
menghasilkan kotak yang tidak diusulkan oleh satu anggota pun. Kedua varian berbobot memberikan
hasil sama ketika cluster tidak ambigu dan dapat sedikit berbeda pada rantai cluster yang
tumpang tindih. `"nms"` memilih kotak yang bertahan alih-alih merata-ratakannya, sehingga
kotak tersebut mempertahankan skor asli, sedangkan bobot hanya memengaruhi kotak mana yang
menang. Karena metode ini memilih alih-alih membuat cluster, metode tersebut tidak dapat
menghitung suara: menggabungkan `fusion="nms"` dengan `min_votes` lebih dari `1` akan
memunculkan `ValueError`.

Weighted boxes fusion menskalakan ulang skor cluster berdasarkan bagian bobot anggota yang
mendukungnya. Dengan dua anggota berbobot sama, kotak yang hanya ditemukan salah satunya
mempertahankan separuh skor: `0.9` menjadi `0.45`. Karena itu, confidence hasil fusion dapat
turun di bawah `conf` yang dipakai untuk menjalankan setiap anggota. Filter berdasarkan skor
hasil fusion, jangan menganggap ambang anggota tetap berlaku.

## Anggota dengan daftar kelas berbeda

Anggota tidak harus memiliki daftar kelas yang sama. Ruang labelnya disatukan berdasarkan nama,
dan setiap anggota mendapat lookup table untuk memetakan ulang id kelasnya sendiri ke hasil
gabungan. `ensemble.names` adalah gabungan tersebut, dan itulah yang dibawa `Results` hasilnya.

Kotak hanya digabungkan dalam nama kelas yang sama. Kelas yang hanya dikenal satu anggota
diteruskan tanpa fusion dan tidak diberi penalti: penskalaan ulang skor memakai denominator
per kelas, sehingga kelas yang dikenal sendiri mempertahankan skornya.

Tumpang tindih parsial mencatat peringatan yang menyebut kelas yang tidak dimiliki bersama oleh
semua anggota. Peringatan tersebut harus dibaca dengan saksama karena checkpoint yang nama
kelasnya berupa placeholder seperti `class_0` membuat gabungan yang terpisah dari setiap
anggota lain, sehingga tidak ada fusion lintas anggota sama sekali.

Anggota yang mengembalikan id kelas di luar `names` miliknya akan memunculkan `RuntimeError`.

## Detektor eksternal

<code-tabs name="external" />

`ExternalDetector(fn, names)` membungkus callable apa pun yang menerima gambar PIL dan
mengembalikan `(boxes, scores, labels)`, dengan kotak sebagai xyxy dalam piksel gambar asli.
Wrapper memvalidasi arity, bentuk kotak, kesesuaian panjang, dan bahwa setiap id kelas ada
dalam `names`, lalu menerapkan sendiri ambang `conf`.

Dengan cara ini, detektor yang tidak dimuat LibreYOLO dapat ikut dalam fusion.

## Memanggil ensemble

<code-tabs name="sources" />

Signature pemanggilan mencerminkan satu model dan menerima sumber yang sama: gambar, folder,
daftar, video, screen capture, webcam, dan stream jaringan. Sumber live memerlukan
`stream=True` karena alasan yang sama seperti di tempat lain.

| Argumen | Default | Catatan |
|---|---|---|
| `conf` | `0.25` | Per anggota; skalar disebarkan, atau satu per anggota |
| `iou` | `0.45` | Ambang NMS setiap anggota sendiri, bukan ambang fusion |
| `imgsz` | `None` | Sebuah `list` dibaca per anggota; `int` atau tuple disebarkan |
| `device` | `None` | Skalar atau satu per anggota, sehingga anggota dapat berada pada perangkat berbeda |
| `classes` | `None` | Memfilter hasil fusion pada id kelas gabungan |
| `max_det` | `300` | Berlaku pada hasil fusion |

Karena `list` berarti per anggota untuk `imgsz`, `imgsz=[480, 640]` berarti 480 untuk
anggota pertama dan 640 untuk anggota kedua, sedangkan `imgsz=(480, 640)` adalah satu ukuran
persegi panjang untuk semuanya. Perbedaan ini mudah terlewat.

Anggota dipanggil dengan `max_det` setidaknya 300, apa pun nilai yang diminta, sehingga
masing-masing berjalan dengan batas longgar dan ensemble memangkasnya sekali di akhir.

Gambar didekode satu kali dan objek yang sama diberikan kepada setiap anggota. `batch`
diterima demi kesetaraan dan diabaikan; gambar diproses secara berurutan.

## Hasil yang dikembalikan

Sebuah `Results` biasa, tipe yang sama seperti hasil satu model, dengan `names` ditetapkan ke
ruang kelas gabungan. Semua yang dijelaskan dalam [Bekerja dengan hasil](/docs/predict/results)
berlaku tanpa perubahan.

Satu perbedaannya adalah `result.speed`, yang memang diisi oleh ensemble. Key-nya adalah
`member_0`, `member_1`, dan seterusnya, ditambah `fusion`, dalam milidetik. Ini satu-satunya
tempat dalam library tempat `speed` diisi.

Baris dengan kotak atau skor non-finite dibuang sebelum fusion. Jika anggota berada pada
perangkat berbeda, fusion berjalan pada perangkat anggota pertama yang mengembalikan sesuatu.

## Hal yang tidak dapat dilakukan ensemble

`val()` dan `export()` sama-sama memunculkan `NotImplementedError` dan mengarahkan ke setiap
anggota: validasi dan ekspor masing-masing secara terpisah. Tidak ada metode `train`, sehingga
memanggilnya akan memunculkan `AttributeError`.

Half precision tidak ditangani pada tingkat ensemble. `half=True` memasuki jalur no-op dengan
peringatan yang sama seperti di tempat lain; konfigurasikan presisi pada setiap anggota.

Tidak ada antarmuka command line untuk ensemble. Gunakan API Python.
