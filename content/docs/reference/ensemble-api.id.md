---
title: API Ensemble
seo_title: API LibreEnsemble dan operasi fusi
description: >-
  LibreEnsemble, ExternalDetector, dan tiga operasi fusi di libreyolo.ops: fusi
  kotak berbobot, variannya yang di-bibit, dan fusi NMS sadar kelas.
lead: >-
  LibreEnsemble menjalankan beberapa detektor pada gambar yang sama dan
  memfusikan deteksi mereka menjadi satu Results. Fusi terjadi setelah
  pemrosesan lanjutan masing-masing anggota, sehingga anggota mempertahankan
  ukuran input, normalisasi, dan penekanannya sendiri.
keywords:
  - LibreEnsemble
  - fusi kotak berbobot
  - wbf
  - ExternalDetector
  - libreyolo.ops.fusion
  - konsensus min_votes
last_verified: 1.5.0
verification: >-
  Tanda tangan dan default dibaca dari libreyolo/ensemble/model.py dan
  libreyolo/ops/fusion.py pada v1.5.0. Niat desain dari
  docs/adr/0004-model-ensembling.md.
snippets:
  usage:
    - label: 'Dua anggota, fusi default'
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ens = LibreEnsemble(["LibreYOLO9t.pt", "LibreYOLO9s.pt"])

        # Satu sumber gambar mengembalikan satu Results, bukan daftar.
        result = ens(SAMPLE_IMAGE, conf=0.25)

        print(result.boxes.xyxy)
        print(result.speed)
    - label: Konsensus dan ambang per anggota
      language: python
      code: |
        from libreyolo import LibreEnsemble, SAMPLE_IMAGE

        ens = LibreEnsemble(
            ["LibreYOLO9t.pt", "LibreYOLO9s.pt"],
            weights=[1.0, 2.0],
            fusion="wbf",
            fusion_iou=0.55,
            min_votes=2,
        )
        result = ens(SAMPLE_IMAGE, conf=[0.25, 0.4])
        print(len(result))
  ops:
    - label: 'Operasi fusi, tanpa model yang terlibat'
      language: python
      code: >
        import torch

        from libreyolo.ops import weighted_boxes_fusion


        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [12.0, 11.0, 51.0,
        49.0]])

        scores = torch.tensor([0.9, 0.8])

        labels = torch.tensor([0, 0])

        model_ids = torch.tensor([0, 1])


        fused = weighted_boxes_fusion(
            boxes, scores, labels, model_ids, num_models=2, iou_thr=0.55
        )

        print(fused)
source_hash: 3834f628efb1193d
---

## LibreEnsemble

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

| Argumen | Default | Makna |
|---|---|---|
| `members` | | Dua atau lebih detektor |
| `weights` | `None` | Faktor kepercayaan per anggota; semua `1.0` jika diabaikan |
| `fusion` | `"wbf"` | `"wbf"`, `"wbf_seeded"`, `"nms"`, atau callable |
| `fusion_iou` | `0.55` | Ambang IoU untuk klasterisasi fusi |
| `min_votes` | `1` | Simpan hanya kotak yang dikonfirmasi oleh setidaknya sebanyak anggota ini |

Seorang anggota adalah jalur bobot yang diselesaikan melalui pabrik `LibreYOLO()`, sebuah
model yang sudah dibuat, backend yang diekspor, atau `ExternalDetector`.
Setiap anggota harus merupakan model detect-task.

<code-tabs name="usage" />

Konstruksi menolak kurang dari dua anggota, daftar `weights` dari yang salah
panjang, berat yang tidak positif, `min_votes` yang bukan bilangan bulat positif,
dan `min_votes` lebih besar dari jumlah anggota. `fusion="nms"` dengan
`min_votes > 1` juga naik, karena NMS membuang keanggotaan klaster dan
tidak bisa menghitung suara.

`weights` menilai kepercayaan yang diberikan kepada setiap anggota. Bobot yang lebih tinggi menarik yang tergabung
koordinat dan skor menuju anggota itu. Konvensinya adalah membuatnya
sebanding dengan mAP validasi.

## Ruang kelas

Anggota dengan `names` yang sama langsung melewati. Jika tidak, kelas
ruang digabungkan berdasarkan nama, ID kelas anggota dipetakan ulang melalui pencarian
tabel, dan `Results.names` yang dilebur adalah gabungannya. Peleburan hanya menggabungkan kotak
dalam kelas tunggal yang sama, jadi sebuah kelas yang hanya diketahui oleh satu anggota melewati
tidak digabung. Ketidaksesuaian mencatat peringatan saat konstruksi.

`min_votes` dibatasi per kelas oleh berapa banyak ruang label anggota yang memuat itu
kelas, sehingga konsensus tetap berarti pada kosa kata yang dibagi sebagian.

## Memanggil ansambel

```python
ens(
    source=None,
    *,
    conf=0.25,
    iou=0.45,
    imgsz=None,
    device=None,
    classes=None,
    max_det=300,
    augment=False,
    save=False,
    output_path=None,
    color_format="auto",
    batch=1,
    stream=False,
    stream_buffer=False,
    vid_stride=1,
    show=False,
    **kwargs,
)
```

`predict` adalah alias untuk `__call__`. Kembalian adalah `Results` biasa, yang
`speed` merinci biaya per anggota dan menambahkan entri `fusion`. Satu
sumber gambar mengembalikan salah satunya, daftar atau direktori mengembalikan daftar, dan
`stream=True` mengembalikan generator.

`conf`, `iou` dan `device` menyiarkan ke setiap anggota dan juga menerima satu nilai
per anggota, jadi `conf=[0.25, 0.4]` memberikan anggota 0 ambang 0,25 dan
anggota 1 ambang 0,4. `imgsz` menyiarkan saat berupa int atau tuple
dan hanya per anggota saat berupa daftar, jadi `imgsz=(480, 640)` adalah satu
ukuran persegi panjang untuk semua orang sementara `imgsz=[480, 640]` adalah 480 untuk anggota 0 dan
640 untuk anggota 1. Setiap entri harus valid untuk family anggota tersebut.

`augment` menyiarkan ke anggota yang mendukung augmentasi saat pengujian, dan
backend yang diekspor mengabaikannya. `classes` mengambil gabungan ID kelas dan `max_det`
diterapkan pada hasil gabungan, sehingga anggota menjalankan secara luas dan ensemble memotong
sekali. `batch` diterima untuk kesetaraan API; gambar diproses secara berurutan.

`val()` dan `export()` menaikkan `NotImplementedError`. Validasi dan ekspor
anggota secara individu.

## DetektorEksternal

```python
ExternalDetector(fn: Callable, names: dict[int, str])
```

Menyesuaikan setiap callable deteksi menjadi anggota. `fn` menerima gambar PIL dan
mengembalikan `(boxes, scores, labels)`, di mana kotak adalah xyxy pada gambar asli
adalah ID kelas yang valid di `names`. Tensor, array, dan daftar
bersarang semuanya berfungsi. LibreYOLO tidak mengimpor apa pun dari kode eksternal.

Adapter memvalidasi hasil pengembalian: itu harus berupa tuple 3, kotak harus memiliki bentuk
`(N, 4)`, ketiga array harus memiliki panjang yang sama, dan setiap ID kelas harus
muncul di `names`. Deteksi pada atau di bawah `conf` dihapus sebelum fusi.

## Operasi fusi

Primitif fusi adalah operasi torch mandiri di `libreyolo.ops`. Mereka
bebas model dan dapat diimpor sendiri, itulah sebabnya mereka diekspor
secara terpisah dari ansambel.

<code-tabs name="ops" />

Ketiganya mengambil argumen posisi yang sama, `boxes, scores, labels,
model_ids`, and return `(kotak, skor, label)`.

| Op | Kunci registri | Perilaku |
|---|---|---|
| `weighted_boxes_fusion` | `wbf` | Penggabungan kotak berbobot secara berurutan, sesuai dengan makalah |
| `wbf_seeded` | `wbf_seeded` | Varian satu-lalu-lintas paralel dari pengurangan yang sama |
| `nms_fusion` | `nms` | Gabungkan semuanya dan terapkan NMS yang mengenal kelas |

`FUSIONS` memetakan ketiga kunci registri ke callable, dan `LibreEnsemble`
mencari `fusion=` di sana.

```python
weighted_boxes_fusion(
    boxes, scores, labels, model_ids,
    *,
    weights=None,
    num_models=None,
    iou_thr=0.55,
    skip_box_thr=0.0,
    conf_type="avg",
    min_votes=1,
    models_per_label=None,
    label_weights=None,
)
```

`wbf_seeded` mengambil tanda tangan yang identik. `nms_fusion` mengambil argumen yang sama
kecuali `conf_type`, dan menimbulkan `ValueError` ketika `min_votes > 1`.

Dalam `weighted_boxes_fusion`, deteksi dikunjungi secara berurutan dari yang menurun
kepercayaan berskala berat. Masing-masing bergabung dengan klaster yang ada yang
kotak gabungan yang berjalan itu tumpang tindih terbaik, pada IoU di atas `iou_thr` dan dengan yang sama
label, atau memulai klaster baru. Kotak gabungan dari klaster adalah
rata-rata berbobot kepercayaan dari koordinat anggotanya, dan skornya adalah
rata-rata tertimbang atau maksimum dari kepercayaan mereka, diskalakan ulang sehingga kotak
dikonfirmasi oleh lebih sedikit model mendapatkan skor lebih rendah.

`wbf_seeded` memilih benih klaster dengan NMS yang sadar kelas di `iou_thr`, menetapkan
setiap deteksi ke benih-IoU terbaiknya dengan label yang sama, kemudian mengurangi masing-masing
mengelompokkan dengan cara yang sama. Bentuk klaster tidak pernah bergeser di tengah proses, jadi seluruh operasi
matematika tensor bentuk tetap. Kedua variasi setuju kapan pun klaster adalah
tidak ambigu dan dapat berbeda sedikit pada rantai klaster yang tumpang tindih.

`nms_fusion` mempertahankan kotak dengan kepercayaan tertinggi dari setiap grup yang tumpang tindih,
tidak berubah. Skor kepercayaan per-model `weights` hanya untuk peringkat penekanan,
dan kotak yang bertahan mempertahankan skor aslinya.

## Fusi kustom

`fusion=` juga menerima callable dengan tanda tangan yang sama seperti operasi di atas.
Namanya dicatat di `ens.fusion`, atau `"custom"` ketika tidak ada. Hasilnya
divalidasi: itu harus merupakan tripel `(boxes, scores, labels)` dengan
bentuk yang konsisten.

