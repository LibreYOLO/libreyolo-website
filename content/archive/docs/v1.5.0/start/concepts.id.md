---
title: Konsep inti
seo_title: Konsep inti LibreYOLO
description: >-
  Hubungan antara task, family model, ukuran, dan nama berkas checkpoint di
  LibreYOLO, serta jaminan setiap tier dukungan.
lead: >-
  Empat gagasan menjelaskan setiap model di LibreYOLO: task yang dikerjakan,
  family asalnya, ukuran dalam family tersebut, dan tier dukungan family itu.
  Nama berkas checkpoint mengenkode tiga hal pertama.
keywords:
  - konsep libreyolo
  - task libreyolo
  - family model libreyolo
  - format nama checkpoint libreyolo
  - tier dukungan libreyolo
last_verified: 1.5.0
meta:
  - label: Skema nama berkas
    value: 'Libre<FAMILY><size>[-<task>].pt'
    mono: true
  - label: Task kanonis
    value: 17
  - label: Tier dukungan
    value: 'Unggulan, Inti, Didukung, Hanya inferensi, Museum, Tingkat saudara'
snippets:
  inspect:
    - label: Cantumkan family
      language: bash
      code: |
        # Task, ukuran, dan resolusi input untuk setiap family terdaftar.
        libreyolo models
    - label: Satu model
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        print(model.family, model.size, model.task)
        print(model.input_size)
        print(model.nb_classes, model.names[0])
    - label: Pilih task
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Alias dinormalisasi pada batas API: "keypoints" diselesaikan menjadi

        # "pose", "det" menjadi "detect", "semantic-segmentation" menjadi
        "semantic".

        model = LibreYOLO("LibreYOLO9t.pt", task="det")

        print(model.task)
source_hash: 23d045463a6a8411
---

## Task

Task adalah yang dikembalikan model. LibreYOLO memiliki tujuh belas nama task
kanonis, dan setiap nama menunjuk kolom pada objek `Results` yang membawa
output-nya.

| Task | Mengembalikan |
|---|---|
| `detect` | Bounding box sejajar sumbu dengan kelas dan confidence |
| `segment` | Mask per instance, satu mask per objek terdeteksi |
| `semantic` | Satu label kelas per piksel, tanpa pemisahan instance |
| `panoptic` | Satu label tidak tumpang tindih per piksel, menggabungkan thing yang dapat dihitung dengan stuff amorf |
| `pose` | Keypoint per instance, dengan baris yang selaras dengan bounding box |
| `classify` | Probabilitas atas kumpulan label untuk seluruh gambar |
| `obb` | Oriented bounding box dengan sudut rotasi |
| `point` | Satu koordinat gambar per deteksi, bukan bounding box |
| `depth` | Map inverse-depth relatif padat |
| `normal` | Kolom surface-normal vektor satuan padat |
| `edge` | Map probabilitas edge padat |
| `restore` | Gambar RGB hasil restorasi untuk deblurring, denoising, atau super-resolution |
| `matte` | Map foreground lunak dari 0 hingga 1 untuk penghapusan latar belakang |
| `ocr` | Quad teks beserta transkrip dalam urutan baca |
| `embed` | Vektor ternormalisasi L2 yang dot product-nya mengukur kecocokan |
| `gaze` | Arah pandang per wajah terdeteksi |
| `mesh` | Tubuh 3D berpose per orang terdeteksi |

Nama tersebut muncul dalam metadata checkpoint dan nama berkas. Alias yang umum
diterima di semua tempat yang menerima task dan dinormalisasi sebelum proses
lain: `detection` dan `det` menjadi `detect`, `keypoints` menjadi `pose`, `cls`
menjadi `classify`, `deblur`, `denoise`, dan `super-resolution` semuanya menjadi
`restore`, sedangkan `face-recognition` dan `reid` menjadi `embed`. Nama yang
tidak dikenal memunculkan error, bukan kembali ke default secara diam-diam.

`segment`, `semantic`, dan `panoptic` adalah tiga task berbeda, bukan tiga kata
untuk hal yang sama. Instance mask, label per piksel, dan map gabungan
thing-plus-stuff memiliki ground truth, metrik, dan kolom hasil berbeda.

## Family model

Family adalah satu lineage arsitektur dengan kode pemuatan, preprocessing, dan
postprocessing sendiri. Setiap family mendeklarasikan identifier `FAMILY`
seperti `yolo9`, `rfdetr`, atau `dfine`, task yang didukung, serta resolusi input
untuk setiap ukuran yang disediakan.

`LibreYOLO()` adalah factory, bukan kelas. Berdasarkan path, factory memuat berkas,
mengidentifikasi family dari metadata checkpoint atau, jika gagal, dari kunci
tensor itu sendiri, lalu mengembalikan instance model family tersebut. Karena
itu, mengganti detektor hanya memerlukan perubahan satu baris: objek yang
dihasilkan menyediakan antarmuka `predict`, `train`, `val`, dan `export` yang
sama serta mengembalikan jenis `Results` yang sama.

<code-tabs name="inspect" />

Family yang melayani lebih dari satu task biasanya menerbitkan checkpoint
terpisah per task, sering dengan kumpulan ukuran berbeda untuk masing-masing;
beberapa family memakai satu artefak bersama untuk dua task runtime. Dalam
kedua kasus, task yang didukung merupakan list tetap. Meminta task di luar list
memunculkan error yang mencantumkan task didukung, bukan memuat pendekatan
perkiraan.

List lengkap beserta benchmark per family dan bobot terbitan tersedia di
[semua model](/docs/models).

## Ukuran

Ukuran adalah varian dalam family, ditulis sebagai kode huruf kecil yang
ditempel langsung pada prefix family. Huruf yang umum adalah `n` untuk nano,
`t` untuk tiny, `s` untuk small, `m` untuk medium, `l` untuk large, dan `x`
untuk xlarge, tetapi kode bersifat khusus family dan beberapa family memakai
hal lain: kode nama backbone seperti `r50` atau `r101` ketika ukuran adalah
depth ResNet, kode compound-scaling seperti `b0` hingga `b3`, atau nama yang
mengidentifikasi satu checkpoint terbitan. YOLOv9 menggunakan `c` untuk compact
ketika family lain memakai `l`.

Ukuran juga menetapkan resolusi input, dan pada family dengan beberapa task,
resolusi dapat berbeda per task. Keduanya dibaca dari family, tidak pernah
diasumsikan; `libreyolo models` mencetaknya.

## Nama berkas checkpoint

Setiap berkas bobot terbitan mengikuti satu skema:

```text
Libre<FAMILY><size>[-<task>].pt
```

Prefix family merupakan string tetap per family, ukuran memakai huruf kecil dan
ditempel tanpa separator, sedangkan suffix task diawali tanda hubung. Deteksi
tidak memakai suffix, mengikuti konvensi checkpoint YOLO sejak awal, sehingga
`LibreYOLO9t.pt` adalah detektor dan `LibreRFDETRn-seg.pt` adalah model
segmentation dari family yang sama.

| Task | Suffix |
|---|---|
| `detect` | |
| `segment` | `-seg` |
| `semantic` | `-sem` |
| `panoptic` | `-panoptic` |
| `pose` | `-pose` |
| `classify` | `-cls` |
| `gaze` | `-gaze` |
| `obb` | `-obb` |
| `point` | `-point` |
| `depth` | `-depth` |
| `edge` | `-edge` |
| `normal` | `-normal` |
| `restore` | `-restore` |
| `matte` | `-matte` |
| `ocr` | `-ocr` |
| `embed` | `-embed` |
| `mesh` | `-mesh` |

Family tanpa task tanpa-suffix dapat mewajibkan suffix, sehingga nama tanpa
suffix bukan checkpoint valid baginya. Family yang menerbitkan bobot hasil
pelatihan pada dataset selain default menambahkan nama dataset sebagai suffix
lanjutan, dan varian tersebut tetap menjadi bagian nama repositori tempat berkas
diunduh.

Tiga tier berada di luar skema ini. Family promptable segmentation, family
vision-language, dan detektor open-vocabulary tidak terdaftar ke factory
checkpoint dan tidak menghasilkan berkas `Libre<FAMILY><size>.pt`. Prefix mereka
menamai snapshot Hugging Face yang diunduh atau checkpoint promptable, dan
kapitalisasi merek upstream sengaja dipertahankan di sana.

## Cara task ditentukan

Ketika beberapa sinyal dapat menentukan task, semuanya diperiksa dalam urutan
tetap dan sinyal pertama yang tersedia menang: argumen `task` yang diberikan,
lalu task dalam metadata checkpoint, kemudian suffix task pada nama berkas, dan
akhirnya task default family. Hasil diperiksa terhadap task yang didukung family
sebelum model dibangun, sehingga ketidakcocokan gagal saat pemuatan, bukan
menghasilkan output salah kemudian.

## Tier dukungan

Family didaftarkan tepat ke satu tier. Tier adalah pernyataan tentang perhatian
engineering, bukan akurasi: tier menjelaskan tempat fitur baru diterapkan lebih
dahulu dan hal yang dijaga tetap berfungsi.

| Tingkat | Artinya |
|---|---|
| Unggulan | Fitur dirancang dan divalidasi penuh pada GPU di sini terlebih dahulu |
| Inti | Detektor inti yang dapat dilatih. Fitur mengikuti family unggulan dalam gelombang rilis yang sama |
| Didukung | Family pendukung yang dapat dilatih. Dijaga tetap hijau dalam CI, fitur diterapkan secara oportunistis |
| Hanya inferensi | Prediksi, validasi, dan ekspor. Fitur pelatihan tidak berlaku |
| Museum | Pameran beku. Hanya perbaikan bug |
| Tingkat saudara | Antarmuka produk terpisah dengan factory dan kontraknya sendiri |

Setiap halaman model memuat tingkat family pada header. Dua family unggulan adalah
[YOLOv9](/docs/models/yolov9) untuk detektor CNN dan
[RF-DETR](/docs/models/rf-detr) untuk detektor transformer; mulailah dari sana
kecuali ada alasan untuk memilih yang lain.

Hanya inferensi menyatakan yang tidak ada, yaitu loop pelatihan dalam LibreYOLO.
Prediksi, validasi, dan ekspor jika didukung family tetap berfungsi. Memanggil
`train()` pada family tersebut memunculkan `NotImplementedError` yang menyebutkan
alasannya.


