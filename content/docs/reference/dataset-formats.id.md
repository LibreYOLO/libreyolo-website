---
title: Format dataset
seo_title: Format dataset LibreYOLO untuk setiap task
description: >-
  Kontrak berkas dataset per task kanonis: kunci YAML, tata letak folder, baris label,
  konvensi mask dan map, serta loader yang membacanya.
lead: >-
  Halaman ini mencerminkan kontrak berkas dataset dalam docs/dataset_schema.md
  milik library. Cakupannya meliputi kunci YAML dan tata letak di disk yang diharapkan
  setiap task kanonis.
keywords:
  - format dataset libreyolo
  - format label yolo
  - data.yaml
  - dataset segmentation mask
  - format coco panoptic
  - dataset depth
  - pose kpt_shape
last_verified: 1.5.0
verification: >-
  Mencerminkan docs/dataset_schema.md dalam repositori libreyolo pada v1.5.0,
  dengan nama loader diperiksa silang terhadap libreyolo/data/.
snippets:
  usage:
    - label: Parse satu baris label deteksi
      language: python
      code: >
        from libreyolo.data import parse_yolo_label_line


        # class_id cx cy w h, dinormalisasi ke [0, 1]

        row = parse_yolo_label_line("0 0.5 0.5 0.25 0.5", 640, 480,
        num_classes=80)


        # (class_id, x1, y1, x2, y2, area) dalam piksel

        print(row)
source_hash: a8282c079624044d
---

## YAML umum

Berlaku untuk `detect`, `segment`, `pose`, dan `obb`.

| Kunci | Wajib | Arti |
|---|---|---|
| `path` | | Root dataset |
| `train` | Untuk pelatihan | Gambar pelatihan |
| `val` | Untuk validasi | Gambar validasi |
| `test` | | Gambar pengujian |
| `names` | Ya | List kelas atau mapping dengan kunci bilangan bulat |
| `nc` | | Jumlah kelas; harus cocok dengan `names` jika ada |
| `download` | | Petunjuk pengunduhan; script Python memerlukan keikutsertaan eksplisit |
| `annotations` | | Split ke berkas JSON COCO native, untuk detect, segment, dan obb |

`train`, `val`, dan `test` dapat berupa direktori gambar, berkas `.txt` berisi
list gambar, atau list dari keduanya. Path label mengikuti satu substitusi:

```text
images/.../image.jpg -> labels/.../image.txt
```

Untuk dataset JSON COCO native, `annotations` memetakan split ke berkas JSON dan
path split menentukan root gambar:

```yaml
path: dataset
train: images/train
val: images/val
annotations:
  train: annotations/train.json
  val: annotations/val.json
```

Jika `names` tersedia, nama kategori JSON COCO native harus cocok dengan nama
kelas YAML, dan nama tersebut menentukan id label model. Tanpa `names`, id
kategori COCO diurutkan dan dipetakan secara padat ke `0..N-1`.

YAML dataset tidak memiliki kunci `task`. Pemilihan model dan task eksplisit
selalu menang.

Aturan yang berlaku untuk setiap berkas label teks:

- satu berkas label `.txt` per gambar;
- berkas label yang hilang atau kosong berarti tidak ada objek;
- `class_id` adalah bilangan bulat dalam `0..nc-1`;
- koordinat berupa float ternormalisasi terbatas dalam `[0, 1]`;
- koordinat relatif terhadap lebar dan tinggi gambar asli;
- baris tidak memiliki confidence maupun id track.

<code-tabs name="usage" />

## detect

Tepat lima kolom per baris:

```text
<class_id> <cx> <cy> <w> <h>
```

`cx cy w h` adalah bounding box sejajar sumbu yang dinormalisasi, sedangkan `w` dan `h`
harus positif.

## segment

Baris poligon:

```text
<class_id> <x1> <y1> ... <xN> <yN>
```

`N` setidaknya 3, jumlah koordinat setelah `class_id` harus genap, dan poligon
tidak boleh degenerat. Baris deteksi lima kolom juga diterima dan mewakili
segmen rectangle.

## pose

YAML menambahkan `kpt_shape`, yang wajib berupa `[K, 2]` atau `[K, 3]`, serta
`flip_idx` opsional, yaitu permutasi bilangan bulat `0..K-1`.

```text
<class_id> <cx> <cy> <w> <h> <k1x> <k1y> [<k1v>] ... <kKx> <kKy> [<kKv>]
```

Jumlah kolom tepat `5 + K * D`, dengan `D` adalah nilai kedua `kpt_shape`.
Koordinat keypoint dinormalisasi. Visibility `v`, jika ada, bernilai `0`, `1`,
atau `2`.

## obb

Tepat sembilan kolom:

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

Keempat titik merupakan koordinat gambar ternormalisasi dalam `[0, 1]` dan
membentuk oriented rectangle yang tidak degenerat. Tidak ada sudut yang
disimpan dalam berkas label.

Parser kanonis bersifat ketat secara default dan menolak koordinat di luar
rentang. Proses ingestion dataset dan validasi dapat membatasi koordinat ke
`[0, 1]` untuk label batas crop yang sebaliknya valid, lalu tetap menolak bounding box
degenerat. Parsing memperhitungkan task: sembilan kolom berarti `obb` hanya
dalam mode `obb`, sedangkan dalam mode `segment` dapat berupa poligon empat
titik.

Secara internal, sudut ternormalisasi dikonversi menjadi `xywhr` kanonis, dengan
sudut dalam radian yang mewakili rotasi sisi lebar di sekitar pusat bounding box. Hasil
publik menyediakan deteksi OBB sebagai baris `xywhr, conf, cls`.

Pemuatan OBB JSON COCO native menerima anotasi dalam urutan prioritas berikut:
`obb` sebagai delapan sudut dalam ruang piksel; `obb` sebagai
`[cx, cy, w, h, angle]` dengan sudut dalam radian; poligon atau RLE
`segmentation` COCO yang dicocokkan ulang menjadi rectangle area minimum; dan
`bbox` COCO yang dibaca sebagai sejajar sumbu lalu dikanonisasi.

Mosaic dan mixup dinonaktifkan untuk pelatihan OBB hingga tersedia augmentasi
OBB yang memperhitungkan sudut.

Parser baris kanonis adalah `libreyolo.data.parse_yolo_obb_label_line`.

## semantic

Setiap gambar dipasangkan dengan mask satu channel padat dalam format lossless,
biasanya PNG, bukan berkas `.txt`:

```text
images/.../image.jpg -> <masks_dir>/.../image.png
```

Mask memiliki satu channel, dan PNG mode palette dibaca sebagai indeks palette.
Setiap nilai piksel merupakan id kelas dalam `0..nc-1`, nilai piksel `255`
berarti ignore dan tidak disertakan dalam loss maupun metrik, serta resolusi mask
harus sama dengan resolusi gambar.

Dua kunci YAML opsional ditambahkan di atas kontrak umum. `masks_dir` adalah nama
direktori mask yang menggantikan `images` dalam setiap path gambar, dengan
default `masks`. `label_mapping` adalah remap `{source_id: train_id}` yang
diterapkan ke nilai piksel mask saat pemuatan. Nilai sumber yang tidak dipetakan
menjadi ignore dan id pelatihan harus berada dalam `0..nc-1`.

Ketika `masks_dir` tidak diberikan, mask dirasterisasi saat pemuatan dari label
poligon `segment` yang diselesaikan melalui konvensi `images` ke `labels`, lalu
kelas `background` ditambahkan setelah kelas objek, sehingga `nc` bertambah satu.

Loader kanonis: `libreyolo.data.SemanticDataset`.

## panoptic

LibreYOLO menggunakan format COCO-panoptic apa adanya (Kirillov et al., CVPR
2019). Tidak ada format panoptic khusus LibreYOLO.

Satu PNG RGB per gambar, pada resolusi gambar, mengenkode id segmen setiap piksel
dalam warnanya:

```text
segment_id = R + 256 * G + 256 * 256 * B
```

Setiap piksel termasuk tepat dalam satu segmen dan segmen tidak pernah tumpang
tindih. Id segmen `0`, RGB hitam, adalah void: piksel tanpa label yang tidak
disertakan dalam metrik.

```json
{
  "images":      [{"id": 139, "file_name": "000000000139.jpg"}],
  "annotations": [{"image_id": 139, "file_name": "000000000139.png",
                   "segments_info": [
                     {"id": 3226956, "category_id": 1, "area": 2840,
                      "bbox": [413, 158, 53, 138], "iscrowd": 0}]}],
  "categories":  [{"id": 1, "name": "person", "isthing": 1, "supercategory": "person"}]
}
```

`annotations[].file_name` menamai PNG id-segmen dalam `panoptic_dir`, dan
`segments_info[].id` cocok dengan nilai dalam PNG tersebut. `iscrowd` menandai
region kelompok: region ini tidak pernah menjadi false negative, dan prediksi
yang sebagian besar menutupinya tidak dianggap false positive.

Thing-versus-stuff adalah properti per kategori. `isthing` berada pada
`categories`, tidak pernah pada `segments_info`.

Nilai `category_id` COCO-panoptic adalah id mentah dataset dan biasanya tidak
kontigu. Model memprediksi `0..nc-1` kontigu, sehingga id mentah dipetakan ulang
melalui `names` YAML berdasarkan nama kategori, dengan aturan yang sama seperti
loader detect JSON COCO native. Kategori JSON yang tidak ada dalam `names`
merupakan error, bukan dihapus diam-diam, karena jika tidak akan selalu dinilai
sebagai false negative.

```yaml
path: coco
val: images/val2017
annotations:
  val: annotations/panoptic_val2017.json
panoptic_dir:
  val: annotations/panoptic_val2017
names: {0: person, 1: bicycle, 132: rug-merged}
```

`annotations` dan `panoptic_dir` menerima satu path atau mapping per split.

Validasi melaporkan Panoptic Quality, yang dihitung pada resolusi ground truth
dan dirata-ratakan pada kategori yang muncul, lalu dibagi menjadi `PQ_things`
dan `PQ_stuff`. Pencocokan bersifat unik: segmen prediksi dan ground truth dari
kategori yang sama cocok ketika IoU di atas 0.5.

Loader kanonis: `libreyolo.data.PanopticDataset`.

## depth

Setiap gambar dipasangkan dengan depth map satu channel padat:

```text
images/.../image.jpg -> <depths_dir>/.../image.png
```

Map berupa PNG atau TIF satu channel, atau berkas `.npy`, pada resolusi gambar.
Nilai adalah depth biasa dalam satuan yang konsisten di dataset. Nilai nol,
negatif, NaN, dan tak hingga menandai piksel tidak valid serta dikecualikan dari
loss dan metrik.

| Kunci | Default | Arti |
|---|---|---|
| `depths_dir` | `depths` | Direktori depth yang menggantikan `images` |
| `depth_stem_suffix` | | Suffix yang ditambahkan ke stem gambar; jika tidak ada, stem sama dan suffix `_depth` sama-sama dicoba |
| `depth_mask_suffix` | `_mask` | Suffix untuk validity mask; nilai mask sama dengan atau di bawah nol, NaN, dan tak hingga membatalkan piksel depth |
| `depth_scale` | `256.0` | Pembagi untuk depth map bertipe bilangan bulat, konvensi umum PNG 16-bit |

Map `.npy` float digunakan apa adanya dan tidak menerapkan `depth_scale`.

Loader kanonis: `libreyolo.data.DepthDataset`.

## edge

Setiap gambar RGB dipasangkan dengan map satu channel lossless dengan stem yang
sama serta validity mask opsional:

```text
images/val/scene.jpg -> edges/val/scene.png
                     -> masks/val/scene.png
```

Map berupa PNG atau TIF satu channel, bukan visualisasi RGB, pada resolusi
gambar. Map bilangan bulat dibagi nilai maksimum dtype-nya; map float harus
sudah terbatas dan berada dalam `[0, 1]`. `0` berarti bukan edge dan `1` berarti
edge. Piksel mask opsional valid jika bukan nol. Pengubahan ukuran menggunakan
interpolasi nearest-neighbor untuk target dan mask, sedangkan piksel padding
tidak valid dan tidak berkontribusi pada validasi.

| Kunci | Default | Arti |
|---|---|---|
| `edges_dir` | `edges` | Direktori edge map yang menggantikan `images` |
| `edge_stem_suffix` | | Suffix yang ditambahkan ke stem gambar |
| `edge_extension` | `.png` | Ekstensi target lossless |
| `edge_invert` | | Tetapkan true jika map sumber menyimpan edge hitam di atas putih |
| `masks_dir` | `masks` | Direktori validity mask opsional |

```yaml
path: edge-dataset
train: images/train
val: images/val
edges_dir: edges
masks_dir: masks
nc: 1
names: {0: edge}
```

Validasi menipiskan prediksi kontinu dengan non-maximum suppression gradien
empat arah dan melaporkan F-measure ODS serta OIS pada sweep ambang batas yang
dapat dikonfigurasi. Piksel prediksi dan ground truth dicocokkan one-to-one
dalam `edge_max_dist * image_diagonal`, dengan toleransi ternormalisasi default
`0.0075`.

Loader kanonis: `libreyolo.data.EdgeDataset`. Loader hanya menangani format:
tidak mengunduh atau mendistribusikan ulang data benchmark.

## normal

Setiap gambar dipasangkan dengan PNG 16-bit tiga channel dengan stem sama,
ditambah validity mask opsional dengan stem sama:

```text
images/val/room.jpg -> normals/val/room.png
                    -> masks/val/room.png
```

PNG harus tepat berupa `uint16` tiga channel yang disimpan sebagai RGB, pada
resolusi gambar. Decode dengan `n = png / 65535 * 2 - 1`, lalu normalisasi ulang
setiap vektor. Vektor hasil decode menggunakan frame kamera OpenCV, `+x` ke
kanan, `+y` ke bawah, `+z` masuk ke scene, dan menghadap kamera. Mask opsional
berupa PNG satu channel dengan nilai bukan nol berarti valid; tanpa mask, setiap
vektor hasil decode yang terbatas dan bukan nol dianggap valid. Piksel target
tidak valid dan padding diwakili secara internal oleh `(0, 0, 0)`. Resize
menginterpolasi tiga komponen secara bilinear lalu menormalisasinya ulang,
validity mask menggunakan nearest-neighbor, dan flip horizontal juga membalik
tanda komponen x.

| Kunci | Default | Arti |
|---|---|---|
| `normals_dir` | `normals` | Direktori normal map yang menggantikan `images` |
| `masks_dir` | `masks` | Direktori validity mask opsional |

Validasi melaporkan mean dan median angular error dalam derajat serta persentase
piksel valid dalam 11.25, 22.5, dan 30 derajat.

Loader kanonis: `libreyolo.data.NormalDataset`.

## restore

Setiap gambar input terdegradasi dipasangkan dengan target RGB bersih:

```text
inputs/.../image.jpg -> targets/.../image.jpg
```

Input dan target berupa berkas gambar yang kompatibel dengan RGB dan resolusinya
harus sama persis. Validasi mempertahankan resolusi native dan hanya menambahkan
padding secukupnya untuk menumpuk batch, sedangkan metrik dihitung pada canvas
gambar asli. Pelatihan menerapkan crop dan flip horizontal yang berpasangan pada
input dan target.

| Kunci | Default | Arti |
|---|---|---|
| `input_dir` | `inputs` | Direktori input terdegradasi yang digunakan dalam path split |
| `target_dir` | `targets` | Direktori target bersih yang menggantikan `input_dir` |
| `target_stem_suffix` | | Suffix yang ditambahkan ke stem input sebelum lookup target |
| `target_stem_suffixes` | | Bentuk list dari `target_stem_suffix` |
| `degradation` | | Label metadata seperti `deblur` atau `denoise` |
| `dataset` | | Label dataset atau asal |

Kolom YAML mirip kelas adalah placeholder skema: gunakan `nc: 1` dan
`names: {0: image}`. Model restore menyediakan `Results.restored`, bukan deteksi.

Loader kanonis: `libreyolo.data.RestoreDataset`.

## matte

Setiap gambar RGB dipasangkan dengan ground-truth matte satu channel dengan stem
yang sama, dengan 0 berarti background dan 255 berarti foreground:

```text
images/subject.jpg -> mattes/subject.png
```

Dua tata letak diterima. Root direktori yang memuat `images/` dan direktori matte,
yang dideteksi otomatis di antara `mattes/`, `matte/`, `gt/`, `masks/`,
`mask/`, dan `alpha/`, diberikan sebagai `data=`. Alternatifnya adalah YAML
dengan `path` ditambah `val_images` dan `val_mattes` per split, serta opsional
`train_images` dan `train_mattes`, masing-masing relatif terhadap `path` atau
absolut.

Matte berupa grayscale dan dibaca sebagai opasitas dalam `[0, 1]`, serta
diubah ukurannya ke canvas prediksi dengan interpolasi bilinear jika bentuk berbeda.
Metriknya adalah MAE dan S-measure (Fan et al., ICCV 2017) pada canvas gambar
asli, dengan S-measure sebagai fitness checkpoint terbaik.

Kolom YAML mirip kelas adalah placeholder skema: gunakan `nc: 1` dan
`names: {0: matte}`. Model matte menyediakan `Results.matte`.

Validasi hanya mendukung inferensi pada versi ini. Resolver pasangan kanonis:
`libreyolo.data.matte_dataset.resolve_matte_pairs`.

## ocr

Label berupa satu berkas JSONL per split, satu objek JSON per gambar:

```text
images/val/receipt.jpg -> labels/val.jsonl
```

```json
{"image": "receipt.jpg", "regions": [{"polygon": [[10, 12], [118, 14], [117, 40], [9, 38]], "text": "TOTAL 12.50"}]}
```

`polygon` adalah quad empat titik dalam koordinat piksel absolut, berurutan kiri
atas, kanan atas, kanan bawah, kiri bawah. Region dengan teks tidak terbaca
menggunakan `"text": "###"`, konvensi do-not-care ICDAR: region tersebut
dikecualikan dari penilaian pengenalan, dan prediksi yang tumpang tindih
dengannya diabaikan, bukan dikenai penalti dalam pencocokan deteksi.

Metriknya adalah hmean deteksi dengan pencocokan poligon one-to-one di atas IoU
0.5, F1 end-to-end yang mewajibkan IoU di atas 0.5 dan transkrip persis sama
setelah normalisasi NFKC serta penghapusan whitespace, dengan kapitalisasi
diperhitungkan, dan 1-NED pada pasangan yang cocok. Fitness checkpoint terbaik
adalah F1 end-to-end.

Dua tata letak diterima: root direktori yang memuat `images/<split>/` dan
`labels/<split>.jsonl`, diberikan sebagai `data=`, atau YAML dengan `path`
ditambah nama direktori `images` dan `labels` opsional.

Kolom YAML mirip kelas adalah placeholder skema: gunakan `nc: 1` dan
`names: {0: text}`. Model OCR menyediakan `Results.ocr`.

Validasi hanya mendukung inferensi pada versi ini. Resolver sampel kanonis:
`libreyolo.data.ocr_dataset.resolve_ocr_samples`.

## classify

Tree direktori bergaya ImageFolder, bukan berkas label:

```text
dataset_root/
  train/
    class_a/*.jpg
    class_b/*.jpg
  val/
    class_a/*.jpg
    class_b/*.jpg
```

`train/` wajib untuk pelatihan dan menentukan mapping kelas-ke-indeks
berdasarkan nama folder yang diurutkan. `val/` wajib untuk validasi. `test/`
dapat tersedia, tetapi perintah train dan val default tidak menggunakannya.
Split selain pelatihan harus memuat nama folder kelas yang sama seperti kumpulan
kelas train atau checkpoint yang diharapkan. Ekstensi gambar yang didukung
didefinisikan dalam `libreyolo.data.classify_dataset.IMAGE_EXTENSIONS`.

## gaze dan point

Tidak ada kontrak berkas dataset pelatihan atau validasi yang diimplementasikan
untuk `gaze`.

`point` adalah task output model, bukan skema label dataset. Family point dapat
mengadaptasi label yang ada secara internal, misalnya dengan menurunkan pusat
objek dari baris bounding box, tetapi format label teks khusus point tidak didefinisikan.


