---
title: Datasets
seo_title: Dataset pelatihan di LibreYOLO
description: >-
  YAML dataset LibreYOLO membaca, tata letak folder yang diharapkannya,
  bagaimana autodownload bekerja, dan perintah doctor yang memeriksa dataset
  sebelum pelatihan.
lead: >-
  LibreYOLO dataset adalah file YAML yang menamai root, pembagiannya, dan nama
  kelasnya. Semua hal lain, termasuk tempat file label berada, diturunkan dari
  file itu berdasarkan konvensi.
keywords:
  - format yolo dataset
  - data.yaml
  - pelatihan dataset kustom
  - format label yolo
  - json coco dataset
  - dataset autodownload
  - libreyolo doctor
  - cek ketidakseimbangan kelas
  - kebocoran pembagian train val
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # Nama bundel, jalur relatif, atau jalur absolut semuanya bisa
        digunakan.

        model.train(data="coco8.yaml", epochs=10)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=coco8.yaml epochs=10
  doctor:
    - label: Periksa dataset
      language: bash
      code: |
        libreyolo doctor my-dataset.yaml
    - label: Gagal pada pekerjaan CI juga karena peringatan
      language: bash
      code: |
        libreyolo doctor my-dataset.yaml strict=true json=true
    - label: Lewati proses dekode gambar
      language: bash
      code: |
        # Membaca label dan YAML saja. Korupsi, duplikat, dan kebocoran terbagi
        # Semua pemeriksaan membutuhkan piksel, jadi dilewati.
        libreyolo doctor my-dataset.yaml fast=true
    - label: Python
      language: python
      code: |
        from libreyolo import doctor

        report = doctor.diagnose("my-dataset.yaml", imgsz=640)

        for finding in report.findings:
            print(finding.severity.value, finding.check_id, finding.message)

        raise SystemExit(report.exit_code(strict=False))
source_hash: 9a12a0551c8b56e9
---

## Arahkan pelatihan ke dataset

`data=` mengambil jalur YAML atau nama konfigurasi yang disertakan dengan paket.

<code-tabs name="train" />

Nama diselesaikan dalam urutan tetap: jalur absolut yang ada, kemudian
nama seperti yang diberikan relatif terhadap direktori kerja, kemudian nama yang sama dengan
`.yaml` ditambahkan, kemudian direktori konfigurasi yang dibundel. Ketika tidak ada yang cocok,
kesalahan menyebutkan setiap direktori yang dicari dan mencantumkan konfigurasi yang dibundel.

## Konfigurasi yang dibundel

Tiga belas konfigurasi dataset dikirim di dalam paket, di bawah
`libreyolo/config/datasets/`.

| Konfigurasi | Task | Catatan |
|---|---|---|
| `coco8.yaml` | deteksi | 8 gambar, unduh dari URL biasa |
| `coco128.yaml` | deteksi | 128 gambar |
| `coco1000.yaml` | deteksi | 800 latihan, 200 validasi |
| `coco5000.yaml` | deteksi | 4000 latihan, 1000 validasi |
| `coco.yaml` | deteksi | full COCO 2017 |
| `coco-val-only.yaml` | deteksi | hanya val2017 |
| `coco8-pose.yaml` | pose | 8 gambar, keypoints COCO-17 |
| `coco-pose.yaml` | pose | keypoints COCO 2017 |
| `ade20k.yaml` | semantik | 150 kelas |
| `cityscapes.yaml` | semantik | 19 kelas, unduh secara manual |
| `cocostuff.yaml` | semantik | 182 kelas, unduh secara manual |
| `gopro.yaml` | pulihkan | pasangan pemburaman |
| `sr8.yaml` | pulihkan | pasangan super-resolusi |

Hanya `coco8.yaml` dan `coco128.yaml` yang memiliki URL unduhan langsung. Sisanya
menggunakan blok unduhan Python, yang memerlukan persetujuan seperti dijelaskan di bawah, atau mengharapkan
data sudah ada di disk.

## Lokasi dataset di disk

Kunci YAML `path` menamai root dataset. `path` absolut digunakan sebagaimana
tertulis. Yang relatif dicari terlebih dahulu di bawah direktori datasets, kemudian
di samping file YAML itu sendiri, dan dataset yang akan diunduh ditempatkan
di direktori datasets.

Direktori itu adalah `~/datasets`, digantikan oleh `LIBREYOLO_DATASETS_DIR`
. Tidak ada file pengaturan untuk itu.

## Kunci YAML

```yaml
path: my-dataset        # dataset root
train: images/train     # diperlukan untuk melatih
val: images/val         # diperlukan untuk memvalidasi
test: images/test       # opsional
nc: 3                   # opsional; harus sesuai dengan nama
names:
  0: person
  1: helmet
  2: vest
download: https://example.com/my-dataset.zip   # opsional
```

`train`, `val` dan `test` masing-masing menerima direktori gambar, file `.txt` yang mencantumkan
satu jalur gambar per baris, atau daftar yang mencampur keduanya. Baris dalam daftar `.txt` dapat
bersifat relatif, dalam hal ini akan diselesaikan terhadap direktori file daftar itu sendiri, dan
baris yang dimulai dengan `#` dilewati.

`names` dapat berupa daftar atau pemetaan dengan kunci integer. `nc` bersifat opsional; ketika keduanya
hadir dan tidak setuju, dokter melaporkannya sebagai kesalahan.

## Tata letak direktori dan file label

Deteksi, segmentasi, pose, dan kotak berorientasi semuanya berbagi satu tata letak. Label
path diturunkan dari path gambar dengan menulis ulang komponen direktori `images`
menjadi `labels` dan mengubah ekstensi menjadi `.txt`:

```text
my-dataset/
  images/train/0001.jpg   ->   labels/train/0001.txt
  images/val/0002.jpg     ->   labels/val/0002.txt
```

Hanya seluruh komponen path `images` yang ditulis ulang, sehingga direktori bernama
`images_old` dibiarkan begitu saja.

Satu baris deteksi memiliki lima bidang, semua dinormalisasi terhadap `[0, 1]` terhadap lebar
dan tinggi gambar asli:

```text
<class_id> <cx> <cy> <w> <h>
```

File label yang hilang atau kosong berarti gambar tidak memiliki objek, dan itu dilatih sebagai
daripada menaikkan. Baris dengan lebih dari lima kolom dibaca sebagai
poligon dan kotaknya menjadi jangkauan poligon, sehingga ekspor segmentasi yang digunakan
untuk pelatihan deteksi dimuat tanpa keluhan. Dokter melaporkan berapa banyak baris
yang mengambil jalur itu.

## Tugas lainnya

Segmentasi mempertahankan tata letak yang sama dengan baris poligon,
`<class_id> <x1> <y1> ... <xN> <yN>`, setidaknya tiga titik. Sebuah baris
deteksi lima-kolom diterima dan berarti sebuah instance persegi panjang.

Pose menambahkan `kpt_shape: [K, D]` dan opsi `flip_idx` permutasi ke YAML.
Setiap baris tepat `5 + K * D` kolom: kotak, kemudian `K` titik kunci `x y` atau
`x y v`, dengan visibilitas `0`, `1`, atau `2`.

Kotak berorientasi menggunakan tepat sembilan bidang, kelas diikuti oleh empat titik sudut
dalam koordinat ternormalisasi. Tidak ada sudut yang disimpan di dalam berkas.

Segmentasi semantik memasangkan setiap gambar dengan topeng satu saluran dengan resolusi yang sama,
diselesaikan dengan menggantikan `masks_dir` (default `masks`) untuk `images`.
Nilai piksel `255` berarti diabaikan. `label_mapping` memetakan ulang ID sumber ke ID pelatihan pada
saat dimuat.

Klasifikasi menggunakan pohon ImageFolder alih-alih berkas label, dengan `train/` dan
`val/` masing-masing memiliki satu direktori per kelas. Pemetaan kelas-ke-indeks adalah urutan
nama folder yang diurutkan.

Pemulihan QZL menggabungkan input yang rusak dengan target bersih dengan resolusi yang identik
melalui `input_dir` dan `target_dir`. Kedalaman, normal permukaan, dan tepi masing-masing menggabungkan
sebuah gambar dengan peta padat melalui kunci direktori mereka sendiri.

Kontrak penuh per-task, termasuk konvensi skala kedalaman dan
pengkodean PNG ID segmen panoptik, ada di `docs/dataset_schema.md` dalam repositori
perpustakaan.

## JSON COCO Asli

File anotasi JSON COCO dapat digunakan langsung. Tambahkan `annotations` pemetaan,
dan jalur split menjadi root gambar:

```yaml
path: my-dataset
train: images/train
val: images/val
annotations:
  train: annotations/train.json
  val: annotations/val.json
```

Ketika `names` hadir, nama kategori JSON harus sesuai dengannya, dan `names`
mendefinisikan id label yang diprediksi model. Tanpa `names`, id kategori COCO adalah
disortir dan dipetakan secara padat ke `0..N-1`.

Jalur ini mengharapkan satu direktori gambar per pembagian. Daftar jalur atau `.txt`
daftar gambar meningkatkan alih-alih diam-diam memuat set yang berbeda.

## Unduh Otomatis

Sebuah dataset dihitung sebagai hadir ketika jalur `train` atau `val`-nya terselesaikan ke
direktori yang tidak kosong atau file yang sudah ada. Ketika tidak ada, dan YAML memiliki
Kunci `download`, nilainya menentukan apa yang terjadi selanjutnya.

Sebuah URL `http` atau `https` diambil dan, jika itu adalah file zip, diekstrak ke dalam
dataset root. Apa pun selain itu dianggap sebagai skrip Python yang terlampir dan hanya dijalankan
ketika `allow_download_scripts=True`. Tanpa itu, skrip dilewati dengan sebuah
peringatan dan pelatihan terus dilakukan terhadap apapun yang ada di disk.

```bash
libreyolo train model=LibreYOLO9s.pt data=coco.yaml allow_download_scripts=true
```

flag adalah gerbang eksekusi kode, bukan gerbang jaringan. Unduhan URL terjadi
bagaimanapun; blok `download: |` yang membutuhkannya. CLI mencetak peringatan
ketika flag menyala, dan dokter tidak pernah mengaktifkannya.

## Periksa dataset sebelum Anda berlatih

`libreyolo doctor` membaca deteksi dataset dan melaporkan apa yang akan salah
sebelum GPU terlibat. Ia keluar dengan kode 1 ketika menemukan kesalahan, sehingga berfungsi sebagai CI
gerbang.

<code-tabs name="doctor" />

Cek datang dalam enam keluarga:

| Family | Mencari |
|---|---|
| `config` | `names`, `nc` yang hilang yang tidak sesuai dengan `names`, split yang hilang atau kosong, nama kelas duplikat |
| `files` | gambar tanpa file label, label tanpa gambar, gambar yang hilang yang tercantum dalam sebuah split, tabrakan stem |
| `labels` | baris yang rusak, ID kelas di luar `[0, nc)`, koordinat di luar `[0, 1]`, kotak dengan area nol, kotak sangat kecil atau sangat besar, kotak duplikat, file label identik byte |
| `balance` | kelas dengan nol atau sedikit instance, rasio ketidakseimbangan kelas, kelas yang hanya ada dalam satu split, bagian gambar latar belakang |
| `images` | file yang tidak dapat didekode, rotasi EXIF, tata letak saluran aneh, gambar seragam, duplikat tepat dan hampir sama |
| `splits` | gambar yang sama muncul di dua bagian, tepat atau hampir identik |

`--only` dan `--skip` mengambil id pemeriksaan atau prefiks family, jadi
`skip=images,labels.tiny_object` adalah valid. `--fast` menghapus setiap pemeriksaan yang membutuhkan
untuk mendekode piksel, yang merupakan keluarga `images` dan `splits`.

Dua perilaku yang perlu diketahui. `--strict` membuat peringatan gagal pada kode keluar sekaligus
kesalahan. Dan dokter hanya mencakup dataset deteksi: pose, segmen atau
kotak berorientasi dataset ditolak dengan pesan yang menyebutkan apa yang terdeteksi, bukan
diperiksa terhadap kontrak yang salah.

## Terkait

- [Hyperparameters](/docs/train/hyperparameters) untuk argumen `train()`
  dilakukan sekali setelah data tersedia.
- [Validasi dan metrik ](/docs/train/validation) untuk evaluasi pada `val`
  atau pembagian `test`.

