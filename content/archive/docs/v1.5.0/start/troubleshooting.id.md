---
title: Pemecahan masalah
seo_title: Memperbaiki error umum LibreYOLO
description: >-
  Error yang paling sering dimunculkan LibreYOLO, arti setiap error, dan
  perbaikannya. Termasuk dua kegagalan yang menghasilkan output salah tanpa
  memunculkan error.
lead: >-
  Error dikelompokkan berdasarkan pesan yang terlihat. Dua entri terakhir
  membahas masalah sebaliknya: kode berjalan, mengembalikan hasil masuk akal,
  tetapi salah.
keywords:
  - error libreyolo
  - modulenotfounderror libreyolo
  - libreyolo cuda kehabisan memori
  - libreyolo notimplementederror
  - troubleshooting libreyolo
last_verified: 1.5.0
source_hash: e271ab29b789865a
---

Error dikelompokkan berdasarkan teks yang terlihat. Jika pesan tidak tersedia di
sini, [FAQ](/docs/faq) menjawab pertanyaan yang bukan kegagalan, dan
`libreyolo models` melaporkan model yang benar-benar dapat dimuat instalasi.

## ModuleNotFoundError menyebut paket yang tidak pernah diimpor

Beberapa family memerlukan ekstra opsional. Pesan menyebut paket yang hilang,
bukan ekstra, sehingga perbaikannya tidak selalu terlihat jelas dari traceback.

Jalankan `libreyolo models`. Setiap family dengan dependency yang hilang dicetak
bersama perintah pip persis yang mengaktifkannya, sehingga paket tidak perlu
dipetakan kembali ke ekstra secara manual. `libreyolo models --json` mencetak
informasi yang sama sebagai objek.

[Halaman instalasi](/docs/install) mencantumkan setiap ekstra dan cakupannya.

## Inferensi ONNX memerlukan onnxruntime

```
ImportError: ONNX inference requires onnxruntime. Install with: pip install onnxruntime
```

Paket dasar tidak bergantung pada runtime karena pilihan runtime bergantung
pada hardware. Instal `onnxruntime` untuk CPU atau `onnxruntime-gpu` untuk CUDA.
Keduanya menyediakan modul `onnxruntime` yang sama, jadi instal salah satu,
bukan keduanya.

## Model ONNX tidak ditemukan

```
FileNotFoundError: ONNX model not found: <path>
```

Path diselesaikan relatif terhadap direktori kerja, bukan script. Pesan ini juga
muncul ketika ekspor menulis ke tempat lain tanpa disadari: `export()`
mengembalikan path yang ditulis, jadi simpan nilai kembalian alih-alih
mengasumsikan nama.

## NotImplementedError dari train()

Tidak setiap family dapat dilatih. Beberapa diadaptasi hanya untuk prediksi,
validasi, dan ekspor, serta `train()`-nya memunculkan error alih-alih berpura-
pura berjalan.

[Entri FAQ](/docs/faq) menjelaskan alasannya. Untuk memeriksa family tertentu
sebelum menulis script pelatihan, lihat dukungan pelatihan pada halaman model.

## NotImplementedError dari ekspor()

Family dapat mendukung task tetapi tidak mendukung ekspornya. EoMT sering
ditemui: `export()` menerima task semantic dan memunculkan error untuk `segment`
serta `panoptic` karena kontrak runtime query-mask yang diperlukan belum
didefinisikan.

```
NotImplementedError: LibreEoMT instance and panoptic export need query-mask runtime contracts.
```

Setiap halaman family memiliki matriks ekspor yang menunjukkan kombinasi task
dan format yang tervalidasi.

## CUDA kehabisan memori

Kurangi `batch` terlebih dahulu, lalu `imgsz`. Keduanya mengubah kebutuhan memori
secara kasar menurut ukurannya, tetapi batch dapat diturunkan tanpa mengubah
yang dilihat model.

Jika kegagalan terjadi saat validasi, bukan pelatihan, validasi menggunakan
ukuran batch sendiri, sehingga turunkan nilai tersebut juga.

Di Windows, GPU display memiliki mode kegagalan kedua yang tampak seperti error
CUDA acak, bukan kehabisan memori: driver me-reset GPU yang tidak merespons
lebih lama daripada timeout dan menghentikan proses. Kernel panjang pada kartu
yang menjalankan monitor dapat memicunya.

## Bobot tidak dapat diunduh

Bobot diambil dari Hugging Face saat penggunaan pertama dan disimpan dalam cache secara
lokal. [FAQ](/docs/faq) menjelaskan lokasi cache dan cara berjalan sepenuhnya
offline.

Jika pengunduhan menghasilkan 404, periksa nama berkas yang diberikan. URL
diturunkan dari nama tersebut, termasuk suffix task, sehingga nama yang tidak
cocok dengan checkpoint terbitan menghasilkan URL yang tidak ada. Tabel
checkpoint pada setiap halaman model mencantumkan nama berkas terbitan persis.

## Pelatihan macet atau dimulai ulang di Windows

Windows tidak memiliki `fork`, sehingga worker dataloader dimulai dengan
mengimpor ulang script. Tanpa guard `if __name__ == "__main__":`, setiap worker
menjalankan ulang pemanggilan pelatihan, yang menyebabkan deadlock atau spawn
process tanpa akhir.

```python
def main():
    ...  # bangun model dan panggil train()

if __name__ == "__main__":
    main()
```

Menetapkan `workers=0` juga menghindarinya dengan konsekuensi throughput. Guard
adalah perbaikan yang lebih baik.

## Dua kegagalan yang tidak memunculkan error

Bagian lain halaman ini membahas error. Dua kasus berikut lebih buruk karena
kode berjalan dan mengembalikan sesuatu yang terlihat benar.

### Melakukan indexing pada satu hasil

`predict()` mengembalikan satu `Results` untuk satu gambar dan list untuk
beberapa gambar. Melakukan indexing pada nilai kembalian satu gambar memilih
sebuah *deteksi*, bukan gambar:

```python
result = model.predict("image.jpg")   # sebuah Results
result.boxes                          # setiap deteksi, benar
result[0].boxes                       # SATU deteksi, diam-diam
```

Tidak ada error karena melakukan indexing pada `Results` adalah operasi valid
yang mengembalikan subset. Kode yang ditulis untuk bentuk list diam-diam
melaporkan satu bounding box per gambar. Hanya lakukan indexing pada objek yang diketahui
berupa list.

### Membaca metrik sebagai atribut

`val()` mengembalikan dictionary biasa dengan kunci nama metrik, bukan objek
dengan akses atribut:

```python
metrics = model.val(data="coco8.yaml")
metrics["metrics/mAP50-95"]   # benar
metrics.box.map               # AttributeError
```

Kunci menggunakan namespace `metrics/` dan `speed/`. Cetak dictionary satu kali
untuk melihat output task karena kumpulannya berbeda per task.

## Memeriksa dataset sebelum pelatihan

Sebagian besar kegagalan pelatihan adalah masalah dataset.
`libreyolo doctor data.yaml` menjalankan health check pada dataset deteksi dan
melaporkan temuan berdasarkan tingkat keparahan, lebih cepat daripada membaca
traceback dari epoch pertama.

```python
from libreyolo import doctor

report = doctor.diagnose("data.yaml", imgsz=640)
if report.errors:
    ...
```

Lihat [perintah doctor](/docs/cli/doctor) untuk katalog pemeriksaan.


