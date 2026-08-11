---
title: Deteksi berorientasi
seo_title: Deteksi berorientasi di LibreYOLO
description: >-
  Deteksi objek yang berotasi di LibreYOLO: family yang melayani box
  berorientasi, baris label empat sudut, serta pemanggilan prediksi, pelatihan,
  validasi, dan ekspor.
lead: >-
  Deteksi objek berorientasi melokalisasi setiap instance dengan persegi panjang
  berotasi, bukan persegi panjang sejajar sumbu, sehingga objek miring dibatasi
  dengan rapat alih-alih oleh box yang penuh background. Key task-nya adalah
  obb.
keywords:
  - deteksi oriented bounding box
  - deteksi objek berotasi
  - OBB Python
  - dataset DOTA
  - deteksi objek aerial
  - rotated IoU
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        # Memerlukan extra rfdetr: pip install "libreyolo[rfdetr]"
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Akhiran -obb pada nama file memilih task, sehingga argumen task
        # tidak diperlukan.
        model = LibreYOLO("LibreRFDETRs-obb.pt")
        result = model(SAMPLE_IMAGE, save=True)

        obb = result.obb
        print(obb.xywhr)   # (N, 5): pusat x, pusat y, lebar, tinggi, radian
        print(obb.conf, obb.cls)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRFDETRs-obb.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Sudut sebagai pengganti angle
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreRFDETRs-obb.pt")(SAMPLE_IMAGE)
        obb = result.obb

        print(obb.xyxyxyxy.shape)    # (N, 4, 2) titik sudut dalam piksel
        print(obb.xyxyxyxyn.shape)   # hal yang sama, ternormalisasi
        print(obb.xyxy.shape)        # (N, 4) box sejajar sumbu yang melingkupi
    - label: Checkpoint lebih kecil
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRFDETRn-obb.pt")
        result = model(SAMPLE_IMAGE)

        print(result.obb.xywhr.shape)
    - label: RT-DETRv2
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Bobot DOTA v1.0, 15 class aerial pada 1024 px. Graph berorientasi
        # dikenali dari tensor checkpoint sendiri, jadi tanpa argumen task.
        model = LibreYOLO("LibreRTDETRv2n-obb.pt")
        result = model("aerial.png", save=True)

        obb = result.obb
        print(obb.xywhr)
        print(result.names)   # plane, ship, harbor, helicopter, dan 11 lainnya
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Melanjutkan dari bobot berorientasi terbitan. data harus menunjuk ke

        # dataset yang baris labelnya memuat empat sudut.

        model = LibreYOLO("LibreRFDETRs-obb.pt")

        model.train(data="my-obb-dataset.yaml", epochs=50, imgsz=512, batch=8,
        lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs-obb.pt data=my-obb-dataset.yaml \
          epochs=50 imgsz=512 batch=8 lr0=1e-4
    - label: Dari bobot deteksi
      language: bash
      code: |
        # Bobot deteksi tidak memuat prediksi angle, jadi ini adalah transfer
        # eksplisit. Meminta task=obb memberikan otorisasi untuknya.
        libreyolo train model=LibreRFDETRs.pt data=my-obb-dataset.yaml \
          task=obb epochs=50 imgsz=512
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs-obb.pt")

        # val() mengembalikan dict biasa, bukan objek.
        metrics = model.val(data="my-obb-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"], metrics["metrics/mAP75"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRFDETRs-obb.pt data=my-obb-dataset.yaml
    - label: RT-DETRv2
      language: bash
      code: |
        libreyolo val model=LibreRTDETRv2n-obb.pt data=my-obb-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs-obb.pt")
        model.export(format="onnx", imgsz=512)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRFDETRs-obb.pt format=onnx imgsz=512
    - label: RT-DETRv2
      language: bash
      code: >
        # ONNX dan TorchScript adalah target tervalidasi di sini, pada FP32,

        # batch 1, di kanvas tetap 1024 kali 1024.

        libreyolo export model=LibreRTDETRv2n-obb.pt format=onnx imgsz=1024

        libreyolo export model=LibreRTDETRv2n-obb.pt format=torchscript
        imgsz=1024
    - label: Gunakan file hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan akhiran file, sehingga artefak hasil

        # ekspor dimuat seperti checkpoint dan mengembalikan objek Results yang
        sama.

        model = LibreYOLO("LibreRFDETRs-obb.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.obb.xywhr)
source_hash: 0d605d956f3ea025
---

## Definisi

Deteksi berorientasi menambahkan satu angka pada deteksi: angle. Setiap instance
memperoleh persegi panjang berotasi, class, dan skor. Keuntungannya adalah batas
yang rapat. Kapal pada sudut 45 derajat, atap gudang, atau deretan truk terparkir
akan dikelilingi box sejajar sumbu yang sebagian besar berisi background, dan dua
box berdekatan saling tumpang tindih meskipun objeknya tidak. Itulah alasan task
ini umum dalam pencitraan aerial dan tata letak dokumen, serta alasan DOTA menjadi
dataset referensinya.

`obb` adalah key task kanonis, dan akhiran `-obb` pada nama file checkpoint
memilihnya, sehingga `task=` tidak diperlukan saat memuat bobot terbitan.

`predict()` mengisi `result.obb`. `.xywhr` adalah bentuk kanonis `(N, 5)`: pusat
x, pusat y, lebar, tinggi, dan angle dalam radian yang menyatakan rotasi sisi
lebar di sekitar pusat. `.conf` dan `.cls` memuat skor serta indeks class dalam
`result.names`, sedangkan `.id` memuat track id saat tracking. `.xyxyxyxy`
mengubah setiap baris menjadi empat titik sudut sebagai piksel `(N, 4, 2)`,
`.xyxyxyxyn` menormalisasi sudut tersebut, dan `.xyxy` memberikan box sejajar
sumbu yang melingkupi, untuk digunakan ketika kode downstream hanya memahami
persegi panjang. `result.boxes` juga diisi dengan bentuk sejajar sumbu.

## Model

Dua family melayani task ini, dan pilihan bergantung pada kebutuhan pelatihan.

[RF-DETR](/docs/models/rf-detr) adalah family yang dapat dilatih. Model ini
memprediksi, melatih, memvalidasi, dan mengekspor box berorientasi, serta
menyertakan checkpoint berorientasi terbitan dalam empat ukuran, n, s, m, dan l.
Model ini memerlukan extra sendiri, `pip install "libreyolo[rfdetr]"`, dan halaman
modelnya memuat lisensi serta asal-usul bobot.

Baca bagian di bawah tentang hal yang sebenarnya diprediksi checkpoint tersebut
sebelum mengandalkannya.

[RT-DETRv2](/docs/models/rt-detr) adalah family dengan bobot aerial. Model ini
menerbitkan `LibreRTDETRv2n-obb.pt` hingga `LibreRTDETRv2x-obb.pt`, yaitu
checkpoint single-scale DOTA v1.0 resmi yang dikonversi ke format LibreYOLO,
mencakup 15 class DOTA pada 1024 px. Tidak diperlukan extra selain package dasar,
graph berorientasi dikenali dari tensor checkpoint sendiri, dan prediksi,
validasi, serta ekspor ONNX dan TorchScript didukung. Pelatihan tidak didukung:
task berorientasi hanya untuk inferensi pada family ini, `train()` memunculkan
error, dan tidak ada transfer dari bobot deteksinya yang menggunakan backbone
berbeda. Tracking dan test-time augmentation juga tidak tersedia untuk box
berorientasi.

Ringkasnya: untuk kategori DOTA siap pakai, gunakan RT-DETRv2. Untuk label
berorientasi Anda sendiri, gunakan RF-DETR.

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam
cache lokal.

<code-tabs name="predict" />

Pahami checkpoint terbitan RF-DETR sebelum menjalankannya. Meskipun DOTA menjadi
benchmark referensi task ini, bobot tersebut tidak dilatih padanya. Keempatnya
diinisialisasi dari bobot deteksi RF-DETR dan di-fine-tune pada satu dataset
Roboflow Universe berisi rekaman UAV, dengan enam class kendaraan: bike, bus,
car, other_vehicle, taxi, dan truck. Model card mendeskripsikannya sebagai bobot
pengembangan yang dihasilkan saat memvalidasi dukungan pelatihan berorientasi,
serta menyatakan bahwa bobot tersebut bukan bobot production atau benchmark resmi.

Dalam praktiknya, bobot tersebut menjadi titik awal yang berfungsi untuk box
berorientasi pada kendaraan yang terlihat dari atas dan untuk memverifikasi bahwa
pipeline berjalan end-to-end. Domain lain memerlukan pelatihan pada label
berorientasi Anda sendiri, dan untuk kategori aerial yang dikenal dari DOTA,
checkpoint RT-DETRv2 adalah yang benar-benar dilatih pada data tersebut. `conf`
dan `max_det` membentuk output seperti pada deteksi. Lihat
[prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Format dataset

Tata letaknya sama dengan deteksi: satu file label `.txt` per gambar, yang
ditemukan dengan mengganti `images` menjadi `labels` pada path gambar dan
mengubah ekstensinya.

```text
dataset/
  data.yaml
  images/
    train/P0001.png
    val/P0101.png
  labels/
    train/P0001.txt
    val/P0101.txt
```

Satu baris berisi tepat sembilan field, yaitu indeks class yang diikuti empat
titik sudut secara berurutan:

```text
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

Keempat titik berupa float ternormalisasi dalam `[0, 1]` dan harus membentuk
persegi panjang berorientasi yang tidak degeneratif. Angle tidak disimpan dalam
file label: loader menurunkan `xywhr` kanonis dari sudut. Secara default, parser
bersifat ketat dan menolak koordinat di luar rentang, sedangkan proses ingestion
dataset serta validasi dapat lebih dahulu memotong ke `[0, 1]` untuk label batas
crop yang valid, lalu tetap menolak box degeneratif.

Parsing baris memahami task. Sembilan field berarti box berorientasi hanya dalam
mode `obb`; dalam mode `segment`, baris yang sama dibaca sebagai poligon empat titik.

YAML-nya sama dengan YAML deteksi:

```yaml
path: dataset
train: images/train
val: images/val
names:
  0: plane
  1: ship
```

JSON COCO native juga dapat dimuat, dengan pemetaan `annotations` dari nama split
ke file JSON. Anotasi dibaca menurut urutan prioritas: field `obb` berisi delapan
sudut dalam ruang piksel, field `obb` berupa `[cx, cy, w, h, angle]` dengan angle
dalam radian, poligon `segmentation` atau RLE yang di-fit ulang ke persegi panjang
minimum-area, atau `bbox` COCO biasa yang diperlakukan sebagai persegi panjang
sejajar sumbu dan dikanonisasi menjadi `xywhr`.

Parser baris kanonis adalah `libreyolo.data.parse_yolo_obb_label_line`.

## Pelatihan

<code-tabs name="train" />

Pelatihan pada task ini berarti RF-DETR. Secara default, pelatihan dilanjutkan
dari checkpoint `-obb` terbitan. Memulai dari bobot deteksi merupakan transfer
yang disengaja: bobot tersebut tidak memprediksi angle, dan memberikan `task=obb`
mengotorisasi penggantian tersebut. Pertahankan `lr0` pada atau di bawah `1e-4`,
seperti task lain dalam family. Checkpoint berorientasi RT-DETRv2 tidak dapat
di-fine-tune; gunakan apa adanya atau latih model RF-DETR pada label Anda sendiri.
Lihat [pelatihan](/docs/train) untuk dataset, augmentasi, multi-GPU, dan logger.

## Validasi

`val()` mengembalikan dictionary biasa berisi key `metrics/`. Pencocokan
menggunakan rotated IoU, yang dihitung antara persegi panjang berorientasi, bukan
box sejajar sumbu yang melingkupinya, sehingga prediksi dengan posisi benar dan
angle salah dinilai meleset.

<code-tabs name="val" />

`metrics/mAP50-95` adalah mean average precision yang dirata-ratakan pada ambang
batas IoU 0,50 hingga 0,95 dalam langkah 0,05, dan menjadi angka utama. Berbeda
dari jalur COCO untuk deteksi, task ini mematuhi `iou_thresholds` dalam konfigurasi
validasi, sehingga sweep dapat diubah. `metrics/mAP50` dan `metrics/mAP75` adalah
versi satu ambang batas. `metrics/precision` dan `metrics/recall` merupakan
precision dan recall sebenarnya pada IoU 0,50, dibaca pada operating point paling
longgar: setiap prediksi yang melewati ambang batas confidence dihitung, dan
ambang batas tersebut default ke 0,001 selama validasi. Menaikkan `conf` akan
mengubah keduanya, sedangkan angka mAP yang menggunakan seluruh kurva
precision-recall tetap. Empat key diulang dengan akhiran `(OBB)`, yaitu
`metrics/mAP50-95(OBB)`, `metrics/mAP50(OBB)`, `metrics/precision(OBB)`, dan
`metrics/recall(OBB)`, agar pemanggil dapat membedakan hasil berorientasi dari
hasil sejajar sumbu saat keduanya berada dalam tabel yang sama.
`metrics/mAP75` tidak memiliki pasangan berakhiran.

Dua opsi tidak berpengaruh pada task ini. `save_json` dan `save_plots` diterima
dan mencatat warning: dump prediksi berorientasi dan plot validasi belum
diimplementasikan.

## Ekspor

<code-tabs name="export" />

Artefak hasil ekspor dimuat kembali melalui `LibreYOLO()` berdasarkan akhiran
filenya, sehingga file `.onnx` atau `.engine` berperilaku seperti checkpoint dan
mengembalikan `Results` yang sama. Cakupan format berbeda berdasarkan task pada
family yang sama, dan matriks halaman model dibuat dari kumpulan tervalidasi serta
menyebutkan alasan target tidak tersedia. Lihat
[ekspor dan deployment](/docs/export) untuk format, extra, dan batasannya.
