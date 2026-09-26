---
title: DEIM
families:
  - deim
seo_title: DEIM dan DEIMv2 di LibreYOLO
description: >-
  Pakai DEIM dan DEIMv2 di LibreYOLO untuk deteksi objek. Instalasi, prediksi,
  pelatihan, validasi dan ekspor, mulai dari ukuran setengah juta parameter ke
  atas.
lead: >-
  Detection transformer yang dilatih dengan dense one-to-one matching, dan
  konvergen dalam epoch jauh lebih sedikit dibanding resep DETR yang
  mendasarinya. LibreYOLO menyediakan dua versinya, dibedakan lewat checkpoint
  yang dimuat.
keywords:
  - DEIM
  - DEIMv2
  - DINOv3
  - detection transformer
  - DETR
  - object detection
  - deteksi objek python
  - real-time detection
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDEIMn.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreDEIMn.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Video
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Versinya menjadi bagian dari nama berkas, dan factory-nya memilih
        # berdasarkan checkpoint, jadi keduanya dimuat dengan cara yang sama.
        model = LibreYOLO("LibreDEIMv2pico.pt")

        # Source apa pun yang diterima library: berkas, folder, URL, indeks
        # webcam, stream RTSP, atau daftar .streams
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")

        # coco128.yaml mengunduh sampel 128 gambar saat pertama kali dipakai.
        # Arahkan `data` ke YAML dataset Anda sendiri untuk pelatihan sungguhan.
        model.train(data="coco128.yaml", epochs=50, batch=8, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDEIMn.pt data=coco128.yaml \
          epochs=50 batch=8 lr0=1e-4
    - label: DEIMv2
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Bila dibiarkan kosong, epochs, batch, imgsz dan lr0 diambil dari
        # resep rilis untuk ukuran yang dimuat.
        model = LibreYOLO("LibreDEIMv2pico.pt")
        model.train(data="coco128.yaml", epochs=50)
    - label: LoRA
      language: python
      code: |
        # Butuh extra lora: pip install "libreyolo[lora]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")
        model.train(data="coco128.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreDEIMn.pt data=coco128.yaml \
          epochs=50 device=0,1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")

        # val() mengembalikan dict biasa, bukan objek
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDEIMn.pt data=coco128.yaml
    - label: Terhadap COCO
      language: bash
      code: |
        # coco-val-only.yaml mengambil 5000 gambar val2017 dan melewati set
        # pelatihan. Di dalamnya ada skrip unduhan tertanam, jadi butuh izin
        # eksplisit kecuali datasetnya sudah tersedia secara lokal.
        libreyolo val model=LibreDEIMn.pt data=coco-val-only.yaml \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        # Butuh extra onnx: pip install "libreyolo[onnx]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDEIMn.pt")
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreDEIMn.pt format=onnx
    - label: Memakai berkas hasil ekspor
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory-nya memilih berdasarkan sufiks berkas, jadi artefak hasil
        # ekspor dimuat seperti checkpoint biasa dan mengembalikan objek
        # Results yang sama.
        model = LibreYOLO("LibreDEIMn.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 6edaac5f05abaabe
---

## Instalasi

Tidak ada versi yang memerlukan extra opsional. Semua yang diimpor keduanya
sudah tersedia di instalasi dasar.

```bash
pip install libreyolo
```

Fine-tuning adapter dengan `lora=True` adalah pengecualiannya, dan memerlukan
extra `lora`.

```bash
pip install "libreyolo[lora]"
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali dipakai lalu disimpan di
cache lokal.

<code-tabs name="predict" />

Objek `Results` yang dikembalikan sama dengan yang dikembalikan setiap family,
jadi mengganti detektor dengan yang lain hanya butuh perubahan satu baris.
`conf` dan `max_det` menyaring decode top-k atas query dan kelas; tidak ada
langkah NMS yang perlu disetel, dan `iou` tetap diterima tetapi tidak dipakai.
Lihat [prediksi](/docs/predict) untuk source, streaming dan penanganan hasil.

## Varian

Versi 1 hadir dengan lima ukuran, semuanya pada ukuran input yang sama. Versi 2
mempertahankan kelima nama itu dan menambahkan tiga ukuran yang lebih kecil,
`atto`, `femto` dan `pico`, dan dua yang pertama native pada ukuran input yang
lebih rendah daripada sisanya. Karena itu lima kode ukuran ada di kedua versi
dan menamai model yang berbeda; versinya ditulis ke dalam nama berkas
checkpoint.

<benchmark-table task="detect" />

<va-embed />

Versi 1 mempertahankan arsitektur D-FINE dan mengganti objective klasifikasinya
dengan loss matchability-aware dari resep dense one-to-one, sehingga kedua
family berbagi hampir semua key state dict dan dibedakan lewat metadata di
dalam checkpoint. Versi 2 mempertahankan kontrak pelatihan itu dan mencampur
backbone: HGNetv2 di bawah `s`, dan vision transformer DINOv3 dengan adapter
spatial tuning pada `s` ke atas. Backbone itulah yang menempelkan lisensi kedua
pada keempat checkpoint tersebut, jadi baca [lisensi](#licensing) sebelum
menerapkan salah satunya ke produksi.

## Pelatihan

Pelatihan dimulai dari checkpoint yang sudah dipublikasikan. `pretrained` tidak
pernah sampai ke trainer: versi 1 memperingatkan bahwa key itu tidak dikenal
lalu mengabaikannya, versi 2 menghapusnya. Keduanya tidak memberi model dengan
inisialisasi acak.

<code-tabs name="train" />

Berikan `lr0` sendiri pada versi 1. Signature `train()` Python-nya memakai
default `4e-4`, yaitu laju dari resep COCO yang dipublikasikan, sementara
config pelatihan family ini memuat `1e-4` sebagai default fine-tune-nya, dan
nilai yang lebih rendah itulah yang dipakai CLI ketika argumennya tidak
diberikan. Config itu mencatat pengukuran di baliknya: pada ukuran batch yang
benar-benar dipakai saat fine-tuning, pada dataset kecil, laju COCO menurunkan
kualitas transfer secara terukur.

Versi 2 menentukan default itu sendiri. Membiarkan `epochs`, `batch`, `imgsz`
dan `lr0` tidak diisi membuatnya membaca masing-masing dari resep rilis untuk
ukuran yang dimuat, sehingga ukuran yang kecil dilatih pada resolusi input
masing-masing tanpa perlu diberi tahu, dan nilai yang Anda berikan menimpa
resep tersebut. `imgsz` adalah argumen yang dibatasinya: nilainya harus
kelipatan positif dari 32, dan jika tidak, versi 2 memunculkan error sebelum
run dimulai.

Lihat [pelatihan](/docs/train) untuk dataset, augmentasi, multi-GPU dan logger.

## Validasi

`val()` mengembalikan dictionary berisi key `metrics/` yang mencakup presisi,
recall, mAP 50 dan mAP 50-95, diukur terhadap dataset apa pun dalam format yang
dipakai saat pelatihan.

<code-tabs name="val" />

Baris pada tabel benchmark di atas berasal dari harness benchmark LibreYOLO;
catatan di bawah tabel itu mencatat dataset mana yang menghasilkannya dan
menautkan catatan run-nya.

## Ekspor

<export-matrix />

Matriks ini mencakup kedua versi dalam satu halaman: bila keduanya berbeda soal
suatu format, selnya menampilkan yang lebih lemah di antara keduanya, sehingga
tidak ada yang dilebih-lebihkan untuk versi mana pun yang dimuat.

Artefak hasil ekspor dimuat kembali lewat `LibreYOLO()` berdasarkan sufiks
berkasnya, jadi berkas `.onnx` atau `.engine` berperilaku seperti checkpoint
dan mengembalikan `Results` yang sama.

<code-tabs name="export" />

## Checkpoint

Semua berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box>
Empat ukuran DEIMv2 dari S ke atas mengambil backbone-nya dari DINOv3,
sehingga repositori bobotnya membawa Apache-2.0 sekaligus DINOv3 License milik
Meta, dan LibreYOLO menyertakan source backbone DINOv3 di bawah perjanjian yang
sama. Sisa family ini, termasuk setiap ukuran DEIMv2 di bawah S, hanya
Apache-2.0.
</provenance-box>

## Sitasi

<citation-block />

DEIMv2 adalah paper terpisah dan punya blok sitasi sendiri di
[github.com/Intellindust-AI-Lab/DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2#5-citation);
kutip yang itu jika Anda memakai checkpoint versi 2.
