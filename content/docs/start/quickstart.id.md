---
title: Mulai cepat
seo_title: Mulai cepat LibreYOLO
description: >-
  Jalankan detektor pada gambar, lakukan fine-tuning dengan dataset kecil, lalu
  ekspor ke TorchScript atau ONNX. Semuanya berjalan pada CPU dengan sekitar
  sepuluh baris Python.
lead: >-
  Jalur tersingkat menggunakan LibreYOLO: lakukan prediksi pada satu gambar,
  latih dengan dataset kecil, lalu ekspor hasilnya. Setiap perintah di sini
  berjalan pada CPU.
keywords:
  - cara cepat libreyolo
  - tutorial libreyolo indonesia
  - prediksi libreyolo python
  - cara training libreyolo
  - export libreyolo onnx
  - contoh yolo python
last_verified: 1.5.0
meta:
  - label: Instalasi
    value: pip install libreyolo
    mono: true
  - label: Checkpoint
    value: LibreYOLO9t.pt
    mono: true
  - label: Hardware
    value: CPU cukup untuk semua contoh pada halaman ini
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Mengunduh checkpoint saat pertama digunakan, lalu menyimpannya di
        weights/.

        model = LibreYOLO("LibreYOLO9t.pt")


        # Satu gambar mengembalikan satu objek Results.

        result = model(SAMPLE_IMAGE, save=True)


        for box in result.boxes:
            print(result.names[int(box.cls)], float(box.conf), box.xyxy.tolist())
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=yolo9-t save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Video dan stream
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # stream=True menghasilkan satu Results per frame, bukan membangun list.
        # Ganti path dengan indeks webcam, URL RTSP, atau folder.
        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9t.pt")


        # coco8 adalah dataset 8 gambar yang disertakan bersama library. Dataset

        # diunduh dari URL saat pertama digunakan, sehingga tidak ada script
        yang dijalankan.

        results = model.train(
            data="coco8.yaml",
            epochs=1,
            imgsz=640,
            batch=4,
            device="cpu",
        )


        print(results["save_dir"])

        print(results["best_checkpoint"])
    - label: CLI
      language: bash
      code: |
        libreyolo train model=yolo9-t data=coco8.yaml \
          epochs=1 imgsz=640 batch=4 device=cpu
    - label: Validasi
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # val() mengembalikan dict biasa, bukan objek.
        metrics = model.val(data="coco8.yaml", device="cpu")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
  export:
    - label: TorchScript
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9t.pt")


        # export() mengembalikan path yang ditulis.

        path = model.export(format="torchscript")

        print(path)


        # Factory mengarahkan berdasarkan suffix file, sehingga artefak dimuat
        kembali

        # seperti checkpoint dan mengembalikan objek Results yang sama.

        exported = LibreYOLO(path)

        result = exported(SAMPLE_IMAGE)

        print(len(result.boxes))
    - label: ONNX
      language: bash
      code: |
        pip install "libreyolo[onnx]"
        libreyolo export model=yolo9-t format=onnx imgsz=640
source_hash: c11b6bdbf0b6fdf1
---

## Instalasi

```bash
pip install libreyolo
```

Itulah semua yang diperlukan bagian predict dan train di bawah. Ekspor ke ONNX
menambahkan satu ekstra; lihat [instalasi](/docs/install) untuk list lengkap.

## Predict

<code-tabs name="predict" />

`LibreYOLO()` adalah factory. Factory membaca file, menentukan family asal
bobot, dan mengembalikan model family tersebut, sehingga mengganti detektor
hanya memerlukan perubahan satu baris. Memberikan `LibreYOLO9t.pt` tanpa
direktori akan mencari `weights/LibreYOLO9t.pt` relatif terhadap direktori kerja
dan mengunduhnya ke sana jika tidak ada. Lihat
[checkpoint dan bobot](/docs/weights) untuk aturan pengunduhan dan cara bekerja
offline.

`save=True` menulis salinan beranotasi di bawah `runs/detect/`, dalam direktori
`predict` yang bertambah per run. `Results` yang dikembalikan memiliki `boxes`,
dan `names` memetakan indeks kelas ke labelnya. Path satu gambar mengembalikan
satu `Results`; direktori, list gambar, atau `stream=True` mengembalikan list
atau generator.

## Train

<code-tabs name="train" />

`data` adalah YAML dataset. `coco8.yaml` disertakan bersama library, sehingga
snippet dapat langsung dijalankan; nama yang tidak disertakan dibaca sebagai
path. Dataset di-resolve di bawah `~/datasets`, atau di bawah
`LIBREYOLO_DATASETS_DIR` jika variabel tersebut ditetapkan.

Run menulis ke `project/name`, dengan default direktori di bawah `runs/train`,
serta `weights/best.pt` dan `weights/last.pt` di dalamnya. `train()`
mengembalikan dictionary yang memuat `save_dir`, `best_checkpoint`,
`last_checkpoint`, loss per epoch, dan metrik validasi per epoch. Checkpoint
hasil pelatihan dimuat melalui `LibreYOLO()` sama seperti checkpoint pretrained.

Tidak setiap family dapat dilatih. Jika family hanya mendukung inferensi,
`train()` memunculkan `NotImplementedError` dan menyatakannya. [Konsep
inti](/docs/concepts) menjelaskan arti setiap tier dukungan.

## Export

<code-tabs name="export" />

TorchScript tidak memerlukan apa pun selain instalasi dasar. Target lain
masing-masing memiliki ekstra sendiri, dan cakupannya berlaku per family serta
per task, bukan seragam: lihat [ekspor dan deployment](/docs/export).

Argumen yang diterima setiap format mencakup `imgsz` (int atau pasangan tinggi
dan lebar), `batch` (default 1), `half`, `int8` dengan YAML `data` untuk
kalibrasi, `dynamic` (default True), `simplify` (default True), `opset`, `device`,
dan `output_path`. Jika `output_path` tidak diberikan, file ditulis di bawah
`weights/` dengan nama yang diturunkan dari checkpoint.

## Langkah berikutnya

- [Konsep inti](/docs/concepts) untuk task, family, ukuran, dan nama checkpoint.
- [Checkpoint dan bobot](/docs/weights) untuk pengunduhan otomatis, penggunaan offline, dan keamanan pemuatan.
- [Impor bobot yang ada](/docs/migrate) jika sudah memiliki checkpoint dari project upstream.
- [Semua model](/docs/models) untuk memilih family yang sesuai masalah.
- [Train](/docs/train), [Predict](/docs/predict), dan [Export](/docs/export) untuk workflow lengkap.
