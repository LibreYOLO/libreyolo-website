---
title: EdgeCrafter
families:
  - ec
seo_title: 'EdgeCrafter: deteksi, pose dan segmentasi di LibreYOLO'
description: >-
  Pakai EdgeCrafter di LibreYOLO untuk deteksi, pose dan segmentasi instance.
  Instalasi, prediksi, validasi dan ekspor, dengan kode berlisensi MIT.
lead: >-
  Vision transformer ringkas untuk prediksi dense di perangkat edge,
  dipublikasikan upstream sebagai tiga model bersaudara: ECDet, ECPose dan
  ECSeg. LibreYOLO memuat ketiganya sebagai satu family, dengan task ditentukan
  oleh checkpoint.
keywords:
  - EdgeCrafter
  - ECDet
  - ECPose
  - ECSeg
  - compact vision transformer
  - deteksi objek python
  - object detection
  - pose estimation
  - instance segmentation
  - inference di perangkat edge
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreECs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreECs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Pose
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Sufiks -pose pada nama berkas memilih head keypoint, jadi argumen
        # task tidak diperlukan di sini.
        model = LibreYOLO("LibreECs-pose.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.keypoints.xy)
        print(result.boxes.conf)
    - label: Segmentasi instance
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreECs-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            imgsz=640,
            batch=8,
            lr0=5e-4,
        )
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreECs.pt data=my-dataset.yaml epochs=50
        imgsz=640 batch=8 lr0=5e-4
    - label: Pose
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Butuh dataset keypoint satu kelas yang data.yaml-nya mendeklarasikan
        # kpt_shape, dan imgsz pada ukuran native checkpoint.
        model = LibreYOLO("LibreECs-pose.pt")
        model.train(
            data="my-pose-dataset.yaml",
            epochs=50,
            imgsz=640,
        )
    - label: Segmentasi instance
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Butuh label poligon, dan imgsz pada ukuran native checkpoint.
        model = LibreYOLO("LibreECs-seg.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            imgsz=640,
        )
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.train(
            data="my-dataset.yaml",
            epochs=50,
            lora=True,
        )
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreECs.pt data=my-dataset.yaml
    - label: Pose
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-pose.pt")
        metrics = model.val(data="my-pose-dataset.yaml")

        print(metrics["metrics/keypoints_mAP50-95"])
        print(metrics["metrics/keypoints_mAP50"])
    - label: Segmentasi instance
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # mask
        print(metrics["metrics/mAP50-95(B)"])   # box
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreECs.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreECs.pt format=onnx imgsz=640
        libreyolo export model=LibreECs-pose.pt format=onnx imgsz=640
        libreyolo export model=LibreECs-seg.pt format=onnx imgsz=640
    - label: Memakai berkas hasil ekspor
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory memilih jalur berdasarkan sufiks berkas, jadi artefak hasil
        # ekspor dimuat seperti checkpoint biasa dan mengembalikan objek
        # Results yang sama.
        model = LibreYOLO("LibreECs.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 39c6975fc16b3ff1
---

## Instalasi

EdgeCrafter tidak memerlukan extra opsional. Semua yang diimpornya sudah
tersedia di instalasi dasar.

```bash
pip install libreyolo
```

Fine-tuning adapter dengan `lora=True` adalah pengecualian, dan memerlukan extra
`lora`.

```bash
pip install "libreyolo[lora]"
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali dipakai lalu disimpan di cache
lokal.

<code-tabs name="predict" />

Task berasal dari nama berkas, jadi checkpoint `-pose` atau `-seg` memilih
head-nya sendiri dan tidak menerima argumen task. Ketiganya mengembalikan objek
`Results` yang dikembalikan setiap family, dengan tambahan `result.keypoints`
untuk pose dan `result.masks` untuk segmentasi. Pose mencakup satu kelas, yaitu
person, dengan 17 keypoint COCO, dan jumlahnya ditetapkan saat model dibangun.
Pose tidak punya head box, jadi setiap box pose adalah rentang batas dari
keypoint-nya sendiri, dan kanal keypoint ketiga berupa konstanta, bukan skor per
titik.

`conf` dan `max_det` menyaring pemilihan query; `iou` tetap diterima demi
keseragaman API tetapi tidak berpengaruh, karena ketiga head melakukan decode
atas sekumpulan query tanpa langkah NMS. Lihat [prediksi](/docs/predict) untuk
source, streaming dan penanganan hasil.

## Varian

Empat ukuran. Semuanya berjalan pada resolusi input yang sama, jadi tabel
membedakannya lewat jumlah parameter dan akurasi.

<benchmark-table task="detect" />

<va-embed />

Upstream memublikasikan ECDet, ECPose dan ECSeg sebagai tiga model terpisah,
bukan satu model dengan tiga head. Ketiganya berbagi backbone ECViT dan encoder
hybrid, dan hanya berbeda pada head, jadi LibreYOLO menyatukannya menjadi satu
family dan membiarkan nama berkas checkpoint yang menentukan task. Karena itu
huruf ukuran berarti backbone dan encoder yang sama di ketiganya, dan prediksi,
validasi serta ekspor memakai argumen yang sama, mana pun yang dimuat.

## Pelatihan

Ketiga task dilatih lewat `train()`, yang membaca task dari checkpoint yang
dimuat lalu memilih trainer yang cocok.

<code-tabs name="train" />

Yang sudah diperiksa untuk deteksi dan segmentasi: kesetaraan inferensi terhadap
upstream pada 1e-5, lapisan demi lapisan dan per ukuran, serta bahwa loss dan
satu langkah pelatihan berjalan pada input sintetis. Yang belum, menurut
docstring `train()` sendiri: konvergensi fine-tune penuh, pelatihan multi-GPU,
langkah stop-augmentation best-reload, dan remap kelas Objects365 ke COCO. Jalur
pose mengikuti resep yang dipublikasikan DETRPose, yaitu matcher Hungarian atas
biaya kelas, L1 keypoint dan OKS dengan denoising keypoint kontrastif, dan
konvergensinya juga belum diperiksa dari ujung ke ujung.

Jika dibiarkan apa adanya, trainer menjalankan 74 epoch pada `lr0=5e-4` dengan
mixed precision aktif, mengikuti resep upstream: AdamW, jadwal cosine datar, EMA
pada 0.9999 dan input yang dinormalisasi ala ImageNet. Pose dan segmentasi
sama-sama mengharuskan `imgsz` pada ukuran native checkpoint, karena grid anchor
untuk evaluasinya dibangun saat model dikonstruksi; nilai yang berbeda akan
memunculkan error sebelum run dimulai. Pose juga mengharuskan dataset satu kelas
yang `data.yaml`-nya mendeklarasikan `kpt_shape`, dengan jumlah keypoint yang
cocok dengan head.

`lora=True` hanya berlaku untuk deteksi; pose dan segmentasi memunculkan
`ValueError` jika memakainya. Di Apple silicon, trainer mempertahankan run di GPU
dan mengirim satu operasi ke CPU, yaitu backward grid-sample di dalam deformable
attention, yang tidak diimplementasikan PyTorch di Metal.

Lihat [pelatihan](/docs/train) untuk dataset, augmentasi, multi-GPU dan logger.

## Validasi

`val()` mengembalikan dictionary dengan key berupa nama metrik, dan mencetak
hasil per kelas jika `verbose` dibiarkan aktif.

<code-tabs name="val" />

Pose melaporkan metrik OKS keypoint di bawah `metrics/keypoints_*`. Segmentasi
melaporkan mask di bawah key `metrics/mAP50-95` biasa dan mengulang kedua sudut
pandang dalam satu pass, box di bawah `(B)` dan mask di bawah `(M)`.

## Ekspor

<export-matrix />

Artefak hasil ekspor dimuat kembali lewat `LibreYOLO()` berdasarkan sufiks
berkasnya, jadi berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan
mengembalikan `Results` yang sama. Pose dan segmentasi diekspor pada input tetap
640 kali 640, bukan shape dinamis, dan beberapa target deteksi juga memakai
kanvas tetap, termasuk OpenVINO, Paddle, MNN, ExecuTorch dan Core AI.
[Ekspor](/docs/export) memuat daftar argumen yang diterima setiap format dan
tambahan yang disediakan sebagian di antaranya.

<code-tabs name="export" />

## Checkpoint

Semua berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
