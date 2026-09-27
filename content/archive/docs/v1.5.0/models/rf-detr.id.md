---
title: RF-DETR
families:
  - rfdetr
seo_title: 'RF-DETR: latih, lakukan fine-tuning, dan ekspor di bawah MIT'
description: >-
  Gunakan RF-DETR di LibreYOLO untuk deteksi, segmentasi instance, pose, dan
  kotak berorientasi. Instal, prediksi, latih, validasi, dan ekspor, semuanya
  berlisensi MIT.
lead: >-
  Detection transformer yang memprediksi sekumpulan objek tetap, bukan grid
  padat, sehingga tidak memerlukan NMS saat inferensi. LibreYOLO mendukungnya
  untuk empat task.
keywords:
  - RF-DETR
  - real-time detection transformer
  - DETR
  - deteksi objek
  - segmentasi instance
  - estimasi pose
  - bounding box berorientasi
last_verified: 1.5.0
hero:
  src: /showcase/parkour-detection.mp4
  poster: /showcase/parkour-detection-poster.jpg
  caption: Deteksi video dengan LibreRFDETRs pada 512 px.
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRFDETRs.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreRFDETRs.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Video
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRFDETRs.pt")


        # Sumber apa pun yang diterima library: berkas, folder, URL, indeks
        webcam,

        # stream RTSP, atau daftar .streams

        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRFDETRs.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=512, batch=8,
        lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 imgsz=512 batch=8 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 device=0,1 batch=-1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")

        # val() mengembalikan dict biasa, bukan objek
        metrics = model.val(data="my-dataset.yaml", imgsz=512)

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRFDETRs.pt data=my-dataset.yaml imgsz=512
    - label: Dibandingkan dengan COCO
      language: bash
      code: |
        # YAML COCO bawaan menyertakan skrip unduhan tertanam, sehingga
        # memerlukan izin eksplisit kecuali dataset sudah tersedia secara lokal.
        libreyolo val model=LibreRFDETRn.pt data=coco.yaml imgsz=384 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRFDETRs.pt")

        model.export(format="onnx", imgsz=512)

        model.export(format="tensorrt", imgsz=512, half=True)


        # Argumen yang diterima untuk setiap format:

        #

        #   format    "onnx" | "torchscript" | "executorch" | "tensorrt"

        #             | "openvino" | "paddle" | "mnn" | "rknn" | "ncnn"

        #             | "tflite" | "coreml" | "coreai".

        #             "engine" adalah alias untuk tensorrt, "litert" untuk
        tflite.

        #   imgsz     int, atau (height, width). Default-nya resolusi native

        #             checkpoint.

        #   batch     int, default 1.

        #   half      bool, ekspor dalam FP16. Default False.

        #   int8      bool, ekspor dalam INT8. Default False. Memerlukan `data`.

        #   data      path ke YAML dataset, digunakan untuk kalibrasi int8.

        #   fraction  float, bagian set kalibrasi yang digunakan. Default 1.0.

        #   dynamic   bool, sumbu dinamis. Default True.

        #   simplify  bool, jalankan penyederhanaan graph ONNX. Default True.

        #   opset     int, opset ONNX. Dipilih per family jika tidak diberikan.

        #   device    str, perangkat untuk tracing. Default-nya perangkat model.

        #   output_path  str, default-nya nama yang diturunkan dari checkpoint.

        #   verbose   bool, default False.

        #   allow_download_scripts  bool, default False. Mengizinkan Python

        #             tertanam dalam YAML dataset yang harus diunduh.

        #

        # Beberapa format menerima argumen tambahan sendiri, seperti platform

        # target RKNN. Argumen tersebut didokumentasikan di halaman tiap format.
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreRFDETRs.pt format=onnx imgsz=512

        libreyolo export model=LibreRFDETRs.pt format=tensorrt imgsz=512
        half=True
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Factory merutekan berdasarkan sufiks berkas, sehingga artefak hasil
        ekspor

        # dimuat seperti checkpoint lain dan mengembalikan objek Results yang
        sama.

        model = LibreYOLO("LibreRFDETRs.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
    - label: Tanpa LibreYOLO
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # Menjalankan graph secara langsung berarti melakukan prapemrosesan dan

        # pascapemrosesan sendiri. Periksa signature sebelum menghubungkannya.

        session = ort.InferenceSession("LibreRFDETRs.onnx")

        name = session.get_inputs()[0].name

        outputs = session.run(None, {name: np.zeros((1, 3, 512, 512),
        dtype=np.float32)})


        for meta, array in zip(session.get_outputs(), outputs):
            print(meta.name, array.shape)
source_hash: 8c464aa759131694
---

## Instalasi

RF-DETR memerlukan komponen tambahannya sendiri, yang memasang `transformers` untuk backbone.

```bash
pip install "libreyolo[rfdetr]"
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam cache lokal.

<code-tabs name="predict" />

Objek `Results` yang dikembalikan sama dengan yang dikembalikan setiap family, jadi mengganti
detektor hanya memerlukan perubahan satu baris. `conf` dan `max_det` memfilter pemilihan
query; tidak ada tahap NMS yang perlu disetel. Lihat [prediksi](/docs/predict) untuk sumber,
streaming, dan penanganan hasil.

## Varian

Empat ukuran dan empat task yang memakai satu arsitektur: segmentasi, pose, dan kotak
berorientasi menggunakan kembali decoder deteksi dengan head berbeda, sehingga semuanya
menerima argumen yang sama. Setiap ukuran memiliki jumlah parameter yang mirip dan terutama
berbeda dalam resolusi input.

<benchmark-table task="detect" />

<va-embed />

## Pelatihan

Pelatihan dimulai dari checkpoint yang dipublikasikan untuk keempat task. RF-DETR
mencantumkan `pretrained` di antara argumen yang diabaikan pelatih native-nya, sehingga
`pretrained=False` tidak menghasilkan model yang diinisialisasi secara acak di sini.

<code-tabs name="train" />

Dua argumen lebih penting di sini daripada pada detektor CNN. Pertahankan `lr0` pada atau
di bawah `1e-4`, karena detektor transformer mengalami divergensi pada learning rate yang
masih dapat ditoleransi model YOLO. Biarkan `imgsz` pada resolusi native checkpoint kecuali
ada alasan untuk mengubahnya. Input harus habis dibagi ukuran patch backbone dikalikan jumlah
window; LibreYOLO memeriksanya sebelum proses dimulai dan menyebutkan ukuran valid terdekat.

Lihat [pelatihan](/docs/train) untuk dataset, augmentasi, multi-GPU, dan logger.

## Validasi

`val()` mengembalikan dictionary dengan key `metrics/` yang mencakup presisi, recall,
mAP 50, dan mAP 50-95, yang diukur terhadap dataset apa pun dalam format yang digunakan
untuk pelatihan.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Artefak hasil ekspor dimuat kembali melalui `LibreYOLO()` berdasarkan sufiks berkasnya,
sehingga berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan mengembalikan
`Results` yang sama. Menjalankan graph pada runtime mandiri tanpa memasang LibreYOLO juga
didukung, tetapi prapemrosesan dan pascapemrosesannya harus ditulis sendiri.

<code-tabs name="export" />

## Checkpoint

Setiap berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
