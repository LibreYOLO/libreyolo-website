---
title: L2CS-Net
families:
  - l2cs
seo_title: 'L2CS-Net: estimasi arah pandang di LibreYOLO'
description: >-
  Gunakan L2CS-Net di LibreYOLO untuk estimasi pitch/yaw arah pandang dua tahap.
  Instal, prediksi, dan ekspor. Checkpoint Gaze360 hanya untuk penelitian.
lead: >-
  L2CS-Net adalah estimator arah pandang dua tahap: detector wajah menemukan
  wajah, lalu trunk ResNet dengan dua head klasifikasi bin sudut memprediksi
  pitch dan yaw per wajah. LibreYOLO membungkusnya hanya untuk inferensi.
keywords:
  - L2CS-Net
  - estimasi arah pandang
  - gaze estimation
  - eye tracking
  - pitch yaw
  - Gaze360
  - deteksi wajah
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Tidak ada face_detector yang diberikan: gunakan detector wajah bawaan
        OpenCV

        # (Haar pada OpenCV 4, YuNet pada OpenCV 5), sehingga kode ini berjalan
        tanpa

        # unduhan tambahan selain checkpoint L2CS itu sendiri.

        model = LibreYOLO("LibreL2CSr50.pt")

        result = model(SAMPLE_IMAGE)


        print(result.gaze.pitch, result.gaze.yaw)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreL2CSr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Sumber wajah
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreL2CSr50.pt")

        # Berikan L2CS box dari detector yang sudah dijalankan.
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])

        # Atau sebutkan detector wajah bawaan tertentu.
        result = model(SAMPLE_IMAGE, face_detector="yunet")
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreL2CSr50.pt")
        model.export(format="onnx")
        model.export(format="tensorrt", half=True)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreL2CSr50.pt format=onnx
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # Graph hasil ekspor hanya berisi trunk ResNet dan dua head bin sudut:

        # menerima crop wajah 448x448 yang telah diproses dan mengembalikan

        # (yaw_logits, pitch_logits) mentah, bukan sudut yang sudah didekode.
        Softmax,

        # ekspektasi bin, dan konversi derajat tetap dalam Python. Lihat

        # libreyolo.models.l2cs.utils.bin_logits_to_angles.

        session = ort.InferenceSession("LibreL2CSr50.onnx")

        name = session.get_inputs()[0].name

        yaw_logits, pitch_logits = session.run(
            None, {name: np.zeros((1, 3, 448, 448), dtype=np.float32)}
        )
source_hash: 4ec43f4673b4be3e
---

## Instalasi

L2CS-Net tidak memerlukan extra untuk membuat, menjalankan prediksi, atau
mengekspor model yang checkpoint-nya sudah tersedia.

```bash
pip install libreyolo
```

Satu checkpoint yang dapat diambil LibreYOLO secara otomatis, yaitu ResNet-50
yang dilatih pada Gaze360, diunduh melalui `gdown`, bukan cermin HTTP biasa,
karena berada di Google Drive milik penulis dan bukan organisasi LibreYOLO.
Jalur tersebut memerlukan extra `gaze`:

```bash
pip install "libreyolo[gaze]"
```

Tanpa extra itu, LibreYOLO mencetak petunjuk unduhan manual alih-alih gagal
secara diam-diam.

## Prediksi

<code-tabs name="predict" />

L2CS-Net adalah estimator dua tahap. Detector wajah berjalan lebih dulu, lalu
head arah pandang membaca pitch dan yaw dari setiap crop wajah yang
dihasilkannya. Jika dibiarkan, prediksi menggunakan detector bawaan OpenCV,
sehingga panggilan polos dapat berjalan tanpa unduhan tambahan setelah
checkpoint L2CS tersedia. `face_boxes` menerima box dari detector yang sudah
dijalankan. `face_detector` menerima `"auto"`, `"haar"`, `"yunet"`, model
deteksi LibreYOLO, atau callable biasa. `result.gaze` memuat pitch dan yaw dalam
radian, yang diselaraskan baris demi baris dengan `result.boxes`, yaitu box wajah
yang terdeteksi. Lihat [prediksi](/docs/predict) untuk sumber, streaming, dan
penanganan hasil.

## Varian

Lima kedalaman backbone memakai satu resolusi input dan menerima argumen yang
sama. Gaze360, dataset di balik satu-satunya checkpoint yang dipublikasikan,
melatih ResNet-50. Empat kedalaman lainnya didukung secara arsitektural, tetapi
tidak memiliki bobot terpublikasi untuk dimuat.

## Ekspor

<export-matrix />

<code-tabs name="export" />

## Lisensi

<provenance-box>

LibreYOLO tidak menghosting atau mencerminkan checkpoint L2CS apa pun. Tidak ada
artefak family ini dalam organisasi Hugging Face LibreYOLO, tidak seperti
sebagian besar family lain di situs ini. Satu checkpoint yang dapat diambil
library secara otomatis berasal langsung dari distribusi Google Drive milik
penulis, dibatasi oleh pemberitahuan lisensi Gaze360 yang dicetak sebelum
transfer dimulai. Checkpoint ini bukan salinan "dipublikasikan ulang di
huggingface.co/LibreYOLO" seperti yang tersirat dalam ringkasan di atas.

</provenance-box>
