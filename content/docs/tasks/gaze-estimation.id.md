---
title: Estimasi arah pandang
seo_title: Estimasi arah pandang di LibreYOLO
description: >-
  Estimasi pitch dan yaw arah pandang per wajah di LibreYOLO. Lakukan prediksi
  dari Python atau CLI, baca sudut dalam radian, dan ekspor head gaze ke ONNX.
lead: >-
  Estimasi arah pandang mengembalikan arah pandangan untuk setiap wajah dalam
  gambar. LibreYOLO memodelkannya sebagai task dua tahap: detektor wajah
  berjalan lebih dulu, lalu head gaze membaca pitch dan yaw dari setiap crop
  wajah yang dikembalikan.
keywords:
  - estimasi arah pandang python
  - eye tracking
  - pitch yaw gaze
  - L2CS-Net
  - arah pandangan
  - head pose
  - task gaze libreyolo
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Tanpa face_detector, prediksi kembali ke detektor bawaan
        # OpenCV, sehingga hanya checkpoint yang diunduh.
        model = LibreYOLO("LibreL2CSr50.pt")
        result = model(SAMPLE_IMAGE)

        gaze = result.gaze
        print(gaze.pitch, gaze.yaw)              # radian, satu baris per wajah
        print(gaze.pitch_deg, gaze.yaw_deg)      # sudut yang sama dalam derajat
        print(gaze.direction_3d)                 # vektor satuan (N, 3)
    - label: CLI
      language: bash
      code: >
        # Berbeda dari jalur Python, CLI tidak memiliki fallback otomatis: model
        gaze

        # memerlukan detektor wajah eksplisit berupa detektor LibreYOLO

        # yang box-nya merupakan wajah.

        libreyolo predict model=LibreL2CSr50.pt source=photo.jpg
        face_detector=face-detector.pt save=True
    - label: Pilih sumber wajah
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreL2CSr50.pt")

        # Berikan box dari detektor yang sudah dijalankan kepada head gaze.
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])

        # Atau gunakan salah satu detektor bawaan.
        result = model(SAMPLE_IMAGE, face_detector="yunet")
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreL2CSr50.pt")
        model.export(format="onnx")
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreL2CSr50.pt format=onnx
source_hash: 22aa3c3d87b0c730
---

## Definisi

Estimasi arah pandang mengembalikan dua sudut per wajah. `result.gaze` adalah
payload `Gaze` berbentuk `(N, 2)`, kolom 0 pitch dan kolom 1 yaw, dalam radian,
dengan baris yang selaras terhadap `result.boxes`, yaitu box wajah terdeteksi.
Konvensinya mengikuti L2CS-Net: yaw positif memutar pandangan ke kiri subjek,
pitch positif memutarnya ke bawah.

Payload yang sama menyediakan `pitch_deg` dan `yaw_deg` untuk derajat, serta
`direction_3d`, vektor satuan `(N, 3)` dalam frame kamera dengan kolom
`(x, y, z)`.

Karena task terdiri dari dua tahap, prediksi bergantung pada dua model. Wajah
yang dilewatkan detektor tidak memiliki baris gaze, dan box yang salah
menghasilkan sudut dari crop wajah yang salah. Key task kanonis adalah `gaze`;
`gaze-estimation` dinormalisasi ke sana.

## Model

[L2CS-Net](/docs/models/l2cs) adalah satu-satunya family untuk task ini. Model
memasangkan trunk ResNet dengan dua head classification bin sudut paralel, satu
untuk pitch dan satu untuk yaw, pada crop wajah 448x448. Lima depth backbone
didukung secara arsitektural dan satu checkpoint ResNet-50 diterbitkan.

Bobot memiliki batasan lisensi. Bobot dilatih pada Gaze360, yang hanya
mengizinkan penelitian dan penggunaan nonkomersial serta melarang distribusi
ulang, sehingga LibreYOLO tidak mencerminkan apa pun untuk family ini. Satu
checkpoint yang dapat diambil otomatis berasal langsung dari distribusi Google
Drive penulis melalui `gdown`, setelah ketentuan lisensi dicetak. Baca
[L2CS-Net](/docs/models/l2cs) sebelum deployment.

Jalur pengunduhan memerlukan ekstra `gaze`:

```bash
pip install "libreyolo[gaze]"
```

Tanpanya, library mencetak petunjuk pengunduhan manual. Prediksi dan ekspor
checkpoint yang sudah tersedia tidak memerlukan ekstra.

## Predict

<code-tabs name="predict" />

Sumber wajah dipilih melalui tiga cara. `face_boxes` memberikan box yang sudah
dihitung dan melewati deteksi. `face_detector` menerima `"auto"`, `"haar"`,
`"yunet"`, model deteksi LibreYOLO, atau callable biasa, dan dapat ditetapkan
pada constructor atau per pemanggilan. Jika tidak ditetapkan dalam Python,
prediksi kembali ke detektor bawaan OpenCV. Pada OpenCV 4, detektor tersebut
adalah cascade Haar dalam wheel tanpa pengunduhan; pada OpenCV 5, yang menghapus
API Haar, detektornya YuNet dan mengambil file model kecil dari OpenCV zoo satu
kali.

CLI tidak memakai fallback tersebut. `libreyolo predict` menolak model gaze
tanpa `face_detector=`, dan nilainya harus berupa nama detektor LibreYOLO atau
path checkpoint. Lihat [prediksi](/docs/predict) untuk sumber, streaming, dan
penanganan hasil.

## Train

Tidak ada family dalam task ini yang berlatih di LibreYOLO.
`LibreL2CS.train()` memunculkan error: lakukan pelatihan pada project L2CS-Net
upstream dan muat state dict hasilnya di sini.

## Validate

Validasi terhadap dataset ground truth gaze berada di luar cakupan, dan `val()`
memunculkan error alih-alih mengembalikan metrik yang tidak dihitung. Tidak ada
dictionary `metrics/` untuk task ini. Lakukan evaluasi di upstream pada dataset
asal checkpoint.

## Export

<code-tabs name="export" />

Kontrak ekspor gaze mencakup ONNX, TorchScript, ExecuTorch, TensorRT, dan
OpenVINO. Yang keluar dari library hanya trunk ResNet dan dua head bin sudut:
graph menerima crop wajah 448x448 yang sudah dipraproses dan mengembalikan logit
yaw serta pitch mentah. Deteksi wajah, cropping, softmax, expectation bin, dan
konversi ke sudut tetap berada di Python dalam `libreyolo.models.l2cs.utils`.
Lihat [ekspor](/docs/export) untuk format dan argumennya.
