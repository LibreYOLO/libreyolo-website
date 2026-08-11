---
title: Estimasi pandangan
seo_title: Estimasi pandangan di LibreYOLO
description: >-
  Estimasi pitch dan yaw pandangan per wajah di LibreYOLO. Prediksi dari Python
  atau CLI, baca sudut dalam radian, dan ekspor pandangan head ke ONNX.
lead: >-
  Estimasi pandangan mengembalikan arah pandangan untuk setiap wajah dalam
  gambar. LibreYOLO memodelkannya sebagai task dua tahap: detektor wajah
  dijalankan terlebih dahulu, dan pandangan head membaca pitch dan yaw dari
  setiap potongan wajah yang dikembalikan.
keywords:
  - estimasi pandangan python
  - pelacakan mata
  - pitch yaw pandangan
  - L2CS-Net
  - arah pandangan
  - pose head
  - pandangan libreyolo task
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Tanpa face_detector yang diberikan, prediksi kembali menggunakan
        detektor bawaan OpenCV

        # sehingga tidak ada yang diunduh selain checkpoint.

        model = LibreYOLO("LibreL2CSr50.pt")

        result = model(SAMPLE_IMAGE)


        gaze = result.gaze

        print(gaze.pitch, gaze.yaw)              # radian, satu baris per wajah

        print(gaze.pitch_deg, gaze.yaw_deg)      # sudut yang sama dalam derajat

        print(gaze.direction_3d)                 # (N, 3) vektor satuan
    - label: CLI
      language: bash
      code: >
        # Tidak seperti jalur Python, CLI tidak memiliki cadangan otomatis:
        pandangan

        # model memerlukan detektor wajah yang eksplisit, dan itu harus

        # LibreYOLO detektor yang kotaknya adalah wajah.

        libreyolo predict model=LibreL2CSr50.pt source=photo.jpg
        face_detector=face-detector.pt save=True
    - label: Pilih sumber wajah
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreL2CSr50.pt")

        # Berikan kotak head pandangan dari detektor yang sudah Anda jalankan.
        result = model(SAMPLE_IMAGE, face_boxes=[[34, 12, 90, 80]])

        # Atau sebutkan salah satu detektor yang sudah disertakan.
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

Estimasi pandangan mengembalikan dua sudut per wajah. `result.gaze` adalah muatan `Gaze`
berbentuk `(N, 2)`, kolom 0 pitch dan kolom 1 yaw, dalam radian, disusun baris per
baris dengan `result.boxes`, kotak wajah yang terdeteksi. Konvensinya adalah yang
L2CS-Net menggunakan: yaw positif memutar pandangan ke arah kiri subjek,
pitch positif memutarnya ke bawah.

Payload yang sama mengekspos `pitch_deg` dan `yaw_deg` untuk derajat, dan
`direction_3d`, sebuah `(N, 3)` vektor satuan dalam frame kamera dengan kolom
`(x, y, z)`.

Karena task adalah dua tahap, prediksi bergantung pada dua model. Wajah yang
terlewat oleh detektor tidak memiliki baris pandangan, dan kotak yang ditempatkan dengan buruk menghasilkan sudut dari
wajah yang dipotong dengan buruk. Kunci task kanonik adalah `gaze`; `gaze-estimation`
menormalkannya.

## Model

[L2CS-Net](/docs/models/l2cs) adalah satu-satunya family yang melayani task ini. Ini memasangkan sebuah
Trunk ResNet dengan dua kepala klasifikasi bin-sudut paralel, satu untuk pitch dan
satu untuk yaw, di atas 448x448 potongan wajah. Lima kedalaman backbone didukung
Secara arsitektural, dan salah satunya, ResNet-50, memiliki checkpoint yang dipublikasikan.

Bobot tersebut memiliki batasan lisensi. Mereka dilatih pada Gaze360, yang
lisensi hanya mengizinkan penelitian dan penggunaan non-komersial serta melarang
redistribusi, jadi LibreYOLO tidak mencerminkan apa pun untuk family ini. Satu checkpoint
perpustakaan dapat mengambil secara otomatis langsung dari Google milik penulis
Distribusikan drive, melalui `gdown`, setelah mencetak syarat lisensi. Baca
[L2CS-Net](/docs/models/l2cs) sebelum menyebarkannya.

Jalur unduhan itu memerlukan tambahan `gaze`:

```bash
pip install "libreyolo[gaze]"
```

Tanpanya, pustaka mencetak instruksi unduhan manual alih-alih mencoba
transfer. Melakukan prediksi dan mengekspor checkpoint yang sudah Anda miliki tidak memerlukan
tambahan sama sekali.

## Prediksi

<code-tabs name="predict" />

Sumber wajah dipilih dengan salah satu dari tiga cara. `face_boxes` melewati kotak yang
sudah Anda hitung dan melewati deteksi. `face_detector` menerima `"auto"`,
`"haar"`, `"yunet"`, model deteksi LibreYOLO, atau callable biasa, dan bisa
disetel pada konstruktor atau per panggilan. Jika tidak disetel di Python, prediksi akan
kembali ke detektor yang dibundel OpenCV, sehingga panggilan kosong bekerja tanpa pengaturan.
OpenCV 4 yaitu Haar cascade yang dikirimkan di dalam wheel, yang tidak membutuhkan
unduh sama sekali; pada OpenCV 5, di mana API Haar dihapus, itu adalah YuNet, yang
mengambil satu file model kecil dari OpenCV zoo sekali.

CLI tidak membagikan fallback itu. `libreyolo predict` menolak model pandangan
tanpa `face_detector=`, dan nilai yang diambilnya adalah nama detektor LibreYOLO atau
Jalur checkpoint. Lihat [prediction](/docs/predict) untuk sumber, streaming dan
penanganan hasil.

## Kereta

Tidak ada family di task kereta di dalam LibreYOLO. `LibreL2CS.train()` meningkatkan:
latih di proyek L2CS-Net hulu dan muat dictionary state yang dihasilkan di sini.

## Validasi

Validasi terhadap dataset gaze ground-truth berada di luar ruang lingkup, dan `val()`
menghasilkan alih-alih mengembalikan metrik yang tidak dihitungnya. Tidak ada `metrics/`
kamus untuk task ini. Evaluasi di hulu, pada dataset checkpoint adalah
dilatih untuk.

## Ekspor

<code-tabs name="export" />

Kontrak ekspor gaze mencakup ONNX, TorchScript, ExecuTorch, TensorRT dan
OpenVINO. Yang keluar dari pustaka adalah batang ResNet dan dua bin sudut
hanya kepala: grafik mengambil potongan wajah 448x448 yang telah diproses sebelumnya dan mengembalikan data mentah
logit yaw dan pitch. Deteksi wajah, pemotongan, softmax, bin
harapan dan konversi ke sudut semuanya tetap di Python, dalam
`libreyolo.models.l2cs.utils`. Lihat [export](/docs/export) untuk format dan
argumennya.

