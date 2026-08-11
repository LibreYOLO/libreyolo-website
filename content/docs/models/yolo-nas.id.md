---
title: YOLO-NAS
families:
  - yolonas
seo_title: 'YOLO-NAS: prediksi, latih, dan ekspor di LibreYOLO'
description: >-
  Gunakan YOLO-NAS di LibreYOLO untuk deteksi dan pose. Bobot Deci.AI bersifat
  proprietary dan nonkomersial, serta tidak ada yang diterbitkan LibreYOLO.
lead: >-
  Detektor konvolusional yang backbone dan neck-nya dihasilkan oleh pencarian
  arsitektur Deci.AI, dibuat dari block RepVGG yang menyadari kuantisasi.
  Bobotnya milik Deci.AI, berlisensi hanya untuk penggunaan nonkomersial, dan
  LibreYOLO tidak menerbitkan satu pun.
keywords:
  - YOLO-NAS
  - YOLONAS
  - Deci AI
  - SuperGradients
  - deteksi objek
  - estimasi pose
  - quantization aware detector
  - AutoNAC
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Nama yang belum tersedia di disk diambil dari CDN Deci. Pengunduhan

        # mencetak ketentuan lisensi Deci terlebih dahulu; mengambil berkas
        berarti menerimanya.

        model = LibreYOLO("LibreYOLONASs.pt")

        result = model(SAMPLE_IMAGE, save=True)


        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLONASs.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Pose
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Sufiks -pose memilih head pose dan kumpulan bobotnya sendiri.
        model = LibreYOLO("LibreYOLONASs-pose.pt")
        result = model(SAMPLE_IMAGE)

        print(result.keypoints.xy)
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLONASs.pt data=my-dataset.yaml \
          epochs=100 imgsz=640 batch=16
    - label: Dari nol
      language: python
      code: >
        from libreyolo import LibreYOLONAS


        # Tidak ada checkpoint Deci yang disentuh: model dimulai dari bobot
        acak,

        # sehingga hasil proses hanya diturunkan dari data Anda.

        model = LibreYOLONAS(None, size="s")

        model.train(data="my-dataset.yaml", imgsz=640, batch=16)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLONASs.pt data=my-dataset.yaml
    - label: Dibandingkan dengan COCO
      language: bash
      code: |
        # YAML COCO bawaan menyertakan skrip unduhan tertanam, sehingga
        # memerlukan izin eksplisit kecuali dataset sudah tersedia secara lokal.
        libreyolo val model=LibreYOLONASl.pt data=coco.yaml imgsz=640 \
          allow_download_scripts=True
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLONASs.pt")
        model.export(format="onnx", imgsz=640)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreYOLONASs.pt format=onnx imgsz=640
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan sufiks berkas, sehingga artefak hasil
        ekspor

        # dimuat seperti checkpoint lain dan mengembalikan objek Results yang
        sama.

        model = LibreYOLO("LibreYOLONASs.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 47c30d6e44024ce7
---

## Instalasi

YOLO-NAS tidak memerlukan komponen tambahan selain paket dasar.

```bash
pip install libreyolo
```

## Prediksi

Nama checkpoint yang belum tersedia di disk diambil dari CDN publik Deci, bukan organisasi
LibreYOLO, yang tidak meng-host bobot ini. Sebelum transfer dimulai, library mencetak
ketentuan lisensi Deci satu kali per proses, dan sebelum berkas unduhan dibuka, SHA-256-nya
diperiksa terhadap nilai yang ditetapkan. Izin yang diberikan ketentuan tersebut dijelaskan
dalam [lisensi](#licensing).

<code-tabs name="predict" />

Objek `Results` yang dikembalikan sama dengan yang dikembalikan setiap family, jadi mengganti
detektor hanya memerlukan perubahan satu baris. `conf` menetapkan ambang batas confidence
dan `iou` menetapkan ambang NMS. Lihat [prediksi](/docs/predict) untuk sumber, streaming,
dan penanganan hasil.

## Varian

Deteksi dan pose memakai arsitektur yang sama di bawah head berbeda, serta menerima argumen
yang sama. Ukuran pada tabel di bawah adalah ukuran deteksi; pose diterbitkan pada ukuran
tersebut dan satu ukuran lebih kecil. Head pose memprediksi kumpulan keypoint COCO.

<benchmark-table task="detect" />

<va-embed />

## Pelatihan

<code-tabs name="train" />

`epochs`, `lr0`, dan `amp` ditentukan per task jika dihilangkan, sehingga proses pose
dimulai dengan default berbeda dari proses deteksi. Optimizer memakai AdamW secara default.
Jumlah kelas berasal dari YAML dataset dan head dibangun ulang agar sesuai sebelum epoch
pertama; pada head pose, jumlah keypoint ditangani dengan cara yang sama, sehingga checkpoint
pose COCO dapat di-fine-tune ke skeleton berukuran berbeda.

Fine-tuning dimulai dari bobot Deci, yang dicakup oleh lisensi Deci. Pelatihan dari model
yang diinisialisasi secara acak sama sekali tidak melibatkan checkpoint Deci, dan itulah
snippet ketiga di atas.

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
didukung, tetapi prapemrosesan dan pascapemrosesannya harus ditulis sendiri. Setiap format
memasang komponen tambahan berbeda dan menerima beberapa argumen sendiri. Keduanya dijelaskan
pada halaman format tersebut.

Ekspor merupakan salinan lain dari bobot yang sama dalam container berbeda. Mengekspor
checkpoint Deci tidak mengubah asal bobot maupun lisensi yang mencakupnya.

<code-tabs name="export" />

## Checkpoint

Tidak ada yang dapat dicantumkan. Lisensi Deci melarang distribusi ulang, sehingga organisasi
LibreYOLO tidak menerbitkan bobot YOLO-NAS dan unduhan diselesaikan di tempat lain: nama
berbentuk `LibreYOLONAS<size>.pt`, atau `LibreYOLONAS<size>-pose.pt` untuk pose, dipetakan
ke objek yang sesuai pada CDN publik Deci.

Hanya checkpoint dengan SHA-256 yang ditetapkan library yang dapat diambil dengan cara itu.
Semua yang lain gagal dalam kondisi tertutup, bukan membuka pickle pihak ketiga yang belum
diverifikasi, dan harus diunduh secara manual lalu diteruskan sebagai path. Berkas yang sudah
ada di disk dimuat dari path-nya tanpa pengunduhan dan tanpa pemeriksaan checksum. Ini termasuk
berkas `.pth` Deci dengan nama aslinya, yang dikenali loader.

## Lisensi

<provenance-box>

LibreYOLO tidak meng-host atau mencerminkan bobot ini: tidak ada apa pun untuk family ini
dalam organisasi Hugging Face LibreYOLO. Setiap pengunduhan otomatis menuju CDN publik Deci,
mencetak ketentuan Deci satu kali per proses sebelum dimulai, dan diperiksa terhadap SHA-256
yang ditetapkan sebelum berkas dibuka.

Pelatihan dari model yang diinisialisasi secara acak adalah alternatifnya. Arsitektur
berlisensi Apache-2.0 di upstream dan MIT di sini, sehingga model yang dilatih dengan cara
tersebut pada data sendiri tidak diturunkan dari checkpoint Deci.

</provenance-box>

## Sitasi

YOLO-NAS dirilis tanpa makalah. Entri di bawah adalah yang diminta penulisnya dan mencakup
SuperGradients, library tempat model ini dirilis.

<citation-block />
