---
title: RT-DETR
families:
  - rtdetr
seo_title: 'RT-DETR, RT-DETRv2, dan RT-DETRv4 di LibreYOLO'
description: >-
  Gunakan RT-DETR, RT-DETRv2, dan RT-DETRv4 di LibreYOLO untuk deteksi objek,
  ditambah kotak berorientasi pada RT-DETRv2. Instal, prediksi, latih, validasi,
  dan ekspor dengan bobot Apache-2.0.
lead: >-
  Detection transformer yang dibuat untuk inferensi real-time: model ini
  mendekode sekumpulan query tetap, bukan grid padat, sehingga tidak menjalankan
  NMS. LibreYOLO menyediakan tiga versinya, yang dibedakan berdasarkan
  checkpoint yang dimuat, dan versi 2 juga melayani kotak berorientasi.
keywords:
  - RT-DETR
  - RT-DETRv2
  - RT-DETRv4
  - real-time detection transformer
  - DETR
  - deteksi objek
  - oriented bounding box detection
  - OBB
  - DOTA
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreRTDETRr18.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreRTDETRr18.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Video
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Versi menjadi bagian dari nama berkas, dan factory merutekan
        berdasarkan

        # checkpoint, sehingga ketiganya dimuat dengan cara yang sama.

        model = LibreYOLO("LibreRTDETRv4s.pt")


        # Sumber apa pun yang diterima library: berkas, folder, URL, indeks
        webcam,

        # stream RTSP, atau daftar .streams

        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
    - label: Kotak berorientasi
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Khusus versi 2. Sufiks -obb memilih task, dan checkpoint dikenali
        sebagai

        # berorientasi dari tensor miliknya, sehingga argumen task tidak
        diperlukan.

        # Bobot ini memakai DOTA v1.0, 15 kelas udara pada 1024 px.

        model = LibreYOLO("LibreRTDETRv2n-obb.pt")

        result = model("aerial.png", save=True)


        obb = result.obb

        print(obb.xywhr)     # (N, 5): cx, cy, w, h, radian

        print(obb.xyxyxyxy)  # baris yang sama sebagai empat titik sudut

        print(result.boxes.xyxy)  # kotak axis-aligned yang mencakupnya
    - label: 'Kotak berorientasi, CLI'
      language: bash
      code: >
        libreyolo predict model=LibreRTDETRv2n-obb.pt source=aerial.png
        save=True
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreRTDETRr18.pt")


        # coco128.yaml mengunduh sampel 128 gambar saat pertama kali digunakan.
        Arahkan `data`

        # ke YAML dataset sendiri untuk proses yang sebenarnya.

        model.train(data="coco128.yaml", epochs=50, batch=4, lr0=1e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRTDETRr18.pt data=coco128.yaml \
          epochs=50 batch=4 lr0=1e-4
    - label: LoRA
      language: python
      code: |
        # Memerlukan komponen tambahan lora: pip install "libreyolo[lora]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        model.train(data="coco128.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreRTDETRr18.pt data=coco128.yaml \
          epochs=50 device=0,1
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")

        # val() mengembalikan dict biasa, bukan objek
        metrics = model.val(data="coco128.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreRTDETRr18.pt data=coco128.yaml
    - label: Dibandingkan dengan COCO
      language: bash
      code: >
        # coco-val-only.yaml mengambil 5000 gambar val2017 dan melewati set

        # pelatihan. Berkas ini menyertakan skrip unduhan tertanam, sehingga
        memerlukan

        # izin eksplisit kecuali dataset sudah tersedia secara lokal.

        libreyolo val model=LibreRTDETRr18.pt data=coco-val-only.yaml \
          allow_download_scripts=True
    - label: Kotak berorientasi
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Validasi berorientasi mencocokkan dengan IoU berotasi, sehingga
        prediksi di

        # tempat yang benar tetapi dengan sudut yang salah dianggap luput.

        model = LibreYOLO("LibreRTDETRv2n-obb.pt")

        metrics = model.val(data="my-obb-dataset.yaml")


        print(metrics["metrics/mAP50-95(OBB)"])

        print(metrics["metrics/mAP50(OBB)"])
  export:
    - label: Python
      language: python
      code: |
        # Memerlukan komponen tambahan onnx: pip install "libreyolo[onnx]"
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRTDETRr18.pt")
        path = model.export(format="onnx")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export model=LibreRTDETRr18.pt format=onnx
    - label: Kotak berorientasi
      language: bash
      code: >
        # ONNX dan TorchScript adalah target tervalidasi untuk task
        berorientasi,

        # pada FP32, batch 1, dengan canvas tetap 1024 kali 1024.

        libreyolo export model=LibreRTDETRv2n-obb.pt format=onnx imgsz=1024

        libreyolo export model=LibreRTDETRv2n-obb.pt format=torchscript
        imgsz=1024
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan sufiks berkas, sehingga artefak hasil
        ekspor

        # dimuat seperti checkpoint lain dan mengembalikan objek Results yang
        sama.

        model = LibreYOLO("LibreRTDETRr18.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: 8022a5a591922a90
---

## Instalasi

RT-DETR tidak memerlukan komponen tambahan opsional. Semua impornya tersedia dalam
instalasi dasar, dan komponen tambahan `rtdetr` adalah nama stabil yang tidak menambahkan apa pun.

```bash
pip install libreyolo
```

Fine-tuning adapter dengan `lora=True` menjadi pengecualian dan memerlukan komponen tambahan `lora`.

```bash
pip install "libreyolo[lora]"
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam cache lokal.

<code-tabs name="predict" />

Objek `Results` yang dikembalikan sama dengan yang dikembalikan setiap family, jadi mengganti
detektor hanya memerlukan perubahan satu baris. `conf` dan `max_det` memfilter dekode top-k
pada query dan kelas; tidak ada tahap NMS yang perlu disetel, dan `iou` diterima tetapi tidak
digunakan. Checkpoint berorientasi mengisi `result.obb` secara native dan juga mengisi
`result.boxes` dengan persegi panjang axis-aligned yang mencakupnya. Lihat
[prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Varian

Tiga versi dengan dua task di antaranya, dan kode ukurannya tidak membentuk satu seri.
Versi 1 menamai ukuran berdasarkan backbone, yaitu ResNet atau HGNetv2. Versi 2 hanya
menggunakan kembali nama ResNet: versi 1 sudah menyediakan dua ukuran HGNetv2, dan hasil
versi 2 pada ukuran tersebut cukup mirip sehingga LibreYOLO tidak memublikasikan bobot
duplikat untuknya. Versi 4 memakai seri huruf biasa, yang bertabrakan dengan nama HGNetv2
versi 1, sehingga kode ukuran saja tidak mengidentifikasi model. Versinya ditulis dalam
nama berkas checkpoint.

<benchmark-table task="detect" />

<va-embed />

Versi 2 mempertahankan arsitektur dan tata letak state dict versi 1 serta mengubah cara
deformable attention mengambil sampel. Karena itu keduanya dibedakan berdasarkan metadata
dalam checkpoint, bukan berdasarkan bentuk. Versi 4 berasal dari garis keturunan berbeda:
versi ini menggunakan kembali arsitektur dan pelatih D-FINE, sedangkan bobotnya berasal dari
distilasi model teacher vision foundation DINOv3 menjadi student HGNetv2. Di LibreYOLO,
`LibreRTDETRv4` adalah subclass `LibreDFINE` dengan head mask dinonaktifkan secara tetap,
sehingga hanya mendukung deteksi.

### Kotak berorientasi pada versi 2

Versi 2 adalah satu-satunya versi yang memiliki task kedua. Task yang didukung adalah
`detect` dan `obb`, dan keduanya tidak memakai graph atau seri ukuran yang sama.
Deteksi memakai ukuran ResNet pada 640 px; deteksi berorientasi memakai seri HGNetv2
n, s, m, l, dan x pada 1024 px, sedangkan ukuran input ditentukan per task, bukan per
family. Checkpoint dikenali sebagai berorientasi dari tensor miliknya sendiri melalui
head kotak lima koordinat dan parameter sampling versi 2, sehingga bobot `-obb` dimuat
ke graph berorientasi tanpa argumen `task`. Ketidakcocokan di antara keduanya menghasilkan
error tegas, bukan reinterpretasi tanpa pemberitahuan.

Berkas yang dipublikasikan adalah `LibreRTDETRv2n-obb.pt` hingga
`LibreRTDETRv2x-obb.pt`. Semuanya merupakan checkpoint single-scale DOTA v1.0 resmi yang
dikonversi ke format LibreYOLO, dengan 15 kelas udara dari pesawat dan kapal hingga pelabuhan
dan helikopter, sedangkan nama kelasnya disimpan dalam checkpoint. Tidak seperti sisi deteksi,
task berorientasi hanya mendukung inferensi: prediksi, validasi, dan ekspor berfungsi,
sedangkan `train()` pada model berorientasi memunculkan error. Pelacakan dan augmentasi saat
pengujian juga tidak mendukung kotak berorientasi. [Deteksi berorientasi](/docs/tasks/oriented-detection)
membahas task, format label, dan metriknya.

## Pelatihan

Pelatihan dimulai dari checkpoint yang dipublikasikan. `pretrained` diterima lalu dibuang
pada ketiga versi, sehingga `pretrained=False` tidak menghasilkan model yang diinisialisasi
secara acak. Seluruh bagian ini membahas deteksi: task berorientasi versi 2 hanya mendukung
inferensi dan tidak ada jalur transfer dari bobot deteksi karena keduanya memakai backbone
yang berbeda.

<code-tabs name="train" />

Learning rate adalah argumen yang harus diatur dengan tepat, dan setiap versi memiliki
nilai default sendiri, bukan nilai umum library. Signature `train()` Python membacanya
dari konfigurasi pelatihan versi tersebut, dan CLI menetapkan nilai yang sama saat `lr0`
tidak diberikan. Versi 1 dan 2 juga menerima `lr_backbone` dengan default seperdua puluh
dari `lr0`, mengikuti resep asli; versi 4 berjalan melalui pelatih D-FINE, yang menskalakan
kelompok parameter backbone dengan `backbone_lr_mult`.

Biarkan `imgsz` pada ukuran native checkpoint kecuali ada alasan untuk mengubahnya.
Validasi dan prediksi pada ukuran lain berfungsi, dengan satu efek sisa: ukuran persegi
panjang yang jumlah tokennya cocok dengan ukuran native tetap menggunakan embedding yang
dibuat untuk rasio aspek yang salah.

Lihat [pelatihan](/docs/train) untuk dataset, augmentasi, multi-GPU, dan logger.

## Validasi

`val()` mengembalikan dictionary dengan key `metrics/` yang mencakup presisi, recall,
mAP 50, dan mAP 50-95, yang diukur terhadap dataset apa pun dalam format yang digunakan
untuk pelatihan.

<code-tabs name="val" />

Baris pada tabel benchmark di atas berasal dari harness benchmark LibreYOLO; catatan di bawah
tabel tersebut menyebutkan dataset yang menghasilkannya dan menautkan catatan proses.

Validasi berorientasi berjalan melalui pemanggilan yang sama dan melaporkan key yang sama,
ditambah empat pengulangan dengan sufiks `(OBB)`. Pencocokan memakai IoU berotasi, bukan IoU
persegi panjang yang mencakupnya, sehingga error sudut dianggap luput. `augment=True`
ditolak pada task ini.

## Ekspor

<export-matrix />

Matriks ini mencakup seluruh garis keturunan dalam satu halaman: jika ketiga versi berbeda
tentang suatu format, sel menampilkan dukungan terlemah dari ketiganya, sehingga tidak ada
kemampuan berlebih yang dijanjikan untuk versi mana pun yang dimuat. Baris berorientasi hanya
milik versi 2. ONNX dan TorchScript telah divalidasi pada task tersebut dengan FP32, batch 1,
dan canvas tetap 1024 kali 1024; OpenVINO, TensorRT, dan ExecuTorch dapat dikonversi serta
dimuat ulang, tetapi belum mencapai kesetaraan output mentah pada seluruh kumpulan query.
Kotak teratas cocok hingga sebagian kecil piksel, sedangkan bagian akhirnya menyimpang.

Artefak hasil ekspor dimuat kembali melalui `LibreYOLO()` berdasarkan sufiks berkasnya,
sehingga berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan mengembalikan
`Results` yang sama.

<code-tabs name="export" />

## Checkpoint

Setiap berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

Nama berkas memuat versi, lalu ukuran, kemudian task. Bobot deteksi adalah
`LibreRTDETR<size>.pt`, `LibreRTDETRv2<size>.pt`, dan `LibreRTDETRv4<size>.pt`, semuanya
pada 640 px. Bobot berorientasi hanya tersedia untuk versi 2 dan menambahkan sufiks task,
mulai `LibreRTDETRv2n-obb.pt` hingga `LibreRTDETRv2x-obb.pt`, semuanya pada 1024 px dan
dilatih pada DOTA v1.0, bukan COCO.

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />

Block di atas adalah yang dipublikasikan penulis untuk deteksi versi 1 dan 2. Bobot
berorientasi versi 2 memiliki upstream ketiga, yaitu repositori RiO-DETR berlisensi
Apache-2.0 di
[github.com/RicePasteM/RiO-DETR](https://github.com/RicePasteM/RiO-DETR), yang menjadi
sumber checkpoint DOTA; kutip proyek tersebut jika menggunakannya. Versi 4 adalah
makalah terpisah dari kelompok berbeda dan memiliki block sitasi sendiri di
[github.com/RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4#4-citation);
kutip makalah itu jika menggunakan checkpoint versi 4.
