---
title: D-FINE
families:
  - dfine
seo_title: 'D-FINE: fine-tuning, validasi dan ekspor dengan lisensi MIT'
description: >-
  Pakai D-FINE di LibreYOLO untuk deteksi objek dan segmentasi instance.
  Instalasi, prediksi, fine-tuning, validasi dan ekspor, dengan kode berlisensi
  MIT.
lead: >-
  Detection transformer yang merumuskan ulang regresi box sebagai distribusi
  probabilitas atas setiap sisi box, lalu menyempurnakannya di sepanjang lapisan
  decoder. LibreYOLO mendukungnya untuk deteksi dan segmentasi instance.
keywords:
  - D-FINE
  - detection transformer
  - deteksi objek python
  - real-time object detection
  - instance segmentation
  - DETR
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDFINEn.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDFINEn.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Segmentasi instance
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Sufiks -seg pada nama berkas memilih mask head, jadi argumen task
        # tidak diperlukan di sini.
        model = LibreYOLO("LibreDFINEn-seg.pt")
        result = model(SAMPLE_IMAGE, save=True)

        print(result.masks.data.shape)
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreDFINEn.pt")

        model.train(data="my-dataset.yaml", epochs=50, imgsz=640, batch=8,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          epochs=50 imgsz=640 batch=8 lr0=2e-4
    - label: Segmentasi instance
      language: bash
      code: >
        # Melanjutkan dari bobot segmentasi yang dipublikasikan, mask head ikut
        serta.

        libreyolo train model=LibreDFINEn-seg.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
    - label: Segmentasi dari bobot detect
      language: bash
      code: |
        # Bobot detect tidak membawa mask head, jadi ini transfer eksplisit:
        # head-nya mulai tanpa pelatihan dan baru berguna setelah dilatih.
        # Meminta task=segment di sini yang mengizinkan transfer tersebut.
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          task=segment epochs=50 imgsz=640
    - label: LoRA
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreDFINEn.pt data=my-dataset.yaml \
          epochs=50 device=0,1 batch=16
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDFINEn.pt data=my-dataset.yaml
    - label: Segmentasi instance
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn-seg.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95(M)"])   # mask
        print(metrics["metrics/mAP50-95(B)"])   # box
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDFINEn.pt")
        model.export(format="onnx", imgsz=640)
        model.export(format="tensorrt", imgsz=640, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDFINEn.pt format=onnx imgsz=640

        libreyolo export model=LibreDFINEn.pt format=tensorrt imgsz=640
        half=True
    - label: Memakai berkas hasil ekspor
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Factory memilih rute lewat sufiks berkas, jadi artefak hasil ekspor
        # dimuat seperti checkpoint biasa dan mengembalikan Results yang sama.
        model = LibreYOLO("LibreDFINEn.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.boxes.xyxy)
source_hash: 0216631a26185524
---

## Instalasi

D-FINE tidak memerlukan extra opsional. Semua yang diimpornya sudah ada di
instalasi dasar.

```bash
pip install libreyolo
```

Fine-tuning berbasis adapter dengan `lora=True` adalah pengecualian, dan
memerlukan extra `lora`.

```bash
pip install "libreyolo[lora]"
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali dipakai dan disimpan di cache
lokal.

<code-tabs name="predict" />

Objek `Results` yang dikembalikan sama dengan yang dikembalikan setiap family,
jadi mengganti detektor cukup lewat perubahan satu baris. Nama berkas dengan
`-seg` mengarah ke task segmentasi dengan sendirinya, dan `result.masks` lalu
membawa mask instance berdampingan dengan box. `conf` dan `max_det` menyaring
pemilihan query; `iou` diterima demi keseragaman API tetapi tidak berpengaruh,
karena decoder-nya adalah set predictor tanpa langkah NMS. Lihat
[prediksi](/docs/predict) untuk sumber, streaming dan penanganan hasil.

## Varian

Lima ukuran. Semuanya berjalan pada resolusi input yang sama, jadi tabel
membedakannya lewat jumlah parameter dan akurasi.

<benchmark-table task="detect" />

<va-embed />

Segmentasi memakai ulang backbone, encoder dan decoder deteksi lalu menambahkan
mask head, jadi checkpoint `-seg` menerima argumen yang sama dengan saudaranya
yang detect. Family RT-DETRv4 di LibreYOLO ditulis sebagai subclass dari wrapper
D-FINE: ia mewarisi jalur decoder ini lalu mengunci daftar task-nya kembali ke
deteksi, karena tidak membawa mask head.

## Pelatihan

Pelatihan dimulai dari checkpoint yang dipublikasikan, untuk kedua task.

<code-tabs name="train" />

Jika dibiarkan dengan nilai bawaan, trainer menjalankan 132 epoch pada
`lr0=2e-4` dengan `amp=False`, batch 16 dan early stopping setelah 50 epoch
tanpa perbaikan. Bobot detect adalah titik awal yang sah untuk pelatihan
segmentasi, tetapi hanya sebagai transfer eksplisit, karena mask head-nya mulai
tanpa pelatihan dan kalau tidak akan mengembalikan mask yang tidak bermakna.
Memberikan `task=segment` ke CLI itulah yang mengizinkannya. Jalur Python lebih
sempit: `LibreDFINE` harus dikonstruksi langsung dengan
`allow_detect_to_segment_transfer=True`, karena factory `LibreYOLO()` tidak
menerima argumen semacam itu, dan konstruksi langsung tidak melakukan
pengunduhan, jadi berkas bobotnya harus sudah ada di disk.

`lora=True` berlaku untuk deteksi. Pelatihan segment menolaknya dan mengarahkan
ke `freeze='backbone'`, karena mask head belum diuji dengan adapter. Di Apple
silicon trainer memindahkan seluruh proses ke CPU: backward pass dari binned
matmul pada modul Integral memicu kegagalan kompilasi Metal. Inferensi di MPS
tidak terpengaruh.

Lihat [pelatihan](/docs/train) untuk dataset, augmentasi, multi-GPU dan logger.

## Validasi

`val()` mengembalikan dictionary dengan kunci berupa nama metrik, dan mencetak
hasil per kelas jika `verbose` dibiarkan aktif.

<code-tabs name="val" />

Pada checkpoint `-seg`, kunci `metrics/mAP50-95` yang polos berisi skor mask,
dan proses yang sama juga melaporkan box di bawah `(B)` dan mask di bawah `(M)`
sehingga keduanya tersedia dari satu kali jalan.

## Ekspor

<export-matrix />

Artefak hasil ekspor dimuat kembali lewat `LibreYOLO()` berdasarkan sufiks
berkasnya, jadi berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan
mengembalikan `Results` yang sama. Jalur OpenVINO, Paddle, MNN dan Core AI
mengekspor pada kanvas tetap, bukan bentuk dinamis. [Ekspor](/docs/export)
memuat daftar argumen yang diterima setiap format dan tambahan yang dipakai
beberapa di antaranya.

<code-tabs name="export" />

## Checkpoint

Semua berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box>

Bobot segmentasi punya upstream kedua: mask decoder, mask matching dan mask
loss-nya berasal dari ArgoHA/D-FINE-seg, juga Apache-2.0, dan maintainer-nya
menyetujui penggunaan ulang dengan atribusi.

</provenance-box>

## Sitasi

<citation-block />
