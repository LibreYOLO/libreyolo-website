---
title: Deteksi open-vocabulary
seo_title: Deteksi open-vocabulary di LibreYOLO
description: >-
  Deteksi objek dari vocabulary teks di LibreYOLO. Muat Grounding DINO, OWLv2,
  OMDet-Turbo, atau OV-DEIM melalui LibreOpenVocab dan tetapkan kelas saat
  runtime.
lead: >-
  Deteksi open-vocabulary mengganti list kelas tetap checkpoint dengan kata yang
  dipilih saat pemanggilan. Di LibreYOLO, ini bukan task terpisah: ini adalah
  task detect yang dilayani tier model terpisah, dimuat melalui factory
  LibreOpenVocab, bukan LibreYOLO.
keywords:
  - deteksi open vocabulary
  - deteksi objek zero shot
  - open set detection
  - grounding dino python
  - owlv2
  - omdet turbo
  - deteksi prompt teks
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("grounding-dino-t")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        print(result.names)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Ganti vocabulary
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("owlv2-b16")

        # set_classes bersifat persisten hingga pemanggilan berikutnya.
        # Label harus unik setelah diubah ke huruf kecil dan artikel dihapus.
        model.set_classes(["a red backpack", "traffic cone"])
        result = model.predict(SAMPLE_IMAGE)

        model.set_classes(["bicycle wheel"])
        result = model.predict(SAMPLE_IMAGE)
    - label: Ambang batas teks Grounding DINO
      language: python
      code: >
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE


        model = LibreOpenVocab("grounding-dino-b")

        model.set_classes(["remote control", "school bus"])


        # conf memfilter berdasarkan score box, text_threshold berdasarkan score
        token

        # frasa hasil decode. Default keduanya 0.25 jika tidak ditetapkan. Hanya
        Grounding

        # DINO yang menerima text_threshold; lainnya memunculkan error.

        result = model.predict(SAMPLE_IMAGE, conf=0.25, text_threshold=0.3)
source_hash: 17197cf4d80f3d6f
---

## Definisi

Deteksi open-vocabulary mengembalikan `Results` deteksi biasa: box, confidence,
dan indeks kelas, dengan `result.names` memetakan indeks ke string yang diminta.
Perubahannya adalah sumber list kelas. Detektor konvensional dilatih terhadap
kumpulan kategori tetap dan tidak pernah menghasilkan kategori di luar
kumpulan tersebut. Model ini menerima vocabulary sebagai teks saat inferensi,
sehingga `set_classes(["forklift", "safety cone"])` cukup untuk menjadikannya
kelas.

LibreYOLO tidak memiliki key task `open-vocabulary`. Model ini mendeklarasikan
`SUPPORTED_TASKS = ("detect",)` seperti detektor lain. Pemisahnya adalah jalur
pemuatan: model berupa snapshot Hugging Face, bukan checkpoint state dict
LibreYOLO, sehingga tidak masuk factory `LibreYOLO()` dan dibuat melalui
`LibreOpenVocab()`. Factory tersebut adalah sibling `LibreSAM()` dan
`LibreVLM()`, bukan pengganti `LibreYOLO()`.

Score merupakan score deteksi nyata, bukan caption hasil generasi yang di-parse.
Setiap family menilai region gambar terhadap embedding teks setiap prompt.

## Model

Empat family membentuk tier ini dan semuanya hanya memprediksi. Muat family
berdasarkan alias melalui `LibreOpenVocab`.

[Grounding DINO](/docs/models/grounding-dino) dari IDEA Research, dalam ukuran
`t` dan `b`. Ini adalah default tier dan satu-satunya family yang menerima
`text_threshold`, cutoff kedua untuk score token frasa hasil decode.

[OWLv2](/docs/models/owlv2) dari Google Research, dalam ukuran `b16` dan `l14`.
Model menilai region gambar terhadap embedding teks dari encoder bergaya CLIP.

[OMDet-Turbo](/docs/models/omdet-turbo) dari Om AI Lab, dalam satu ukuran `t`.
Model memisahkan embedding kelas dari prompt task bahasa dan merupakan satu-
satunya family di sini yang menekan box tumpang tindih dalam postprocessing,
sehingga `iou=` dipatuhi.

[OV-DEIM](/docs/models/ov-deim), dalam ukuran `s`, `m`, dan `l`, adalah detektor
bergaya DETR yang mencocokkan query decoder terhadap embedding teks dari text
tower MobileCLIP bawaan. Model memakai pencocokan one-to-one dengan pemilihan
top-K, sehingga NMS tidak dijalankan.

Bobot OV-DEIM adalah kasus terbatas dalam tier ini. Bobot detektor berlisensi
CC BY-NC 4.0 dan nonkomersial. Text tower bawaan berada di bawah Apple's Machine
Learning Research Model license, khusus penggunaan penelitian. Checkpoint `l`
menambahkan fine-tuning backbone DINOv3-S di bawah DINOv3 License milik Meta.
Ketiga teks lisensi tersedia dalam repository bobot, dan library mencatat
ringkasan yang sama ketika me-resolve bobot, sebelum model dibangun. Baca
[OV-DEIM](/docs/models/ov-deim) sebelum deployment.

Tier memerlukan satu ekstra:

```bash
pip install "libreyolo[openvocab]"
```

Ekstra mencakup `transformers` dan `timm` untuk tiga family wrapper, serta
`huggingface_hub`, `safetensors`, `regex`, dan `ftfy` yang diperlukan OV-DEIM
sebagai port native.

Tier kedua juga menerima vocabulary teks: `LibreVLM()` memuat vision-language
model generatif seperti [Qwen3-VL](/docs/models/qwen3-vl) dan
[Florence-2](/docs/models/florence-2), lalu mengubah output-nya menjadi
`Results` yang sama. Tier tersebut berbagi antarmuka `set_classes()`.
Perbedaannya adalah penghasil box: family halaman ini berupa detektor
diskriminatif yang langsung menghasilkan score, sedangkan tier VLM melakukan
generasi.

## Predict

<code-tabs name="predict" />

`set_classes()` menerima list string label yang tidak kosong dan bertahan
hingga dipanggil kembali. Label harus unik setelah diubah ke huruf kecil dan
artikel di awal dihapus, sehingga `"a bus"` dan `"bus"` tidak dapat berada dalam
satu vocabulary. Frasa multi-kata merupakan label biasa, dan setiap family
mengubah list menjadi input teksnya sendiri sebelum tokenization, sehingga
`"traffic cone"` adalah query berbeda dari `"cone"`.

Tiga argumen prediksi berperilaku berbeda dari detektor native. `imgsz=` ditolak
karena processor mengendalikan resize. `augment=True` ditolak karena augmentasi
saat pengujian berada di luar cakupan tier. `iou=` hanya berlaku pada family
yang processor-nya menjalankan suppression; jika tidak ada suppression,
memberikannya menghasilkan peringatan dan diabaikan.

Jika tidak ditetapkan, `conf` mengambil default milik family yang dimuat, bukan
default umum `predict()` 0.25, dan nilainya berbeda dalam tier. Tetapkan secara
eksplisit saat membandingkan dua family pada gambar sama.

`track()` memunculkan error di seluruh tier. Jalankan `predict()` per frame.
Lihat [prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Train

Tidak ada family dalam tier yang berlatih di LibreYOLO. `train()` memunculkan
error: lakukan fine-tuning di upstream dan muat bobot hasilnya. Vocabulary pada
`set_classes()` adalah satu-satunya pengaturan yang mengubah yang dideteksi
model termuat.

## Validate

Tidak ada validator untuk tier ini dan `val()` memunculkan error. Validasi
open-vocabulary memerlukan validator khusus karena validator deteksi standar
memberikan tensor gambar langsung kepada model, sedangkan family ini memerlukan
input yang dikondisikan teks.

## Export

Ekspor berada di luar cakupan tier dan `export()` memunculkan error. Model
berjalan melalui `predict()` di PyTorch.
