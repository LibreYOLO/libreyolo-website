---
title: OV-DEIM
families:
  - ov_deim
seo_title: 'OV-DEIM di LibreYOLO: deteksi open-vocabulary'
description: >-
  Gunakan OV-DEIM di LibreYOLO untuk deteksi open-vocabulary real-time bergaya
  DETR. Instal extra openvocab dan lakukan prediksi dengan vocabulary teks
  bebas.
lead: >-
  OV-DEIM adalah detector objek open-vocabulary bergaya DETR yang mencocokkan
  query decoder dengan embedding teks dari text tower MobileCLIP bawaan.
  LibreYOLO menyediakan port native sebagai family khusus prediksi dalam tingkat
  detector open-vocabulary.
keywords:
  - OV-DEIM
  - DEIMv2
  - open-vocabulary object detection
  - deteksi real-time
  - zero-shot detection
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("ov-deim-s")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Ganti vocabulary
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("ov-deim-l")
        model.set_classes(["traffic light", "bicycle"])
        first = model.predict(SAMPLE_IMAGE, conf=0.3)

        # Panggilan kedua ke set_classes() mengganti vocabulary sepenuhnya dan
        # membuat embedding ulang melalui text tower. Hasil kosong merupakan
        # hasil valid, bukan error.
        model.set_classes(["giraffe"])
        second = model.predict(SAMPLE_IMAGE, conf=0.5)
        print(second.names, len(second))
source_hash: 0c295f555a9eb303
---

## Instalasi

OV-DEIM dimuat melalui tingkat detector open-vocabulary LibreYOLO yang
memerlukan extra `openvocab`:

```bash
pip install "libreyolo[openvocab]"
```

Tidak seperti family lain dalam tingkat ini, OV-DEIM adalah port native
LibreYOLO, bukan wrapper `transformers`. Tidak ada kelas model `transformers`
untuknya, tetapi extra yang sama mencakup paket `huggingface_hub`,
`safetensors`, `regex`, dan `ftfy` yang diperlukan saat prediksi.

## Prediksi

OV-DEIM bukan checkpoint yang dimuat LibreYOLO melalui `LibreYOLO()`. Model ini
dimuat melalui factory pendamping `LibreOpenVocab`, yang mengunduh snapshot
Hugging Face saat pertama kali digunakan dan menyimpannya dalam cache di bawah
`weights/`.

<code-tabs name="predict" />

`set_classes()` menetapkan vocabulary teks yang persisten. Panggil lagi untuk
mengganti daftar sepenuhnya atau lewati untuk mempertahankan label COCO-80
default. Hasil kosong merupakan hasil valid, bukan error. Setiap query decoder
dinilai berdasarkan cosine similarity terhadap embedding teks dari text tower
MobileCLIP-B(LT) bawaan. Embedding dihitung secara online untuk vocabulary apa
pun yang ditetapkan dan disimpan dalam cache sampai berubah, sehingga prompt
arbitrer bekerja tanpa berkas embedding yang telah dihitung sebelumnya.

OV-DEIM tidak memiliki ambang batas token teks. Hanya `conf` yang memfilter
deteksi, dan meneruskan `text_threshold` memunculkan error. Pencocokan memakai
pemilihan top-K satu-ke-satu, sehingga tidak ada non-maximum suppression yang
dijalankan. `iou` diterima untuk kompatibilitas API, tetapi mengeluarkan
peringatan dan tidak melakukan apa pun. `imgsz` dan `augment=True` langsung
ditolak. Model mengendalikan input letterbox tetap, sedangkan augmentasi waktu
pengujian tidak termasuk cakupan tingkat ini. `predict()` pada satu gambar
mengembalikan satu `Results`, bukan daftar. Teruskan direktori, daftar gambar,
atau `stream=True` untuk sumber video agar memperoleh beberapa hasil. Tidak ada
jalur CLI untuk family ini. `libreyolo predict` hanya memuat checkpoint `.pt`
melalui `LibreYOLO()`, sehingga family `LibreOpenVocab` dijalankan dari Python.
Lihat [prediksi](/docs/predict) untuk jenis sumber dan streaming.

Setiap panggilan `predict()` juga menjalankan text tower MobileCLIP-B(LT) bawaan
untuk membuat embedding vocabulary saat ini. Lihat Lisensi untuk ketentuan
tambahan yang ditimbulkannya.

## Varian

Ada tiga checkpoint, `s`, `m`, dan `l`. `s` adalah ukuran default tingkat ini
jika tidak ada ukuran yang diberikan. Tidak seperti family lain dalam tingkat
ini, OV-DEIM adalah port native, bukan wrapper `transformers`. LibreYOLO
menyertakan modul detector dengan lisensi Apache-2.0 yang sama seperti kode
upstream dan menggunakan kembali adapter backbone DINOv3 yang telah dibuat
untuk family DEIMv2. Backbone checkpoint `l` adalah hasil fine-tuning DINOv3-S
yang dilisensikan terpisah berdasarkan DINOv3 License milik Meta. Belum ada
angka akurasi atau latensi yang dipublikasikan untuk family ini.

Pelatihan, validasi dataset, dan ekspor berada di luar cakupan tingkat ini:
`train()`, `val()`, dan `export()` semuanya selalu memunculkan
`NotImplementedError`. Ini adalah wrapper khusus prediksi untuk checkpoint yang
telah dipublikasikan.

## Checkpoint

Semua berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box>

OV-DEIM menambahkan tiga lisensi upstream pada setiap panggilan prediksi: bobot
detector di bawah CC BY-NC 4.0 milik OV-DEIM, text tower online di bawah Apple
Machine Learning Research Model license (hanya untuk penelitian), dan untuk
checkpoint `l`, hasil fine-tuning backbone DINOv3-S di bawah DINOv3 License
milik Meta. Ketiga teks lisensi disertakan dalam repository bobot LibreYOLO.

</provenance-box>

## Sitasi

<citation-block />
