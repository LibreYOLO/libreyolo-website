---
title: OWLv2
families:
  - owlv2
seo_title: 'OWLv2 di LibreYOLO: deteksi objek zero-shot'
description: >-
  Gunakan OWLv2 di LibreYOLO untuk mendeteksi objek apa pun yang dijelaskan
  dengan teks. Instal extra openvocab dan lakukan prediksi dengan vocabulary
  teks bebas.
lead: >-
  OWLv2 adalah detector objek open-vocabulary yang dikembangkan Google Research
  dan menilai region gambar terhadap embedding teks dari encoder bergaya CLIP.
  LibreYOLO membungkusnya sebagai family khusus prediksi dalam tingkat detector
  open-vocabulary.
keywords:
  - OWLv2
  - OWL-ViT
  - open-vocabulary object detection
  - zero-shot detection
  - text-conditioned detector
  - LibreOpenVocab
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE

        model = LibreOpenVocab("owlv2-b16")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.1)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Vocabulary default
      language: python
      code: >
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE


        # Melewati set_classes() mempertahankan vocabulary COCO-80 default
        tingkat ini.

        model = LibreOpenVocab("owlv2-l14")

        result = model.predict(SAMPLE_IMAGE, conf=0.1)

        print(result.names)
source_hash: 2d0ce68af0daabb7
---

## Instalasi

OWLv2 dimuat melalui tingkat detector open-vocabulary LibreYOLO yang memerlukan
extra `openvocab`:

```bash
pip install "libreyolo[openvocab]"
```

Extra tersebut memasang `transformers` dan `timm`, yaitu library Hugging Face
yang dipanggil oleh tingkat ini.

## Prediksi

OWLv2 bukan checkpoint yang dimuat LibreYOLO melalui `LibreYOLO()`. Model ini
dimuat melalui factory pendamping `LibreOpenVocab`, yang mengunduh snapshot
Hugging Face saat pertama kali digunakan dan menyimpannya dalam cache di bawah
`weights/`.

<code-tabs name="predict" />

`set_classes()` menetapkan vocabulary teks yang persisten. Panggil lagi untuk
mengganti daftar atau lewati untuk mempertahankan label COCO-80 default. Setiap
label dibungkus dalam template prompt tetap sebelum mencapai text tower, sesuai
cara `Owlv2ForObjectDetection` milik `transformers` dilatih.

OWLv2 tidak memiliki ambang batas token teks. Hanya `conf` yang memfilter
deteksi, dan meneruskan `text_threshold` memunculkan error. `iou` diterima untuk
kompatibilitas API, tetapi mengeluarkan peringatan dan tidak melakukan apa pun
karena tidak ada non-maximum suppression yang dijalankan. `imgsz` dan
`augment=True` langsung ditolak. Processor `transformers` mengendalikan resize,
sedangkan augmentasi waktu pengujian tidak termasuk cakupan tingkat ini.
`predict()` pada satu gambar mengembalikan satu `Results`, bukan daftar. Teruskan
direktori, daftar gambar, atau `stream=True` untuk sumber video agar memperoleh
beberapa hasil. Tidak ada jalur CLI untuk family ini. `libreyolo predict` hanya
memuat checkpoint `.pt` melalui `LibreYOLO()`, sehingga family
`LibreOpenVocab` dijalankan dari Python. Lihat [prediksi](/docs/predict) untuk
jenis sumber dan streaming.

## Varian

Ada dua checkpoint, `b16` (base, ukuran patch 16) dan `l14` (large, ukuran patch
14). `b16` adalah ukuran default tingkat ini jika tidak ada ukuran yang
diberikan. Keduanya mencerminkan rilis resmi Google Research melalui
`Owlv2ForObjectDetection` milik `transformers`, yang diunduh satu kali ke
snapshot Hugging Face yang dihosting LibreYOLO dan tetap mempertahankan berkas
upstream. Belum ada angka akurasi atau latensi yang dipublikasikan untuk family
ini.

Pelatihan, validasi dataset, dan ekspor berada di luar cakupan tingkat ini:
`train()`, `val()`, dan `export()` semuanya selalu memunculkan
`NotImplementedError`. Ini adalah wrapper khusus prediksi untuk checkpoint yang
telah dipublikasikan.

## Checkpoint

Semua berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
