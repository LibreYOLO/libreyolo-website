---
title: Grounding DINO
families:
  - grounding_dino
seo_title: 'Grounding DINO di LibreYOLO: deteksi open-set'
description: >-
  Gunakan Grounding DINO di LibreYOLO untuk mendeteksi objek apa pun yang
  dijelaskan dengan teks. Instal extra openvocab dan lakukan prediksi dengan
  vocabulary teks bebas.
lead: >-
  Grounding DINO adalah detector objek open-set yang dikembangkan IDEA Research.
  Model ini menilai gambar terhadap prompt teks bebas, bukan daftar kelas tetap.
  LibreYOLO membungkusnya sebagai family khusus prediksi dalam tingkat detector
  open-vocabulary.
keywords:
  - Grounding DINO
  - open-vocabulary object detection
  - open-set detection
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

        model = LibreOpenVocab("grounding-dino-t")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.25)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Ambang batas teks
      language: python
      code: >
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE


        model = LibreOpenVocab("grounding-dino-b")

        model.set_classes(["remote control", "school bus"])


        # conf memfilter berdasarkan skor box, sedangkan text_threshold
        berdasarkan skor token

        # frasa yang didekode. Keduanya memakai nilai default 0.25 jika tidak
        ditetapkan.

        result = model.predict(SAMPLE_IMAGE, conf=0.25, text_threshold=0.3)

        print(result.names)
source_hash: 06bd13b8e6a66038
---

## Instalasi

Grounding DINO dimuat melalui tingkat detector open-vocabulary LibreYOLO yang
memerlukan extra `openvocab`:

```bash
pip install "libreyolo[openvocab]"
```

Extra tersebut memasang `transformers` dan `timm`, yaitu library Hugging Face
yang dipanggil oleh tingkat ini.

## Prediksi

Grounding DINO bukan checkpoint yang dimuat LibreYOLO melalui `LibreYOLO()`.
Model ini dimuat melalui factory pendamping `LibreOpenVocab`, yang mengunduh
snapshot Hugging Face saat pertama kali digunakan dan menyimpannya dalam cache
di bawah `weights/`.

<code-tabs name="predict" />

`set_classes()` menetapkan vocabulary teks yang persisten. Panggil lagi untuk
mengganti daftar atau lewati untuk mempertahankan label COCO-80 default.
Grounding DINO mendekode frasa bebas dari output teksnya sendiri dan memetakan
frasa itu kembali ke vocabulary secara mandiri. Kecocokan ternormalisasi yang
persis diprioritaskan, kecocokan token utuh diterima, sedangkan frasa ambigu
atau tidak cocok dibuang alih-alih ditebak. Karena itu, `school bus` tidak
pernah dipetakan hanya ke `bus` atau `school`. Vocabulary yang cukup panjang
hingga melampaui batas token encoder teks dibagi menjadi beberapa prompt,
dijalankan sebagai forward pass terpisah, lalu digabungkan kembali menjadi
satu set deteksi yang dibatasi oleh `max_det`.

`iou` diterima untuk kompatibilitas API, tetapi mengeluarkan peringatan dan
tidak melakukan apa pun karena tidak ada non-maximum suppression yang
dijalankan. `imgsz` dan `augment=True` langsung ditolak. Processor
`transformers` mengendalikan resize, sedangkan augmentasi waktu pengujian tidak
termasuk cakupan tingkat ini. `predict()` pada satu gambar mengembalikan satu
`Results`, bukan daftar. Teruskan direktori, daftar gambar, atau `stream=True`
untuk sumber video agar memperoleh beberapa hasil. Tidak ada jalur CLI untuk
family ini. `libreyolo predict` hanya memuat checkpoint `.pt` melalui
`LibreYOLO()`, sehingga family `LibreOpenVocab` dijalankan dari Python. Lihat
[prediksi](/docs/predict) untuk jenis sumber dan streaming.

## Varian

Ada dua checkpoint, `t` dan `b`. `t` adalah ukuran default tingkat ini bila
tidak ada ukuran yang diberikan. Keduanya mencerminkan rilis resmi IDEA
Research melalui `GroundingDinoForObjectDetection` milik `transformers`, yang
diunduh satu kali ke snapshot Hugging Face yang dihosting LibreYOLO dan tetap
mempertahankan berkas upstream. Belum ada angka akurasi atau latensi yang
dipublikasikan untuk family ini.

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
