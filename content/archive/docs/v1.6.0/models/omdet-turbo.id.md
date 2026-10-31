---
title: OMDet-Turbo
families:
  - omdet_turbo
seo_title: 'OMDet-Turbo di LibreYOLO: deteksi zero-shot real-time'
description: >-
  Gunakan OMDet-Turbo di LibreYOLO untuk deteksi open-vocabulary real-time.
  Instal extra openvocab dan lakukan prediksi dengan vocabulary teks bebas.
lead: >-
  OMDet-Turbo adalah detector objek open-vocabulary real-time yang dikembangkan
  Om AI Lab dan memisahkan embedding kelas dari prompt task bahasa. LibreYOLO
  membungkusnya sebagai family khusus prediksi dalam tingkat detector
  open-vocabulary.
keywords:
  - OMDet-Turbo
  - OmDet
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

        model = LibreOpenVocab("omdet-turbo")
        model.set_classes(["person", "dog", "skateboard"])

        result = model.predict(SAMPLE_IMAGE, conf=0.3)
        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Ambang batas NMS khusus
      language: python
      code: >
        from libreyolo import LibreOpenVocab, SAMPLE_IMAGE


        model = LibreOpenVocab("omdet-turbo")

        model.set_classes(["traffic light", "bicycle"])


        # OMDet-Turbo adalah satu-satunya family dalam tingkat ini yang
        menghormati iou=:

        # post-processing miliknya menerima ambang batas suppression sebagai
        argumen,

        # dengan nilai default 0.5 jika iou= tidak ditetapkan.

        result = model.predict(SAMPLE_IMAGE, conf=0.3, iou=0.7)

        print(result.names, len(result))
source_hash: c2a375d234341b7e
---

## Instalasi

OMDet-Turbo dimuat melalui tingkat detector open-vocabulary LibreYOLO yang
memerlukan extra `openvocab`:

```bash
pip install "libreyolo[openvocab]"
```

Extra tersebut memasang `transformers` dan `timm`, yaitu library Hugging Face
yang dipanggil tingkat ini. Backbone Swin OMDet-Turbo dimuat melalui wrapper
`TimmBackbone` milik `transformers`.

## Prediksi

OMDet-Turbo bukan checkpoint yang dimuat LibreYOLO melalui `LibreYOLO()`. Model
ini dimuat melalui factory pendamping `LibreOpenVocab`, yang mengunduh snapshot
Hugging Face saat pertama kali digunakan dan menyimpannya dalam cache di bawah
`weights/`.

<code-tabs name="predict" />

`set_classes()` menetapkan vocabulary teks yang persisten. Panggil lagi untuk
mengganti daftar sepenuhnya atau lewati untuk mempertahankan label COCO-80
default. Hasil kosong merupakan hasil valid, bukan error. Tidak seperti
Grounding DINO, OMDet-Turbo memisahkan embedding kelas dari prompt task bahasa,
sehingga post-processing `transformers` mengembalikan label yang langsung
dipetakan ke daftar kelas yang diminta tanpa langkah disambiguasi frasa.

OMDet-Turbo tidak memiliki ambang batas token teks. Hanya `conf` yang memfilter
deteksi, dan meneruskan `text_threshold` memunculkan error. Ini adalah
satu-satunya family dalam tingkat ini yang menjalankan non-maximum suppression
sendiri di dalam `post_process_grounded_object_detection`, sehingga `iou`
dihormati di sini dan tidak sekadar memunculkan peringatan. `imgsz` dan
`augment=True` langsung ditolak. Processor `transformers` mengendalikan resize,
sedangkan augmentasi waktu pengujian tidak termasuk cakupan tingkat ini.
`predict()` pada satu gambar mengembalikan satu `Results`, bukan daftar. Teruskan
direktori, daftar gambar, atau `stream=True` untuk sumber video agar memperoleh
beberapa hasil. Tidak ada jalur CLI untuk family ini. `libreyolo predict` hanya
memuat checkpoint `.pt` melalui `LibreYOLO()`, sehingga family
`LibreOpenVocab` dijalankan dari Python. Lihat [prediksi](/docs/predict) untuk
jenis sumber dan streaming.

## Varian

Ada satu checkpoint, `t`, satu-satunya ukuran tingkat ini. Checkpoint tersebut
mencerminkan `omlab/omdet-turbo-swin-tiny-hf` pada revisi upstream yang dipatok
melalui `OmDetTurboForObjectDetection` milik `transformers`. Berkas bobot yang
dicerminkan byte-identik dengan snapshot upstream. Belum ada angka akurasi atau
latensi yang dipublikasikan untuk family ini.

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
