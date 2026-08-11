---
title: PP-OCRv5
families:
  - ppocr
seo_title: 'PP-OCRv5: deteksi dan pengenalan teks di LibreYOLO'
description: >-
  Gunakan PP-OCRv5 di LibreYOLO untuk OCR teks dalam adegan multibahasa. Instal,
  prediksi, dan validasi checkpoint t dan l berlisensi Apache-2.0.
lead: >-
  PP-OCRv5 adalah pipeline deteksi dan pengenalan teks milik PaddleOCR: detektor
  binarisasi terdiferensiasi menemukan kuadrilateral teks dan recognizer
  SVTR/CTC membacanya. LibreYOLO mem-porting-nya ke PyTorch dalam dua tier.
keywords:
  - PP-OCRv5
  - PaddleOCR
  - OCR
  - deteksi teks
  - pengenalan teks
  - OCR tulisan di gambar
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for text, conf in zip(result.ocr.texts, result.ocr.conf):
            print(text, float(conf))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibrePPOCRl-ocr.pt source=receipt.jpg save=True
    - label: Kuadrilateral
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        result = model(SAMPLE_IMAGE)

        # Poligon (N, 4, 2) dalam urutan baca: kiri atas, kanan atas,
        # kanan bawah, kiri bawah. Kuadrilateral deteksi adalah poligon asli
        # (teks berotasi), sehingga mengisi result.ocr, bukan result.boxes.
        print(result.ocr.data.shape)
        print(result.ocr.det_conf)
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPOCRl-ocr.pt")
        metrics = model.val(data="my-dataset")

        print(metrics["metrics/det_hmean"])
        print(metrics["metrics/e2e_f1"])       # metrik utama
        print(metrics["metrics/rec_1-NED"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibrePPOCRl-ocr.pt data=my-dataset
source_hash: 9835057f8bd95bc1
---

## Instalasi

PP-OCRv5 tidak memerlukan komponen tambahan selain paket dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam cache lokal.

<code-tabs name="predict" />

Setiap checkpoint menyertakan kedua tahap, deteksi dan pengenalan, dalam satu
berkas `.pt`, sedangkan set karakter pengenalan dan nilai default pipeline disimpan
dalam metadata checkpoint. Recognizer membaca bahasa Tionghoa Sederhana dan Tradisional,
Inggris, Jepang, serta pinyin dengan satu kamus. `result.ocr` adalah payload
`OCRRegions`: `.data` menyimpan poligon empat titik, `.texts` menyimpan
transkrip, `.conf` menyimpan skor pengenalan per wilayah, dan
`.det_conf` menyimpan skor deteksi. Sumber multi-gambar dijalankan secara berurutan:
pipeline dua tahap ini tidak melakukan batch lintas gambar. Lihat
[prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Varian

Dua tier: `t`, dibangun pada backbone PP-LCNetV3/PP-OCRv5_mobile yang lebih ringan
untuk penggunaan CPU, dan `l`, dibangun pada backbone server PP-HGNetV2 untuk akurasi
lebih tinggi. Kedua tier menjalankan deteksi dengan batas sisi panjang yang tetap dan
mengenali crop dalam batch; `rec_batch` mengontrol jumlah crop yang melewati recognizer
per forward pass.

## Validasi

`val()` mengukur pipeline terhadap direktori gambar beserta berkas
`labels/<split>.jsonl`, atau YAML dataset yang setara. Setiap label mencantumkan
poligon wilayah teks per gambar dan transkripnya. Metode ini melaporkan hmean deteksi
(presisi/recall/F1 yang dicocokkan dengan IoU), F1 end-to-end (hmean ditambah kecocokan
transkrip persis setelah normalisasi, yaitu metrik fitness checkpoint), serta 1-NED,
jarak edit ternormalisasi rata-rata pada pasangan yang cocok.

<code-tabs name="val" />

## Ekspor

<export-matrix />

PP-OCRv5 adalah pipeline dua jaringan, yaitu deteksi dan pengenalan yang bergerak
bersama, bukan satu graph yang dapat di-trace. Ekspor belum diimplementasikan untuknya:
belum ada format yang didukung. Lakukan fine-tuning langsung pada kode pelatihan upstream
berlisensi Apache-2.0 dan konversi hasilnya dengan `weights/convert_ppocr_weights.py`
jika memerlukan checkpoint di luar format ini.

## Checkpoint

Setiap berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
