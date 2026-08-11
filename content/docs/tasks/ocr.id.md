---
title: OCR
seo_title: 'OCR: deteksi dan pengenalan teks di LibreYOLO'
description: >-
  Temukan dan baca teks dalam gambar dengan LibreYOLO. Prediksi quad dan
  transkrip, beri label dataset JSONL, lalu validasi dengan hmean, F1
  end-to-end, dan 1-NED.
lead: >-
  OCR melokalisasi teks dalam gambar dan membacanya. LibreYOLO menyediakannya
  sebagai task ocr, yang mengembalikan satu poligon empat titik ditambah satu
  transkrip per region teks dalam urutan baca.
keywords:
  - library ocr python
  - pengenalan teks scene
  - text detection quad
  - PP-OCRv5 python
  - text spotting end-to-end
last_verified: 1.5.0
snippets:
  predict:
    - label: Baca teks dalam gambar
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Tier t adalah yang lebih ringan dan dibuat untuk CPU. SAMPLE_IMAGE
        # menjaga contoh dapat dijalankan; arahkan ke gambar berisi teks sendiri.
        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        result = model(SAMPLE_IMAGE)

        regions = result.ocr
        print(len(regions), "regions")
        for text, score in zip(regions.texts, regions.conf):
            print(repr(text), float(score))
    - label: Baca quad
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        result = model(SAMPLE_IMAGE)

        regions = result.ocr
        print(regions.data.shape)   # poligon (N, 4, 2), TL TR BR BL
        print(regions.xyxy)         # hull sejajar sumbu dari poligon
        print(regions.det_conf)     # skor deteksi, terpisah dari .conf
    - label: Filter berdasarkan confidence pengenalan
      language: python
      code: |
        import numpy as np
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        result = model(SAMPLE_IMAGE)

        # Lakukan indexing dengan posisi, bukan boolean mask: slicing membawa
        # transkrip dan kedua array skor bersama geometri.
        regions = result.ocr.numpy()
        keep = regions[np.flatnonzero(regions.conf >= 0.9)]
        print(keep.texts)
  val:
    - label: Validasi dan baca kunci metrik
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibrePPOCRt-ocr.pt")
        metrics = model.val(data="my-ocr-dataset")

        print(metrics["metrics/det_precision"], metrics["metrics/det_recall"])
        print(metrics["metrics/det_hmean"])
        print(metrics["metrics/e2e_f1"])       # fitness
        print(metrics["metrics/rec_1-NED"])
source_hash: 58ad5305c9dd458c
---

## Definisi

Task `ocr` melakukan dua hal dalam satu pemanggilan: melokalisasi setiap region
teks dalam gambar dan mentranskripsikannya. Region dikembalikan sebagai poligon
empat titik, bukan bounding box sejajar sumbu, karena teks scene sering berotasi, serta
dalam urutan baca dari atas ke bawah lalu kiri ke kanan.

Prediksi mengisi `result.ocr`, payload `OCRRegions`. `.data` adalah array float
poligon `(N, 4, 2)` dalam piksel gambar asli, berurutan kiri atas, kanan atas,
kanan bawah, kiri bawah; `.texts` adalah list N transkrip; `.conf` adalah skor
pengenalan per region dan `.det_conf` skor deteksi; `.xyxy` memberikan hull
sejajar sumbu setiap poligon. Karena quad merupakan poligon sebenarnya,
`result.boxes` tidak diisi. Slicing `OCRRegions` membawa transkrip dan kedua
array skor bersama geometri.

## Model

Dua family melayani `ocr`.

[PP-OCRv5](/docs/models/pp-ocrv5) adalah pipeline khusus: detektor
differentiable-binarization menemukan quad teks dan recognizer SVTR/CTC
membacanya, dengan kedua tahap disatukan dalam satu berkas `.pt` beserta charset
pengenalan. Model tersedia dalam dua tier, satu lebih ringan untuk CPU dan satu
server untuk akurasi lebih tinggi, serta satu dictionary mencakup Mandarin
Sederhana dan Tradisional, Inggris, Jepang, dan pinyin.

[SenseNova-Vision](/docs/models/sensenova-vision) menangani OCR dengan
menghasilkan kata sebagai teks bertag dari checkpoint 7B yang sama dengan enam
task lain, dimuat melalui `LibreVLM("sensenova-vision", task="ocr")`. Model
memerlukan ekstra `sensenova`, dan bobotnya terbatas untuk penggunaan
nonkomersial; lisensi tersedia pada halamannya.

## Prediksi

Bobot diunduh dari Hugging Face saat penggunaan pertama dan disimpan dalam cache secara
lokal.

<code-tabs name="predict" />

PP-OCRv5 menjalankan deteksi pada batas sisi panjang tetap lalu mengenali region
hasil crop dalam batch, dengan `rec_batch` mengatur jumlah crop per forward pass
recognizer. Sumber multi-gambar berjalan berurutan karena pipeline dua tahap
tidak melakukan batching lintas gambar. Lihat [prediksi](/docs/predict) untuk
sumber, streaming, dan penanganan hasil.

## Format dataset

Label OCR berupa satu berkas JSONL per split, satu objek JSON per gambar, di
samping gambar.

```text
my-ocr-dataset/
  images/
    val/receipt.jpg
  labels/
    val.jsonl
```

Setiap baris menamai gambar dan mencantumkan region-nya:

```json
{"image": "receipt.jpg", "regions": [{"polygon": [[10, 12], [118, 14], [117, 40], [9, 38]], "text": "TOTAL 12.50"}]}
```

`polygon` adalah quad empat titik dalam koordinat piksel absolut, berurutan kiri
atas, kanan atas, kanan bawah, kiri bawah. Region dengan teks tidak terbaca diberi
label `"text": "###"`, konvensi don't-care ICDAR: region dikecualikan dari
penilaian pengenalan, dan prediksi yang tumpang tindih diabaikan, bukan dianggap
false positive.

Memberikan root direktori sebagai `data=` sudah cukup. Alternatifnya adalah YAML
dataset dengan `path` serta nama direktori `images` dan `labels` opsional, lalu
`nc: 1` bersama `names: {0: text}` sebagai placeholder skema karena model OCR
mengembalikan `Results.ocr`, bukan deteksi. Lihat
[format dataset](/docs/reference/dataset-formats) untuk kontrak lengkap.

## Pelatihan

Kedua family OCR tidak memiliki implementasi pelatihan: `train()` memunculkan
`NotImplementedError`, dan dukungan OCR mencakup prediksi serta validasi.
Halaman PP-OCRv5 menyebut kode pelatihan upstream Apache-2.0 dan script konversi
untuk membawa checkpoint hasil fine-tuning kembali ke LibreYOLO.

## Validasi

`val()` menilai seluruh pipeline, deteksi dan pengenalan bersama, dengan
mencocokkan poligon prediksi ke ground truth secara one-to-one pada IoU di atas
0.5.

<code-tabs name="val" />

`metrics/det_precision`, `metrics/det_recall`, dan `metrics/det_hmean` hanya
menilai lokalisasi: kecocokan hanya memerlukan tumpang tindih poligon, apa pun
transkripnya. `metrics/e2e_precision`, `metrics/e2e_recall`, dan
`metrics/e2e_f1` menambahkan pembacaan: kecocokan memerlukan tumpang tindih yang
sama dan transkrip persis setelah normalisasi NFKC serta penghapusan whitespace,
dengan kapitalisasi tetap diperhitungkan. `metrics/e2e_f1` juga menjadi
`fitness`, angka yang digunakan pemilihan checkpoint terbaik.

`metrics/rec_1-NED` menilai recognizer sendiri pada pasangan yang sudah cocok
dalam deteksi: satu dikurangi normalized edit distance, sehingga transkrip yang
berbeda satu karakter mendapat skor mendekati 1 ketika F1 end-to-end menilainya
0.

## Ekspor

Tidak ada format ekspor untuk task ini. PP-OCRv5 terdiri dari dua jaringan yang
bergerak bersama, bukan satu graph yang dapat diproses dengan tracing, dan `export()`
memunculkan error untuk setiap format pada kedua family. Untuk deployment di
luar LibreYOLO, lakukan fine-tuning dan gunakan jalur deployment upstream.


