---
title: Deteksi tepi
seo_title: Deteksi tepi di LibreYOLO
description: >-
  Prediksi peta probabilitas tepi padat dari satu gambar di LibreYOLO. Konversi
  checkpoint, ambang peta, validasi dengan ODS dan OIS, dan ekspor.
lead: >-
  Deteksi tepi memprediksi seberapa besar kemungkinan setiap piksel berada di
  batas objek. LibreYOLO mengeksposnya sebagai tepi task, yang mengembalikan
  peta probabilitas padat pada kanvas gambar asli daripada sekumpulan segmen
  garis.
keywords:
  - deteksi tepi python
  - deteksi batas pembelajaran mendalam
  - peta probabilitas tepi
  - ODS OIS F-measure
  - prediksi tepi padat
last_verified: 1.5.0
snippets:
  predict:
    - label: Prediksi peta tepi
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Tidak ada checkpoint tepi yang dilengkapi dengan LibreYOLO; konversi terlebih dahulu (di bawah).
        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE, save=True)

        edges = result.edges
        print(edges.array.shape)          # (H, W) float32 di [0, 1]
        print(edges.binary(0.5).sum())    # jumlah piksel tepi pada 0.5
    - label: Pilih ambang batas Anda sendiri
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE)

        # Peta kontinu tetap sehingga ambang tetap sesuai keputusan Anda.
        for t in (0.3, 0.5, 0.7):
            print(t, int(result.edges.binary(t).sum()))
    - label: Simpan visualisasi
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        result = model(SAMPLE_IMAGE)

        # plot() menampilkan peta; ini didefinisikan untuk results tepi dan normal.
        result.plot().save("edges.png")
  val:
    - label: Validasi dan baca kunci metrik
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(data="my-dataset.yaml", imgsz=352)

        print(metrics["metrics/ODS"])              # kecocokan
        print(metrics["metrics/OIS"])
        print(metrics["metrics/best_threshold"])
    - label: Ubah sapuan dan toleransi cocok
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        metrics = model.val(
            data="my-dataset.yaml",
            imgsz=352,
            edge_thresholds=(0.1, 0.2, 0.3, 0.4, 0.5),
            edge_max_dist=0.0075,
        )

        print(metrics["metrics/ODS"], metrics["metrics/best_threshold"])
  export:
    - label: Ekspor
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("weights/LibreDexiNedb-edge.pt")
        model.export(format="onnx", imgsz=352)
    - label: Jalankan berkas yang diekspor
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Pabrik mengarahkan pada akhiran berkas, sehingga artefak yang diekspor dimuat
        # seperti halnya checkpoint dan mengembalikan objek Results yang sama.
        model = LibreYOLO("weights/LibreDexiNedb-edge.onnx")
        result = model(SAMPLE_IMAGE)

        print(result.edges.array.shape)
source_hash: bc286345540ed966
---

## Definisi

`edge` task memprediksi satu probabilitas per piksel dari satu gambar RGB:
`0` berarti non-tepi dan `1` berarti tepi. Peta tetap kontinu, jadi memilih
ambang yang mengubahnya menjadi gambar batas biner diserahkan kepada pemanggil, dan
ambang yang tepat bergantung pada dataset dan penggunaan hilirnya.

Sebuah prediksi mengisi `result.edges`, sebuah muatan `EdgeMap` yang memegang `(H, W)`
array float32 di `[0, 1]` pada kanvas gambar asli. `.array` mengembalikan itu
peta sebagai NumPy dan `.binary(threshold)` mengembalikan topeng boolean. `result.boxes`
tetap kosong, jadi `conf`, `iou`, dan `max_det` tidak berpengaruh. `Results.plot()`
menutupi task ini dan menghasilkan peta secara langsung.

## Model

Tiga keluarga melayani `edge`.

[DexiNed](/docs/models/dexined), Jaringan Inception Ekstrem Padat, menggabungkan
beberapa keluaran samping menjadi satu peta probabilitas dan berjalan pada 352 px asli.

[TEED](/docs/models/teed), Detektor Tepi Kecil dan Efisien, adalah jaringan kecil
pada 352 px asli yang sama, dengan langkah downsample 4 dibanding
DexiNed yang 16, sehingga menerima lebih banyak nilai dari `imgsz`.

[LibreMODUS](/docs/models/libremodus) menghasilkan tepi gaya Canny sebagai satu target
dari model apa pun ke apa pun. Ini membutuhkan `modus` tambahan dan akun Hugging Face
Anda sendiri yang sah, dan tidak menawarkan `val()` maupun `export()`, sehingga tidak
ikut serta dalam bagian validasi dan ekspor di bawah ini.

## Prediksi

LibreYOLO menerbitkan checkpoint tanpa tepi. DexiNed dan TEED yang dirilis secara resmi
bobot dilatih pada BIPED, yang istilah dataset yang dipublikasikan membatasi penggunaan untuk
tujuan non-komersial, jadi LibreYOLO tidak mencerminkan mereka. Konversi checkpoint
Anda memiliki lisensi untuk menggunakan, kemudian muat berkas yang dikonversi melalui jalur:

```bash
python weights/convert_dexined_weights.py upstream.pth weights/LibreDexiNedb-edge.pt --verify
```

<code-tabs name="predict" />

Nama berkas harus memiliki akhiran `-edge` task agar loader dapat mengenalinya
itu. `imgsz` harus dapat dibagi oleh langkah downsample jaringan, dan LibreYOLO
menimbulkan kesalahan yang jelas dengan menyebut pembagi saat itu tidak ada. Lihat
[prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Format dataset

Validasi tepi memasangkan setiap gambar RGB dengan peta satu saluran berbatang sama dari
resolusi yang sama, ditambah dengan mask validitas opsional.

```text
dataset/
  data.yaml
  images/
    val/scene.jpg
  edges/
    val/scene.png
  masks/
    val/scene.png
```

```yaml
path: dataset
train: images/train
val: images/val
edges_dir: edges
masks_dir: masks
nc: 1
names: {0: edge}
```

Targetnya adalah PNG atau TIF saluran tunggal, bukan visualisasi RGB. Bilangan bulat
peta dibagi berdasarkan maksimum dtype mereka; peta float harus sudah
terbatas dan berada di `[0, 1]`. Piksel mask dihitung sebagai valid ketika tidak nol, dan dipadatkan
piksel tidak pernah berkontribusi pada metrik. `edge_invert: true` mencakup sumber yang
simpan tepi hitam pada putih. Lihat
[dataset memformat](/docs/reference/dataset-formats) untuk kontrak penuh.

## Kereta

Tidak ada edge family di LibreYOLO yang memiliki implementasi pelatihan: `train()` memunculkan
`NotImplementedError` pada ketiganya. Setiap halaman model menyebutkan skrip konversi
yang mengubah checkpoint yang dilatih di tempat lain menjadi LibreYOLO yang dapat dimuat.

## Validasi

`val()` melaporkan ukuran F gaya BSDS. Prediksi kontinu ditipiskan
pertama dengan penekanan maksimum non-empat arah gradien, kemudian diprediksi dan
piksel tepi ground-truth dipasangkan satu-ke-satu dalam toleransi jarak.

<code-tabs name="val" />

`metrics/ODS` adalah ukuran F skala optimal-dataset: jumlah kecocokan digabungkan
di seluruh dataset pada setiap ambang batas, dan F-measure gabungan terbaik dari itu adalah
dilaporkan. Ini juga `fitness`, nomor baca pemilihan terbaik-checkpoint.
`metrics/OIS` adalah F-measure skala-gambar-optimal, rata-rata di atas gambar dari masing-masing
F-measure terbaik gambar itu sendiri, sehingga membiarkan setiap gambar memilih ambang batasnya sendiri.
`metrics/best_threshold` adalah ambang tunggal yang menghasilkan ODS, yang adalah
satu untuk digunakan kembali di `edges.binary()` saat inferensi.

Dua argumen membentuk rentang. `edge_thresholds` adalah kumpulan ambang batas yang dicoba,
mengatur default ke 0,01 hingga 0,99 dalam satuan perseratus. `edge_max_dist` adalah kecocokannya
toleransi sebagai pecahan dari diagonal gambar, secara default adalah `0.0075`; sepasang
lebih jauh dari itu bukanlah pasangan yang cocok.

## Ekspor

Model tepi yang diekspor dimuat kembali melalui `LibreYOLO()` berdasarkan sufiks filenya, jadi a
Berkas `.onnx` berperilaku seperti checkpoint dan mengembalikan `Results` yang sama.

<code-tabs name="export" />

Ekspor Edge menggunakan resolusi tetap, kontrak runtime batch-1: `dynamic` dan sebuah
`batch` selain 1 ditolak, dan grafik yang diekspor memancarkan sebuah gabungan tunggal
peta probabilitas. Cakupan per format ada pada [DexiNed](/docs/models/dexined)
dan [TEED](/docs/models/teed) halaman dan di
[matriks ekspor penuh](/docs/reference/export-matrix). [Ekspor](/docs/export)
mencantumkan argumen yang diterima setiap format.


