---
title: SAM
families:
  - sam
seo_title: 'SAM (Segment Anything): prediksi mask di LibreYOLO'
description: >-
  Gunakan SAM di LibreYOLO untuk segmentasi dengan prompt titik dan kotak.
  Instal dan prediksi dengan checkpoint base, large, dan huge di bawah
  Apache-2.0.
lead: >-
  SAM (Segment Anything) mengubah klik titik atau kotak menjadi mask objek.
  LibreYOLO memuatnya melalui factory LibreSAM khusus yang terpisah dari factory
  detektor LibreYOLO(), karena model berbasis prompt memerlukan bentuk
  pemanggilan berbeda.
keywords:
  - SAM
  - Segment Anything
  - promptable segmentation
  - segmentasi interaktif
  - point prompt
  - box prompt
  - Meta AI
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompt titik dan kotak
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # "base" mengunduh facebook/sam-vit-base secara otomatis saat penggunaan
        pertama.

        # Ukuran lain: "large", "huge" (juga "b"/"l"/"h").

        model = LibreSAM("base")


        # Prompt titik: [x, y] dalam koordinat piksel, label 1 = foreground.

        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])

        print(result.masks.xy)      # poligon per mask

        print(result.boxes.xyxy)    # kotak rapat yang diturunkan dari mask


        # Prompt kotak sebagai pengganti titik.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])


        # Tanpa prompt akan menyegmentasikan seluruh gambar (generator mask
        otomatis

        # yang disederhanakan, bukan versi referensi lengkap).

        result = model.predict(SAMPLE_IMAGE)
    - label: 'Enkode sekali, gunakan banyak prompt'
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        model = LibreSAM("base")


        # Encoder gambar adalah bagian yang mahal. set_image() menjalankannya
        sekali;

        # setiap pemanggilan predict() setelahnya menggunakan kembali embedding
        dari cache.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[640, 420], labels=[1])

        b = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
source_hash: f8904d241ef8a929
---

## Instalasi

SAM memerlukan komponen tambahan `sam`, yang memasang `transformers` dan `timm`.

```bash
pip install "libreyolo[sam]"
```

## Prediksi

`LibreSAM(...)` adalah entry point terpisah dari `LibreYOLO(...)`: hasilnya merupakan
segmenter yang dapat diberi prompt, bukan detektor, karena forward pass di sini tidak
bermakna tanpa prompt spasial. Tidak ada perintah CLI `libreyolo predict` untuk family ini;
gunakan API Python.

<code-tabs name="predict" />

Prompt titik menerima `[x, y]` untuk satu objek, `[[x, y], ...]` untuk beberapa objek,
atau array numpy; `labels` menandai setiap titik sebagai `1` (foreground) atau `0`
(background), dengan default semuanya foreground. Prompt kotak menerima
`[x1, y1, x2, y2]` atau daftar kotak, satu mask per kotak. Menghilangkan kedua prompt
akan menyegmentasikan seluruh gambar dengan memberi prompt pada grid padat dan mempertahankan
mask yang meyakinkan serta tidak tumpang tindih; mode "segment everything" ini disederhanakan
dibandingkan generator mask otomatis referensi dan dapat kurang menyegmentasikan adegan padat,
sehingga prompt titik atau kotak yang nyata menjadi jalur presisi. `conf` memfilter berdasarkan
kualitas mask yang diprediksi (IoU), bukan confidence deteksi: berikan `0.0` untuk
mempertahankan setiap kandidat. `multimask=True` mengembalikan ketiga mask ambiguitas
keseluruhan-lawan-bagian milik SAM per prompt, bukan hanya mask terbaik. `device=`
memindahkan model dan, jika sesi `set_image()` aktif, embedding dari cache. Setiap mask
memiliki id kelas `0` dengan nama `"object"`, karena mask berbasis prompt tidak memiliki
kumpulan kelas tetap. `train()`, `val()`, `export()`, dan `track()` semuanya memunculkan
`NotImplementedError` untuk family ini: SAM hanya mendukung prediksi di LibreYOLO, sedangkan
pelacakan video berada di luar cakupan. Lihat [prediksi](/docs/predict) untuk jenis sumber.

## Varian

Tiga ukuran encoder gambar ViT: base, large, dan huge, semuanya pada input tetap 1024 px.
Belum ada benchmark akurasi atau latensi yang dipublikasikan untuk family ini, sehingga
pemilihan ukuran menukar bobot encoder dengan kualitas mask secara langsung: base paling
cepat untuk enkode, sedangkan huge paling berat.

## Lisensi

<provenance-box>

LibreYOLO tidak meng-host salinan bobot SAM-1 sendiri. `LibreSAM("base")`, `"large"`, dan
`"huge"` mengunduh langsung dari repositori `facebook/sam-vit-base`,
`facebook/sam-vit-large`, dan `facebook/sam-vit-huge` milik Meta di Hugging Face, yang
masing-masing diberi tag Apache-2.0 secara independen dari LibreYOLO.

</provenance-box>

## Sitasi

<citation-block />
