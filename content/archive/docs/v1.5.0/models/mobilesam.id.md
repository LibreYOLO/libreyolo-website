---
title: MobileSAM
families:
  - mobilesam
seo_title: 'MobileSAM: segmentasi dengan prompt yang ringan di LibreYOLO'
description: >-
  Gunakan MobileSAM di LibreYOLO untuk segmentasi dengan prompt titik dan box
  menggunakan encoder TinyViT. Instal dan prediksi dengan checkpoint tiny
  berlisensi Apache-2.0.
lead: >-
  MobileSAM mengganti encoder gambar ViT-H milik SAM dengan encoder TinyViT yang
  didistilasi, sehingga alur prompt titik dan box yang sama dapat berjalan pada
  hardware lebih ringan. LibreYOLO menyediakan port native melalui factory
  LibreSAM khusus, terpisah dari factory detector LibreYOLO().
keywords:
  - MobileSAM
  - Segment Anything
  - TinyViT
  - promptable segmentation
  - segmentasi interaktif
  - point prompt
  - box prompt
  - segmentasi ringan
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompt titik dan box
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # MobileSAM memiliki satu ukuran, "tiny", sehingga tidak memerlukan
        alias lain.

        model = LibreSAM("mobilesam")


        # Prompt titik: [x, y] dalam koordinat piksel, label 1 = foreground.

        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])

        print(result.masks.xy)      # poligon per mask

        print(result.boxes.xyxy)    # box rapat yang diturunkan dari mask


        # Prompt box sebagai pengganti titik.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])


        # Tanpa prompt sama sekali, seluruh gambar disegmentasi (generator mask
        otomatis

        # yang disederhanakan, bukan versi referensi yang menyeluruh).

        result = model.predict(SAMPLE_IMAGE)
    - label: 'Encode sekali, gunakan banyak prompt'
      language: python
      code: >
        from libreyolo import LibreMobileSAM, SAMPLE_IMAGE


        model = LibreMobileSAM()


        # Encoder gambar adalah bagian yang mahal. set_image() menjalankannya
        sekali;

        # setiap panggilan predict() sesudahnya menggunakan kembali embedding
        dalam cache.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[640, 420], labels=[1])

        b = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
source_hash: f96e885d93f72bdd
---

## Instalasi

MobileSAM memerlukan extra `sam`. Unduhan bobot milik LibreYOLO tetap melalui
tool snapshot Hugging Face dari `transformers`, meskipun inferensi berjalan pada
decoder native yang tidak berbasis `transformers`.

```bash
pip install "libreyolo[sam]"
```

## Prediksi

`LibreSAM(...)` (atau `LibreMobileSAM(...)` khusus family) adalah entry point
yang terpisah dari `LibreYOLO(...)`. Fungsi ini mengembalikan segmenter dengan
prompt, bukan detector, karena forward pass tidak bermakna tanpa prompt spasial.
Tidak ada perintah CLI `libreyolo predict` untuk family ini. Gunakan API Python.

<code-tabs name="predict" />

Prompt titik menerima `[x, y]` untuk satu objek, `[[x, y], ...]` untuk beberapa
objek, atau array numpy. `labels` menandai setiap titik sebagai `1` (foreground)
atau `0` (background) dan default-nya semua foreground. Prompt box menerima
`[x1, y1, x2, y2]` atau daftar box, satu mask per box. Menghilangkan kedua prompt
akan menyegmentasi seluruh gambar dengan memberi prompt pada grid padat dan
mempertahankan mask ber-confidence tinggi yang tidak tumpang tindih. Mode
"segment everything" ini disederhanakan dibandingkan generator mask otomatis
referensi dan dapat menghasilkan segmentasi yang kurang lengkap pada adegan
padat. Prompt titik atau box nyata adalah jalur yang lebih presisi. `conf`
memfilter berdasarkan prediksi kualitas mask (IoU), bukan confidence deteksi.
Teruskan `0.0` untuk mempertahankan setiap kandidat. `multimask=True`
mengembalikan ketiga mask ambiguitas whole-versus-part milik SAM untuk setiap
prompt, bukan hanya mask terbaik. `device=` memindahkan model dan, jika sesi
`set_image()` aktif, embedding dalam cache. Setiap mask memakai ID kelas `0`
bernama `"object"` karena mask dengan prompt tidak memiliki set kelas tetap.
`train()`, `val()`, `export()`, dan `track()` semuanya memunculkan
`NotImplementedError` untuk family ini. MobileSAM hanya untuk prediksi di
LibreYOLO. Lihat [prediksi](/docs/predict) untuk jenis sumber.

## Varian

Ada satu ukuran, tiny, dengan input tetap 1024 px. MobileSAM menyediakan satu
encoder TinyViT, bukan jenjang base/large/huge milik SAM-1.

## Checkpoint

Semua berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
