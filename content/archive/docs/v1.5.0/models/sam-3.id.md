---
title: SAM 3
families:
  - sam3
seo_title: 'SAM 3: segmentasi dengan prompt dan konsep di LibreYOLO'
description: >-
  Gunakan SAM 3 di LibreYOLO untuk segmentasi titik, kotak, dan konsep teks.
  Instal dan prediksi dengan checkpoint large yang dibatasi oleh SAM License
  dari Meta.
lead: >-
  SAM 3 memperluas SAM dengan prompt konsep teks di atas titik dan kotak yang
  biasa, sehingga frasa seperti "yellow school bus" mengembalikan setiap
  instance yang cocok. LibreYOLO mendukung jalur gambarnya melalui factory
  LibreSAM khusus yang terpisah dari factory detektor LibreYOLO().
keywords:
  - SAM 3
  - Segment Anything
  - promptable segmentation
  - segmentasi konsep
  - text prompt
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


        # "sam3" adalah satu-satunya ukuran ("large"); alias: "sam3", "sam-3",
        "sam3-large".

        model = LibreSAM("sam3")


        # Prompt titik: [x, y] dalam koordinat piksel, label 1 = foreground.

        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])

        print(result.masks.xy)      # poligon per mask

        print(result.boxes.xyxy)    # kotak rapat yang diturunkan dari mask


        # Prompt kotak sebagai pengganti titik.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
    - label: Prompt teks (konsep)
      language: python
      code: >
        from libreyolo import LibreSAM3, SAMPLE_IMAGE


        model = LibreSAM3("large")


        # Menemukan setiap instance yang cocok dengan frasa, bukan hanya satu
        objek.

        # text= saling eksklusif dengan points, bboxes, labels, dan masks.

        result = model.predict(SAMPLE_IMAGE, text="a person")

        print(result.names)         # {0: "a person"}

        print(result.boxes.conf)    # skor deteksi PCS per instance
    - label: 'Enkode sekali, gunakan banyak prompt'
      language: python
      code: >
        from libreyolo import LibreSAM3, SAMPLE_IMAGE


        model = LibreSAM3("large")


        # Encoder gambar adalah bagian yang mahal. set_image() menjalankannya
        sekali;

        # setiap pemanggilan predict() setelahnya memakai embedding dari cache.
        Pemanggilan

        # text= melakukan enkode ulang secara internal karena tracker dan
        encoder

        # segmentasi konsep tidak berbagi cache.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[640, 420], labels=[1])

        b = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
source_hash: c4fb6d5a622f99ff
---

## Instalasi

SAM 3 memerlukan komponen tambahan `sam`, yang memasang `transformers` dan `timm`.

```bash
pip install "libreyolo[sam]"
```

Bobotnya dibatasi: kunjungi
[huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3), terima SAM License
dari Meta, lalu jalankan `hf auth login` (atau tetapkan `HF_TOKEN`) sebelum pengunduhan
pertama. LibreYOLO mencatat pemberitahuan lisensi saat pertama kali mengunduh family ini.

## Prediksi

`LibreSAM(...)` (atau `LibreSAM3(...)` yang khusus untuk family ini) adalah entry point
terpisah dari `LibreYOLO(...)`: hasilnya merupakan segmenter yang dapat diberi prompt,
bukan detektor, karena forward pass di sini tidak bermakna tanpa prompt. Tidak ada perintah
CLI `libreyolo predict` untuk family ini; gunakan API Python. Hanya inferensi gambar yang
didukung; model video SAM 3 berada di luar cakupan.

<code-tabs name="predict" />

Jalur titik dan kotak cocok dengan family SAM lainnya: prompt titik menerima `[x, y]`
untuk satu objek atau `[[x, y], ...]` untuk beberapa objek, `labels` menandai setiap titik
sebagai `1` (foreground) atau `0` (background), dan prompt kotak menerima
`[x1, y1, x2, y2]` atau daftar kotak. `conf` pada jalur ini memfilter berdasarkan kualitas
mask yang diprediksi (IoU), bukan confidence deteksi.

Jalur `text=` merupakan tambahan SAM 3: string konsep mengembalikan setiap instance yang
cocok dalam gambar melalui Promptable Concept Segmentation dan tidak dapat digabungkan
dengan titik, kotak, label, atau mask. `conf` di sana adalah skor deteksi PCS, bukan IoU
mask; membiarkannya pada default akan menerapkan ambang 0.3 milik model, sedangkan
`conf=0.0` mempertahankan setiap kandidat. `names` yang dikembalikan memetakan id kelas
`0` ke string konsep yang diminta karena mask berbasis prompt tidak memiliki kumpulan kelas
tetap. `device=` memindahkan model dan, jika sesi `set_image()` aktif, embedding dari cache.
`train()`, `val()`, `export()`, dan `track()` semuanya memunculkan `NotImplementedError`
untuk family ini: SAM 3 hanya mendukung prediksi di LibreYOLO, sedangkan pelacakan video
berada di luar cakupan. Lihat [prediksi](/docs/predict) untuk jenis sumber.

## Varian

Satu ukuran, large, dengan input tetap 1008 px. SAM 3.1 tidak didukung: implementasinya
memakai lisensi khusus yang tidak dapat disertakan ke repositori MIT ini, dan versi
Transformers yang menjadi dependensi LibreYOLO belum dapat memuat format checkpoint-nya.

## Lisensi

<provenance-box>

LibreYOLO tidak meng-host salinan bobot SAM 3 sendiri dan tidak mendistribusikannya ulang.
`LibreSAM("sam3")` mengunduh langsung dari repositori `facebook/sam3` milik Meta yang
dibatasi di Hugging Face. Repositori tersebut mewajibkan penerimaan SAM License dari Meta
dan autentikasi sebelum pengunduhan pertama.

</provenance-box>

## Sitasi

<citation-block />
