---
title: EdgeTAM
families:
  - edgetam
seo_title: 'EdgeTAM: segmentasi promptable on-device di LibreYOLO'
description: >-
  Gunakan EdgeTAM di LibreYOLO untuk segmentasi promptable berbasis titik dan
  box yang dibuat untuk kecepatan on-device. Pasang dan prediksi checkpoint-nya
  di bawah Apache-2.0.
lead: >-
  EdgeTAM adalah varian on-device dari SAM 2, dibuat untuk kecepatan inferensi
  di perangkat mobile sambil mempertahankan alur kerja promptable berbasis titik
  dan box yang sama. LibreYOLO mendukung jalur segmentasi gambarnya lewat
  factory LibreSAM khusus, terpisah dari factory detektor LibreYOLO().
keywords:
  - EdgeTAM
  - SAM 2
  - promptable segmentation
  - interactive segmentation
  - segmentasi gambar python
  - point prompt
  - box prompt
  - segment anything python
  - Meta Reality Labs
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompt titik dan box
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        # EdgeTAM hanya punya satu ukuran, "edge". Alias: "edgetam", "edge-tam",
        # "edgetam-edge".
        model = LibreSAM("edgetam")

        # Prompt titik: [x, y] dalam koordinat piksel, label 1 = foreground.
        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])
        print(result.masks.xy)      # poligon per mask
        print(result.boxes.xyxy)    # box ketat yang diturunkan dari mask

        # Prompt box sebagai ganti titik.
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])

        # Tanpa prompt sama sekali, seluruh gambar disegmentasi (generator
        # mask otomatis yang disederhanakan, bukan versi referensi yang
        # menyeluruh).
        result = model.predict(SAMPLE_IMAGE)
    - label: 'Encode sekali, prompt berkali-kali'
      language: python
      code: |
        from libreyolo import LibreEdgeTAM, SAMPLE_IMAGE

        model = LibreEdgeTAM()

        # Encoder gambar adalah bagian yang mahal. set_image() menjalankannya
        # sekali; setiap panggilan predict() sesudahnya memakai ulang embedding
        # yang tersimpan di cache.
        model.set_image(SAMPLE_IMAGE)
        a = model.predict(points=[640, 420], labels=[1])
        b = model.predict(bboxes=[300, 200, 900, 700])
        model.reset_image()
source_hash: e6cce8faad18e73d
---

## Instalasi

EdgeTAM membutuhkan extra `sam`, yang menarik `transformers` dan `timm`.

```bash
pip install "libreyolo[sam]"
```

## Prediksi

`LibreSAM(...)` (atau `LibreEdgeTAM(...)` yang khusus untuk family ini) adalah
entry point terpisah dari `LibreYOLO(...)`: ia mengembalikan segmenter
promptable, bukan detektor, karena satu forward pass di sini tidak berarti
apa-apa tanpa prompt spasial. Tidak ada perintah CLI `libreyolo predict` untuk
family ini; pakai API Python. Hanya segmentasi gambar yang didukung; pelacakan
video EdgeTAM berada di luar cakupan halaman ini.

<code-tabs name="predict" />

Prompt titik menerima `[x, y]` untuk satu objek, `[[x, y], ...]` untuk beberapa
objek, atau array numpy; `labels` menandai setiap titik dengan `1` (foreground)
atau `0` (background) dan secara default menganggap semuanya foreground. Prompt
box menerima `[x1, y1, x2, y2]` atau sebuah daftar box, satu mask per box. Bila
kedua prompt dihilangkan, seluruh gambar disegmentasi dengan memberi prompt
berupa grid rapat lalu menyimpan mask yang meyakinkan dan tidak saling tumpang
tindih; mode "segment everything" ini disederhanakan dibandingkan generator mask
otomatis versi referensi dan bisa kurang memisahkan objek pada adegan yang
padat, jadi prompt titik atau box yang sebenarnya adalah jalur yang presisi.
`conf` menyaring berdasarkan kualitas mask yang diprediksi (IoU), bukan skor
keyakinan (confidence) deteksi: berikan `0.0` untuk menyimpan semua kandidat.
`multimask=True` mengembalikan ketiga mask ambiguitas keseluruhan-versus-bagian
milik SAM untuk setiap prompt, bukan hanya satu yang terbaik. `device=`
memindahkan model dan, bila sesi `set_image()` sedang aktif, embedding miliknya
yang tersimpan di cache. Setiap mask membawa id kelas `0` dengan nama
`"object"`, karena mask promptable tidak punya himpunan kelas yang tetap.
`train()`, `val()`, `export()` dan `track()` semuanya memunculkan
`NotImplementedError` untuk family ini: inferensi gambar adalah yang didukung
LibreYOLO di sini. Lihat [prediksi](/docs/predict) untuk jenis sumber.

## Varian

Satu ukuran, edge, pada resolusi input tetap, jadi memilih family ini ketimbang
sisa tier SAM adalah keputusan perangkat keras, bukan keputusan ukuran: EdgeTAM
ada khusus untuk inferensi on-device pada perangkat terbatas.

## Checkpoint

Semua berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
