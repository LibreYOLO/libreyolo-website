---
title: API segmentasi yang dapat dipicu lewat prompt
seo_title: 'API LibreSAM: prompt, alias, dan tanda tangan'
description: >-
  Pabrik LibreSAM, alias ukuran, tipe prompt titik, kotak, dan konsep-teks,
  siklus hidup set_image encode-once, dan apa yang tidak didukung oleh tier.
lead: >-
  LibreSAM adalah pabrik untuk segmentasi yang dapat dipicu lewat prompt. Satu
  proses forward membutuhkan prompt per gambar yang diberikan saat pemanggilan,
  sehingga tier memiliki permukaan prediksi sendiri daripada melalui runner
  inferensi tanpa prompt.
keywords:
  - LibreSAM
  - segmentasi yang dapat dipicu lewat prompt
  - prompt titik SAM
  - prompt kotak SAM
  - set_image
  - segmentasi semua
  - libreyolo sam ekstra
last_verified: 1.5.0
verification: >-
  , ukuran dan repositori dibaca dari libreyolo/models/sam/model.py, sam2.py,
  edgetam.py, sam3.py, libreyolo/models/mobilesam/model.py dan
  libreyolo/models/picosam3/model.py. Kontrak dan default prompt dibaca dari
  libreyolo/models/sam/base.py. Niat desain dari
  docs/adr/0007-libresam-contract.md, semuanya di v1.5.0.
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[sam]'
  usage:
    - label: Prompt titik dan kotak
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        r = model.predict(SAMPLE_IMAGE, points=[900, 370], labels=[1])
        print(r.masks.xy)
        print(r.boxes.xyxy)

        r = model.predict(SAMPLE_IMAGE, bboxes=[100, 100, 200, 200])
        print(len(r))
    - label: 'Encode sekali, prompt berkali-kali'
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")
        model.set_image(SAMPLE_IMAGE)

        a = model.predict(points=[500, 375], labels=[1])
        b = model.predict(bboxes=[100, 100, 200, 200])
        print(len(a), len(b))

        model.reset_image()
source_hash: 18e8206c10ce17fd
---

## Instal

Tier ini membutuhkan tambahan `sam`.

<code-tabs name="install" />

## Pabrik

```python
LibreSAM(model: str = "base", **kwargs) -> LibreSAMModel
```

`model` adalah alias ukuran, bukan path. `**kwargs` mencapai family
konstruktor, yang menerima `device` dan `multimask`. Alias yang tidak dikenal memunculkan
`ValueError` dan pesan mencantumkan setiap alias yang dikenal.

<code-tabs name="usage" />

## Alias

| Family | Alias | Ukuran | Berat |
|---|---|---|---|
| SAM-1 | `base`, `large`, `huge`, `b`, `l`, `h`, `sam-base`, `sam-large`, `sam-huge`, `sam_b`, `sam_l`, `sam_h` | `base`, `large`, `huge` | `facebook/sam-vit-base`, `-large`, `-huge` |
| SAM-2 | `sam2-tiny`, `sam2-small`, `sam2-base-plus`, `sam2-baseplus`, `sam2-large`, dan bentuk pendek `sam2-t`, `sam2-s`, `sam2-bp`, `sam2-l`, `sam2_t`, `sam2_s`, `sam2_bp`, `sam2_l` | `tiny`, `small`, `base-plus`, `large` | `LibreYOLO/LibreSAM2tiny`, `-small`, `-base-plus`, `-large` |
| EdgeTAM | `edgetam`, `edge-tam`, `edgetam-edge` | `edge` | `LibreYOLO/LibreEdgeTAM` |
| SAM 3 | `sam3`, `sam-3`, `sam3-large` | `large` | `facebook/sam3` |
| MobileSAM | `mobilesam`, `mobilesam-tiny`, `mobilesam_t`, `mobile-sam`, `mobile-sam-tiny` | `tiny` | `LibreYOLO/LibreMobileSAM` |
| PicoSAM3 | `picosam3`, `picosam3-pico`, `picosam3_pico`, `pico-sam3` | `pico` | `LibreYOLO/LibrePicoSAM3` |

Default adalah `base`. SAM-1, SAM-2, EdgeTAM dan MobileSAM berjalan pada kanvas nominal
1024 piksel, SAM 3 pada 1008, PicoSAM3 pada 96.

Bobot SAM 3 dibatasi. Mereka diunduh dari `facebook/sam3` di bawah Lisensi SAM khusus Meta,
yang bukan MIT maupun Apache-2.0 dan tidak
didistribusikan ulang oleh LibreYOLO. Terima syarat di halaman repositori dan
autentikasi dengan Hugging Face sebelum memuat; pemuat mencatat pemberitahuan
terlebih dahulu.

Kelas family juga diekspor, sehingga `LibreSAM1`, `LibreSAM2`,
`LibreSAM3`, `LibreEdgeTAM`, `LibreMobileSAM` dan `LibrePicoSAM3` dapat
dibangun langsung dengan `size=`.

## prediksi

```python
model.predict(
    source=None,
    *,
    points=None,
    bboxes=None,
    labels=None,
    masks=None,
    text=None,
    conf=None,
    multimask=None,
    max_det=300,
    device=None,
    color_format="auto",
    points_per_side=None,
) -> Results
```

| Argumen | Default | Makna |
|---|---|---|
| `source` | `None` | Gambar untuk disegmentasi; `None` menggunakan kembali gambar yang disimpan oleh `set_image()` |
| `points` | `None` | Prompt titik dalam koordinat piksel |
| `bboxes` | `None` | Prompt kotak sebagai `[x1, y1, x2, y2]`, atau daftar dari mereka untuk satu mask per kotak |
| `labels` | `None` | Label titik, `1` positif dan `0` negatif, disesuaikan untuk cocok dengan `points`; semua positif jika dihilangkan |
| `masks` | `None` | Cadangan; melewati satu memunculkan `NotImplementedError` |
| `text` | `None` | Prompt konsep; hanya SAM 3 |
| `conf` | `None` | Prediksi lantai mask-IoU |
| `multimask` | `None` | Kembalikan semua mask ambiguitas per prompt; default ke pengaturan konstruksi |
| `max_det` | `300` | Batas pada mask yang dikembalikan |
| `device` | `None` | Pindahkan model untuk panggilan ini dan panggilan berikutnya, membatalkan penyematan cache |
| `color_format` | `"auto"` | Petunjuk format warna untuk array dalam memori |
| `points_per_side` | `None` | Kepadatan grid untuk segment-everything; default ke 32 |

Kembalian adalah `Results` biasa yang membawa `masks`, plus `boxes` ketat
yang diperoleh dari mask-mask tersebut, dengan kelas `0` bernama `"object"`.

## Bentuk prompt

`points` menerima bentuk bersarang `[x, y]` untuk satu objek, `[[x, y], ...]` untuk
N objek, dan `[[[x, y], ...], ...]` untuk titik yang dikelompokkan per objek. Numpy
array bekerja di mana pun sebuah daftar bekerja. Koordinat adalah piksel biasa pada sumber
gambar.

Melewatkan setiap prompt spasial menjalankan segment-everything, sebuah mask otomatis grid
generator dengan ambang IoU yang diprediksi dan deduplikasi box-IoU. The
default `points_per_side` dari 32 menjalankan kira-kira 1024 pass decoder, yang
lambat pada CPU; turunkan untuk penggunaan interaktif. Generator mengabaikan
penyaringan skor-stabilitas, multi-crop dan deduplikasi mask-IoU, jadi itu adalah
perkiraan dari jalur yang diminta daripada kecocokan dengannya.

## Kepercayaan

`conf` memfilter berdasarkan mask-IoU yang diprediksi, yang merupakan skor kualitas mask dan bukan
kepercayaan deteksi. `None` mempertahankan setiap mask di jalur yang diarahkan dan menerapkan
ambang grid family dalam segment-everything. `0.0` menonaktifkan penyaringan dalam
salah satu mode.

Pada jalur teks SAM 3, `conf` adalah skor deteksi Promptable Concept Segmentation
sebagai gantinya. `None` di sana berarti ambang standar 0,3, dan `0.0` mempertahankan
semua kandidat.

## Prompt teks

`text=` hanya untuk SAM 3; setiap prompt spasial family meningkatkan
`NotImplementedError` untuk itu. Teks bersifat saling eksklusif dengan titik dan
kotak. `names` yang dikembalikan memetakan kelas `0` ke konsep yang diminta. Sebuah teks
panggilan dengan `source=None` meng-encode ulang gambar yang di-cache, karena pelacak dan
encoder konsep tidak berbagi cache.

Kata kunci `exemplars=` dicadangkan untuk ekstensi contoh-gambar di masa depan dan
tidak diimplementasikan.

## Siklus hidup encode-sekali

```python
model.set_image(source, color_format="auto") -> LibreSAMModel
model.reset_image() -> LibreSAMModel
```

`set_image` menjalankan pengkode gambar berat sekali dan menyimpan embedding, jadi
setiap `predict()` berikutnya dengan `source=None` itu murah. Kedua metode mengembalikan
model sehingga panggilan dapat berantai. Melewatkan `device=` ke `predict` memindahkan model dan
membatalkan cache.

## PicoSAM3

PicoSAM3 hanya menerima `bboxes=`. Titik, teks, mask, multimask dan
prompts segment-everything meningkat. Kotaknya diperluas sebesar 10 persen dan dijalankan
melalui jaringan ROI 96 piksel, dan PicoSAM3 adalah family di tingkat tersebut
yang mengekspor, hanya ke ONNX.

## Tidak didukung

`train()`, `val()` dan `track()` menaikkan `NotImplementedError` pada setiap family
dalam tingkatan. Mask yang dapat dijalankan tidak memiliki satu set kelas tetap untuk dinilai, jadi
mAP tidak ada artinya di sini. `export()` menaikkan untuk SAM-1, SAM-2, SAM 3, EdgeTAM
dan MobileSAM.

Jalur video dan memori untuk SAM-2, SAM 3, dan EdgeTAM berada di luar cakupan ini
versi, seperti halnya contoh gambar SAM 3 dan petunjuk mask.

