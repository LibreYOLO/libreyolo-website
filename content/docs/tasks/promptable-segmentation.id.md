---
title: Promptable segmentation
seo_title: Promptable segmentation di LibreYOLO
description: >-
  Ubah titik, box, atau konsep teks menjadi mask objek di LibreYOLO. Muat SAM,
  SAM 2, SAM 3, EdgeTAM, MobileSAM, atau PicoSAM3 melalui LibreSAM.
lead: >-
  Promptable segmentation mengubah klik menjadi mask: tunjuk objek atau gambar
  box di sekitarnya, lalu model mengembalikan outline. Di LibreYOLO, ini bukan
  key task terpisah, melainkan tier model yang dimuat melalui factory LibreSAM,
  dengan hasil berupa Results segmentation biasa.
keywords:
  - promptable segmentation
  - segmentasi interaktif
  - segment anything python
  - prompt titik
  - prompt box
  - SAM python
  - mask dari klik
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompt titik dan box
      language: python
      code: |
        from libreyolo import LibreSAM, SAMPLE_IMAGE

        model = LibreSAM("base")

        # Titik adalah [x, y] dalam piksel; label 1 positif dan 0 negatif.
        result = model.predict(SAMPLE_IMAGE, points=[640, 420], labels=[1])
        print(result.masks.xy)      # poligon
        print(result.boxes.xyxy)    # box rapat yang diturunkan dari mask

        # Prompt box memberikan satu mask per box.
        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])
    - label: 'Encode sekali, berikan prompt berkali-kali'
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        model = LibreSAM("base")


        # set_image menjalankan image encoder yang berat sekali dan menyimpan
        cache.

        model.set_image(SAMPLE_IMAGE)

        first = model.predict(points=[640, 420], labels=[1])

        second = model.predict(bboxes=[300, 200, 900, 700])

        model.reset_image()
    - label: Segmentasikan semuanya
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        model = LibreSAM("base")


        # Tanpa prompt berarti grid titik di seluruh gambar. Default grid 32 per
        sisi

        # menghasilkan sekitar 1024 decoder pass, yang lambat pada CPU.

        result = model.predict(SAMPLE_IMAGE, points_per_side=8)

        print(len(result.masks))
    - label: Mask ambiguitas
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        model = LibreSAM("base")


        # Satu titik dapat berarti lengan baju, baju, atau orang. multimask=True

        # mengembalikan ketiga mask keseluruhan-versus-bagian, bukan yang
        terbaik saja.

        result = model.predict(
            SAMPLE_IMAGE, points=[640, 420], labels=[1], multimask=True
        )

        print(len(result.masks))
source_hash: bb70ff24e6c0a767
---

## Definisi

Promptable segmentation menerima gambar ditambah prompt spasial dan
mengembalikan mask objek yang ditunjuk prompt. Tidak ada classification: tidak
ada list kelas, dan `result.boxes` menyimpan box rapat yang diturunkan dari
mask, bukan deteksi mandiri. `result.masks` memuat data mask dan
`result.masks.xy` poligonnya.

Prompt adalah antarmukanya. `points` berupa koordinat piksel `[x, y]`, satu
kumpulan per objek, dengan `labels` menandai setiap titik positif (1, sertakan)
atau negatif (0, kecualikan). `bboxes` berupa `[x1, y1, x2, y2]`, satu mask per
box. Titik dan box dapat digabungkan, dengan pasangan per objek dan panjang
yang harus sama. Tanpa prompt, jalur segment-everything menjalankan grid titik
di seluruh gambar.

Satu titik secara inheren ambigu. Klik pada lengan baju dapat berarti lengan,
baju, atau orang, sehingga `multimask=True` mengembalikan ketiga mask
keseluruhan-versus-bagian per prompt, bukan satu mask terbaik. `conf` memfilter
berdasarkan predicted IoU model, yaitu score kualitas mask, bukan confidence
deteksi.

LibreYOLO tidak memiliki key task `promptable`. Tier mendaftar sebagai
`segment`, key yang sama dengan instance segmentation. Perbedaannya adalah
bentuk pemanggilan, sehingga tier memiliki factory `LibreSAM()` sendiri,
sibling dari `LibreYOLO()`, `LibreOpenVocab()`, dan `LibreVLM()`. Signature
`predict(image)` tunggal tidak dapat mewakili loop model ini: `set_image()`
menjalankan image encoder sekali dan menyimpan embedding, setiap `predict()`
berikutnya dengan `source=None` hanya membayar prompt decoding, dan
`reset_image()` membersihkan cache. Image encoder adalah biaya dominan dan
berjalan sekali per gambar, sehingga prompt kedua pada gambar sama melewatinya.

## Model

Enam family dimuat melalui `LibreSAM` berdasarkan alias.

[SAM](/docs/models/sam) adalah default dalam ukuran `base`, `large`, dan `huge`,
juga ditulis `b`, `l`, dan `h`.

[SAM 2](/docs/models/sam-2) memakai alias `sam2-tiny`, `sam2-small`,
`sam2-base-plus`, dan `sam2-large`. LibreYOLO mendukung jalur gambarnya.

[SAM 3](/docs/models/sam-3), alias `sam3`, adalah satu-satunya family yang
menerima prompt konsep teks: `text="yellow school bus"` mengembalikan setiap
instance cocok. Memberikan `text=` ke family lain memunculkan error yang
menyebut SAM 3. Bobot berasal dari Meta berdasarkan SAM License khusus, bukan
lisensi MIT LibreYOLO, dan repository bersifat gated: terima ketentuan pada
halaman model dan autentikasi dengan `hf auth login` sebelum pengunduhan
pertama. Baca [SAM 3](/docs/models/sam-3) sebelum deployment.

[EdgeTAM](/docs/models/edgetam), alias `edgetam`, adalah varian SAM 2 untuk
on-device. LibreYOLO mendukung jalur gambarnya.

[MobileSAM](/docs/models/mobilesam), alias `mobilesam`, mengganti encoder ViT-H
SAM dengan TinyViT hasil distillation.

[PicoSAM3](/docs/models/picosam3), alias `picosam3`, adalah CNN ringkas untuk
region dengan prompt box pada edge sensor. Prompt box adalah seluruh kontraknya:
titik, teks, mask, multimask, dan segment-everything memunculkan error yang
menunjuk ke SAM 2 atau SAM 3.

Ekstra tier mencakup empat family yang dimuat melalui `transformers`:

```bash
pip install "libreyolo[sam]"
```

MobileSAM dan PicoSAM3 adalah port native LibreYOLO dan tidak memerlukan
instalasi `transformers`.

## Predict

<code-tabs name="predict" />

`source` dan `set_image()` adalah alternatif, bukan urutan: berikan gambar ke
`predict()` untuk pemanggilan sekali jalan, atau panggil `set_image()` lalu
`predict(source=None)` untuk setiap prompt. Memberikan `device=` kepada
`predict()` memindahkan model untuk pemanggilan ini dan berikutnya, serta
membatalkan embedding yang di-cache.

Segment-everything adalah mode mahal. Default `points_per_side` 32 menghasilkan
sekitar 1024 decoder pass; turunkan nilainya untuk penggunaan interaktif pada
CPU. Dalam mode ini, `conf` yang tidak ditetapkan memakai ambang batas grid
family, sedangkan pada jalur dengan prompt, `conf` kosong mempertahankan setiap
mask. Berikan `conf=0.0` untuk menonaktifkan filtering dalam kedua mode, dan
`max_det` untuk membatasi jumlah mask.

Prompt mask belum didukung dan `masks=` memunculkan error, bukan diabaikan.
`track()` juga memunculkan error di seluruh tier: model ini merupakan image
segmenter, jadi jalankan `predict()` per frame. Lihat [prediksi](/docs/predict)
untuk sumber dan penanganan hasil.

## Train

Tidak ada family dalam tier ini yang berlatih di LibreYOLO. `train()`
memunculkan error: lakukan fine-tuning di upstream dan muat bobot hasilnya.

## Validate

Tidak ada validator untuk tier ini dan `val()` memunculkan error. Promptable
mask tidak memiliki kumpulan kelas tetap untuk dinilai, sehingga metrik deteksi
dan segmentation biasa tidak memiliki key. Penilaian prompt mask berarti
membandingkannya dengan mask referensi sendiri pada prompt yang relevan.

## Export

Ekspor berada di luar cakupan tier dan `export()` memunculkan error, dengan satu
pengecualian. [PicoSAM3](/docs/models/picosam3) mengekspor CNN region mentah
96x96 ke ONNX sebagai `roi_image -> mask_logits`; cropping box dan resize mask
kembali ke koordinat gambar tetap di Python. Family lain berjalan melalui
`predict()` di PyTorch. Lihat [ekspor](/docs/export) untuk format lain dalam
library.
