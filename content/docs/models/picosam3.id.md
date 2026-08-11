---
title: PicoSAM3
families:
  - picosam3
seo_title: 'PicoSAM3: segmentasi edge dengan prompt kotak di LibreYOLO'
description: >-
  Gunakan PicoSAM3 di LibreYOLO untuk segmentasi wilayah dengan prompt kotak
  pada sensor edge. Instal, prediksi, dan ekspor checkpoint pico di bawah
  Apache-2.0.
lead: >-
  PicoSAM3 adalah CNN ringkas yang didistilasi dari SAM 2.1 dan SAM 3, dibuat
  untuk segmentasi region of interest dengan prompt kotak pada sensor seperti
  Sony IMX500. LibreYOLO mendukungnya melalui factory LibreSAM khusus, terpisah
  dari factory detektor LibreYOLO(), dan hanya dengan prompt kotak.
keywords:
  - PicoSAM3
  - Segment Anything
  - segmentasi edge
  - region of interest
  - box prompt
  - inferensi dalam sensor
  - IMX500
  - knowledge distillation
last_verified: 1.5.0
snippets:
  predict:
    - label: Prompt kotak
      language: python
      code: >
        from libreyolo import LibreSAM, SAMPLE_IMAGE


        # PicoSAM3 hanya memiliki satu ukuran, "pico", jadi alias lain tidak
        diperlukan.

        model = LibreSAM("picosam3")


        # bboxes= adalah satu-satunya prompt yang didukung: [x1, y1, x2, y2]
        atau daftar

        # kotak, satu mask per kotak. Setiap kotak diperluas 10%, dibuat
        persegi,

        # dipotong sesuai gambar, dan diubah ukurannya menjadi 96x96 sebelum CNN
        berjalan.

        result = model.predict(SAMPLE_IMAGE, bboxes=[300, 200, 900, 700])

        print(result.masks.xy)      # poligon per mask

        print(result.boxes.xyxy)    # kotak rapat yang diturunkan dari mask
    - label: 'Enkode sekali, gunakan banyak prompt'
      language: python
      code: >
        from libreyolo import LibrePicoSAM3, SAMPLE_IMAGE


        model = LibrePicoSAM3()


        # set_image() menyimpan gambar sumber dalam cache; PicoSAM3 menjalankan
        satu

        # forward CNN penuh per kotak, sehingga ini menghemat pemuatan/dekode
        gambar,

        # bukan tahap encoder seperti pada family SAM lain.

        model.set_image(SAMPLE_IMAGE)

        a = model.predict(bboxes=[300, 200, 900, 700])

        b = model.predict(bboxes=[100, 100, 400, 400])

        model.reset_image()
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibrePicoSAM3


        model = LibrePicoSAM3()

        model.export(format="onnx", output_path="LibrePicoSAM3pico.onnx")


        # opset (default 13) dan dynamic (default True, hanya sumbu batch)
        adalah

        # satu-satunya argumen ekspor yang diterima family ini.
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        import numpy as np

        import onnxruntime as ort


        # PicoSAM3 mengekspor CNN ROI 96x96 mentahnya: roi_image -> mask_logits.

        # Tidak ada prapemrosesan/pascapemrosesan dari LibreYOLO untuk digunakan
        kembali,

        # karena export() tidak dirutekan kembali melalui LibreYOLO() seperti
        checkpoint detektor.

        session = ort.InferenceSession("LibrePicoSAM3pico.onnx")

        name = session.get_inputs()[0].name

        outputs = session.run(None, {name: np.zeros((1, 3, 96, 96),
        dtype=np.float32)})


        for meta, array in zip(session.get_outputs(), outputs):
            print(meta.name, array.shape)
source_hash: 5d60ff14fe61ba29
---

## Instalasi

PicoSAM3 memerlukan komponen tambahan `sam`: pengunduhan bobot milik LibreYOLO tetap
melalui alat Hugging Face dari `transformers`, meskipun inferensi berjalan pada CNN
native yang tidak memakai `transformers`.

```bash
pip install "libreyolo[sam]"
```

## Prediksi

`LibreSAM(...)` (atau `LibrePicoSAM3(...)` yang khusus untuk family ini) adalah
entry point terpisah dari `LibreYOLO(...)`: hasilnya merupakan segmenter yang dapat
diberi prompt, bukan detektor, karena forward pass di sini tidak bermakna tanpa
prompt. Tidak ada perintah CLI `libreyolo predict` untuk family ini; gunakan API Python.

<code-tabs name="predict" />

PicoSAM3 hanya menerima `bboxes=`; meneruskan `points=`, `labels=`, `masks=`,
`text=`, `multimask=True`, atau menghilangkan kotak untuk menyegmentasikan semuanya
akan memunculkan `ValueError` yang jelas karena mode tersebut tidak ada di model
upstream. `conf` memfilter berdasarkan kualitas mask yang diprediksi (IoU), bukan
confidence deteksi, dan harus berada di antara `0.0` dan `1.0`. Setiap mask memiliki id kelas
`0` dengan nama `"object"`. `train()`, `val()`, dan `track()` memunculkan
`NotImplementedError`; gunakan LibreSAM2 atau LibreSAM3 untuk prompt titik, teks, mask,
atau segment-everything. Lihat [prediksi](/docs/predict) untuk jenis sumber.

## Varian

Satu ukuran, pico, dengan input ROI tetap 96 px: PicoSAM3 menjalankan satu forward CNN
penuh per kotak, bukan mengenkode seluruh gambar satu kali.

## Ekspor

<export-matrix />

PicoSAM3 adalah satu-satunya family dalam tier SAM yang dapat diekspor: model ini
mengirim CNN ROI 96x96 mentahnya ke ONNX, `roi_image -> mask_logits`, tanpa NMS atau
pascapemrosesan mask yang tertanam. Family SAM lain memunculkan `NotImplementedError`
pada `export()` karena pemisahan encoder/decoder mereka belum memiliki kontrak ekspor
runtime yang ditentukan. Graph PicoSAM3 hasil ekspor tidak dimuat kembali melalui
`LibreYOLO()`; jalankan langsung dengan runtime seperti `onnxruntime`, sambil menerapkan
prapemrosesan ROI persegi dengan padding 10% yang sama seperti ditunjukkan di atas.

<code-tabs name="export" />

## Checkpoint

Setiap berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box>

PicoSAM3 didistilasi dari SAM 2.1 dan SAM 3 sebagai model teacher. LibreYOLO
tidak menyertakan atau mendistribusikan ulang kode maupun bobot kedua teacher dalam
family ini; hanya CNN student yang ringkas dan checkpoint hasil konversinya yang
disediakan.

</provenance-box>

## Sitasi

<citation-block />
