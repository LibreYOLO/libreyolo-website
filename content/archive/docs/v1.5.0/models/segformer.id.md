---
title: SegFormer
families:
  - segformer
seo_title: 'SegFormer: segmentasi semantik di LibreYOLO'
description: >-
  Gunakan SegFormer di LibreYOLO untuk segmentasi semantik ADE20K pada ukuran
  b0-b5. Instal, prediksi, latih, dan ekspor; bobot pretrained bersifat
  nonkomersial.
lead: >-
  SegFormer adalah transformer segmentasi semantik yang memasangkan encoder Mix
  Transformer (MiT) hierarkis dengan decode head all-MLP ringan, sehingga tidak
  memerlukan decoder berat dan positional encoding tetap seperti transformer
  segmentasi terdahulu. LibreYOLO mendukungnya untuk satu task, yaitu segmentasi
  semantik, dalam enam ukuran.
keywords:
  - SegFormer
  - segmentasi semantik
  - Mix Transformer
  - MiT
  - transformer segmentation
  - ADE20K
  - dense prediction
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        result = model(SAMPLE_IMAGE, save=True)

        mask = result.semantic_mask
        print(mask.data.shape, mask.classes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreSegformerb0-sem.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  train:
    - label: Python (fine-tuning)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 imgsz=512 batch=8
    - label: Dari nol
      language: python
      code: >
        from libreyolo.models.segformer.model import LibreSegformer


        # Tanpa model_path: inisialisasi acak, tidak ada yang diunduh. Ini
        satu-satunya jalur

        # menuju bobot yang bebas dari ketentuan nonkomersial checkpoint
        pretrained.

        model = LibreSegformer(size="b0", nb_classes=150)

        model.train(data="my-dataset.yaml", epochs=160, imgsz=512, batch=8)
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreSegformerb0-sem.pt data=my-dataset.yaml \
          epochs=160 device=0,1 batch=16
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mIoU"])
        print(metrics["metrics/pixel_accuracy"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSegformerb0-sem.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSegformerb0-sem.pt")
        model.export(format="onnx", imgsz=512)
        model.export(format="tensorrt", imgsz=512, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreSegformerb0-sem.pt format=onnx imgsz=512

        libreyolo export model=LibreSegformerb0-sem.pt format=tensorrt imgsz=512
        half=True
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory merutekan berdasarkan sufiks berkas, sehingga artefak hasil
        ekspor

        # dimuat seperti checkpoint lain dan mengembalikan objek Results yang
        sama.

        model = LibreYOLO("LibreSegformerb0-sem.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.semantic_mask.data.shape)
source_hash: c236895b991beabf
---

## Instalasi

SegFormer tidak memerlukan komponen tambahan opsional. Semua impornya tersedia dalam instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam cache lokal.

<code-tabs name="predict" />

`result.semantic_mask` menyimpan peta kelas padat: `.data` adalah tensor `(H, W)`
berisi id kelas pada ukuran gambar asli, sedangkan `.classes` mencantumkan id kelas yang
benar-benar ada. `result.boxes` adalah `None` karena tidak ada deteksi per instance.
`conf` dan `iou` diterima demi kesetaraan API tetapi tidak mengubah output: model
mengembalikan satu kelas per piksel, bukan deteksi per instance yang perlu difilter atau
dihilangkan duplikasinya. Lihat [prediksi](/docs/predict) untuk sumber, streaming, dan
penanganan hasil.

## Varian

Enam ukuran, b0 hingga b5, yang memperlebar dan memperdalam encoder Mix Transformer pada
setiap tahap sambil mempertahankan desain decode head all-MLP yang sama.

<checkpoint-table />

## Pelatihan

Secara default, `train()` melakukan fine-tuning pada checkpoint yang dipublikasikan.
Sebagai gantinya, jangan berikan `model_path` ke `LibreSegformer(...)` agar model dibuat
dengan encoder dan head yang diinisialisasi secara acak serta dilatih dari nol. Ini adalah
satu-satunya jalur menuju bobot yang tidak memiliki pembatasan nonkomersial checkpoint
pretrained (lihat [Lisensi](#licensing)).

<code-tabs name="train" />

Jika dibiarkan, pelatih mengikuti resep ADE20K dari makalah SegFormer: AdamW pada learning
rate dasar backbone dengan decode head dilatih pada 10x laju tersebut, weight decay di semua
bagian kecuali LayerNorm dan konvolusi posisi Mix-FFN, serta jadwal decay linear dengan
warmup. Konvergensi untuk ukuran lebih besar, b3 hingga b5, belum divalidasi secara end-to-end.

Lihat [pelatihan](/docs/train) untuk dataset, augmentasi, multi-GPU, dan logger.

## Validasi

`val()` mengembalikan dictionary dengan key `metrics/`: mIoU dan akurasi piksel,
yang diukur terhadap dataset apa pun dalam format yang digunakan untuk pelatihan.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Artefak hasil ekspor dimuat kembali melalui `LibreYOLO()` berdasarkan sufiks berkasnya,
sehingga berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan mengembalikan
`Results` yang sama. [Ekspor](/docs/export) mencantumkan argumen yang diterima setiap format.

<code-tabs name="export" />

## Checkpoint

Setiap berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box>

Encoder dan decode head LibreSegformer adalah port PyTorch dari implementasi SegFormer
Apache-2.0 milik Hugging Face Transformers, bukan dari NVlabs/SegFormer: repositori asli
NVIDIA tidak pernah dibaca atau disalin dan hanya dicantumkan di sini untuk atribusi kepada
penulis makalah. Hanya checkpoint pretrained di atas yang membawa pembatasan nonkomersial
NVIDIA; arsitektur dan kode LibreYOLO sendiri tetap berlisensi MIT.

</provenance-box>

## Sitasi

<citation-block />
