---
title: DINO-DETR
families:
  - dinodetr
seo_title: 'DINO-DETR: prediksi dan ekspor di bawah Apache-2.0'
description: >-
  Jalankan DINO-DETR di LibreYOLO untuk deteksi objek. Instalasi, prediksi,
  validasi dan ekspor tiga ukuran denoising-anchor, semuanya berlisensi
  Apache-2.0.
lead: >-
  DINO-DETR, yang dipublikasikan IDEA Research dengan nama DINO, memadukan
  pelatihan contrastive denoising dengan mixed query selection di atas sparse
  attention milik Deformable DETR. LibreYOLO menyediakan tiga ukuran untuk
  deteksi, khusus inferensi.
keywords:
  - DINO-DETR
  - DINO
  - detection transformer
  - denoising anchor boxes
  - mixed query selection
  - deteksi objek python
  - IDEA Research
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreDINODETRr50.pt")
        result = model(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDINODETRr50.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDINODETRr50.pt")

        # val() mengembalikan dict biasa, bukan objek
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["metrics/precision"], metrics["metrics/recall"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDINODETRr50.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDINODETRr50.pt")
        model.export(format="onnx", imgsz=800)
        model.export(format="tensorrt", imgsz=800, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreDINODETRr50.pt format=onnx imgsz=800

        libreyolo export model=LibreDINODETRr50.pt format=tensorrt imgsz=800
        half=True
    - label: Memakai berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        # Factory-nya memilih berdasarkan sufiks berkas, jadi artefak hasil
        ekspor

        # dimuat seperti checkpoint biasa dan mengembalikan objek Results yang
        sama.

        model = LibreYOLO("LibreDINODETRr50.onnx")

        result = model(SAMPLE_IMAGE)


        print(result.boxes.xyxy)
source_hash: dda176ebee3a83de
---

## Instalasi

DINO-DETR tidak memerlukan extra opsional. Semua yang diimpornya sudah tersedia
di instalasi dasar, memakai core multi-scale deformable attention pure-PyTorch
yang sama dengan family Deformable DETR di LibreYOLO.

```bash
pip install libreyolo
```

Memasang `libreyolo[hub-kernels]` bersifat opsional. Begitu paket `kernels`
tersedia, LibreYOLO mengambil kernel multi-scale deformable attention yang sudah
dikompilasi dari Hugging Face Hub saat runtime dan memakainya sebagai pengganti
core pure-PyTorch; `LIBREYOLO_HUB_KERNELS=0` mematikannya kembali.

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali dipakai lalu disimpan di
cache lokal.

<code-tabs name="predict" />

Objek `Results` yang dikembalikan sama dengan yang dikembalikan setiap family,
jadi mengganti detektor dengan yang lain hanya butuh perubahan satu baris.
`conf` dan `max_det` menyaring query selection; `iou` tetap diterima demi
keseragaman API tetapi tidak berpengaruh, karena decoder-nya adalah set
predictor tanpa langkah NMS. Lihat
[prediksi](/docs/predict) untuk source, streaming dan penanganan hasil.

Di LibreYOLO, DINO-DETR hanya untuk inferensi. Upstream melatihnya dengan
contrastive denoising dan Hungarian matching; resep itu tidak diimplementasikan
di sini, jadi `train()` memunculkan `NotImplementedError`.

## Varian

Tiga checkpoint, semuanya pada resolusi input yang sama. `r50` dan `r50s5`
memakai backbone ResNet-50 yang sama dan berbeda pada jumlah skala feature map
yang masuk ke decoder, empat berbanding lima. `swinl` mengganti backbone-nya
dengan Swin-L dan juga mengambil lima skala.

## Validasi

`val()` mengembalikan dictionary berisi key `metrics/` yang mencakup presisi,
recall, mAP 50 dan mAP 50-95, diukur terhadap dataset apa pun dalam format yang
dipakai saat pelatihan.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Artefak hasil ekspor dimuat kembali lewat `LibreYOLO()` berdasarkan sufiks
berkasnya, jadi berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan
mengembalikan `Results` yang sama. [Ekspor](/docs/export) mencantumkan argumen
yang diterima setiap format.

<code-tabs name="export" />

## Checkpoint

Semua berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box>

Ketiga checkpoint resminya berasal dari folder rilis Google Drive milik para
penulis, bukan dari model card Hugging Face. Repositori upstream mendeklarasikan
Apache-2.0 di tingkat repositori tetapi tidak melampirkan berkas lisensi maupun
metadata lisensi pada checkpoint itu sendiri, jadi dasar redistribusinya adalah
deklarasi tingkat repositori tersebut, bukan pemberian lisensi khusus per
checkpoint. Setiap mirror LibreYOLO menyertakan teks lisensi Apache-2.0 upstream
secara verbatim beserta catatan yang menjelaskan hal ini.

</provenance-box>

## Sitasi

<citation-block />
