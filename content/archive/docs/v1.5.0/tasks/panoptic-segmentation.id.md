---
title: Segmentasi panoptik
seo_title: Segmentasi panoptik di LibreYOLO
description: >-
  Tetapkan setiap piksel ke satu segment di LibreYOLO: family yang melayani
  task, format dataset COCO-panoptic, serta pemanggilan prediksi dan validasi.
lead: >-
  Segmentasi panoptik menetapkan setiap piksel ke tepat satu segment yang tidak
  tumpang tindih, menyatukan instance objek yang dapat dihitung dengan region
  background amorf. Kunci task-nya adalah panoptic.
keywords:
  - segmentasi panoptik Python
  - panoptic quality
  - segmentasi things dan stuff
  - format COCO panoptic
  - peta segment id
  - metrik PQ
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Akhiran -panoptic pada nama berkas memilih task, sehingga argumen task
        # tidak diperlukan.
        model = LibreYOLO("LibreEoMTl-panoptic.pt")
        result = model(SAMPLE_IMAGE, save=True)

        pan = result.panoptic
        print(pan.data.shape)       # (H, W) id segment
        print(pan.segments_info)    # [{"id": ..., "category_id": ...}, ...]
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreEoMTl-panoptic.pt save=True \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
    - label: Satu segment pada satu waktu
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        result = LibreYOLO("LibreEoMTl-panoptic.pt")(SAMPLE_IMAGE)
        pan = result.panoptic

        for segment in pan.segments_info:
            pixels = pan.segment_mask(segment["id"])   # boolean (H, W)
            print(result.names[segment["category_id"]], int(pixels.sum()))
    - label: Checkpoint lebih kecil
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreEoMTs-panoptic.pt")
        result = model(SAMPLE_IMAGE)

        print(len(result.panoptic.segment_ids))
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreEoMTl-panoptic.pt")

        # val() mengembalikan dict biasa, bukan objek.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PQ"])
        print(metrics["metrics/SQ"], metrics["metrics/RQ"])
        print(metrics["metrics/PQ_things"], metrics["metrics/PQ_stuff"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreEoMTl-panoptic.pt data=my-dataset.yaml
source_hash: b8adc9ccde7a4e6c
---

## Definisi

Segmentasi panoptik adalah gabungan dua task segmentasi lainnya. Setiap piksel
memperoleh tepat satu segment, segment tidak pernah tumpang tindih, dan segment
merupakan thing, yaitu instance objek yang dapat dihitung, atau stuff, yaitu
region amorf seperti langit atau jalan. Karena itu, task ini lebih ketat daripada
[segmentasi instance](/docs/tasks/instance-segmentation), yang membiarkan piksel
background tanpa assignment dan memungkinkan mask tumpang tindih, serta lebih
ketat daripada [segmentasi semantik](/docs/tasks/semantic-segmentation), yang
memberi label setiap piksel tetapi menggabungkan instance bersentuhan dari satu kelas.

`panoptic` adalah kunci task kanonis, dan akhiran `-panoptic` pada nama berkas
checkpoint memilihnya, sehingga `task=` tidak diperlukan saat memuat bobot terbitan.

`predict()` mengisi `result.panoptic`. `.data` adalah peta id-segment integer
`(H, W)` pada kanvas gambar asli. `.segments_info` adalah daftar dict, satu per
segment, yang masing-masing setidaknya memuat `{"id", "category_id"}`, dengan
`id` cocok dengan nilai dalam peta dan `category_id` mengindeks `result.names`.
`.segment_ids` mencantumkan id yang ada dalam urutan terurut dan
`.segment_mask(id)` mengembalikan pilihan boolean `(H, W)` untuk satu segment.
Id segment `0` adalah nilai void: piksel tanpa label yang dikecualikan dari
metrik dan tidak disertakan dalam `.segment_ids`.

Thing atau stuff merupakan properti kategori, bukan segment individual. Properti
ini dibawa dalam metadata kategori kumpulan label, dan payload prediksi dapat
menyalinnya ke setiap segment sebagai `"isthing"` demi kemudahan, tetapi metadata
kategori tetap menjadi sumber otoritatif.

## Model

[EoMT](/docs/models/eomt) adalah family yang melayani task ini melalui
`LibreYOLO()`. Model ini berjalan pada paket dasar dan menyertakan checkpoint
panoptik dalam tiga ukuran, s, b, dan l, yang dilatih pada COCO.

[SenseNova-Vision](/docs/models/sensenova-vision) juga menghasilkan peta panoptik.
Model generatif berbasis prompt ini memiliki factory sendiri, `LibreVLM`, dan
extra sendiri; jika tidak ada vocabulary yang ditetapkan, model kembali ke
kategori panoptik COCO yang digunakan dalam tuning. Bobotnya bersifat
nonkomersial. Latensi per gambar jauh lebih tinggi daripada segmenter khusus,
karena setiap prediksi merupakan diffusion decode.

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam
cache lokal.

<code-tabs name="predict" />

`conf` memfilter pemilihan query. Lihat [prediksi](/docs/predict) untuk sumber,
streaming, dan penanganan hasil.

## Format dataset

LibreYOLO menggunakan format COCO-panoptic apa adanya, dari Kirillov et al.,
CVPR 2019. Tidak ada tata letak panoptik khusus LibreYOLO.

```text
dataset/
  data.yaml
  images/
    val/000000000139.jpg
  annotations/
    panoptic_val.json
    panoptic_val/000000000139.png
```

Setiap gambar dipasangkan dengan satu PNG RGB beresolusi sama, dengan warna
setiap piksel mengodekan id segment pemiliknya:

```text
segment_id = R + 256 * G + 256 * 256 * B
```

Id segment `0`, yaitu RGB hitam, merupakan void: piksel tanpa label yang tidak
memberi reward maupun penalti pada prediksi. Setiap piksel lain tepat menjadi
milik satu segment.

JSON mencantumkan PNG id-segment dan segment di dalamnya untuk setiap gambar:

```json
{
  "images":      [{"id": 139, "file_name": "000000000139.jpg"}],
  "annotations": [{"image_id": 139, "file_name": "000000000139.png",
                   "segments_info": [
                     {"id": 3226956, "category_id": 1, "area": 2840,
                      "bbox": [413, 158, 53, 138], "iscrowd": 0}]}],
  "categories":  [{"id": 1, "name": "person", "isthing": 1}]
}
```

`annotations[].file_name` menamai PNG dalam direktori panoptik, dan
`segments_info[].id` cocok dengan nilai dalam PNG tersebut. `iscrowd` menandai
region kelompok: region tersebut tidak pernah dihitung sebagai false negative,
dan prediksi yang sebagian besar menutupinya bukan false positive. `isthing`
berada pada `categories`, bukan pada segment individual.

YAML menunjuk ke keduanya:

```yaml
path: dataset
val: images/val
annotations:
  val: annotations/panoptic_val.json
panoptic_dir:
  val: annotations/panoptic_val
names:
  0: person
  1: bicycle
```

`annotations` dan `panoptic_dir` masing-masing menerima satu path atau pemetaan
per split. Id kategori mentah COCO biasanya tidak berurutan, sedangkan model
memprediksi rentang berurutan `0..nc-1`, sehingga id dipetakan ulang melalui
`names` berdasarkan nama kategori. Kategori JSON yang tidak ada dalam `names`
merupakan error, bukan diabaikan diam-diam, karena mengabaikannya akan dinilai
sebagai false negative permanen.

Loader kanonis adalah `libreyolo.data.PanopticDataset`.

## Pelatihan

Saat ini tidak ada family yang melatih segmentasi panoptik di LibreYOLO:
`train()` EoMT memunculkan `NotImplementedError`, sehingga checkpoint panoptik
digunakan sebagaimana diterbitkan.

## Validasi

`val()` mengembalikan dictionary biasa berisi kunci `metrics/`, yang dihitung pada
resolusi ground truth di split bernama `val` dalam YAML dataset. Segment prediksi
dan ground truth dari kategori yang sama cocok jika IoU-nya melebihi 0,5, dan
pencocokan tersebut unik.

<code-tabs name="val" />

`metrics/PQ` adalah Panoptic Quality dan menjadi angka utama. Dalam satu kategori,
nilai ini merupakan hasil kali dua faktor. Segmentation quality adalah mean IoU
pada segment yang cocok dan menunjukkan seberapa baik bentuk hasil pencocokan
sejajar. Recognition quality adalah `TP / (TP + 0.5 FP + 0.5 FN)`, yaitu skor F1
dari pencocokan itu sendiri, dan menunjukkan berapa banyak segment yang ditemukan.
Ketiga angka kemudian dirata-ratakan pada kategori yang muncul dan dilaporkan
sebagai `metrics/PQ`, `metrics/SQ`, dan `metrics/RQ`, sehingga PQ yang dilaporkan
merupakan rata-rata hasil kali per kategori, bukan hasil kali dua rata-rata terlapor.

`metrics/PQ_things` dan `metrics/PQ_stuff` merata-ratakan PQ per kategori yang
sama secara terpisah pada kategori thing dan stuff, sedangkan `metrics/categories`
menghitung kategori yang muncul dan karenanya masuk ke rata-rata. Dictionary juga
memuat `fitness`, salinan nilai PQ.

## Ekspor

Checkpoint panoptik tidak dapat diekspor. `export()` memunculkan
`NotImplementedError` untuk task ini karena output query-mask belum memiliki
kontrak ekspor runtime. Task semantic EoMT dapat diekspor; lihat
[segmentasi semantik](/docs/tasks/semantic-segmentation) dan
[ekspor dan deployment](/docs/export).

