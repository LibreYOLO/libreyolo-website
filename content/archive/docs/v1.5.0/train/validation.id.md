---
title: Validasi dan metrik
seo_title: Validasi dan metrik di LibreYOLO
description: >-
  Jalankan val() pada model apa pun, baca kunci metrik yang dikembalikan setiap
  task, pilih backend evaluasi, dan nyalakan validasi loss bersama dengan metrik
  akurasi.
lead: >-
  Validasi menjalankan model pada pembagian dataset melalui val() dan
  mengembalikan kamus datar dari kunci metrik dan nilai float. Kuncinya adalah
  string literal, dan kunci yang Anda dapatkan tergantung pada task, bukan
  family.
keywords:
  - map50-95
  - evaluasi coco
  - metrik validasi
  - faster-coco-eval
  - pycocotools
  - validasi loss
  - miou
  - kualitas panoptic
  - akurasi top1
last_verified: 1.5.0
snippets:
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
        print(metrics["speed/total_ms"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml
    - label: Pada pembagian lain
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        metrics = model.val(data="coco8.yaml", split="train", batch=4)

        print(metrics)
  valloss:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, val_loss=True)
  json:
    - label: Tulis prediksi dalam format COCO
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.val(data="coco8.yaml", save_json=True, save_dir="runs/val/exp")
source_hash: d907183492fa3f57
---

## Jalankan validasi

`val()` mengambil dataset dan mengembalikan metrik.

<code-tabs name="val" />

Nilai kembaliannya adalah `dict[str, float]` biasa. Setiap kunci bersifat literal, jadi bacalah
berdasarkan nama daripada posisi.

Argumen utama adalah `data`, `split`, `batch`, `imgsz`, `conf`, `iou`,
`workers`, `device`, `augment`, `save_json` dan `verbose`. `conf` secara default adalah
`0.001` dan `iou` ke `0.6`, keduanya jauh lebih longgar daripada default prediksi, karena
pencarian mAP membutuhkan ekor dengan kepercayaan rendah. `imgsz` secara default adalah ukuran input model itu sendiri
daripada angka tetap. `split` menerima `val`, `test` atau `train`
dan tidak ada lainnya.

Setiap bidang lain dari konfigurasi validasi diteruskan sebagai argumen kata kunci,
termasuk `save_dir`, `max_det`, `eval_max_det`, `half`, `amp_dtype`, `cache`
dan `save_plots`.

## Kunci metrik per task

Deteksi mengembalikan family COCO dari angka:

```text
metrics/mAP50-95   metrics/mAP50    metrics/mAP75
metrics/mAP_small  metrics/mAP_medium  metrics/mAP_large
metrics/AR1  metrics/AR10  metrics/AR100  metrics/AR_max_det
metrics/AR_small  metrics/AR_medium  metrics/AR_large
metrics/precision  metrics/recall
metrics/precision(B)  metrics/recall(B)  metrics/mAP50(B)  metrics/mAP50-95(B)
```

Dua dari mereka adalah jebakan. `metrics/precision` dan `metrics/recall` adalah alias yang dijaga
untuk kompatibilitas belakang: mereka membawa nilai mAP 50-95 dan AR@100, bukan
pasangan presisi dan recall. Gunakan kunci bernama.

Segmentasi instance mengembalikan angka mAP dan AR di atas sebagai angka mask
di bawah kunci tanpa akhiran, dengan versi kotak berada di bawah akhiran `(B)` dan
diulang di bawah `(M)`. Presisi dan recall hanya ada dalam bentuk akhiran
untuk task ini, seperti `metrics/precision(B)`/`metrics/recall(B)` dan
`metrics/precision(M)`/`metrics/recall(M)`, dan kedua pasangan membawa nilai alias
yang sama seperti detect: pasangan `(B)` adalah bounding box mAP50-95 dan bounding box AR@100, pasangan
`(M)` adalah mask mAP50-95 dan mask AR@100.

| Task | Kunci |
|---|---|
| detect | `metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75`, plus pembagian ukuran dan recall di atas |
| segment | versi mask dari kunci detect di atas (kunci tanpa akhiran adalah mask); `precision`/`recall` hanya ada sebagai `(B)`/`(M)`, keduanya dialiaskan dengan cara yang sama |
| pose | `metrics/keypoints_mAP50-95`, `metrics/keypoints_mAP50`, `metrics/keypoints_mAP75`, `metrics/keypoints_mAP_M`, `metrics/keypoints_mAP_L`, dan kunci `keypoints_AR` yang sesuai |
| obb | `metrics/mAP50-95`, `metrics/mAP50`, `metrics/mAP75`, `metrics/precision`, `metrics/recall`, plus salinan dengan akhiran `(OBB)` |
| classify | `metrics/accuracy_top1`, `metrics/accuracy_top5` |
| semantic | `metrics/mIoU`, `metrics/pixel_accuracy` |
| panoptic | `metrics/PQ`, `metrics/SQ`, `metrics/RQ`, `metrics/PQ_things`, `metrics/PQ_stuff`, `metrics/categories` |
| depth | `metrics/abs_rel`, `metrics/rmse`, `metrics/delta1`, `metrics/delta2`, `metrics/delta3` |
| normal | `metrics/mean_angular_error`, `metrics/median_angular_error`, `metrics/within_11_25`, `metrics/within_22_5`, `metrics/within_30` |
| edge | `metrics/ODS`, `metrics/OIS`, `metrics/best_threshold` |
| restore | `metrics/PSNR`, `metrics/SSIM` |
| matte | `metrics/MAE`, `metrics/Smeasure` |
| ocr | `metrics/det_precision`, `metrics/det_recall`, `metrics/det_hmean`, `metrics/e2e_precision`, `metrics/e2e_recall`, `metrics/e2e_f1`, `metrics/rec_1-NED` |
| titik | `metrics/precision`, `metrics/recall`, `metrics/f1`, `metrics/MLE`, `metrics/MAE`, `metrics/RMSE`, ditambah kunci sweep mAP |

OBB `metrics/precision` dan `metrics/recall` bukan alias: mereka adalah
presisi nyata dan recall pada IoU 0,50, diambil pada titik operasi paling longgar
(setiap prediksi yang bertahan `conf`, default `0.001`). Salinan yang berakhiran `(OBB)`
mengulangi keempat nilai yang sama dengan nama spesifik task, konvensi yang sama
seperti `(B)` dan `(M)` di atas.

`accuracy_top5` sebenarnya top-`min(5, num_classes)`, jadi pada dataset tiga kelas
itu adalah top-3, yang dipenuhi setiap sampel dan akibatnya terbaca 1,0.

Kunci sweep titik task dibangun dari ambang jarak, jadi dengan
secara default terbaca `metrics/mAP@[0.01:0.10]` dan tombol ambang tunggal terbaca
`metrics/mAP@0.01`. Melewati `dist_thresholds` mengubah kedua string.

Sebagian besar tugas juga mengembalikan kunci `fitness`, angka tunggal terbaik-checkpoint
seleksi digunakan secara default. Deteksi, segmentasi, dan OBB tidak membawa satu;
keluarga mereka dipilih pada `metrics/mAP50-95`, yang dilakukan oleh kamus mereka
kembali. Pose tidak mengembalikan `fitness` maupun `metrics/mAP50-95`; pelatihnya
atur `best_metric_key` ke `metrics/keypoints_mAP50-95` sebagai gantinya.

## Tombol pintas

Setiap validator menambahkan waktu:

```text
speed/preprocess_ms   speed/inference_ms   speed/postprocess_ms
speed/total_ms        speed/total_s        speed/images_seen
```

Ini adalah milidetik per gambar yang dirata-ratakan selama seluruh proses. Mereka menggambarkan mesin
dan pengaturan yang Anda jalankan, sehingga angka yang diambil dari mereka hanya berarti jika dilaporkan
dengan perangkat kerasnya, batch ukuran dan presisi.

## Backend Evaluasi

Metrik deteksi dan segmentasi dihitung melalui evaluator COCO, dan
`faster_coco_eval=True`, default-nya, memilih backend C++ ketika
paket `faster-coco-eval` terinstal. Ketika tidak, proses berjalan kembali ke
pycocotools dengan satu peringatan per proses:

```text
faster_coco_eval requested but not installed; falling back to pycocotools.
Install with: pip install faster-coco-eval
```

Backend yang sebenarnya dijalankan dicatat pada model sebagai `last_eval_backend`, dan
CLI melaporkannya dalam output untuk tugas gaya deteksi. Atur
`LIBREYOLO_FASTER_COCO_EVAL` untuk mengganti nilai konfigurasi dari lingkungan.

`iou_thresholds` hanya dihormati pada jalur OBB. Jalur COCO dievaluasi melalui
sweep 0,50 hingga 0,95 yang tetap sendiri dan mengabaikan nilainya.

## Validasi loss

Secara default, validasi hanya melaporkan akurasi. `val_loss=True` juga menghitung
tujuan pelatihan family pada batch validasi.

<code-tabs name="valloss" />

Ini mengeluarkan `metrics/loss` ditambah satu `metrics/loss/<component>` per istilah, diberi bobot
persis seperti bobot pelatihan, sehingga komponen-komponennya berjumlah total. Melalui
logger, mereka muncul sebagai `val/loss` dan `val/loss/<component>`, dan `libreyolo
monitor` overlays `metrics/loss` with `train/loss`.

Komponen-komponen ini adalah milik family sendiri:

| Task | Families | Komponen |
|---|---|---|
| detect | `yolo9`, `yolo9_p2`, `yolo9_e2e` | `box`, `cls`, `dfl` |
| detect | `yolonas` | `cls`, `iou`, `dfl` |
| mendeteksi | `rfdetr` | `ce`, `bbox`, `giou` |
| mendeteksi | `rtdetr`, `rtdetrv2` | `vfl`, `bbox`, `giou` |
| mendeteksi | `dfine` | `vfl`, `bbox`, `giou`, `fgl`, `ddf` |
| mendeteksi | `domedetr` | `vfl`, `bbox`, `giou`, `fgl`, `ddf`, `defe_density`, `defe_reg` |
| mendeteksi | `deim`, `deimv2`, `rtdetrv4`, `ec` | `mal`, `bbox`, `giou`, `fgl`, `ddf` |
| mendeteksi | `rtmdet` | `cls`, `bbox` |
| mendeteksi | `picodet` | `cls`, `bbox`, `dfl` |
| mendeteksi | `yolox` | `iou`, `obj`, `cls`, `l1` |
| mendeteksi | `yolo7` | `iou`, `obj`, `cls` |
| menunjukkan | `fomo` | `ce` |
| klasifikasi | `resnet`, `convnext`, `mobilenetv4`, `efficientnetv2` | `ce` |
| semantik | `segformer`, `lingbotvision`, `dinov2` | `sem` |
| pulihkan | `nafnet` | `restore` |

Ini dimatikan secara default karena penugasan target menambahkan waktu dan memori ke
validasi. Validator menggunakan kembali output model yang sudah dihasilkan untuk
metrik akurasi daripada menjalankan lintasan maju kedua, ia dijalankan di bawah
`no_grad` pada model evaluasi atau EMA, dan di bawah pelatihan multi-GPU itu
dihitung secara lokal pada peringkat 0 tanpa kolektif. Pemilihan Best-checkpoint tetap
pada metrik akurasi.

Tiga hal yang dengan sengaja tidak dilakukannya. Ia tidak pernah termasuk denoising-kontrastas
persyaratan, karena itu memerlukan ground truth pada waktu maju dan validasi
maju tanpa itu. Ini melaporkan model dalam mode evaluasi, jadi di mana family's
train dan eval benar-benar berbeda di depan, dalam statistik BatchNorm atau stokastik
kedalaman, angka tersebut mencerminkan mode evaluasi; itu adalah perbandingan yang dimaksud. Dan sebuah task
family belum menerapkannya sehingga menyebabkan kesalahan konfigurasi saat pengaturan
daripada diam-diam melewatkan:

```text
val_loss=True currently supports RF-DETR detection only; segment, pose, OBB,
classify, and semantic tasks are not supported
```

FOMO adalah pengecualian yang tidak mengubah apa pun: validatornya selalu menghitung ini
loss, dan `val_loss=True` hanya memengaruhi tombol mana yang diterbitkan dengannya.

Validasi yang ditingkatkan dan validasi loss tidak dapat dikombinasikan, dan meminta keduanya
meningkatkan.

## Berkas menulis validasi

`val()` selalu menulis `config.yaml` ke direktori simpanannya, secara default ke
`runs/val/<model>_<size>_<timestamp>` ketika `save_dir` tidak diberikan.

<code-tabs name="json" />

`save_json=True` menulis `predictions.json` untuk deteksi, dan
`predictions_bbox.json` ditambah `predictions_masks.json` untuk segmentasi. OBB tidak
mendukungnya dan menyatakannya.

`save_plots=True` menulis ke dalam subdirektori `plots/`. Deteksi mendapatkan
`box_metrics.png`, AP per kelas dan grafik recall, presisi-recall dan
kurva kepercayaan, matriks kebingungan, dan contoh gambar beranotasi ketika OpenCV
terpasang. Segmentasi menambahkan salinan sisi mask dari masing-masing, dan pose mendapatkan miliknya sendiri
. Validator lain tidak mengimplementasikan plot; klasifikasi,
semantik, panoptik, kedalaman, normal, tepi, pemulihan, matte, OCR, OBB, dan poin semuanya
tidak menulis apa pun di sana. Kegagalan plot akan memberi peringatan dan tidak pernah menghentikan jalannya.

## Validasi selama pelatihan

Pelatihan memvalidasi setiap `eval_interval` epoch terhadap dataset's `val`
split, dan metrik yang dihasilkannya adalah yang menjadi penggerak pemilihan `best.pt`,
`patience` early stop, dan `val/` kunci di setiap logger. Validasi dijalankan
pada bobot EMA ketika EMA aktif.

Lihat [Hyperparameters](/docs/train/hyperparameters) untuk `eval_interval`,
`patience` dan `save_plots`, serta [Experiment loggers](/docs/train/loggers) untuk
di mana angkanya pergi.

## Terkait

- [Dataset](/docs/train/datasets) untuk kunci split dan validator format dibaca.



