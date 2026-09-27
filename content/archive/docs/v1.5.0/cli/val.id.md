---
title: libreyolo val
seo_title: referensi perintah libreyolo val
description: >-
  Mengevaluasi checkpoint pada satu split dataset dari baris perintah: setiap
  argumen dengan nilai bawaannya, dan kunci metrik yang dikembalikan tiap task.
lead: >-
  Mengevaluasi satu model terhadap satu split dataset lalu mencetak metriknya.
  Kumpulan metriknya mengikuti task model, dan angkanya adalah angka yang
  dipakai untuk menyusun satu baris benchmark.
keywords:
  - libreyolo val cli
  - perintah validasi libreyolo
  - cara evaluasi model yolo di terminal
  - hitung mAP50-95 command line
  - argumen libreyolo val
last_verified: 1.5.0
meta:
  - label: Perintah
    value: libreyolo val
    mono: true
  - label: Wajib
    value: 'model, data'
    mono: true
  - label: Keluaran
    value: Metrik di stdout. Plot dan JSON COCO di runs/val/exp bila diminta
snippets:
  examples:
    - label: Dasar
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml
    - label: Plot dan JSON COCO
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml \
          imgsz=640 batch=8 save_json=true save_plots=true \
          project=runs/val name=yolo9s-coco8 exist_ok=true
    - label: Terbaca mesin
      language: bash
      code: |
        libreyolo val model=LibreYOLO9s.pt data=coco8.yaml json=true quiet=true
source_hash: f6507840568c3725
---

## Sinopsis

```bash
libreyolo val model=<name|path> data=<dataset.yaml> [key=value ...]
```

Argumen berupa pasangan `key=value`, dan bentuk POSIX juga berlaku, jadi
`batch=8` dan `--batch 8` adalah argumen yang sama.

## Argumen

| Argumen | Default | Arti |
|---|---|---|
| `model` | | Path bobot model atau nama CLI. Wajib |
| `data` | | Path ke YAML dataset (format YOLO, misalnya `coco8.yaml`). Wajib |
| `data_dir` | | Direktori dataset langsung, mengabaikan path yang ada di YAML |
| `split` | `val` | Split dataset: `val`, `test`, `train` |
| `batch` | `16` | Ukuran batch |
| `imgsz` | | Ukuran gambar: `640` (persegi) atau `480x640` (tinggi x lebar). Ukuran input model itu sendiri bila tidak diisi |
| `conf` | `0.001` | Ambang batas skor keyakinan (confidence) |
| `iou` | `0.6` | Ambang IoU untuk NMS |
| `max_det` | `300` | Prediksi maksimum per gambar setelah NMS |
| `eval_max_det` | | Batas atas evaluator COCO. Mengikuti konvensi AP@100 pycocotools bila tidak diisi |
| `faster_coco_eval` | `true` | Memakai backend C++ faster-coco-eval untuk metrik COCO bila terpasang; jika tidak ada, kembali ke pycocotools |
| `half` | `false` | Inferensi FP16 |
| `amp_dtype` | `float16` | Dtype autocast CUDA saat `half=true`: `float16` atau `bfloat16` |
| `save_json` | `false` | Menyimpan hasil JSON berformat COCO |
| `save_plots` | `false` | Menyimpan plot validasi: metrik, AP per kelas, confusion matrix, sampel |
| `workers` | `4` | Worker dataloader |
| `device` | `auto` | Perangkat |
| `project` | `runs/val` | Root direktori keluaran |
| `name` | `exp` | Nama eksperimen |
| `exist_ok` | `false` | Memakai ulang direktori keluaran |
| `allow_download_scripts` | `false` | Mengizinkan Python yang tertanam di blok download pada YAML dataset |
| `json` | `false` | Keluaran JSON ke stdout |
| `quiet` | `false` | Membungkam stderr |
| `verbose` | `true` | Keluaran verbose |
| `help_json` | `false` | Mencetak skema perintah sebagai JSON lalu keluar |

## Contoh

<code-tabs name="examples" />

## Catatan

### Apa saja metriknya

Kumpulan metrik yang dicetak mengikuti task model, dan keluaran JSON memakai
kunci yang sama.

Deteksi, segmentasi dan bounding box berorientasi melaporkan `mAP50`,
`mAP50_95`, `precision` dan `recall`. Bila sebuah model memprediksi lebih dari
satu jenis keluaran, grup per jenis muncul berdampingan sebagai `box_metrics`,
`mask_metrics` dan `obb_metrics`, masing-masing membawa empat kunci yang sama.

Klasifikasi melaporkan `accuracy_top1` dan `accuracy_top5`. Deteksi titik
melaporkan `precision`, `recall`, `f1`, `MLE`, `MAE`, `RMSE` dan `mAP_sweep`.
Kedalaman melaporkan `abs_rel`, `rmse`, `delta1`, `delta2` dan `delta3`.
Segmentasi semantik melaporkan `mIoU` dan `pixel_accuracy`. Restorasi
melaporkan `PSNR` dan `SSIM`.

Hasil JSON juga membawa `eval_backend`, yang menyebutkan library evaluasi COCO
beserta versinya yang menghasilkan angka tersebut, sehingga dua run bisa
dibandingkan dengan kepastian apakah keduanya dinilai oleh backend yang sama.

### Ambang batas

Nilai bawaan di sini adalah nilai bawaan untuk evaluasi, bukan untuk prediksi:
`conf` bernilai `0.001` dan `iou` bernilai `0.6`, sedangkan
[`libreyolo predict`](/docs/cli/predict) memakai `0.25` dan `0.45`. Menaikkan
`conf` ke ambang tampilan menurunkan recall dan ikut menurunkan mAP, jadi angka
yang dihasilkan dengan cara itu tidak sebanding dengan angka yang
dipublikasikan.

`imgsz` secara bawaan tidak diisi, yang berarti memakai ukuran input model itu
sendiri. Mengisinya membuat evaluasi berjalan pada ukuran yang diberikan, dan
begitulah cara sebuah checkpoint diukur di luar resolusi aslinya.

### Dataset yang mengunduh sendiri

YAML dataset yang kolom `download`-nya berupa URL akan mengunduh saat pertama
kali dipakai tanpa izin tambahan. YAML yang membawa skrip unduhan Python
tertanam memerlukan `allow_download_scripts=true`, dan perintah ini memberi
peringatan di stderr bahwa eksekusi kode lokal telah diaktifkan. `coco8.yaml`
dan `coco128.yaml` bawaan berbasis URL, jadi keduanya tidak memerlukan apa-apa.

### Keluaran dan kode keluar

stdout membawa metriknya; progres masuk ke stderr. `json=true` mencetak satu
objek dengan `schema_version`, dan `quiet=true` membungkam stderr.

Kode keluar adalah `0` bila berhasil, `2` untuk kesalahan penggunaan atau
konfigurasi, `3` bila dataset tidak ditemukan, `4` bila model tidak bisa
dimuat, dan `1` untuk kegagalan runtime lainnya.

Terkait: [`libreyolo train`](/docs/cli/train), yang menjalankan evaluasi yang
sama ini menurut jadwalnya sendiri lewat `eval_interval`.
