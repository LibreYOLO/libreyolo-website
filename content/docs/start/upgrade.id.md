---
title: Meningkatkan ke 1.6.0
seo_title: Tingkatkan LibreYOLO 1.5.0 ke 1.6.0
description: Langkah migrasi prapemrosesan, default pelatihan, direktori proses, aset yang dipatok,
  QAT, dan loader data di LibreYOLO 1.6.0.
lead: Versi 1.6.0 mengubah prapemrosesan, default pelatihan, dan penanganan checkpoint. Validasi
  ulang baseline tersimpan dan atur default lama secara eksplisit saat mereproduksi proses sebelumnya.
keywords:
  - upgrade libreyolo
  - migrasi libreyolo 1.5.0
  - allow_experimental dihapus
  - breaking change libreyolo
  - yolox bn eps
  - default faster-coco-eval
last_verified: 1.6.0
source_hash: f4fda6ef286113ab
---

## 1.5.0 ke 1.6.0

- Tingkatkan lingkungan yang dipatok di bawah ONNX Runtime 1.18 ke `onnxruntime>=1.18.0` sebelum memasang extra ONNX.

- SAM 3D Body menerima kumpulan aset snapshot yang sudah ditinjau. Pasang `libreyolo[hf]`, dapatkan akses terbatas, lalu gunakan pengambilan otomatis, atau sediakan direktori snapshot tanpa perubahan dan aset MHR yang dipatok pada sistem berkas lokal tepercaya.

- Jalankan ulang validasi setelah koreksi pengubahan ukuran RF-DETR non-pose dan normalisasi pengklasifikasi. Periksa ulang ambang skor keyakinan yang diterapkan sebelum membandingkan hasil dengan 1.5.0; tidak ada flag prapemrosesan lama.

- Deteksi D-FINE, DEIM, RT-DETRv4, dan YOLO-NAS mengaktifkan FP16 AMP secara default. Berikan `amp=False` untuk mempertahankan FP32.

- Untuk pilihan fine-tuning YOLO9 sebelumnya, atur `aux_weight=0`, `max_labels=100`, dan `warmup_momentum=0.937`. Atur `letterbox_pad="topleft"` jika konversi baru bertanda center harus mereproduksi geometri lama. Checkpoint lama dengan satu head melanjutkan pelatihan dengan graf aslinya.

- RF-DETR dan DINOv2 membuat direktori proses bernomor dengan nama family. Perbarui pengguna path artefak, atau atur `output_dir="runs/train", exist_ok=True` untuk mempertahankan lokasi lama dan perilaku penggunaan ulang.

- Pertahankan `mixup + cutmix <= 1` untuk klasifikasi; kombinasi tidak valid kini menimbulkan galat saat penyiapan.

- QAT menonaktifkan EMA, SyncBatchNorm, dan perataan checkpoint. Gunakan checkpoint best/last QAT tanpa bergantung pada status tersebut.

- Loader kustom dengan hook mutasi dataset harus memakai `persistent_workers=False` atau membangun ulang worker setelah mutasi. Salinan persisten multi-worker yang tidak kompatibel kini menimbulkan galat.

Lihat [changelog](/docs/changelog) untuk rilis lengkap dan [mengimpor bobot](/docs/migrate) untuk konversi checkpoint.

## 1.4.0 ke 1.5.0

Halaman ini membahas upgrade LibreYOLO. Untuk memuat checkpoint dari project
upstream, lihat [impor bobot yang ada](/docs/migrate), yang merupakan topik lain.

Entri rilis lengkap tersedia pada [changelog](/docs/changelog). Bagian berikut
hanya membahas tindakan yang diperlukan.

### Perubahan kode yang wajib dilakukan

#### `allow_experimental=True` tidak lagi ada

Gate konfirmasi telah dihapus beserta mekanisme
`ddp_aware(experimental_key=...)` di baliknya. Pelatihan dan ekspor EC, RTMDet,
PicoDet, dan FOMO sebelumnya memerlukan argumen tersebut, sehingga semua script
yang melatih family itu terpengaruh.

```python
# 1.4.0
model.train(data="data.yaml", epochs=100, allow_experimental=True)

# 1.5.0: hapus argumen
model.train(data="data.yaml", epochs=100)
```

Tidak ada shim deprecation. Pemanggilan yang masih memberikannya memunculkan
`TypeError`. `BaseModel.EXPERIMENTAL_WEIGHT_FILENAMES` juga dihapus. Hook
`get_download_notice()` tetap ada dan masih diganti oleh MiDaS, SegFormer,
dan YOLO9-P2.

Tingkat dukungan tetap dipublikasikan, tetapi bukan lagi argumen: lihat
[tier stabilitas](/docs/reference/stability-tiers).

#### Tier ekspor `"experimental"` tidak lagi ada

```python
from libreyolo.export.support import Tier

# 1.4.0: Literal["validated", "experimental", "blocked"]
# 1.5.0: Literal["validated", "available", "blocked"]
```

Kode yang bercabang berdasarkan string tier harus membaca `"available"` di
tempat yang sebelumnya membaca `"experimental"`. `BaseExporter` tidak lagi
menghasilkan `RuntimeWarning` untuk format tersebut. Status per format tersedia
dalam [matriks ekspor](/docs/reference/export-matrix).

#### `pretrained=False` bersama `resume` kini ditolak

Kombinasi ini sebelumnya berjalan secara tidak koheren. Kini muncul error:

```
ValueError: pretrained=False cannot be combined with resume.
```

Pilih salah satu. `pretrained=False` memulai dari initialization baru dengan
seed, yang pada 1.5.0 berfungsi bagi setiap family yang dapat dilatih, bukan
hanya tiga. `resume` melanjutkan run yang terputus dari checkpoint. Keduanya
didokumentasikan dalam [pelatihan](/docs/train).

#### `--imgsz` CLI berupa string, bukan int

Perubahannya lebih sempit daripada kedengarannya. Kedua contoh ini tidak
terpengaruh:

```bash
libreyolo predict --model yolo9-t --source img.jpg --imgsz 640   # tetap berfungsi
```

```python
model.predict("img.jpg", imgsz=640)   # tetap berfungsi
```

Hanya kode yang memanggil fungsi perintah [CLI](/docs/cli) langsung dari Python
yang harus berubah, karena `predict`, `train`, dan `val` memperluas `--imgsz`
dari `int` ke `str` agar dapat menerima ukuran rectangle:

```python
from libreyolo.cli.commands.predict import predict_cmd

predict_cmd(..., imgsz=640)      # 1.4.0
predict_cmd(..., imgsz="640")    # 1.5.0, dan "480x640" kini juga berfungsi
```

Default `train` kini string `"640"`. `export --imgsz` sudah berupa string, dan
`profile` tidak berubah.

### Angka yang berubah

Tiga perubahan menggeser metrik pada pengaturan default. Jika melacak hasil
lintas versi, baca bagian ini sebelum membandingkan run 1.5.0 dengan 1.4.0.

#### faster-coco-eval menjadi backend metrik COCO default

`val()` dan validasi pelatihan per epoch kini menghitung metrik COCO dengan
backend C++ faster-coco-eval, bukan pycocotools.

Peralihan diputuskan berdasarkan paritas terukur pada semua 100 split pengujian
RF100-VL: 1381 dari 1400 nilai metrik identik secara bit, deviasi maksimum
2.22e-16, delta headline tepat 0, dengan kecepatan keseluruhan 15.6x dan 56x
pada dataset padat deteksi. Angka seharusnya tidak berubah. Namun, angka tetap
dihasilkan implementasi berbeda, sehingga perubahan ini dicantumkan.

pycocotools tetap menjadi fallback otomatis jika faster-coco-eval tidak
diinstal. Untuk memaksanya:

```bash
libreyolo val --model yolo9-t --data coco.yaml --no-faster-coco-eval
```

```python
model.val(data="coco.yaml", faster_coco_eval=False)
```

`LIBREYOLO_FASTER_COCO_EVAL=0` melakukan hal yang sama secara global. Backend
yang benar-benar digunakan dicatat pada INFO, tersedia sebagai
`model.last_eval_backend` setelah `val()`, dan disertakan sebagai `eval_backend`
dalam payload JSON [CLI](/docs/cli/val). Instal jalur cepat dengan
`pip install libreyolo[fast-eval]`.

#### Checkpoint YOLOX sebelum 1.5.0 memerlukan override eps

Ini adalah jebakan dalam rilis. Baca jika memiliki hasil fine-tuning
[YOLOX](/docs/models/yolox).

YOLOX menetapkan BatchNorm `eps=1e-3` dan `momentum=0.03`. Hingga 1.5.0, nilai
tersebut diterapkan sebagai perbaikan setelah konstruksi yang tidak bertahan
melalui rebuild jumlah kelas yang dilakukan `train()` ketika `nc` dataset
berbeda dari checkpoint. Fine-tuning tersebut berlatih dan melaporkan validasi
dalam pelatihan pada default torch `eps=1e-5`, lalu dimuat ulang untuk inferensi
pada `1e-3`: tensor sama di bawah normalisasi berbeda.

Ukuran regular-conv hampir tidak berubah. Depthwise `n` berubah besar karena
`running_var` per channel cukup kecil hingga eps mendominasi. Pada RF100-VL
`ball`, checkpoint nano yang sama mendapat **0.566** mAP50-95 saat dievaluasi
dengan eps pelatihan dan **0.151** setelah pemuatan ulang standar.

Checkpoint yang dilatih sebelum 1.5.0 memiliki semantik eps=1e-5. Untuk
melaporkan angka yang tepat, evaluasi dengan eps BN diganti ke 1e-5:

```python
import torch
from libreyolo import LibreYOLOX

model = LibreYOLOX("my-yolox-finetune.pt")
for module in model.model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.eps = 1e-5

model.val(data="data.yaml")
```

Alternatifnya, lipat `sqrt((var + 1e-3) / (var + 1e-5))` ke bobot BN sekali dan
simpan hasilnya. Checkpoint yang dilatih pada 1.5.0 dan setelahnya tidak
memerlukan keduanya.

#### Pelatihan multi-scale D-FINE memakai resep upstream per ukuran

`base_size_repeat` sebelumnya ditetapkan langsung ke 3 untuk setiap ukuran. Kini nilai
diselesaikan per ukuran sesuai upstream: **n** berlatih pada ukuran tetap dengan
multi-scale nonaktif, **s** 20, **m** 6, **l** 4, **x** 3. Hanya x yang cocok
sebelumnya, sehingga n, s, m, dan l melihat distribusi skala berbeda dan
konvergen ke metrik berbeda.

Untuk memulihkan perilaku lama, tetapkan secara eksplisit:

```python
from libreyolo.training.config import DFINEConfig

config = DFINEConfig(base_size_repeat=3)
```

DEIM tetap memakai nilai 3 yang ditetapkan langsung. Detail family tersedia pada
[D-FINE](/docs/models/d-fine).

### Perlu diketahui, tanpa tindakan

- **Hasil `imgsz` rectangle berubah karena sebelumnya salah.** Koordinat bounding box, pengubahan ukuran mask RTMDet, rescaling YOLO-NAS, dan scaling ground truth validator kini menggunakan tinggi dan lebar per sumbu, bukan satu skalar. `imgsz` persegi tidak berubah secara bit. Inferensi atau validasi rectangle pada 1.4.0 salah skala. YOLO-NAS kini menolak `imgsz` rectangle alih-alih diam-diam menghasilkan output salah.
- **Dictionary metrik mendapatkan kunci baru.** `max_det`, `ar_max_det`, dan `AR_max_det` dari evaluator COCO, serta `metrics/loss` dan `metrics/loss/ce` dari FOMO. Nilai default tidak berubah, tetapi semua proses yang mengiterasi kunci metrik, termasuk [logger](/docs/train/loggers) kustom dan header CSV, melihat kolom baru.
- **Run YOLO9 dengan seed yang memicu rebuild head** memulai dari initialization berbeda karena seed kini diterapkan sebelum rebuild, bukan setelahnya. Fine-tuning 1.4.0 dengan seed ke jumlah kelas berbeda tidak dapat direproduksi bit demi bit pada 1.5.0.
- **`libreyolo[hub-kernels]` pada CUDA kini benar-benar mengaktifkan kernel MS-deform-attn native.** 1.4.0 membatasinya di balik kondisi yang tidak pernah diambil RF-DETR, sehingga kernel tidak pernah berjalan. Prediksi dapat bergeser dalam toleransi float untuk RF-DETR dan family deformable-attention lain. Instalasi standar tidak terpengaruh, dan `LIBREYOLO_HUB_KERNELS=0` menonaktifkannya.
- **`libreyolo predict` membuang opsi tidak didukung, bukan memunculkan error.** CLI memfilter kwargs terhadap signature `__call__` model, sehingga opsi yang tidak diterima family diabaikan alih-alih memunculkan `TypeError`. Salah ketik nama flag kini diam-diam diabaikan.
- **Live sumber mengubah bentuk output JSON.** Webcam, stream RTSP, dan screen capture secara implisit mengaktifkan streaming, yang menghasilkan satu rekaman per frame, bukan satu untuk pemanggilan. [Sumber](/docs/predict/sources) tersebut baru pada 1.5.0, jadi script 1.4.0 tidak terpengaruh.
- **Ekspor ulang `rfdetr-pose` atau `yolonas-pose` ke ONNX menghasilkan nama output berbeda.** 1.4.0 keliru membaca head pose multi-tensor sebagai segmentation melalui heuristic jumlah output. Berkas `.onnx` yang sudah ada di disk tidak berubah.
- **Pada instalasi tanpa torch**, hasil memuat array numpy, bukan `torch.Tensor`, sehingga `.boxes.data` mengembalikan jenis berbeda dan tie-breaking NMS dapat berbeda dari torchvision. Jika torch terinstal, perilaku identik per byte. Lihat [instalasi ringan](/docs/lightweight-install).
- **Objek konfigurasi melakukan lebih banyak validasi saat konstruksi.** `TrainConfig` mendapatkan `__post_init__`, sehingga konfigurasi yang sebelumnya sudah tidak valid kini langsung memunculkan error, bukan gagal jauh di dalam run. Serialisasi `ValidationConfig` mendapatkan kunci `edge_thresholds`, yang merusak round-trip ketat `ValidationConfig(**dump)` dari dump 1.4.0.
- **Nama berkas bobot untuk family dengan suffix task diselesaikan secara berbeda.** `segformer-b0` kini diselesaikan menjadi `LibreSegformerb0-sem.pt`. Ini memperbaiki 404 pengunduhan otomatis dan merusak script yang melakukan hard-code nama tanpa suffix lama.
- **Marker pytest `experimental_backend` kini menjadi `extended_backend`.** Hanya relevan jika menjalankan test suite dengan `-m`.

### Checkpoint dan dataset

Checkpoint yang ditulis oleh 1.4.0 dimuat tanpa perubahan.
[Skema](/docs/reference/checkpoint-schema) mendapatkan `imgsz_h` dan `imgsz_w`
untuk model rectangle dan tetap menulis skalar `imgsz = max(h, w)` bagi reader
lama. Ekspor [ExecuTorch](/docs/export/executorch) dan [MNN](/docs/export/mnn)
kini memerlukan sidecar, masing-masing `<program>.pte.json` dan
`<model>.mnn.json`, sedangkan ekspor HRNet memuat
`pose_input: "person_crop"`. Format dataset tidak berubah.
