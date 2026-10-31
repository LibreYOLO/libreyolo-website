---
title: Meningkatkan ke 1.7.0
seo_title: Tingkatkan LibreYOLO 1.6.0 ke 1.7.0
description: >-
  Langkah migrasi untuk checkpoint YOLO9, nama modul dan resep pelatihan, folder
  run CLI, confusion matrix, dan error baru di LibreYOLO 1.7.0.
lead: >-
  Versi 1.7.0 tetap memuat checkpoint YOLO9 lama dengan box float yang sama.
  Versi ini mengubah pelatihan family YOLO9, nama yang dilihat kode di dalam
  jaringan, dan beberapa kontrak validasi dan predict.
keywords:
  - upgrade libreyolo
  - migrasi libreyolo 1.6.0
  - breaking change libreyolo
  - nama modul yolo9
  - clip_max_norm
  - confusion matrix
last_verified: 1.7.0
source_hash: 895bcd7425f0052f
---

## 1.6.0 ke 1.7.0

- Checkpoint YOLO9, YOLO9-E2E, dan YOLO9-P2 yang disimpan oleh 1.7.0 memakai nama tensor baru dan tidak dapat dimuat di 1.6.x: 1.6.x memuat yang YOLO9 dan YOLO9-P2 dengan peringatan missing-keys dan tidak memprediksi box, dan memunculkan error ketidakcocokan `model_family` untuk yang YOLO9-E2E. Tingkatkan setiap lingkungan yang membacanya. Checkpoint float, terkuantisasi, dan pelatihan yang ditulis rilis sebelumnya dapat dimuat di 1.7.0, dan ekspor ONNX tidak terpengaruh.

- Kode yang mengakses `model.model` pada model family YOLO9 melihat nama modul MultimediaTechLab/YOLO, tanpa alias. `DDetect` menjadi `YOLO9Head` (`anchor_convs` / `class_convs`), `RepConvN` menjadi `RepConv`, `RepNBottleneck` menjadi `Bottleneck`, `DFL` menjadi `Anchor2Vec` (`anchor2vec`), `nc` menjadi `num_classes`, dan sublayer blok menjadi `conv1`/`conv2`/`bottleneck`. Pada YOLO9-E2E, `YOLO9E2EDetect` menjadi `YOLO9E2EHead` dan `one2one_cv2` / `one2one_cv3` menjadi `one_to_one_anchor_convs` / `one_to_one_class_convs`. `LibreYOLO9Model.fuse()`, `RepConvN.fuse_convs()`, `Conv.forward_without_bn()`, `Concat`, dan `yolo9.utils.decode_boxes` dihapus. [Changelog](/docs/changelog) mencantumkan setiap penggantian nama.

- Nama node ONNX mengikuti nama modul baru pada head, backbone, dan neck, misalnya `/head/anchor_convs.0/...`. Perbarui alat yang bergantung pada nama node, seperti konfigurasi end-node Hailo. Input, output, dan nama output model tidak berubah.

- Fine-tuning YOLO9 dari bobot tanpa tensor PGI, termasuk `LibreYOLO9{t,s,m,c}.pt` yang dipublikasikan, hanya melatih head utama, seperti di 1.5.0; 1.6.0 memasang cabang PGI yang diinisialisasi acak dengan `aux_weight=0.25`. Bobot yang membawa tensor PGI, dan run dari nol, tetap dilatih dengan PGI. Berikan `aux_weight=0.25` untuk tetap memasang cabang baru, atau `aux_weight=0` untuk memaksanya mati.

- Saat PGI dipasang, YOLO9-m dan YOLO9-c membangun cabang auxiliary MultimediaTechLab: 12.8M (m) dan 25.7M (c) parameter khusus pelatihan pada 80 kelas, naik dari 7.5M dan 10.2M, sehingga pelatihan PGI pada ukuran ini memerlukan lebih banyak memori dan waktu per langkah. Checkpoint pelatihan m/c dari 1.6.0 dapat dilanjutkan dan di-fine-tune dengan cabang yang dipakai saat pelatihannya. Inferensi serta YOLO9-t/s tidak berubah.

- YOLO9, YOLO9-E2E, dan YOLO9-P2 memotong norma L2 gradien pada 10, dan assigner pelatihannya menilai anchor dalam fp32, sehingga positif yang dipilih di bawah fp16 AMP bisa berbeda. Berikan `clip_max_norm=0` untuk pemotongan 1.6.0; run yang dimulai di 1.6.0 dan dilanjutkan di 1.7.0 juga memotong kecuali Anda memberikannya. Penetapan fp32 tidak punya sakelar.

- `perspective` bukan nol sekarang membengkokkan lewat pergeseran sudut di setiap family yang memakai affine warp bersama (YOLO9 beserta varian E2E dan P2, YOLOX, YOLOv7, YOLO-NAS, PP-YOLOE, RTMDet, PicoDet, RT-DETR, dan RT-DETRv2), termasuk run yang dilanjutkan dari checkpoint 1.6.0 yang menetapkannya. Nilai `0.002` ke atas memberi warp maksimum yang sama, dan nilai negatif memunculkan `ValueError`. Tidak ada pengaturan yang mereproduksi warp 1.6.0; default `perspective=0` tidak berubah.

- Pipeline mosaic YOLO9 menarik rasio mixup dari Beta(1, 1), bukan Beta(32, 32), tanpa pengaturan untuk nilai lama. Pelatihan YOLOv7 default berubah, karena YOLOv7 mengaktifkan mixup secara default. YOLO9, YOLO9-E2E, YOLO9-P2, RT-DETR, dan RT-DETRv2 hanya berubah bila mixup diaktifkan.

- `libreyolo train` tanpa `name=` menulis run EC, D-FINE, SegFormer, dan LingBot-Vision ke `runs/train/ec_exp`, `dfine_exp`, `segformer_exp`, dan `lingbotvision_exp`, bukan `runs/train/exp`, probe V-JEPA 2 ke `runs/train/train`, dan run YOLO9-P2 ke `runs/train/yolo9_exp`. Berikan `name=exp` untuk path lama, atau `name=yolo9_p2_exp` untuk YOLO9-P2.

- `libreyolo train` memulai D-FINE, SegFormer, dan LingBot-Vision dari resepnya sendiri. Untuk D-FINE itu berarti 132 epoch, `lr0=2e-4`, `weight_decay=1e-4`, 2 epoch warmup, dan tanpa augmentasi mosaic, mixup, HSV, maupun affine. Untuk mengirim nilai yang dikirim 1.6.0, berikan `epochs=300 optimizer=sgd lr0=0.01 weight_decay=5e-4 warmup_epochs=5 mosaic=1.0 mixup=1.0 hsv_prob=1.0 degrees=10 translate=0.1 shear=2.0`.

- `libreyolo.validation.ConfusionMatrix` mengindeks `matrix[predicted, true]` (dulu `[actual, predicted]`), menghitung deteksi pada `conf_thres=0.25` (dulu 0.15), menerima `names=` dan `task=` hanya sebagai argumen keyword, dan `matrix`-nya tidak dapat lagi ditetapkan ulang. Transpos kode yang mengindeks matriks lama, dan berikan `conf_thres=0.15` untuk hitungan lama. `plots=True` menulis hitungan ke `confusion_matrix.png` dan plot ternormalisasi ke `confusion_matrix_normalized.png`.

- `predict(agnostic_nms=True)` menekan kotak tumpang tindih lintas kelas, bukan lagi memberi peringatan. Task tanpa kotak memunculkan `ValueError`, `LibreEnsemble` dan HRNet memunculkan `NotImplementedError`, dan tier SAM menolak keyword itu dengan `TypeError`; hapus argumen dari pemanggilan tersebut.

- `interpolation` classifier yang tidak dikenal memunculkan `ValueError`, termasuk mode torchvision `LANCZOS`, `BOX`, `HAMMING`, dan `NEAREST_EXACT`. Gunakan `bilinear`, `bicubic`, `nearest`, atau `bilinear_no_antialias`.

- Jalankan ulang validasi sebelum membandingkan dengan angka 1.6.0 untuk `val(single_cls=True)` pada model yang dilatih dengan beberapa kelas, dataset segmentasi yang mencampur baris box-saja dan poligon, dan RTMDet-Ins, yang mempertahankan hingga `max_det=300` mask per gambar secara default; berikan `max_det=100` untuk pengaturan MMDetection.

- Tanpa `libreyolo[rfdetr]`, pemanggilan CLI dengan nama model RF-DETR atau DINOv2 keluar dengan kode error JSON `model_load_failed`, bukan `model_not_found`, dan pesannya menyebut `pip install libreyolo[rfdetr]`.

Lihat [changelog](/docs/changelog) untuk rilis lengkapnya.

## 1.5.0 ke 1.6.0

- Tingkatkan lingkungan yang dipatok di bawah ONNX Runtime 1.18 ke `onnxruntime>=1.18.0` sebelum memasang extra ONNX.

- SAM 3D Body menerima kumpulan aset snapshot yang sudah ditinjau. Pasang `libreyolo[hf]`, dapatkan akses terbatas, lalu gunakan pengambilan otomatis, atau sediakan direktori snapshot tanpa perubahan dan aset MHR yang dipatok pada sistem berkas lokal tepercaya.

- Jalankan ulang validasi setelah koreksi pengubahan ukuran RF-DETR non-pose dan normalisasi pengklasifikasi. Periksa ulang ambang skor keyakinan yang diterapkan sebelum membandingkan hasil dengan 1.5.0; tidak ada flag prapemrosesan lama.

- Deteksi D-FINE, DEIM, RT-DETRv4, dan YOLO-NAS mengaktifkan FP16 AMP secara default. Berikan `amp=False` untuk mempertahankan FP32.

- Untuk pilihan fine-tuning YOLO9 sebelumnya, atur `aux_weight=0`, `max_labels=100`, dan `warmup_momentum=0.937`. Atur `letterbox_pad="topleft"` jika konversi baru bertanda center harus mereproduksi geometri lama. Checkpoint lama dengan satu head melanjutkan pelatihan dengan graf aslinya. Run YOLO9 baru memakai lebih banyak memori GPU, jadi turunkan `batch` jika ukuran batch dari 1.5.0 kehabisan memori.

- RF-DETR dan DINOv2 membuat direktori proses bernomor dengan nama family. Perbarui pengguna path artefak, atau atur `output_dir="runs/train", exist_ok=True` untuk mempertahankan lokasi lama dan perilaku penggunaan ulang.

- Pertahankan `mixup + cutmix <= 1` untuk klasifikasi; kombinasi tidak valid kini menimbulkan galat saat penyiapan.

- QAT menonaktifkan EMA, SyncBatchNorm, dan perataan checkpoint. Gunakan checkpoint best/last QAT tanpa bergantung pada status tersebut.

- Loader kustom dengan hook mutasi dataset harus memakai `persistent_workers=False` atau membangun ulang worker setelah mutasi. Salinan persisten multi-worker yang tidak kompatibel kini menimbulkan galat.

- Array gambar NumPy dibaca sebagai BGR, urutan OpenCV. Berikan `color_format="rgb"` saat memberikan array RGB seperti `np.asarray(pil_image)`; keluaran `cv2.imread()` dan frame video tidak perlu diubah.

- Array NumPy atau tensor 4D adalah batch: `predict()` mengembalikan daftar berisi satu `Results` per gambar, alih-alih hanya memakai gambar pertama.

- `train(resume=True)` dan `train(resume="path/to/last.pt")` memulihkan argumen pelatihan tersimpan milik proses tersebut dan tetap menulis ke direktori prosesnya; argumen yang diberikan secara eksplisit lebih diutamakan. Melanjutkan bobot rilis, atau proses yang sudah mencapai `epochs`-nya, menimbulkan `ValueError`.

- Dengan validasi aktif, epoch terakhir selalu divalidasi, sehingga proses yang lebih pendek dari `eval_interval` kini melaporkan metrik dan menulis `best.pt`. `val=False` mematikan validasi, termasuk pada epoch terakhir. Validasi selama pelatihan kini menulis ke `<run>/val`, bukan `runs/val/`.

- Kegagalan `train` di CLI melaporkan jenis galatnya, sehingga galat konfigurasi keluar dengan kode 2, bukan 1 dengan `io_error`.

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

Tidak ada shim deprecation. Pemanggilan yang masih memberikannya mendapat
peringatan `Unknown training config keys (ignored)`, dan argumen itu tidak
berpengaruh. `BaseModel.EXPERIMENTAL_WEIGHT_FILENAMES` juga dihapus. Hook
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
from libreyolo import LibreYOLO

model = LibreYOLO("my-yolox-finetune.pt")
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
- **`libreyolo predict` membuang opsi tidak didukung, bukan memunculkan error.** CLI memfilter kwargs terhadap signature `__call__` model, sehingga opsi yang tidak diterima family diabaikan alih-alih memunculkan `TypeError`. Nama flag yang tidak dikenal tetap ditolak dengan `No such option`.
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
