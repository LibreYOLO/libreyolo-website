---
title: Skema checkpoint
seo_title: Skema metadata checkpoint LibreYOLO v1.0
description: >-
  Metadata yang dibawa setiap checkpoint .pt LibreYOLO: kunci wajib, tambahan per
  task, kunci runtime ekspor, manifest terkuantisasi, dan kolom pelatihan.
lead: >-
  Berkas .pt LibreYOLO adalah dictionary flat yang disimpan dengan torch.save. Kunci
  model menyimpan state dict; kunci top-level lainnya adalah metadata yang
  mengidentifikasi checkpoint tanpa parsing nama berkas atau sniffing state dict.
keywords:
  - skema checkpoint libreyolo
  - schema_version 1.0
  - model_family
  - metadata checkpoint libreyolo
  - manifest quant
  - wrap_libreyolo_checkpoint
last_verified: 1.5.0
verification: >-
  Mencerminkan docs/checkpoint_schema.md dalam repositori libreyolo pada v1.5.0,
  diperiksa silang terhadap libreyolo/utils/serialization.py dan BaseModel.save.
snippets:
  usage:
    - label: Baca metadata dari checkpoint
      language: python
      code: >
        from libreyolo import LibreYOLO

        from libreyolo.utils.serialization import unwrap_libreyolo_checkpoint

        import torch


        # Unduh checkpoint, lalu simpan ulang agar tersedia path lokal.

        LibreYOLO("LibreYOLO9t.pt").save("roundtrip.pt")


        loaded = torch.load("roundtrip.pt", map_location="cpu",
        weights_only=False)

        state_dict, metadata = unwrap_libreyolo_checkpoint(loaded)


        print(metadata["schema_version"], metadata["model_family"])

        print(metadata["size"], metadata["task"], metadata["nc"],
        metadata["imgsz"])

        print(len(state_dict), "tensors")
source_hash: ce760f1bed97bfd0
---

## Skema v1.0

Setiap checkpoint `.pt` resmi LibreYOLO memuat:

```python
{
    "model": state_dict,
    "schema_version": "1.0",
    "libreyolo_version": "0.x.y",
    "model_family": "yolo9",
    "size": "t",
    "task": "detect",
    "nc": 80,
    "names": {0: "cat", 1: "dog"},
    "imgsz": 640,
}
```

| Kunci | Jenis | Arti |
|---|---|---|
| `model` | state dict | Bobot model |
| `schema_version` | str | Versi kontrak metadata; v1.0 menggunakan string `"1.0"` |
| `libreyolo_version` | str | Versi yang menghasilkan checkpoint |
| `model_family` | str | Family terdaftar, seperti `yolo9`, `rfdetr`, `dfine`, `ec` |
| `size` | str | Varian dalam family, seperti `t`, `s`, `r18`, `atto` |
| `task` | str | Nama task kanonis |
| `nc` | int | Jumlah kelas positif |
| `names` | dict | `dict[int, str]` dengan kunci dalam `0..nc-1` |
| `imgsz` | int | Resolusi input persegi positif, atau skalar lama untuk kontrak rectangle |

`task` adalah salah satu dari `detect`, `segment`, `semantic`, `panoptic`,
`pose`, `classify`, `gaze`, `obb`, `point`, `depth`, `edge`, `normal`,
`restore`, `matte`, `ocr`, `embed`, atau `mesh`.

Checkpoint resmi menulis setiap kunci `names`. Reader dapat menambahkan label
`class_i` untuk kunci yang hilang pada mapping sparse lama, tetapi kunci di luar
rentang tidak valid.

Checkpoint rectangle mempertahankan `imgsz` skalar bagi reader lama, dengan
nilai `max(imgsz_h, imgsz_w)`, dan juga menulis `imgsz_h` serta `imgsz_w` dengan
dimensi sebenarnya. Reader yang memahami kolom rectangle harus
memprioritaskannya daripada skalar. Family dengan kontrak rectangle tetap,
seperti pose HRNet, menolak ukuran runtime yang tidak kompatibel.

Skema sengaja dibuat flat, dan `model` sengaja berupa state dict.

<code-tabs name="usage" />

## Tambahan pose

Pose biasanya satu kelas, `nc: 1` dengan `person`, tetapi head pose YOLO-NAS
juga mendukung pose multi-kelas dengan satu skeleton keypoint bersama. Dalam
kasus itu, `nc` dan `names` menjelaskan kelas seperti dalam deteksi. Ekspor pose
runtime menghasilkan `scores` berbentuk `[batch, anchors, nc]`.

| Kunci | Arti |
|---|---|
| `num_keypoints` | Jumlah keypoint positif yang digunakan head pose |
| `keypoint_dim` | `2` untuk label `x,y` atau `3` untuk label `x,y,visibility`; output model selalu menyediakan `x,y,visibility` |
| `oks_sigmas` | Sigma OKS opsional per keypoint; default task untuk `num_keypoints` digunakan jika tidak ada |
| `num_keypoints_per_class` | Jumlah keypoint opsional per kelas untuk head bergaya GroupPose yang tensor keypoint-nya diberi padding menurut kelas; `0` untuk kelas tanpa keypoint |

## Tambahan mesh

Checkpoint mesh menggunakan `task: "mesh"`, `nc: 1`, dan
`names: {0: "person"}`. Tata letak parameter berbeda antar body model, sehingga
dimensinya dicatat, bukan diasumsikan.

| Kunci | Arti |
|---|---|
| `body_model` | Parameterization seperti `mhr`; wajib dan digunakan untuk menafsirkan setiap kolom di bawah |
| `num_betas` | Jumlah koefisien identitas dan bentuk; 45 untuk MHR |
| `num_body_pose` | Lebar block parameter body-pose; 130 untuk MHR. Berupa vektor flat, bukan satu triplet per joint, karena joint rig memiliki degree of freedom berbeda |
| `num_vertices` | Jumlah vertex yang dihasilkan decoder; 18439 untuk MHR |
| `num_joints` | Jumlah joint yang dihasilkan decoder; 127 untuk MHR |
| `rotation_format` | Cara rotasi dikodekan, seperti `euler_zyx` untuk MHR atau `axis_angle`. Tidak pernah disimpulkan dari bentuk tensor karena vektor-3 bersifat ambigu |

## Placeholder task padat

Beberapa task memprediksi map padat, bukan kelas, sehingga slot mirip kelas
hanya ada untuk kompatibilitas skema.

| Task | `nc` | `names` |
|---|---|---|
| `depth` | 1 | `{0: "depth"}` |
| `edge` | 1 | `{0: "edge"}` |
| `restore` | 1 | `{0: "image"}` |
| `ocr` | 1 | `{0: "text"}` |

Prediksi edge adalah map probabilitas float32 padat dalam `[0, 1]`.

Checkpoint restore dapat menambahkan `degradation`, label kerusakan pendek
seperti `deblur`, `denoise`, atau `super-resolution`; `dataset`, label asal
seperti `GoPro` atau `SIDD`; serta `scale`, faktor upscale output-ke-input
bilangan bulat positif, misalnya `4` untuk model super-resolution x4. Nilai yang
tidak ada atau `1` berarti gambar hasil restorasi mempertahankan resolusi input.
Runtime juga menurunkan skala dari family dan ukuran, sehingga `scale` merupakan
metadata asal, bukan persyaratan saat pemuatan.

## Tambahan OCR

Family `ppocr` menyediakan satu checkpoint komposit per tier, dengan state dict
`model` yang menyimpan dua submodel di bawah namespace kunci `det.*` dan `rec.*`.

| Kunci | Arti |
|---|---|
| `charset` | Alfabet CTC lengkap dalam urutan indeks output: indeks 0 adalah CTC blank, diikuti dictionary pengenalan, lalu karakter spasi. Loader harus membacanya dari checkpoint, bukan berkas samping |
| `pipeline` | Default pipeline yang ditanam saat konversi: `det_limit_side_len`, `det_db_thresh`, `det_db_box_thresh`, `det_db_unclip_ratio`, `rec_image_shape`. Argumen runtime dapat menimpanya per pemanggilan |
| `components` | Dicadangkan untuk tahap pipeline opsional seperti orientasi dokumen, unwarping, dan rotasi baris teks. Kosong pada v1 |

## Metadata runtime ekspor

Artefak hasil ekspor menggunakan konvensi dual-write rectangle yang sama:
`imgsz_h` dan `imgsz_w` ditulis di samping `imgsz` skalar lama, dan reader yang
tidak memahami kolom rectangle tidak boleh diam-diam memperlakukan skalar
sebagai kontrak persegi.

Dukungan runtime rectangle dibatasi family dan format. Ekspor family YOLO9,
HRNet, NAFNet, dan Real-ESRGAN dapat menggunakan `imgsz_h` dan `imgsz_w`
nonpersegi dalam format yang didukung; family atau format tanpa dukungan
rectangle eksplisit menolak metadata, bukan melakukan preprocessing artefak
sebagai persegi. Ekspor HRNet adalah head person-crop tetap, batch satu, FP32,
dengan W32 menerima 256x192 dan W48 menerima 384x288, serta detektor orang
tidak disematkan dalam graph.

Ekspor dengan NMS tertanam dapat menambahkan kunci flat berikut:

| Kunci | Arti |
|---|---|
| `nms` | Boolean string; `"true"` berarti graph memuat output postprocessing tertanam |
| `nms_conf` | Ambang batas confidence yang ditanam ke output |
| `nms_iou` | Ambang batas IoU yang ditanam ke output |
| `max_det` | Jumlah maksimum baris deteksi pasca-NMS yang dihasilkan output |
| `nms_raw_output` | Boolean string; `"true"` berarti graph juga menyediakan output detektor mentah tambahan |

Untuk ekspor deteksi ONNX YOLO9 dengan `nms=true`, output `0` (bernama `output`)
adalah tensor pasca-NMS mandiri pada ambang batas waktu ekspor. Ketika
`nms_raw_output=true`, output `1` (bernama `raw`) dicadangkan bagi backend
LibreYOLO agar dapat menerapkan clipping canvas asli native dan semantik runtime
`predict(conf=..., iou=..., max_det=...)`. Konsumen pihak ketiga sebaiknya
menggunakan output pertama.

Ekspor pose dapat menambahkan `num_keypoints`; `keypoint_dim`, dengan ekspor
mentah bergaya GroupPose mungkin menggunakan nilai lebih besar seperti `8`
ketika tensor memuat kolom presisi atau logit kelas; `num_keypoints_per_class`
sebagai list berformat JSON, dengan slot kelas tanpa keypoint harus dipertahankan
karena mendefinisikan skema; dan `pose_input`, dengan `"person_crop"` berarti
graph menerima satu crop yang sudah diekstrak dan tidak memuat detektor. Ekspor
runtime HRNet mewajibkan nilai tersebut.

Ekspor classification dapat menambahkan `crop_pct`, rasio center-crop float
dengan target resize pracrop `round(imgsz / crop_pct)` dan default `0.875` jika
tidak ada, serta `interpolation`, `"bilinear"` atau `"bicubic"`, dengan default
`"bilinear"`.

Ekspor ExecuTorch menulis metadata flat ke sidecar wajib `<program>.pte.json`.
Kontrak v1 adalah CPU, FP32, batch 1, dan canvas input tetap, serta mewajibkan
`executorch_version`, `executorch_delegate` bernilai `"xnnpack"`, dan
`executorch_delegate_partitions` positif. Loader menolak sidecar yang menyatakan
delegate lain, bentuk dinamis, atau presisi selain FP32.

Ekspor MNN menulis metadata flat ke sidecar wajib `<model>.mnn.json`. Kontrak v1
adalah CPU, FP32, khusus deteksi, dan bentuk input NCHW tetap, serta mewajibkan
`mnn_version`, `mnn_backend` bernilai `"cpu"`, `mnn_input_names` dan
`mnn_output_names` terurut serta tidak kosong, `mnn_input_shape` berupa empat
bilangan bulat positif dalam urutan `[batch, channels, height, width]`, dan
`mnn_batch` sama dengan `mnn_input_shape[0]`. Loader menolak metadata dinamis,
selain FP32, selain deteksi, family tidak didukung, atau bentuk tidak konsisten.

Berkas `.pte` dan `.mnn` merupakan artefak khusus backend, bukan checkpoint
PyTorch.

## Checkpoint terkuantisasi

Model terkuantisasi menambahkan satu kunci flat opsional, `quant`, yang menyimpan
dict manifest dengan `schema`, `recipe`, `keep_high_precision`, `execution`,
asal kalibrasi, `module_count`, dan `state`. Manifest FP8 juga dapat memuat
`fp8_tensorwise_weights`, list tepat nama modul `QuantLinear` yang skala bobotnya
bersifat tensorwise, bukan per output channel. Loader yang melihat `quant`
membangun ulang struktur modul terkuantisasi dan kebijakan scaling sebelum
`load_state_dict`.

`state` membedakan dua bentuk artefak.

`"prepared"`, yang merupakan default, menyimpan master weight FP32 ditambah
buffer skala `_q_*` dan dapat dilatih. Reader tanpa dukungan kuantisasi dapat
mengabaikan kunci `quant` dan memuat master sebagai model float.

`"finalized"` adalah bentuk deployment yang ditulis oleh `export(format="pt")`.
Master dihapus dan setiap modul terkuantisasi menyimpan bobot terkemas:

| Resep | Tensor terkemas | Dekuantisasi |
|---|---|---|
| int8 | `weight_packed` int8 dengan bentuk bobot asli, `_q_w_scale` FP32 per channel | `weight_packed * scale` |
| fp8 | `weight_packed` float8_e4m3fn dengan bentuk asli, `_q_w_scale` FP32 satu entri per output channel | `weight_packed * scale` |
| w4a16, w4a8 | `weight_packed` uint8, dua kode 4-bit per byte, nibble rendah lebih dulu, kode `q + 8`; `_q_w_gscale` FP32 `[out, ngroups]`, kelompok 128 sepanjang in_features | Skala per kelompok |
| int2 | Empat kode 2-bit per byte, kode `q + 2`, kelompok 64 | Skala per kelompok |
| nvfp4 | `weight_packed` uint8 `[out, ceil(in/16)*8]`, kode `sign<<3 \| E2M1 level`; `weight_block_scale` float8_e4m3fn `[out, ceil(in/16)]`; `_q_w_amax` FP32 per tensor | `block_scale * amax / (448 * 6)` |
| mxfp4 | Seperti nvfp4, tetapi block 32 elemen, ditambah `weight_block_exp` int8 `[out, ceil(in/32)]` | `2 ** exponent` |

Buffer rentang aktivasi `_q_act_lo`, `_q_act_hi`, dan `_q_calibrated`
dipertahankan untuk int8. Manifest mencatat `remainder`, `"fp16"` atau
`"fp32"`, bagi tensor yang tidak dikuantisasi. Unpacking mereproduksi simulasi
bit demi bit, sehingga inferensi finalized sama persis dengan inferensi prepared
pada device finalisasi. Tata letak ini adalah kontrak stabil bagi exporter dan
runtime eksternal.

## Checkpoint pelatihan

Checkpoint trainer menggunakan inti metadata wajib yang sama dan dapat
menambahkan kolom flat pelatihan serta resume:

```python
{
    "model": state_dict,
    "epoch": 42,
    "optimizer": optimizer_state_dict,
    "config": {},
    "loss": 1.23,
    "best_metric_key": "metrics/mAP50-95",
    "best_metric_value": 0.51,
    "best_epoch": 39,
    "is_ema_weights": True,
    "train_model": raw_state_dict,
    "ema": ema_state_dict,
    "ema_updates": 12345,
}
```

`is_ema_weights` menyatakan apakah `model` top-level telah dihaluskan EMA.
Ketika EMA diaktifkan, `train_model`, `ema`, dan `ema_updates` mempertahankan
status resume. Bobot inferensi yang dipublikasikan sebaiknya ramping dan tidak
menyertakan optimizer, epoch, konfigurasi, loss, atau status resume EMA kecuali
sengaja didistribusikan sebagai checkpoint pelatihan.

Untuk kompatibilitas rilis, reader menerima alias metrik terbaik lama
`best_mAP50_95`, `best_mAP50`, `best_metric`, dan `best_metric_name`.

## Snapshot eksternal

Skema mengatur berkas `.pt` yang dibuat LibreYOLO. Skema tidak mengganti nama atau
membungkus snapshot upstream multi-berkas yang digunakan oleh tier model terpisah.

LibreMODUS ukuran `14b-a7b` adalah pengecualian eksplisit: alias diselesaikan
melalui `LibreVLM(...)` ke direktori berkas upstream yang dikunci versinya, dan LibreYOLO
tidak menambahkan metadata v1.0 maupun memublikasikannya kembali sebagai `.pt`.

## Bobot lama dan asing

Writer baru melakukan validasi ketat dan wajib menghasilkan metadata v1.0. Jika
metadata tidak ada atau tidak lengkap, checkpoint yang tampak seperti LibreYOLO
lama dimuat melalui jalur kompatibilitas dengan peringatan dan petunjuk
konversi, sedangkan checkpoint upstream asing diarahkan ke konversi otomatis.
Lihat [checkpoint upstream](/docs/reference/upstream-checkpoints).

## Helper

Helper skema berada dalam `libreyolo.utils.serialization`:

```python
wrap_libreyolo_checkpoint(
    state_dict,
    *,
    model_family,
    size,
    task,
    nc,
    names=None,
    imgsz=None,
    libreyolo_version=None,
    schema_version="1.0",
    **extra_metadata,
) -> dict

validate_checkpoint_metadata(checkpoint, *, strict=False) -> list[str]

unwrap_libreyolo_checkpoint(loaded, *, strict=False) -> tuple[dict, dict]
```

`validate_checkpoint_metadata` tidak melakukan mutasi dan mengembalikan list
error; dengan `strict=True`, metode ini memunculkan `CheckpointMetadataError`.
`model.save(path)` adalah cara yang didukung untuk menulis checkpoint yang
sesuai.


