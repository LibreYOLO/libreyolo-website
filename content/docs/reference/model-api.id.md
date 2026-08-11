---
title: API model
seo_title: Metode dan signature objek model LibreYOLO
description: >-
  Setiap metode pada model LibreYOLO yang dimuat: predict, embed, track, val,
  train, export, save, quantize, info, dan kontrol CUDA graph, dengan default
  sebenarnya.
lead: >-
  Model LibreYOLO yang dimuat adalah instance BaseModel. Halaman ini
  mencantumkan metode yang dimiliki instance tersebut, dengan signature dan
  default yang dibaca dari libreyolo/models/base/model.py.
keywords:
  - metode model libreyolo
  - argumen predict libreyolo
  - argumen val libreyolo
  - argumen export libreyolo
  - model.track
  - model.quantize
  - capture_graph
last_verified: 1.5.0
verification: >-
  Signature dan default dibaca dari libreyolo/models/base/model.py dan
  libreyolo/models/base/inference.py pada v1.5.0. Kelas family dapat
  mempersempit atau memperluasnya; train() didefinisikan per family dan hanya
  wrapper cfg= bersama yang didokumentasikan di sini.
snippets:
  usage:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9t.pt")

        model.info()
        result = model(SAMPLE_IMAGE, conf=0.25, iou=0.45)

        print(result.boxes.xyxy)
        print(result.speed)
  stream:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9t.pt")


        # stream=True mengembalikan generator, satu Results per frame atau
        gambar.

        for result in model([SAMPLE_IMAGE, SAMPLE_IMAGE], stream=True):
            print(len(result))
source_hash: da0776970ded8716
---

## Konstruksi

Factory mengembalikan instance kelas family. Membuat kelas tersebut secara
langsung menerima argumen yang sama, kecuali `size` wajib diberikan:

```python
Family(model_path, size, nb_classes=80, device="auto", task=None, **kwargs)
```

`device="auto"` memilih CUDA jika tersedia, lalu MPS, lalu CPU. Bilangan bulat
atau string digit dibaca sebagai ordinal CUDA, sehingga `device=0` dan
`device="0"` sama-sama berarti `cuda:0`. `task` divalidasi terhadap
`SUPPORTED_TASKS` milik family. Memberikan `model_path=None` membangun arsitektur
dan membiarkannya dalam mode pelatihan; memberikan `dict` memuat state dict
tersebut secara langsung.

## predict dan \_\_call\_\_

`predict` adalah alias untuk `__call__`.

```python
model(
    source=None,
    *,
    conf=0.25,
    iou=0.45,
    imgsz=None,
    device=None,
    classes=None,
    max_det=300,
    augment=False,
    save=False,
    batch=1,
    stream=False,
    stream_buffer=False,
    vid_stride=1,
    show=False,
    output_path=None,
    color_format="auto",
    tiling=False,
    overlap_ratio=0.2,
    output_file_format=None,
    cuda_graph=False,
    **kwargs,
)
```

| Argumen | Default | Arti |
|---|---|---|
| `source` | `None` | Gambar, list atau tuple gambar dalam memori, direktori, file video, atau sumber layar seperti `"screen"`, `"screen 1"`, `"screen 1 100 200 512 256"` |
| `conf` | `0.25` | Ambang batas confidence |
| `iou` | `0.45` | Ambang batas IoU untuk NMS |
| `imgsz` | `None` | Override ukuran input; `None` menggunakan ukuran native model |
| `device` | `None` | Override device untuk pemanggilan ini |
| `classes` | `None` | Hanya mempertahankan id kelas berikut |
| `max_det` | `300` | Jumlah maksimum deteksi per gambar |
| `augment` | `False` | Augmentasi saat pengujian |
| `save` | `False` | Menulis gambar atau video beranotasi |
| `batch` | `1` | Jumlah gambar per forward pass untuk sumber direktori dan list |
| `stream` | `False` | Mengembalikan generator, bukan list yang diwujudkan |
| `stream_buffer` | `False` | Mempertahankan setiap frame live yang ditangkap, bukan hanya yang terbaru |
| `vid_stride` | `1` | Memproses setiap frame video atau layar ke-N |
| `show` | `False` | Menampilkan frame beranotasi dalam window |
| `output_path` | `None` | Path output ketika `save=True` |
| `color_format` | `"auto"` | Petunjuk format warna untuk array dalam memori |
| `tiling` | `False` | Inferensi berubin untuk gambar besar |
| `overlap_ratio` | `0.2` | Rasio tumpang tindih ubin |
| `output_file_format` | `None` | `"jpg"`, `"png"`, atau `"webp"` |
| `cuda_graph` | `False` | `True` menangkap saat penggunaan pertama per bentuk input, `"auto"` menunggu hingga bentuk berulang |

Sumber satu gambar mengembalikan satu `Results`. List, tuple, atau direktori
mengembalikan list `Results`, dan `stream=True` mengembalikan generator dalam
setiap kasus.

Live stream tidak memiliki batas akhir dan memerlukan `stream=True`. `tiling`
dan `augment` tidak dapat digabungkan. Augmentasi saat pengujian memunculkan
error untuk task `embed`, `point`, dan `edge`.

<code-tabs name="usage" />

Dengan `batch > 1`, family dengan `SUPPORTS_BATCHED_PREDICT` bernilai true
menjalankan satu forward bertumpuk per kelompok; `batch=1` mempertahankan satu
forward per gambar.

<code-tabs name="stream" />

## embed

```python
model.embed(source=None, **kwargs) -> torch.Tensor
```

Wrapper praktis di atas `predict` yang menumpuk setiap baris embedding menjadi
satu tensor `(N_total, D)`. Model harus dibuat dengan `task="embed"`; jika
tidak, metode ini memunculkan `NotImplementedError`.

## track

```python
model.track(
    source,
    *,
    track_conf=0.25,
    iou=0.45,
    imgsz=None,
    classes=None,
    max_det=300,
    save=False,
    show=False,
    vid_stride=1,
    output_path=None,
    tracker="bytetrack",
    tracker_config=None,
    augment=False,
    **tracker_kwargs,
) -> Generator[Results, None, None]
```

Menghasilkan satu `Results` per frame dengan `track_id` yang telah ditetapkan.
`tracker` adalah `"bytetrack"`, `"botsort"`, `"ocsort"`, atau
`"deepocsort"`, dan diabaikan ketika `tracker_config` diberikan karena jenis
konfigurasi memilih tracker. `track_conf` dipetakan ke `track_high_thresh` untuk
ByteTrack dan BoT-SORT, serta ke `det_thresh` untuk OC-SORT dan Deep OC-SORT.
Default `output_path` adalah `runs/track/<video_stem>.mp4`.

## val

```python
model.val(
    data=None,
    batch=16,
    imgsz=None,
    conf=0.001,
    iou=0.6,
    workers=4,
    allow_download_scripts=False,
    device=None,
    split="val",
    augment=False,
    save_json=False,
    verbose=True,
    *,
    plots=None,
    **kwargs,
) -> Dict
```

Mengembalikan dictionary metrik yang key-nya bergantung pada task; deteksi
mengembalikan `metrics/precision`, `metrics/recall`, `metrics/mAP50`, dan
`metrics/mAP50-95`. `imgsz` menerima int persegi atau tuple `(height, width)`
dan default-nya menggunakan ukuran input native model. `plots` adalah alias
untuk `save_plots`. `allow_download_scripts` mengendalikan Python tertanam yang
dapat dibawa YAML dataset dalam field `download`.

`faster_coco_eval` diterima melalui `**kwargs` dengan default `True`, lalu
kembali ke pycocotools ketika package tidak diinstal. Backend yang dijalankan
dilaporkan pada `model.last_eval_backend`.

Validasi dengan augmentasi memunculkan error untuk task `obb` dan `pose`.

## train

`train` didefinisikan per family, sehingga argumennya berbeda. Dua perilaku
digunakan bersama karena base class membungkus `train` setiap family:

- `cfg=` menerima path YAML yang key-nya digabungkan ke pemanggilan. Keyword argument eksplisit mengalahkan file.
- `pretrained=False` pada family dalam kelompok cakupan `g0` atau `g1` melakukan reinitialization model dari awal sebelum pelatihan dan tidak dapat digabungkan dengan `resume=True`.

Knob augmentasi yang benar-benar dipatuhi family merupakan pertanyaan per
family; lihat [matriks augmentasi](/docs/reference/augmentation-matrix).

## export

```python
model.export(format="onnx", **kwargs) -> str
```

Mengembalikan path artefak yang ditulis. `format` di-resolve melalui registry
exporter, dengan `engine` sebagai alias `tensorrt` dan `litert` sebagai alias
`tflite`. Argumen yang digunakan bersama oleh setiap exporter:

| Argumen | Default | Arti |
|---|---|---|
| `output_path` | `None` | Path file output; dibuat di bawah `weights/` ketika tidak diberikan |
| `imgsz` | `None` | Tuple `(height, width)` atau satu int; default-nya ukuran native |
| `opset` | `None` | Versi opset ONNX |
| `simplify` | `True` | Menjalankan simplifikasi graph ONNX |
| `dynamic` | `True` | Mengaktifkan dynamic axis |
| `half` | `False` | Presisi FP16 |
| `int8` | `False` | Presisi INT8 |
| `batch` | `1` | Ukuran batch yang ditanam ke artefak |
| `device` | `None` | Device untuk tracing |
| `data` | `None` | data.yaml untuk kalibrasi INT8 |
| `fraction` | `1.0` | Fraksi dataset kalibrasi yang digunakan |
| `allow_download_scripts` | `False` | Mengizinkan Python tertanam dalam pengunduhan YAML dataset |
| `verbose` | `False` | Logging exporter mendetail |

Kombinasi blocked memunculkan `NotImplementedError` saat preflight, sebelum
tracing. Cakupan dan aturannya tersedia pada halaman
[matriks ekspor](/docs/reference/export-matrix). Jika terdapat adapter LoRA
aktif, adapter dilipat ke bobot padat, dan penggabungan ini hanya dilakukan
setelah semua penolakan permintaan.

## save

```python
model.save(path) -> str
```

Menulis checkpoint LibreYOLO skema v1.0: state dict ditambah metadata yang
dijelaskan dalam [skema checkpoint](/docs/reference/checkpoint-schema). Model
terkuantisasi juga membawa manifest `quant`, sehingga `LibreYOLO(path)`
memulihkan struktur dan skala terkuantisasi.

## quantize, quant_info, dan dequantize

```python
model.quantize(
    recipe,
    calib="coco128.yaml",
    samples=128,
    batch=8,
    algorithm="auto",
    keep_high_precision=None,
    allow_download_scripts=False,
    verbose=True,
)
```

Melakukan kuantisasi di tempat dan mengembalikan model. `recipe` adalah salah
satu casting `fp16` dan `bf16`, resep Conv dan Linear `int8` dan `fp8`, atau
resep khusus Linear `w4a16`, `w4a8`, `nvfp4`, `mxfp4`, dan `int2`, yang
didukung family transformer seperti RF-DETR. `int2` memerlukan QAT. `calib`
menerima path data.yaml atau nama dataset bawaan dan membaca gambar hanya untuk
forward; label tidak pernah dibaca. Berikan `calib=None` untuk melewati
kalibrasi. `algorithm` adalah `"minmax"`, `"percentile"`, atau `"auto"`.

`model.quant_info()` mengembalikan ringkasan status kuantisasi, atau `None` untuk
model float. `model.dequantize()` memulihkan modul float di tempat sambil
mempertahankan master weight hasil pelatihan kuantisasi, yang menjadi penghubung
dari QAT ke `export(format="onnx", int8=True, data=...)`.

## info dan lapisan

```python
model.info(detailed=False, verbose=True) -> Dict[str, Any]
model.get_available_layer_names() -> List[str]
model.get_distill_config() -> Dict
```

`info` mengembalikan dictionary yang ramah JSON dan mencatat ringkasan yang dapat
dibaca manusia ketika `verbose` bernilai true. `get_available_layer_names`
mencantumkan lapisan yang dapat disebut konfigurasi distillation atau ekstraksi
fitur.

## CUDA graph

Tersedia pada family yang atribut kelas `SUPPORTS_CUDA_GRAPH`-nya bernilai true.
Replay identik secara bit dengan eksekusi eager.

```python
model.capture_graph(imgsz=None, batch=1, dtype=None) -> None
model.cuda_graph_scope(mode=True)          # context manager
model.graph_info() -> Dict[str, Any]
model.release_graphs() -> None
```

Graph yang ditangkap hanya valid untuk bentuk persis saat penangkapan, sehingga
`batch` dan `imgsz` harus cocok dengan pemanggilan `predict` berikutnya.
`capture_graph` memindahkan biaya capture dari permintaan pertama. `mode`
menerima `True` atau `"on"` untuk menangkap saat penggunaan pertama, `"auto"`
untuk menunggu hingga bentuk berulang, dan `False` sebagai no-op.
`capture_graph` memunculkan `NotImplementedError` ketika family belum ikut serta
dan `CudaGraphUnavailable` ketika capture gagal.

## Device dan dtype

Objek `Results` memiliki `.to()`, `.cpu()`, `.cuda()`, dan `.numpy()`; lihat
[Jenis Results](/docs/reference/results-types). Model dipindahkan dengan
memberikan `device=` kepada `predict`, atau saat konstruksi.
