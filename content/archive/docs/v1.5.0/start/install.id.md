---
title: Instalasi
seo_title: Menginstal LibreYOLO
description: >-
  Instal LibreYOLO dari PyPI, pilih ekstra opsional yang diperlukan family model
  atau target ekspor, dan pastikan PyTorch melihat GPU.
lead: >-
  LibreYOLO diterbitkan di PyPI sebagai libreyolo. Paket dasar mencakup
  prediksi, pelatihan, validasi, dan family model yang hanya memerlukan PyTorch;
  ekstra opsional menambahkan sisanya.
keywords:
  - cara install libreyolo
  - pip install libreyolo
  - ekstra libreyolo
  - libreyolo cuda
  - libreyolo gpu
  - dependency libreyolo
last_verified: 1.5.0
meta:
  - label: Paket
    value: libreyolo
    mono: true
  - label: Python
    value: 3.10 atau lebih baru
  - label: Lisensi kode
    value: MIT
  - label: Dependency inti
    value: PyTorch 2.4 atau lebih baru
snippets:
  install:
    - label: pip
      language: bash
      code: |
        pip install libreyolo
    - label: Dengan ekstra
      language: bash
      code: >
        # Pisahkan dengan koma untuk menggabungkan beberapa ekstra dalam satu
        instalasi.

        pip install "libreyolo[rfdetr,onnx]"
    - label: Semuanya
      language: bash
      code: |
        pip install "libreyolo[all]"
    - label: Dari sumber
      language: bash
      code: |
        git clone https://github.com/LibreYOLO/libreyolo.git
        cd libreyolo
        pip install -e .
  verify:
    - label: CLI
      language: bash
      code: |
        # Python, Torch, CUDA, cuDNN, setiap GPU yang terlihat, dan
        # paket opsional yang terinstal.
        libreyolo checks
    - label: Python
      language: python
      code: |
        import libreyolo

        print(libreyolo.__version__)
    - label: Inventaris model
      language: bash
      code: |
        # Setiap family terdaftar beserta task, ukuran, dan resolusi input.
        # Family yang ekstra-nya belum ada dicantumkan bersama perintah pip
        # yang mengaktifkannya.
        libreyolo models
source_hash: 34fc6d3e24d03fb4
---

## Instalasi

<code-tabs name="install" />

Python 3.10 atau lebih baru diperlukan. Instalasi dasar menarik PyTorch,
torchvision, NumPy, Pillow, OpenCV, PyYAML, requests, mss, tqdm, pycocotools,
typer, click, safetensors, dan SciPy, sehingga YOLOv9 serta family lain yang
tidak memerlukan paket tambahan langsung berfungsi setelah
`pip install libreyolo`.

Clone mengambil branch `release`, yaitu branch stabil dengan kode yang cocok
dengan dokumentasi ini. Branch integrasi yang memuat pekerjaan belum dirilis
adalah `dev`.

## Ekstra opsional

Ekstra adalah nama dalam kurung siku yang menambahkan dependency yang diperlukan
satu family model atau target ekspor. Tidak ada perubahan lain: API tetap sama
baik ekstra tersedia maupun tidak.

### Family model

| Ekstra | Menambahkan |
|---|---|
| `rfdetr` | `transformers`, yang menyediakan backbone RF-DETR |
| `eomt` | `transformers` |
| `midas` | `timm` 1.0.x, yang menyediakan encoder ViT-L/16 dan EfficientNet-Lite3 milik MiDaS |
| `vlm` | `transformers`, `num2words`, `decord`, `lmdb`, `peft` |
| `sam` | `transformers`, `timm` |
| `openvocab` | `transformers`, `timm`, `regex`, `ftfy` |
| `sensenova` | `transformers`, `accelerate`, dan `bitsandbytes` kecuali macOS |
| `modus` | `transformers`, `accelerate` |
| `clip` | `regex` dan `ftfy`, diperlukan tokenizer teks CLIP yang disertakan |
| `siglip2` | `sentencepiece`, diperlukan tokenizer SigLIP 2 multibahasa |
| `gaze` | `gdown`, yang mengaktifkan pengunduhan otomatis checkpoint L2CS |
| `rtdetr` | Tidak ada. RT-DETR tidak memerlukan dependency ekstra; namanya dipertahankan stabil |

### Ekspor dan runtime

| Ekstra | Menambahkan |
|---|---|
| `onnx` | `onnx`, `onnxsim`, `onnxruntime` |
| `tensorrt` | `tensorrt-cu12` 10.16.1.11 dan `pycuda`, kecuali macOS |
| `openvino` | `openvino` |
| `coreml` | `coremltools` |
| `coreai` | `coreai-torch`, khusus macOS |
| `tflite`, alias `litert` | `libreyolo[onnx]` ditambah `onnx2tf`, `ai-edge-litert`, `onnx-graphsurgeon`, dan `onnx-simplifier` |
| `mnn` | `libreyolo[onnx]` ditambah `MNN` |
| `ncnn` | `pnnx` dan `ncnn` |
| `paddle` | `libreyolo[onnx]` ditambah `paddlepaddle` 2.6.2 dan `x2paddle` 1.6.0 |
| `executorch` | `executorch` |
| `triton` | `tritonclient[http]` untuk inferensi V2 HTTP dan HTTPS |

### Pelatihan, evaluasi, dan logging

| Ekstra | Menambahkan |
|---|---|
| `lora` | `libreyolo[rfdetr]` ditambah `peft`, untuk fine-tuning `lora=True` |
| `plots` | `matplotlib` |
| `fast-eval` | `faster-coco-eval`, backend evaluasi COCO C++ |
| `tensorboard` | `tensorboard` |
| `mlflow` | `mlflow` |
| `wandb` | `wandb` |
| `comet` | `comet-ml` |
| `clearml` | `clearml` |
| `neptune` | `neptune-scale` |
| `dvclive`, alias `dvc` | `dvclive` |

`fast-eval` bersifat opt-in, bukan dependency wajib, agar platform tanpa wheel
prebuilt tidak merusak instalasi biasa. Jika paket tidak tersedia, evaluasi
COCO kembali ke pycocotools dan run berlanjut.

### Alat

| Ekstra | Menambahkan |
|---|---|
| `stream` | `yt-dlp`, hanya diperlukan untuk me-resolve URL halaman YouTube |
| `tracking` | Tidak ada. Setiap dependency tracking sudah menjadi dependency inti |
| `label` | `libreyolo[sam]`, yang mengaktifkan bantuan click-to-mask dalam `libreyolo label` |
| `hub-kernels` | `kernels`, loader opsional untuk kernel Hub terkompilasi. Lihat [kernel](/docs/reference/kernels), yang mencatat bahwa instalasinya dapat menggeser prediksi RF-DETR dalam toleransi float |
| `clip-convert` | `libreyolo[clip]` ditambah `open_clip_torch`, untuk konversi bobot dan pemeriksaan paritas |
| `siglip2-convert` | `libreyolo[siglip2]` ditambah `transformers`, untuk alasan yang sama |

Webcam, RTSP, RTMP, TCP, UDP, HLS, dan list multi-stream lokal tidak memerlukan
ekstra. Hanya URL halaman YouTube yang memerlukannya.

### Ekstra agregat

`libreyolo[all]` menginstal ekstra model, ekspor, tracking, dan logging dalam
satu perintah. Beberapa sengaja tidak disertakan. `neptune` dikecualikan karena
`neptune-scale` stabil memerlukan protobuf di bawah 7, sedangkan jalur TFLite
memerlukan protobuf 7. `executorch` dikecualikan karena ExecuTorch membatasi
versi PyTorch pasangannya, dan `coreai` karena `coreai-torch` mengunci PyTorch
ke 2.11.x serta akan memindahkan seluruh environment ke versi tersebut.
`fast-eval`, `hub-kernels`, `clip-convert`, dan `siglip2-convert` juga tidak
disertakan. Instal ekstra tersebut berdasarkan nama.

## Constraint platform

Tiga ekstra dibatasi platform melalui marker dependency, sehingga instalasi
berhasil di semua tempat dan hanya menginstal lebih sedikit jika wheel tidak ada.

| Ekstra | Constraint |
|---|---|
| `coreai` | Khusus macOS. Toolchain Core AI tidak melakukan konversi maupun berjalan di tempat lain |
| `tensorrt` | Dilewati pada macOS, yang tidak memiliki CUDA |
| `tflite`, `litert` | `onnx2tf` dan `ai-edge-litert` memerlukan Python 3.12 atau lebih baru |

`sensenova` melewati `bitsandbytes` pada macOS karena tidak ada wheel yang
diterbitkan; bagian lain ekstra tetap diinstal secara normal.

Jika disk menjadi constraint, sebagian besar ruang digunakan PyTorch, dan
sebagian besar PyTorch adalah payload CUDA dalam wheel default. Wheel khusus
CPU menghapusnya tanpa mengurangi fitur. Untuk deteksi ONNX pada mesin yang
tidak boleh memiliki torch, lihat [instalasi ringan](/docs/lightweight-install).

## GPU dan CUDA

Pemilihan device dilakukan saat model dibuat. Default `device="auto"`
menggunakan CUDA ketika `torch.cuda.is_available()` bernilai true, kemudian
Metal Performance Shaders ketika `torch.backends.mps.is_available()` bernilai
true, dan CPU jika tidak. Tidak ada bagian library lain yang memeriksa hardware,
sehingga jika PyTorch tidak melihat GPU, LibreYOLO juga tidak dapat melihatnya.

Untuk menetapkan device, berikan `device` kepada model atau `predict`, `train`,
`val`, dan `export`. Nilai yang diterima adalah `"cpu"`, `"cuda"`, `"cuda:0"`,
`"mps"`, bilangan bulat seperti `0`, atau string digit seperti `"0"`; dua nilai
terakhir dikembangkan menjadi `cuda:<n>`.

Mulailah dengan `libreyolo checks`, yang mencetak versi Torch, versi CUDA dan
cuDNN yang menjadi dasar build Torch, serta setiap GPU terlihat beserta
memorinya. Jika perintah melaporkan tidak ada CUDA pada mesin dengan kartu
NVIDIA, wheel PyTorch yang diselesaikan pip adalah build CPU. Instal build CUDA
dari indeks PyTorch terlebih dahulu, lalu instal LibreYOLO:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install libreyolo
```

Itulah indeks yang sama dengan yang dikunci versinya repositori untuk environment kelolaan
uv pada Linux dan Windows. Driver NVIDIA 555 atau lebih baru diperlukan, sesuai
persyaratan runtime CUDA 12.8. macOS tetap menggunakan wheel PyPI karena host
unduhan PyTorch tidak menerbitkan build Darwin.

## Periksa instalasi

<code-tabs name="verify" />

`libreyolo models` adalah cara tercepat melihat apakah ekstra telah aktif:
family dengan dependency yang hilang dicetak bersama perintah pip persis yang
mengaktifkannya. Kedua perintah juga menerima `--json`, yang mencetak data sama
sebagai objek machine-readable ke stdout.


