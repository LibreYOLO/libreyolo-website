---
title: NVIDIA Jetson
seo_title: Memasang LibreYOLO dan PyTorch di NVIDIA Jetson
description: >-
  Pasang LibreYOLO di NVIDIA Jetson: empat library CUDA yang tidak disertakan
  JetPack, langkah --no-deps yang dibutuhkan PyTorch, dan angka hasil ukur di
  Orin Nano.
lead: >-
  Board NVIDIA Jetson menjalankan LibreYOLO dengan wheel PyTorch aarch64
  standar. Tidak ada build torch khusus Jetson yang terlibat, tetapi JetPack
  tidak menyertakan empat library yang dirujuk torch saat linking, dan instalasi
  harus menyediakannya.
keywords:
  - NVIDIA Jetson
  - Jetson Orin Nano
  - JetPack 7.2
  - install pytorch di jetson
  - nvidia-cudnn-cu13
  - nvidia-nccl-cu13
  - nvidia-cusparselt-cu13
  - nvidia-nvshmem-cu13
  - torch.cuda.is_available
  - no kernel image is available for execution on the device
  - tensorrt di jetson
  - wheel aarch64
last_verified: 1.4.0
meta:
  - label: Board
    value: 'Jetson Orin Nano Super Developer Kit, 8 GB, compute capability GPU 8.7'
  - label: Platform
    value: 'JetPack 7.2 (L4T R39.2), Ubuntu 24.04, CUDA 13, Python 3.12.3, aarch64'
  - label: Stack yang diuji
    value: >-
      libreyolo 1.4.0, torch 2.13.0+cu130, torchvision 0.28.0+cu130, opencv
      5.0.0, numpy 2.5.1, pada 2026-07-27
  - label: Tidak ada di JetPack
    value: >-
      nvidia-cudnn-cu13, nvidia-nccl-cu13, nvidia-cusparselt-cu13,
      nvidia-nvshmem-cu13
    mono: true
  - label: Benchmark
    value: >-
      223 run terverifikasi di board ini, 58 model dari 12 family, di PyTorch,
      ONNX Runtime dan TensorRT
    links:
      - label: visionanalysis.org/hardware/jetson_orin
        href: 'https://www.visionanalysis.org/hardware/jetson_orin'
  - label: Dilacak di
    value: Bagian Jetson dari issue 648
    links:
      - label: issue 648
        href: 'https://github.com/LibreYOLO/libreyolo/issues/648'
verification: >-
  Resep instalasi dan output yang diharapkan diambil dari instalasi 2026-07-27
  di sebuah Jetson Orin Nano Super. Baris latensi dan akurasi berasal dari
  snapshot hasil terverifikasi di balik visionanalysis.org, difilter ke hardware
  jetson_orin, diukur pada Juni 2026 dengan libreyolo 1.2.0.dev0. Perilaku
  ekspor dan loader dibaca dari libreyolo/export/exporter.py,
  libreyolo/export/tensorrt.py dan libreyolo/models/__init__.py.
snippets:
  prep:
    - label: Paket sistem dan virtual environment
      language: bash
      code: |
        # JetPack tidak memasang pip atau modul venv secara bawaan.
        sudo apt update
        sudo apt install -y python3.12-venv python3-pip

        python3 -m venv ~/libreyolo
        source ~/libreyolo/bin/activate
        pip install -U pip wheel setuptools
  torch:
    - label: 'PyTorch, dari index wheel CUDA 13'
      language: bash
      code: |
        pip install torch torchvision \
          --index-url https://download.pytorch.org/whl/cu130 \
          --extra-index-url https://pypi.org/simple
    - label: Empat library yang tidak disertakan JetPack
      language: bash
      code: |
        pip install nvidia-cudnn-cu13 nvidia-nccl-cu13 \
                    nvidia-cusparselt-cu13 nvidia-nvshmem-cu13
    - label: 'Jika pip menuntut cuda-toolkit 13.0.3, pasang dengan --no-deps'
      language: bash
      code: |
        # --no-deps berarti dependensi Python torch juga harus disebut manual.
        pip install --no-deps \
          torch torchvision \
          nvidia-cudnn-cu13 nvidia-nccl-cu13 \
          nvidia-cusparselt-cu13 nvidia-nvshmem-cu13 \
          filelock typing_extensions sympy networkx jinja2 markupsafe mpmath \
          fsspec numpy pillow
  ldd:
    - label: 'Sebutkan library hilang berikutnya, jangan menebak'
      language: bash
      code: >
        ldd
        "$VIRTUAL_ENV/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so" \
          | grep "not found"

        # Semua yang masih hilang di seluruh library torch, dalam satu langkah:

        ldd "$VIRTUAL_ENV"/lib/python3.12/site-packages/torch/lib/*.so
        2>/dev/null \
          | grep "not found" | sort -u
  install:
    - label: 'Pasang LibreYOLO setelah torch, bukan sebelumnya'
      language: bash
      code: |
        # torch sudah terpenuhi, jadi pip membiarkan build CUDA apa adanya.
        pip install libreyolo

        # Extra ONNX hanya dibutuhkan untuk mengekspor. Ekspor TensorRT berjalan
        # lewat ONNX, jadi pasang sebelum bagian ekspor di bawah.
        pip install "libreyolo[onnx]"
  verify:
    - label: Versi dan perangkat
      language: python
      code: |
        import cv2
        import numpy
        import torch

        import libreyolo

        print("torch", torch.__version__, "cuda", torch.cuda.is_available())
        print("gpu", torch.cuda.get_device_name(0))
        print("libreyolo", libreyolo.__version__)
        print("cv2", cv2.__version__, "numpy", numpy.__version__)
      expect: |
        torch 2.13.0+cu130 cuda True
        gpu Orin
        libreyolo 1.4.0
        cv2 5.0.0 numpy 2.5.1
    - label: Lalu jalankan kernel sungguhan
      language: python
      code: |
        import torch

        x = torch.rand(2000, 2000, device="cuda")
        print(float((x @ x).sum()))
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO9, SAMPLE_IMAGE

        # Mengunduh checkpoint saat pertama kali dipakai.
        model = LibreYOLO9("libreyolo9s.pt", size="s")

        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes)
    - label: CLI
      language: bash
      code: >
        libreyolo predict --source
        https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        --model libreyolo9s.pt --save
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO, LibreYOLO9, SAMPLE_IMAGE


        # Menulis libreyolo9s.onnx, lalu membangun libreyolo9s.engine darinya.

        LibreYOLO9("libreyolo9s.pt", size="s").export(format="tensorrt",
        half=True)


        # Engine dimuat kembali lewat entry point yang sama.

        result = LibreYOLO("libreyolo9s.engine").predict(SAMPLE_IMAGE)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model libreyolo9s.pt --format tensorrt --half
  power:
    - label: Mode daya dan clock
      language: bash
      code: >
        sudo nvpmodel -q      # mode apa saja yang tersedia di board ini, dan
        mana yang aktif

        sudo nvpmodel -m 0    # mode tertinggi di board yang diuji di sini

        sudo jetson_clocks


        tegrastats            # beban live; nvidia-smi terbatas di Tegra
source_hash: c07ff908503e89b5
---

## Yang dicatat halaman ini

Halaman ini mencatat satu konfigurasi yang diverifikasi dari ujung ke ujung,
bukan matriks dukungan. Board yang dipakai adalah Jetson Orin Nano Super
Developer Kit dengan memori 8 GB yang menjalankan JetPack 7.2 (L4T R39.2, Ubuntu
24.04, CUDA 13, Python 3.12.3), dan stack yang berhasil berjalan di atasnya
adalah `libreyolo 1.4.0` dengan `torch 2.13.0+cu130`, OpenCV 5.0.0 dan NumPy
2.5.1. `torch.cuda.is_available()` mengembalikan `True` dan GPU melaporkan
dirinya sebagai `Orin`.

Rilis JetPack lain, board Jetson lain dan versi CUDA lain tidak diuji. Resep di
bawah adalah yang berhasil pada kombinasi tersebut.

Run itu dilakukan pada 2026-07-27 terhadap LibreYOLO 1.4.0, dan belum diulang di
perangkat 1.5.0: ini satu-satunya halaman di tree 1.5.0 yang masih membawa
verifikasi 1.4.0, karena itu front matter halaman ini menyebut
`last_verified: "1.4.0"`. Tidak ada perubahan di 1.5.0 yang menyentuh jalur
instalasi, empat library yang hilang atau flag ekspor yang dijelaskan di sini,
jadi perintahnya diperkirakan tetap berlaku, tetapi nomor versi pada output di
bawah adalah yang dicetak 1.4.0, bukan hasil ukur 1.5.0.

Ada dua hal di dalamnya yang bertentangan dengan yang dikatakan kebanyakan
panduan Jetson. Wheel yang dipakai adalah build aarch64 biasa yang dipublikasikan
untuk CUDA 13, jadi tidak perlu build torch khusus Jetson. Dan JetPack tidak
menyertakan empat library yang dirujuk wheel tersebut, sehingga `import torch`
gagal satu library demi satu library sampai keempatnya terpasang.

## Instalasi

Image JetPack datang tanpa pip dan tanpa modul `venv`, jadi keduanya dipasang
lebih dulu.

<code-tabs name="prep" />

Board 8 GB terasa sempit untuk checkpoint yang lebih besar. Menambahkan swap di
NVMe sebelum memuatnya menghindari proses dihentikan karena kehabisan memori di
tengah jalan.

Lalu PyTorch. Index CUDA 13 menyediakan wheel aarch64; index tambahan memasok
dependensi pure-Python dari PyPI.

<code-tabs name="torch" />

Empat wheel `nvidia-*-cu13` adalah bagian yang mudah terlewat. JetPack
menyediakan driver GPU, bukan cuDNN, NCCL, cuSPARSELt atau NVSHMEM, dan torch
menolak diimpor tanpa keempatnya. Memasang keempatnya sekaligus lebih cepat
daripada menemukannya satu exception demi satu exception.

Snippet ketiga menangani satu kegagalan spesifik: metadata dependensi torch untuk
build CUDA 13 meminta `cuda-toolkit==13.0.3`, yang tidak punya wheel aarch64 di
PyPI, sehingga resolusi gagal sebelum apa pun terunduh. `--no-deps` melewati
resolver, yang berarti setiap dependensi harus disebut di baris perintah.

LibreYOLO dipasang terakhir. Memasangnya lebih dulu membuat pip memilih torch
pilihannya sendiri, yang di platform ini bukan build CUDA.

<code-tabs name="install" />

Semua dependensi sisanya menemukan wheel aarch64 siap pakai, termasuk OpenCV,
NumPy, SciPy, pycocotools dan safetensors. Tidak ada yang dikompilasi dari
source.

## Memastikan CUDA berfungsi

<code-tabs name="verify" />

Snippet kedua sama pentingnya dengan yang pertama. Wheel yang dibangun untuk
arsitektur GPU yang salah tetap melaporkan `torch.cuda.is_available() == True`
lalu gagal pada operasi nyata pertama dengan `CUDA error: no kernel image is
available for execution on the device`. Perkalian matriks di perangkat adalah
pemeriksaan yang menangkapnya.

## Menjalankan prediksi

<code-tabs name="predict" />

`predict` mengembalikan objek `Results` yang sama seperti di platform lain, jadi
halaman model berlaku tanpa perubahan.

## Ekspor ke TensorRT

Di board ini, TensorRT lebih cepat daripada PyTorch maupun ONNX Runtime untuk
seluruh 55 model yang diukur di semua runtime.

<code-tabs name="export" />

`format="tensorrt"` menulis graph ONNX lebih dulu lalu membangun engine darinya,
jadi extra `onnx` harus terpasang. `LibreYOLO()` memilih jalur berdasarkan
akhiran berkas, jadi berkas `.engine` dimuat lewat pemanggilan yang sama dengan
checkpoint `.pt`.

Jangan pakai extra pip `tensorrt` di Jetson. Extra itu mengunci `tensorrt-cu12`,
sebuah build CUDA 12, di platform CUDA 13. Pakai TensorRT bawaan JetPack. Jika
`import tensorrt` gagal di dalam virtual environment padahal berhasil di luar,
buat ulang environment dengan `--system-site-packages` agar modul sistem
terlihat.

Engine TensorRT terserialisasi terikat pada perangkat, arsitektur GPU dan versi
TensorRT yang membangunnya. Engine yang dibangun di workstation tidak akan dimuat
di Jetson, jadi langkah build dijalankan di board.

## Hasil ukur di board ini

Latensi per gambar, ukuran batch 1, dari ujung ke ujung termasuk preprocessing
dan postprocessing, di COCO val2017 (subset 500 gambar) pada `conf=0.001` dan
`max_det=300`. Lima model dari 58 yang diukur:

| Model | Input (px) | PyTorch FP32 (ms) | ONNX FP32 (ms) | TensorRT FP32 (ms) | TensorRT FP16 (ms) | mAP 50-95 |
|---|---:|---:|---:|---:|---:|---:|
| DEIMv2-Atto | 320 | 64.9 | 22.8 | 12.3 | 11.2 | 27.49 |
| YOLOX-Tiny | 416 | 49.2 | 31.8 | 23.0 | 19.4 | 35.45 |
| YOLO9-t | 640 | 101.2 | 53.8 | 36.0 | 29.1 | 41.78 |
| RT-DETR-r18 | 640 | 98.3 | 103.7 | 45.3 | 25.7 | 49.72 |
| D-FINE-s | 640 | 96.8 | 96.1 | 44.7 | 33.1 | 53.45 |

Kolom mAP adalah skor dari run TensorRT FP16 itu sendiri. Di antara 55 model yang
diukur di keempat runtime, selisih terbesar antara skor PyTorch FP32 dan skor
TensorRT FP16 adalah 0.59 poin, pada DEIMv2-X. Runtime berbeda dalam kecepatan,
bukan dalam akurasi.

TensorRT FP32 lebih cepat daripada PyTorch maupun ONNX Runtime untuk seluruh 55
model tersebut. TensorRT FP16 juga lebih cepat daripada PyTorch FP32 untuk
kelima puluh lima model itu, dengan rentang 1.68x sampai 6.22x dan median 3.39x.
ONNX Runtime yang paling bervariasi: ia lebih lambat daripada PyTorch di 23 dari
55 model, termasuk baris RT-DETR-r18.

Kondisi di balik setiap angka: `libreyolo 1.2.0.dev0`, `torch 2.12.0+cu130`,
Python 3.12.3, CUDA 13, driver 595.78, ONNX Runtime 1.24.0, diukur pada Juni
2026. Latensi di Jetson juga bergantung pada mode daya yang aktif, yang tidak
tercatat dalam rekaman benchmark.

<code-tabs name="power" />

Seluruh 223 run, termasuk 53 model lainnya dan kolom akurasi lengkap,
dipublikasikan di
[halaman Jetson Orin di Vision Analysis](https://www.visionanalysis.org/hardware/jetson_orin).

## Pemecahan masalah

### import torch gagal dan menyebut sebuah shared library

Salah satu dari empat library di atas tidak ada. Alih-alih menebak yang mana,
baca langsung dari binernya:

<code-tabs name="ldd" />

Setiap entri yang hilang memetakan ke satu wheel:

| Library yang hilang | Wheel |
|---|---|
| cuDNN | `nvidia-cudnn-cu13` |
| NCCL | `nvidia-nccl-cu13` |
| cuSPARSELt | `nvidia-cusparselt-cu13` |
| NVSHMEM | `nvidia-nvshmem-cu13` |

### torch memperingatkan bahwa tidak ada build yang mendukung GPU ini

Panggilan CUDA pertama pada konfigurasi yang berhasil mencetak ini:

```text
UserWarning: Found GPU0 Orin which is of compute capability (CC) 8.7.
The following list shows the CCs this version of PyTorch was built for and the hardware CCs it supports:
- 8.0 which supports hardware CC >=8.0,<9.0 except {8.7}
- 9.0 which supports hardware CC >=9.0,<10.0
- 10.0 which supports hardware CC >=10.0,<11.0 except {10.1}
- 11.0 which supports hardware CC >=11.0,<12.0
- 12.0 which supports hardware CC >=12.0,<13.0
No published PyTorch CUDA builds for release 2.13.0+cu130 support this GPU.
```

Peringatan itu hanya kosmetik di board ini. Wheel membawa kernel `sm_80` dan Orin
menjalankannya. Peringatan yang sama muncul pada wheel sebelumnya dari index
tersebut, yang menghasilkan semua baris benchmark di atas. Pastikan dengan
perkalian matriks dari pemeriksaan CUDA, bukan dengan mempercayai atau tidak
mempercayai pesan itu.

### CUDA error: no kernel image is available for execution on the device

Wheel yang terpasang dibangun untuk arsitektur GPU yang berbeda. Ini yang terjadi
dengan wheel dari index `sbsa` milik NVIDIA, yang menyasar GPU ARM server, bukan
silikon Jetson. Pasang ulang dari index CUDA 13 di bagian instalasi.

### pip tidak menemukan cuda-toolkit 13.0.3

Tidak ada wheel aarch64 untuknya. Pakai bentuk `--no-deps` di bagian instalasi
dan sebutkan dependensi torch secara eksplisit.

### libnvpl_lapack_lp64_gomp.so.0: cannot open shared object file

Wheel torch aarch64 menautkan NVIDIA Performance Libraries untuk perhitungan CPU.
Pasang keduanya dan letakkan di library path:

```bash
pip install nvpl-lapack nvpl-blas --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130/
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvpl/lib:$LD_LIBRARY_PATH"
```

Index itu aman untuk dua library CPU tersebut. Build torch dari index itulah yang
menghasilkan kegagalan "no kernel image" di atas.

### Sumber wheel yang tidak cocok dengan JetPack 7.2

| Sumber | Hasil di Orin Nano Super |
|---|---|
| torch dari `pypi.jetson-ai-lab.io/sbsa/cu130` | Dibangun untuk GPU ARM server. Berhasil diimpor, melaporkan CUDA tersedia, lalu gagal dengan "no kernel image is available for execution on the device". |
| torch dari `pypi.jetson-ai-lab.io/jp6/*` | Build CUDA 12 dan Python 3.10. Keduanya tidak bisa dipasang di Python 3.12 image ini. |
| Container PyTorch JetPack 6 | Inisialisasi CUDA gagal dengan error 801 di host JetPack 7. |
| Membangun torch dari source | Berhasil, tetapi memakan waktu berjam-jam di board 8 GB dan tidak perlu setelah wheel CUDA 13 terpasang. |

## DeepStream

Untuk pipeline video penuh, bukan loop Python, ekspor dengan `deepstream=True`
dan jalankan graph-nya lewat `nvinfer`. Jalur itu punya halaman sendiri, termasuk
config `nvinfer` yang dihasilkan, build parser bounding box dan jebakan yang
sudah diketahui: [DeepStream](/docs/export/deepstream).

Pipeline DeepStream itu sendiri divalidasi di GPU diskret x86, bukan di Jetson.
Kontrak ekspornya tidak bergantung pada arsitektur, tetapi menjalankan pipeline
di aarch64 masih belum dilakukan.

## Tidak diverifikasi

- Rilis JetPack selain 7.2, dan rilis L4T selain R39.2.
- Board Jetson selain Orin Nano Super 8 GB.
- Pelatihan di board. Inferensi dan ekspor sudah dicoba; pelatihan belum.
- Engine INT8. Hanya baris FP32 dan FP16 yang ada untuk board ini.
- Ukuran batch di atas 1. Semua pengukuran di atas adalah batch 1.
