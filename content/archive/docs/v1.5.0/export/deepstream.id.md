---
title: NVIDIA DeepStream
seo_title: Menjalankan model YOLO di NVIDIA DeepStream
description: >-
  Ekspor model LibreYOLO untuk NVIDIA DeepStream: satu graf ONNX plus config
  nvinfer yang dihasilkan. Perintah persis untuk build parser dan untuk
  pipeline.
lead: >-
  NVIDIA DeepStream menjalankan inferensi lewat elemen nvinfer miliknya, yang
  butuh graf ONNX, berkas config yang cocok, dan parser bounding box. Menyetel
  deepstream=True pada ekspor ONNX menulis dua yang pertama dan menyambungkannya
  ke yang ketiga.
keywords:
  - deepstream yolo
  - ekspor yolo ke deepstream
  - nvinfer config yolo
  - custom bounding box parser deepstream
  - config_infer_primary
  - NvDsInferParseYolo
  - deepstream-app
  - tensorrt engine deepstream
  - jetson deepstream
meta:
  - label: Flag
    value: 'export(format="onnx", deepstream=True)'
    mono: true
  - label: Menulis
    value: 'Satu graf ONNX, config_infer_primary_<stem>.txt, dan <stem>_labels.txt'
  - label: Cakupan
    value: 43 kombinasi family dan task yang tersebar di sembilan task
  - label: Parser
    value: >-
      NvDsInferParseYolo, dari proyek DeepStream-Yolo berlisensi MIT karya
      Marcos Luciano. Dibangun sekali per perangkat.
    links:
      - label: github.com/marcoslucianops/DeepStream-Yolo
        href: 'https://github.com/marcoslucianops/DeepStream-Yolo'
  - label: Ketersediaan
    value: >-
      Hadir di v1.5.0. Digabungkan ke dev pada 2026-08-08 dalam pull request
      728.
    links:
      - label: pull request 728
        href: 'https://github.com/LibreYOLO/libreyolo/pull/728'
      - label: issue 648
        href: 'https://github.com/LibreYOLO/libreyolo/issues/648'
  - label: Divalidasi saat runtime
    value: 'DeepStream 8.0.0 pada RTX 5070 Ti, hanya deteksi, 2026-08-08'
verification: >-
  Ditulis dari validasi runtime 2026-08-08. Daftar family, kunci config, dan
  nilai bawaan dibaca dari libreyolo/export/deepstream.py dan
  libreyolo/export/exporter.py pada commit 5f81e11e, yang digabungkan ke dev
  pada hari yang sama dalam pull request 728.
snippets:
  install:
    - label: Instalasi
      language: bash
      code: |
        pip install "libreyolo[onnx]"
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO9, LibreDFINE


        # Menulis libreyolo9s.onnx, config_infer_primary_libreyolo9s.txt

        # dan libreyolo9s_labels.txt ke direktori kerja.

        LibreYOLO9("libreyolo9s.pt", size="s").export(format="onnx",
        deepstream=True)


        # Simpan setiap model deteksi di direktorinya sendiri: setiap config

        # deteksi menamai berkas cache engine yang sama. Lihat "Jebakan yang
        diketahui".

        LibreDFINE("LibreDFINEs.pt", size="s").export(format="onnx",
        deepstream=True)
    - label: Argumen
      language: python
      code: |
        model.export(
            format="onnx",     # deepstream=True ditolak untuk semua format lain
            deepstream=True,
            conf=0.25,         # mengisi pre-cluster-threshold (dan classifier-threshold,
                               # segmentation-threshold pada task tersebut)
            iou=0.45,          # mengisi nms-iou-threshold, dihilangkan saat cluster-mode=4
            batch=1,           # mengisi batch-size dan nama berkas cache engine
            half=False,        # True menandai config network-mode=2 (build fp16)
            int8=False,        # True menandai config network-mode=1
            dynamic=True,      # sumbu batch dinamis di graf ONNX
            imgsz=640,         # mengisi infer-dims=3;H;W
        )

        # deepstream=True dan nms=True saling eksklusif: DeepStream menjalankan
        # supresi di tahap clustering-nya, jadi tidak ada yang ditanam di graf.
    - label: Ambil bobot D-FINE lebih dulu
      language: bash
      code: |
        curl -L -o LibreDFINEs.pt \
          https://huggingface.co/LibreYOLO/LibreDFINEs/resolve/main/LibreDFINEs.pt
  gpu:
    - label: Pastikan passthrough GPU sebelum hal lain
      language: bash
      code: |
        docker run --rm --gpus all nvcr.io/nvidia/tritonserver:26.04-py3 \
          nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
      expect: |
        name, driver_version, compute_cap
        NVIDIA GeForce RTX 5070 Ti, 591.86, 12.0
  parser:
    - label: 'build_parser.sh, jalankan di dalam container DeepStream'
      language: bash
      code: >
        set -e

        git clone --depth 1
        https://github.com/marcoslucianops/DeepStream-Yolo.git


        # /usr/local/cuda-12 pada image ini hanya stub dan build mati di sana
        dengan

        # "fatal error: crt/host_defines.h: No such file or directory". Cari

        # toolkit yang benar-benar memuat header itu; pada image 8.0 itu
        cuda-12.5.

        CUDA_DIR=$(readlink -f /usr/local/cuda)

        [ -f "$CUDA_DIR/include/crt/host_defines.h" ] || \
          CUDA_DIR=$(ls -d /usr/local/cuda-*.* | sort -Vr | \
                     while read d; do [ -f "$d/include/crt/host_defines.h" ] && echo "$d" && break; done)

        # Image ini menyertakan libcublas.so.12 dan libcublas.so.12.8.4.1 tetapi
        bukan

        # libcublas.so tanpa versi yang dibutuhkan -lcublas, jadi tahap link
        gagal dengan

        # "/usr/bin/ld: cannot find -lcublas". Beri linker nama yang dicarinya.

        mkdir -p /tmp/cudalibs

        for lib in cublas cublasLt cudart; do
          real=$(find /usr/local -name "lib${lib}.so.1*" | grep -v stubs | sort -V | tail -1)
          ln -sf "$real" "/tmp/cudalibs/lib${lib}.so"
        done

        export LIBRARY_PATH="/tmp/cudalibs:$LIBRARY_PATH"


        make -C DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
        CUDA_VER="${CUDA_DIR##*/cuda-}"
    - label: Segmentasi instance memakai parser yang berbeda
      language: bash
      code: >
        git clone --depth 1
        https://github.com/marcoslucianops/DeepStream-Yolo-Seg.git

        make -C DeepStream-Yolo-Seg/nvdsinfer_custom_impl_Yolo_seg \
          CUDA_VER="${CUDA_DIR##*/cuda-}"
  run:
    - label: deepstream_app_config.txt
      language: text
      code: >
        [application]

        enable-perf-measurement=1

        perf-measurement-interval-sec=5

        gie-kitti-output-dir=kitti


        [tiled-display]

        enable=0


        [source0]

        enable=1

        type=3

        uri=file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4

        num-sources=1

        gpu-id=0


        [streammux]

        gpu-id=0

        batch-size=1

        batched-push-timeout=40000

        width=1920

        height=1080

        live-source=0


        [primary-gie]

        enable=1

        gpu-id=0

        gie-unique-id=1

        config-file=config_infer_primary_libreyolo9s.txt


        [osd]

        enable=1

        border-width=2

        text-size=15


        [sink0]

        enable=1

        type=1

        sync=0


        [tests]

        file-loop=0
    - label: Jalankan
      language: bash
      code: |
        deepstream-app -c deepstream_app_config.txt
      expect: |
        App run successful
    - label: Kedua langkah dalam satu container
      language: bash
      code: |
        docker run --rm --gpus all -v "$PWD:/work" -w /work \
          nvcr.io/nvidia/deepstream:8.0-samples-multiarch \
          bash -c "bash build_parser.sh && deepstream-app -c deepstream_app_config.txt"
source_hash: 1ee91c265753dd9a
---

## Ketersediaan

Ekspor DeepStream hadir di v1.5.0. Sudah digabungkan ke `dev` pada 2026-08-08
dalam pull request 728, jadi instalasi terbaru sudah memilikinya dan tidak perlu
pin ke branch mana pun.

<code-tabs name="install" />

Jika Anda sudah melakukan clone branch `deepstream-export` sebelum 2026-08-08,
ganti dengan yang baru. Branch itu sudah melalui rebase dan force-push, dan
riwayat lamanya kehilangan perbaikan yang membuat ekspor ini bisa berjalan di
mesin CUDA sama sekali.

## Apa yang ditulis ekspor

`model.export(format="onnx", deepstream=True)` menulis tiga berkas berdampingan.
Untuk `libreyolo9s.pt`:

- `libreyolo9s.onnx`, graf deteksi, satu tensor keluaran berbentuk
  `(batch, num_detections, 6)`, setiap baris `[x1, y1, x2, y2, score, class_id]`
  dalam koordinat piksel input jaringan.
- `config_infer_primary_libreyolo9s.txt`, konfigurasi `nvinfer` yang membawa
  konstanta preprocessing family tersebut, jumlah kelas, ambang batas, dan
  sambungan ke parser.
- `libreyolo9s_labels.txt`, satu nama kelas per baris.

Berkas label muncul setiap kali checkpoint membawa nama kelas. Model kedalaman
tidak punya nama kelas, jadi tidak mendapat berkas itu maupun kunci
`labelfile-path`.

LibreYOLO tidak menghasilkan `.so`. `.so` yang dimuat DeepStream adalah parser
bounding box dari `marcoslucianops/DeepStream-Yolo`, dikompilasi sekali per
perangkat, dan biner yang sama dipakai untuk detektor LibreYOLO mana pun yang
Anda arahkan ke sana. Modelnya adalah ONNX. Klasifikasi dan segmentasi semantik
sama sekali tidak butuh parser, karena `nvinfer` melakukan post-processing
sendiri untuk keduanya.

## Mengekspor model

<code-tabs name="export" />

`LibreDFINE._load_weights` melempar `FileNotFoundError` bila berkasnya belum ada
di disk, tanpa mencoba mengunduh, jadi ambil `LibreDFINEs.pt` sendiri lebih
dulu. Celah itu dilacak sebagai
[issue #727](https://github.com/LibreYOLO/libreyolo/issues/727). Bobot YOLO9
diunduh saat pertama kali dipakai.

Flag ini hanya ada di Python. `libreyolo export` pada branch ini tidak punya
opsi `deepstream`, dan CLI menyusun argumen ekspornya dari daftar tetap alih-alih
meneruskan kunci yang tidak dikenal.

## Membangun parser bounding box

Deteksi butuh library parser, segmentasi instance butuh parser yang berbeda, dan
task selebihnya tidak butuh sama sekali. Dua hal pada image DeepStream 8.0
merusak perintah build yang didokumentasikan, dan keduanya masalah lingkungan,
bukan masalah LibreYOLO.

Image itu menyertakan `cuda`, `cuda-12`, `cuda-12.5`, `cuda-12.8` dan `cuda-12.9`
di bawah `/usr/local`. Hanya `cuda-12.5` yang toolkit-nya lengkap. Image itu juga
menyertakan `libcublas.so.12` dan `libcublas.so.12.8.4.1` tetapi bukan
`libcublas.so` tanpa versi yang menjadi acuan `-lcublas`. Skrip di bawah
menyiasati keduanya.

<code-tabs name="parser" />

Lalu arahkan `custom-lib-path` di config yang dihasilkan ke
`libnvdsinfer_custom_impl_Yolo.so` hasil build. Nilai yang dihasilkan adalah path
relatif `nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so`, yang cocok
bila `deepstream-app` dijalankan dari checkout `DeepStream-Yolo` dan perlu
disunting bila tidak.

## Menjalankan pipeline

Pastikan container bisa melihat GPU sebelum membuang waktu untuk hal lain. Ini
pemeriksaan pertama yang dilakukan pada validasi, pada kartu Blackwell di bawah
WSL2.

<code-tabs name="gpu" />

Validasi menjalankan `deepstream-app` dengan satu sumber berkas, tanpa sink
tampilan, on-screen display menyala, dan `gie-kitti-output-dir` diatur sehingga
deteksi setiap frame tersimpan ke disk sebagai teks KITTI. Config dengan
pengaturan tersebut:

<code-tabs name="run" />

`nvinfer` membangun engine TensorRT dari ONNX pada run pertama dan menyimpannya
di cache di sebelah model, jadi run pertama membayar biaya build engine dan run
berikutnya tinggal memuat cache.

## Config yang dihasilkan

Kedua config di bawah ditulis oleh exporter untuk validasi, tanpa disunting
sesudahnya.

| Kunci | YOLO9-s | D-FINE-s |
|---|---|---|
| `net-scale-factor` | 0.003921568627 | 0.003921568627 |
| `model-color-format` | 0 | 0 |
| `infer-dims` | 3;640;640 | 3;640;640 |
| `maintain-aspect-ratio` | 1 | 0 |
| `symmetric-padding` | 0 | 0 |
| `network-type` | 0 | 0 |
| `num-detected-classes` | 80 | 80 |
| `cluster-mode` | 2 | 4 |
| `parse-bbox-func-name` | NvDsInferParseYolo | NvDsInferParseYolo |
| `pre-cluster-threshold` | 0.25 | 0.25 |
| `nms-iou-threshold` | 0.45 | |
| `topk` | 300 | 300 |

Kedua config berbeda di tiga tempat: `maintain-aspect-ratio`, `cluster-mode`, dan
ada tidaknya `nms-iou-threshold`. Config D-FINE menghilangkan kunci itu
sepenuhnya, dan itulah yang diminta `cluster-mode=4`.

Head yang menghasilkan paling banyak satu prediksi per objek mendapat
`cluster-mode=4`, jadi DeepStream tidak menjalankan clustering atasnya;
clustering justru akan menggabungkan deteksi yang benar-benar berbeda. Itu
mencakup `rfdetr`, `dfine`, `deim`, `deimv2`, `ec`, `rtdetr`, `rtdetrv2`,
`rtdetrv4` dan `yolo9_e2e`. Head grid dan anchor mendapat `cluster-mode=2` plus
`nms-iou-threshold`.

Config deteksi juga membawa `engine-create-func-name=NvDsInferYoloCudaEngineGet`,
yang menyerahkan pembangunan engine ke library parser. Itulah yang mengunci nama
berkas cache engine, dan itu sumber tabrakan yang dijelaskan di bagian jebakan
yang diketahui.

## Task dan family yang didukung

Empat puluh tiga kombinasi family dan task bisa diekspor.
`deepstream_supported_tasks()` dan `deepstream_supported_families(task)` di
`libreyolo/export/deepstream.py` mengembalikan daftar yang sama saat runtime.

| Task | `network-type` | Library parser | Family |
|---|---|---|---|
| Deteksi | 0 | DeepStream-Yolo | yolo9, yolo9_p2, yolo9_e2e, yolo1, yolo2, yolo3, yolo4, yolo7, yolox, yolonas, rtmdet, picodet, rfdetr, dfine, deim, deimv2, ec, rtdetr, rtdetrv2, rtdetrv4 |
| Klasifikasi | 1 | Tidak perlu | mobilenetv4, convnext, efficientnetv2, resnet, dinov2 |
| Segmentasi semantik | 2 | Tidak perlu | pidnet, eomt, dinov2, lingbotvision |
| Segmentasi instance | 3 | DeepStream-Yolo-Seg | rfdetr, dfine, ec |
| Pose | 100 | Tidak perlu | yolo9, yolonas, rfdetr, ec |
| Kedalaman | 100 | Tidak perlu | depth_anything, zipdepth |
| Restorasi | 100 | Tidak perlu | nafnet, realesrgan, swinir |
| Matting | 100 | Tidak perlu | birefnet |
| Gaze | 100 | Tidak perlu | l2cs |

`network-type=100` berarti DeepStream tidak punya post-processor untuk task
tersebut. Config itu menyetel `output-tensor-meta=1`, keluaran asli graf lewat
tanpa diubah, dan aplikasi mendekodenya dari metadata tensor. Graf dengan banyak
keluaran tidak masalah di sana: setiap lapisan keluaran sampai ke metadata dengan
nama keluaran dan sumbu dinamis yang sama seperti ekspor ONNX biasa.

Baris segmentasi instance adalah baris deteksi diikuti mask instance tersebut,
diratakan pada `(netH / 4, netW / 4)`, resolusi yang ditetapkan secara tetap di
dalam parser seg, sebagai probabilitas untuk `segmentation-threshold`.

Klasifikasi dan gaze berjalan sebagai inferensi sekunder. Setel `process-mode=2`
dan `operate-on-gie-id` di config yang dihasilkan untuk menempatkan classifier di
belakang detektor. Gaze adalah kontrak head-only, satu crop wajah per input, jadi
butuh detektor wajah di depannya.

Tiga family sengaja tidak ada. `segformer` tidak tersambung ke kontrak ekspor
semantik bersama dan tidak bisa diekspor ke ONNX dalam format apa pun. RTMDet-Ins
dan YOLO9 punya ekspor segmentasi instance yang diblokir di dalam LibreYOLO
sendiri. `depth_anything3` belum punya implementasi ekspor.

Dua baris di tabel punya celah checkpoint di baliknya. Hanya checkpoint semantik
EoMT `l` yang dipublikasikan, dan klasifikasi DINOv2 sama sekali belum punya
checkpoint yang dipublikasikan, jadi kombinasi itu butuh bobot hasil fine-tuning
Anda sendiri.

## Perbedaan preprocessing

`nvinfer` menghitung `net-scale-factor * (x - offsets)` per kanal dengan skala
skalar, yang tidak bisa menyatakan standar deviasi per kanal. Family yang
membutuhkannya (`rfdetr`, `ec`, ukuran `deimv2` yang backbone-nya DINO, `rtmdet`,
`picodet`, dan semua family klasifikasi) menanamkan normalisasi itu di dalam graf
yang diekspor, dan config yang dihasilkan memberi graf tersebut ruang input mentah
yang cocok.

Geometri adalah tempat pipeline Python milik LibreYOLO sendiri dan `nvinfer` masih
berbeda:

- Family letterbox (`yolo9`, `yolox`, `yolonas`, `rtmdet`, `yolo2`, `yolo3`,
  `yolo4`, `yolo7`) secara native mengisi padding dengan abu-abu. `nvinfer`
  mengisinya dengan hitam.
- Deteksi `yolonas` secara native mengubah ukuran sisi terpanjang menjadi 636 di
  dalam kanvas 640-nya. `maintain-aspect-ratio` milik `nvinfer` memakai 640 penuh.
- Klasifikasi secara native mengubah ukuran sisi terpendek lalu melakukan
  center-crop. `nvinfer` merentangkan frame atau ROI objek ke input jaringan, jadi
  subjek dengan crop ketat akan berbeda.
- EoMT secara native menjalankan tile sliding-window untuk segmentasi semantik.
  Graf yang diekspor adalah satu kanvas yang direntangkan, yang lebih cepat dan
  kurang akurat.
- `pidnet` menghasilkan class map pada 1/8 resolusi input dan `lingbotvision` pada
  1/16. DeepStream melakukan upsample class map itu untuk ditampilkan.

Gerbang parity ONNX memberi tensor yang sudah melalui preprocessing, jadi ia
memeriksa keluaran graf dan tidak bisa menangkap urutan warna atau kebijakan
padding yang salah di config. Validasi pada data Anda sendiri sebelum menerapkan
beban kerja yang menuntut parity persis.

## Jebakan yang diketahui

### Dua model deteksi dalam satu direktori saling memuat engine

Setiap config deteksi membawa baris yang sama:

```ini
model-engine-file=model_b1_gpu0_fp32.engine
```

Pembangun engine milik parser mengharuskan basename itu dan namanya tidak berubah
menurut model. Ekspor model deteksi kedua ke direktori yang sama, dan run kedua
akan memuat engine hasil cache model pertama. Tidak ada yang crash; hanya box-nya
yang salah. Beri setiap model deteksi direktorinya sendiri. Validasi harus
mengisolasi D-FINE ke direktori tersendiri sebelum bisa diuji sama sekali.

### Satu box hanya bisa membawa satu kelas

Format baris `nvinfer` adalah `[x1, y1, x2, y2, score, class_id]`, satu kelas per
box, jadi ekspor meruntuhkan skor kelas menjadi argmax-nya. Box yang dilaporkan
`predict` di bawah dua kelas hanya bertahan di bawah satu kelas. Kasus terukur:
LibreYOLO melaporkan `vase 0.773` dan `bottle 0.383` pada box yang sama, dan graf
DeepStream mempertahankan `vase`. Ini konsekuensi dari format baris parser dan
tidak bisa diubah tanpa keluar dari kontrak itu, jadi ini perilaku yang
diharapkan, bukan regresi.

## Tervalidasi

`deepstream-app` berjalan sampai EOS dengan `App run successful` pada kedua tipe
head detektor, atas `sample_1080p_h264.mp4` bawaan NVIDIA (1443 frame), dengan
dump KITTI per frame diaktifkan.

| | YOLO9-s | D-FINE-s |
|---|---|---|
| Tipe head | grid | one-to-one |
| `cluster-mode` | 2 | 4 |
| `maintain-aspect-ratio` | 1 | 0 |
| Frame dengan deteksi | 1443 | 1443 |
| Total deteksi | 18031 | 71105 |

Histogram kelas atas seluruh 1443 frame menempatkan mobil di urutan pertama dan
orang di urutan kedua untuk kedua model, yang wajar untuk pemandangan jalan.
Selisih empat kali lipat pada jumlah deteksi adalah perbedaan `cluster-mode` yang
sedang bekerja: D-FINE pada `cluster-mode=4` tidak menjalankan clustering, jadi
setiap query di atas ambang batas bertahan, termasuk yang hampir duplikat.

Dua model yang dilatih secara independen menempatkan objek dominan di tempat yang
sama:

```text
YOLO9  bus  [706.72,  0.82, 1916.34, 1062.97]  conf 0.965
D-FINE bus  [702.73,  2.93, 1916.24, 1069.32]  conf 0.965
```

Run itu memastikan lima hal: TensorRT membangun engine dari ONNX yang diekspor
pada sm_120, `nvinfer` menerima setiap kunci di config yang dihasilkan,
`NvDsInferParseYolo` membaca layout tensor dengan benar, box mendarat di koordinat
resolusi sumber 1920x1080, dan label cocok dengan berkas label yang dihasilkan.

Lingkungan tempat validasi berjalan:

| Komponen | Nilai |
|---|---|
| Sistem operasi host | Windows 11 Pro 26200 |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB |
| Driver | 591.86 |
| Compute capability | 12.0 (Blackwell, sm_120) |
| Runtime container | Docker Desktop 29.4.3, backend WSL2 |
| Image DeepStream | `nvcr.io/nvidia/deepstream:8.0-samples-multiarch` |
| Versi DeepStream | 8.0.0 |
| CUDA container | 12.8.1 |
| Parser | `marcoslucianops/DeepStream-Yolo` pada HEAD |

Selain menjalankan pipeline, `tests/unit/test_deepstream_export.py` mencakup
adapter graf dan kunci config yang dihasilkan, dan 35 tesnya lulus pada commit
ini.

## Belum tervalidasi

Disebutkan agar cakupan di atas tidak dibaca lebih luas daripada yang sebenarnya.

- Jetson dan aarch64. Kontrak ekspor tidak bergantung pada arsitektur, tetapi
  pipeline baru dijalankan pada GPU diskrit x86.
- Empat puluh satu dari 43 kombinasi. Hanya deteksi dengan `yolo9` dan deteksi
  dengan `dfine` yang melewati DeepStream. Klasifikasi, segmentasi semantik,
  segmentasi instance, dan task tensor mentah tercakup oleh unit test dan
  pemeriksaan parity ONNX, bukan oleh run pipeline.
- FP16 dan INT8. Hanya `network-mode=0` yang diuji.
- Multi-stream dan batching. Satu sumber, `batch-size=1`.
- Akurasi terhadap dataset ground truth. Deteksi diperiksa untuk kewajaran
  semantik dan kecocokan antarmodel, bukan diskor sebagai mAP lewat DeepStream.
