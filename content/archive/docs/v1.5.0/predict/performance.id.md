---
title: Performa inferensi
seo_title: Inferensi lebih cepat di LibreYOLO
description: >-
  CUDA graph, half precision, batching, inferensi berbasis tile, dan augmentasi
  saat pengujian pada waktu prediksi, lengkap dengan default sebenarnya serta
  family yang mendukungnya.
lead: >-
  Lima kontrol saat prediksi mengubah throughput atau akurasi: replay CUDA
  graph, presisi, batching, tiling, dan augmentasi saat pengujian. Masing-masing
  berlaku pada kumpulan family tertentu, dan dua di antaranya mengorbankan
  akurasi atau latensi, bukan menghematnya.
keywords:
  - cuda graphs pytorch inference
  - yolo batch inference python
  - fp16 inference
  - tiled inference objek kecil
  - sliced inference gambar besar
  - test time augmentation detection
  - capture_graph
  - batch predict folder
last_verified: 1.5.0
verification: >-
  Nilai default argumen berasal dari InferenceRunner.__call__ dalam
  libreyolo/models/base/inference.py. API CUDA graph berasal dari
  BaseModel.capture_graph, graph_info, release_graphs, dan cuda_graph_scope
  dalam libreyolo/models/base/model.py; keikutsertaan family berasal dari
  variabel kelas SUPPORTS_CUDA_GRAPH. Perilaku half precision berasal dari
  NOOP_PREDICT_KWARGS dalam libreyolo/utils/predict_args.py, peringatan CLI
  dalam libreyolo/cli/commands/predict.py, serta CAST_RECIPES dan
  SUPPORTED_FAMILIES dalam libreyolo/quant/api.py. Kondisi batching berasal dari
  InferenceRunner._process_in_batches dan _predict_batch. Tiling berasal dari
  _predict_tiled dan _merge_tile_detections. Augmentasi saat pengujian berasal
  dari BaseModel._predict_augment dan _merge_tta, dengan TTA_ENABLED,
  TTA_SCALES, dan TTA_FIXED_SIZE yang dibaca di seluruh libreyolo/models/.
snippets:
  batch:
    - label: Inferensi batch pada folder
      language: python
      code: >
        from pathlib import Path

        from PIL import Image


        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        folder = Path("batch_demo")

        folder.mkdir(exist_ok=True)

        image = Image.open(SAMPLE_IMAGE)

        for index in range(8):
            image.save(folder / f"frame_{index}.jpg")

        model = LibreYOLO("LibreYOLO9s.pt")


        # Satu forward bertumpuk per bagian berisi 4 pada family yang
        mendukungnya.

        results = model(str(folder), batch=4)

        print(len(results), "results")
    - label: Streaming agar daftar tidak pernah dimaterialisasi
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        for result in model("batch_demo", batch=4, stream=True):
            print(len(result.boxes))
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt source=batch_demo batch=4
  graphs:
    - label: 'Capture di awal, lalu replay (memerlukan CUDA)'
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # Bayar biaya warmup dan capture sekali, di luar permintaan pertama.
        model.capture_graph()

        result = model(SAMPLE_IMAGE, cuda_graph=True)
        print(len(result.boxes))
        print(model.graph_info())
    - label: Capture hanya setelah bentuk berulang (memerlukan CUDA)
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")


        # "auto" menunggu bentuk terlihat dua kali, sehingga pekerjaan sekali
        jalan

        # tidak pernah membayar biaya capture.

        for _ in range(3):
            model(SAMPLE_IMAGE, cuda_graph="auto")

        print(model.graph_info())

        model.release_graphs()
  precision:
    - label: Pasang komponen tambahan ekspor
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: Ekspor dan muat kembali dengan presisi default
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")
        path = model.export(format="onnx")

        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: Ekspor FP16 (buat dan jalankan pada mesin CUDA)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")
        path = model.export(format="onnx", half=True)

        exported = LibreYOLO(path)
        result = exported(SAMPLE_IMAGE)
        print(len(result.boxes))
    - label: FP16 di PyTorch melalui resep cast (memerlukan CUDA)
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt", device="cuda")

        # Resep cast tidak membaca data kalibrasi.
        model.quantize(recipe="fp16", calib=None)

        result = model(SAMPLE_IMAGE)
        print(len(result.boxes))
  tiling:
    - label: Inferensi berbasis tile pada gambar besar
      language: python
      code: |
        from PIL import Image

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Tiling hanya aktif jika gambar lebih besar daripada ukuran input.
        large = Image.open(SAMPLE_IMAGE).resize((2048, 1536))
        large.save("large.jpg")

        model = LibreYOLO("LibreYOLO9s.pt")

        result = model("large.jpg", tiling=True, overlap_ratio=0.2)
        print(result.num_tiles, "tiles", len(result.boxes), "detections")
  tta:
    - label: Augmentasi saat pengujian
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        plain = model(SAMPLE_IMAGE)
        flipped = model(SAMPLE_IMAGE, augment=True)

        print(len(plain.boxes), "->", len(flipped.boxes))
source_hash: 3914665d0e7f892c
---

## Kontrol dan nilai default

Semuanya merupakan argumen untuk `predict`, dan semua default-nya nonaktif.

| Argumen | Default | Efek |
|---|---|---|
| `batch` | `1` | Gambar per forward pass untuk sumber folder dan daftar |
| `cuda_graph` | `False` | Replay forward dari CUDA graph yang sudah di-capture |
| `tiling` | `False` | Membagi gambar besar menjadi tile yang tumpang tindih |
| `overlap_ratio` | `0.2` | Tumpang tindih tile saat `tiling` aktif |
| `augment` | `False` | Menjalankan tampilan yang dibalik dan menggabungkannya |
| `half` | | Diterima, diberi peringatan, dan diabaikan |
| `device` | `None` | Memindahkan model sebelum prediksi |

`imgsz` juga memengaruhi biaya karena menetapkan resolusi tempat model berjalan, tetapi ini
terutama merupakan argumen akurasi dan lebih tepat dibahas bersama model.

## Batching

<code-tabs name="batch" />

`batch` berlaku untuk sumber folder dan daftar. Dengan `batch=1`, setiap gambar menjalankan
satu forward pass. Di atas `1`, setiap bagian dipraproses, ditumpuk menjadi satu tensor,
dijalankan sekali, lalu dipotong kembali agar pascapemrosesan satu gambar milik setiap family
tetap menerima bentuk yang diharapkan.

Jalur bertumpuk hanya digunakan jika semua kondisi berikut terpenuhi:

- `batch` lebih besar daripada `1`
- `tiling` nonaktif
- augmentasi saat pengujian tidak aktif
- family menetapkan `SUPPORTS_BATCHED_PREDICT`
- jaringan dasar tidak berada dalam mode pelatihan

Kondisi terakhir bukan sekadar detail teknis. Jaringan dalam mode pelatihan akan
menormalisasi bagian bertumpuk dengan statistik batch lintas gambar, sehingga gambar dalam
bagian yang sama saling mengubah prediksi; proses seperti itu tetap berjalan berurutan.

`SUPPORTS_BATCHED_PREDICT` memakai true secara default. Family berikut memilih keluar dan
menjalankan satu gambar per forward, terlepas dari `batch`: Depth Anything V2, Depth Anything
3, EoMT, Faster R-CNN, FCOS, HRNet, L2CS-Net, LibreMODUS, MiDaS, MoGe-2, PP-OCRv5,
Real-ESRGAN, RetinaNet, SAM 3D Body, SwinIR, YOLOv1, ZipDepth, setiap detektor
open-vocabulary, dan setiap vision language model.

Ada satu fallback lagi. Jika prapemrosesan tidak mengembalikan tensor `(1, C, H, W)` seragam
dengan bentuk, dtype, dan perangkat yang cocok di seluruh bagian, bagian tersebut berjalan
secara berurutan, bukan ditumpuk, sehingga kebenaran tidak pernah bergantung pada kebetulan
ukuran gambar sama.

Gabungkan `batch` dengan `stream=True` pada folder besar untuk memperoleh forward batch tanpa
menyimpan setiap hasil dalam memori.

## CUDA graph

<code-tabs name="graphs" />

CUDA graph merekam forward pass sekali dan melakukan replay sebagai satu peluncuran. Detektor
kecil menghabiskan sebagian besar waktu batch-1 untuk meluncurkan kernel, sehingga menyatukan
peluncuran tersebut meningkatkan throughput, dan output replay identik hingga tingkat bit
dengan eksekusi eager.

`cuda_graph` menerima tiga nilai. `False` adalah default dan tidak melakukan apa pun. `True`
melakukan capture saat penggunaan pertama untuk setiap bentuk input. `"auto"` menunggu sampai
bentuk berulang sebelum melakukan capture, sehingga pekerjaan sekali jalan dan bentuk yang
bervariasi tidak pernah membayar biayanya.

`capture_graph(imgsz=None, batch=1, dtype=None)` memindahkan biaya tersebut dari permintaan
pertama. Graph hanya valid untuk bentuk tepat yang di-capture, sehingga `batch` di sini harus
sesuai dengan cara `predict` dipanggil kemudian.

`graph_info()` melaporkan graph yang di-capture, jumlah replay, dan alasan apa pun proses
kembali ke eager. `release_graphs()` membebaskan graph serta buffer statisnya.

Capture memerlukan CUDA dan family yang ikut serta melalui `SUPPORTS_CUDA_GRAPH`, karena
proses ini memerlukan forward tanpa pekerjaan yang terlihat host dan diverifikasi per family.
Memintanya pada family yang belum ikut serta akan memunculkan `NotImplementedError`, bukan
diam-diam berjalan secara eager.

Graph merekam alamat memori, bukan nilai, sehingga tindakan apa pun yang memindahkan parameter
akan menghapusnya. Mengubah perangkat melalui `predict(device=...)`, melakukan kuantisasi,
dan dekuantisasi semuanya membatalkan graph yang sudah di-capture.

Matriks dukungan lengkap per family, pemisahan seam, dan kontrak numerik dijelaskan pada
[CUDA graph](/docs/reference/cuda-graphs).

## Presisi

<code-tabs name="precision" />

`half=True` saat prediksi tidak melakukan apa pun. Argumen ini diterima demi kompatibilitas
command line, memunculkan peringatan bahwa argumen tersebut no-op, lalu dibuang sebelum
mencapai family mana pun. Flag `--half` CLI mencetak peringatan yang sama untuk model `.pt`.

Ada dua jalur nyata menuju presisi lebih rendah.

Untuk artefak hasil ekspor, presisi dipilih saat ekspor dengan
`export(format=..., half=True)`, lalu berkas hasilnya dimuat kembali melalui `LibreYOLO()`
tanpa perubahan.

Untuk eksekusi PyTorch, `model.quantize(recipe="fp16")` melakukan cast model ke float16 dan
memasang hook yang mempertahankan float32 pada input serta output model. `"bf16"` melakukan
hal yang sama dengan bfloat16. Keduanya tidak membaca data kalibrasi, sehingga `calib`
diabaikan. Kuantisasi saat ini mencakup empat family: YOLOv9, RF-DETR, BiRefNet, dan FeyNobg.
Cast pada perangkat CPU mencatat peringatan bahwa proses akan lambat, sehingga resep ini
ditujukan untuk GPU.

Kedua jalur mengubah numerik. Tidak ada yang menjamin deteksi tetap sama, jadi lakukan
validasi sebelum deployment.

## Inferensi berbasis tile

<code-tabs name="tiling" />

Tiling memotong gambar besar menjadi tile persegi yang tumpang tindih, memprediksi masing-masing,
dan menggabungkan hasilnya. Ini adalah opsi untuk objek kecil dalam gambar beresolusi tinggi,
ketika pengubahan ukuran seluruh gambar menyusutkan target hingga di bawah kemampuan model.

Ukuran tile adalah ukuran input model, atau `imgsz` jika diberikan, dan harus persegi.
`overlap_ratio` memakai default `0.2`. Tile yang tumpang tindih direkonsiliasi dengan
non-maximum suppression per kelas pada ambang `iou`, lalu daftar gabungan dipangkas ke
`max_det`. Karena itu, `iou` memengaruhi prediksi berbasis tile bahkan pada family yang
tidak menjalankan NMS sendiri.

Tiling dilewati sepenuhnya, bukan hanya murah, jika gambar sudah muat: bila kedua dimensi
berada pada atau di bawah ukuran input, hanya satu forward biasa yang berjalan. Tiling juga
dilewati untuk klasifikasi, segmentasi semantik, dan task `embed`, yang kembali ke satu tahap
karena tiling tidak bermakna di sana.

Tiling memunculkan error untuk task yang payload-nya tidak dapat disambung kembali: mask
segmentasi instance, kotak berorientasi, titik, kedalaman, tepi, dan normal. Tiling tidak dapat
digabungkan dengan `augment`.

Hasil memiliki `result.tiled` dan `result.num_tiles`. Dengan `save=True`, proses berbasis
tile menulis direktori di bawah `runs/tiled_detections` yang memuat setiap tile, gambar
beranotasi, visualisasi grid, dan `metadata.json` yang mencatat ukuran tile, tumpang tindih,
serta ambang batas. `result.tiles_path` dan `result.grid_path` menunjuk ke hasil tersebut.

## Augmentasi saat pengujian

<code-tabs name="tta" />

`augment=True` menjalankan gambar lebih dari satu kali dan menggabungkan deteksi dengan
non-maximum suppression per kelas pada ambang `iou`. Seperti tiling, ini membuat `iou`
berpengaruh pada family yang biasanya mengabaikannya.

Dalam praktiknya, ini adalah pembalikan horizontal. Daftar skala `TTA_SCALES` memakai satu
skala `1.0` secara default dan tidak ada family yang disediakan mengubahnya, sehingga setiap
family menjalankan dua tahap: gambar asli dan cerminnya. Family yang ditandai
`TTA_FIXED_SIZE` mengubah ukuran ke persegi tetap, yang membuat multiskala menjadi no-op.

Segmentasi semantik dan panoptik memakai penggabungan berbeda. Tampilan yang dibalik akan
dikembalikan orientasinya dan kedua distribusi softmax dirata-ratakan sebelum argmax, bukan
digabungkan sebagai kotak.

Augmentasi saat pengujian tidak tersedia untuk setiap task. Fitur ini memunculkan error untuk
kotak berorientasi, pose, titik, kedalaman, normal, tepi, restorasi, OCR, dan model embedding,
serta tidak dapat digabungkan dengan tiling.

Family berikut menonaktifkannya sepenuhnya, sehingga `augment=True` menjalankan satu tahap
biasa: BiRefNet, CenterNet, CLIP, DexiNed, FOMO, HRNet, L2CS-Net, LibreMODUS, NAFNet,
PP-OCRv5, Real-ESRGAN, RetinaNet, SAM 3D Body, SigLIP2, SwinIR, TEED, setiap varian SAM,
setiap detektor open-vocabulary, dan setiap vision language model.

## Pengukuran

Halaman ini tidak mencantumkan angka latensi karena nilai milidetik tanpa perangkat keras,
runtime, presisi, dan ukuran batch bukanlah fakta. Angka hasil pengukuran lintas perangkat
keras dan runtime dipublikasikan di [visionanalysis.org](https://www.visionanalysis.org),
sedangkan `libreyolo profile` mengukur model tertentu pada mesin yang sedang digunakan.
