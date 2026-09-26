---
title: ncnn
seo_title: Ekspor ke ncnn dari LibreYOLO
description: >-
  Mengekspor model LibreYOLO ke ncnn lewat PNNX: pasangan param dan bin, kanvas
  ekspor yang tetap, penulisan ulang Focus YOLOX, dan family mana saja yang bisa
  dikonversi.
lead: >-
  ncnn adalah library inferensi CPU dari Tencent untuk target mobile. LibreYOLO
  mengonversi lewat PNNX, menulis graph model.ncnn.param berdampingan dengan
  berkas bobot model.ncnn.bin dan sebuah metadata.yaml yang membawa family, task
  serta nama kelas.
keywords:
  - export yolo ke ncnn
  - pnnx
  - model.ncnn.param
  - inference cpu mobile
  - ncnn extractor
  - focus pixel_unshuffle
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="ncnn")
    mono: true
  - label: Menulis
    value: 'Sebuah direktori berisi model.ncnn.param, model.ncnn.bin dan metadata.yaml'
  - label: Tambahan
    value: 'pip install "libreyolo[ncnn]"'
    mono: true
  - label: Dimuat kembali
    value: LibreYOLO("weights/LibreYOLO9t_ncnn")
    mono: true
  - label: Bentuk
    value: Tetap. Metadata mencatat dynamic=False terlepas dari isi flagnya.
  - label: Presisi
    value: Hanya FP32. half=True dan int8=True ditolak.
verification: >-
  Dibaca dari libreyolo/export/ncnn.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/ncnn.py dan pyproject.toml di
  branch dev.
snippets:
  install:
    - label: Instalasi
      language: bash
      code: |
        # pnnx melakukan konversi, ncnn menjalankan hasilnya.
        pip install "libreyolo[ncnn]"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Menulis direktori weights/LibreYOLO9t_ncnn
        path = model.export(format="ncnn", imgsz=640)
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format ncnn --imgsz 640
    - label: Argumen
      language: python
      code: |
        model.export(
            format="ncnn",
            imgsz=640,        # int, atau (height, width)
            batch=1,
            simplify=True,    # hanya berlaku untuk jalur cadangan ONNX
            opset=None,       # auto; hanya berlaku untuk jalur cadangan ONNX
            output_path=None, # None menulis weights/<stem>_ncnn
        )

        # half=True dan int8=True ditolak saat validasi.
  run:
    - label: Lewat LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_ncnn")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Langsung dengan ncnn
      language: python
      code: >
        import ncnn

        import numpy as np

        import yaml


        directory = "weights/LibreYOLO9t_ncnn"

        net = ncnn.Net()

        net.load_param(f"{directory}/model.ncnn.param")

        net.load_model(f"{directory}/model.ncnn.bin")


        # ncnn menerima satu gambar CHW, bukan satu batch.

        mat_in = ncnn.Mat(np.zeros((3, 640, 640), dtype=np.float32))

        extractor = net.create_extractor()

        extractor.input("in0", mat_in)

        ret, mat_out = extractor.extract("out0")

        print(ret, np.array(mat_out).shape)


        meta = yaml.safe_load(open(f"{directory}/metadata.yaml"))

        print(meta["model_family"], meta["task"], meta["names"])


        # Preprocessing dan postprocessing menjadi tanggung jawab Anda di jalur
        ini.
  support:
    - label: Memeriksa satu family dan task sebelum mengekspor
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: 9a849a16a3b32334
---

## Instalasi

<code-tabs name="install" />

Extra ini menarik kedua bagian toolchain: `pnnx` melakukan konversi dan `ncnn`
menjalankan hasilnya. Pada jalur utama, keduanya tidak melewati ONNX.

## Ekspor

<code-tabs name="export" />

Artefaknya berupa direktori. `weights/LibreYOLO9t_ncnn` berisi
`model.ncnn.param`, `model.ncnn.bin` dan `metadata.yaml`; ketiganya adalah satu
artefak dan berpindah bersama.

Konversi lebih dulu mencoba PNNX langsung dari PyTorch. Jika gagal, prosesnya
mengekspor graph ONNX statis ke direktori sementara lalu memanggil command line
tool `pnnx` pada graph itu, dan ekspor baru melempar error ketika kedua jalur
gagal, dengan melaporkan kedua error tersebut. Karena itu, `opset` dan
`simplify` hanya memengaruhi jalur cadangan.

YOLOX butuh satu penulisan ulang agar bisa dikonversi sama sekali. Layer Focus
miliknya memakai strided slicing, yang tidak bisa diturunkan oleh PNNX, sehingga
ekspor menggantinya dengan `pixel_unshuffle` dan mempermutasi kanal input
konvolusi berikutnya untuk mengompensasi urutan kanal yang berbeda. Outputnya
identik secara numerik, dan bobot aslinya dipulihkan setelah ekspor.

## Menjalankan artefak

<code-tabs name="run" />

`LibreYOLO()` mengenali direktori mana pun yang berisi `model.ncnn.param` dan
`model.ncnn.bin`, membaca `metadata.yaml`, lalu mengembalikan objek `Results`
yang sama seperti pada checkpoint.

Snippet kedua adalah jalur runtime langsung, dan ada dua detail yang berbeda
dari semua format lain di sini. ncnn bekerja pada satu gambar CHW, bukan satu
batch, sehingga tidak ada sumbu batch di depan. Nama blob berasal dari berkas
`.param`; PNNX menulis `in0` dan `out0` sesuai konvensi, dan backend membaca
berkas itu alih-alih mengasumsikannya. Preprocessing, decoding, NMS dan
penyesuaian ulang koordinat menjadi tanggung jawab Anda di jalur tersebut.

## Batasan

FP32 pada kanvas tetap. `half=True` dan `int8=True` sama-sama ditolak saat
validasi, dan metadata hasil ekspor mencatat `dynamic=False` apa pun isi
flagnya, sehingga tidak ada backend yang mengasumsikan sumbu yang tidak dimiliki
graph.

Semua family bergaya DETR ditolak pada preflight: `detr`, `deformable_detr`,
`dinodetr`, `dfine`, `lwdetr`, `deim`, `deimv2`, `rtdetr`, `rtdetrv2`,
`rtdetrv4`, `rfdetr` dan `ec`. Pesannya sama untuk semuanya, bahwa model
tersebut membutuhkan operasi decoder atau sampling yang tidak tersedia di ncnn,
dan pesan itu mengarahkan ke ONNX, OpenVINO, TorchScript atau TensorRT sebagai
gantinya.

Yang bisa dikonversi cukup luas di sisi konvolusional: YOLO9 dan YOLO9-E2E,
YOLOX, PicoDet, deteksi dan pose YOLO-NAS, detektor lama YOLO1, YOLO3, YOLO4 dan
YOLO7, empat family klasifikasi CNN, segmentasi semantik PIDNet, deteksi titik
FOMO pada ukuran tetap 96 kali 96, ZipDepth, NAFNet dan Real-ESRGAN.

Entri yang diblokir menyebutkan penyebab kegagalannya secara konkret. Graph
transformer umumnya menyisakan node `pnnx.Expression` yang tidak didukung, yang
menghasilkan jaringan tanpa blob input yang bisa dijalankan, dan itulah yang
menghentikan DINOv2, CLIP, SigLIP2 dan SegFormer. BiRefNet membutuhkan
deformable convolution dari torchvision, yang tidak bisa diturunkan oleh PNNX.
Graph hasil konversi YOLO2 menghentikan runtime ncnn di Windows dengan integer
divide by zero native saat ekstraksi output.

Untuk grid family dan task selengkapnya, lihat
[matriks ekspor](/docs/reference/export-matrix). Untuk satu kombinasi:

<code-tabs name="support" />
