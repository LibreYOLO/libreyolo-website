---
title: Hailo
seo_title: Menjalankan model LibreYOLO di akselerator Hailo
description: >-
  Menerapkan model LibreYOLO ke Hailo-8 atau Hailo-8L: ekspor ONNX statis, tahap
  Dataflow Compiler yang dijalankan sendiri, dan arsitektur mana saja yang bisa
  dikompilasi.
lead: >-
  Akselerator Hailo dikompilasi dengan Hailo Dataflow Compiler, sebuah SDK
  proprietary yang didistribusikan lewat Developer Zone milik Hailo. Bagian
  LibreYOLO dalam alur ini hanyalah ekspor ONNX statis biasa; parsing,
  kuantisasi dan kompilasi menjadi HEF terjadi di dalam DFC setelahnya.
keywords:
  - libreyolo hailo
  - hailo-8
  - hailo-8l
  - raspberry pi ai kit
  - ai hat+
  - hailo dataflow compiler
  - cara compile hef hailo
  - hailortcli
last_verified: 1.5.0
meta:
  - label: Langkah LibreYOLO
    value: 'export(format="onnx", imgsz=640, dynamic=False)'
    mono: true
  - label: Bukan sebuah format
    value: Tidak ada format="hef". DFC tidak bisa menjadi dependensi pip.
  - label: Extra
    value: 'pip install "libreyolo[onnx]"'
    mono: true
  - label: Host kompilasi
    value: >-
      Linux x86_64, termasuk WSL2 Ubuntu 22.04. Kompilasi tidak bisa berjalan di
      ARM.
  - label: Yang bisa dikompilasi
    value: >-
      Graph CNN murni dengan bentuk tetap. Attention, bentuk dinamis dan desain
      yang didominasi LayerNorm tidak bisa.
  - label: Status
    value: >-
      Belum ada family LibreYOLO yang dibawa dari awal sampai akhir melalui DFC
      hingga menjadi HEF yang berjalan.
verification: >-
  Dibaca dari skills/libreyolo-export-hailo/SKILL.md, libreyolo/export/onnx.py
  dan libreyolo/cli/commands/export.py di branch dev. Batasan DFC adalah batasan
  yang tercatat di skill tersebut; belum ada HEF LibreYOLO yang dikompilasi dan
  diukur.
snippets:
  install:
    - label: Sisi LibreYOLO
      language: bash
      code: |
        pip install "libreyolo[onnx]"
    - label: 'Sisi Hailo, dipasang sendiri'
      language: text
      code: >
        Prerequisites, none of them installable from PyPI:


        - A Linux x86_64 machine. WSL2 Ubuntu 22.04 works. The Raspberry Pi is a
          runtime target, never the compile host.
        - The Dataflow Compiler wheel (hailo_sdk_client) from the Hailo
        Developer
          Zone, which is free to register for.
        - For Hailo-8 and Hailo-8L, the Hailo Model Zoo v2.x line, for its
          recipes and NMS configurations.
        - A GPU on the compile host is strongly recommended: the quantization
          step takes hours without one.
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Hailo butuh batch 1, resolusi tetap dan tanpa sumbu dinamis.
        # API Python memakai dynamic=True secara bawaan, jadi matikan eksplisit.
        model = LibreYOLO("LibreYOLOXs.pt")
        model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
    - label: CLI
      language: bash
      code: |
        # CLI sudah memakai bentuk statis secara bawaan.
        libreyolo export --model LibreYOLOXs.pt --format onnx --imgsz 640
    - label: Memastikan graph statis sebelum dikompilasi
      language: python
      code: |
        import onnx

        graph = onnx.load("weights/LibreYOLOXs.onnx").graph
        shape = graph.input[0].type.tensor_type.shape
        print([d.dim_value or d.dim_param for d in shape.dim])
  compile:
    - label: 'Parsing, kuantisasi dan kompilasi'
      language: python
      code: >
        from pathlib import Path


        import numpy as np

        from hailo_sdk_client import ClientRunner

        from PIL import Image


        ONNX = "weights/LibreYOLOXs.onnx"

        HW_ARCH = "hailo8"     # hailo8 | hailo8l | hailo10h

        IMGSZ = 640


        runner = ClientRunner(hw_arch=HW_ARCH)


        # Untuk YOLOX, jalankan translate sekali tanpa end_node_names: log DFC

        # mencetak end node yang disarankan. Jalankan ulang dengan nama
        tersebut.

        runner.translate_onnx_model(ONNX)


        # Normalisasi harus cocok dengan preprocessing LibreYOLO. YOLOX dan
        YOLO9

        # tidak butuh mean atau standar deviasi, hanya skala 0-255 ke 0-1.

        script = "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0,
        255.0])\n"


        # Opsional: biarkan Hailo yang menangani NMS. Konfigurasinya spesifik

        # untuk jumlah kelas sekaligus ukuran masukan, jadi config COCO-80 salah

        # untuk model tiga kelas hasil fine-tuning. Tanpa baris ini, HEF

        # mengeluarkan tensor head mentah dan aplikasi yang melakukan decode.

        # script += 'nms_postprocess("yolox_nms_config.json", meta_arch=yolox,
        engine=cpu)\n'


        runner.load_model_script(script)


        # Gambar kalibrasi harus mewakili data deployment.

        # Gambar acak tetap terkompilasi dan diam-diam merusak akurasi.

        calib_paths = sorted(Path("calib_images").glob("*.jpg"))[:128]

        calib = np.stack([
            np.asarray(
                Image.open(p).convert("RGB").resize((IMGSZ, IMGSZ)),
                dtype=np.float32,
            )
            for p in calib_paths
        ])


        runner.optimize(calib)

        Path("libreyoloxs.hef").write_bytes(runner.compile())
    - label: End node YOLO9
      language: python
      code: >
        # Graph LibreYOLO memakai prefix "/head/...", bukan prefix "model.N"
        yang

        # muncul di konfigurasi yang ditulis untuk hasil ekspor lain. Config

        # salinan tidak akan cocok. Periksa namanya di graph sendiri bila
        parsing

        # gagal.

        END_NODES = [
            "/head/cv2.0/cv2.0.2/Conv", "/head/cv3.0/cv3.0.2/Conv",
            "/head/cv2.1/cv2.1.2/Conv", "/head/cv3.1/cv3.1.2/Conv",
            "/head/cv2.2/cv2.2.2/Conv", "/head/cv3.2/cv3.2.2/Conv",
        ]

        runner.translate_onnx_model(ONNX, end_node_names=END_NODES)
  device:
    - label: Raspberry Pi 5 dengan AI Kit atau AI HAT+
      language: bash
      code: >
        sudo apt install dkms hailo-all

        hailortcli fw-control identify       # cek perangkat, sekaligus menyebut
        arsitekturnya

        hailortcli run libreyoloxs.hef       # smoke test dan throughput
source_hash: 33b077f1c23d5535
---

## Instalasi

Tidak ada `format="hef"` di LibreYOLO dan tidak akan pernah ada. Hailo Dataflow
Compiler adalah SDK proprietary yang didistribusikan sebagai wheel privat di
balik pendaftaran Developer Zone, jadi ia tidak bisa menjadi dependensi maupun
extra. Deployment berlangsung dalam dua tahap: LibreYOLO menulis berkas ONNX
statis, lalu DFC dijalankan di atasnya.

```text
Libre<Model>.pt  ->  ONNX  ->  HAR (parse)  ->  HAR (quantize INT8)  ->  HEF
                 [libreyolo]           [Hailo DFC, installed by you]
```

<code-tabs name="install" />

## Ekspor

<code-tabs name="export" />

Jangan berikan `half=True`. DFC menerima ONNX FP32 dan melakukan kuantisasi INT8
sendiri. Jangan berikan `nms=True` juga: NMS ditangani Hailo lewat
`nms_postprocess` atau ditangani aplikasi, dan subgraph NMS hanya jadi beban mati
di luar end node. Opset bawaan sudah bekerja; bila parser DFC menolaknya, lakukan
ekspor ulang dengan `opset=11`.

DFC memotong graph pada end node yang diberikan, yaitu konvolusi detection head,
lalu membuang semua yang ada di hilirnya. Karena itu ONNX LibreYOLO biasa yang
sudah menyertakan decode tetap merupakan masukan yang bisa diterima: bagian ekor
decode cukup diabaikan oleh parser.

## Kompilasi

<code-tabs name="compile" />

Pilih `hw_arch` sesuai target: `hailo8` untuk Hailo-8, AI HAT+ 26 TOPS serta
modul M.2 dan PCIe; `hailo8l` untuk Hailo-8L, Raspberry Pi AI Kit dan AI HAT+ 13
TOPS; `hailo10h` untuk Hailo-10H, yang butuh DFC dan Model Zoo versi lebih baru
yang cocok. Bila ragu, `hailortcli fw-control identify` di perangkat tersebut
menjawab pertanyaannya.

Dua family cocok dengan meta-arsitektur NMS HailoRT, sehingga Hailo bisa
menangani suppression di dalam pipeline hasil kompilasi: YOLOX lewat
`meta_arch=yolox`, dan YOLO9 lewat meta-arsitektur decoupled head milik Hailo,
yang tata letak head-nya identik. Ambil konfigurasi `nms_postprocess` yang sesuai
dari Hailo Model Zoo lalu sesuaikan dengan jumlah kelas dan ukuran masukan Anda.
Setiap detektor konvolusional lain dikompilasi sebagai graph tanpa
meta-arsitektur yang cocok: HEF mengeluarkan tensor head mentah dan aplikasi
menjalankan decode dan NMS di CPU.

Simpan log kompilasi bila ada yang gagal. Setiap perbaikan bergantung pada nama
lapisan atau operator yang persis gagal.

## Menjalankan artefaknya

<code-tabs name="device" />

Inferensi di aplikasi memakai API Python `hailo_platform`. Bila
`nms_postprocess` ikut dikompilasi, keluarannya berbentuk
`(batch, num_classes, max_dets, 5)` yang membawa `[y1, x1, y2, x2, score]` dalam
koordinat model, dan penskalaan kembali ke gambar sumber dilakukan sendiri.
Pipeline `Results` milik LibreYOLO tidak terlibat saat runtime; HEF adalah
artefak yang berdiri sendiri, dan preprocessing serta postprocessing menjadi
urusan aplikasi.

## Batasan

Apakah sebuah model bisa menargetkan Hailo-8 atau Hailo-8L adalah sifat
arsitekturnya, bukan namanya, jadi aturan di bawah ini juga berlaku untuk family
yang ditambahkan setelah halaman ini ditulis.

Sebuah model tidak akan bisa dikompilasi bila mengandung salah satu dari ini:

- Attention dalam bentuk apa pun, baik self, cross, deformable maupun windowed.
  Itu menyingkirkan setiap detektor bergaya DETR, setiap detektor
  open-vocabulary atau yang dikondisikan teks, setiap backbone ViT, dan setiap
  tower bahasa atau vision-language. Zoo milik Hailo sendiri menyediakan
  beberapa HEF transformer yang disetel manual; itu pekerjaan khusus dari vendor
  dan bukan bukti bahwa graph attention sembarangan bisa dikompilasi.
- Bentuk dinamis atau alur kontrol yang bergantung pada data. DFC mengompilasi
  satu bentuk masukan tetap dan satu graph statis, jadi jumlah query yang
  berubah-ubah, prompt teks, top-k dinamis, `NonZero`, `Gather` atau `TopK`
  dengan indeks dinamis, dan `grid_sample` semuanya tidak bisa.
- Desain yang didominasi LayerNorm atau GELU. BatchNorm melebur ke dalam
  konvolusi dengan rapi; dukungan LayerNorm buruk dan GELU bukan aktivasi
  native, jadi stack bergaya ConvNeXt kurang cocok meski secara nama bersifat
  konvolusional.
- Pekerjaan image-to-image pada resolusi asli. Model restorasi berjalan pada
  resolusi masukan penuh dan melampaui anggaran SRAM Hailo yang praktis.

Sebuah family menjadi kandidat bila ia murni konvolusi, memakai BatchNorm dengan
ReLU atau SiLU, dan punya ukuran masukan tetap. Di library ini artinya detektor
single-stage berbasis CNN, dengan YOLOX dan YOLO9 sebagai target utama; detektor
konvolusional lain seperti PicoDet, YOLO-NAS dan RTMDet, dengan decode di sisi
aplikasi; classifier CNN ResNet, MobileNetV4-conv dan EfficientNetV2, yang di
antaranya ResNet paling baik dukungannya karena Hailo Model Zoo menyediakan
resep untuknya; serta task head konvolusional kecil seperti deteksi titik FOMO
dan gaze L2CS di atas backbone ResNet, yang secara prinsip bisa dikompilasi
tetapi tidak punya resep Hailo.

Satu catatan status, yang menjadi alasan tidak ada satu pun di halaman ini
disebut sudah didukung: belum ada family LibreYOLO yang dibawa dari awal sampai
akhir melalui DFC hingga menjadi HEF yang berjalan. Aturan di atas memprediksi
kemampuan kompilasi dari arsitekturnya. Perilaku parser, kuantisasi dan akurasi
tetap belum terbukti sampai sebuah HEF benar-benar dikompilasi dan diukur, jadi
anggap setiap kandidat butuh buktinya sendiri yang tercatat: HEF hasil kompilasi
dari checkpoint yang persis sama dengan versi DFC, Model Zoo dan HailoRT yang
dicatat, kalibrasi yang terdokumentasi, dan perbandingan akurasi di perangkat
terhadap baseline FP32, bukan sekadar angka throughput.

Bila modelnya tidak memenuhi syarat, alternatifnya adalah runtime dengan paritas
yang tercatat: [ONNX](/docs/export/onnx), [TensorRT](/docs/export/tensorrt) dan
[OpenVINO](/docs/export/openvino).
