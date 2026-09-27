---
title: Paddle
seo_title: Ekspor ke PaddlePaddle dari LibreYOLO
description: >-
  Ubah detektor LibreYOLO menjadi model inferensi PaddlePaddle lewat X2Paddle:
  toolchain yang dipatok, graf FP32 statis batch 1, dan inferensi CPU.
lead: >-
  Model inferensi PaddlePaddle berupa graf model.pdmodel di samping berkas bobot
  model.pdiparams. LibreYOLO mengekspor graf ONNX statis opset 15,
  mengonversinya dengan X2Paddle, lalu mengemas hasilnya bersama metadata.yaml
  sehingga dimuat lewat factory yang sama seperti semua runtime lain.
keywords:
  - export yolo ke paddle
  - paddlepaddle inference
  - x2paddle
  - model.pdmodel
  - model.pdiparams
  - onnx opset 15
last_verified: 1.5.0
meta:
  - label: Flag
    value: export(format="paddle")
    mono: true
  - label: Menulis
    value: 'Sebuah direktori berisi model.pdmodel, model.pdiparams dan metadata.yaml'
  - label: Ekstra
    value: 'pip install "libreyolo[paddle]"'
    mono: true
  - label: Dimuat kembali
    value: 'LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")'
    mono: true
  - label: Backend
    value: libreyolo.backends.paddle.PaddleBackend
    mono: true
  - label: Shape
    value: 'Statis, batch 1, opset 15. Ketiganya diwajibkan.'
  - label: Presisi
    value: 'Hanya FP32, hanya CPU.'
  - label: Toolchain
    value: >-
      PaddlePaddle 2.6.2, X2Paddle 1.6.0, ONNX 1.17 atau lebih lama, diperiksa
      persis
verification: >-
  Dibaca dari libreyolo/export/paddle.py, libreyolo/export/exporter.py,
  libreyolo/export/support.py, libreyolo/backends/paddle.py, docs/paddle.md dan
  pyproject.toml di branch dev.
snippets:
  install:
    - label: Instalasi
      language: bash
      code: >
        # Python 3.10 sampai 3.12. WSL2 dengan Ubuntu 22.04 adalah jalur Windows
        yang tervalidasi.

        pip install "libreyolo[paddle]"
    - label: Memastikan versi yang dipatok
      language: bash
      code: >
        python -c "from importlib.metadata import version;
        print(version('paddlepaddle'), version('x2paddle'), version('onnx'))"
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9t.pt")

        # Menulis direktori weights/LibreYOLO9t_paddle
        path = model.export(format="paddle")
        print(path)
    - label: CLI
      language: bash
      code: |
        libreyolo export --model LibreYOLO9t.pt --format paddle
    - label: Argumen
      language: python
      code: |
        model.export(
            format="paddle",
            imgsz=640,        # int; kanvas persegi milik family ini
            batch=1,          # nilai lain memicu ValueError
            dynamic=False,    # True memicu ValueError
            simplify=True,    # False memicu ValueError
            opset=15,         # nilai lain memicu ValueError
            output_path=None, # None menulis weights/<stem>_paddle
        )
  run:
    - label: Lewat LibreYOLO
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("weights/LibreYOLO9t_paddle", device="cpu")
        result = model.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: CLI
      language: bash
      code: |
        libreyolo predict --model weights/LibreYOLO9t_paddle \
          --source https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg --device cpu --save
    - label: Langsung lewat backend
      language: python
      code: |
        from libreyolo.backends.paddle import PaddleBackend

        # Yang dibangun LibreYOLO() untuk direktori Paddle. Objek Results
        # yang sama, tanpa routing factory di antaranya.
        backend = PaddleBackend("weights/LibreYOLO9t_paddle", device="cpu")
        result = backend.predict("parkour.jpg")
        print(result.boxes.xyxy[:3])
    - label: Paddle murni
      language: python
      code: >
        import numpy as np

        import paddle.inference as paddle_infer

        import yaml


        directory = "weights/LibreYOLO9t_paddle"

        config = paddle_infer.Config(
            f"{directory}/model.pdmodel", f"{directory}/model.pdiparams"
        )

        config.disable_gpu()

        config.disable_mkldnn()

        config.switch_ir_optim(False)


        predictor = paddle_infer.create_predictor(config)

        handle = predictor.get_input_handle(predictor.get_input_names()[0])

        handle.reshape([1, 3, 640, 640])

        handle.copy_from_cpu(np.zeros((1, 3, 640, 640), dtype=np.float32))

        predictor.run()

        for name in predictor.get_output_names():
            print(name, predictor.get_output_handle(name).copy_to_cpu().shape)

        meta = yaml.safe_load(open(f"{directory}/metadata.yaml"))

        print(meta["model_family"], meta["task"], meta["names"])


        # Preprocessing dan postprocessing jadi tanggung jawab Anda di jalur
        ini.
  support:
    - label: Memeriksa satu family dan task sebelum mengekspor
      language: bash
      code: |
        libreyolo formats --family yolo9 --task detect
source_hash: cdd8bf12286e2f53
---

## Instalasi

<code-tabs name="install" />

Ekstra ini mematok stack persis seperti yang diukur pada kerja parity:
PaddlePaddle 2.6.2, X2Paddle 1.6.0 dan ONNX 1.17 atau lebih lama. Patokan itu
diperiksa saat ekspor, bukan hanya saat pemasangan, dan versi yang berbeda
memicu `ImportError` yang menyebutkan versi yang diharapkan. Rilis Paddle yang
lebih baru menolak sebagian kode statis yang dihasilkan X2Paddle 1.6.0, jadi
gagal sejak awal lebih baik daripada menghasilkan artefak yang belum
divalidasi siapa pun.

## Ekspor

<code-tabs name="export" />

Empat argumen bersifat tetap, bukan sekadar bernilai bawaan. `dynamic` harus
`False`, `batch` harus 1, `simplify` harus `True` agar graf konversi sepenuhnya
statis, dan `opset` harus 15, yaitu batas tertinggi yang diterima X2Paddle
1.6.0. Memberikan nilai lain akan memicu error sebelum tracing.

Satu normalisasi dijalankan pada graf perantara. ONNX mendefinisikan dilasi
MaxPool yang dihilangkan sebagai satu, PyTorch menuliskan atribut semua-satu
secara eksplisit, dan X2Paddle 1.6.0 menolaknya, jadi exporter menghapus nilai
bawaan yang berlebihan itu dan membiarkan operasi yang ditentukan tetap sama.

Artefaknya berupa sebuah direktori: `model.pdmodel`, `model.pdiparams` dan
`metadata.yaml`. Kode Python yang dihasilkan X2Paddle selama konversi bukan
bagian darinya.

## Menjalankan artefak

<code-tabs name="run" />

`LibreYOLO()` mengenali direktori mana pun yang memuat `model.pdmodel` sekaligus
`model.pdiparams`, membaca `metadata.yaml`, dan mengembalikan objek `Results`
yang sama seperti checkpoint. Device selain `auto` atau `cpu` akan memicu error:
backend ini hanya untuk CPU.

Yang dibangun factory adalah `PaddleBackend`, diekspor dari `libreyolo` dan bisa
diimpor sebagai `libreyolo.backends.paddle.PaddleBackend`. Bangun sendiri
objeknya bila ingin memakai backend tanpa routing sufiks milik factory,
misalnya untuk memberikan `task=` secara eksplisit pada direktori yang
`metadata.yaml`-nya bukan Anda yang menulis. Method `predict()` miliknya
menerima sumber yang sama dan mengembalikan hasil yang sama.

Snippet runtime murni mencerminkan apa yang dikonfigurasi backend, dan tiga
opsi yang dimatikan itu disengaja. Pipeline fusion CPU pada Paddle 2.6 bisa
crash saat mengoptimalkan graf gather dan scatter berukuran besar yang
dihasilkan untuk deformable attention, jadi graf statis tanpa fusion yang
portabel itulah yang dipakai sebagai acuan pengukuran parity. Preprocessing,
decoding, NMS dan penskalaan ulang koordinat menjadi tanggung jawab Anda di
jalur itu.

## Batasan

Tidak ada shape dinamis, tidak ada FP16, tidak ada INT8, tidak ada NMS
tertanam, tidak ada runtime GPU.

Kombinasi yang tervalidasi adalah deteksi YOLO9, deteksi YOLO9-E2E dan YOLO9-P2,
deteksi, pose dan segmentasi EC, deteksi RT-DETRv4, D-FINE, DEIM dan DEIMv2,
serta deteksi dan pose YOLO-NAS. Masing-masing dicakup oleh konversi, pemuatan
ulang di runtime CPU, parity keluaran mentah dan hasil publik yang cocok.

Diblokir, dengan alasan yang dicatat per kombinasi:

| Kombinasi | Alasan |
|---|---|
| RF-DETR, semua task | Butuh ONNX opset 17 dan GridSample; X2Paddle 1.6.0 menerima opset 15 atau lebih rendah dan tidak punya mapper GridSample |
| Deteksi RT-DETR dan RT-DETRv2 | Graf hasil pelatihannya butuh GridSample pada opset 16 atau lebih baru |
| Segmentasi D-FINE | Berhasil dikonversi dan dimuat ulang, tetapi error RMS relatif mask-logit 3.52% dan IoU mask tercocok minimum 0.582 |
| Segmentasi YOLO9 | YOLO9 hanya untuk deteksi di LibreYOLO |
| Segmentasi RTMDet-Ins | Decode mask dengan kernel dinamis tidak punya kontrak runtime hasil ekspor |

Apa pun yang tidak terdaftar sebagai tervalidasi atau diblokir akan ditolak
dengan catatan bahwa kombinasi itu belum divalidasi melalui jalur konversi ONNX
ke Paddle.

Untuk grid family dan task lengkap, lihat
[matriks ekspor](/docs/reference/export-matrix). Untuk satu kombinasi:

<code-tabs name="support" />
