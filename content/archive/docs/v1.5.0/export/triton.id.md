---
title: Triton Inference Server
seo_title: Menyajikan model LibreYOLO di NVIDIA Triton
description: >-
  Menyajikan hasil ekspor ONNX LibreYOLO lewat NVIDIA Triton: layout model
  repository, config.pbtxt yang dihasilkan, dan memprediksi ke sebuah URL model
  HTTP.
lead: >-
  Triton Inference Server menampung sebuah model repository dan menjawab
  permintaan inferensi lewat HTTP. LibreYOLO mengekspor graph ONNX, menghasilkan
  config.pbtxt yang membawa metadata ekspor sebagai satu parameter Triton, dan
  memperlakukan URL model sebagai path model yang bisa dimuat.
keywords:
  - libreyolo triton
  - triton inference server
  - config.pbtxt
  - tritonclient http
  - model repository
  - cara deploy yolo ke triton server
last_verified: 1.5.0
meta:
  - label: Panggilan
    value: 'LibreYOLO("http://127.0.0.1:8000/yolo9")'
    mono: true
  - label: Helper
    value: >-
      create_triton_config(onnx_path, config_path, model_name=...,
      max_batch_size=8)
    mono: true
  - label: Ekstra
    value: 'pip install "libreyolo[onnx,triton]"'
    mono: true
  - label: Protokol
    value: >-
      Hanya inferensi V2 lewat HTTP dan HTTPS. Tidak ada gRPC, autentikasi,
      shared memory, maupun load dan unload model.
  - label: Timeout
    value: Timeout koneksi dan jaringan secara default 30 detik
verification: >-
  Dibaca dari libreyolo/backends/triton.py, libreyolo/models/__init__.py,
  docs/triton.md dan pyproject.toml di branch dev. Perintah container memakai
  versi yang dipatok di docs/triton.md.
snippets:
  install:
    - label: Instalasi
      language: bash
      code: |
        pip install "libreyolo[onnx,triton]"
  repo:
    - label: Mengekspor ke dalam layout repository
      language: python
      code: |
        from pathlib import Path

        from libreyolo import LibreYOLO

        model_dir = Path("triton_repo/yolo9/1")
        model_dir.mkdir(parents=True, exist_ok=True)

        LibreYOLO("LibreYOLO9t.pt").export(
            format="onnx",
            output_path=str(model_dir / "model.onnx"),
            dynamic=True,
            simplify=False,
        )
    - label: Menghasilkan config.pbtxt
      language: python
      code: |
        from libreyolo import create_triton_config

        create_triton_config(
            "triton_repo/yolo9/1/model.onnx",
            "triton_repo/yolo9/config.pbtxt",
            model_name="yolo9",
            max_batch_size=8,
        )
    - label: Layout yang dihasilkan
      language: text
      code: |
        triton_repo/
          yolo9/
            config.pbtxt
            1/
              model.onnx
  serve:
    - label: Memulai server
      language: bash
      code: |
        docker run --rm --name libreyolo-triton \
          -p 8000:8000 -p 8002:8002 \
          -v "$(pwd)/triton_repo:/models:ro" \
          nvcr.io/nvidia/tritonserver:26.04-py3 \
          tritonserver --model-repository=/models --exit-on-error=true
    - label: Menunggu server siap
      language: bash
      code: >
        until curl --fail --silent http://127.0.0.1:8000/v2/health/ready; do
        sleep 1; done
    - label: Menghentikannya
      language: bash
      code: |
        docker stop libreyolo-triton
  run:
    - label: Memprediksi ke model yang disajikan
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        remote = LibreYOLO("http://127.0.0.1:8000/yolo9")
        result = remote.predict(SAMPLE_IMAGE)
        print(result.boxes.xyxy[:3])
    - label: Membandingkan dengan model lokal
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        remote = LibreYOLO("http://127.0.0.1:8000/yolo9").predict(SAMPLE_IMAGE)
        native = LibreYOLO("LibreYOLO9t.pt").predict(SAMPLE_IMAGE)

        print(len(remote.boxes), len(native.boxes))
        print(remote.boxes.xyxy[:3])
        print(native.boxes.xyxy[:3])
    - label: 'Mengunci versi, atau mengubah timeout'
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.backends.triton import TritonBackend

        # Segmen path kedua memilih versi model. Tanpa segmen itu,
        # version policy yang dikonfigurasi di Triton yang menentukan.
        pinned = LibreYOLO("http://127.0.0.1:8000/yolo9/1")

        # Timeout koneksi dan jaringan secara default 30 detik.
        patient = TritonBackend("http://127.0.0.1:8000/yolo9", timeout=120)
source_hash: 0652e4faf0224df3
---

## Instalasi

<code-tabs name="install" />

Ekstra `triton` memasang `tritonclient[http]`. Ekstra gRPC dan shared memory
sengaja dikecualikan: integrasi ini hanya melayani inferensi V2 lewat HTTP dan
HTTPS. `onnx` dibutuhkan karena artefak yang disajikan dan generator config
sama-sama bekerja dari graph ONNX.

## Membangun model repository

Ekspor dengan sumbu batch dinamis, ke dalam layout direktori yang diharapkan
Triton.

<code-tabs name="repo" />

Triton tidak mempertahankan metadata kustom ONNX di dalam respons model-config
miliknya, jadi metadata ekspor yang lengkap harus dikirim lewat jalur lain.
`create_triton_config` menyandikannya sebagai satu parameter string JSON bernama
`libreyolo_metadata` di `config.pbtxt`, menuliskan deklarasi input dan output
sesuai urutan graph, menangani escaping JSON, dan mengunci model ke `KIND_CPU`.

Helper ini memvalidasi sebelum menulis. Ia menuntut tepat satu input pada graph
ONNX, minimal satu output, bentuk tensor yang bisa diresolusi, dan metadata yang
map `names`-nya mendefinisikan setiap indeks kelas dari 0 sampai `nc - 1`. Model
yang gagal di salah satu pemeriksaan tersebut ditolak saat config dibuat, bukan
saat permintaan pertama.

`max_batch_size: 8` cocok dengan ekspor dinamis dan membuat server dapat
memproses hingga delapan gambar dalam satu batch per permintaan. Untuk graph
ONNX dengan batch tetap 1, pakai `max_batch_size=0`; LibreYOLO lalu mengirim
gambar satu per satu.

## Menjalankan server

<code-tabs name="serve" />

Perintah ini mengunci Triton Server 26.04 dan sengaja tidak memakai flag GPU
Docker, karena `KIND_CPU` di config yang dihasilkan memang sudah mencegah
penempatan di GPU.

## Menjalankan artefak

URL model Triton adalah sebuah path model. `LibreYOLO()` memeriksa skema `http`
atau `https` sebelum menangani path lokal apa pun, lalu mengembalikan backend
yang berbicara dengan server, sehingga sisi pemanggilannya identik dengan
checkpoint lokal, begitu pula objek `Results` yang dikembalikan.

<code-tabs name="run" />

Bentuk URL-nya adalah `http(s)://host:port/model` dengan segmen versi yang
opsional. Port harus ditulis eksplisit. Kredensial yang ditanam di URL, query
string dan fragment semuanya ditolak, begitu juga path dengan lebih dari dua
segmen.

`device` diterima lalu diabaikan disertai satu baris log, karena penempatan
adalah keputusan server.

## Batasan

Backend gagal dengan error langsung alih-alih memberi hasil yang menurun ketika
kontraknya tidak terpenuhi: metadata LibreYOLO tidak ada di config model, input
model lebih dari satu, ketidakcocokan antara output yang dikonfigurasi dan
metadata model, tipe data input yang tidak didukung, atau server maupun model
yang belum siap.

Di luar kontrak pada versi ini: gRPC, autentikasi, shared memory, serta memuat
atau melepas model lewat API.

Format apa pun yang didukung Triton sendiri bisa disajikan, tetapi parameter
metadata dan config yang dihasilkan di sini berbentuk ONNX, jadi jalur LibreYOLO
adalah [ONNX](/docs/export/onnx) ke dalam repository. Untuk pipeline video utuh
alih-alih server request-response, lihat
[DeepStream](/docs/export/deepstream).
