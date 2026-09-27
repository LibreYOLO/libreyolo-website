---
title: Instalasi ringan
seo_title: Menjalankan inferensi ONNX LibreYOLO tanpa PyTorch
description: >-
  Instal LibreYOLO dengan --no-deps dan jalankan deteksi ONNX hanya dengan
  numpy, tanpa torch di disk. Teknik, batasan, dan list paket persisnya.
lead: >-
  Jalur inferensi ONNX LibreYOLO menggunakan numpy dari awal hingga akhir,
  termasuk decode dan NMS. Tidak ada bagian runtime yang memerlukan PyTorch,
  sehingga instalasi yang melewati resolve dependency dapat menjalankan deteksi
  tanpa torch pada mesin.
keywords:
  - inferensi tanpa torch
  - libreyolo tanpa pytorch
  - onnx inference tanpa torch
  - instalasi ringan libreyolo
  - pip install no-deps
  - hemat ruang libreyolo
  - onnxruntime inference python
last_verified: 1.5.0
meta:
  - label: Berlaku untuk
    value: 'Deteksi ONNX, tujuh family model'
  - label: Entry point
    value: libreyolo.backends.onnx.OnnxBackend
    mono: true
  - label: Tingkat dukungan
    value: 'Upaya terbaik, bukan distribusi terpisah'
snippets:
  install:
    - label: Ringan
      language: bash
      code: |
        # Instal paket tanpa list dependency-nya, lalu sediakan
        # empat paket yang benar-benar diimpor jalur deteksi ONNX.
        pip install --no-deps libreyolo
        pip install numpy pillow opencv-python-headless onnxruntime
    - label: Torch khusus CPU
      language: bash
      code: |
        # Coba ini terlebih dahulu. Semua fitur tetap tersedia tanpa wheel CUDA,
        # yang menggunakan sebagian besar ruang disk.
        pip install libreyolo --index-url https://download.pytorch.org/whl/cpu
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo.backends.onnx import OnnxBackend

        model = OnnxBackend("libreyolo9t.onnx")
        result = model.predict("https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg")

        # xyxy di sini adalah ndarray numpy, bukan tensor torch.
        print(result.boxes.xyxy)
        print(result.boxes.conf)
        print(result.boxes.cls)
source_hash: e60e83d32d13026e
---

## Mengapa cara ini berfungsi

`pip install --no-deps libreyolo` menginstal paket dan melewati seluruh list
dependency. Tidak ada yang diselesaikan secara otomatis, dan pengguna bertanggung
jawab menginstal yang benar-benar digunakan.

Cara ini hanya berguna jika jalur kode yang diinginkan benar-benar tidak
memerlukan dependency yang dilewati, dan deteksi ONNX memang tidak memerlukannya.
Decode, termasuk non-maximum suppression, menggunakan numpy. Resep preprocessing
menggunakan numpy. PyTorch adalah dependency pelatihan dan inferensi eager, dan
tidak pernah dipanggil pada jalur ini.

Sebelum rilis ini, import tetap gagal: mengimpor apa pun di bawah
`libreyolo.models` membangun setiap kelas model untuk mengisi registry deteksi
otomatis checkpoint, dan kelas tersebut merupakan subclass `torch.nn.Module`.
Resep preprocessing kini berada dalam paket sendiri, `libreyolo.preprocess`,
dan import torch ditunda hingga atribut torch disentuh, sehingga jalur ONNX
dapat diimpor tanpa torch pada mesin. Paket tersebut memuat preprocessor
native numpy per family: `yolo9`, `yolonas`, `yolox`, `ec`, `rtdetr`, `rfdetr`,
`dfine`, `deim`, dan `deimv2`, dua lebih banyak daripada tujuh family yang
diverifikasi end-to-end di bawah. Setiap
`libreyolo/models/<family>/utils.py` mengekspor ulang darinya, sehingga path
import lama tetap berfungsi.

## Coba wheel khusus CPU terlebih dahulu

Sebagian besar pengguna yang memerlukan cara ini ingin menghindari instalasi
beberapa gigabyte, dan ukurannya terkonsentrasi di satu tempat: wheel `torch`
default menyertakan CUDA. Build khusus CPU jauh lebih kecil dan tidak memerlukan
jalur instalasi khusus.

<code-tabs name="install" />

Opsi khusus CPU mempertahankan setiap fitur LibreYOLO: pelatihan, validasi,
setiap task, setiap family, dan CLI. Gunakan jalur ringan jika benar-benar
memerlukan mesin tanpa torch, bukan sekadar instalasi yang lebih kecil.

## Cakupan instalasi ringan

| | |
|---|---|
| Task | Deteksi |
| Format | ONNX |
| Entry point | `OnnxBackend` |
| Antarmuka | Library Python |

Tujuh family telah diverifikasi pada jalur ini: [YOLOv9](/docs/models/yolov9),
[YOLO-NAS](/docs/models/yolo-nas), [EdgeCrafter](/docs/models/edgecrafter),
[RT-DETR](/docs/models/rt-detr), [RF-DETR](/docs/models/rf-detr),
[D-FINE](/docs/models/d-fine), dan [DEIM](/docs/models/deim), termasuk varian
setiap family.

Itulah cakupan terverifikasi, bukan batas yang diterapkan library. Task dan
family lain hanya berada di luar pemeriksaan: beberapa akan menarik torch saat
dipanggil dan beberapa mungkin berfungsi. Perlakukan semua di luar list sebagai
belum diuji, bukan didukung atau rusak.

Dalam cakupan tersebut, hasil identik dengan instalasi normal, bukan sekadar
mendekati. Setiap family diekspor ke ONNX dan dijalankan dua kali, sekali secara
normal dan sekali dengan torch diblokir; bounding box, skor, dan kelas cocok persis.
Pengujian paritas dalam suite menjaga kontrak agar tidak menyimpang.

## Lima hal yang sering menjebak pengguna

**Gunakan `OnnxBackend`, bukan kelas model.** `LibreYOLO9("model.onnx")` tetap
memerlukan torch karena `LibreYOLO9` sendiri adalah subclass `nn.Module`. Ini
kesalahan paling mungkin karena setiap halaman lain memuat model melalui kelas
atau `LibreYOLO()`.

**Lakukan ekspor di tempat lain.** Pembuatan berkas `.onnx` memerlukan torch,
sehingga mesin ringan tidak dapat membuatnya. Ekspor pada mesin development atau
CI lalu kirim artefak ke target ramping.

**Results membawa array numpy.** `result.boxes.xyxy` adalah `ndarray` di sini.
Container menerima kedua jenis sehingga nama atribut tidak berubah, tetapi kode
yang memanggil `.cpu()` atau `.numpy()` pada hasil akan gagal.

**Satu gambar mengembalikan satu `Results`.** `predict()` mengembalikan satu
`Results` untuk satu gambar dan list untuk beberapa gambar. Melakukan indexing
pada satu hasil dengan `[0]` memilih deteksi pertama, bukan gambar pertama, yang
diam-diam menghasilkan hasil satu bounding box alih-alih error.

**CLI tidak akan berfungsi.** `typer` dan `click` tidak termasuk empat paket,
sehingga perintah `libreyolo` tidak tersedia. Ini adalah instalasi library.

## Prediksi

<code-tabs name="predict" />

Ganti `onnxruntime` dengan `onnxruntime-gpu` untuk berjalan pada CUDA. Empat
paket tersebut adalah paket yang benar-benar diimpor oleh `predict()` tanpa
torch penuh, dicatat selama pemanggilan, bukan disimpulkan. Paket
`opencv-python-headless` menggantikan `opencv-python` yang dideklarasikan:
modulnya sama, tanpa library GUI, dan lebih kecil di disk.

Di antara dependency lain yang dideklarasikan, `requests` hanya diperlukan untuk
memuat gambar dari URL, `pycocotools` dan `scipy` untuk validasi serta evaluasi,
sedangkan `typer` dan `click` untuk CLI.

## List ini sengaja dapat berubah

List paket di atas tepat untuk rilis yang disebutkan di bagian atas halaman.
`--no-deps` membuat pengguna keluar dari resolve dependency, sehingga tidak ada
yang memeriksanya secara otomatis, dan rilis berikutnya dapat mengimpor hal
yang tidak tercantum.

Jika muncul `ModuleNotFoundError`, tekniknya sudah jelas: instal paket yang
hilang. Itulah model maintenance yang dimaksud, bukan bug report. Jalur ini
bersifat upaya terbaik dan bukan distribusi yang didukung secara terpisah. Karena
itu, tidak ada paket ringan kedua pada PyPI dan tidak ada rencana membuatnya.

Untuk memastikan environment benar-benar tanpa torch, bukan diam-diam kembali
ke salinan terinstal, gunakan assertion:

```python
import importlib.util

assert importlib.util.find_spec("torch") is None, "torch is installed"
```

Pemeriksaan ini layak dipertahankan dalam CI untuk gambar ramping. Tanpanya,
environment yang kebetulan memiliki torch akan lolos setiap pengujian tanpa
memberikan informasi apa pun.


