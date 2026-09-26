---
title: Sumber prediksi
seo_title: Sumber prediksi di LibreYOLO
description: >-
  Setiap sumber yang diterima predict: gambar, folder, URL, berkas video,
  webcam, RTSP, YouTube, screen capture, daftar gambar, dan berkas .streams.
lead: >-
  Argumen source diklasifikasikan sebelum apa pun dibuka, sehingga satu
  pemanggilan menangani JPEG, folder, MP4, indeks webcam, URL RTSP, wilayah
  layar, atau daftar kamera.
keywords:
  - yolo video inference python
  - rtsp
  - webcam object detection python
  - prediksi folder gambar
  - screen capture object detection
  - multiple rtsp streams
  - streams file
  - youtube inference
  - vid_stride
  - stream=True
last_verified: 1.5.0
verification: >-
  Klasifikasi sumber dibaca dari libreyolo/utils/source.py (classify_source,
  SourceKind, StreamSource, MultiStreamSource). Jenis gambar dan ekstensi
  direktori yang diterima berasal dari libreyolo/utils/image_loader.py. Ekstensi
  video dan path penyimpanan berasal dari libreyolo/utils/video.py. Sintaks
  layar berasal dari libreyolo/utils/screen.py. Bentuk hasil dan nilai default
  argumen berasal dari InferenceRunner.__call__ dalam
  libreyolo/models/base/inference.py.
snippets:
  images:
    - label: Satu gambar
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        # Sumber satu gambar mengembalikan satu Results, bukan daftar.
        result = model(SAMPLE_IMAGE)
        print(len(result.boxes), "detections")
    - label: Gambar dalam memori
      language: python
      code: |
        import numpy as np
        from PIL import Image

        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreYOLO9s.pt")

        pil_image = Image.open(SAMPLE_IMAGE)
        array = np.asarray(pil_image)
        raw_bytes = open(SAMPLE_IMAGE, "rb").read()

        for source in (pil_image, array, raw_bytes):
            result = model(source)
            print(type(source).__name__, len(result.boxes))
    - label: Folder
      language: python
      code: >
        from pathlib import Path

        from PIL import Image


        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        folder = Path("sample_folder")

        folder.mkdir(exist_ok=True)

        image = Image.open(SAMPLE_IMAGE)

        for index in range(3):
            image.save(folder / f"frame_{index}.jpg")

        model = LibreYOLO("LibreYOLO9s.pt")


        # Folder mengembalikan daftar, satu Results per gambar, diurutkan
        berdasarkan path.

        results = model(str(folder))

        print(len(results), "images")
    - label: CLI
      language: bash
      code: |
        libreyolo predict model=LibreYOLO9s.pt \
          source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
  video:
    - label: Berkas video (sediakan klip sendiri)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Ganti clip.mp4 dengan berkas video di disk.
        for result in model("clip.mp4", stream=True):
            print(result.frame_idx, len(result.boxes))
    - label: 'Setiap frame ketiga, ditulis ke disk'
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        for result in model("clip.mp4", stream=True, vid_stride=3, save=True):
            pass
  live:
    - label: Webcam (memerlukan kamera terpasang)
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Indeks webcam 0. Sumber live tidak berakhir, jadi batasi loop.
        for result in itertools.islice(model(0, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
    - label: RTSP (memerlukan URL kamera yang dapat dijangkau)
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        source = "rtsp://user:password@192.168.1.64:554/Streaming/Channels/101"

        for result in itertools.islice(model(source, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
  streams:
    - label: Berkas .streams (sediakan kamera sendiri)
      language: python
      code: >
        import itertools

        from pathlib import Path


        from libreyolo import LibreYOLO


        Path("cameras.streams").write_text(
            "# one source per line, blank lines and comments are skipped\n"
            "rtsp://192.168.1.64:554/Streaming/Channels/101\n"
            "rtsp://192.168.1.65:554/Streaming/Channels/101\n",
            encoding="utf-8",
        )


        model = LibreYOLO("LibreYOLO9s.pt")

        for result in itertools.islice(model("cameras.streams", stream=True),
        100):
            print(result.frame_idx, len(result.boxes))
    - label: Daftar kamera
      language: python
      code: |
        import itertools

        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        cameras = [0, "rtsp://192.168.1.64:554/Streaming/Channels/101"]

        for result in itertools.islice(model(cameras, stream=True), 100):
            print(result.frame_idx, len(result.boxes))
  screen:
    - label: Satu screenshot (memerlukan mss dan sesi desktop)
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Tanpa stream=True, ini mengambil satu frame.
        result = model("screen")
        print(len(result.boxes), "detections")
    - label: 'Wilayah pada satu monitor, secara kontinu'
      language: python
      code: >
        import itertools


        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # "screen <monitor> <left> <top> <width> <height>"

        for result in itertools.islice(model("screen 1 100 200 512 256",
        stream=True), 50):
            print(len(result.boxes))
source_hash: c371965951dd0181
---

## Cara sumber diklasifikasikan

`classify_source` memeriksa nilai sebelum apa pun dibuka atau diunduh, dalam urutan berikut.
Aturan pertama yang cocok akan digunakan.

| Sumber | Dibaca sebagai |
|---|---|
| `"screen"`, `"screen 1"`, `"screen 1 100 200 512 256"` | Screen capture |
| `int` nonnegatif, atau string digit tanpa berkas bernama sama | Webcam |
| URL `rtsp://`, `rtmp://`, `tcp://`, atau `udp://` | Stream jaringan |
| URL `http(s)://` yang path-nya berakhir dengan `.m3u8` | Stream jaringan |
| URL halaman YouTube | Stream jaringan |
| Daftar atau tuple yang semua entrinya live atau video | Beberapa stream live |
| Daftar atau tuple lainnya | Batch gambar |
| Path yang berakhir dengan `.streams` | Beberapa stream live |
| Path dengan ekstensi video | Berkas video |
| Direktori yang ada | Folder gambar |
| Semua yang lain | Satu gambar |

Daftar yang mencampur sumber live dengan gambar akan memunculkan `TypeError`. Indeks webcam
negatif akan memunculkan `ValueError`.

Classifier tidak pernah menyentuh jaringan, sehingga URL yang salah muncul saat capture
dibuka, bukan ketika `predict` dipanggil.

## Gambar

<code-tabs name="images" />

Sumber satu gambar menerima tujuh tipe.

| Tipe | Dibaca sebagai |
|---|---|
| `str` atau `pathlib.Path` | Berkas lokal, `http(s)://`, `s3://`, atau `gs://` |
| `PIL.Image.Image` | Dikonversi ke RGB |
| `numpy.ndarray` | Grayscale 2D, atau HWC maupun CHW 3D; array 4D memakai gambar pertama |
| `torch.Tensor` | CHW atau NCHW, dibaca sebagai RGB; tensor batch memakai gambar pertama |
| `bytes` | Data gambar terenkode |
| `io.BytesIO` | Data gambar terenkode |

Semuanya dikonversi ke RGB sebelum prapemrosesan. Array NumPy adalah satu-satunya kasus dengan
urutan channel ambigu, sehingga `color_format` mengontrolnya: `"auto"` (default) membiarkan
array apa adanya, sedangkan `"bgr"` membalik channel, seperti yang dibutuhkan frame OpenCV.

Array float diskalakan ulang berdasarkan rentangnya sendiri: nilai hingga `1.0` dikalikan 255,
sedangkan nilai lebih tinggi dipotong ke `[0, 255]`. Array RGBA membuang channel alfa.

Path remote memerlukan satu paket per jenis dan tidak ada yang dipasang secara default:
`requests` untuk `http(s)://`, `boto3` untuk `s3://`, dan `gcsfs` untuk `gs://`.

## Folder

Direktori dipindai secara rekursif dan diurutkan, lalu setiap berkas dengan salah satu sufiks
berikut menjadi gambar: `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`, `.tiff`, `.tif`.
Semua yang lain dilewati. Folder kosong mengembalikan daftar kosong, bukan memunculkan error.

Folder dan daftar adalah dua sumber yang menerima `batch`, yang menjalankan satu forward
bertumpuk per bagian pada family yang mendukungnya. Lihat
[Performa inferensi](/docs/predict/performance).

## Berkas video

<code-tabs name="video" />

Path dihitung sebagai video jika sufiksnya salah satu dari `.asf`, `.avi`, `.gif`, `.m4v`,
`.mkv`, `.mov`, `.mp4`, `.mpeg`, `.mpg`, `.ts`, `.wmv`, `.webm`.

`.gif` muncul di kedua daftar. Path `.gif` yang diteruskan langsung ke `predict` dibuka sebagai
video karena pemeriksaan video berjalan lebih dulu; `.gif` dalam folder yang dipindai dimuat
sebagai gambar diam.

`vid_stride` memproses setiap frame ke-N dan memakai default `1`. Tanpa `stream=True`, seluruh
video didekode menjadi daftar, dan hasil lebih dari 500 frame setelah striding memunculkan
peringatan yang menyarankan `stream=True`.

Setiap `Results` dari video membawa `frame_idx`.

## Webcam, stream jaringan, dan YouTube

<code-tabs name="live" />

Sumber live tidak terbatas, sehingga memerlukan `stream=True`. Tanpanya, `predict` memunculkan
`ValueError`, bukan mencoba mengumpulkan daftar tanpa akhir.

Frame dibaca pada thread latar belakang, satu per capture. Secara default, antrean hanya
menyimpan frame terbaru, sehingga model yang lebih lambat daripada kamera melewati frame,
bukan tertinggal. `stream_buffer=True` mempertahankan setiap frame yang diambil, dengan biaya
latensi yang terus bertambah.

Indeks webcam berupa `int` atau string digit. Di Windows, capture dibuka melalui backend
DirectShow terlebih dahulu dan kembali ke backend default jika gagal.

URL halaman YouTube diselesaikan menjadi URL media langsung tanpa mengunduh video, yang
memerlukan `yt-dlp`:

```bash
pip install "libreyolo[stream]"
```

Label stream disensor sebelum dicatat atau digunakan sebagai nama berkas. URL dengan kredensial
ditampilkan sebagai `user:***@host`, sedangkan query string dibuang dari label stream langsung
karena URL bertanda tangan dan bearer token berada di sana. Id video YouTube dipertahankan
karena bukan kredensial.

## Beberapa kamera sekaligus

<code-tabs name="streams" />

Berkas `.streams` memuat satu sumber per baris. Baris kosong dan baris yang diawali `#`
diabaikan. Setiap baris lain harus berupa indeks webcam, stream jaringan, URL YouTube, atau
path berkas video; yang lain memunculkan `ValueError` dengan nomor baris. Berkas kosong akan
memunculkan error, bukan memulai tanpa kamera.

Daftar atau tuple sumber live melakukan hal yang sama tanpa berkas.

Setiap capture mendapat thread sendiri, dan frame semuanya dimultipleks menjadi satu generator.
Setiap tahap melakukan polling pada tiap stream aktif dan menghasilkan apa pun yang siap,
sehingga kamera lambat tidak menghambat kamera cepat, dan frame dari kamera berbeda saling
berselang. Stream yang berakhir keluar dari rotasi sementara yang lain berlanjut.

## Screen capture

<code-tabs name="screen" />

Sumber layar adalah kata `screen` yang diikuti nol, satu, empat, atau lima bilangan bulat.
Jumlah lain akan memunculkan `ValueError`.

| Bentuk | Mengambil |
|---|---|
| `"screen"` | Semua monitor, digabungkan |
| `"screen 1"` | Monitor 1 |
| `"screen 100 200 512 256"` | Kotak pada desktop gabungan |
| `"screen 1 100 200 512 256"` | Kotak pada monitor 1 |

Koordinat kotak adalah `left top width height`, relatif terhadap sudut kiri atas monitor
terpilih. Sumber layar melaporkan frame rate sebagai 30 dibagi `vid_stride`, yaitu laju untuk
menulis video simpanan. Capture memerlukan paket `mss`:

```bash
pip install mss
```

Tanpa `stream=True`, sumber layar mengambil satu frame dan mengembalikan satu `Results`,
setara screenshot dengan prediksi pada berkas gambar. Dengan `stream=True`, capture berlanjut
hingga loop dihentikan.

## Hasil yang dikembalikan predict

Bentuk nilai hasil bergantung pada sumber dan `stream`.

| Sumber | `stream=False` | `stream=True` |
|---|---|---|
| Satu gambar | Satu `Results` | Generator satu `Results` |
| Daftar gambar | Daftar `Results` | Generator |
| Folder | Daftar `Results` | Generator |
| Berkas video | Daftar `Results` | Generator |
| Layar | Satu `Results` | Generator, tidak terbatas |
| Webcam, stream jaringan, `.streams` | `ValueError` | Generator, tidak terbatas |

Satu gambar mengembalikan objek `Results` itu sendiri. Mengindeksnya memilih deteksi, bukan
gambar, sehingga `result[0]` pada prediksi satu gambar adalah kotak pertama, bukan gambar
pertama. Untuk isinya, lihat [Bekerja dengan hasil](/docs/predict/results).

## Lokasi penulisan hasil simpan

`save=True` menulis output beranotasi ke direktori proses, bukan mengembalikannya.

Gambar masuk ke `runs/detect/predict`, `runs/detect/predict2`, dan seterusnya yang bertambah
otomatis, dengan nama berkas sumber dipertahankan. Setiap gambar dalam satu proses masuk ke
direktori yang sama, sehingga dua folder input dengan nama berkas sama saling menimpa.
Gambar dalam memori tidak memiliki nama berkas dan diberi nomor `image0`, `image1`, dan seterusnya.

Video dan sumber live ditulis sebagai satu `.mp4` yang dinamai berdasarkan sumber.

`output_path` mengganti direktori. Path dengan sufiks dianggap sebagai berkas, sedangkan path
tanpa sufiks dianggap sebagai direktori. `output_file_format` memilih encoding gambar diam
dan menerima `jpg`, `png`, atau `webp`.

Setelah penyimpanan, path yang ditulis juga dilampirkan ke hasil sebagai `result.saved_path`.
