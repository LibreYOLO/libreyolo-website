---
title: Tracking objek
seo_title: Tracking objek di LibreYOLO
description: >-
  Lacak objek lintas frame video di LibreYOLO dengan ByteTrack, BoT-SORT,
  OC-SORT, atau Deep OC-SORT pada model deteksi, segmentation, atau pose apa
  pun.
lead: >-
  Tracking memberikan identitas stabil kepada setiap deteksi lintas frame video.
  LibreYOLO tidak memodelkannya sebagai task dengan bobot sendiri: ini adalah
  mode predict, model.track(), yang menjalankan tracker terpilih pada output per
  frame dari model deteksi, segmentation, atau pose.
keywords:
  - object tracking python
  - multi object tracking
  - bytetrack
  - botsort
  - ocsort
  - deep ocsort
  - track id
  - reid tracking
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # track() adalah generator: satu Results per frame yang diproses.
        for result in model.track("video.mp4"):
            print(result.track_id)        # tensor int (N,), selaras dengan box
            print(result.boxes.xyxy)
    - label: Pilih tracker
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # "bytetrack" (default), "botsort", "ocsort", atau "deepocsort".
        for result in model.track("video.mp4", tracker="botsort"):
            print(result.track_id)
    - label: Simpan video beranotasi
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Tanpa output_path, file berada di runs/track/<video_stem>.mp4.
        for result in model.track("video.mp4", save=True, vid_stride=2):
            pass
    - label: Sesuaikan tracker
      language: python
      code: >
        from libreyolo import BoTSortConfig, LibreYOLO


        model = LibreYOLO("LibreYOLO9s.pt")


        # Jenis konfigurasi memilih tracker, sehingga tracker= redundan di sini.

        config = BoTSortConfig(track_buffer=60, frame_rate=25, enable_cmc=False)

        for result in model.track("video.mp4", tracker_config=config):
            print(result.track_id)

        # Atau berikan field yang sama sebagai keyword argument dan biarkan
        track() membuatnya.

        for result in model.track("video.mp4", tracker="botsort",
        track_buffer=60):
            print(result.track_id)
source_hash: f1fa7dcf60597d6b
---

## Definisi

Tracking bukan salah satu key task LibreYOLO dan tidak ada checkpoint tracking
untuk diunduh. Tracking adalah metode pada model, `model.track(source)`, yang
menjalankan deteksi pada setiap frame dan mengasosiasikan hasil lintas waktu.
Metode ini berupa generator: menghasilkan satu `Results` per frame yang diproses,
dengan `result.track_id` berupa tensor bilangan bulat `(N,)` yang selaras dengan
`result.boxes`. Id yang sama juga tersedia pada `result.boxes.id`.

Hanya objek terkonfirmasi yang sedang dilacak yang dihasilkan. Track yang
kehilangan asosiasi tetap hidup selama jumlah frame yang dikonfigurasi sebelum
dihapus, `track_buffer` untuk ByteTrack dan BoT-SORT serta `max_age` untuk dua
varian OC-SORT, sehingga objek yang ditemukan kembali dalam window tersebut
mempertahankan id asli.

Karena asosiasi terjadi setelah deteksi, payload lain pada frame tetap tersedia:
`Results` tracking adalah `Results` deteksi yang di-slice ke baris cocok, sehingga
mask dan keypoint diteruskan bersama box.

## Model

Dua pilihan independen membentuk run tracking: model yang menghasilkan box per
frame dan tracker yang menghubungkannya.

Setiap model LibreYOLO native dengan task deteksi, segmentation, atau pose
menyediakan `track()`. Lihat [indeks model](/docs/models) untuk list lengkap,
atau mulai dari [YOLO9](/docs/models/yolov9),
[RF-DETR](/docs/models/rf-detr), [D-FINE](/docs/models/d-fine), atau
[RTMDet](/docs/models/rtmdet). Task tanpa box untuk diasosiasikan menolak
pemanggilan: classification, oriented box, point, depth, surface normal, edge,
semantic dan panoptic segmentation, restoration, OCR, serta body mesh.

Dua tier model juga menolaknya. Model `LibreSAM` adalah image segmenter dan
model `LibreOpenVocab` adalah detektor per frame; keduanya memunculkan error dari
`track()` dan menggunakan `predict()` per frame.

Tracking berjalan pada model PyTorch native. Artefak hasil ekspor yang dimuat
melalui `LibreYOLO("model.onnx")` mengembalikan objek backend runtime dengan
`predict()`, tetapi tanpa `track()`.

Empat tracker disertakan dan dipilih melalui argumen `tracker`:

`"bytetrack"` adalah default. Tracker ini hanya menggunakan motion, dengan
Kalman filter dan asosiasi tiga tahap: deteksi confidence tinggi, pass kedua
yang memberi deteksi confidence rendah kesempatan mencocokkan track sebelum
dibuang, lalu track belum terkonfirmasi. Dikonfigurasi dengan `TrackConfig`.

`"botsort"` mempertahankan lifecycle tiga tahap ByteTrack, tetapi memakai status
Kalman center-width-height dan mengompensasi track prediksi terhadap gerakan
kamera sebelum pencocokan. Ini adalah varian motion-only BoT-SORT tanpa model
appearance. Dikonfigurasi dengan `BoTSortConfig`, yang menambahkan `enable_cmc`,
`cmc_method`, dan `cmc_downscale`.

`"ocsort"` juga hanya memakai motion dan menambahkan term arah velocity ke biaya
asosiasi, pass asosiasi kedua terhadap observasi nyata terakhir setiap track,
serta smoothing status Kalman sepanjang trajectory virtual ketika track
ditemukan kembali. Dikonfigurasi dengan `OCSortConfig`.

`"deepocsort"` memperluas OC-SORT dengan appearance. Setiap track mempertahankan
moving average embedding re-identification yang dibobot confidence, dan cosine
similarity masuk biaya asosiasi, sehingga identitas bertahan melalui occlusion
panjang dan target yang bersilangan. Biayanya satu forward network embedding
kecil per frame, dan bobot OSNet diunduh saat penggunaan pertama. Dikonfigurasi
dengan `DeepOCSortConfig`.

## Predict

<code-tabs name="predict" />

`track_conf` menetapkan ambang batas tahap asosiasi pertama:
`track_high_thresh` untuk ByteTrack dan BoT-SORT, `det_thresh` untuk OC-SORT dan
Deep OC-SORT. Nilai ini bukan `conf` milik `predict()`. Untuk ByteTrack,
BoT-SORT, dan OC-SORT, detektor berjalan pada ambang batas lebih rendah secara
internal agar deteksi lemah tersedia bagi recovery pass. Deep OC-SORT menjalankan
detektor pada `det_thresh`. Untuk ByteTrack dan BoT-SORT, `track_conf` harus sama
dengan atau di atas `track_low_thresh`, dengan default 0.1.

Pengaturan tracker masuk melalui dua cara. Berikan instance konfigurasi ke
`tracker_config=`, dan jenisnya memilih tracker sehingga `tracker=` redundan.
Atau berikan field sebagai keyword argument dan biarkan `track()` membuat
konfigurasi untuk tracker terpilih; key tidak dikenal menghasilkan peringatan.
Dalam kedua cara, `track_conf` diabaikan setelah key pencocokan ditetapkan
eksplisit.

Argumen lain mencerminkan prediksi: `iou`, `imgsz`, `classes`, `max_det`,
`vid_stride`, `show`, dan `save` bersama `output_path`. Source adalah path file
video. Lihat [prediksi](/docs/predict) untuk penanganan hasil.

## Train

Tracker tidak dilatih. Tiga dari empat tracker merupakan model motion murni
tanpa parameter terpelajari, sedangkan network appearance Deep OC-SORT adalah
checkpoint re-identification terbitan yang diunduh saat penggunaan pertama.
Peningkatan kualitas tracking berarti memperbaiki detektor atau menyesuaikan
ambang batas asosiasi di atas.
