---
title: Dome-DETR
families:
  - domedetr
seo_title: 'Dome-DETR: deteksi objek sangat kecil di LibreYOLO'
description: >-
  Gunakan Dome-DETR di LibreYOLO untuk mendeteksi objek sangat kecil pada gambar
  udara dan drone. Konversi bobot upstream, lakukan prediksi, fine-tuning, dan
  validasi dengan kode berlisensi MIT.
lead: >-
  Spesialis objek sangat kecil yang dibangun di atas D-FINE: head kepadatan
  menentukan lokasi objek, attention encoder dibatasi pada window yang
  memuatnya, dan jumlah query ditentukan dari kepadatan tersebut alih-alih
  dibuat tetap. LibreYOLO mendukungnya untuk deteksi.
keywords:
  - Dome-DETR
  - deteksi objek kecil
  - tiny object detection
  - gambar udara
  - deteksi drone
  - remote sensing
  - VisDrone
  - AI-TOD
  - DETR
  - density adaptive queries
last_verified: 1.5.0
snippets:
  predict:
    - label: 'Konversi, lalu prediksi'
      language: bash
      code: |
        # LibreYOLO tidak menghosting bobot Dome-DETR, sehingga checkpoint
        # diambil dari repository upstream dan dikonversi satu kali.
        hf download RicePasteM/Dome-DETR --include 'best_ckpts_dome_2026/*' \
          --local-dir dome-ckpts

        python weights/convert_domedetr_weights.py \
          dome-ckpts/best_ckpts_dome_2026/dome-s-visdrone_converted.pth \
          LibreDOMEDETRs-visdrone.pt --size s --variant visdrone
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Gunakan path lokal, bukan nama polos. Tidak ada yang diunduh untuk
        family ini.

        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")

        result = model("drone-frame.jpg", save=True)


        for box in result.boxes:
            print(result.names[int(box.cls)], box.conf, box.xyxy)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreDOMEDETRs-visdrone.pt
        source=drone-frame.jpg save=True
    - label: Nama kelas
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Tidak ada checkpoint COCO, sehingga kelas berasal dari dataset yang
        dipakai

        # untuk melatih bobot dan dibaca dari metadata checkpoint.

        aitod = LibreYOLO("LibreDOMEDETRs-aitod.pt")

        print(aitod.model.names)     # 9 kelas AI-TOD-V2


        visdrone = LibreYOLO("LibreDOMEDETRs-visdrone.pt")

        print(visdrone.model.names)  # 12 kelas VisDrone
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")

        model.train(data="my-dataset.yaml", epochs=160, imgsz=800, batch=4,
        lr0=2e-4)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml \
          epochs=160 imgsz=800 batch=4 lr0=2e-4
    - label: Multi-GPU
      language: bash
      code: |
        libreyolo train model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml \
          epochs=160 device=0,1 batch=4
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreDOMEDETRs-visdrone.pt")
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/mAP50-95"])
        print(metrics["metrics/mAP50"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreDOMEDETRs-visdrone.pt data=my-dataset.yaml
source_hash: 381f01d769e7c420
---

## Instalasi

Dome-DETR tidak memerlukan extra opsional. Semua yang diimpornya tersedia
dalam instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Tidak ada yang dapat diunduh otomatis. LibreYOLO tidak menghosting bobot ini,
sehingga alurnya adalah mengambil checkpoint upstream, mengonversinya satu
kali, lalu memuat berkas hasil konversi berdasarkan path. [Lisensi](#licensing)
menjelaskan alasannya.

<code-tabs name="predict" />

Objek `Results` yang dikembalikan sama seperti yang dikembalikan setiap family,
sehingga mengganti detector hanya memerlukan perubahan satu baris. `conf` dan
`max_det` memfilter pemilihan query. `iou` diterima demi paritas API, tetapi
tidak berpengaruh karena decoder adalah set predictor tanpa langkah NMS. Lihat
[prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

Dua kemampuan dinonaktifkan untuk family ini. Capture CUDA graph dinonaktifkan
karena jumlah query PAQI bergantung pada data, sehingga bentuk forward pass
berubah dari satu gambar ke gambar lain. Perubahan seperti ini tidak dapat
ditampung oleh graph capture. Augmentasi waktu pengujian berjalan pada satu
ukuran persegi tetap, sehingga permintaan TTA multi-scale tidak berpengaruh.

## Varian

Ada tiga ukuran, s, m, dan l, semuanya pada 800 kali 800. Ukuran memilih
backbone, sedangkan dataset asal bobot memilih kedalaman decoder dan anggaran
query, sehingga kode ukuran saja tidak mengidentifikasi sebuah graph. Bobot
AI-TOD-V2 memilih antara 300 dan 1500 query per gambar, bobot VisDrone antara
250 dan 500, dan model large menjalankan empat lapisan decoder pada AI-TOD-V2
dibandingkan dengan enam pada VisDrone.

Dome-DETR adalah D-FINE dengan tiga tambahan. DeFE memprediksi peta kepadatan.
MWAS menggunakan peta tersebut untuk membatasi attention encoder pada window
yang benar-benar memuat objek, alih-alih melakukan attention di semua tempat.
PAQI menentukan ukuran set query dari kepadatan yang sama, bukan mendekode 300
query tetap. Peningkatan terpusat pada objek yang paling kecil dan menyempit
saat ukurannya bertambah. Ablation upstream sendiri mengubah AP pada objek yang
sangat kecil dari 14.0 menjadi 17.8, sedangkan AP pada objek sedang hanya
berubah dari 45.4 menjadi 46.4. Perlakukan model ini sebagai pendamping
[D-FINE](/docs/models/d-fine) untuk gambar udara, drone, dan remote sensing,
bukan sebagai penggantinya.

LibreYOLO tidak memublikasikan baris benchmark untuk family ini karena tidak
ada checkpoint yang dipublikasikan untuk diukur.

## Pelatihan

Dome-DETR dapat dilatih. Pelatihan menjalankan objective upstream secara penuh:
loss D-FINE ditambah supervisi kepadatan dan jumlah DeFE, dengan query padding
yang dikeluarkan dari term klasifikasi serta mask attention denoising per gambar
agar padding dari satu gambar tidak bocor ke gambar lain.

<code-tabs name="train" />

Konfigurasi mewarisi resep D-FINE dan mengubah hal yang diperlukan MWAS.
`imgsz` adalah 800, `lr0` adalah `2e-4`, grup parameter backbone diskalakan
dengan `backbone_lr_mult=0.1`, dan `multi_scale` dipaksa nonaktif karena window
MWAS mengharuskan input tetap habis dibagi stride 8. Nilai default `batch`
adalah 4, bukan 16 seperti D-FINE. PAQI mengisi setiap batch hingga selebar
anggota terlebarnya, sehingga penggunaan memori mengikuti gambar tersibuk dalam
batch, bukan rata-ratanya.

Ada satu batasan akurasi yang perlu dinyatakan dengan jelas. Upstream melatih
selama 160 epoch pada `MultiStepLR(milestones=[80, 120], gamma=0.8)`, sedangkan
nilai default ini menjalankan schedule flat-cosine D-FINE selama 160 epoch yang
sama. Schedule tersebut belum direproduksi di sini, begitu pula angka AP dari
makalah. Karena itu, baca angka tersebut sebagai hasil penulis upstream, bukan
janji bahwa resep ini akan mencapainya. Gunakan schedule upstream jika tujuannya
adalah mencocokkan makalah.

Lihat [pelatihan](/docs/train) untuk dataset, augmentasi, multi-GPU, dan logger.

## Validasi

`val()` mengembalikan dictionary yang menggunakan nama metrik sebagai key dan
mencetak hasil per kelas saat `verbose` dibiarkan aktif.

<code-tabs name="val" />

Validasi berjalan terhadap dataset Anda sendiri dalam format yang digunakan
saat pelatihan. Gate validasi COCO milik library tidak berlaku di sini karena
tidak ada checkpoint COCO untuk family ini yang dapat diukur terhadapnya.

## Ekspor

Ekspor tidak didukung untuk format apa pun, dan permintaan ekspor akan
memunculkan error alih-alih menghasilkan berkas.

Penyebabnya adalah PAQI. Komponen ini menentukan jumlah query per gambar dari
proposal yang difilter berdasarkan kepadatan dan loop suppression adaptif
kepadatan yang greedy, sehingga panjang output decoder menjadi properti input,
bukan graph. Tracing memasukkan jumlah apa pun yang kebetulan dihasilkan gambar
tracing, sehingga menciptakan artefak yang diam-diam mengembalikan hasil salah
untuk setiap gambar lain. Formulasi statis harus menguraikan suppression itu
untuk seluruh 250 hingga 1500 kandidat. Mengubahnya menjadi top-k tetap justru
akan menghapus recall objek sangat kecil yang menjadi tujuan family ini. Jika
memerlukan detection transformer yang dapat diekspor, gunakan
[D-FINE](/docs/models/d-fine).

## Checkpoint

Tidak ada yang dapat dicantumkan. LibreYOLO tidak memublikasikan bobot Dome-DETR,
dan nama dengan bentuk `LibreDOMEDETR<size>-<dataset>.pt` tidak akan mengarah ke
unduhan.

Upstream memublikasikan enam checkpoint, yaitu s, m, dan l untuk masing-masing
dari dua dataset: AI-TOD-V2 dengan 9 kelas dan VisDrone dengan 12 kelas. Tidak
ada checkpoint COCO, sehingga nama berkas kanonis selalu memuat akhiran dataset,
dan nama kelas disimpan dalam metadata checkpoint, bukan berasal dari konstanta
family. Meminta `LibreDOMEDETRs.pt` tanpa akhiran langsung memunculkan pesan yang
menyebutkan dua nama berkas sebenarnya dan perintah konversinya, alih-alih
mencoba unduhan yang akan menghasilkan 404.

`weights/convert_domedetr_weights.py` melakukan konversi. Skrip ini membangun
ulang graph LibreYOLO, memuat tensor upstream ke dalamnya, dan menolak menulis
apa pun jika satu key saja hilang, tidak diharapkan, atau memiliki bentuk yang
salah. Dengan demikian, berkas hasil konversi merupakan kecocokan persis atau
tidak dibuat sama sekali. Arahkan skrip ke `.pth` upstream serta teruskan ukuran
dan varian:

```bash
python weights/convert_domedetr_weights.py \
    dome-ckpts/best_ckpts_dome_2026/aitod-s-best.pth \
    LibreDOMEDETRs-aitod.pt --size s --variant aitod
```

Untuk kesetiaan numerik, `weights/parity_domedetr.py` membandingkan port ini
dengan implementasi upstream pada keenam checkpoint dan melaporkan
`max_abs_diff == 0.0` untuk `pred_logits` maupun `pred_boxes`, setelah lebih
dulu memeriksa mask window MWAS bit demi bit, kemudian secara terpisah
membandingkan setiap term loss dengan criterion upstream. Perjelas cakupannya:
ini adalah skrip manual yang memerlukan checkout upstream dan checkpoint yang
dipublikasikan tersedia di disk, lalu dijalankan secara manual. Skrip ini bukan
bagian dari continuous integration dan tidak ada job CI yang mereproduksinya.

## Lisensi

<provenance-box>

Bobot menjadi alasan family ini tidak dicerminkan. Model card upstream tidak
memiliki field lisensi dalam metadatanya, sedangkan prosanya menyatakan bahwa
proyek ini berlisensi Apache-2.0 sambil membatasi materi hanya untuk tujuan
penelitian akademis. Kedua tafsir itu tidak sejalan, dan tafsir yang lebih ketat
bukanlah izin redistribusi. Karena itu, LibreYOLO menautkan repository upstream
alih-alih menyalin berkas, sambil menunggu klarifikasi. Pertimbangan yang sama
juga mengatur [YOLO-NAS](/docs/models/yolo-nas) di sini.

Kode adalah persoalan terpisah dengan jawaban yang lebih jelas. Repository
upstream berlisensi Apache-2.0, port LibreYOLO berlisensi MIT, dan bobot yang
Anda latih sendiri pada data Anda tetap menjadi milik Anda.

</provenance-box>

## Sitasi

Dome-DETR dipublikasikan pada ACM Multimedia 2025 dengan judul "Dome-DETR: DETR
with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object
Detection". Preprint tersedia di
[arxiv.org/abs/2505.05741](https://arxiv.org/abs/2505.05741). Penulis tidak
memublikasikan blok BibTeX dalam repository mereka, sehingga tidak ada blok yang
direproduksi di sini alih-alih dirangkai secara manual.

<citation-block />
