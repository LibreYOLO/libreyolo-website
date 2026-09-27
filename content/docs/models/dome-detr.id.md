---
title: Dome-DETR
families:
  - domedetr
seo_title: 'Dome-DETR: deteksi objek sangat kecil di LibreYOLO'
description: Gunakan Dome-DETR untuk deteksi objek sangat kecil, pelatihan, dan validasi. Checkpoint
  pretrained di mirror mempertahankan ketentuan untuk riset akademik saja.
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
last_verified: 1.6.0
snippets:
  predict:
  - label: Python
    language: python
    code: |
      from libreyolo import LibreYOLO, SAMPLE_IMAGE

      # Bobot pretrained terbatas untuk riset akademik.
      model = LibreYOLO("LibreDOMEDETRs-visdrone.pt", device="cpu")
      print(model(SAMPLE_IMAGE).boxes)
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
source_hash: 8482301790a9b8d9
---

## Instalasi

Dome-DETR tidak memerlukan extra opsional. Semua yang diimpornya tersedia
dalam instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Enam checkpoint hasil konversi diunduh otomatis dari mirror LibreYOLO. Ketentuan upstream membatasi penggunaannya untuk riset akademik.

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

Tidak ada baris benchmark Vision Analysis yang tercatat untuk family ini.

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

<checkpoint-table />

## Lisensi

<provenance-box>

Keenam mirror mempertahankan batasan upstream untuk riset akademik saja. Kode memiliki lisensi tersendiri. Repositori upstream berlisensi Apache-2.0, port LibreYOLO berlisensi MIT, dan bobot yang Anda latih dengan data sendiri tetap milik Anda.

</provenance-box>
