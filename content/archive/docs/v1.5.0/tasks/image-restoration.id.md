---
title: Pemulihan gambar
seo_title: Pemulihan dan peningkatan gambar dalam LibreYOLO
description: >-
  Mengurangi noise, menghapus blur, dan meningkatkan resolusi gambar dalam
  LibreYOLO. Memprediksi gambar RGB yang dipulihkan, melatih NAFNet pada data
  berpasangan, dan membaca kunci PSNR dan SSIM.
lead: >-
  Pemulihan gambar mengambil gambar yang rusak dan menghasilkan gambar yang
  bersih. LibreYOLO menampilkannya sebagai pemulihan task, yang mencakup
  pengurangan noise, penghapusan blur, dan super-resolusi di belakang satu
  kontrak keluaran: satu gambar RGB masuk, satu gambar RGB keluar.
keywords:
  - Pemulihan gambar python
  - model denoising gambar
  - super resolusi gambar python
  - model deblurring
  - validasi PSNR SSIM
last_verified: 1.5.0
snippets:
  predict:
    - label: Meningkatkan resolusi gambar
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Generator kompak 4x; batas ubin memuncak pada penggunaan memori sumber yang besar.
        model = LibreYOLO("LibreRealESRGANx4t-restore.pt")
        result = model(SAMPLE_IMAGE, tile=512, tile_pad=10)

        result.restored.save("upscaled.png")
        print(result.restored.array.shape)   # 4x input di setiap sumbu
    - label: Menghilangkan noise dari gambar
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Dilatih pada noise gambar nyata SIDD; output tetap pada ukuran input.
        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model(SAMPLE_IMAGE)

        result.restored.save("denoised.png")
        print(result.restore_scale)   # 1: tidak ada peningkatan skala untuk ini checkpoint
  train:
    - label: Menyesuaikan NAFNet pada gambar berpasangan
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        model.train(data="my-dataset.yaml", epochs=100, imgsz=256, batch=16, lr0=1e-3)
    - label: Mencatat asal-usul pada checkpoint
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # degradasi dan dataset ditulis ke dalam checkpoint yang disimpan untuk
        # asal-usul; mereka tidak berperan dalam pelatihan.
        model.train(
            data="my-dataset.yaml",
            epochs=100,
            degradation="denoise",
            dataset="MyDataset",
        )
  val:
    - label: Memvalidasi dan membaca kunci metrik
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # val() mengembalikan dict biasa, bukan objek.
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PSNR"])   # kecocokan
        print(metrics["metrics/SSIM"])
  export:
    - label: Ekspor
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # imgsz diperbaiki dalam grafik, jadi masukkan ukuran yang sebenarnya digunakan deployment Anda
        # untuk memberi makan model.
        model.export(format="onnx", imgsz=256)
    - label: Jalankan berkas yang diekspor
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Pabrik mengalihkan berdasarkan akhiran berkas, jadi artefak yang diekspor dimuat
        # seperti halnya checkpoint dan mengembalikan objek Results yang sama.
        model = LibreYOLO("LibreNAFNetl-restore-sidd.onnx")
        result = model(SAMPLE_IMAGE)

        result.restored.save("denoised.png")
source_hash: 9dc81cadb3ebf18b
---

## Definisi

`restore` task memetakan satu gambar ke gambar lain. Penghilangan noise, penghilangan blur, dan
super-resolusi semuanya adalah task yang sama di sini, karena mereka berbagi satu kontrak:
model mengkonsumsi gambar RGB dan mengembalikan gambar RGB, dan degradasi yang
dilatih untuk dibatalkan adalah sifat dari checkpoint daripada API.

Sebuah prediksi mengisi `result.restored`, muatan `RestoredImage` yang memegang
array RGB uint8 `(H, W, 3)`. `.array` mengembalikannya sebagai NumPy dan `.save(path)`
menuliskannya ke disk. `result.restore_scale` mencatat faktor peningkatan yang
membawa, yang merupakan `1` untuk checkpoint yang mempertahankan resolusi.
`result.boxes` tetap kosong, sehingga `conf`, `iou`, dan `max_det` diterima untuk
paritas tanda tangan tetapi tidak memiliki efek, dan `save=True` menulis gambar yang dipulihkan
secara langsung daripada foto yang diberi anotasi.

## Model

Tiga keluarga melayani `restore`, dibagi menurut degradasi yang mereka batalkan.

[NAFNet](/docs/models/nafnet) adalah denoiser, dan satu-satunya restore family
LibreYOLO yang dapat dilatih. Arsitekturnya menggantikan aktivasi nonlinier dari
blok UNet dengan perkalian elemen demi elemen, dan checkpoint yang dipublikasikan
dilatih pada noise gambar nyata SIDD. Keluaran tetap pada resolusi input.

[Real-ESRGAN](/docs/models/real-esrgan) adalah penguat skala praktis: tiga
titik pemeriksaan dilatih terhadap degradasi sintetis daripada hanya bicubic
penurunan skala, pada 4x, 2x, dan generator 4x yang lebih kecil dan lebih cepat yang dibuat untuk lebih rendah
latensi.

[SwinIR](/docs/models/swinir) meningkatkan skala 4x dengan Swin Transformer backbone, dalam
tiga ukuran mencakup generator ringan resmi dan dua dunia nyata
generator.

## Prediksi

Bobot diunduh dari Hugging Face saat penggunaan pertama dan disimpan secara lokal.

<code-tabs name="predict" />

Restorasi dijalankan pada resolusi gambar sumber itu sendiri daripada resolusi tetap
kanvas jaringan, padding hanya pada faktor downsample jaringan, sehingga keduanya waktu
dan memori meningkat seiring dengan jumlah piksel dari input Anda. `tile` membagi maju
melewati ubin yang saling tumpang tindih dan menyatukan kembali jahitannya, dan `tile_pad`
apakah halo ditambahkan di sekitar setiap ubin sebelum dipotong kembali; keduanya adalah
Argumen kata kunci Python. Lihat [prediksi](/docs/predict) untuk sumber, streaming
dan penanganan hasil.

## Format dataset

Restorasi memasangkan setiap gambar input yang rusak dengan gambar target yang bersih secara tepat
resolusi yang sama, dicocokkan berdasarkan nama berkas.

```text
dataset/
  data.yaml
  inputs/
    train/photo.jpg
    val/photo.jpg
  targets/
    train/photo.jpg
    val/photo.jpg
```

```yaml
path: dataset
train: inputs/train
val: inputs/val
input_dir: inputs
target_dir: targets
degradation: denoise
dataset: MyDataset
nc: 1
names: {0: image}
```

`nc` dan `names` adalah placeholder skema; sebuah model pemulihan mengembalikan
`Results.restored`, bukan deteksi. `degradation` dan `dataset` bersifat opsional
. `target_stem_suffix` mencakup dataset yang memberi nama gambar
bersih secara berbeda dari pasangannya yang terdegradasi. Validasi mempertahankan resolusi asli dan
hanya menambahkan padding secukupnya untuk menumpuk batch, sehingga metrik dihitung pada kanvas asli
. Lihat [dataset formats](/docs/reference/dataset-formats) untuk kontrak lengkap
.

## Latih

NAFNet adalah satu-satunya restore family dengan implementasi pelatihan.
`Real-ESRGAN.train()` dan `SwinIR.train()` sama-sama meningkatkan `NotImplementedError`:
checkpoint tersebut berasal dari pelatihan GAN melalui jalur degradasi sintetis,
dan trainer restore berpasangan akan berjalan tanpa mereproduksi resep tersebut.

<code-tabs name="train" />

Pelatih mengambil pasangan hasil panen dari input dan target, sehingga kedua sisi tetap
selaras. Lihat [pelatihan](/docs/train) untuk dataset, multi-GPU dan pencatat log, dan
halaman [NAFNet](/docs/models/nafnet) untuk default family ini dan
pooling saat inferensi yang dilepaskan saat pelatihan.

## Validasi

`val()` membandingkan keluaran yang dipulihkan dengan target bersih, dalam RGB, pada
kanvas asli, tanpa pemotongan tepi dan tanpa pengubahan ukuran.

<code-tabs name="val" />

`metrics/PSNR` adalah rasio puncak sinyal terhadap kebisingan dalam desibel, dan itu juga
`fitness`, pembacaan pemilihan best-checkpoint. `metrics/SSIM` adalah
kesamaan struktural dalam `[0, 1]`, dihitung dengan jendela Gaussian 11x11 pada
sigma 1.5 dan dirata-ratakan di ketiga saluran warna. Semakin tinggi semakin baik untuk kedua-duanya.

## Ekspor

Model restore yang diekspor dimuat kembali melalui `LibreYOLO()` pada akhiran berkasnya,
sehingga berkas `.onnx` atau `.engine` berperilaku seperti checkpoint dan mengembalikan `.onnx` yang sama,
dengan `restored` membawa gambar keluaran.

<code-tabs name="export" />

Ekspor restore menetapkan resolusi spasial ke dalam grafik, jadi masukkan `imgsz`
yang akan digunakan dalam penerapan Anda untuk benar-benar memberi makan model. Untuk NAFNet, ukuran itu harus dapat dibagi
oleh faktor downsample jaringan, dan hanya dimensi batch yang tetap dinamis
di bawah `dynamic=True`. Untuk Real-ESRGAN dan SwinIR, meninggalkan `imgsz` akan kembali ke default
ke ukuran patch internal kecil daripada resolusi kerja Anda. Per-format
cakupan ada di setiap halaman model dan di
[matriks ekspor penuh](/docs/reference/export-matrix). [Ekspor](/docs/export)
mencantumkan argumen yang diterima setiap format.


