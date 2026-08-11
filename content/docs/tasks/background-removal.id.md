---
title: Penghapusan latar belakang
seo_title: Penghapusan latar belakang di LibreYOLO
description: >-
  Memotong subjek dari latar belakangnya di LibreYOLO. Prediksi matte alpha
  lembut, tulis PNG transparan, dan validasi dengan MAE dan S-measure.
lead: >-
  Penghapusan latar belakang memisahkan subjek dari segala sesuatu di
  belakangnya. LibreYOLO menampilkannya sebagai matte task, yang mengembalikan
  nilai alpha lembut per piksel daripada mask depan keras.
keywords:
  - penghapusan latar belakang python
  - model alpha matting
  - segmentasi gambar dikotomis
  - potongan PNG transparan
  - matte alpha lembut
last_verified: 1.5.0
snippets:
  predict:
    - label: Prediksi matte
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        matte = result.matte
        print(matte.array.shape, matte.array.dtype)   # (H, W) float32 di [0, 1]
    - label: Tulis PNG transparan
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        # save() mengkomposit sumber dengan matte sebagai saluran alpha.
        result.save("subject.png")

        rgba = result.cutout()   # array uint8 (H, W, 4) yang sama di memori
        print(rgba.shape)
    - label: Gabungkan ke latar belakang baru
      language: python
      code: |
        import numpy as np
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        result = model(SAMPLE_IMAGE)

        rgba = result.cutout()
        alpha = rgba[..., 3:4].astype(np.float32) / 255.0
        backdrop = np.full_like(rgba[..., :3], 255)          # putih
        composited = (rgba[..., :3] * alpha + backdrop * (1 - alpha)).astype(np.uint8)
        print(composited.shape)
  val:
    - label: Validasi dan baca kunci metrik
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")

        # Sebuah direktori yang memuat direktori gambar/ dan matte dapat digunakan sebagai pengganti
        # sebuah dataset YAML.
        metrics = model.val(data="my-matte-dataset/")

        print(metrics["metrics/MAE"])        # lebih rendah lebih baik
        print(metrics["metrics/Smeasure"])   # kebugaran, lebih tinggi lebih baik
  export:
    - label: Ekspor
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreBiRefNetl-matte.pt")
        model.export(format="torchscript")
    - label: Jalankan berkas yang diekspor
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        # Pabrik menentukan rute berdasarkan akhiran berkas, sehingga artefak yang diekspor dapat dimuat
        # seperti checkpoint lainnya dan mengembalikan objek Results yang sama.
        model = LibreYOLO("LibreBiRefNetl-matte.torchscript")
        result = model(SAMPLE_IMAGE)

        print(result.matte.array.shape)
source_hash: f7d88c74d9729268
---

## Definisi

`matte` task memprediksi satu nilai alpha per piksel dari satu gambar RGB: `1`
sepenuhnya latar depan dan `0` sepenuhnya latar belakang. Nilainya bersifat kontinu daripada
daripada biner, yang merupakan inti dari task. Sebuah mask keras hanya satu ambang saja,
pada 0,5, sementara matte lembut selain itu juga memberikan coverage parsial pada rambut,
bulu dan tepi yang buram karena gerakan yang dibuang oleh topeng biner.

Sebuah prediksi mengisi `result.matte`, sebuah payload `Matte` yang memuat `(H, W)`
array float32 di `[0, 1]` pada kanvas gambar asli, dapat diakses sebagai NumPy
melalui `.array`. `result.cutout()` menyusun gambar sumber dengan itu
alpha menjadi array RGBA uint8 `(H, W, 4)`, dan `result.save(path)` menulis
hal yang sama ke PNG dengan latar belakang transparan. `result.boxes` tetap kosong,
jadi `conf`, `iou` dan `max_det` tidak berpengaruh.

## Model

Dua keluarga melayani `matte`, dan mereka berbagi jalur maju.

[BiRefNet](/docs/models/birefnet) adalah jaringan referensi bilateral task adalah]
dibangun di sekitar, dipublikasikan di sini sebagai satu tingkat Swin-L checkpoint.

[FeyNobg](/docs/models/feynobg) adalah varian yang diperdalam dari Feyn Inc.: BiRefNet's]
arsitektur dengan tahap Swin ketiga yang dikembangkan dari 18 menjadi 24 blok, kemudian
dilatih ulang. LibreYOLO menggunakan kembali jalur maju BiRefNet, pra-pemrosesan dan
output single-logit untuk itu, jadi prediksi, validasi dan penanganan checkpoint berperilaku
identik; bobot dan identitas family adalah milik FeyNobg sendiri.

Keduanya memiliki lisensi dengan bobot yang berbeda. Keduanya tercantum di halaman model, dan
lisensi di repositori Hugging Face dari checkpoint tertentu adalah
yang berwibawa.

## Prediksi

Bobot diunduh dari Hugging Face saat penggunaan pertama dan disimpan secara lokal.

<code-tabs name="predict" />

Kedua keluarga berjalan pada kanvas asli tetap 1024x1024 dan mengubah ukuran matte kembali
ke gambar asli. Resolusi yang berbeda tidak didukung, karena Swin
Tabel posisi relatif backbone terikat pada ukuran itu, dan ketidaksesuaian
menginterpolasinya dengan buruk daripada meningkatkan. `Results.save()` didefinisikan untuk
hanya matte results dan membutuhkan gambar sumber, yang dimuat ulang darinya
`Results.path` kecuali Anda melewati satu. Lihat [prediksi](/docs/predict) untuk sumber,
streaming dan penanganan hasil.

## Format dataset

Validasi matte memadankan setiap gambar RGB dengan alpha ground-truth satu saluran
matte yang memiliki stem yang sama, dimana 0 adalah latar belakang dan 255 adalah latar depan.

```text
my-matte-dataset/
  images/
    subject.jpg
  mattes/
    subject.png
```

Melewati root itu sebagai `data=` sudah cukup: direktori matte terdeteksi secara otomatis
di antara `mattes/`, `matte/`, `gt/`, `masks/`, `mask/` dan `alpha/`. dataset YAML
adalah alternatifnya, dengan `path` plus penamaan `val_images` dan `val_mattes`
direktori relatif terhadapnya:

```yaml
path: my-matte-dataset
val_images: images
val_mattes: mattes
nc: 1
names: {0: matte}
```

`nc` dan `names` adalah placeholder skema; model matte mengembalikan `Results.matte`,
bukan deteksi. Nilai matte dibaca sebagai alpha di `[0, 1]` dengan membagi dengan 255,
dan sebuah matte yang bentuknya berbeda dari kanvas prediksi diubah ukurannya secara bilinear
untuk mencocokkan. Lihat [dataset formats](/docs/reference/dataset-formats) untuk selengkapnya
kontrak.

## Kereta

Tidak ada matte family yang memiliki implementasi pelatihan: `train()` memunculkan
`NotImplementedError` di kedua-duanya, dan dukungan matte mencakup prediksi, validasi
dan hanya ekspor. Setiap halaman model menyebutkan proyek hulu yang mengirimkan pelatihan
kode dan skrip konversi yang mengembalikan checkpoint.

## Validasi

`val()` menggerakkan `predict` milik model itu sendiri, jadi validasi menggunakan family yang sama persis
pra-pemrosesan, dan kedua metrik dihitung pada kanvas gambar asli.

<code-tabs name="val" />

`metrics/MAE` adalah kesalahan absolut rata-rata terhadap alpha yang sebenarnya, dalam
`[0, 1]`, dan semakin rendah semakin baik. `metrics/Smeasure` adalah S-measure dari Fan et al.
(ICCV 2017), sebuah kesamaan struktural yang menilai ketepatan bentuk subjek
dan lubangnya, yang tidak bisa hanya dinilai dengan rata-rata per-piksel; semakin tinggi semakin baik.
S-measure juga `fitness`, jumlah pembacaan terbaik-checkpoint. Tidak ada
metrik yang bergantung pada resolusi.

## Ekspor

Model matte yang diekspor dapat dimuat kembali melalui `LibreYOLO()` berdasarkan akhiran filenya, sehingga
artefak berperilaku seperti checkpoint dan mengembalikan `Results` yang sama.

<code-tabs name="export" />

TorchScript adalah jalur yang tervalidasi untuk task ini. Konversi ONNX berjalan tetapi memiliki
belum melewati batang paritas yang sama, dan format yang tersisa tidak tersedia.
Cakupan per format ada di [BiRefNet](/docs/models/birefnet) dan
halaman [FeyNobg](/docs/models/feynobg) dan di
[matriks ekspor penuh](/docs/reference/export-matrix).


