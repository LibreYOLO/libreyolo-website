---
title: NAFNet
families:
  - nafnet
seo_title: 'NAFNet: denoise, pelatihan, dan ekspor berlisensi MIT'
description: >-
  Gunakan NAFNet di LibreYOLO untuk denoise dan restorasi gambar. Instal,
  prediksi, latih, validasi, dan ekspor checkpoint SIDD berlisensi MIT.
lead: >-
  NAFNet adalah jaringan konvolusional untuk restorasi gambar yang menghapus
  fungsi aktivasi nonlinear dari blok UNet biasa dan menggantinya dengan
  perkalian elemen demi elemen. LibreYOLO mendukungnya untuk satu task,
  restorasi, dengan checkpoint denoise gambar nyata yang dilatih pada SIDD.
keywords:
  - NAFNet
  - restorasi gambar
  - denoise gambar Python
  - image deblurring
  - nonlinear activation free network
  - SIDD
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model("noisy.jpg", save=True)

        restored = result.restored
        print(restored.array.shape)
    - label: CLI
      language: bash
      code: >
        libreyolo predict model=LibreNAFNetl-restore-sidd.pt source=noisy.jpg
        save=True
    - label: Simpan gambar hasil restorasi
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        result = model.predict("noisy.jpg")

        result.restored.save("denoised.png")
  train:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        model.train(data="my-dataset.yaml", epochs=100, imgsz=256, batch=16,
        lr0=1e-3)
    - label: CLI
      language: bash
      code: >
        libreyolo train model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
        \
          epochs=100 imgsz=256 batch=16 lr0=1e-3
    - label: Provenans checkpoint
      language: python
      code: >
        from libreyolo import LibreYOLO


        # degradation dan dataset direkam pada checkpoint yang disimpan.
        Keduanya

        # tidak mengubah apa yang dilatih.

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        model.train(
            data="my-dataset.yaml",
            epochs=100,
            degradation="denoise",
            dataset="MyDataset",
        )
    - label: Multi-GPU
      language: bash
      code: >
        libreyolo train model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
        \
          epochs=100 device=0,1 batch=32
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")

        # val() mengembalikan dict biasa, bukan objek
        metrics = model.val(data="my-dataset.yaml")

        print(metrics["metrics/PSNR"])
        print(metrics["metrics/SSIM"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreNAFNetl-restore-sidd.pt data=my-dataset.yaml
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreNAFNetl-restore-sidd.pt")
        model.export(format="onnx", imgsz=256)
        model.export(format="tensorrt", imgsz=256, half=True)
    - label: CLI
      language: bash
      code: >
        libreyolo export model=LibreNAFNetl-restore-sidd.pt format=onnx
        imgsz=256

        libreyolo export model=LibreNAFNetl-restore-sidd.pt format=tensorrt
        imgsz=256 half=True
    - label: Gunakan berkas hasil ekspor
      language: python
      code: >
        from libreyolo import LibreYOLO


        # Factory merutekan berdasarkan akhiran berkas, sehingga artefak hasil
        ekspor dimuat

        # seperti checkpoint lain dan mengembalikan objek Results yang sama.

        model = LibreYOLO("LibreNAFNetl-restore-sidd.onnx")

        result = model("noisy.jpg")


        result.restored.save("denoised.png")
source_hash: 9bae9f82bee741bf
---

## Instalasi

NAFNet tidak memerlukan extra opsional. Semua yang diimpornya tersedia dalam
instalasi dasar.

```bash
pip install libreyolo
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam
cache lokal.

<code-tabs name="predict" />

Objek `Results` yang dikembalikan memuat satu field untuk family ini, yaitu
`restored`, gambar RGB uint8 HWC padat pada kanvas asli. Tidak ada box untuk
diiterasi. `save=True` langsung menulis gambar hasil restorasi ke disk,
alih-alih menggambar anotasi di atas input. `conf`, `iou`, dan `max_det`
diterima demi paritas signature dengan semua family lain, tetapi tidak
berpengaruh karena restorasi tidak menghasilkan deteksi untuk difilter. Lihat
[prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Varian

Dua lebar memakai arsitektur ini: `s` (lebar 32) dan `l` (lebar 64), keduanya
dibangun dengan patch pelatihan 256 px. Prediksi dan validasi berjalan pada
resolusi native gambar terlepas dari ukurannya, dengan padding hanya ke faktor
downsample jaringan. Saat ini hanya lebar `l` yang dipublikasikan, sebagai
checkpoint denoise gambar nyata yang dilatih pada SIDD.

## Pelatihan

NAFNet melakukan fine-tuning pada pasangan gambar rusak/bersih milik Anda:
YAML dataset yang menunjuk ke folder gambar rusak `inputs/<split>/` dan folder
target bersih `targets/<split>/`, yang dicocokkan berdasarkan stem nama berkas.
`degradation` dan `dataset` adalah string opsional yang direkam pada checkpoint
tersimpan untuk provenans. Keduanya tidak berperan dalam pelatihan.

<code-tabs name="train" />

Jika dibiarkan, trainer berjalan selama 100 epoch dengan AdamW pada `lr0=1e-3`,
batch 16, crop 256 px, dan early stopping setelah 50 epoch tanpa peningkatan
PSNR. Tidak ada jalur LoRA untuk family ini. `lora=True` memunculkan error,
bukan menjalankan fine-tuning, karena `NAFNetTrainer` tidak pernah mengaktifkan
fine-tuning adapter.

Selama pelatihan, jaringan berjalan dengan global-average pooling biasa.
Windowed local pooling khusus inferensi NAFNet (Test-time Local Converter)
dilepas sebelum epoch pertama dan dipasang kembali setelah pelatihan selesai,
karena backpropagation melalui local pool ber-window tetap tidak akan cocok
dengan cara checkpoint digunakan saat inferensi.

Lihat [pelatihan](/docs/train) untuk dataset, augmentasi, multi-GPU, dan logger.

## Validasi

`val()` mengembalikan dictionary dengan `metrics/PSNR` dan `metrics/SSIM`, yang
dihitung dalam RGB pada seluruh kanvas valid. SSIM memakai window Gaussian 11x11
dengan sigma 1.5, sedangkan `fitness` untuk pemilihan checkpoint terbaik adalah
nilai PSNR. `data` menunjuk ke format dataset gambar berpasangan yang sama dengan
pelatihan.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Artefak hasil ekspor dapat dimuat kembali melalui `LibreYOLO()` berdasarkan
akhiran berkasnya, sehingga berkas `.onnx` atau `.engine` berperilaku seperti
checkpoint dan mengembalikan `Results` yang sama, dengan `restored` memuat
gambar output. NAFNet diekspor pada resolusi spasial tetap. `imgsz` harus habis
dibagi faktor downsample jaringan (16 untuk kedua lebar arsitektur), dan hanya
dimensi batch yang dinamis saat `dynamic=True`. Tinggi dan lebar tetap pada
waktu ekspor.

<code-tabs name="export" />

## Checkpoint

Semua berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
