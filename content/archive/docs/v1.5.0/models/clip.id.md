---
title: CLIP
families:
  - clip
seo_title: 'CLIP di LibreYOLO: klasifikasi zero-shot dan embedding'
description: >-
  Pakai CLIP di LibreYOLO untuk klasifikasi gambar zero-shot serta embedding
  gambar dan teks. Tanpa pelatihan: set_classes() menentukan kumpulan label saat
  runtime.
lead: >-
  CLIP adalah model dua menara yang menilai sebuah gambar terhadap prompt teks,
  bukan terhadap kumpulan label tetap. LibreYOLO mendukungnya untuk klasifikasi
  zero-shot serta embedding gambar dan teks, tanpa langkah pelatihan.
keywords:
  - CLIP
  - OpenCLIP
  - zero-shot classification
  - klasifikasi gambar tanpa training
  - image embedding python
  - text embedding
  - cari gambar pakai teks
  - open vocabulary
  - LAION-2B
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreCLIPb32-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        result = model(SAMPLE_IMAGE, save=True)

        print(model.names[result.probs.top1], float(result.probs.top1conf))
    - label: CLI
      language: bash
      code: >
        # Tanpa panggilan set_classes(), predict CLI memakai 1.000 nama

        # kelas ImageNet yang dimuat model secara default.

        libreyolo predict model=LibreCLIPb32-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Embedding gambar dan teks
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")

        image_embed = model(SAMPLE_IMAGE).embeddings.data

        text_embed = model.embed_text("a photo of a forklift")


        # Keduanya ternormalisasi L2, jadi dot product biasa adalah cosine
        similarity.

        similarity = (image_embed @ text_embed.T).item()
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt")

        # data adalah root ImageFolder dengan split train/; nama foldernya
        # menjadi prompt kelas zero-shot untuk proses ini.
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreCLIPb32-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreCLIPb32-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        model.export(format="onnx")

        # Label set_classes() saat ini dan resolusi input ditanamkan ke
        # dalam graph. Lakukan ekspor ulang setelah mengubah salah satunya.
    - label: CLI
      language: bash
      code: |
        # Tidak ada panggilan set_classes() di sini, jadi ini menanamkan
        # 1.000 kelas ImageNet bawaan yang dimuat model.
        libreyolo export model=LibreCLIPb32-cls.pt format=onnx
    - label: Ekspor embedding
      language: python
      code: |
        from libreyolo import LibreYOLO

        # task="embed" hanya menelusuri menara gambar; tidak perlu kelas.
        model = LibreYOLO("LibreCLIPb32-cls.pt", task="embed")
        model.export(format="onnx")
source_hash: ac7cfd75ad6c0fa7
---

## Instalasi

CLIP membutuhkan extra tersendiri, yang ikut memasang paket yang dipakai tokenizer BPE bawaannya untuk menghasilkan token id yang persis sama.

```bash
pip install "libreyolo[clip]"
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali dipakai dan disimpan di cache lokal.

<code-tabs name="predict" />

`set_classes()` adalah satu-satunya primitif yang menjadikan ini classifier open-vocabulary: fungsi ini menyisipkan setiap label ke seluruh template prompt, melakukan encoding lalu merata-ratakan hasilnya, dan menyimpan matriks `[K, D]` yang terbentuk di cache sebagai head classifier, sehingga matriks itu tidak dihitung ulang untuk tiap gambar. Panggil lagi kapan saja untuk mengganti kelas. Tanpa panggilan itu, LibreCLIP dimuat dengan 1.000 nama kelas ImageNet-1k yang sudah tersetel.

Dengan `task="embed"`, prediksi mengembalikan satu vektor gambar ternormalisasi L2 per input, bukan probabilitas kelas, dan `embed_text()` mengembalikan baris teks ternormalisasi di ruang vektor yang sama, sehingga dot product biasa di antara keduanya adalah cosine similarity. `iou` tidak berpengaruh pada kedua task itu; tidak ada langkah NMS. Lihat [prediksi](/docs/predict) untuk sumber, streaming dan penanganan hasil.

## Validasi

`val()` membaca nama folder kelas di bawah split `train/` sebuah ImageFolder, memanggil `set_classes()` dengan nama tersebut, lalu mengukur akurasi zero-shot top-1 dan top-5. Akurasi bergantung pada seberapa baik nama kelas terbaca sebagai prompt, bukan pada pembaruan bobot apa pun, karena tidak ada yang perlu dilatih. Validasi hanya mencakup `task="classify"`; `task="embed"` tidak punya validator dataset.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Ekspor menanamkan kondisi model saat ini ke dalam graph yang tetap. Untuk `task="classify"`, label apa pun yang terakhir ditetapkan `set_classes()`, beserta resolusi pada saat ekspor, ditanamkan ke lapisan linear terakhir, sehingga graph ONNX atau TensorRT hasil ekspor adalah classifier gambar `[B, K]` biasa tanpa menara teks dan tanpa tokenizer; lakukan ekspor ulang setelah mengubah kelas atau ukurannya. Ekspor dengan `task="embed"` hanya menelusuri menara gambar. Keduanya membutuhkan ONNX opset 14 atau lebih tinggi, yang memang disetel exporter secara default.

<code-tabs name="export" />

## Checkpoint

Semua berkas bobot yang dipublikasikan untuk family ini. Keduanya dikonversi dari checkpoint OpenCLIP yang dilatih pada LAION-2B (`ViT-B-32` dan `ViT-B-16`), bukan dari pelatihan COCO mana pun.

<checkpoint-table />

Data pelatihan LAION-2B punya riwayat terdokumentasi berisi konten CSAM (Stanford Internet Observatory, Desember 2023). Sejak itu LAION merilis Re-LAION, rilis ulang yang sudah dibersihkan; jika Anda melakukan hosting ulang bobot ini lebih lanjut, pilih checkpoint turunan Re-LAION bila tersedia.

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
