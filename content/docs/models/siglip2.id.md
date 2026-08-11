---
title: SigLIP2
families:
  - siglip2
seo_title: 'SigLIP2 di LibreYOLO: klasifikasi zero-shot dan embedding'
description: >-
  Gunakan SigLIP2 di LibreYOLO untuk klasifikasi gambar zero-shot serta
  embedding gambar/teks dengan penilaian multi-label sigmoid. Tidak memerlukan
  pelatihan.
lead: >-
  SigLIP2 adalah model dual-tower yang menilai gambar terhadap prompt teks
  dengan sigmoid independen per kelas, bukan softmax bersama pada kumpulan label
  tetap. LibreYOLO mendukungnya untuk klasifikasi zero-shot serta embedding
  gambar/teks, tanpa tahap pelatihan.
keywords:
  - SigLIP2
  - SigLIP 2
  - klasifikasi zero-shot
  - image embedding
  - text embedding
  - open vocabulary
  - multilingual
  - sigmoid loss
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")
        model.set_classes(["a forklift", "an empty aisle", "a spill"])
        result = model(SAMPLE_IMAGE, save=True)

        print(model.names[result.probs.top1], float(result.probs.top1conf))
    - label: CLI
      language: bash
      code: >
        # Tanpa pemanggilan set_classes(), prediksi CLI memakai 1.000 nama kelas

        # ImageNet yang dimuat model secara default.

        libreyolo predict model=LibreSigLIP2b16-cls.pt
        source=https://raw.githubusercontent.com/LibreYOLO/libreyolo/release/libreyolo/assets/parkour.jpg
        save=True
    - label: Penilaian sigmoid multi-label
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreSigLIP2b16-cls.pt")

        model.set_classes(["a dog", "a cat", "outdoors"], multi_label=True)

        r = model(SAMPLE_IMAGE)


        # Probabilitas independen per kelas: lebih dari satu, atau tidak ada,
        dapat

        # meraih skor tinggi sekaligus. Softmax (default) justru
        menormalisasikannya

        # menjadi distribusi satu label, sesuai perilaku LibreCLIP.

        for i, name in model.names.items():
            print(name, float(r.probs.data[i]))
    - label: Embedding gambar dan teks
      language: python
      code: >
        from libreyolo import LibreYOLO, SAMPLE_IMAGE


        model = LibreYOLO("LibreSigLIP2b16-cls.pt", task="embed")

        image_embed = model(SAMPLE_IMAGE).embeddings.data

        text_embed = model.embed_text("a photo of a forklift")


        # Keduanya dinormalisasi L2, sehingga dot product biasa adalah cosine
        similarity.

        similarity = (image_embed @ text_embed.T).item()
  val:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreSigLIP2b16-cls.pt")

        # data adalah root ImageFolder dengan split train/; nama foldernya
        # menjadi prompt kelas zero-shot untuk proses ini.
        metrics = model.val(data="imagenette160")

        print(metrics["metrics/accuracy_top1"])
        print(metrics["metrics/accuracy_top5"])
    - label: CLI
      language: bash
      code: |
        libreyolo val model=LibreSigLIP2b16-cls.pt data=imagenette160
  export:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreYOLO


        model = LibreYOLO("LibreSigLIP2b16-cls.pt")

        model.set_classes(["a forklift", "an empty aisle", "a spill"])

        model.export(format="onnx")


        # Label set_classes() saat ini dan resolusi input ditanam ke dalam
        graph.

        # Ekspor ulang setelah mengubah salah satunya. multi_label harus

        # False (default) saat ekspor.
    - label: CLI
      language: bash
      code: >
        # Tidak ada pemanggilan set_classes() di sini, sehingga 1.000 kelas
        ImageNet

        # default yang dimuat model akan ditanam.

        libreyolo export model=LibreSigLIP2b16-cls.pt format=onnx
    - label: Ekspor embedding
      language: python
      code: >
        from libreyolo import LibreYOLO


        # task="embed" melakukan trace pada tower gambar saja; tidak memerlukan
        kelas.

        model = LibreYOLO("LibreSigLIP2b16-cls.pt", task="embed")

        model.export(format="onnx")
source_hash: f992655747fd8819
---

## Instalasi

SigLIP2 memerlukan komponen tambahannya sendiri, yang memasang paket SentencePiece untuk
tokenizer multibahasa miliknya.

```bash
pip install "libreyolo[siglip2]"
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam cache lokal.

<code-tabs name="predict" />

`set_classes()` adalah satu-satunya primitif yang menjadikannya pengklasifikasi
open-vocabulary: metode ini merender setiap label ke dalam semua template prompt, mengenkode
dan merata-ratakan hasilnya, lalu menyimpan matriks `[K, D]` yang dihasilkan sebagai head
pengklasifikasi dalam cache, sehingga tidak dihitung ulang per gambar. Panggil lagi kapan pun
untuk mengubah kelas. Tanpa pemanggilan tersebut, LibreSigLIP2 dimuat dengan 1.000 nama kelas
ImageNet-1k yang sudah ditetapkan.

SigLIP menilai setiap kelas secara independen: `logit = scale * (image . text) + bias`.
Secara default, kumpulan logit tersebut tetap diteruskan melalui softmax, menghasilkan
distribusi satu label yang sesuai dengan perilaku `top1`/`top5` LibreCLIP. Meneruskan
`multi_label=True` ke `set_classes()` (atau saat konstruksi) beralih ke probabilitas sigmoid
independen, sehingga lebih dari satu kelas, atau tidak ada kelas, dapat meraih skor tinggi
pada gambar yang sama. Tokenizer-nya adalah model SentencePiece multibahasa (kosakata Gemma),
sehingga nama kelas dalam bahasa selain Inggris berfungsi dengan cara yang sama.

Dengan `task="embed"`, prediksi mengembalikan satu vektor gambar yang dinormalisasi L2 per
input, bukan probabilitas kelas, sedangkan `embed_text()` mengembalikan baris teks ternormalisasi
dalam ruang vektor yang sama, sehingga dot product biasa di antara keduanya adalah cosine
similarity. `iou` tidak berpengaruh pada kedua task; tidak ada tahap NMS. Lihat
[prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Validasi

`val()` membaca nama folder kelas di bawah split `train/` ImageFolder, memanggil
`set_classes()` dengannya, lalu mengukur akurasi zero-shot top-1 dan top-5 dengan penilaian
softmax. Akurasi bergantung pada cara nama kelas dibaca sebagai prompt, bukan pada pembaruan
bobot apa pun, karena tidak ada yang perlu dilatih. Validasi hanya mencakup
`task="classify"`; `task="embed"` tidak memiliki validator dataset.

<code-tabs name="val" />

## Ekspor

<export-matrix />

Ekspor menanam status model saat ini ke dalam graph tetap. Untuk `task="classify"`, label
apa pun yang terakhir ditetapkan `set_classes()` dan resolusi saat ekspor ditanam ke lapisan
linear akhir dengan scale dan bias yang dipelajari, sehingga graph hasil ekspor menjadi
pengklasifikasi gambar `[B, K]` biasa tanpa tower teks dan tanpa tokenizer; ekspor ulang
setelah mengubah kelas atau ukuran. Ekspor dalam mode `multi_label=True` belum
diimplementasikan; kembalikan ke `False` terlebih dahulu. Ekspor `task="embed"` melakukan
trace pada tower gambar saja. Keduanya memerlukan ONNX opset 14 atau lebih tinggi, yang
ditetapkan secara default oleh exporter.

<code-tabs name="export" />

## Checkpoint

Setiap berkas bobot yang dipublikasikan untuk family ini. Keduanya dikonversi dari checkpoint
Apache-2.0 Google `siglip2-base-patch16-256` dan `siglip2-so400m-patch14-384`, bukan dari
proses pelatihan COCO apa pun.

<checkpoint-table />

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
