---
title: LocateAnything
families:
  - locateanything
seo_title: 'LocateAnything: deteksi open-vocabulary dan penunjukan'
description: >-
  Gunakan LocateAnything di LibreYOLO untuk deteksi open-vocabulary dan
  penunjukan. Prediksi dengan label teks apa pun. Pelatihan, validasi, dan
  ekspor tidak didukung.
lead: >-
  LocateAnything adalah model grounding vision-language yang dirilis NVIDIA dan
  mendekode bounding box serta titik secara paralel, bukan satu token koordinat
  pada satu waktu. LibreYOLO membungkusnya sebagai detector dan penunjuk
  open-vocabulary: daftar label teks apa pun menjadi set kelas, tanpa head tetap
  dan tanpa memerlukan fine-tuning.
keywords:
  - LocateAnything
  - NVIDIA
  - vision-language model
  - open-vocabulary detection
  - deteksi titik
  - VLM
  - grounding
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE


        model = LibreLocateAnything(size="3b")


        # Vocabulary terbuka: kata apa pun dapat digunakan, bukan head kelas
        tetap. Persisten

        # pada setiap panggilan predict()/track() berikutnya sampai ditetapkan
        kembali.

        model.set_classes(["person", "bicycle", "dog"])

        result = model(SAMPLE_IMAGE, save=True)


        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Prompt titik
      language: python
      code: >
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE


        # task="point" mengembalikan satu titik per objek yang cocok, bukan box.

        # Beralih task pada model yang sudah dimuat dengan
        model.set_task("point").

        model = LibreLocateAnything(size="3b", task="point")

        model.set_classes(["the person closest to the camera"])

        result = model(SAMPLE_IMAGE, save=True)


        for pt in result.points:
            print(pt.cls, pt.conf, pt.xy)
    - label: Chat mentah
      language: python
      code: |
        from libreyolo import LibreLocateAnything, SAMPLE_IMAGE

        model = LibreLocateAnything(size="3b")

        # Jalur langsung di bawah kemudahan deteksi: pertanyaan bebas,
        # penghitungan, atau prompt apa pun yang tidak dicakup wrapper box.
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
source_hash: 378ea758e507a096
---

## Instalasi

LocateAnything memerlukan extra `vlm`, yang memasang `transformers` beserta
paket `decord`, `lmdb`, dan `peft` yang diimpor kode remote Hugging Face saat
model dimuat.

```bash
pip install "libreyolo[vlm]"
```

## Prediksi

`LibreLocateAnything` adalah kelas Python, bukan checkpoint `.pt`. Model ini
tidak dimuat melalui factory `LibreYOLO()`, dan CLI `libreyolo` tidak dapat
menentukannya. Factory `LibreVLM(...)` (`from libreyolo import LibreVLM`) juga
dapat mencapai family ini berdasarkan alias, misalnya
`LibreVLM("locate-anything")`. Kelas yang digunakan di bawah adalah objek yang
dibangun factory tersebut. Memuatnya akan mengunduh dan menjalankan kode model
remote milik NVIDIA dari Hugging Face, sehingga LibreYOLO mematok unduhan ke
satu revisi commit tetap alih-alih branch `main` yang dapat berubah, serta
mencatat pemberitahuan lisensi satu kali sebelum unduhan pertama.

<code-tabs name="predict" />

`result.boxes` (task `detect`) dan `result.points` (task `point`) memuat output
yang telah di-parse seperti family lain. Confidence merupakan placeholder.
LocateAnything tidak menghasilkan skor per box, sehingga setiap deteksi
memperoleh confidence konstan yang sama, dan `conf=` hanya membuang baris di
bawah konstanta itu, bukan memberi peringkat. Jika `set_classes()` dilewati,
vocabulary memakai nama COCO-80 sebagai default. Lihat
[prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Varian

Ada satu ukuran yang dipublikasikan, 3b. Dua task memakai bobot yang sama:
`detect` (default) mengembalikan box, sedangkan `task="point"` mengembalikan satu
titik per objek yang cocok dalam `result.points`. Beralih di antara keduanya
pada model yang sudah dimuat dengan `model.set_task("point")`. Harness benchmark
LibreYOLO belum mengukur family ini, sehingga belum ada angka akurasi yang
dipublikasikan sebagai pembanding.

LibreYOLO menyediakan family ini hanya untuk prediksi. `train()`, `val()`, dan
`export()` semuanya memunculkan `NotImplementedError`. Lakukan fine-tuning di
upstream dan muat hasilnya. Validasi dataset dilewati karena confidence
placeholder akan membuat mAP COCO menyesatkan, sedangkan ekspor berada di luar
cakupan model generatif tanpa state dict yang dapat di-trace.

## Lisensi

<provenance-box>

NVIDIA License mengizinkan penggunaan, reproduksi, dan modifikasi, tetapi
membatasi model dan semua turunannya hanya untuk penggunaan nonkomersial,
penelitian, atau evaluasi bagi siapa pun selain NVIDIA dan afiliasinya. Tidak
ada ambang pendapatan atau pengecualian berbayar. LocateAnything-3B juga
menggabungkan dua komponen berlisensi lain: backbone bahasa
Qwen2.5-3B-Instruct di bawah Qwen Research License dan encoder vision
MoonViT-SO-400M berlisensi MIT. LibreYOLO tidak menghosting, mencerminkan, atau
mendistribusikan ulang bagian apa pun. `LibreLocateAnything` mengunduh bobot dan
kode remote yang diperlukan langsung dari `nvidia/LocateAnything-3B` di Hugging
Face, dipatok ke satu commit tetap, saat pertama kali dijalankan.

</provenance-box>

## Sitasi

<citation-block />
