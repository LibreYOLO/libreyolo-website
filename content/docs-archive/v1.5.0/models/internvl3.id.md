---
title: InternVL3
families:
  - internvl3
seo_title: 'InternVL3: deteksi open-vocabulary di LibreYOLO'
description: >-
  Gunakan InternVL3 di LibreYOLO untuk deteksi objek open-vocabulary. Prediksi
  dengan label teks apa pun. Pelatihan, validasi, dan ekspor tidak didukung.
lead: >-
  InternVL3 adalah large language model multimodal native yang dirilis OpenGVLab
  dan mempelajari vision serta bahasa bersama-sama dalam satu tahap
  pre-training. LibreYOLO membungkusnya sebagai detector objek open-vocabulary:
  daftar label teks apa pun menjadi set kelas, tanpa head tetap dan tanpa
  memerlukan fine-tuning.
keywords:
  - InternVL3
  - InternVL
  - vision-language model
  - open-vocabulary detection
  - VLM
  - OpenGVLab
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreInternVL3, SAMPLE_IMAGE


        model = LibreInternVL3(size="2b")


        # Vocabulary terbuka: kata apa pun dapat digunakan, bukan head kelas
        tetap. Persisten

        # pada setiap panggilan predict()/track() berikutnya sampai ditetapkan
        kembali.

        model.set_classes(["person", "bicycle", "dog"])

        result = model(SAMPLE_IMAGE, save=True)


        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Chat mentah
      language: python
      code: |
        from libreyolo import LibreInternVL3, SAMPLE_IMAGE

        model = LibreInternVL3(size="2b")

        # Jalur langsung di bawah kemudahan deteksi: pertanyaan bebas,
        # penghitungan, atau prompt apa pun yang tidak dicakup wrapper box.
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
source_hash: 6305f020d3079d71
---

## Instalasi

InternVL3 memerlukan extra `vlm`, yang memasang `transformers` untuk backbone
template chat.

```bash
pip install "libreyolo[vlm]"
```

## Prediksi

`LibreInternVL3` adalah kelas Python, bukan checkpoint `.pt`. Model ini tidak
dimuat melalui factory `LibreYOLO()`, dan CLI `libreyolo` tidak dapat
menentukannya. Factory `LibreVLM(...)` (`from libreyolo import LibreVLM`) juga
dapat mencapai family ini berdasarkan alias, misalnya
`LibreVLM("internvl3-2b")`. Kelas yang digunakan di bawah adalah objek yang
dibangun factory tersebut. Bobot berasal dari repository Hugging Face `-hf`
milik OpenGVLab sendiri, bukan cermin LibreYOLO. Panggilan pertama mengunduh dan
menyimpannya dalam cache lokal, serta mencatat pemberitahuan lisensi satu kali
untuk bobot Qwen yang dibatasi sebelum melakukannya.

<code-tabs name="predict" />

`result.boxes` memuat deteksi yang telah di-parse seperti family lain.
Confidence merupakan placeholder. InternVL3 tidak menghasilkan skor per box,
sehingga setiap deteksi memperoleh confidence konstan yang sama, dan `conf=`
hanya membuang baris di bawah konstanta itu, bukan memberi peringkat. `iou`
membuang box nyaris duplikat dari kelas yang sama di atas overlap tertentu,
sebagai efek samping greedy decoding yang mengulang sebuah objek. Ini bukan
proses NMS per kelas. Jika `set_classes()` dilewati, vocabulary memakai nama
COCO-80 sebagai default. Lihat [prediksi](/docs/predict) untuk sumber, streaming,
dan penanganan hasil.

## Varian

Ada tiga ukuran: 1b, 2b, dan 8b. Semuanya adalah checkpoint `-hf` native milik
OpenGVLab (backbone LLM Qwen, bukan arsitektur two-tower yang dijelaskan makalah
InternVL asli). Harness benchmark LibreYOLO belum mengukur family ini, sehingga
belum ada angka akurasi yang dipublikasikan sebagai pembanding. Pilih ukuran
sesuai anggaran komputasi Anda.

LibreYOLO menyediakan family ini hanya untuk prediksi. `train()`, `val()`, dan
`export()` semuanya memunculkan `NotImplementedError`. Lakukan fine-tuning di
upstream dan muat hasilnya. Validasi dataset dilewati karena confidence
placeholder akan membuat mAP COCO menyesatkan, sedangkan ekspor berada di luar
cakupan model generatif tanpa state dict yang dapat di-trace.

## Lisensi

<provenance-box>

Kode InternVL3 sendiri berlisensi MIT, bersifat permisif, serta dapat digunakan
dalam produk komersial dan closed source. Checkpoint `-hf` yang dimuat family
ini memakai backbone LLM Qwen dan dilisensikan secara terpisah berdasarkan Qwen
License dari Alibaba Cloud: bebas digunakan, diubah, dan didistribusikan ulang
dengan persyaratan atribusi "Built with Qwen" atau "Improved using Qwen", serta
batas 100 juta pengguna aktif bulanan untuk penggunaan komersial. Di atas batas
tersebut, otorisasi Alibaba diperlukan. LibreYOLO tidak menghosting atau
mendistribusikan ulang bobot ini. `LibreInternVL3` mengunduh ukuran yang cocok
langsung dari `OpenGVLab/InternVL3-<size>-hf` di Hugging Face saat pertama kali
dijalankan, dan mencatat pemberitahuan satu kali tentang Qwen License sebelum
unduhan.

</provenance-box>

## Sitasi

<citation-block />
