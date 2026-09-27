---
title: LFM2-VL
families:
  - lfm2vl
seo_title: 'LFM2-VL: deteksi open-vocabulary di LibreYOLO'
description: >-
  Gunakan LFM2-VL di LibreYOLO untuk deteksi objek open-vocabulary pada
  perangkat. Prediksi dengan label teks apa pun. Pelatihan, validasi, dan ekspor
  tidak didukung.
lead: >-
  LFM2-VL adalah vision-language model ringkas untuk perangkat yang dirilis
  Liquid AI. LibreYOLO membungkusnya sebagai detector objek open-vocabulary:
  daftar label teks apa pun menjadi set kelas, tanpa head tetap dan tanpa
  memerlukan fine-tuning.
keywords:
  - LFM2-VL
  - LFM2
  - Liquid AI
  - vision-language model
  - open-vocabulary detection
  - VLM
  - edge VLM
  - LibreVLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import LibreLFM2VL, SAMPLE_IMAGE


        model = LibreLFM2VL(size="450m")


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
        from libreyolo import LibreLFM2VL, SAMPLE_IMAGE

        model = LibreLFM2VL(size="450m")

        # Jalur langsung di bawah kemudahan deteksi: pertanyaan bebas,
        # penghitungan, atau prompt apa pun yang tidak dicakup wrapper box.
        text = model.chat(SAMPLE_IMAGE, "Describe the scene in one sentence.")
        print(text)
source_hash: 40237f0ecc0d2cd5
---

## Instalasi

LFM2-VL memerlukan extra `vlm`, yang memasang `transformers` untuk backbone
template chat.

```bash
pip install "libreyolo[vlm]"
```

## Prediksi

`LibreLFM2VL` adalah kelas Python, bukan checkpoint `.pt`. Model ini tidak
dimuat melalui factory `LibreYOLO()`, dan CLI `libreyolo` tidak dapat
menentukannya. Factory `LibreVLM(...)` (`from libreyolo import LibreVLM`) juga
dapat mencapai family ini berdasarkan alias, misalnya
`LibreVLM("lfm2-vl-450m")`. Kelas yang digunakan di bawah adalah objek yang
dibangun factory tersebut. Bobot berasal dari repository Hugging Face milik
Liquid AI sendiri, bukan cermin LibreYOLO. Panggilan pertama mengunduh dan
menyimpannya dalam cache lokal, serta mencatat pemberitahuan lisensi satu kali
sebelum melakukannya.

<code-tabs name="predict" />

`result.boxes` memuat deteksi yang telah di-parse seperti family lain.
Confidence merupakan placeholder. LFM2-VL tidak menghasilkan skor per box,
sehingga setiap deteksi memperoleh confidence konstan yang sama, dan `conf=`
hanya membuang baris di bawah konstanta itu, bukan memberi peringkat. `iou`
membuang box nyaris duplikat dari kelas yang sama di atas overlap tertentu,
sebagai efek samping greedy decoding yang mengulang sebuah objek. Ini bukan
proses NMS per kelas. Jika `set_classes()` dilewati, vocabulary memakai nama
COCO-80 sebagai default. Lihat [prediksi](/docs/predict) untuk sumber, streaming,
dan penanganan hasil.

## Varian

Ada dua ukuran, 450m dan 1.6b, keduanya berasal dari rilis LFM2.5-VL Liquid AI
yang dibuat untuk deployment pada perangkat. Harness benchmark LibreYOLO belum
mengukur family ini, sehingga belum ada angka akurasi yang dipublikasikan
sebagai pembanding. Pilih ukuran sesuai anggaran komputasi Anda.

LibreYOLO menyediakan family ini hanya untuk prediksi. `train()`, `val()`, dan
`export()` semuanya memunculkan `NotImplementedError`. Lakukan fine-tuning di
upstream dan muat hasilnya. Validasi dataset dilewati karena confidence
placeholder akan membuat mAP COCO menyesatkan, sedangkan ekspor berada di luar
cakupan model generatif tanpa state dict yang dapat di-trace.

## Lisensi

<provenance-box>

LFM Open License v1.0 mengizinkan penggunaan komersial, reproduksi, dan
modifikasi, tetapi hanya di bawah ambang pendapatan tahunan $10 juta. Entitas
hukum pada atau di atas ambang tersebut sama sekali tidak memperoleh lisensi
penggunaan komersial berdasarkan perjanjian ini dan harus menghubungi Liquid AI
secara langsung. Organisasi nonprofit yang memenuhi syarat dikecualikan dari
ambang tersebut untuk penggunaan nonkomersial atau penelitian. LibreYOLO tidak
menyertakan kode sumber LiquidAI karena model dimuat melalui library
`transformers` berlisensi Apache-2.0, dan tidak menghosting atau
mendistribusikan ulang bobotnya. `LibreLFM2VL` mengunduh ukuran yang cocok
langsung dari repository Hugging Face milik Liquid AI saat pertama kali
dijalankan dan mencatat pemberitahuan satu kali sebelum unduhan tersebut.

</provenance-box>

## Sitasi

<citation-block />
