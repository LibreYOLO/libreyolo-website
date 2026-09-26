---
title: Qwen3-VL
families:
  - qwen3vl
seo_title: 'Qwen3-VL di LibreYOLO: deteksi open-vocabulary'
description: >-
  Qwen3-VL di LibreYOLO: instal, tetapkan kosakata terbuka, lalu lakukan
  prediksi atau chat dengan model vision-language Apache-2.0 dari Alibaba.
lead: >-
  Qwen3-VL adalah model vision-language Alibaba dengan grounding 2D native.
  LibreYOLO membungkusnya sebagai detektor objek open-vocabulary dan mengekspos
  chat bebasnya secara langsung: berikan daftar kelas untuk mendeteksi, atau
  ajukan pertanyaan.
keywords:
  - Qwen3-VL
  - vision-language model
  - deteksi open-vocabulary
  - grounding objek
  - Alibaba
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("qwen3-vl-4b")
        model.set_classes(["forklift", "pallet", "safety vest"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Chat
      language: python
      code: >
        from libreyolo import LibreVLM, SAMPLE_IMAGE


        model = LibreVLM("qwen3-vl-4b")


        # Jalur keluar di balik kemudahan deteksi: pertanyaan apa pun,

        # tidak terbatas pada kueri bounding box.

        answer = model.chat(SAMPLE_IMAGE, "How many people are wearing a safety
        vest?")

        print(answer)
source_hash: ee225b6221d624d9
---

## Instalasi

Qwen3-VL termasuk dalam tier VLM-sebagai-detektor LibreYOLO, permukaan produk
yang terpisah dari family berbasis checkpoint dan memiliki factory sendiri. Model ini
memerlukan komponen tambahan `vlm`.

```bash
pip install "libreyolo[vlm]"
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam cache lokal.
`LibreVLM()` yang dipanggil tanpa argumen memakai Qwen3-VL-4B secara default.

<code-tabs name="predict" />

Family ini dimuat melalui factory `LibreVLM()`, bukan `LibreYOLO()`: family VLM
tidak mendeklarasikan loader checkpoint, sehingga perutean sufiks berkas yang dijelaskan
di halaman model lain tidak berlaku di sini. `set_classes()` menetapkan kosakata
yang diminta untuk ditemukan Qwen3-VL; pengaturannya persisten, sehingga tetap berlaku
pada setiap pemanggilan `predict()`/`track()` berikutnya sampai ditetapkan lagi. Setiap
deteksi membawa confidence placeholder yang sama, sehingga pemfilteran `conf` bersifat
semua-atau-tidak-sama-sekali, bukan pemeringkatan; `iou` tetap berpengaruh pada family ini,
dengan membuang kotak berikutnya dari kelas yang sama setelah tumpang tindihnya dengan
kotak yang sudah dipertahankan melewati ambang batas, karena generator berulang dapat
menghasilkan kotak yang hampir duplikat untuk satu objek. Tidak seperti Florence-2 dan
Kosmos-2, Qwen3-VL juga menjawab pertanyaan bebas melalui `chat()`, yaitu jalur keluar
yang sama seperti dalam dokumentasi factory `LibreVLM`. CLI LibreYOLO tidak mencakup
tier ini: tidak ada bentuk `libreyolo predict model=...` untuknya. Lihat
[prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Varian

Tiga ukuran: Qwen3-VL-2B-Instruct, Qwen3-VL-4B-Instruct, dan Qwen3-VL-8B-Instruct,
yang dimuat sebagai `LibreVLM("qwen3-vl-2b")`, `LibreVLM("qwen3-vl-4b")`, dan
`LibreVLM("qwen3-vl-8b")`. Ketiganya mendeklarasikan input nominal 1024 px, tetapi
smart-resize milik prosesor Qwen menentukan canvas sebenarnya yang diteruskan ke
jaringan, sehingga angka tersebut bukan resolusi operasi tetap seperti pada family
lain di situs ini. LibreYOLO belum menerbitkan benchmark yang membandingkan akurasi
ketiga ukuran tersebut.

LibreYOLO tidak melatih, memvalidasi, atau mengekspor Qwen3-VL: `train()`, `val()`, dan
`export()` semuanya memunculkan `NotImplementedError` untuk setiap family dalam tier ini
(lihat tier dukungan di atas). Lakukan fine-tuning Qwen3-VL melalui upstream dan muat
bobot hasilnya jika memerlukan kosakata khusus yang tertanam; periksa output `predict()`
secara visual alih-alih memakai tahap validasi bergaya COCO karena setiap deteksi membawa
confidence placeholder yang sama.

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
