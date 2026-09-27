---
title: SmolVLM2
families:
  - smolvlm2
seo_title: 'SmolVLM2 di LibreYOLO: deteksi open-vocabulary'
description: >-
  SmolVLM2 di LibreYOLO: instal, tetapkan kosakata terbuka, lalu lakukan
  prediksi atau chat dengan model vision-language Apache-2.0 dari Hugging Face.
lead: >-
  SmolVLM2 adalah model vision-language kecil dari Hugging Face. LibreYOLO
  membungkusnya sebagai detektor objek open-vocabulary dan mengekspos chat
  bebasnya secara langsung: berikan daftar kelas untuk mendeteksi, atau ajukan
  pertanyaan.
keywords:
  - SmolVLM2
  - vision-language model
  - deteksi open-vocabulary
  - small multimodal model
  - Hugging Face
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("smolvlm2-500m")
        model.set_classes(["cat", "dog"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Chat
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("smolvlm2-500m")

        # Jalur keluar di balik kemudahan deteksi: pertanyaan apa pun,
        # tidak terbatas pada kueri bounding box.
        answer = model.chat(SAMPLE_IMAGE, "What is the cat doing?")
        print(answer)
source_hash: b30823b62d6347b5
---

## Instalasi

SmolVLM2 termasuk dalam tier VLM-sebagai-detektor LibreYOLO, permukaan produk yang terpisah
dari family berbasis checkpoint dan memiliki factory sendiri. Model ini memerlukan komponen
tambahan `vlm`, yang juga memasang `num2words`, dependensi prosesor SmolVLM2 sendiri.

```bash
pip install "libreyolo[vlm]"
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam cache lokal.

<code-tabs name="predict" />

Family ini dimuat melalui factory `LibreVLM()`, bukan `LibreYOLO()`: family VLM tidak
mendeklarasikan loader checkpoint, sehingga perutean sufiks berkas yang dijelaskan di halaman
model lain tidak berlaku di sini. `set_classes()` menetapkan kosakata yang diminta untuk
ditemukan SmolVLM2; pengaturannya persisten, sehingga tetap berlaku pada setiap pemanggilan
`predict()`/`track()` berikutnya sampai ditetapkan lagi. SmolVLM2 tidak memerlukan override
parser di LibreYOLO: model ini mengikuti output chat-template-plus-JSON yang sama dengan
default bersama tier, sehingga prompt deteksi dan format kotaknya tidak khusus untuk family.
Setiap deteksi membawa confidence placeholder yang sama, sehingga pemfilteran `conf` bersifat
semua-atau-tidak-sama-sekali, bukan pemeringkatan; `iou` tetap berpengaruh dengan membuang
kotak berikutnya dari kelas yang sama setelah tumpang tindihnya dengan kotak yang sudah
dipertahankan melewati ambang batas, karena generator berulang dapat menghasilkan kotak
yang hampir duplikat untuk satu objek. SmolVLM2 juga menjawab pertanyaan bebas melalui
`chat()`, yaitu jalur keluar yang sama seperti dalam dokumentasi factory `LibreVLM`.
CLI LibreYOLO tidak mencakup tier ini: tidak ada bentuk `libreyolo predict model=...`
untuknya. Lihat [prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Varian

Satu ukuran dalam registry: SmolVLM2-500M-Video-Instruct, yang dimuat sebagai
`LibreVLM("smolvlm2-500m")`. SmolVLM2 adalah detektor yang lebih lemah daripada model
grounding khusus dalam tier ini; wrapper LibreYOLO sendiri menjelaskannya sebagai demonstrasi
bahwa family baru tidak memerlukan parsing khusus agar dapat berfungsi di sini, bukan sebagai
opsi open-vocabulary terkuat.

LibreYOLO tidak melatih, memvalidasi, atau mengekspor SmolVLM2: `train()`, `val()`, dan
`export()` semuanya memunculkan `NotImplementedError` untuk setiap family dalam tier ini
(lihat tier dukungan di atas). Lakukan fine-tuning SmolVLM2 melalui upstream dan muat bobot
hasilnya jika memerlukan kosakata khusus yang tertanam; periksa output `predict()` secara
visual alih-alih memakai tahap validasi bergaya COCO karena setiap deteksi membawa confidence
placeholder yang sama.

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
