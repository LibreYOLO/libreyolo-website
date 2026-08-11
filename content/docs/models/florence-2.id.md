---
title: Florence-2
families:
  - florence2
seo_title: 'Florence-2 di LibreYOLO: deteksi open-vocabulary'
description: >-
  Florence-2 di LibreYOLO: instal, tetapkan vocabulary terbuka, dan prediksi box
  dengan model vision berlisensi MIT milik Microsoft.
lead: >-
  Florence-2 adalah vision foundation model milik Microsoft yang diberi prompt
  dengan token task, bukan dijalankan melalui head deteksi tetap. LibreYOLO
  membungkusnya sebagai detector objek open-vocabulary: berikan daftar kelas
  saat prediksi.
keywords:
  - Florence-2
  - vision-language model
  - open-vocabulary detection
  - grounding
  - Microsoft
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("florence-2-base")
        model.set_classes(["car", "person", "traffic light"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Video
      language: python
      code: >
        from libreyolo import LibreVLM


        model = LibreVLM("florence-2-base")

        model.set_classes(["car", "person", "traffic light"])


        # Semua sumber yang diterima library: berkas, folder, URL, indeks
        webcam,

        # stream RTSP, atau daftar .streams

        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
source_hash: ad26d9056465d662
---

## Instalasi

Florence-2 termasuk dalam tingkat VLM-sebagai-detector LibreYOLO, permukaan
produk yang terpisah dari family berbasis checkpoint dan memiliki factory
sendiri. Model ini memerlukan extra `vlm`.

```bash
pip install "libreyolo[vlm]"
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam
cache lokal. LibreYOLO mengunduh unggahan ulang checkpoint dari
florence-community alih-alih repository asli `microsoft/Florence-2-*`. Lihat
Lisensi untuk alasannya.

<code-tabs name="predict" />

Family ini dimuat melalui factory `LibreVLM()`, bukan `LibreYOLO()`. Family VLM
tidak mendeklarasikan loader checkpoint, sehingga perutean berdasarkan akhiran
berkas yang dijelaskan pada halaman model lain tidak berlaku di sini.
`set_classes()` menetapkan vocabulary yang diminta untuk ditemukan Florence-2
dalam gambar. Pengaturan ini persisten dan tetap berlaku pada setiap panggilan
`predict()`/`track()` berikutnya sampai ditetapkan kembali. `Results` yang
dikembalikan memuat `boxes` dengan bentuk yang sama seperti family lain, tetapi
setiap deteksi memiliki confidence placeholder yang sama. Akibatnya, filter
`conf` berlaku untuk semua atau tidak satu pun, bukan sebagai peringkat, dan
`iou` tidak berpengaruh. Wrapper Florence-2 membangun daftar deteksi langsung
dari output token task yang telah di-parse, tanpa langkah deduplikasi. `chat()`
memunculkan `NotImplementedError` di sini karena Florence-2 digerakkan oleh token
task `<OPEN_VOCABULARY_DETECTION>`, bukan template chat. CLI LibreYOLO tidak
mencakup tingkat ini: tidak ada bentuk `libreyolo predict model=...` untuknya.
Lihat [prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Varian

Ada dua ukuran, Florence-2-base dan Florence-2-large, keduanya pada 768 px, yang
dimuat sebagai `LibreVLM("florence-2-base")` atau
`LibreVLM("florence-2-large")`. LibreYOLO belum memublikasikan benchmark yang
membandingkan akurasi keduanya.

LibreYOLO tidak melatih, memvalidasi, atau mengekspor Florence-2: `train()`,
`val()`, dan `export()` semuanya memunculkan `NotImplementedError` untuk setiap
family pada tingkat ini (lihat tingkat dukungan di atas). Lakukan fine-tuning
Florence-2 di upstream dan muat bobot hasilnya jika memerlukan vocabulary khusus
yang sudah tertanam. Periksa output `predict()` secara visual alih-alih melalui
validasi bergaya COCO karena setiap deteksi memakai confidence placeholder yang
sama.

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
