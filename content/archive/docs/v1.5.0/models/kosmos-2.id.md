---
title: Kosmos-2
families:
  - kosmos2
seo_title: 'Kosmos-2 di LibreYOLO: deteksi objek grounded'
description: >-
  Kosmos-2 di LibreYOLO: instal, tetapkan vocabulary terbuka, dan prediksi box
  grounded dengan model berlisensi MIT milik Microsoft.
lead: >-
  Kosmos-2 adalah model grounding milik Microsoft: model ini membuat caption
  gambar, lalu menemukan setiap frasa nomina dalam caption tersebut dengan
  sebuah box. LibreYOLO membungkusnya sebagai detector objek open-vocabulary:
  berikan daftar kelas saat prediksi.
keywords:
  - Kosmos-2
  - vision-language model
  - grounding
  - open-vocabulary detection
  - Microsoft
  - VLM
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("kosmos-2")
        model.set_classes(["boat", "person"])
        result = model.predict(SAMPLE_IMAGE, save=True)

        for box in result.boxes:
            print(box.cls, box.conf, box.xyxy)
    - label: Video
      language: python
      code: >
        from libreyolo import LibreVLM


        model = LibreVLM("kosmos-2")

        model.set_classes(["boat", "person"])


        # Semua sumber yang diterima library: berkas, folder, URL, indeks
        webcam,

        # stream RTSP, atau daftar .streams

        for result in model.predict("clip.mp4", stream=True, save=True):
            print(len(result.boxes))
source_hash: 60e0796f34be6d59
---

## Instalasi

Kosmos-2 termasuk dalam tingkat VLM-sebagai-detector LibreYOLO, permukaan
produk yang terpisah dari family berbasis checkpoint dan memiliki factory
sendiri. Model ini memerlukan extra `vlm`.

```bash
pip install "libreyolo[vlm]"
```

## Prediksi

Bobot diunduh dari Hugging Face saat pertama kali digunakan dan disimpan dalam
cache lokal. LibreYOLO memuat repository `microsoft/kosmos-2-patch14-224` milik
Microsoft secara langsung. Tidak seperti Florence-2, unggahan ulang komunitas
tidak diperlukan di sini.

<code-tabs name="predict" />

Family ini dimuat melalui factory `LibreVLM()`, bukan `LibreYOLO()`. Family VLM
tidak mendeklarasikan loader checkpoint, sehingga perutean berdasarkan akhiran
berkas yang dijelaskan pada halaman model lain tidak berlaku di sini.
`set_classes()` menetapkan vocabulary yang diminta untuk ditemukan Kosmos-2.
Pengaturan ini persisten dan tetap berlaku pada setiap panggilan
`predict()`/`track()` berikutnya sampai ditetapkan kembali. Kosmos-2 melakukan
grounding pada frasa nomina, bukan mencocokkan label secara persis, sehingga
wrapper LibreYOLO menerima kecocokan parsial. Kelas bernama `"boat"` juga cocok
dengan frasa yang dihasilkan seperti "the boats". Setiap deteksi memiliki
confidence placeholder yang sama, sehingga filter `conf` berlaku untuk semua
atau tidak satu pun, bukan sebagai peringkat, dan `iou` tidak berpengaruh di
sini karena wrapper membangun daftar deteksi langsung dari entitas grounded
tanpa langkah deduplikasi. `chat()` memunculkan `NotImplementedError` karena
Kosmos-2 digerakkan oleh prompt `<grounding>`, bukan template chat. CLI
LibreYOLO tidak mencakup tingkat ini: tidak ada bentuk
`libreyolo predict model=...` untuknya. Lihat [prediksi](/docs/predict) untuk
sumber, streaming, dan penanganan hasil.

## Varian

Ada satu ukuran, `kosmos-2-patch14-224`, pada 224 px, yang dimuat sebagai
`LibreVLM("kosmos-2")`. Ini adalah model dari era 2023, dan wrapper LibreYOLO
sendiri mencatat bahwa grounding-nya lebih kasar daripada detector yang lebih
baru dalam tingkat ini.

LibreYOLO tidak melatih, memvalidasi, atau mengekspor Kosmos-2: `train()`,
`val()`, dan `export()` semuanya memunculkan `NotImplementedError` untuk setiap
family dalam tingkat ini (lihat tingkat dukungan di atas). Lakukan fine-tuning
Kosmos-2 di upstream dan muat bobot hasilnya jika memerlukan vocabulary khusus
yang sudah tertanam. Periksa output `predict()` secara visual alih-alih melalui
validasi bergaya COCO karena setiap deteksi memakai confidence placeholder yang
sama.

## Lisensi

<provenance-box></provenance-box>

## Sitasi

<citation-block />
