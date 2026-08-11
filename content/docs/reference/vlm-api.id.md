---
title: API visi-bahasa
seo_title: 'API LibreVLM: alias, set_classes dan chat'
description: >-
  Pabrik LibreVLM, setiap alias model, kosakata set_classes yang permanen,
  set_task, jalur keluar chat, dan mengapa kepercayaan adalah placeholder.
lead: >-
  LibreVLM memuat model visi-bahasa generatif dan menggunakannya sebagai
  detektor objek. Daftar kelas adalah prompt daripada head tetap, dan model
  mengembalikan Results yang sama seperti family lain mengembalikan.
keywords:
  - LibreVLM
  - deteksi model bahasa-visual
  - Qwen3-VL
  - LFM2-VL
  - InternVL3
  - SmolVLM2
  - Florence-2
  - libreyolo chat
last_verified: 1.5.0
verification: >-
  Alias dibaca dari libreyolo/models/vlm/__init__.py; repositori, ukuran, dan
  daftar task dari modul family di bawah libreyolo/models/vlm/ ditambah
  libreyolo/models/sensenova/model.py; aturan panggilan dan raise dari
  libreyolo/models/vlm/base.py, semuanya di v1.5.0.
snippets:
  install:
    - label: bash
      language: bash
      code: |
        pip install 'libreyolo[vlm]'
  usage:
    - label: Deteksi kosakata terbuka
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("lfm2-vl-450m")
        model.set_classes(["person", "skateboard"])

        result = model.predict(SAMPLE_IMAGE)
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            print(result.names[int(cls)], box.tolist())
    - label: Ajukan pertanyaan bebas
      language: python
      code: |
        from libreyolo import LibreVLM, SAMPLE_IMAGE

        model = LibreVLM("lfm2-vl-450m")
        print(model.chat(SAMPLE_IMAGE, "How many people are in this image?"))
source_hash: 57ddac08bc4d4e05
---

## Instal

Tingkat tersebut membutuhkan tambahan `vlm`.

<code-tabs name="install" />

## Pabrik

```python
LibreVLM(model: str = "qwen3-vl-4b", **kwargs) -> LibreVLMModel
```

`model` adalah alias, bukan jalur. `**kwargs` mencapai konstruktor family,
yang mengambil `device`, `names` (kosakata awal, setara dengan memanggil
`set_classes` setelah memuat), `prompt` (menimpa prompt deteksi) dan
`max_new_tokens`. Alias yang tidak dikenal memunculkan `ValueError` yang mencantumkan setiap alias.

<code-tabs name="usage" />

## Alias

| Family | Alias | Ukuran | Berat |
|---|---|---|---|
| Qwen3-VL | `qwen3-vl`, `qwen3-vl-2b`, `qwen3-vl-4b`, `qwen3-vl-8b` | `2b`, `4b`, `8b` | `Qwen/Qwen3-VL-2B-Instruct`, `-4B-`, `-8B-` |
| LFM2-VL | `lfm2-vl`, `lfm2-vl-450m`, `lfm2-vl-1.6b` | `450m`, `1.6b` | `LiquidAI/LFM2.5-VL-450M`, `-1.6B` |
| InternVL3 | `internvl3`, `internvl3-1b`, `internvl3-2b`, `internvl3-8b` | `1b`, `2b`, `8b` | `OpenGVLab/InternVL3-1B-hf`, `-2B-hf`, `-8B-hf` |
| SmolVLM2 | `smolvlm2`, `smolvlm2-2.2b`, `smolvlm2-500m` | `2.2b`, `500m` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct`, `SmolVLM2-500M-Video-Instruct` |
| Florence-2 | `florence-2`, `florence2`, `florence-2-base`, `florence-2-large` | `base`, `large` | `florence-community/Florence-2-base`, `-large` |
| Kosmos-2 | `kosmos-2`, `kosmos2` | `224` | `microsoft/kosmos-2-patch14-224` |
| LocateAnything | `locate-anything`, `locateanything`, `locate-anything-3b`, `locateanything-3b` | `3b` | `nvidia/LocateAnything-3B` |
| SenseNova-Vision | `sensenova-vision`, `sensenova-vision-7b`, `sensenovavision` | `7b` | `LibreYOLO/SenseNovaVision7b` |
| LibreMODUS | `libremodus`, `libremodus-14b-a7b`, `modus`, `modus-14b-a7b` | `14b-a7b` | Snapshot hulu yang dipasang |

Alias default adalah `qwen3-vl-4b`. Ukuran untuk alias default masing-masing family
adalah yang tercantum pertama: `qwen3-vl` mengarah ke `4b`, `lfm2-vl` ke `450m`,
`internvl3` ke `2b`, `smolvlm2` ke `2.2b`, `florence-2` ke `base`.

`LibreVLM`, `LibreLFM2VL`, `LibreQwen3VL`, `LibreSmolVLM2`, `LibreInternVL3`,
`LibreFlorence2`, `LibreKosmos2`, `LibreLocateAnything` dan `LibreMODUS`
(juga dieja `LibreModus`) diekspor pada tingkat paket.

## Tugas

Sebagian besar keluarga hanya melayani `detect`. Dua melayani lebih banyak:

| Family | Tugas yang didukung |
|---|---|
| LocateAnything | `detect`, `point` |
| SenseNova-Vision | `detect`, `segment`, `panoptic`, `pose`, `point`, `depth`, `ocr` |

Karena task digerakkan oleh prompt daripada tertanam dalam checkpoint, itu dapat
diaktifkan pada model yang sudah dimuat:

```python
model.set_task(task: str) -> LibreVLMModel
```

task divalidasi terhadap daftar yang didukung oleh family, tetap lengket di seluruh
panggilan `predict()` dan `track()` berikutnya, dan model dikembalikan sehingga panggilan dapat
berantai.

## set_classes

```python
model.set_classes(classes: list[str]) -> LibreVLMModel
```

Menetapkan kosakata terbuka. Kata apa pun bisa digunakan, karena model diberi prompt dengan
mereka daripada dibatasi pada head yang tetap. Daftar harus tidak kosong dan
entri-entri tersebut harus unik jika dibandingkan tanpa memperhatikan huruf besar-kecil. Melewatkan sebuah
string memunculkan `TypeError`, karena itu akan menghitung menjadi satu karakter
kelas. Kosakatanya bersifat tetap: atur sekali setelah memuat dan itu akan bertahan
sampai diatur lagi.

## obrolan

```python
model.chat(image, prompt, max_new_tokens=None, color_format="auto") -> str
```

Generasi multimodal mentah: gambar dan prompt masuk, teks yang didekodekan keluar, kata demi kata.
Ini adalah pintu keluar darurat di bawah kenyamanan deteksi, untuk bentuk bebas
pertanyaan, penghitungan, atau format keluaran yang tidak dilakukan oleh pembungkus deteksi
menutupi. `max_new_tokens` kembali ke family's `MAX_NEW_TOKENS`, yang adalah
1024 pada kelas dasar. Dekoding dilakukan secara rakus dengan penalti pengulangan yang ringan.

## Kepercayaan Diri

Output yang dihasilkan tidak memiliki kepercayaan per-kotak yang terkalibrasi. Versi ini menetapkan
placeholder konstan sehingga `predict`, menggambar dan `track` berperilaku, yang membuat
Penyaringan `conf=` dan mAP bersifat lembut daripada bermakna. Inilah juga alasannya
`val()` mengangkat: COCO mAP atas skor placeholder akan menyesatkan.

## Prediksi dan lacak

Permukaan prediksi standar berlaku, dan `track()` berfungsi, jadi sebuah detektor VLM
jatuh ke dalam pipeline yang sama seperti family lainnya. Dua kebijakan tingkat kelas
berbeda dari detektor konvolusional: augmentasi saat uji dinonaktifkan,
karena augmentasi multi-skala tidak berarti untuk generator resolusi tetap,
dan prediksi berkumpulan dimatikan, karena generasi bersifat autoregresif
dan pra-pemrosesan mengembalikan pengkodean teks-dan-gambar daripada tumpukan
tensor gambar.

## Tidak didukung

`train()`, `val()` dan `export()` menimbulkan `NotImplementedError`. Lakukan pelatihan ulang
hulu dan muat bobot yang dihasilkan.

## Kode jarak jauh

Setiap family yang dikirim dimuat melalui kelas model asli, jadi LibreYOLO tidak
mengeksekusi kode repositori pihak ketiga secara default. Sebuah family yang benar-benar membutuhkan
harus memilih secara eksplisit dan menyematkan revisi snapshot; LocateAnything adalah
yang melakukannya, disematkan ke commit `c32291ca5e996f5a7a485845b4f57a233936bba0`.

LibreMODUS adalah pengecualian eksplisit terhadap skema checkpoint: aliasnya
merujuk ke direktori file upstream yang disematkan daripada LibreYOLO
, dan LibreYOLO tidak menambahkan metadata v1.0 ke dalamnya maupun memublikasikannya kembali.

