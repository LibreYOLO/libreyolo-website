---
title: Pembekuan lapisan
seo_title: Bekukan lapisan selama pelatihan di LibreYOLO
description: >-
  Bekukan sebagian model untuk transfer learning: jumlah integer freeze group
  family, daftar indeks eksplisit, atau selector nama modul dan parameter.
lead: >-
  Pembekuan mempertahankan bobot terpilih tetap sama saat bagian model lainnya
  berlatih. Selector mengacu pada freeze group terurut atau nama modul milik
  family, bukan nomor lapisan mentah dari graph YAML.
keywords:
  - membekukan lapisan model
  - transfer learning
  - freeze backbone
  - frozen batchnorm
  - freeze group
  - fine tune head saja
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")

        # Sepuluh group pertama adalah seluruh backbone YOLOv9.
        model.train(data="my-dataset.yaml", epochs=50, freeze=10)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreYOLO9s.pt data=my-dataset.yaml \
          epochs=50 freeze=10
    - label: Berdasarkan nama
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, freeze="backbone")
    - label: Beberapa selector
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="my-dataset.yaml", freeze=["backbone", "neck"])
  groups:
    - label: Cantumkan freeze group family secara berurutan
      language: python
      code: |
        from libreyolo import LibreYOLO9
        from libreyolo.models.yolo9.trainer import YOLO9Trainer

        model = LibreYOLO9("LibreYOLO9s.pt", size="s")
        trainer = YOLO9Trainer(model=model.model, wrapper_model=model, size="s")

        for index, (name, _module) in enumerate(trainer.get_freeze_groups()):
            print(index, name)
source_hash: 9f1e7551af6b16fe
---

## Bekukan sesuatu

`freeze` bersifat opsional dan default-nya tanpa pembekuan.

<code-tabs name="train" />

Pembekuan berjalan setelah model dibangun dan setelah head dibangun ulang untuk
jumlah class baru, serta sebelum optimizer dibuat, sehingga optimizer hanya
menerima parameter yang dapat dilatih.

## Bentuk selector

| Nilai | Arti |
|---|---|
| `None`, `False`, `""`, `"none"` | Latih setiap parameter |
| `10` atau `"10"` | Bekukan sepuluh freeze group family pertama |
| `[0, 3, 7]` | Bekukan group berbasis nol tersebut |
| `"backbone"` | Bekukan group, modul, atau awalan parameter yang cocok |
| `["backbone", "neck"]` | Bekukan setiap selector dalam daftar |
| `["backbone", 3]` | Daftar campuran dapat digunakan |

String di-parse sebelum ditafsirkan, sehingga CLI dan konfigurasi YAML menerima
bentuk yang sama dengan Python. `freeze="[0, 3, 'head']"` di-parse sebagai daftar
literal, `freeze="backbone,neck"` dipecah pada koma, dan string desimal tunggal
menjadi jumlah.

`freeze=True` ditolak karena ambigu.

Selector nama mencocokkan nama freeze group, nama modul, atau awalan nama
parameter, dan karakter glob `*`, `?`, serta `[` dapat digunakan. Awalan `model.`
diperlakukan secara fleksibel, sehingga `backbone` dan `model.backbone` sama-sama
mengenai ejaan mana pun yang digunakan family secara internal.

## Group didefinisikan family

Integer mengacu pada daftar freeze group terurut milik family, bukan posisi pada
graph bersama. Tidak semua family LibreYOLO merupakan satu model sequential
berindeks YAML, sehingga nomor lapisan mentah akan bermakna berbeda pada tiap family.

YOLOv9 mengurutkan group dari sisi input: sepuluh tahap backbone, lalu enam tahap
neck, kemudian head. Karena itu, `freeze=10` tepat berarti backbone. `backbone`,
`neck`, dan `head` adalah selector nama stabil di atasnya.

Group RF-DETR adalah `backbone.encoder`, `backbone.projector`, `decoder`,
`queries`, `transformer.encoder_output`, dan `head`. Nama merupakan pilihan yang
lebih baik di sini karena komponen transformer tidak dipetakan ke jumlah lapisan.
`backbone` mencocokkan kedua group backbone berdasarkan awalan.

Family yang tidak mendefinisikan group semantik kembali ke default konservatif:
setiap child langsung model yang memiliki setidaknya satu parameter, dalam urutan
deklarasi. Daftar ini biasanya pendek, sehingga integer besar tidak akan
menemukan cukup group:

```text
freeze index 10 is out of range for 3 available freeze groups.
```

Untuk melihat daftar sebenarnya tanpa menebak:

<code-tabs name="groups" />

## Kegagalan ditampilkan dengan jelas

Setiap penggunaan yang salah memunculkan error, bukan melatih sesuatu yang tidak
diminta.

Selector yang tidak cocok dengan apa pun memunculkan error dan menyebutkan
selector yang gagal:

```text
freeze selector(s) matched no parameters: 'backbon'
```

Pembekuan yang tidak menyisakan apa pun untuk dilatih memunculkan error saat
pembekuan dan kembali saat optimizer dibangun:

```text
freeze would leave no trainable parameters. Use a smaller freeze value or
target a narrower module.
```

Itulah yang dilakukan `freeze="all"`, karena `all` mencocokkan setiap parameter.

Jika pembekuan berhasil, satu baris mencatat hasilnya:

```text
Layer freezing: selectors=[10], tensors=124, params=2103776, trainable=1863456/3967232
```

## BatchNorm beku berhenti diperbarui

Parameter beku masih berada dalam modul yang running statistics-nya dapat terus
berubah. Setiap modul bergaya BatchNorm yang parameternya termasuk set beku
diubah ke mode eval, dan trainer menerapkannya kembali setelah pemanggilan
`model.train()` setiap epoch, sehingga statistik tetap sama selama seluruh proses.

Perilaku ini aktif secara default dan memastikan pembekuan backbone benar-benar
membekukannya.

## Digabungkan dengan LoRA

`freeze` dan `lora=True` dapat digunakan bersama. Pada RF-DETR, DEIM, dan ConvNeXt,
parameter adapter dipertahankan agar dapat dilatih meskipun parent group-nya
dibekukan. Inilah kombinasi yang diinginkan: backbone beku dengan adapter yang
belajar di atasnya. Lihat [fine-tuning LoRA](/docs/train/lora).

## Cakupan

Ini adalah pembekuan statis yang ditentukan saat startup. Unfreezing terjadwal
dan pembekuan progresif bukan bagian interface.

## Terkait

- [Hyperparameter](/docs/train/hyperparameters) untuk bagian lain `train()`.
- [Distilasi](/docs/train/distillation) untuk cara lain memindahkan pengetahuan
  model besar ke proses pelatihan.
