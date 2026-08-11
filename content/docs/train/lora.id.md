---
title: Fine-tuning LoRA
seo_title: Fine-tuning LoRA di LibreYOLO
description: >-
  Fine-tune detector transformer dengan VRAM rendah menggunakan lora=True.
  Sembilan family yang mendukungnya, resep adapter per family, dan perilaku
  checkpoint.
lead: >-
  LoRA membekukan bagian berat model yang telah dilatih sebelumnya dan melatih
  adapter low-rank kecil di sampingnya, ditambah lapisan yang harus tetap rapat.
  Di LibreYOLO, seluruh interface publiknya hanya satu boolean.
keywords:
  - fine tuning LoRA
  - parameter efficient fine tuning
  - PEFT
  - DoRA
  - training VRAM rendah
  - LoRA RF-DETR
  - LoRA D-FINE
  - merge adapter
last_verified: 1.5.0
snippets:
  install:
    - label: pip
      language: bash
      code: |
        pip install "libreyolo[lora]"
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreRFDETRs.pt")
        model.train(data="my-dataset.yaml", epochs=50, lora=True)
    - label: CLI
      language: bash
      code: |
        libreyolo train model=LibreRFDETRs.pt data=my-dataset.yaml \
          epochs=50 lora=true
  merge:
    - label: Ekspor menggabungkan adapter
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("runs/train/exp/weights/best.pt")
        model.export(format="onnx")
    - label: Gabungkan langsung
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training.lora import merge_lora_adapters

        model = LibreYOLO("runs/train/exp/weights/best.pt")
        merged = merge_lora_adapters(model.model)

        print(f"{merged} adapter layers folded into dense weights")
source_hash: 603fdddf5ec0c316
---

## Instalasi

LoRA menggunakan dependency opsional `peft`.

<code-tabs name="install" />

Tanpanya, `lora=True` memunculkan `ImportError` yang menyebutkan perintah tersebut,
bukan tanpa sengaja menjalankan fine-tuning penuh.

## Penggunaan

<code-tabs name="train" />

`lora=True` adalah seluruh interface. Rank, alpha, dropout, dan modul target
ditetapkan per family agar sesuai dengan referensi upstream masing-masing, serta
bukan pengaturan yang dihadapkan kepada pengguna.

Family yang tidak mendukung LoRA memunculkan error saat setup, bukan mengabaikan
flag:

```text
LoRA fine-tuning (lora=True) is not supported for yolo9. LoRA targets
transformer components with nn.Linear layers (e.g. RF-DETR, D-FINE, DEIM).
```

CLI menolaknya lebih awal, sebelum model dibangun, menggunakan allowlist sendiri
yang berisi sembilan family yang sama.

## Family yang didukung

RF-DETR, D-FINE, DEIM, DEIMv2, RT-DETR v1, v2, dan v4, EC, serta ConvNeXt.
Gate-nya adalah atribut `supports_lora` pada class trainer setiap family, dan CLI
memiliki allowlist yang cocok.

Cakupan task lebih sempit daripada cakupan family. D-FINE dan EC hanya mendukung
deteksi, dan jalur segment serta pose-nya memunculkan error. Jalur semantic
RF-DETR memunculkan error. ConvNeXt digunakan untuk classification.

Semua yang lain memunculkan error. Tidak ada mode parsial atau diam-diam.

## Yang dilakukan setiap resep

Resep berbeda karena arsitektur berbeda, dan resep yang bekerja pada backbone
ViT tidak memiliki tempat untuk dipasang pada backbone konvolusional.

RF-DETR menggunakan DoRA, yaitu LoRA dengan weight decomposition, pada rank 16
dan alpha 16 di proyeksi attention `query`, `key`, serta `value` milik backbone
DINOv2, sesuai referensi RF-DETR. Backbone ViT dibekukan; projector, decoder, dan
head deteksi tetap berlatih secara normal.

D-FINE, DEIM, serta RT-DETR v1, v2, dan v4 memasangkan backbone konvolusional
dengan hybrid encoder transformer dan deformable decoder, sehingga pembagiannya
bergeser. Backbone konvolusional dibekukan sepenuhnya, yang juga melewati backward
pass-nya. Blok transformer membekukan bobot dasar dan melatih adapter LoRA biasa
pada rank 16 serta alpha 16 di lapisan linear: feed-forward `linear1` dan
`linear2`, gate, serta proyeksi deformable attention. Bagian lainnya, yaitu fusion
konvolusi encoder, proyeksi input, head prediksi, dan embedding query, tetap
berlatih secara rapat.

Dua detail resep tersebut disengaja. Self-attention decoder tetap beku tanpa
adapter karena `nn.MultiheadAttention` PyTorch membaca `out_proj.weight` secara
langsung dan akan diam-diam melewati adapter yang diinjeksi. Resep ini juga
menggunakan LoRA biasa, bukan DoRA, karena beberapa lapisan linear decoder
diinisialisasi nol dan normalisasi magnitude DoRA membagi dengan norm bobot.

DEIMv2 menggunakan resep yang sama dengan lapisan feed-forward SwiGLU `w12` dan
`w3` sebagai target. Ukuran S, M, L, dan X juga memiliki backbone ViT DINOv3,
dengan base ViT dibekukan dan lapisan fused attention `qkv` mendapat adapter,
sementara pyramid konvolusi Spatial Tuning Adapter tetap berlatih sebagai analog
projector. Adapter `qkv` tersebut tetap dipasang meskipun konfigurasi menyertakan
ViT dalam keadaan beku, karena mengadaptasi backbone beku adalah tujuannya.
Ukuran di bawah S menggunakan backbone konvolusional dan resep biasa.

EC adalah DETR dengan backbone ViT yang dikelilingi pyramid projector konvolusi
yang dapat dilatih. Base ViT dibekukan dan lapisan `qkv`-nya mendapat adapter,
blok transformer menggunakan resep bersama, serta projector dan head tetap rapat.

Blok ConvNeXt memiliki MLP linear channels-last, `fc1` dan `fc2`, yang menerima
adapter biasa. Konvolusi depthwise, norm, dan parameter layer-scale dibekukan.
Head classification tetap rapat agar jumlah class kustom terus berfungsi.

Head deteksi dan classification selalu dapat dilatih di semua resep karena jumlah
class kustom memerlukan head yang dilatih dari awal.

## Checkpoint dan ekspor

`best.pt` dan `last.pt` mempertahankan tensor adapter, sehingga proses LoRA dapat
dilanjutkan atau diperiksa seperti proses lain. Memuat salah satu checkpoint itu
memerlukan extra `lora`, karena loader mengulang injeksi adapter agar key cocok.

`export()` menggabungkan adapter ke bobot rapat, sehingga artefak hasil ekspor
tidak bergantung pada `peft`. Penggabungan yang sama tersedia langsung untuk
model dalam memori.

<code-tabs name="merge" />

Setelah penggabungan, hierarki modul sepenuhnya rapat dan penggabungan kedua
merupakan no-op.

## Yang dihemat dan tidak dihemat

LoRA mengurangi memori optimizer dan gradien, serta pada family yang membekukan
backbone sepenuhnya juga melewati backward pass backbone tersebut.

Memori aktivasi tidak berubah. Aktivasi forward tetap harus dipertahankan untuk
bagian mana pun yang dapat dilatih, dan biasanya bagian inilah yang menentukan
puncak. Untuk anggaran VRAM paling ketat, turunkan juga `batch` atau `imgsz`.

## Terkait

- [Pembekuan lapisan](/docs/train/layer-freezing) untuk cara lain melatih subset
  bobot, yang berfungsi pada setiap family dan tidak memerlukan dependency extra.
  `freeze` dan `lora=True` dapat digabungkan: parameter adapter tetap dapat dilatih
  meskipun parent group backbone dibekukan.
- [Hyperparameter](/docs/train/hyperparameters) untuk `batch`, `imgsz`, dan bagian
  lain `train()`.
