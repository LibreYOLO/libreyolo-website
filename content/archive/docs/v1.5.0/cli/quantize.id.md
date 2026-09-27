---
title: libreyolo quantize
seo_title: Referensi perintah libreyolo quantize
description: >-
  Kuantisasi checkpoint di PyTorch dari baris perintah: resep, argumen
  kalibrasi, nilai bawaan, dan family yang diterima setiap resep.
lead: >-
  Mengganti modul float pada model dengan modul terkuantisasi, mengkalibrasinya
  pada gambar tanpa label ketika resep membutuhkan statistik, lalu menyimpan
  hasilnya sebagai checkpoint PyTorch.
keywords:
  - libreyolo quantize cli
  - quantization int8 model
  - kuantisasi model fp8
  - post training quantization
  - argumen libreyolo quantize
last_verified: 1.5.0
meta:
  - label: Perintah
    value: libreyolo quantize
    mono: true
  - label: Wajib
    value: model
    mono: true
  - label: Output
    value: 'Path sumber dengan -<recipe> sebelum akhiran, misalnya LibreYOLO9s-int8.pt'
    mono: true
snippets:
  examples:
    - label: Dasar
      language: bash
      code: |
        # Mengkalibrasi pada coco128 dan menulis LibreYOLO9s-int8.pt
        libreyolo quantize model=LibreYOLO9s.pt recipe=int8
    - label: 'Hanya cast, tanpa kalibrasi'
      language: bash
      code: |
        libreyolo quantize model=LibreYOLO9s.pt recipe=fp16 calib=none \
          out=weights/LibreYOLO9s-fp16.pt
    - label: 'Kalibrasi lebih luas, lalu pemulihan'
      language: bash
      code: >
        libreyolo quantize model=LibreYOLO9s.pt recipe=int8 \
          calib=coco128.yaml samples=256 batch=16 algorithm=minmax

        # Pelatihan quantization-aware pada checkpoint terkuantisasi memulihkan
        akurasi.

        libreyolo train model=LibreYOLO9s-int8.pt data=coco8.yaml epochs=10
        lr0=0.001
source_hash: 7ae663e9f117826e
---

## Sinopsis

```bash
libreyolo quantize model=<name|path> [recipe=<recipe>] [key=value ...]
```

Argumen berupa pasangan `key=value`, dan bentuk POSIX juga berlaku, jadi
`recipe=int8` dan `--recipe int8` adalah argumen yang sama.

## Argumen

| Argumen | Default | Arti |
|---|---|---|
| `model` | | Bobot model `.pt`. Wajib |
| `recipe` | `int8` | Resep kuantisasi: `fp16`, `bf16`, `fp8`, `int8`, `w4a16`, `w4a8`, `nvfp4`, `mxfp4`, `int2` |
| `calib` | `coco128.yaml` | Gambar kalibrasi: YAML data atau nama dataset bawaan. Tanpa label, hanya forward. `none` melewati kalibrasi |
| `samples` | `128` | Jumlah maksimum gambar kalibrasi |
| `batch` | `8` | Ukuran batch kalibrasi |
| `algorithm` | `auto` | Estimasi rentang aktivasi: `auto`, yang memilih minmax, atau `minmax`, atau `percentile` |
| `out` | | Path checkpoint keluaran. Secara bawaan berupa path sumber dengan `-<recipe>` sebelum akhiran |
| `device` | `auto` | Perangkat |
| `allow_download_scripts` | `false` | Mengizinkan Python yang tertanam di blok download pada YAML dataset |
| `json` | `false` | Output JSON ke stdout |
| `quiet` | `false` | Menyembunyikan stderr |
| `help_json` | `false` | Menampilkan skema perintah sebagai JSON lalu keluar |

## Contoh

<code-tabs name="examples" />

## Catatan

### Family yang menerimanya

Kuantisasi mencakup empat family: `yolo9`, `rfdetr`, `birefnet`, dan
`feynobg`. Family lain akan keluar dengan `quantize_failed` yang membawa daftar tersebut.

### Apa yang disentuh setiap resep

`fp16` dan `bf16` adalah cast. Keduanya hanya mengubah dtype, tidak perlu
kalibrasi, dan `calib=none` adalah pengaturan yang tepat untuk keduanya.

`int8` dan `fp8` mengkuantisasi modul `Conv2d` dan `Linear`, itulah sebabnya
keduanya cocok untuk family konvolusional.

`w4a16`, `w4a8`, `nvfp4`, `mxfp4`, dan `int2` hanya mengkuantisasi `nn.Linear`,
jadi sasarannya adalah family transformer. Meminta salah satunya pada `yolo9`
akan ditolak disertai penjelasan, bukan diam-diam menghasilkan model yang tidak
terkuantisasi, karena akselerasi di bawah 8 bit di sana hanya berlaku untuk GEMM
dan konvolusinya akan tetap berada pada presisi yang lebih tinggi.

`int8`, `fp8`, `w4a8`, dan `int2` membutuhkan statistik kalibrasi untuk
aktivasinya. `int2` juga membutuhkan pelatihan pemulihan setelahnya, jadi resep
ini ditolak pada `birefnet` dan `feynobg`, yang tidak punya trainer.

Setiap family mempertahankan sejumlah modul dalam float apa pun resepnya:
lapisan pertama, head prediksi, dan pada YOLOv9 konvolusi DFL, yang merupakan
operator ekspektasi integral tetap dan tidak boleh dikuantisasi.

### Data kalibrasi bukan data pelatihan

`calib` menunjuk ke kumpulan gambar kecil tanpa label, dipakai hanya untuk
forward, guna menurunkan rentang aktivasi. Model tidak dievaluasi terhadapnya
dan labelnya tidak pernah dibaca. Nilai bawaan `coco128.yaml` diunduh dari
sebuah URL saat pertama kali dipakai, jadi tidak butuh izin tambahan; YAML
dengan skrip download Python yang tertanam membutuhkan
`allow_download_scripts=true`.

`algorithm=percentile` tersedia dan dapat menurunkan akurasi pada family
transformer, itulah sebabnya `auto` memilih minmax.

### Memulihkan akurasi

Keluarannya adalah checkpoint PyTorch biasa, jadi
[`libreyolo train`](/docs/cli/train) menerimanya secara langsung. Melatih
checkpoint terkuantisasi berarti melakukan quantization-aware training;
menambahkan `distill_model=<teacher>` menjadikannya quantization-aware
distillation.

### Output dan kode keluar

Hasilnya mencetak path yang disimpan, resep, mode eksekusi, apakah kalibrasi
dijalankan, dan jumlah modul yang ditukar per jenis. Kode keluar adalah `0` jika
berhasil, `4` jika model tidak dapat dimuat, `5` jika kuantisasi atau
penyimpanan gagal, dan `1` untuk kegagalan runtime lainnya.

Terkait: [`libreyolo export`](/docs/cli/export), yang keluar dari PyTorch dan
justru menulis artefak deployment.
