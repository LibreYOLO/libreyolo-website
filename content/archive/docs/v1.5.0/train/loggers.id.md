---
title: Pencatat eksperimen
seo_title: Pencatat eksperimen dan callback di LibreYOLO
description: >-
  Kirim metrik pelatihan ke TensorBoard, MLflow, Weights & Biases, Comet,
  ClearML, Neptune atau DVCLive, dan tulis callback Anda sendiri pada empat hook
  pelatihan.
lead: >-
  Setiap family yang dapat dilatih memancarkan empat peristiwa pelatihan.
  Pencatat bawaan adalah objek callback yang mendengarkan peristiwa yang sama,
  sehingga integrasi backend dan hook kustom menggunakan satu antarmuka.
keywords:
  - pelatihan tensorboard
  - pelacakan mlflow
  - weights and biases
  - clearml
  - comet ml
  - neptune
  - dvclive
  - callback pelatihan
  - metrik pelatihan csv
  - pemantau libreyolo
last_verified: 1.5.0
snippets:
  logger:
    - label: Berdasarkan nama
      language: python
      code: |
        from libreyolo import LibreYOLO

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, loggers="tensorboard")
    - label: Instansi yang dikonfigurasi
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training import MLflowLogger

        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(
            data="coco8.yaml",
            epochs=10,
            loggers=[MLflowLogger(tracking_uri="sqlite:///mlflow.db"), "tensorboard"],
        )
  callback:
    - label: Sebuah fungsi biasa
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training import TrainEpochEvent


        def on_epoch(event: TrainEpochEvent) -> None:
            print(f"epoch {event.epoch}/{event.total_epochs} loss={event.train_loss:.4f}")


        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, callbacks=on_epoch)
    - label: Sebuah objek dengan beberapa hook
      language: python
      code: |
        from libreyolo import LibreYOLO
        from libreyolo.training import TrainEndEvent, TrainEpochEvent, TrainStartEvent


        class RunLog:
            def on_train_start(self, event: TrainStartEvent) -> None:
                print(f"{event.model_family}{event.model_size} -> {event.save_dir}")

            def on_train_epoch_end(self, event: TrainEpochEvent) -> None:
                if event.is_best:
                    print(f"new best at epoch {event.epoch}: {event.best_metric}")

            def on_train_end(self, event: TrainEndEvent) -> None:
                print(f"done in {event.total_seconds:.0f}s")


        model = LibreYOLO("LibreYOLO9s.pt")
        model.train(data="coco8.yaml", epochs=10, callbacks=RunLog())
  monitor:
    - label: Tonton sebuah run di peramban
      language: bash
      code: |
        libreyolo monitor                     # run terbaru di bawah runs/
        libreyolo monitor runs/train/exp      # sebuah run tertentu
source_hash: de035acbaed32804
---

## Nyalakan logger

`loggers=` mengambil nama yang terdaftar, sebuah instance yang dikonfigurasi, atau iterable yang mencampur
keduanya.

<code-tabs name="logger" />

Nama tidak sensitif terhadap huruf besar/kecil. Set terdaftar adalah `tensorboard`, `mlflow`,
`wandb`, `comet`, `clearml`, `neptune`, `dvclive` dan `dvc`, yang terakhir merupakan
alias untuk `dvclive`. Apa pun yang lain akan memunculkan kesalahan segera dan menampilkan nama yang valid.
Tidak ada nilai yang mengaktifkan semuanya, dan tidak ada CLI flag: `loggers=`
adalah argumen Python.

## Apa yang dicatat setiap backend

Semua backend menulis nama metrik yang sama, sehingga dasbor terlihat sama di mana pun
Anda memilihnya:

| Kunci | Nilai |
|---|---|
| `train/loss` | rata-rata pelatihan loss dari epoch |
| `train/loss/<component>` | setiap komponen loss laporan family |
| `lr/<group>` | learning rate dari setiap kelompok parameter optimizer |
| `val/<metric>` | setiap metrik validasi, dengan awalan `metrics/`-nya dihapus |
| `time/epoch_seconds` | jam dinding untuk epoch |

Langkahnya adalah epoch berbasis 1. Konfigurasi pelatihan yang sepenuhnya terselesaikan adalah
dicatat sebagai parameter saat mulai pelatihan, dan nama jalannya default ke
`<family><size>-<task>`, misalnya `yolo9s-detect`.

Pada akhir pelatihan, backend yang mendukung pengunggahan artefak `results.csv`,
`train_config.yaml` dan `summary.json` ketika ada, ditambah
`weights/best.pt` dengan `log_checkpoints=True`. TensorBoard tidak mengunggah apa pun,
karena tidak memiliki konsep artefak. Tidak ada logger yang mengunggah gambar plot validasi.

## Perilaku kegagalan

Paket backend yang hilang akan memunculkan kesalahan saat konstruksi, menyebutkan perintah instalasi,
karena meminta logger dan diam-diam tidak mendapatkan apa pun menyembunyikan bug.

Kegagalan backend selama proses berjalan melakukan hal sebaliknya. Pengecualian pertama dari
sebuah handler menonaktifkan logger tersebut untuk sisa proses, mencatatnya, merusak backend
dicatat sebagai gagal, dan pelatihan berlanjut. Server pelacakan yang mati tidak
akan merugikan pelatihan Anda.

## Backend

Masing-masing membutuhkan tambahan sendiri.

| Nama | Tambahan | Konstruktor |
|---|---|---|
| `tensorboard` | `libreyolo[tensorboard]` | `TensorBoardLogger(log_dir=None)` |
| `mlflow` | `libreyolo[mlflow]` | `MLflowLogger(tracking_uri, experiment_name, run_name, log_artifacts=True, log_checkpoints=False)` |
| `wandb` | `libreyolo[wandb]` | `WandbLogger(project, name, entity, log_checkpoints=False)` |
| `comet` | `libreyolo[comet]` | `CometLogger(project_name, workspace, name, api_key, online, log_artifacts=True, log_checkpoints=False)` |
| `clearml` | `libreyolo[clearml]` | `ClearMLLogger(project_name="LibreYOLO", task_name, tags, output_uri, log_artifacts=True, log_checkpoints=False)` |
| `neptune` | `libreyolo[neptune]` | `NeptuneLogger(project, api_token, name, run_id, tags, mode, capture_console=False, log_artifacts=True, log_checkpoints=False)` |
| `dvclive`, `dvc` | `libreyolo[dvclive]` | `DVCLiveLogger(log_dir, resume, report, save_dvc_exp=False, dvcyaml=None, monitor_system=False, log_checkpoints=False)` |

Impor kelas-kelas dari `libreyolo.training`.

Catatan khusus backend yang perlu diketahui sebelum jalankan pertama:

Berkas-event TensorBoard default ke `<save_dir>/tensorboard`. Lihat dengan
`tensorboard --logdir runs/train`.

MLflow 3.x menghentikan penggunaan local `./mlruns` berkas store dan akan menimbulkan kesalahan kecuali
`MLFLOW_ALLOW_FILE_STORE=true`. Untuk pelacakan lokal tanpa server, gunakan URI database
sebagai gantinya, seperti pada cuplikan di atas, dan baca dengan
`mlflow ui --backend-store-uri sqlite:///mlflow.db`.

Weights & Biases beralih ke environment variable `WANDB_PROJECT` dan kemudian
ke `libreyolo`. Comet beralih ke `COMET_PROJECT_NAME` dan kemudian ke `libreyolo`,
dan mengambil kredensial dari konfigurasinya sendiri; `online=False` memberikan eksperimen offline.
ClearML membuat task baru, melaporkan konfigurasi di bawah
`TrainConfig`, dan menonaktifkan pengambilan framework otomatis sehingga metrik tidak
dilaporkan dua kali. Neptune menggunakan client `neptune-scale` saat ini daripada
paket warisan, dan `mode="offline"` mencatat secara lokal.

DVCLive menulis ke `<save_dir>/dvclive`. Ini membangun pohon ringkasannya dari `/`, dan
tidak dapat menahan float di jalur yang juga merupakan induk, jadi `train/loss/box` adalah
ditulis sebagai `train/loss.box` sementara `train/loss` tetap namanya. LibreYOLO juga
mematikan default biasa DVCLive untuk menyimpan eksperimen DVC dan menulis root
`dvc.yaml`, jadi pencatat opsional tidak membuat status kontrol versi di luar jalannya
direktori; lewati `save_dvc_exp=True` atau `dvcyaml=` yang eksplisit untuk mendapatkannya kembali.

Neptune sengaja dikecualikan dari `libreyolo[all]`: kliennya yang stabil
membutuhkan protobuf di bawah 7 sementara tambahan TFLite membutuhkan protobuf 7. Pasang
`libreyolo[neptune]` dalam lingkungan tanpa tambahan TFLite.

## Menulis callback

Keempat kejadian yang sama mengendalikan semuanya.

<code-tabs name="callback" />

| Kejadian | Kapan | Membawa |
|---|---|---|
| `TrainStartEvent` | setelah pengaturan, sebelum epoch 1 | `start_epoch`, `total_epochs`, `model_family`, `model_size`, `task`, `save_dir`, `config` |
| `TrainEpochEvent` | setelah setiap epoch, pelatihan dan validasi | `epoch`, `train_loss`, `train_loss_items`, `lr`, `val_metrics`, `validated`, `is_best`, `current_metric`, `best_metric`, `best_epoch`, `epoch_seconds` |
| `TrainEndEvent` | setelah pelatihan selesai | `completed_epochs`, `final_loss`, `best_metric`, `best_epoch`, `total_seconds`, `results` |
| `TrainExceptionEvent` | jika pelatihan menghasilkan | `epoch`, `exception`, `exception_type`, `exception_message`, `elapsed_seconds` |

Sebuah callable biasa hanya menerima `TrainEpochEvent`. Sebuah objek dapat mengimplementasikan apa saja
subkumpulan dari `on_train_start`, `on_train_epoch_end`, `on_train_end` dan
`on_train_exception`; metode yang hilang dilewati.

`TrainStartEvent.config` adalah konfigurasi yang sepenuhnya terselesaikan, kwargs pengguna digabungkan
dengan default family, sebagai pemetaan hanya-baca. Peristiwa-peristiwa tersebut adalah datakelas yang dibekukan
dan pemetaannya bersifat hanya-baca, jadi sebuah callback tidak dapat mengubah jalannya dengan menulis
kepada satu.

Sebuah pengecualian yang timbul dari `on_train_start`, `on_train_epoch_end` atau
`on_train_end` menyebar dan mengakhiri jalannya. Hanya `on_train_exception` yang dijaga,
sehingga tidak dapat menutupi kegagalan asli.

Dalam pelatihan multi-GPU, callback hanya dijalankan pada rank 0. Dengan DDP otomatis
untuk spawn mereka juga harus dapat dipicklable, yang berarti kelas di tingkat modul atau
fungsi daripada sebuah closure atau lambda. Lihat
[Pelatihan Multi-GPU](/docs/train/multi-gpu).

## Apa yang ditulis setiap lari bagaimanapun

Tiga berkas mendarat di direktori run tanpa konfigurasi sama sekali, pada setiap
family:

| Berkas | Tertulis | Isi |
|---|---|---|
| `status.json` | secara atomik, setiap epoch dan saat mulai, akhir, dan kegagalan | `state` dari `running`, `completed` atau `failed`, `current_epoch`, `total_epochs`, `progress`, `eta_seconds`, `metrics` terbaru, `best_metric`, `best_epoch`, dan sebuah objek `error` saat kegagalan |
| `metrics.jsonl` | ditambahkan sekali per epoch | satu baris JSON per epoch, skema yang sama seperti `results.csv` |
| `train.log` | langsung | output konsol dari run |

`status.json` adalah pembacaan murah untuk skrip atau agen yang memeriksa run, dan
penulisan atomik berarti pembaca tidak pernah melihat berkas yang setengah ditulis.

`results.csv` dan `summary.json` terpisah dan digerakkan oleh family. Mereka ditulis
untuk YOLOv9, YOLOv9-E2E, YOLOv9-P2, YOLOv7, YOLO-NAS, RF-DETR, EC dan DINOv2, dan
tidak untuk keluarga lainnya. `results.csv` mendapatkan satu baris per epoch dengan loss
komponen, metrik validasi dan learning rate sebagai kolom, dan headernya
melebar saat kolom baru muncul. Saat dilanjutkan, itu dipangkas kembali ke baris
sebelum epoch yang dilanjutkan daripada menggandakannya.

Bersamaan dengan itu, pelatih selalu menulis `train_config.yaml` saat pengaturan dan
titik pemeriksaan di bawah `weights/`.

## Tonton jalannya secara langsung

<code-tabs name="monitor" />

`libreyolo monitor` menyediakan dasbor peramban atas berkas-berkas di atas hanya menggunakan
perpustakaan standar: grafik metrik, ekor log, dan semua gambar validasi,
menyegarkan saat lari aktif. Ini hanya-baca dan tidak pernah menyentuh
proses pelatihan, sehingga menempel pada jalannya yang sedang berlangsung, membuka kembali yang sudah selesai, atau
memeriksa yang jatuh.

## Terkait

- [Validasi dan metrik](/docs/train/validation) untuk apa arti tombol `val/`
  dan bagaimana cara menambahkan validasi loss.
- [Kinerja pelatihan](/docs/train/performance) untuk profiler, yang merupakan
  alat yang berbeda dengan pertanyaan yang berbeda.
