---
title: Pelatihan multi-GPU
seo_title: Pelatihan multi-GPU di LibreYOLO
description: >-
  Latih pada beberapa GPU dengan device="0,1". Cara library membuat worker DDP,
  alasan batch merupakan batch global, kapan menetapkan sync_bn, dan jalur
  torchrun.
lead: >-
  Pelatihan multi-GPU di LibreYOLO menggunakan PyTorch DistributedDataParallel:
  satu proses per GPU, masing-masing memuat replika model lengkap dan shard
  setiap batch, dengan gradien dirata-ratakan pada semua rank di setiap langkah.
keywords:
  - training PyTorch DDP
  - training multi GPU
  - torchrun nproc_per_node
  - distributed data parallel
  - syncbatchnorm
  - global batch size
  - backend NCCL Gloo
  - multi GPU Windows
last_verified: 1.5.0
snippets:
  train:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        # Guard __main__ wajib: setiap worker yang dibuat mengimpor ulang modul
        # ini, dan tanpanya pelatihan akan diluncurkan ulang secara rekursif.
        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(
                data="my-dataset.yaml",
                epochs=100,
                batch=32,     # batch global: 16 gambar per GPU pada dua GPU
                device="0,1",
            )
  torchrun:
    - label: train.py
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(data="my-dataset.yaml", epochs=100, batch=32)
    - label: Jalankan
      language: bash
      code: |
        torchrun --nproc_per_node=2 train.py
  syncbn:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreRTDETRr18.pt")
            model.train(
                data="my-dataset.yaml",
                batch=32,
                device="0,1",
                sync_bn=True,
            )
  autobatch:
    - label: Python
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            # Diperiksa sekali pada GPU 0, lalu diskalakan ke kelipatan world-size.
            model.train(data="my-dataset.yaml", batch=-1, device="0,1")
source_hash: 83c1563d68068cd0
---

## Jalankan pada dua GPU

Berikan daftar perangkat. Tidak ada hal lain yang berubah.

<code-tabs name="train" />

Jika lebih dari satu perangkat diberikan tanpa environment torchrun, `train()`
menyimpan bobot ke berkas sementara, menyelesaikan autobatch jika diminta, lalu
membuat satu proses worker per GPU dengan `torch.multiprocessing.spawn`. Setiap
worker mengimpor ulang kelas model, membangunnya dari bobot tersimpan, dan
menjalankan jalur satu-perangkat biasa karena variabel environment torchrun telah
ditetapkan di dalam worker. Checkpoint terbaik rank 0 dimuat kembali ke instance
model pemanggil setelah proses selesai.

`device` menerima `"0,1"`, `[0, 1]`, `0`, `"cuda:0"`, `"cpu"`, `"mps"`, dan
`"auto"`. Hanya daftar lebih dari satu indeks CUDA yang memicu spawn.

## Guard `__main__` wajib

Worker mengimpor ulang modul asal. Tanpa guard `if __name__ == "__main__":`,
import tersebut menjalankan ulang pemanggilan pelatihan dan setiap worker membuat
worker sendiri. Library mendeteksinya dan memunculkan error:

```text
spawn_ddp_train() was called from inside a spawned subprocess. This usually
means your script calls model.train(device=...) at the top level without a
'if __name__ == "__main__":' guard.
```

Semua yang dikirim ke worker diserialisasi dengan pickle, sehingga `callbacks=` harus dapat
diserialisasi dengan pickle. Kelas tingkat modul dapat digunakan; closure atau lambda tidak, dan
error menjelaskannya serta menunjuk logger bawaan sebagai alternatif.

## batch adalah batch global

`batch` adalah jumlah gambar per langkah optimizer pada seluruh GPU. Dataloader
setiap rank dibangun dengan `batch // world_size` dan `DistributedSampler`,
sehingga `batch=32` pada dua GPU berarti 16 gambar per GPU, bukan 32.

Batch yang tidak habis dibagi world size memunculkan error:

```text
batch=6 is the global batch and must be divisible by world_size=4: each rank
trains at batch // world_size, so this value would silently train at a
different global batch than requested. Use batch=4 or batch=8.
```

Gradien dirata-ratakan oleh DDP, sehingga loss diteruskan tanpa penskalaan.
Mengalikannya lagi dengan world size akan menaikkan learning rate efektif kira-kira
sejumlah GPU.

## Autobatch di bawah DDP

`batch=-1` berfungsi dan mengembalikan batch global yang habis dibagi world size.

<code-tabs name="autobatch" />

Pada jalur spawn, pemeriksaan berjalan di proses parent pada perangkat pertama
sebelum worker dibuat, sehingga setiap worker menerima integer konkret. Di bawah
torchrun, rank 0 memeriksa dan menyiarkan hasil sebagai satu tensor long.

Pemeriksaan mengukur kapasitas satu GPU dan mengalikannya dengan world size. Jika
`nbs` ditetapkan, batch global dibatasi ke `nbs` dan dibulatkan turun ke kelipatan
world size, sehingga penambahan GPU mengurangi langkah akumulasi, bukan mengecilkan
batch per GPU. Mekanisme pemeriksaan tersedia di
[Hyperparameter](/docs/train/hyperparameters).

## SyncBatchNorm

Di bawah DDP, lapisan BatchNorm setiap rank hanya melihat shard-nya. Jika
`batch // world_size` kecil, running statistics dapat menurunkan hasil model
dibanding proses satu GPU.

`sync_bn=True` mengubah setiap BatchNorm menjadi SyncBatchNorm agar statistik
dihitung pada batch global. Konversi hanya terjadi saat distributed aktif.

Pengaturan ini sudah aktif secara default untuk family konvolusional yang banyak
menggunakan BatchNorm: YOLOX, YOLOv7, YOLOv9 dan variannya, YOLO-NAS, PicoDet,
RTMDet, serta FOMO. Family lain default nonaktif. Jika model memiliki BatchNorm,
`sync_bn` nonaktif, dan batch per rank di bawah 16, trainer memberi warning.

<code-tabs name="syncbn" />

Tidak ada flag CLI untuk `sync_bn`; ini argumen Python.

## Menjalankan dengan torchrun

torchrun juga dapat digunakan dan tepat ketika scheduler cluster mengelola
peluncuran proses. Tulis skrip untuk satu perangkat dan biarkan torchrun
menetapkan environment rank.

<code-tabs name="torchrun" />

Jangan gabungkan keduanya. Jika environment torchrun tersedia, `device="0,1"`
tidak melakukan spawn; trainer memakai `cuda:LOCAL_RANK` dan torchrun mengelola
jumlah proses.

## Perilaku rank

Rank 0 menangani semua efek samping. Rank ini menyelesaikan direktori proses dan
menyiarkan namanya, menulis checkpoint serta artefak, dan menjalankan callback
serta logger pengguna. Rank lain berlatih dan menyumbangkan gradien.

Setiap rank memberi seed berbeda pada dataloader dan RNG augmentasi, yang
diturunkan dari `seed`, sehingga rank tidak mengambil augmentasi identik.

## Platform dan backend

Backend dipilih otomatis: NCCL jika CUDA dan NCCL tersedia, Gloo jika tidak.
NCCL tidak dibangun di Windows, sehingga Windows memakai Gloo tanpa konfigurasi.
Process group diinisialisasi dengan timeout tiga jam.

## Yang tidak berjalan di bawah DDP

- CUDA graph capture. `cuda_graph=True` mencatat satu baris dan berlatih secara
  eager. Lihat [Performa pelatihan](/docs/train/performance).
- Profiler pelatihan. `profile=True` diabaikan dengan warning.

Tidak semua family mendukung spawn otomatis. Dua puluh empat family mendukungnya,
mencakup family deteksi, classification, semantic, dan restoration yang dapat
dilatih. Family tanpa dukungan yang menerima multi-GPU memunculkan error berisi
API model dan perintah torchrun, bukan diam-diam berlatih pada satu GPU.

## Terkait

- [Hyperparameter](/docs/train/hyperparameters) untuk `batch`, `nbs`, dan resume.
- [Logger eksperimen](/docs/train/loggers) untuk batasan picklability callback.
- [GPU cloud](/docs/train/cloud-gpus) untuk menyewa mesin multi-GPU.


