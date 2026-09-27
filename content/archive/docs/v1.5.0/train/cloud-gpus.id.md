---
title: Pelatihan pada GPU yang disewa
seo_title: Melatih LibreYOLO pada GPU cloud yang disewa
description: >-
  Jalankan pekerjaan pelatihan LibreYOLO pada GPU yang disewa atau tanpa server:
  siapkan data, instal, luncurkan, tonton langsung, ambil bobot, dan hentikan
  pembayaran.
lead: >-
  GPU yang disewa mengubah latihan menjadi pekerjaan dengan awal, akhir, dan
  tagihan. Pekerjaannya sama seperti melatih secara lokal; yang berubah adalah
  memasukkan data, menonton dari luar, mengambil bobot, dan mematikan mesin.
keywords:
  - pelatihan gpu cloud
  - sewa gpu
  - pelatihan vast.ai
  - gpu serverless modal
  - beam gpu
  - pelatihan jarak jauh
  - staging hugging face dataset
  - biaya gpu per epoch
last_verified: 1.5.0
snippets:
  install:
    - label: Di kotak
      language: bash
      code: >
        pip install libreyolo


        # Tambahkan hanya tambahan yang dibutuhkan oleh jalannya. rfdetr untuk
        pelatihan RF-DETR,

        # lora untuk parameter-efisien fine-tuning, onnx untuk ekspor
        setelahnya.

        pip install "libreyolo[rfdetr,lora]"
    - label: Periksa GPU sebelum hal lainnya
      language: python
      code: >
        import torch


        print(torch.__version__, torch.cuda.is_available())

        print(torch.cuda.get_device_name(0))


        # Sebuah wheel yang dibuat untuk arsitektur lain melaporkan True dan
        kemudian gagal

        # pada kernel nyata pertama, jadi jalankan satu.

        x = torch.rand(2000, 2000, device="cuda")

        print(float((x @ x).sum()))
  stage:
    - label: 'Kemasi dan unggah satu kali, dari mesin Anda'
      language: bash
      code: >
        tar cf my-dataset.tar my-dataset/

        huggingface-cli upload my-org/my-dataset my-dataset.tar --repo-type
        dataset
    - label: Tahapkan di bounding box
      language: python
      code: |
        import tarfile

        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            "my-org/my-dataset", "my-dataset.tar", repo_type="dataset"
        )
        with tarfile.open(path) as archive:
            archive.extractall("/root/data")
  launch:
    - label: 'Terpisah, sehingga pekerjaan bertahan dari putusnya koneksi'
      language: bash
      code: |
        nohup libreyolo train \
          model=LibreYOLO9s.pt \
          data=/root/data/my-dataset/data.yaml \
          epochs=100 batch=-1 imgsz=640 \
          project=/root/runs name=run1 \
          > /root/train.log 2>&1 &
    - label: 'Multi-GPU, dari berkas Python'
      language: python
      code: |
        from libreyolo import LibreYOLO

        if __name__ == "__main__":
            model = LibreYOLO("LibreYOLO9s.pt")
            model.train(
                data="/root/data/my-dataset/data.yaml",
                epochs=100,
                batch=64,          # global batch di semua GPU
                device="0,1,2,3",
                project="/root/runs",
                name="run1",
            )
  watch:
    - label: Satu baca murah
      language: bash
      code: |
        cat /root/runs/run1/status.json
    - label: Dari skrip
      language: python
      code: |
        import json

        with open("/root/runs/run1/status.json") as handle:
            status = json.load(handle)

        print(status["state"], status["current_epoch"], status["eta_seconds"])
        print(status.get("metrics"))
    - label: 'Di browser, melalui terowongan SSH'
      language: bash
      code: |
        # Pada bounding box (mengikat 127.0.0.1:8420 secara default):
        libreyolo monitor /root/runs/run1 --no-browser

        # Dari mesin Anda, lalu buka http://localhost:8420 secara lokal:
        #   ssh -L 8420:localhost:8420 <user>@<host>
  push:
    - label: Dorong beban ke tempat yang permanen
      language: bash
      code: |
        huggingface-cli upload my-org/my-run \
          /root/runs/run1/weights/best.pt best.pt
source_hash: 75d314de06aca3b6
---

## Sebelum Anda menyewa apa pun

Dua keputusan akan lebih mahal nanti daripada sekarang.

Dapatkan dataset ke dalam CDN dulu. Mengemasnya sebagai tar tunggal di Hugging Face
repositori dataset bekerja sama di setiap penyedia, melayani cepat untuk semuanya,
dan tidak membutuhkan apa pun kecuali `HF_TOKEN` di lingkungan pekerjaan ketika repositori
bersifat pribadi. Menyalin dataset dari koneksi rumah, atau menariknya dari
asal yang lambat di kotak, akan dihitung sebagai waktu GPU yang dihabiskan menunggu.

<code-tabs name="stage" />

Kemudian ukur disk. Penyedia yang menagih penyimpanan menagih berdasarkan kapasitas yang dialokasikan, bukan
kapasitas yang digunakan, dan sebuah disk tidak dapat diperkecil setelah dibuat. Tambahkan yang ditahap
data, titik pemeriksaan, dan sekitar 30 persen ruang cadangan, dan berhenti di situ.

## Pasang di kotak

<code-tabs name="install" />

Pasang PyTorch terlebih dahulu jika gambar belum memiliki build CUDA yang sesuai
kartunya, lalu LibreYOLO, jadi pip tidak menyelesaikan torchnya sendiri yang hanya untuk CPU. The
cuplikan kedua bukan upacara opsional: sebuah roda yang dibuat untuk GPU yang salah
arsitektur melaporkan `torch.cuda.is_available() == True` dan kemudian gagal pada
operasi nyata pertama dengan `CUDA error: no kernel gambar is available for execution
di perangkat`. Satu perkalian matriks menangkapnya sebelum satu jam penyiapan tidak.

Tunjuk `HF_HOME` pada penyimpanan persisten jika penyedia menawarkan volume, jadi
Unduhan checkpoint dan dataset bertahan di antara sesi.

## Peluncuran

Jalankan pekerjaan secara terpisah. Sesi interaktif yang mati dengan jaringan Anda
koneksi membawa pelatihan bersamanya.

<code-tabs name="launch" />

`batch=-1` layak digunakan di sini secara khusus, karena Anda biasanya menggunakan kartu
Anda belum pernah dilatih sebelumnya. Ini menyelidiki model dalam mode pelatihan dengan yang nyata
lalu mundur dan memilih pangkat dua terbesar yang muat, yang lebih cepat daripada
menemukan batas maksimum dengan kesalahan kehabisan memori dua puluh menit kemudian. Lihat
[Hyperparameters](/docs/train/hyperparameters).

Pada sebuah kotak multi-GPU, `device="0,1,2,3"` secara otomatis membuat satu pekerja per GPU, dan
`batch` tetap menjadi batch global di seluruh GPU tersebut. Penjaga `__main__`
wajib, karena setiap pekerja mengimpor ulang skrip. Hal itu, dan perilaku terdistribusi lainnya,
ada di [Pelatihan Multi-GPU](/docs/train/multi-gpu).

## Mengamatinya dari luar

Setiap jalankan menulis `status.json` ke dalam direktori jalankannya, ditulis ulang secara atomik setiap
epoch. Ini adalah pembacaan yang murah: beberapa ratus byte membawa status, epoch saat ini,
perkiraan waktu tiba (ETA), dan metrik terbaru, tanpa harus mengurai log.

<code-tabs name="watch" />

`metrics.jsonl` di sampingnya memiliki riwayat lengkap per-epoch, dan `train.log` memiliki
keluaran konsol. `libreyolo monitor` menyajikan dasbor browser di semua
tiga menggunakan hanya pustaka standar, jadi tidak memerlukan instalasi apapun di mesin
selain LibreYOLO itu sendiri. Akses melalui penerusan port SSH.

Tidak ada dari ini yang menyentuh proses pelatihan, jadi mereka dapat menempel pada jalannya yang sedang berlangsung, membuka kembali yang
selesai, atau memeriksa yang mengalami kegagalan.

## Ambil bobot sebelum Anda berhenti membayar

Mesin ini bisa dibuang. Dorong titik pemeriksaan pada tonggak, bukan hanya di akhir,
karena kegagalan, prapemrosesan, atau kehabisan kredit akan menyebabkan kehilangan seluruh
jalannya.

<code-tabs name="push" />

`weights/best.pt` dan `weights/last.pt` ditulis setiap epoch dan pada setiap
. `save_period=N` menambahkan snapshot `weights/epoch_<N>.pt` di atasnya, yang
adalah yang membuat dorongan tengah berjalan menjadi murah. `summary.json` dan `results.csv`, di mana
family menulis mereka, kecil dan juga layak diambil.

Sebuah callback pada `on_train_epoch_end` adalah cara yang bersih untuk mengotomatiskan dorongan. Lihat
[Pencatat percobaan](/docs/train/loggers), di mana backend yang dihosting juga memberikan
Anda metrik tanpa menyentuh kotak sama sekali.

## Berhenti membayar

Inilah bagian yang menelan biaya nyata ketika salah, dan aturannya berbeda
menurut model penyedia.

Di pasar tempat Anda menyewa mesin mentah, penagihan berjalan berdasarkan jam dinding sampai
instansinya dihancurkan. GPU yang menganggur ditagih persis seperti yang sibuk, jadi membunuh
proses pelatihan tidak menyimpan apa pun dengan sendirinya. Sebuah instance yang dihentikan tetap menagih
disk.

Pada platform tanpa server di mana pekerjaan adalah fungsi yang dihias, kontainer
mengurangi menjadi nol ketika fungsi kembali, sehingga kotak yang terlupakan jauh lebih kecil kemungkinannya.
Pekerjaan yang macet tanpa batas waktu tetap dikenai biaya, jadi selalu tetapkan satu.

Berhenti alih-alih menghancurkan adalah tuas yang nyata, dan jebakan yang nyata. Diukur pada sebuah
menyewa 8x RTX 4090 dengan disk 250 GB pada 2026-07-31: berjalan ditagih $3,4828 per
jam, dihentikan ditagih $0,0694 per jam hanya untuk disk, dan dihancurkan ditagih
tidak ada. Itu adalah penghematan 98 persen sambil menjaga lingkungan, yang bertahap
data dan titik pemeriksaan yang ada.

Tarif berhenti adalah aritmetika yang bisa Anda lakukan sebelum menyewa:

```text
stopped $/hr = allocated_GB * storage_cost_per_GB_per_month / 730
             = 250 * 0.20 / 730 = $0.0694/hr
```

Bandingkan dengan biaya membangun ulang: menyewa lagi, menarik gambar,
menginstal, dan menata ulang data. Di kotak yang sama, pembangunan ulang sekitar 15
menit pengaturan ditambah 43 GB transfer masuk, sekitar $1,00 semuanya. Dibandingkan
dengan $0,0694 per jam, kembali dalam sekitar 14 jam lebih menguntungkan untuk berhenti dan
jeda yang lebih lama lebih menguntungkan untuk menghancurkan dan membangun ulang dari salinan yang ditata.

Satu risiko membuat berhenti tidak aman untuk perangkat keras yang langka: berhenti melepaskan GPU.
Tidak ada yang memesan mereka, jadi memulai ulang hanya berhasil jika host masih memilikinya
gratis. Disk Anda aman; GPU Anda tidak.

## Tanpa server, sebagai sebuah fungsi

Jika Anda lebih memilih untuk tidak mengelola mesin, baik Modal maupun Beam menjalankan versi yang dihias
Fungsi Python pada GPU dan skala ke nol ketika ia mengembalikan. Milik LibreYOLO sendiri
suite pengujian malam dijalankan di Modal, dan `tools/ci/modal_nightly.py` di perpustakaan
repositori adalah contoh in-repo yang sedang digunakan untuk disalin.

```python
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")   # Perpustakaan sistem OpenCV
    .pip_install("libreyolo[rfdetr]")
)
app = modal.App("libreyolo-train")
cache = modal.Volume.from_name("libreyolo-cache", create_if_missing=True)


@app.function(gpu="A100", timeout=6 * 60 * 60, volumes={"/cache": cache})
def train():
    import os

    os.environ["HF_HOME"] = "/cache/hf"          # menyimpan bobot di seluruh percobaan

    from libreyolo import LibreYOLO

    model = LibreYOLO("LibreYOLO9s.pt")
    model.train(data="coco8.yaml", epochs=100, project="/cache/runs")
    cache.commit()                                # pertahankan volume


@app.local_entrypoint()
def main():
    train.remote()
```

Jalankan dengan `modal run modal_train.py`. Sistem berkas kontainer bersifat sementara, jadi
apa pun yang layak disimpan dimasukkan ke dalam volume atau didorong keluar. Set `timeout=`
secara eksplisit; itu adalah satu-satunya hal yang memisahkan antara jalan yang macet dan yang terbuka
tagihan.

Balok mengambil bentuk yang sama dengan dekorator `@function`, `Volume`, dan
`train.remote()` menelepon dari `__main__`.

## Sesuaikan ukuran berdasarkan biaya per pekerjaan

$/jam adalah angka yang salah untuk dioptimalkan. Model kecil setengah-menganggur pada kartu besar, jadi
GPU yang lebih murah dan lebih lambat seringkali lebih murah per epoch. Jalankan profiler selama beberapa
langkah-langkah pada kartu sewaan sebelum berkomitmen untuk lari jarak jauh: jika keputusannya adalah
`dataloader` atau `host / launch`, GPU yang lebih cepat tidak membeli apa-apa dan lebih banyak pekerja atau sebuah
batch yang lebih besar membeli banyak. Lihat
[Kinerja pelatihan](/docs/train/performance).

## Terkait

- [Dataset](/docs/train/datasets) untuk tata letak arsip yang dipentaskan harus memiliki,
  dan perintah dokter yang menangkap masalah sebelum GPU ditagih.
- [Pelatihan Multi-GPU](/docs/train/multi-gpu) untuk kotak multi-kartu.


