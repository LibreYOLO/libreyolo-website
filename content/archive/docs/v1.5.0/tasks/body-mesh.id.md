---
title: Mesh tubuh
seo_title: Rekonstruksi body mesh di LibreYOLO
description: >-
  Rekonstruksi body mesh 3D parametrik per orang di LibreYOLO. Lakukan prediksi
  dari bounding box orang atau detektor, lalu baca vertex, joint, dan translasi kamera.
lead: >-
  Rekonstruksi body mesh mengubah satu gambar dan kumpulan bounding box orang menjadi
  tubuh 3D parametrik per orang: parameter bentuk dan pose, vertex berpose,
  joint 3D, serta translasi kamera yang menempatkannya di depan lensa.
keywords:
  - human mesh recovery python
  - body mesh
  - pose tubuh 3d
  - SAM 3D Body
  - MHR
  - model tubuh parametrik
  - task mesh libreyolo
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: |
        from libreyolo import SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # Family ini tidak terdaftar pada factory LibreYOLO(), sehingga dibuat
        # secara langsung. model_path=None memicu pengunduhan Hugging Face gated;
        # string diperlakukan sebagai checkpoint lokal yang sudah ada dan tidak pernah
        # diambil. Inferensi memerlukan CUDA.
        model = LibreSAM3DBody(None, size="d3", device="cuda")
        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])

        meshes = result.meshes
        print(meshes.body_model)      # parameterization yang digunakan tensor ini
        print(meshes.vertices.shape)  # (N, V, 3), frame kamera, meter
        print(meshes.joints3d.shape)  # (N, J, 3)
        print(meshes.joints2d.shape)  # (N, J, 2), piksel pada gambar sumber
    - label: Dengan detektor orang
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # person_detector menerima detektor LibreYOLO yang sudah dibuat, callable biasa,
        # atau instance PersonDetector. Tidak ada shortcut nama.
        detector = LibreYOLO("LibreYOLO9s.pt")
        model = LibreSAM3DBody(None, size="d3", device="cuda")

        result = model(SAMPLE_IMAGE, person_detector=detector)
source_hash: 31c5b44171cbcd0e
---

## Definisi

Rekonstruksi body mesh mengembalikan payload `Meshes` per gambar, dengan baris
yang diselaraskan terhadap `result.boxes`: baris `i` menjelaskan orang dalam
bounding box `i`, sama seperti kontrak keypoint task pose.

Semuanya dinyatakan dalam frame kamera gambar asli. `transl` bersifat metrik
dalam meter, dengan +z menjauhi kamera. `vertices` dan `joints3d` bersifat
metrik serta sudah menyertakan `transl`, sehingga tidak memerlukan komposisi
lanjutan. `joints2d` berada dalam piksel pada canvas gambar asli, bukan crop yang
dilihat jaringan. `faces` menyimpan topologi mesh satu kali untuk seluruh gambar,
bukan per baris, karena setiap orang menggunakannya bersama. Versi ini tidak
memiliki frame dunia atau gravitasi, dan tidak ada kolom yang diam-diam
menggantikannya.

Tata letak parameter berbeda antar body model, sehingga tidak ada bentuk tetap:
`body_model` menamai parameterization dan jumlahnya dibaca dari tensor. Untuk
`"mhr"`, Momentum Human Rig, rotasi berupa sudut Euler dalam radian, bukan
axis-angle, `body_pose` adalah vektor parameter per joint yang flat, bukan satu
triplet per joint, dan `betas` adalah koefisien identity blendshape. Skala
skeleton, pose tangan, dan ekspresi wajah berada dalam `extras`.

Kunci task kanonis adalah `mesh`. `body-mesh`, `hmr`, dan
`human-mesh-recovery` dinormalisasi ke sana.

## Model

[SAM 3D Body](/docs/models/sam-3d-body) adalah satu-satunya family untuk task
ini dan berupa wrapper, bukan port: paket `sam-3d-body` milik Meta diterbitkan
berdasarkan SAM License, yang tidak mengizinkan kode LibreYOLO diturunkan
darinya, sehingga tidak ada yang disertakan. Dua backbone berbagi body model MHR
yang sama, `d3` pada encoder DINOv3 ViT-H/16+ dan `h` pada ViT-H asli.

Tiga persyaratan berlaku sebelum prediksi pertama dan semuanya wajib.

Paket upstream diinstal oleh pengguna, bukan LibreYOLO:

```bash
git clone https://github.com/facebookresearch/sam-3d-body
pip install roma einops yacs omegaconf braceexpand pytorch-lightning timm
```

Arahkan library ke clone dengan `sam_3d_body_path=` atau variabel environment
`SAM_3D_BODY_PATH`. Pengguna yang tidak pernah membuat family ini tidak memicu
import.

Mirror checkpoint bersifat gated. Terima lisensi pada halaman model Hugging Face
dan lakukan autentikasi dengan `hf auth login`, atau pengunduhan pertama gagal.
Body model MHR sendiri merupakan rilis Apache-2.0 terpisah, diambil dari
lokasi publiknya dan disimpan dalam cache secara lokal.

Inferensi memerlukan device CUDA. Estimator upstream memindahkan batch ke GPU
tanpa pemeriksaan, sehingga tidak ada jalur CPU dan `device="cpu"` memunculkan
error.

## Prediksi

<code-tabs name="predict" />

Orang mencapai model melalui dua cara. `person_boxes` memberikan bounding box yang sudah
tersedia, khusus satu gambar: kumpulan bounding box tetap tidak dapat mengikuti orang
lintas frame video, sehingga memberikannya bersama sumber video memunculkan
error, bukan diam-diam menggunakan kembali bounding box frame pertama. `person_detector`
menerima detektor LibreYOLO yang sudah dibuat, callable, atau `PersonDetector`,
dan merupakan jalur untuk video. `focal_length` memberikan intrinsic kamera yang
diketahui; jika tidak diberikan, model memakai estimasinya sendiri, yang
dilaporkan `meshes.focal_length`.

Family ini tidak dihubungkan ke factory `LibreYOLO()` atau perintah CLI
`libreyolo predict`. `LibreSAM3DBody` adalah satu-satunya entry point. Lihat
[prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Pelatihan

Tidak ada family task ini yang berlatih dalam LibreYOLO.
`LibreSAM3DBody.train()` memunculkan error: lakukan pelatihan pada project
upstream dan muat checkpoint hasilnya di sini.

## Validasi

Tidak ada validator mesh dan `val()` memunculkan error. Benchmark yang umum
digunakan hanya berlisensi penelitian, sehingga tidak disertakan atau diambil
otomatis.

Metriknya tersedia sebagai `libreyolo.validation.mesh_metrics` untuk evaluasi
terhadap dataset yang sudah dimiliki. Fungsi menerima joint prediksi dan target,
vertex prediksi serta target opsional, lalu mengembalikan dictionary dengan kunci
seperti validator:

`metrics/mpjpe` adalah mean per-joint position error setelah menyelaraskan root
joint, sehingga menilai pose sambil mengabaikan lokasi orang dalam scene.
`metrics/pa_mpjpe` adalah nilai sama setelah alignment Procrustes penuh, yaitu
rotasi, skala seragam, dan translasi, yang menghapus error orientasi global dan
ukuran tubuh serta menyisakan pose artikulasi. `metrics/pve` adalah mean
per-vertex error di seluruh permukaan mesh setelah alignment pada centroid
vertex; berbeda dari metrik joint, nilai ini sensitif terhadap bentuk tubuh dan
hanya muncul jika kedua array vertex diberikan. Ketiganya lebih rendah lebih
baik. Input diasumsikan metrik dalam meter, dan `scale_to_mm` mengonversi hasil
ke milimeter seperti laporan literatur.

## Ekspor

Ekspor mesh belum diimplementasikan. LibreYOLO belum mendefinisikan kontrak
metadata graph hasil ekspor untuk task ini, termasuk cara membawa tata letak
parameter MHR di luar PyTorch, sehingga `export()` memunculkan error alih-alih
menghasilkan graph dengan output yang tidak dapat ditafsirkan.


