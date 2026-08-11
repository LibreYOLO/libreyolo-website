---
title: SAM 3D Body
families:
  - sam3dbody
seo_title: 'SAM 3D Body: pemulihan mesh seluruh tubuh di LibreYOLO'
description: >-
  Gunakan SAM 3D Body di LibreYOLO untuk pemulihan mesh manusia seluruh tubuh.
  Instal dan prediksi; checkpoint dibatasi oleh SAM License dari Meta dan
  memerlukan CUDA.
lead: >-
  SAM 3D Body adalah model berbasis prompt dari Meta untuk memulihkan mesh 3D
  seluruh tubuh, termasuk tangan dan kaki, dari satu gambar dan kotak orang.
  LibreYOLO membungkus paket upstream, bukan mem-porting-nya.
keywords:
  - SAM 3D Body
  - human mesh recovery
  - body mesh
  - MHR
  - Momentum Human Rig
  - pose 3D
last_verified: 1.5.0
snippets:
  predict:
    - label: Python
      language: python
      code: >
        from libreyolo import SAMPLE_IMAGE

        from libreyolo.models.sam3dbody import LibreSAM3DBody


        # Family ini tidak terdaftar pada factory LibreYOLO(), sehingga dibuat

        # secara langsung. model_path=None memicu pengunduhan Hugging Face yang

        # dibatasi; sebuah string dianggap sebagai path checkpoint lokal yang
        ada

        # dan tidak pernah diambil secara otomatis. Inferensi memerlukan
        perangkat

        # CUDA; tidak ada jalur CPU.

        model = LibreSAM3DBody(None, size="d3", device="cuda")

        result = model(SAMPLE_IMAGE, person_boxes=[[34, 12, 220, 400]])


        meshes = result.meshes

        print(meshes.vertices.shape)    # (N, V, 3), frame kamera, meter

        print(meshes.joints3d.shape)    # (N, J, 3)
    - label: Dengan detektor orang
      language: python
      code: |
        from libreyolo import LibreYOLO, SAMPLE_IMAGE
        from libreyolo.models.sam3dbody import LibreSAM3DBody

        # Tidak ada shortcut string bernama di sini: berikan detektor LibreYOLO
        # yang sudah dibuat, callable biasa, atau instance PersonDetector.
        detector = LibreYOLO("LibreRFDETRn.pt")
        model = LibreSAM3DBody(None, size="d3", device="cuda")

        result = model(SAMPLE_IMAGE, person_detector=detector)
source_hash: 8edc8d7872f3f875
---

## Instalasi

```bash
pip install libreyolo
```

Perintah tersebut hanya memberikan adapter LibreYOLO. SAM 3D Body sendiri tidak disertakan
karena lisensinya tidak mengizinkan kode LibreYOLO diturunkan darinya: clone repositori
upstream dan pasang dependensinya sendiri, lalu arahkan LibreYOLO ke clone tersebut.

```bash
git clone https://github.com/facebookresearch/sam-3d-body
pip install roma einops yacs omegaconf braceexpand pytorch-lightning timm
```

```python
from libreyolo.models.sam3dbody import LibreSAM3DBody

model = LibreSAM3DBody(
    None,
    size="d3",
    sam_3d_body_path="/path/to/sam-3d-body",
    device="cuda",
)
```

atau tetapkan variabel lingkungan `SAM_3D_BODY_PATH` sebagai pengganti penerusan
`sam_3d_body_path` pada setiap pemanggilan. Pengguna yang tidak pernah membuat family ini
tidak akan memicu impor dan tidak akan menemui SAM License. Family ini tidak terhubung
ke factory `LibreYOLO()` atau perintah CLI `libreyolo predict`; `LibreSAM3DBody` adalah
satu-satunya entry point.

## Prediksi

<code-tabs name="predict" />

Pengunduhan checkpoint dibatasi: pengguna harus menerima lisensi Meta pada halaman model
Hugging Face dan melakukan autentikasi dengan `hf auth login` sebelum pengunduhan pertama
berhasil. Inferensi sendiri selalu memerlukan perangkat CUDA: estimator upstream memindahkan
batch ke GPU tanpa pemeriksaan, sehingga mesin khusus CPU akan memunculkan error, bukan
beralih ke jalur cadangan. `result.meshes` adalah payload `Meshes` yang barisnya sejajar
dengan `result.boxes` (satu baris per orang yang terdeteksi): `vertices` dan `joints3d`
bersifat metrik serta sudah menyertakan estimasi translasi kamera, `joints2d` berada dalam
piksel pada gambar asli, dan rotasinya mengikuti konvensi MHR, yaitu sudut Euler, bukan
axis-angle. Lihat [prediksi](/docs/predict) untuk sumber, streaming, dan penanganan hasil.

## Varian

Dua backbone di balik model tubuh MHR yang sama: `d3` memakai encoder DINOv3 ViT-H/16+,
sedangkan `h` memakai encoder ViT-H asli.

## Ekspor

<export-matrix />

Ekspor mesh tubuh belum diimplementasikan: LibreYOLO belum menentukan kontrak graph hasil
ekspor untuk task mesh, termasuk cara merepresentasikan tata letak parameter MHR di luar
PyTorch.

## Checkpoint

Setiap berkas bobot yang dipublikasikan untuk family ini.

<checkpoint-table />

## Lisensi

<provenance-box>

Model tubuh yang dijalankan checkpoint, MHR (Momentum Human Rig), adalah rilis Meta
terpisah di bawah Apache-2.0. LibreYOLO mengambil aset TorchScript dari rilis publik MHR
sendiri saat runtime dan menyimpannya dalam cache lokal; berkas tersebut tidak dicerminkan
oleh LibreYOLO dan membawa ketentuan Apache-2.0-nya sendiri, bukan SAM License.

</provenance-box>

## Sitasi

<citation-block />
