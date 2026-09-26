---
title: Sitasi
seo_title: Mengutip LibreYOLO dan penulis upstream
description: >-
  Cara mengutip LibreYOLO dalam makalah dan cara mengutip penulis family model
  yang digunakan. Keduanya perlu dicantumkan dalam bagian metode yang sama.
lead: >-
  Sitasi LibreYOLO yang lengkap terdiri dari dua bagian: library dan karya
  terbitan yang mendasari family model penghasil hasil tersebut.
keywords:
  - cara sitasi libreyolo
  - bibtex libreyolo
  - libreyolo citation cff
  - sitasi model computer vision
  - referensi makalah libreyolo
last_verified: 1.5.0
source_hash: 0f3f23e4e85e38be
---

## Mengutip LibreYOLO

Repositori menerbitkan metadata sitasinya sebagai
[`CITATION.cff`](https://github.com/LibreYOLO/libreyolo/blob/release/CITATION.cff),
bukan sebagai block BibTeX. GitHub membaca berkas tersebut dan menyediakan tombol
Cite this repositori pada halaman repositori, yang menghasilkan format APA dan
BibTeX. Ambil entri dari sana, bukan mengetiknya sendiri.

Isi lengkap berkas:

```yaml
cff-version: 1.2.0
message: "If you use LibreYOLO in your research or software, please cite it as below."
title: "LibreYOLO"
type: software
authors:
  - family-names: Ceccon
    given-names: Xuban
  - name: "The LibreYOLO contributors"
license: MIT
url: "https://github.com/LibreYOLO/libreyolo"
repository-code: "https://github.com/LibreYOLO/libreyolo"
```

Berkas ini sengaja tidak memuat versi maupun tanggal rilis.
[`RELEASING.md`](https://github.com/LibreYOLO/libreyolo/blob/release/RELEASING.md)
meminta maintainer agar tidak pernah menaikkan versi, memberi tanggal, atau
mengubah judul `CITATION.cff` maupun `.zenodo.json` saat rilis, sehingga
setiap sitasi mengarah ke satu rekaman dan tidak tersebar antarversi. Sebutkan
versi yang digunakan dalam teks sendiri dan biarkan sitasinya tetap.

## Mengutip family model

LibreYOLO adalah sebuah port. Menjalankan `LibreRFDETRm.pt` berarti menjalankan
RF-DETR, dan penulis RF-DETR adalah pihak yang seharusnya diberi kredit oleh
reviewer. Mengutip library saja akan mengatribusikan karya mereka kepada project
yang salah.

Semua yang diperlukan tersedia pada halaman family. Baris Upstream di header
menyebutkan karya asli dan organisasi di baliknya, serta menautkan paper dan
repositori sumber. Bagian Sitasi di bawahnya memuat BibTeX.

BibTeX tersebut disalin persis dari block sitasi milik penulis, biasanya bagian
Citation pada README upstream atau sebuah `CITATION.cff`. Tampilannya menyertakan
tautan kembali ke block asal agar dapat diperiksa terhadap sumber. Entri tidak
pernah dirangkai dari metadata paper. Entri yang dibuat ulang secara manual
dapat gagal tanpa terlihat dan berdampak besar: rekan penulis terlewat, venue
salah, jenis entri salah, atau tahun yang sebenarnya milik preprint. Preprint
juga dapat diterima kemudian, sehingga sebuah entri mungkin berbentuk
`@inproceedings` meskipun versi yang dibaca berada di arXiv.

Salin block sebagaimana adanya. Jika gaya bibliografi memerlukan jenis entri
lain, konversikan entri tersebut, jangan mengetiknya ulang, dan pertahankan
urutan asli list penulis.

## Hal yang diperlukan bagian metode

Tiga hal membuat hasil LibreYOLO dapat direproduksi dan diatribusikan dengan
benar:

- Library, dikutip dari `CITATION.cff`, beserta versi yang digunakan. `libreyolo version` mencetaknya bersama versi Python, torch, dan CUDA yang digunakan saat berjalan.
- Karya upstream, dikutip dari bagian Sitasi pada halaman family.
- Nama berkas checkpoint yang persis, seperti `LibreRFDETRm.pt`. Ukuran dalam satu family berperilaku berbeda, dan beberapa family menerbitkan checkpoint yang dilatih pada dataset berbeda dengan prefix yang sama. Karena itu, nama family saja tidak mengidentifikasi yang dijalankan.

Atribusi juga merupakan ketentuan lisensi bagi banyak hal yang diterbitkan
LibreYOLO. Apache-2.0 dan family CC BY sama-sama mewajibkan pemberitahuan tetap
menyertai bobot yang didistribusikan ulang. Kewajiban ini terpisah dari mengutip
paper. Lihat [lisensi](/docs/licensing) untuk mengetahui ketentuan yang berlaku
pada setiap checkpoint.

