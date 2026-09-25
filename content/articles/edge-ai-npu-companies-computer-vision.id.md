---
title: "69 Perusahaan NPU Edge AI untuk YOLO: Chip dan SDK (2026)"
description: "Panduan berbasis sumber tentang 69 perusahaan edge AI dan ekosistem NPU: chip, SDK, artefak, kuantisasi, serta dukungan computer vision dan YOLO yang terdokumentasi."
date: 2026-08-15
author: Xuban
tags: [edge-ai, npu, computer-vision, yolo, hardware, ai-accelerators, amlogic, hailo, rockchip]
faq:
  - q: "Ada berapa perusahaan NPU edge AI?"
    a: "Tidak ada jumlah perusahaan yang berlaku universal karena vendor produk, pemberi lisensi IP, sensor pintar, GPU, dan ASIC otomotif privat saling tumpang tindih. Panduan non-ekshaustif ini mencatat 69 perusahaan dan ekosistem platform yang memiliki chip edge AI, platform akselerator, IP NPU berlisensi, atau silikon watchlist yang layak diperhatikan per Agustus 2026."
  - q: "Apa itu NPU?"
    a: "Neural processing unit adalah perangkat keras khusus untuk operasi jaringan saraf seperti konvolusi dan perkalian matriks. Istilah vendor mencakup NPU, AIPU, BPU, KPU, DLA, HTP, TPU, dan MLA. Model hasil kompilasinya umumnya tidak portabel antarperusahaan."
  - q: "Perusahaan NPU mana yang secara resmi mendukung model YOLO?"
    a: "Amlogic, Hailo, Axelera AI, Rockchip, Qualcomm, MediaTek, DEEPX, D-Robotics, AXERA, SOPHGO, Huawei, Cambricon, Sony IMX500 melalui repositori AI Camera resmi Raspberry Pi, STMicroelectronics, Renesas, Lattice, Himax, Microchip, dan beberapa lainnya memiliki entri model zoo, tutorial, atau hasil validasi pihak pertama untuk setidaknya satu generasi YOLO. Generasi, task, target chip, dan versi SDK yang tepat tetap penting."
  - q: "Bisakah model ONNX apa pun dijalankan di NPU apa pun?"
    a: "Tidak. ONNX adalah format pertukaran, bukan jaminan kompatibilitas perangkat keras. Compiler NPU harus mendukung setiap operator, bentuk tensor, dan tipe data di dalam graf. Bentuk statis, pemotongan graf, operasi khusus, dan fallback CPU merupakan hal umum."
  - q: "Apa NPU terbaik untuk YOLO?"
    a: "Tidak ada pemenang universal. Hailo, Rockchip, Axelera AI, DEEPX, dan Amlogic memiliki cakupan YOLO terdokumentasi yang kuat, sedangkan jangkauan perangkat Qualcomm sangat luas. Pilih berdasarkan akurasi setelah kuantisasi, latensi seluruh pipeline, daya berkelanjutan, harga, ketersediaan, akses SDK, dan dukungan sistem operasi."
  - q: "Mengapa angka TOPS NPU tidak bisa dibandingkan langsung?"
    a: "Vendor dapat menghitung presisi, operasi sparse, operasi multiply-accumulate, dan asumsi utilisasi puncak yang berbeda. TOPS tidak mencakup lalu lintas memori, operator yang tidak didukung, prapemrosesan, pascapemrosesan, dan overhead host. Bandingkan model, presisi, resolusi, akurasi, dan seluruh pipeline yang sama."
  - q: "Apa itu kalibrasi NPU?"
    a: "Kalibrasi menjalankan gambar deployment yang representatif, biasanya tanpa label, melalui model agar compiler dapat memperkirakan rentang aktivasi untuk INT8 atau format presisi rendah lainnya. Gambar acak atau tidak representatif tetap dapat menghasilkan artefak terkompilasi, tetapi merusak akurasi task."
  - q: "Apakah LibreYOLO mendukung NPU edge?"
    a: "LibreYOLO mengekspor ONNX dan beberapa format runtime, memiliki jalur compiler RKNN langsung untuk model Rockchip tertentu, serta mendokumentasikan alur kompilasi Hailo eksternal. Setiap artefak native vendor lainnya tetap memerlukan SDK vendornya dan sebaiknya disebut kandidat integrasi sampai dikompilasi, divalidasi, dan dijalankan pada perangkat keras."
---

**Tidak ada satu pasar NPU yang tunggal. Panduan non-ekshaustif ini mencatat 69 perusahaan dan ekosistem platform: 51 vendor dengan chip atau platform yang dapat digunakan, sembilan pemasok IP NPU, dan sembilan perusahaan watchlist yang diberi label jelas.** Mereka mencakup beberapa kelas akselerator yang tidak kompatibel dan hampir sama banyaknya stack compiler. Kebanyakan menjanjikan alur yang sama: ekspor model, lakukan kuantisasi, kompilasi, salin artefak proprietary ke board, lalu panggil runtime vendor. Detailnya menentukan apakah alur itu selesai dalam satu sore atau satu kuartal kerja rekayasa.

Panduan ini memetakan perusahaan yang menyediakan perangkat keras computer vision yang kredibel pada 2026. Panduan ini mencatat chip terkait, nama SDK dan compiler, artefak deployment, tingkat akses publik, serta model yang benar-benar didokumentasikan vendor. Perhatian khusus juga diberikan kepada Amlogic, karena stack ADLA yang lebih baru berbeda secara material dari ekosistem NPU A311D yang lebih lama.

> **Cakupan riset:** Sumber diperiksa pada 15 Agustus 2026. Klaim model bernama bersumber dari halaman produk, dokumentasi, model zoo, repositori, atau tutorial vendor pihak pertama, kecuali dinyatakan lain. Angka TOPS yang diiklankan adalah angka vendor, bukan hasil benchmark yang dapat dibandingkan secara independen. Ini panduan developer, bukan saran investasi atau pemeringkatan berbayar.

**Navigasi cepat:** [tabel perusahaan](#npu-companies-and-platforms-at-a-glance) | [ulasan mendalam Amlogic](#amlogic-two-npu-generations-not-one) | [profil vendor](#the-strongest-public-deployment-ecosystems) | [tabel artefak terkompilasi](#what-you-actually-deploy-the-artifact-lock-in-table) | [akses dan sinyal siklus hidup](#tool-access-and-lifecycle-signals) | [peta jalan LibreYOLO](#what-this-means-for-libreyolo)

## Temuan terpenting

Perangkat kerasnya terfragmentasi, tetapi pola deployment-nya sangat konsisten:

```text
PyTorch checkpoint
    -> ONNX or TFLite interchange graph
    -> representative calibration data
    -> vendor quantizer and graph compiler
    -> chip-specific binary or model package
    -> vendor runtime on the target
    -> application preprocessing, decode, NMS and rendering
```

[ONNX secara eksplisit mengizinkan runtime, generator kode, dan implementasi perangkat keras](https://onnx.ai/onnx/repo-docs/IR.html), tetapi berkas ONNX hanya menjadi titik serah. Itu tidak berarti setiap operator ONNX dapat diturunkan ke setiap akselerator. Bentuk input statis, batasan operator yang didukung, aturan kuantisasi, dan fallback host menentukan apa yang benar-benar berjalan pada NPU.

Karena itu, stack perangkat lunak merupakan bagian dari chip. NPU yang secara nominal cepat dengan compiler yang aksesnya dibatasi atau rapuh dapat menjadi target deployment yang lebih buruk daripada perangkat lebih kecil dengan toolchain publik, model zoo, simulator, dan runtime stabil.

## Arti "didukung" dalam panduan ini

Literatur vendor menggunakan kata *dukungan* dengan sangat longgar. Artikel ini membedakan empat tingkat bukti:

| Tingkat bukti | Yang dibuktikan | Yang tidak dibuktikan |
|---|---|---|
| **Model tervalidasi** | Vendor menerbitkan entri model khusus chip, hasil, atau tabel kompatibilitas | Checkpoint yang Anda modifikasi mempertahankan akurasi yang sama |
| **Contoh resmi** | Vendor menyediakan tutorial atau demo end-to-end untuk model tertentu | Ukuran, task, atau generasi lain dapat dikompilasi |
| **Kemampuan compiler** | SDK mengimpor framework atau mengekspos operator yang didukung | Graf YOLO tertentu berfungsi end-to-end |
| **Pemasaran atau akses terbatas** | Produk ditujukan untuk vision dan SDK privat tersedia | Kompatibilitas model dapat direproduksi secara publik |

Pembedaan ini penting. "Mengimpor ONNX" tidak sama dengan "mendukung segmentasi YOLO11." Model mungkin terbaca, tetapi kemudian berjalan melalui CPU, dikompilasi dengan nilai numerik keluaran yang keliru, melampaui memori akselerator, atau kehilangan akurasi selama kuantisasi.

## Perusahaan dan platform NPU secara ringkas

Tabel sengaja mencampur SoC, akselerator diskret, sensor pintar, serta target GPU/FPGA yang berdekatan. Semuanya bersaing untuk keputusan deployment yang sama meskipun vendor memakai nama seperti NPU, AIPU, BPU, KPU, DLA, HTP, TPU, atau MLA.

### SoC tertanam, sensor pintar, dan prosesor industri

| Perusahaan | Chip atau platform terkait | SDK, compiler, dan artefak | Bukti computer vision yang terdokumentasi publik | Akses |
|---|---|---|---|---|
| [Amlogic](https://github.com/Amlogic-NN/amlnn-toolkit) | A311D2, S928X, S905X5/S905D5, A311Y3, C308L/C302X/C302X2, T968D4, C305X2 dan A123X | AMLNN Toolkit dan `libnnsdk.so`, `.adla` terkompilasi; stack Acuity `.nb` lama yang terpisah untuk A311D | [Model playground](https://github.com/Amlogic-NN/amlnn-model-playground) mendokumentasikan YOLOv5/6/7/8/10/11, YOLOX, YOLOE, YOLO-World, PP-YOLOE, segmentasi, pose, dan OBB pada A311D2, S905X5, A311Y3, C305X2, dan A123X; chip lain yang tercantum adalah target compiler, bukan entri dalam matriks dukungan itu | GitHub dan wheel publik; BSP/driver yang kompatibel tetap diperlukan |
| [Rockchip](https://github.com/airockchip/rknn-toolkit2) | RK3562/3566/3568, RK3576, RK3588, RV1126B | RKNN-Toolkit2, RKNN Runtime, `.rknn` | YOLOv5/6/7/8/10/11, YOLOX, YOLO-World, PP-YOLOE, segmentasi, pose, dan OBB di [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) | GitHub publik; wheel biner |
| [Qualcomm](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | Platform Snapdragon dan Dragonwing, QCS6490, QCS8550, dan Dragonwing IQ-9075 | QAIRT/QNN, SNPE, dan AI Hub; biner konteks QNN atau DLC | Koleksi model [AI Hub](https://github.com/qualcomm/ai-hub-models) mencakup YOLOv3/5/6/7/8/9/10/11/26, YOLOX, YOLO-World, YOLOR, RF-DETR, serta model deteksi/segmentasi/pose | Dokumentasi publik; persyaratan SDK/akun bervariasi |
| [Texas Instruments](https://github.com/TexasInstruments/edgeai-tidl-tools) | AM62A, AM67A, AM68A, AM69A dan TDA4x; TDA54-Q1 pratinjau | Processor SDK Edge AI dan TIDL | [Edge AI Model Zoo](https://github.com/TexasInstruments/edgeai-modelzoo) menerbitkan model deteksi, segmentasi, pose, dan klasifikasi yang dioptimalkan untuk jalur AM6xA/TDA4 saat ini; ini belum menjadi matriks target untuk TDA54-Q1 pratinjau | GitHub publik dan Processor SDK |
| [NXP](https://www.nxp.com/design/design-center/software/development-software/eiq-ml-development-environment:EIQ) | i.MX 8M Plus, i.MX 93, i.MX 95 serta Kinara Ara-1/Ara240 yang diakuisisi | eIQ dengan TIM-VX, Vela, Neutron Converter, dan Ara SDK | [eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo) mencakup aset atau resep YOLOv4-tiny/v8, NanoDet, CenterNet, FastestDet, SSD Lite, dan YOLACT; entri YOLOv5 hanya dokumentasi | Komponen campuran, sebagian publik dan sebagian perlu akun |
| [STMicroelectronics](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) | Varian STM32N6x7, termasuk STM32N657/647, dengan akselerator Neural-ART | STM32Cube AI Studio dan ST Edge AI Core | Repositori layanan saat ini mencantumkan Tiny YOLOv2, YOLOv5u, YOLOv8, YOLO11, YOLO26, dan ST-YOLOX, juga pose dan segmentasi | Tool dan repositori publik |
| [Infineon](https://documentation.infineon.com/psocedge/) | PSOC Edge E83/E84 dengan Cortex-M55 plus Ethos-U55; akselerator NNLite terpisah di sisi M33 | ModusToolbox, [DEEPCRAFT Model Converter](https://www.infineon.com/design-resources/embedded-software/deepcraft-edge-ai-solutions/deepcraft-model-converter), TFLite Micro, dan Arm Vela | Materi arsitektur menyebut MobileNetV1/V2 sebagai kernel representatif dan kit AI E84 mencakup kamera, tetapi tidak ditemukan matriks validasi YOLO pihak pertama untuk PSOC Edge | Dokumentasi publik, komponen SDK, dan kit evaluasi E84 saat ini |
| [Renesas](https://github.com/renesas-rz/rzv_drp-ai_tvm) | RZ/V2L, V2M, V2MA, V2H, dan V2N | DRP-AI Translator, DRP-AI TVM, dan RUHMI | [Daftar model RZ/V2H](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/model_list/Model_List_V2H.md) memvalidasi ukuran YOLOv5/v8/v11/26, YOLOX, serta varian pose dan segmentasi | GitHub publik dan SDK board |
| [Sony](https://www.aitrios.sony-semicon.com/edge-ai-devices/imx500) | Sensor vision cerdas IMX500 dan Raspberry Pi AI Camera | Edge-MDT, Model Compression Toolkit, dan IMX500 packer; paket `.rpk` | [Zoo Raspberry Pi IMX500](https://github.com/raspberrypi/imx500-models) mencakup YOLOv8n, YOLO11n, EfficientDet Lite, NanoDet+, dan SSD | Alur Raspberry Pi publik; ketentuan tool Sony berlaku |
| [Ambarella](https://www.ambarella.com/developer/model-garden/) | Keluarga CV72/CV75, CV5/CV52, dan CV3-AD | Cooper Developer Platform, compiler CVflow, dan runtime DAG | Model garden publik mencantumkan YOLOX-S, RTMDet-nano, DeepLabV3+, TopFormer, OWL-ViT, dan LLaVA OneVision | Terutama terbatas untuk mitra |
| [Synaptics](https://developer.synaptics.com/docs/sl/overview) | Astra SL1600/SL1680; SL2611/13/15/17/19; keluarga MCU SR100 terpisah | SyNAP `.synap` pada SL16xx; Torq/IREE `.vmfb` pada SL261x; SDK SR terpisah berorientasi Ethos-U55 | Benchmark SyNAP YOLOv8n/v8s resmi dan contoh Torq YOLOv8 SL261x; bukti SR perlu dievaluasi terpisah | Portal developer; beberapa unduhan terbatas |
| [MediaTek](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/index.html) | Genio 360/360P/420/520/720 dengan NP8 dan MDLA 5.3; Genio 510/700/1200 dengan NP6/MDLA lebih lama | NeuroPilot Converter, `ncc-tflite`, Neuron Runtime, dan `.dla`; jalur ONNX Runtime pada sistem baru tertentu | IoT AI Hub resmi menerbitkan halaman model dan benchmark YOLOv5 dan YOLOv8, serta model klasifikasi dan wajah | Dokumentasi Yocto publik; bundel NP8 utama dan materi Android memerlukan akses pelanggan langsung/NDA |
| [Allwinner](https://docs.aw-ol.com/v853/en/npu/dev_npu/) | SoC vision V853 dengan NPU Vivante 1-TOPS | Konversi Acuity/Pegasus dan runtime `viplite` | Materi resmi V853 menyebut YOLOv2/3/4/4-tiny/5/5s, RetinaNet, MobileNet, ResNet, dan jaringan wajah/orang | Dokumentasi publik; unduhan paket SDK mungkin perlu akun |
| [Canaan/Kendryte](https://github.com/kendryte/nncase) | K230/K230D saat ini; KPU K210/K510 lama | Compiler dan runtime `nncase`; `.kmodel` | Panduan nncase K230 mencakup kompilasi/simulasi/runtime YOLOv5s; katalog demo resmi dan [changelog CanMV](https://github.com/kendryte/canmv_k230/blob/canmv_k230/CHANGELOG.md) saat ini mendokumentasikan task YOLOv8/11/26 | Compiler/PyPI publik; plug-in chip berbentuk biner |
| [Alif Semiconductor](https://alifsemi.com/support/kits/ensemble-e7appkit/) | Ensemble E7 dengan dua akselerator ML termasuk Ethos-U55; [E8](https://alifsemi.com/ensemble-e8-series/) dengan satu NPU Ethos-U85 dan dua Ethos-U55 | Deployment berbasis Arm Vela dan TFLite Micro | Alif menerbitkan benchmark detektor wajah YOLO-Fastest INT8 tingkat keluarga, tetapi tidak ada matriks YOLO modern yang luas per target | Dokumentasi publik dan sumber kit evaluasi |
| [Himax](https://www.himax.com.tw/products/wiseeye-ai-sensing/wiseeye2-ai-processor/) | MCU AI endpoint HX6538 WiseEye2 dengan Cortex-M55 dan Ethos-U55 | TFLite Micro plus Arm Vela, dengan CMSIS-NN dan fallback kernel referensi | [Contoh WiseEye2](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2) resmi mencakup deteksi, pose, klasifikasi YOLOv8n, deteksi YOLO11n, face mesh, dan PeopleNet | GitHub dan dokumentasi publik serta board mitra yang dapat dibeli |
| [Analog Devices](https://www.analog.com/en/products/max78002.html) | MCU AI MAX78000 dan MAX78002 dengan akselerator CNN berdaya rendah | [`ai8x-training`](https://github.com/analogdevicesinc/ai8x-training), `ai8x-synthesis`/`izer`, dan MSDK; C dan bobot yang dihasilkan | Materi resmi mencakup identifikasi/deteksi wajah, RetinaNet, Visual Wake Words, classifier, dan pengenalan aksi; tidak ditemukan deployment YOLO pihak pertama | Tool GitHub publik, dokumentasi, dan board evaluasi |
| [D-Robotics](https://github.com/D-Robotics/rdk_model_zoo) | Board BPU RDK X3/X5/Ultra dan seri S100 yang lebih baru | OpenExplorer/Algorithm Toolchain dan `hbm_runtime`; X5 `.bin`, `rdk_s` saat ini `.hbm` | Branch `rdk_x5` saat ini mendokumentasikan YOLOv5/v5u/v8/v9/v10/11/12/13/26, YOLOE, YOLO-World, segmentasi, pose, klasifikasi, OCR, dan CLIP untuk RDK X5; X3 dan seri S memakai branch terpisah | GitHub publik; beberapa paket toolchain khusus platform |
| [AXERA](https://github.com/AXERA-TECH/ax-samples) | AX650, AX637, AX630C, AX620Q, dan AX615 | Compiler Pulsar2 dan AXEngine; `.axmodel` | Sampel resmi saat ini mencakup YOLOv5/6/7/8/9/10/11/13/26, YOLOX, dan YOLO-World; task dan target berbeda menurut chip | Sampel dan dokumentasi publik; compiler didistribusikan terpisah |
| [SOPHGO](https://github.com/sophgo/tpu-mlir) | BM1684/1684X/1688/1690 dan CV186X/CV18xx | TPU-MLIR dan runtime SOPHON; `.bmodel` atau `.cvimodel` | Demo resmi mencakup YOLOv3/4/5/7/8/9/10/11/12/26, YOLOX, PP-YOLOE, YOLO-World, OBB, segmentasi, wajah, SAM, dan OCR | Compiler open source plus runtime vendor |
| [Huawei Ascend](https://www.hiascend.com/en/software/cann) | Produk edge Ascend 310/310P/310B dan Atlas 200I/300I | CANN, compiler ATC, dan AscendCL; `.om` | [Ascend ModelZoo](https://github.com/Ascend/modelzoo) resmi mencakup YOLOv2/3/4/5, YOLOX, YOLOR, SSD, RetinaNet, Mask R-CNN, dan model segmentasi | Dokumentasi publik; paket CANN dan kernel harus cocok dengan target |
| [Cambricon](https://github.com/Cambricon/magicmind_cloud) | MLU370-X4/S4 dalam matriks publik yang dirujuk; MLU270 adalah perangkat keras lama yang tidak dicakup matriks itu | Neuware dan MagicMind | Matriks MagicMind 1.7 di repositori mencantumkan model YOLOv3/4/5/7/8, PP-YOLOE, SSD, RetinaFace, RetinaNet, Mask R-CNN, dan segmentasi | Contoh model historis publik; akses SDK/container saat ini terbatas |
| [Sunplus](https://sunplus.atlassian.net/wiki/spaces/C3/pages/1971126471/SP7350%2BSpecification) | SP7350/C3V, sekitar 4.1 sampai 4.6 TOPS yang diiklankan | Docker NPU Vivante Acuity, runtime SNNF, dan `.nb` | Dokumentasi resmi menyediakan YOLOv5 dan [deployment YOLOv8 end-to-end](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2427224156/SP7350) khusus | Sumber/dokumentasi publik dan ekosistem board |
| [ESWIN Computing](https://www.eswincomputing.com/en/news/info/96.html) | SoC RISC-V EIC7700/EIC7700X dan [EIC7702/EIC7702X](https://www.eswincomputing.com/en/news/info/104.html) dual-die | ENNP: EsQuant, compiler EsAAC saat ini, simulator, runtime ESSDK; `.model`; dokumentasi lama memakai `ennc-compile` | Dokumentasi ENNP yang dihosting Milk-V mencakup tutorial [YOLOv3 end-to-end](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/yolov3), MobileNetV2, dan ResNet | Dokumentasi campuran dari pihak pertama dan board; unduhan regional |
| [Nuvoton](https://www.nuvoton.com/products/microcontrollers/arm-cortex-m55-mcus/m55m1-series/index.html) | MCU M55M1 dengan Ethos-U55-256 | TFLite Micro, Arm Vela, dan NuEdgeWise/NuML | Contoh resmi [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise) menyebut YOLOv8-nano, YOLOX-nano, YOLO Fastest, SSD-MobileNet, dan model klasifikasi | Toolchain MCU sebagian besar publik |
| [Realtek](https://github.com/Ameba-AIoT/ameba-arduino-pro2) | AmebaPro2 RTL8735B / platform kamera AMB82-mini | SDK Arduino/FreeRTOS, VoE, dan NeuralNetwork API; `.nb` | Model terpaket mencakup YOLOv3-tiny, YOLOv4-tiny, YOLOv7-tiny, SCRFD, dan MobileFaceNet | Runtime publik; akses [converter khusus](https://ameba-doc-arduino-sdk.readthedocs-hosted.com/en/latest/FAQ/offline_ai_model_conversion_steps.html) perlu menghubungi vendor |
| [Telechips](https://docs.topst.ai/product/p/ai) | Board AI TCC7500 / TOPST, diiklankan 8 TOPS | TC-NN-Toolkit / Enlight SDK; `.enlight` perantara dan bundel deployment terkompilasi | [Proyek TOPST resmi](https://docs.topst.ai/blog/31) men-deploy YOLOv8s; konversi UFLD v1/v2 gagal pada layer yang tidak didukung dan hanya mengusulkan solusi pemisahan/pascapemrosesan. [YOLOv4](https://community.topst.ai/t/segmentation-fault/358) dan [alur YOLOv8 lain](https://community.topst.ai/t/segmentation-fault-error/415) muncul di utas dukungan komunitas | Dokumentasi board publik; unduhan compiler/operator sering memerlukan izin |
| [T-Head](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) | TH1520, diiklankan 4-TOPS INT8, pada LicheePi 4A | Compiler HHB dan CSI-NN2/SHL; `hhb.bm` plus kode/parameter yang dihasilkan | Contoh resmi vendor board menjalankan YOLOv5n/s dan MobileNetV2 pada NPU | Contoh publik; toolchain menua dan status pemeliharaan tidak pasti |
| [SigmaStar](https://wx.comake.online/doc/doc/SigmaStarDocs-SSU9383CM-SIGMASTAR-202507071022/platform/MI/ipu_zh.html) | SoC kamera pintar SSU9383CM saat ini dan keluarga IPU lama | Runtime MI_IPU saat ini; toolchain [SGS_IPU lama](https://wx.comake.online/doc/doc/SigmaStarDocs-SSC9381G_9351_Pudding-ULS00V040-20210913/customer/development/dla/tools.html) menghasilkan `.sim` menjadi `sgsimg.img` | [Postprocessor SDK resmi lama](https://wx.comake.online/doc/doc/Sigmastar_SDK_v1.2.2/module/User_Guide/Common/SigmaStar_Post_Processing_Module.html) menyebut SSD dan YOLOv1/2/3; kompatibilitas publik contoh lama dengan SSU9383CM tidak terverifikasi | Dokumentasi vendor, bukti model terutama historis |

| [HiSilicon](https://www.hisilicon.com/cn/products/smart-vision/machine-vision/hi3516cv610) | SoC smart vision Hi3516CV610/DV500, Hi3519DV500, dan Hi3403V100 | ATC ke `.om` dengan runtime board NNN/SVP-NNN | Matriks HiSpark khusus target, terutama untuk Hi3403 dan Hi3591P, mencantumkan YOLOv3 sampai YOLO11, pose, segmentasi, OBB, OCR, dan entri kedalaman; ini bukan validasi untuk setiap SoC pada baris ini | Aktif dan berbeda dari Ascend; model repositori berlabel hanya untuk penggunaan nonkomersial |

### Akselerator dan modul edge khusus

| Perusahaan | Silikon terkait | SDK dan artefak | Bukti model yang terdokumentasi publik | Akses |
|---|---|---|---|---|
| [Hailo](https://github.com/hailo-ai/hailo_model_zoo) | Keluarga Hailo-8, Hailo-8L, Hailo-10H, dan Hailo-15 | Dataflow Compiler dan HailoRT; `.hef` | YOLOv3/v4/v5/v6/v7/v8/v9/v10/11/12 dan YOLO26, serta YOLOX, DAMO-YOLO, SSD, EfficientDet, segmentasi, pose, dan OBB, dengan tabel model khusus generasi | Model Zoo publik; compiler melalui Developer Zone |
| [Axelera AI](https://docs.axelera.ai/sdk/reference/models/model-zoo/) | Produk Metis yang dikirim; produk portofolio Europa dan Titania yang diumumkan | Voyager SDK publik; Pipeline Builder alfa `.axm`/`.axe`, bundel klasik `.axmodel` | YOLOv3/5/7/8/9/10/11/26 tervalidasi, YOLOX, YOLO-NAS, OBB, pose, segmentasi, dan banyak model non-YOLO | SDK publik di GitHub; dukungan pelanggan perlu akun |
| [DEEPX](https://developer.deepx.ai/modelzoo/) | DX-M1 dan DX-M1M | DXNN SDK, DX-COM, dan DX-RT; `.dxnn` | Model Zoo mencatat 354 entri dengan DX-COM 2.4.0/DX-RT 3.4.0 saat diperiksa pada 15 Agustus 2026, termasuk YOLOv3 sampai YOLO11 dan YOLO26, YOLOX, SSD, EfficientDet, NanoDet, segmentasi, pose, dan OBB | Sumber daya developer dan suite publik |
| [MemryX](https://developer.memryx.com/tools/neural_compiler.html) | Akselerator MX3 dan modul multi-chip | MemryX SDK, Neural Compiler, dan runtime; `.dfp` | Contoh resmi dan catatan rilis mencakup deteksi, segmentasi, pose, termasuk YOLOv10, YOLO11, dan YOLO26 | Dokumentasi dan SDK publik |
| [Kneron](https://doc.kneron.com/docs/) | KL520, KL530, KL630, KL720, KL730 memiliki dokumentasi compiler saat ini; KL830 muncul pada sejumlah API PLUS, tetapi tidak dalam daftar target compiler saat ini | Kneron PLUS dan Model Toolchain; `.nef` pada target compiler yang terdokumentasi | [Alur YOLO](https://doc.kneron.com/docs/toolchain/appendix/yolo_example/) resmi berfokus pada Tiny-YOLOv3; materi lain mencakup pelatihan dan deployment YOLOv5 | Dokumentasi publik; paket tool/unduhan berbeda menurut chip |
| [SiMa.ai](https://docs.sima.ai/pages/palette/modelsdk.html) | MLSoC dan platform Modalix 50-TOPS produksi | Palette, ModelSDK, MLA Compiler, dan ModelExecutor | Rilis publik memvalidasi YOLOv7/v8, YOLOX, pose/segmentasi, DETR, Mask R-CNN, dan EfficientDet | Dokumentasi publik; SDK produk bersifat komersial |
| [EdgeCortix](https://www.edgecortix.com/en/hardware) | SAKURA-II, akselerator 60-TOPS yang diiklankan dalam produk M.2 dan PCIe | Compiler dan framework MERA | Vendor menyatakan dukungan vision hingga generative AI, tetapi tidak menerbitkan matriks kompatibilitas YOLO terkini yang cukup terperinci | Uji coba/permintaan pesanan komersial; validasi dipimpin mitra |
| [BrainChip](https://brainchip.com/metatf-dev-tools/) | Koprosesor/modul M.2 AKD1500 yang dikirim; platform AKD1000 lama; IP Akida 2 | Paket MetaTF: `akida-models`, `quantizeml`, `cnn2snn`, dan runtime `akida` | Kartu model Akida 2 mencantumkan detektor YOLOv2 AkidaNet0.5, CenterNet, AkidaUNet, dan pengenalan wajah; hasil itu tidak otomatis memvalidasi AKD1500 | Paket inti Python publik; pemetaan perangkat keras tergantung target |
| [Blaize](https://www.blaize.com/products/) | Produk P1600, Pathfinder, dan Xplorer | Picasso SDK, NetDeploy, dan AI Studio | Vision merupakan target pasar, tetapi tidak ditemukan matriks kompatibilitas publik model bernama yang dapat diaudit | Komersial/akses terbatas |
| [Google Coral](https://github.com/google-coral/edgetpu) | Produk Edge TPU USB, PCIe, M.2, dan Dev Board lama; [Coral NPU IP](https://github.com/google-coral/coralnpu) open source yang aktif secara terpisah | Edge TPU Compiler/`libedgetpu`/PyCoral lama; Coral NPU RISC-V baru mengekspos IP, RTL/simulasi, dan contoh ELF | Contoh lama menekankan SSD MobileNet, klasifikasi, DeepLab, dan MoveNet; IP baru tidak memiliki matriks deployment YOLO publik dan bukan penerus langsung produk Edge TPU | Repositori inti lama diarsipkan; Coral NPU IP baru dikembangkan aktif |
| [Lattice Semiconductor](https://www.latticesemi.com/en/Solutions/Solutions/SolutionsDetails02/sensAI?ActiveTab=Design+File) | FPGA ECP5, iCE40 UltraPlus, CrossLink-NX, CertusPro-NX, Avant-E, dan Avant-X | sensAI Studio, Neural Network Compiler, dan IP akselerator yang dapat dikonfigurasi | Tabel compiler saat ini mencantumkan YOLOv1, YOLOv5, YOLOv8, dan YOLO11 dalam mode khusus target, serta SSD, MobileNetV2-SSD, ResNet, dan ENet | Compiler/manual publik; ketentuan IP bervariasi dan Advanced CNN Accelerator bersifat komersial |
| [Mobilint](https://www.mobilint.com/sdk-qb) | Akselerator REGULUS 10-TOPS dan ARIES MLA100 80-TOPS | qb SDK; compiler INT8 dan `.mxq` | [Model zoo](https://github.com/mobilint/mblt-model-zoo/blob/master/mblt_model_zoo/vision/README.md) publik mencantumkan deteksi YOLOv3/5/7/8/9/10/11/12/26 serta varian segmentasi, pose, dan OBB | Dokumentasi/model publik; SDK dan perangkat keras komersial |
| [Rebellions](https://rebellions.ai/developers/) | ATOM+ CA22 dan ATOM-Max CA25 aktif; ATOM CA02/ATOM+ CA12 EoL; status tool ATOM-Lite CA21 belum jelas | Compiler/runtime/profiler RBLN SDK; `.rbln` | Rilis dukungan resmi menyebut keluarga YOLOv3, YOLOv5/6/7/8 dan tutorial YOLOv8 saat ini | Dokumentasi publik; wheel compiler perlu kredensial portal |
| [FuriosaAI](https://furiosa.ai/warboy) | Warboy Gen1 berorientasi vision; generasi RNGD terpisah | Furiosa SDK; INT8 `.enf` di Warboy, `.fxb` terpisah di RNGD | Zoo Warboy mendokumentasikan SSD, YOLOv5M/L dan YOLOv7-w6-pose; peta jalan RNGD menandai dukungan YOLOv8m selesai pada Q4 2024, tetapi tidak menyediakan matriks vision publik terkini secara terperinci | Akses IAM/akun; kedua generasi harus dibedakan |

### Platform berdekatan yang dibandingkan developer dengan NPU

| Perusahaan | Perangkat keras | Stack deployment | Alasan dimasukkan dalam perbandingan |
|---|---|---|---|
| [NVIDIA](https://github.com/NVIDIA-AI-IOT/deepstream_tools/tree/main/yolo_deepstream) | GPU Jetson Orin plus DLA; GPU Jetson Thor tanpa DLA | JetPack, TensorRT, DeepStream; `.engine` | Deployment vision sangat matang; tool DeepStream resmi mendokumentasikan YOLOv4/v7/v8/v9/11, termasuk konfigurasi YOLO11 OBB. Banyak hasil memakai GPU dan layer Orin DLA yang tidak didukung dapat beralih ke GPU. |
| [AMD](https://vitisai.docs.amd.com/en/6.2/) | Target DPU Kria/lama; Vitis AI 6.2 GA untuk Versal AI Edge Gen1 VEK280/VE2802 dan Gen2 VEK385; NPU klien Ryzen AI terpisah | `.xmodel` lama; snapshot NPU Gen1 yang terikat target; direktori cache terkompilasi Gen2 atau paket produksi `.rai` | Vitis AI menerbitkan materi vision, tetapi DPU lama, Versal Gen1, Versal Gen2, dan Ryzen AI memiliki kontrak kompilasi/deployment terpisah. |
| [Microchip](https://www.microchip.com/en-us/products/fpgas-and-plds/fpga-and-soc-design-tools/vectorblox) | FPGA PolarFire SoC dengan IP akselerator CoreVectorBlox yang dapat dikonfigurasi | VectorBlox SDK 3.1 publik dan Libero; aset deployment `.vnnx`, `.hex`, dan `.ucomp` | [Tutorial terkini](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/README.md) mencakup YOLOv5n, deteksi/klasifikasi/OBB/pose/segmentasi YOLOv8, YOLOv9t, dan model vision lain; SDK 3.1 saat ini ditujukan untuk PolarFire SoC Video Kit. |
| [Intel](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md) | NPU Core Ultra, CPU, dan GPU terintegrasi/diskret | OpenVINO IR atau cache model terkompilasi | OpenVINO menyediakan jalur lintas-XPU publik; gunakan kolom NPU pada [matriks model terverifikasi](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html), jangan berasumsi setiap contoh YOLO OpenVINO tervalidasi untuk NPU. |
| [Apple](https://developer.apple.com/machine-learning/core-ml/) | Apple Neural Engine pada chip seri A mulai A11 dan chip seri M | Core ML dan `coremltools`; `.mlpackage`/`.mlmodelc` | NPU konsumen yang sangat mudah diakses lewat Core ML, tetapi Apple mengabstraksi pembagian CPU/GPU/Neural Engine alih-alih mengekspos compiler perangkat keras khusus YOLO. |
| [GreenWaves Technologies](https://github.com/GreenWaves-Technologies/nn_menu_gap9) | GAP9 dengan neural engine NE16 | GAP SDK, NNTool, dan kode hasil AutoTiler | Repositori resmi mencakup MobileNet SSD, deteksi wajah, klasifikasi, dan [detektor orang YOLOX](https://github.com/GreenWaves-Technologies/yolox_people_detection) khusus. SDK penuh terbatas untuk pelanggan berkualifikasi. |
| [Syntiant](https://www.syntiant.com/hardware) | Neural Decision Processor NDP200 produksi massal dan NDP250 tahap sampling | Syntiant SDK dan paket deployment | Prosesor vision/sensor selalu aktif berdaya ultra-rendah: NDP200 didokumentasikan di bawah 1 mW, sedangkan pengenalan gambar NDP250 di bawah 30 mW. Tidak ditemukan matriks YOLO publik yang luas. |


## Amlogic: dua generasi NPU, bukan satu

Amlogic perlu dijelaskan lebih mendalam karena informasi daring kerap mencampur produk yang tidak kompatibel. Perusahaan ini memiliki jalur lama berbasis VeriSilicon dan jalur ADLA proprietary yang lebih baru. Keduanya tidak menggunakan compiler, artefak, atau runtime yang sama.

| Generasi | Chip representatif | NPU dan toolchain | Artefak terkompilasi | Status praktis |
|---|---|---|---|---|
| Lama | A311D dan S905D3, umum pada Khadas VIM3/VIM3L | VeriSilicon Vivante VIPNano-QI, toolkit Acuity, KSNN, dan `aml_npu_sdk` lama | `.nb` di dalam direktori keluaran `nbg_unify` | Board dan demo yang sudah ada; integrasi lama terpisah |
| ADLA2 saat ini | C308L/C302X, S928X, A311D2, T968D4, S905X5/S905D5, dan C302X2 | Amlogic ADLA2, AMLNN Toolkit, dan NNSDK2 | `.adla` khusus target; W8A8 atau W8A16 | Stack saat ini berfokus pada integer |
| ADLA3 saat ini | A311Y3, C305X2, dan A123X | Amlogic ADLA3, AMLNN Toolkit, dan NNSDK2 | `.adla` khusus target; W4A8, W8A8, W4A16, W8A16, atau W16A16 | Menambahkan kemampuan native INT4, FP16, dan BF16 |

[Datasheet A311D](https://dl.khadas.com/products/vim3/datasheet/a311d-datasheet.pdf) menyebut NPU INT8 asli sebesar 5-TOPS. [Halaman aplikasi Khadas VIM3](https://docs.khadas.com/products/sbc/vim3/npu/npu-app) dan [ikhtisar NPU](https://docs.khadas.com/products/sbc/vim3/npu/start) lama mendokumentasikan DenseNet CTC, MTCNN, RetinaFace, YOLOFace, YOLOv2, YOLOv3, YOLOv3-tiny, YOLOv4, YOLOv7-tiny, dan YOLOv8n; ikhtisar itu juga mendokumentasikan YOLOv8n-pose dan menandai FaceNet sebagai deprecated. Contoh tersebut nyata, tetapi menjadi bukti untuk ekosistem Acuity `.nb` lama, bukan chip ADLA saat ini. A311D bukan A311D2, dan C305X bukan C305X2.

Ada jebakan perangkat keras kedua. Panduan revisi [Khadas VIM4](https://docs.khadas.com/products/sbc/vim4/configurations/identify-version) menyatakan board V12/A311D2 revisi-B asli tidak memiliki NPU; NPU 3.2-TOPS yang diiklankan ada pada board V13A dan yang lebih baru dengan komponen A311D2-N0D revisi-C. Nama produk saja tidak cukup untuk memilih perangkat uji.

### Stack AMLNN dan ADLA saat ini

[AMLNN Toolkit](https://github.com/Amlogic-NN/amlnn-toolkit/tree/7d3cc9a36179b756ce79c953c14d43acab1f2af5) saat ini mencakup konversi model, kuantisasi, kompilasi, inferensi, dan profiling. Panduan pengguna [Mei 2026 yang dipatok versinya](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/02_Amlogic_NPU_User_Guide_V0.1.pdf) mendokumentasikan importer ONNX, TFLite floating-point atau terkuantisasi, TorchScript `.pt`, serta PyTorch 2 ExportedProgram/PT2. Jalur TensorFlow, Paddle, dan Keras ditandai sebagai rencana dalam panduan terperinci, walaupun ikhtisar repositori memakai uraian yang lebih luas. Compiler menghasilkan `.adla`. Deployment Python memakai AMLNN atau `amlnn_edge_toolkit_lite`; C/C++ native menautkan `libnnsdk.so` melalui `nnsdk2.h`. Pengembangan di host dapat memakai ADB dengan `nnserver`; runtime juga menargetkan Android, Buildroot, Yocto, dan lingkungan Debian/Armbian tertentu.

Identifier platform yang diterbitkan toolkit saat ini meliputi:

| ID target | Platform |
|---:|---|
| `001` | C308L / C302X |
| `002` | S928X |
| `003` | A311D2 |
| `004` | T968D4 |
| `005` | S905X5 / S905D5 |
| `006` | C302X2 |
| `007` | A311Y3 |
| `008` | C305X2 / A123X di dokumentasi terperinci dan matriks model |

Kolom target ini bukan hiasan. Amlogic mengharuskan `.adla`, `nnsdk2.h`, `libnnsdk.so`, BSP, dan generasi compiler/runtime saling cocok. [Quick Start](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/01_Amlogic_NPU_Quick_Start_guide_V0.1.pdf) yang dipatok versinya mensyaratkan driver ADLA 2.0.2 atau lebih baru, NNSDK 3.0.0 atau lebih baru, dan NNSDK2 1.0.0 atau lebih baru; [panduan deployment Android](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/06_Amlogic_NPU_Android_Deployment_Guide_V0.1.pdf) memperjelas hubungan library, header, driver, dan BSP. Perlakukan `.adla` sebagai artefak build yang terikat ABI, bukan model portabel.

Kuantisasi juga bergantung pada generasi NPU. Target ADLA2 001 sampai 006 mendokumentasikan kompilasi W8A8 dan W8A16. Target ADLA3 007 dan 008 menambahkan kombinasi W4A8, W4A16, W8A8, W8A16, dan W16A16, dengan dukungan native INT4, FP16, dan BF16 dalam [panduan operator](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/05_Amlogic_NPU_Support_Operator_List_V0.1.pdf). Amlogic merekomendasikan 200 sampai 500 sampel kalibrasi yang representatif. Opsi data acaknya secara eksplisit ditujukan untuk pengujian performa, bukan kuantisasi yang menjaga akurasi.

### Cakupan model Amlogic yang terdokumentasi

[Model playground Amlogic yang dipatok versinya](https://github.com/Amlogic-NN/amlnn-model-playground/blob/03ca63ef20c4f1560722de26ab910826884f334a/README.md) merupakan bukti yang jauh lebih kuat daripada klaim dukungan ONNX umum. Matriks dukungan/contoh yang dipublikasikan menyebut A311D2, S905X5, A311Y3, C305X2, dan A123X. Diterimanya model oleh compiler pada ID target lain tidak membuktikan seluruh daftar model ini tervalidasi di sana.

| Task | Model terdokumentasi |
|---|---|
| Klasifikasi | MobileNetV2 dan ResNet50-v2; DINO pada ADLA3 |
| Deteksi objek | PP-YOLOE, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv10, YOLO11, YOLOE, YOLO-World, YOLOX, dan deteksi QR |
| Wajah dan gestur | RetinaFace dan Gesture Recognition |
| Segmentasi | DeepLabV3, PP-LiteSeg, YOLOv5-seg, dan YOLOv8-seg |
| Deteksi berorientasi | YOLOv8 OBB |
| Pose | Detektor/landmark BlazePose dan YOLOv8 pose |
| OCR dan ucapan | LPRNet, varian PaddleOCR, Whisper Tiny, dan SenseVoice pada chip baru tertentu |
| Transformer baru dan kerja multimodal | DETR, MobileSAM, CLIP/MobileCLIP, serta contoh LLM/VLM bit rendah tertentu pada platform yang lebih baru |

Repositori yang sama menerbitkan angka runtime model W8A8 berikut. Ini pengukuran vendor untuk jaringan saraf di NPU, bukan pipeline kamera lengkap.

| Model dan input | S905X5 | A311D2 | A311Y3 |
|---|---:|---:|---:|
| YOLOv8n, 640 x 640 | 101.72 FPS | 95.14 FPS | 191.06 FPS |
| YOLOv8s, 640 x 640 | 42.33 FPS | 42.77 FPS | 83.08 FPS |
| YOLOv8m, 640 x 640 | 19.67 FPS | 19.82 FPS | 35.30 FPS |
| YOLOv8l, 640 x 640 | 10.53 FPS | 10.12 FPS | 18.37 FPS |
| YOLO11n, 640 x 640 | 41.14 FPS | 41.48 FPS | 62.24 FPS |

Catatannya sangat penting. Panduan operator Amlogic menempatkan NMS di perangkat lunak pada ADLA2 dan ADLA3. README mendefinisikan angka tersebut sebagai hasil runtime model NPU native dan mengecualikan prapemrosesan serta pascapemrosesan; apakah setiap transfer memori disertakan tidak diungkapkan. Baris akurasi YOLO memakai subset COCO 300 gambar, bukan seluruh set validasi, sedangkan baris klasifikasi memakai ImageNet `val1000`. Tabel juga melaporkan latensi YOLOv8n A311Y3 yang berbeda dari tabel FPS pada kondisi yang tidak dijelaskan rekonsiliasinya. Clock, mode daya, pendinginan, kondisi termal stabil, versi SDK/BSP yang tepat, dan durasi pengujian tidak diungkapkan. Gunakan angka untuk memahami satu stack vendor, bukan untuk memeringkatnya terhadap vendor lain.

### Mengapa Amlogic menarik secara strategis

Amlogic memiliki empat ciri yang tidak biasa: jejak SoC tertanam yang luas, stack compiler yang baru dibuka untuk publik, matriks model saat ini yang banyak beririsan dengan ruang task LibreYOLO, dan integrasi kelas satu yang relatif sedikit di library pelatihan utama. Repositori saat ini juga masih baru: muncul ke publik pada akhir 2025 dan awal 2026, dengan manual substantif bertanggal Mei sampai Juli 2026. Ini menciptakan peluang first-mover yang nyata sekaligus risiko kestabilan API.

Integrasi yang rapi sebaiknya menggunakan ekspor ONNX deterministik LibreYOLO sebagai input compiler, mewajibkan gambar kalibrasi representatif untuk build presisi rendah, menghasilkan `.adla` beserta manifest metadata, lalu memuatnya melalui adapter NNSDK2. A311D lama harus memakai target `.nb` yang jelas berbeda karena menyajikan `.nb` dan `.adla` sebagai satu backend akan menyebabkan kegagalan kompatibilitas tersembunyi. Repositori Amlogic menggunakan lisensi top-level Apache-2.0, tetapi hak redistribusi wheel, shared library, `nnserver`, dan biner Android AAR yang disertakan perlu dikonfirmasi langsung sebelum dicerminkan oleh LibreYOLO.

## Ekosistem deployment publik terkuat

Perusahaan berikut saat ini menawarkan kombinasi paling jelas antara perangkat keras yang bisa diperoleh, materi teknis publik, dan bukti model bernama. Itu tidak berarti performanya selalu lebih cepat. Klaim kompatibilitasnya lebih mudah dievaluasi sebelum membeli perangkat keras.

### Hailo

Hailo menjadi contoh acuan ekosistem akselerator vision khusus. Model diurai menjadi Hailo Archive, dioptimalkan dan dikuantisasi memakai gambar representatif, lalu dikompilasi menjadi berkas Hailo Executable Format (`.hef`). Aplikasi memuat HEF itu melalui HailoRT.

[Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) berisi resep, model pretrained, konfigurasi pascapemrosesan, dan data performa untuk deteksi, segmentasi, klasifikasi, pose, serta task lain. Cakupan YOLO-nya meliputi YOLOv3/v4/v5/v6/v7/v8/v9/v10/11/12 dan YOLO26 bersama YOLOX, DAMO-YOLO, SSD, dan EfficientDet. [Tabel deteksi objek Hailo-8](https://github.com/hailo-ai/hailo_model_zoo/blob/v2.19.0/docs/public_models/HAILO8/HAILO8_object_detection.rst) yang dipatok versi merupakan sumber kompatibilitas lebih baik daripada halaman produk umum, sedangkan [aplikasi mandiri](https://github.com/hailo-ai/hailo-apps/blob/main/hailo_apps/python/standalone_apps/object_detection/README.md) mendokumentasikan pipeline YOLO yang dapat di-deploy.

Ada batas generasi versi yang penting. Hailo-8 dan Hailo-8L menggunakan Model Zoo 2.x, Dataflow Compiler 3.x, dan HailoRT 4.x yang lebih lama, sementara Hailo-10 dan Hailo-15 memakai generasi 5.x saat ini. Resep, perilaku parser, dan HEF tidak dapat dipertukarkan hanya karena semua perangkat memakai nama Hailo.

Hailo-15 juga memiliki lapisan aplikasi berbeda. Perangkat itu menggunakan Vision Processor Software Package dan Hailo Media Library, bukan jalur host bergaya TAPPAS untuk Hailo-8/10H. Repositori referensi publik [`hailo-camera-apps`](https://github.com/hailo-ai/hailo-camera-apps) diarsipkan pada 3 Mei 2026, sedangkan [`hailo-apps-core`](https://github.com/hailo-ai/hailo-apps-core) tetap menjadi framework aplikasi publik terpisah. Pengarsipan repositori bukan bukti EOL produk, tetapi proyek Hailo-15 perlu mengonfirmasi paket aplikasi terkini di Developer Zone.

[Panduan deployment Hailo](/docs/export/hailo) LibreYOLO berhenti pada serah-terima ONNX statis karena compiler proprietary tidak dapat disertakan sebagai dependensi Python. Itulah batas yang jujur antara menghasilkan graf yang sesuai dan mengklaim HEF yang sudah diuji.

### Rockchip

Rockchip memiliki salah satu jejak Linux tertanam terbesar untuk penghobi dan komersial, terutama lewat board RK3588, RK3576, dan RK356x. [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2) publik mengonversi dan mengkuantisasi model pada host Linux x86, lalu RKNN Runtime atau Toolkit-Lite menjalankan berkas `.rknn` hasilnya di board.

[RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) sangat jelas mengenai chip, keluarga model, dan presisi. Tabel saat ini mencantumkan jalur FP16 dan INT8 untuk YOLOv5/6/7/8/10/11, YOLOX, PP-YOLOE, YOLO-World, dan varian segmentasi YOLOv5/v8; YOLOv8 OBB dan YOLOv8 pose tercantum hanya untuk INT8. Zoo itu juga mencakup OCR, wajah, dan jaringan segmentasi lain.

LibreYOLO sudah memiliki [eksporter RKNN](/docs/export/rknn) langsung yang konservatif. Eksporter memvalidasi empat varian deteksi yang tepat pada RK3588, dapat membandingkan simulator compiler dengan ONNX Runtime, serta menolak keluarga yang belum tervalidasi alih-alih menyamakan build yang berhasil dengan prediksi yang benar. Klaim dukungan yang sempit ini menjadi pola berguna untuk setiap backend NPU mendatang.

### Axelera AI

Axelera AI, yang sering salah dieja "Accelera", membuat AIPU Metis dan menjualnya dalam produk M.2, PCIe, dan multi-chip. Metis merupakan keluarga yang bisa dipesan publik; Europa dan Titania adalah produk portofolio yang diumumkan, sehingga status ketersediaannya tidak boleh dilebur menjadi satu. [Voyager SDK tersedia publik di GitHub](https://github.com/axelera-ai-hub/voyager-sdk) dan menangani pemilihan model, kompilasi, kuantisasi, eksekusi, serta pipeline aplikasi; dukungan pelanggan tetap perlu akun.

[Voyager Model Zoo](https://docs.axelera.ai/sdk/reference/models/model-zoo/) publik termasuk yang paling informatif di pasar. Zoo mencantumkan varian model, ukuran input, presisi, akurasi, dan performa terukur untuk YOLOv3, YOLOv5, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO26 n/s/m/l/x, YOLOX, YOLO-NAS, OBB, pose, dan segmentasi, beserta banyak model klasifikasi dan prediksi padat. Menerbitkan penurunan kuantisasi dan akurasi berdampingan dengan kecepatan lebih berguna daripada hanya mengumumkan angka TOPS puncak.

Penamaan artefak berubah menurut generasi API. Alur kompilasi [Pipeline Builder](https://docs.axelera.ai/sdk/reference/pipeline-builder/model-compilation/) alfa menghasilkan model `.axm` dan dapat mengemas pipeline portabel sebagai `.axe`. [Dokumentasi pipeline klasik](https://docs.axelera.ai/sdk/reference/pipeline/model-formats/) menjelaskan `.axmodel` beserta metadata model dan manifest. Integrasi harus mendeteksi generasi Voyager yang terpasang, bukan menetapkan satu ekstensi secara permanen.

### Qualcomm

Qualcomm menawarkan jangkauan deployment yang sangat luas, tetapi istilah "NPU Qualcomm" menyembunyikan beberapa antarmuka. Developer menjumpai Hexagon HTP melalui QAIRT/QNN, alur SNPE yang lebih lama, layanan kompilasi Qualcomm AI Hub, LiteRT, atau execution provider QNN di ONNX Runtime, bergantung pada perangkat dan kelas produk.

Repositori [Qualcomm AI Hub Models](https://github.com/qualcomm/ai-hub-models) resmi menjadi bukti kuat di tingkat model. Repositori itu menerbitkan paket teroptimasi untuk deteksi/segmentasi/pose YOLOv3, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO26, YOLOX, YOLO-World, YOLOR, RF-DETR, serta banyak model klasifikasi, kedalaman, dan pose. AI Hub juga menyediakan profiling khusus perangkat, yang penting karena model yang dikompilasi untuk satu generasi HTP tidak otomatis portabel ke generasi lain.

Biaya integrasi utamanya adalah banyaknya kombinasi: Android dibanding Linux tertanam, komponen pesanan QCS dibanding Snapdragon konsumen, arsitektur HTP, versi QNN/QAIRT, dan skema kuantisasi semuanya berpengaruh. Halaman resmi Qualcomm saat ini berbeda pendapat tentang status [Dragonwing IQ-9075](https://www.qualcomm.com/internet-of-things/products/iq9-series/iq-9075): halaman produk menyebut Active dan [EVK](https://www.qualcomm.com/developer/hardware/qualcomm-iq-9075-evaluation-kit-evk)-nya tersedia untuk evaluasi, sedangkan [Product Longevity Program](https://www.qualcomm.com/internet-of-things/products/product-longevity-program) masih menyebut IQ-9075 Sampling dan mencantumkan dukungan hingga 2038. Prefix SKU pemesanan adalah QCS9075; Qualcomm mengiklankan konfigurasi 50 dan 100 dense INT8 TOPS serta dukungan Ubuntu/Yocto. Status produksi perlu dikonfirmasi untuk SKU pesanan yang tepat, dan integrasi LibreYOLO sebaiknya mencatat SKU itu alih-alih membuat folder generik bernama "Qualcomm".

### DEEPX

Akselerator DX-M1/DX-M1M DEEPX memakai DXNN SDK. DX-COM mengompilasi dan mengkuantisasi model, DX-RT menjalankannya, dan artefak deployment memakai format `.dxnn`. Perusahaan menerbitkan [DX-AllSuite](https://github.com/DEEPX-AI/dx-all-suite), contoh aplikasi di [DX-APP](https://github.com/DEEPX-AI/dx_app), serta [Model Zoo daring](https://developer.deepx.ai/modelzoo/) yang besar.

Katalog publik mencatat 354 entri dengan DX-COM 2.4.0 dan DX-RT 3.4.0 saat diperiksa pada 15 Agustus 2026. Cakupan terdokumentasinya mencakup YOLOv3 hingga YOLO11 dan YOLO26, YOLOX, SSD, EfficientDet, NanoDet, DAMO-YOLO, segmentasi YOLO, pose, dan bounding box berorientasi. DEEPX merupakan kandidat integrasi yang masuk akal karena batas compiler/runtime dan artefak deployment dinamai dengan jelas, sementara cakupan vision publiknya luas.

## SoC industri adalah beberapa ekosistem, bukan satu

Vendor industri sering menawarkan siklus hidup produk lebih panjang serta integrasi kamera, keselamatan, dan real-time yang lebih baik dibanding SoC board maker. Stack AI-nya dapat lebih rumit karena satu vendor mungkin mengirim beberapa arsitektur NPU yang tidak berkaitan sekaligus.

### Texas Instruments

Prosesor edge AI TI saat ini mencakup AM62A, AM67A, AM68A, AM69A, dan perangkat TDA4 terkait. Jalur perangkat lunaknya menggabungkan Processor SDK Linux, compiler/runtime TIDL, [Edge AI TIDL Tools](https://github.com/TexasInstruments/edgeai-tidl-tools), dan [Edge AI Model Zoo](https://github.com/TexasInstruments/edgeai-modelzoo). TIDL dapat terintegrasi dengan ONNX Runtime, TensorFlow Lite, dan runtime aplikasi lain sambil meng-offload subgraf yang didukung.

TI menerbitkan lebih dari sekadar klaim impor framework: zoo berisi model hasil konversi dan metadata performa untuk deteksi objek, segmentasi, pose, klasifikasi, kedalaman, dan task lain. TI juga memperingatkan bahwa artefak model zoo merupakan titik awal pengembangan, bukan otomatis aset siap produksi. Catatan ini berharga dan sering tidak ada dalam perbandingan vendor.

TI juga mencantumkan [TDA54-Q1](https://www.ti.com/product/TDA54-Q1) sebagai Preview, dengan hingga empat NPU C7 dan hingga 400 TOPS pada komponen itu; keluarga TDA5 yang lebih luas diiklankan hingga 1.200 TOPS. Itu klaim produk vendor untuk generasi baru, bukan izin untuk menganggap validasi model AM6xA/TDA4 TIDL berlaku di TDA54-Q1 sebelum TI menerbitkan matriks khusus target.

### NXP

NXP memerlukan setidaknya empat deskripsi backend:

| Target NXP | Akselerator | Jalur compiler/runtime |
|---|---|---|
| i.MX 8M Plus | NPU VeriSilicon Vivante 2.3-TOPS | eIQ dengan TIM-VX/VX delegate |
| i.MX 93 | Arm Ethos-U65 | TFLite terkuantisasi plus Arm Vela |
| i.MX 95 | NPU NXP eIQ Neutron | Neutron Converter dan runtime eIQ |
| Ara-1 / [Ara240](https://www.nxp.com/products/ARA240) | NPU diskret turunan Kinara; Ara240 mengiklankan hingga 40 eTOPS | Ara SDK, terintegrasi dengan narasi eIQ yang lebih luas |

[eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo) memuat aset atau resep YOLOv4-tiny, YOLOv8, NanoDet, CenterNet, FastestDet, SSD Lite, YOLACT, dan model lain; entri YOLOv5 hanya dokumentasi. Entri yang divalidasi untuk satu backend bukan bukti untuk keempatnya. [Panduan ekspor YOLO NXP untuk platform i.MX](https://community.nxp.com/t5/i-MX-Processors-Knowledge-Base/Exporting-YOLO-Models-for-NXP-i-MX-Platforms/ta-p/2381361) menunjukkan persoalan konversi khusus target ini. NXP mengatakan [eIQ Toolkit monolitik berhenti diperbarui setelah versi 1.17 pada Q3 2025](https://community.nxp.com/t5/eIQ-Machine-Learning-Software/eIQ-FAQ/ta-p/1099741); alur saat ini memakai paket mandiri seperti eIQ Neutron SDK.

NXP [menyelesaikan akuisisi Kinara pada Oktober 2025](https://media.nxp.com/news-releases/news-release-details/nxp-completes-acquisitions-aviva-links-and-kinara-advance/). [Lembar fakta Ara240](https://www.nxp.com/docs/en/fact-sheet/ARA240DNPUFS.pdf) menyebut hingga 40 eTOPS yang diiklankan vendor dan 313 gambar per detik untuk YOLOv8n. NXP mencantumkan Ara SDK dan eIQ Toolkit di halaman produk, tetapi itu tidak membuktikan pertukaran artefak dengan [jalur Neutron i.MX 95](https://eiq.nxp.com/learning-hub/convQuant/neutron.html) yang terpisah. [Modul M.2 16 GB](https://www.nxp.com/design/design-center/development-boards-and-designs/ARA2-M2-16G-GT) aktif, sementara opsi USB masih praproduksi. NXP mendefinisikan "e" pada eTOPS sebagai "equivalent", bukan "effective"; ukuran itu bukan metrik dense-INT8 TOPS vendor lain dan tidak seharusnya dibandingkan langsung.

### STMicroelectronics

[Perangkat STM32N6x7](https://www.st.com/en/microcontrollers-microprocessors/stm32n6-series.html), termasuk STM32N657 dan STM32N647, menghadirkan akselerator Neural-ART 600-GOPS ke produk kelas MCU; lini umum N6x5 tidak memiliki akselerator itu. [STM32Cube AI Studio](https://www.st.com/en/development-tools/stedgeai-cubeai.html) dan ST Edge AI Core menganalisis serta mengoptimalkan model yang diimpor, sementara [STM32 AI Model Zoo Services](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) menyediakan alur deployment, optimasi, benchmarking, dan contoh.

[Ikhtisar deteksi objek saat ini](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/main/object_detection/docs/README_OVERVIEW.md) mendokumentasikan Tiny YOLOv2, YOLOv5u, YOLOv8, YOLO11, YOLO26, dan ST-YOLOX, juga layanan klasifikasi, pose, dan segmentasi. Kelas performanya tidak sama dengan kartu PCIe 200-TOPS. Platform ini menarik karena input kamera, inferensi, dan kontrol dapat berjalan dalam batas daya dan memori tertanam yang ketat.

### Renesas

Prosesor Renesas RZ/V mengintegrasikan akselerator DRP-AI. Perangkat lunaknya berkembang dari DRP-AI Translator dan DRP-AI TVM menuju [RUHMI](https://www.renesas.com/en/software-tool/ruhmi-framework), yang ditenagai teknologi MERA dari EdgeCortix. [Repositori RZ/V DRP-AI TVM](https://github.com/renesas-rz/rzv_drp-ai_tvm) terbuka dan [daftar validasi RZ/V2H](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/model_list/Model_List_V2H.md) menyebut varian n/s/m YOLOv5, YOLOv8, YOLO11, dan YOLO26, YOLOX, jaringan pose dan segmentasi, serta model klasifikasi.

Renesas adalah target kredibel untuk robotika dan industri, tetapi board, generasi DRP-AI, versi translator, dan pascapemrosesan CPU yang tepat perlu dicatat agar hasil dapat direproduksi.

## Sensor pintar dan prosesor yang berfokus pada kamera

### Sony IMX500

IMX500 Sony bukan prosesor aplikasi biasa. Sensor ini menumpuk pengindraan gambar dan pemrosesan AI agar inferensi dapat terjadi di sensor, sehingga mengurangi jumlah data gambar yang keluar dari kamera. Raspberry Pi AI Camera membuat arsitektur itu mudah diakses.

Repositori model [Raspberry Pi IMX500](https://github.com/raspberrypi/imx500-models) resmi menerbitkan model YOLOv8n, YOLO11n, EfficientDet Lite0, NanoDet+, dan SSD MobileNetV2 FPN Lite terpaket. [Dokumentasi AI Camera](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) menjelaskan konversi, pengemasan, dan pascapemrosesan di sensor. Unit deployment-nya paket RPK, bukan berkas ONNX generik, dan memori sensor serta batasan operator menjadi batas desain utama. [Halaman produk](https://www.raspberrypi.com/products/ai-camera/) Raspberry Pi menyatakan kamera ini akan tetap diproduksi setidaknya sampai Januari 2028.

### Ambarella

Prosesor CVflow Ambarella telah lama digunakan pada kamera, drone, dan vision otomotif. Keluarga produk saat ini mencakup CV72/CV75 untuk kamera, CV5/CV52 untuk sistem vision berperforma lebih tinggi, serta perangkat otomotif CV3-AD. Cooper Developer Platform dan alur kompilasi/runtime CVflow merupakan stack perangkat lunak yang disebut secara publik.

[Model garden developer publik](https://www.ambarella.com/developer/model-garden/) mencantumkan YOLOX, RTMDet, DeepLabV3+, TopFormer, OWL-ViT, dan model multimodal. Namun, dokumentasi compiler terperinci dan unduhan tetap ditujukan kepada mitra. Karena itu, integrasi Ambarella sebaiknya dimulai lewat hubungan dengan vendor atau OEM, bukan dengan menganggap paket pip publik tersedia.

### Synaptics Astra

Synaptics memiliki tiga target terkait. Sistem Astra SL1600/SL1680 memakai [toolkit SyNAP](https://developer.synaptics.com/docs/synap/introduction), yang mengonversi model sumber beserta deskripsi YAML menjadi paket `model.synap`. Perangkat SL2611/13/15/17/19 yang lebih baru memakai [platform Torq](https://developer.synaptics.com/docs/torq/introduction), alur berbasis IREE/MLIR yang menghasilkan `.vmfb`. [Seri SR100](https://developer.synaptics.com/docs/sr/introduction-sr) adalah keluarga MCU terpisah berbasis Cortex-M55 dan Ethos-U55 dengan SDK tersendiri. Artefak dan API tersebut tidak saling dapat dipertukarkan.

Tutorial [benchmark YOLO](https://developer.synaptics.com/docs/sl/tutorials/vision/benchmark-yolo) resmi sangat konkret: mengekspor YOLOv8 ke TFLite, melakukan kalibrasi UINT8 asimetris di SyNAP, mengompilasi untuk SL1680, lalu menjalankan `synap_cli` dan aplikasi deteksi objek. Tutorial itu membuktikan YOLOv8n/v8s pada SL1680 dan menjelaskan dukungan SyNAP hingga YOLO11. [Panduan deteksi objek SL261x](https://developer.synaptics.com/docs/sl/sl2600/getting-started/object-detection) terpisah menjalankan YOLOv8 dari `.vmfb` Torq. Ini contoh resmi khusus target, bukan klaim bahwa semua graf YOLO didukung pada ketiga keluarga.

### MediaTek Genio

MediaTek pantas mendapat entri penuh karena [IoT AI Hub](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/index.html) saat ini mengekspos matriks perangkat keras/perangkat lunak, paket model, dan tabel benchmark yang dulu sulit diaudit oleh survei pasar. Genio 360/360P/420/520/720 menggunakan NeuroPilot 8 dengan MDLA 5.3; Genio 510/700 menggunakan NP6 dengan MDLA 3.0; Genio 1200 menggunakan NP6 dengan MDLA 2.0. MediaTek menyatakan bahwa generasi NeuroPilot dan MDLA, set operator, compiler, dan runtime terikat versi sepanjang siklus hidup tiap SoC.

Jalur AI analitis berpusat pada TFLite:

```text
PyTorch model
    -> NeuroPilot Converter and representative calibration
    -> quantized .tflite
    -> version-matched ncc-tflite compiler
    -> chip-generation-specific .dla
    -> Neuron Runtime on MDLA
```

Produk Genio yang lebih baru juga menyediakan delegasi LiteRT daring dan jalur CPU/NPU ONNX Runtime pada sistem operasi yang didukung. Mode itu berbeda dari `.dla` offline. [Panduan arsitektur perangkat lunak](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/software_architecture.html) dan [matriks sumber daya](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/related_resource.html) menjelaskan pembedaan tersebut.

[Tabel model analitis](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/model_zoo/litert_analytical.html) resmi menerbitkan entri benchmark YOLOv5s dan YOLOv8s serta panduan konversi untuk beberapa generasi MDLA. Tabel itu secara eksplisit tidak mendistribusikan artefak YOLO hasil konversi karena pembatasan AGPL-3.0. Pengukuran offline Quant8 640 x 640 mencakup 5.35 ms dan 8.04 ms pada Genio 720. [Halaman model YOLOv8s](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/model_zoo/litert_analytical/YOLOv8s.html) menerbitkan tensor input/keluaran, waktu khusus backend, dan peringatan tentang operasi khusus MediaTek. Angka tersebut adalah benchmark vendor, bukan hasil kamera end-to-end.

Akses menjadi batas utama. Dokumentasi Yocto publik cukup terperinci, tetapi bundel converter/compiler NP8 all-in-one saat ini dan banyak materi Android dinyatakan sebagai sumber daya NDA atau pelanggan langsung. Akun developer biasa tidak memberikan akses yang sama dengan akun pelanggan MediaTek Online. Karena itu, LibreYOLO sebaiknya menganggap Genio terbukti secara teknis, tetapi integrasi compiler bring-your-own-model yang dapat direproduksi bergantung pada kemitraan.

## Ekosistem edge AI yang berfokus pada Tiongkok

Beberapa platform vision berbiaya rendah yang paling mumpuni dan mudah diakses kurang terwakili dalam daftar pasar berbahasa Inggris.

### D-Robotics dan platform BPU turunan Horizon

D-Robotics memelihara ekosistem board developer RDK berbasis akselerator BPU pada produk RDK X3, X5, Ultra, dan seri S100 yang lebih baru. Branch `rdk_x5` saat ini dalam [RDK Model Zoo](https://github.com/D-Robotics/rdk_model_zoo) mendokumentasikan YOLOv5/v5u, YOLOv8/9/10/11/12/13/26, YOLOE, YOLO-World, serta jalur deteksi, segmentasi, pose, klasifikasi, OCR, dan CLIP untuk RDK X5. X3 memakai branch terpisah, sedangkan delivery seri S saat ini ada di branch [`rdk_s`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_s) pada zoo utama; repositori [`rdk_model_zoo_s`](https://github.com/D-Robotics/rdk_model_zoo_s) yang lebih lama berisi demo historis. Jadi, daftar X5 modern bukan bukti bahwa setiap model divalidasi pada X3, Ultra, atau S100.

OpenExplorer/Algorithm Toolchain mengimpor ONNX atau Caffe untuk PTQ dan mendukung alur QAT berorientasi PyTorch. Deployment berbeda menurut platform: RDK X5 saat ini memakai `.bin` melalui `hbm_runtime` di atas `libdnn`, sedangkan alur utama `rdk_s` saat ini memakai `.hbm` melalui API `hbm_runtime` bernama sama di atas `libhbucp`; X3 mempertahankan branch dan antarmuka inferensi yang lebih lama. Materi lama `rdk_model_zoo_s` juga menjelaskan jalur historis `.bin` dan `.hbm`, sehingga artefak harus tetap dipasangkan dengan branch dan toolchain yang sesuai. Operator yang tidak didukung dapat beralih ke CPU. Contoh publiknya bagus, tetapi pemasangan versi RDK OS, firmware board, toolchain, dan biner model tetap menjadi bagian kontrak kompatibilitas.

### AXERA

Keluarga AX650/AX630/AX620 AXERA dipakai di kamera AI ringkas dan board seperti lini MaixCAM dari Sipeed. Pulsar2 mengimpor ONNX, melakukan kalibrasi dan analisis presisi, lalu menghasilkan `.axmodel`; AXEngine menjalankan artefak tersebut.

[Sampel AXERA resmi](https://github.com/AXERA-TECH/ax-samples) mencakup YOLOv5/6/7/8/9/10/11/13/26, YOLOX, dan YOLO-World, dengan dukungan task dan chip berbeda menurut platform. Pada target yang didukung, contoh YOLO26 saat ini mencakup deteksi, pose, segmentasi, dan OBB. [Contoh konversi Pulsar2](https://pulsar2-docs.readthedocs.io/en/latest/appendix/model_convert_examples.html) menunjukkan pekerjaan sebenarnya: nama tensor yang tepat, pemotongan keluaran, transformasi layout, arsip kalibrasi, dan pengaturan perangkat keras target.

### SOPHGO

TPU-MLIR SOPHGO termasuk stack compiler yang menarik untuk integrasi independen karena compiler itu sendiri open source. Compiler mengimpor ONNX, PyTorch, TFLite, dan Caffe, menurunkan serta mengkuantisasi graf, lalu menghasilkan `.bmodel` untuk target BM atau `.cvimodel` untuk perangkat keras CV18xx.

[Proyek TPU-MLIR](https://github.com/sophgo/tpu-mlir) mendukung alur FP32, BF16, FP16, dan INT8 sesuai target. [Koleksi demo SOPHON](https://github.com/sophgo/sophon-demo/blob/release/README_EN.md) menyebut YOLOv3/4/5/7/8/9/10/11/12/26, YOLOX, PP-YOLOE, YOLO-World, varian OBB/segmentasi, SSD, CenterNet, RetinaFace, SAM/SAM2, dan OCR. Seperti biasa, demo pada BM1684X tidak memvalidasi artefak yang sama untuk BM1688 atau CV18xx.

### Huawei Ascend

Lini inferensi edge Huawei mencakup silikon Ascend 310/310P/310B pada produk Atlas 200I dan 300I. CANN menyediakan compiler ATC, runtime AscendCL, dan kernel operator khusus chip; ATC mengubah ONNX dan representasi sumber lain menjadi model `.om` offline.

Dokumentasi [Atlas 200I DK A2](https://www.hiascend.com/en/hardware/developer-kit-a2/specification) saat ini serta sampel CANN menyediakan jalur `.om` YOLOv5 untuk Ascend 310B. [Ascend ModelZoo](https://github.com/Ascend/modelzoo) yang lebih luas dan lama berisi YOLOv2/3/4/5, YOLOX, YOLOR, SSD, RetinaNet, Mask R-CNN, pose, dan model segmentasi, tetapi tidak boleh disajikan seolah tiap entri divalidasi ulang pada 310B. CANN, kernel, firmware, dan SoC harus cocok. `.om` untuk Ascend310P bukan artefak Ascend generik.

### Cambricon

Akselerator MLU Cambricon memakai Neuware dan mesin inferensi MagicMind. [Repositori MagicMind Cloud](https://github.com/Cambricon/magicmind_cloud) publik dipatok ke MagicMind 1.7 dan matriks MLU370-X4/S4; itu bukti kompatibilitas historis, bukan matriks SDK terkini atau validasi MLU270. Repositori itu mendokumentasikan jalur PyTorch, ONNX, Caffe, dan Paddle; dukungan framework TensorFlow dihapus pada 1.7 walaupun contoh TensorFlow historis masih ada. [Portal developer seri 3](https://developer.cambricon.com/index/document/index/classid/3.html) Cambricon saat ini mencantumkan rilis SDK keseluruhan yang lebih baru, sehingga repositori contoh publik dan stack komersial saat ini tidak boleh dianggap satu versi.

Repositori model cloud [MagicMind](https://github.com/Cambricon/magicmind_cloud) resmi menerbitkan matriks MLU370-X4/S4 yang sangat langsung. Matriks itu menandai contoh YOLOv3, Tiny-YOLOv3, YOLOv4, YOLOv5, YOLOv7, YOLOv8, PP-YOLOE, SSD, RetinaFace, RetinaNet, Mask R-CNN, DeepLab, dan UNet, termasuk ketersediaan contoh C++ atau Python. Buktinya kuat; hambatan utamanya adalah akses SDK dan container melalui jalur Cambricon.

### Canaan/Kendryte

Chip KPU K230/K230D Kendryte saat ini dan K210/K510 lama penting pada segmen board murah dan kamera pintar. [Compiler nncase](https://github.com/kendryte/nncase) mengimpor ONNX atau TFLite, memakai data kalibrasi untuk konversi fixed-point, mensimulasikan hasil pada host, lalu menghasilkan `.kmodel`.

[Panduan pengembangan nncase K230](https://github.com/kendryte/k230_docs/blob/main/en/01_software/board/ai/K230_nncase_Development_Guide.md) resmi memandu kompilasi, simulasi, dan menjalankan YOLOv5s. [Katalog demo AI](https://github.com/kendryte/k230_docs/blob/main/en/02_applications/ai_demos/K230_AI_Demo_Introduction.md) terpisah mencakup artefak deteksi, segmentasi, dan pose YOLOv8n, sedangkan [changelog CanMV](https://github.com/kendryte/canmv_k230/blob/canmv_k230/CHANGELOG.md) saat ini menambahkan contoh klasifikasi, deteksi, segmentasi, OBB, dan pose YOLOv8/11/26 yang dioptimalkan. Hasil [YOLOv5s pada datasheet](https://github.com/kendryte/k230_docs/blob/main/en/00_hardware/K230_datasheet.md) merupakan benchmark yang benar-benar diterbitkan; versi compiler dan plug-in KPU biner harus cocok dengan SDK board.

### Allwinner

V853 Allwinner menggabungkan perangkat media berorientasi kamera dengan NPU Vivante 1-TOPS. [Panduan NPU berbahasa Inggris resmi](https://docs.aw-ol.com/v853/en/npu/dev_npu/) menjelaskan impor Acuity/Pegasus, kuantisasi, verifikasi, dan deployment melalui `viplite`. Halaman model Allwinner dan [panduan YOLOv5](https://docs.aw-ol.com/v853/npu/npu_yolov5/) menyebut YOLOv2/v3/v4/v4-tiny/v5/v5s, RetinaNet, MobileNet, ResNet, serta jaringan wajah/orang. Dokumentasi publiknya berguna, walaupun akses paket SDK mungkin perlu akun.

## Platform Korea, Taiwan, dan Tiongkok yang kurang diperhatikan

### Mobilint

Akselerator REGULUS dan ARIES dari Mobilint menggunakan qb SDK dan artefak `.mxq`. [Model zoo](https://github.com/mobilint/mblt-model-zoo/blob/master/mblt_model_zoo/vision/README.md) publik mencantumkan deteksi YOLOv3/v5/v7/v8/v9/v10/11/12/26, serta varian segmentasi YOLO, pose, dan OBB. Halaman [ARIES](https://www.mobilint.com/aries/mla100) bahkan menerbitkan hasil khusus YOLO11s dan YOLO26m. Bukti integrasinya kuat, walaupun akses SDK/perangkat keras tetap komersial.

### Sunplus

SP7350/C3V Sunplus memakai stack NPU turunan Vivante, tetapi paketnya berbeda dari Allwinner atau Amlogic lama. Alur publiknya menggabungkan konversi Acuity, lingkungan Docker NPU, runtime SNNF, dan keluaran `.nb`. [Panduan YOLOv8 khusus](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2427224156/SP7350) resmi menjelaskan konversi serta deployment, bukan sekadar mengklaim dukungan framework. IP terkait tidak membuat `.nb` portabel ke SoC lain berbasis Vivante.

### ESWIN Computing

Keluarga RISC-V ESWIN mencakup EIC7700/EIC7700X dan EIC7702/EIC7702X dual-die, dengan EIC7700 muncul pada board yang dikirim seperti Milk-V Megrez. ENNP berisi EsQuant, compiler EsAAC saat ini, pembuatan data acuan, simulasi, dan runtime ESSDK, dengan keluaran `.model`; materi ENNP lama memakai nama `ennc-compile`. [Halaman EsAAC](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/esaac) saat ini mendokumentasikan input ONNX, sehingga framework lain perlu diekspor atau dikonversi ke IR yang didukung. [Ikhtisar ENNP](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/introduction) dan [tutorial YOLOv3](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/yolov3) yang dihosting Milk-V menunjukkan jalur end-to-end publik, tetapi bukan matriks model/akurasi luas terkini.

### Nuvoton M55M1

Nuvoton M55M1 adalah desain MCU Cortex-M55 plus Ethos-U55-256, bukan prosesor aplikasi Linux. NuEdgeWise/NuML dan Arm Vela menyiapkan model TFLite Micro terkuantisasi. [Repositori NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise) publik menyebut YOLOv8-nano, YOLOX-nano, YOLO Fastest v1.1, SSD-MobileNet FPNLite, dan beberapa classifier. Contoh jaringan kecil ini merupakan bukti yang kredibel untuk kelas memori/daya perangkat, bukan untuk varian YOLO standar 640 piksel.

### Rebellions dan FuriosaAI

Lini ATOM Rebellions memakai RBLN SDK dan artefak `.rbln` terkompilasi. [Rilis dukungan resmi](https://docs.rbln.ai/v0.8.2/supports/release_note.html) mencantumkan YOLOv3 tiny/full/SPP, YOLOv5 n hingga x, YOLOv6 n hingga l, varian YOLOv7, dan YOLOv8 n hingga x. [Matriks dukungan kartu](https://docs.rbln.ai/latest/supports/version_matrix.html) saat ini menandai ATOM CA02 dan ATOM+ CA12 sebagai EoL, sedangkan ATOM+ CA22 dan ATOM-Max CA25 berstatus Active. ATOM-Lite CA21 memiliki halaman produk, tetapi tidak ada dalam matriks SDK itu, sehingga status toolchain saat ini belum jelas. Dokumentasi publik, tetapi wheel compiler membutuhkan kredensial Rebellions Portal.

FuriosaAI harus dipisahkan menurut generasi. Warboy adalah akselerator vision yang lebih lama, dengan artefak INT8 `.enf` dan [model zoo resmi](https://developer.furiosa.ai/furiosa-models/latest/) berisi SSD, YOLOv5M/L, dan YOLOv7-w6-pose. RNGD memakai [stack `.fxb`](https://developer.furiosa.ai/latest/en/furiosa_llm/fxb.html) berbeda yang terutama ditujukan untuk beban kerja LLM/VLM. [Peta jalan](https://developer.furiosa.ai/latest/en/overview/roadmap.html) Furiosa saat ini mencatat dukungan vision YOLOv8m selesai pada Q4 2024, tetapi permukaan model dan performa publik terkini berpusat pada LLM/VLM dan tidak menampilkan matriks akurasi/performa YOLOv8m terperinci. Ini bukti generasi yang dirilis, bukan klaim kompatibilitas Warboy.

### Realtek, Telechips, T-Head, dan SigmaStar

Keempat platform ini nyata, tetapi permukaan bring-your-own-model-nya lebih sempit, lebih lama, atau aksesnya terbatas:

| Platform | Bukti publik yang dapat direproduksi | Batasan yang perlu dipertahankan |
|---|---|---|
| Realtek AmebaPro2 | [SDK Arduino/FreeRTOS](https://github.com/Ameba-AIoT/ameba-arduino-pro2) mengemas model `.nb` YOLOv3/4/7-tiny, SCRFD, dan MobileFaceNet | Konversi model khusus memerlukan pendaftaran minat/menghubungi Realtek |
| Telechips TOPST TCC7500 | [Proyek 2026 resmi](https://docs.topst.ai/blog/31) mengonversi dan men-deploy YOLOv8s | Konversi UFLD v1/v2 gagal pada layer yang tidak didukung; artikel hanya mengusulkan solusi pemisahan/pascapemrosesan. Sumber daya compiler/operator penuh biasanya perlu izin melalui email |
| T-Head TH1520 | [Panduan aplikasi Sipeed](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) menjalankan YOLOv5n/s dengan HHB dan CSI-NN2/SHL | Stack publik tersedia, tetapi pemeliharaan dan kejelasan versi tertinggal dari pemimpin saat ini |
| SigmaStar SSU9383CM | Dokumentasi saat ini menetapkan API runtime MI_IPU; materi SGS_IPU lama mendokumentasikan `.sim` ke `sgsimg.img`, dan postprocessor lama menyebut SSD serta YOLOv1/2/3 | Tidak ada bukti publik bahwa artefak compiler lama atau jalur model bernama berlaku pada SSU9383CM |

### Smart vision HiSilicon berbeda dari Huawei Ascend

SoC kamera Hi3516CV610/DV500, Hi3519DV500, dan Hi3403V100 dari HiSilicon mengekspos alur ATC-ke-`.om` melalui runtime NNN/SVP-NNN. [Model zoo HiSpark](https://gitee.com/HiSpark/modelzoo/blob/master/README.md) resmi mencantumkan YOLOv3/4/5/6/7/8/9/10/11, segmentasi/pose YOLO11, OBB/World/segmentasi YOLOv8, OCR, kedalaman, dan TinySAM dalam matriks khusus target, terutama untuk Hi3403 dan Hi3591P. Itu tidak membuktikan setiap entri berjalan pada semua SoC smart vision di atas. Sebagian entri masih berupa rencana, sehingga contoh yang sudah dirilis dan dukungan yang direncanakan harus tetap dipisahkan. Repositori menyatakan model ModelZoo yang disediakan hanya untuk penggunaan nonkomersial. Istilah ATC/`.om` yang sama tidak membuktikan kompatibilitas artefak atau runtime dengan lini produk Ascend berbasis CANN.

## Perusahaan akselerator baru dan khusus

### MemryX

Modul MX3 MemryX menggunakan arsitektur dataflow dan dapat menggabungkan beberapa chip. [Neural Compiler](https://developer.memryx.com/tools/neural_compiler.html) menerima model ONNX, TensorFlow Lite, Keras, dan TensorFlow, lalu menghasilkan paket DFP yang dipakai runtime akselerator. [API runtime](https://developer.memryx.com/api/accelerator/accelerator.html) mendukung pipeline multi-model dan streaming.

Dokumentasi dan contoh MemryX mencakup prapemrosesan dan pascapemrosesan YOLO yang dioptimalkan untuk deteksi, segmentasi, dan pose. [Catatan rilis SDK](https://developer.memryx.com/release_notes.html) secara eksplisit menyebut dukungan YOLOv10, YOLO11, dan YOLO26. Materi publiknya bermanfaat secara teknis, walaupun tidak memiliki satu tabel model/akurasi/perangkat sederhana seperti milik Axelera.

### Kneron

Produk KL520/KL530/KL630/KL720/KL730 Kneron mencakup akselerator USB dan tertanam berukuran kecil. Kneron PLUS mencakup runtime aplikasi, sedangkan Model Toolchain 0.33.1 saat ini mendokumentasikan konversi, kuantisasi, evaluasi, simulasi, dan kompilasi `.nef` untuk target tersebut. KL830 muncul di beberapa [API runtime PLUS](https://doc.kneron.com/docs/plus_c/introduction/run_examples/), tetapi tidak pada [daftar target compiler](https://doc.kneron.com/docs/toolchain/manual_1_overview/) saat ini; klaim kompilasi `.nef` generik tidak boleh dialihkan ke KL830 tanpa konfirmasi vendor.

[Contoh YOLO](https://doc.kneron.com/docs/toolchain/appendix/yolo_example/) resmi menunjukkan proses memakai Tiny-YOLOv3, sementara materi Kneron lain mencakup pelatihan dan deployment YOLOv5. Buktinya kredibel, tetapi lebih sempit daripada matriks YOLO modern Hailo, Axelera, Rockchip, atau DEEPX.

### SiMa.ai

MLSoC SiMa.ai dan [platform Modalix 50-TOPS produksi](https://sima.ai/press-release/sima-ai-next-gen-platform-for-physical-ai-in-production/) memakai lingkungan perangkat lunak Palette. ModelSDK, compiler MLA, dan ModelExecutor menangani impor, kuantisasi, kompilasi, profiling, serta eksekusi. Tool mendukung alur berorientasi INT8, INT16, dan BF16 bergantung pada beban kerja dan produk.

Catatan rilis perusahaan menyebut pipeline YOLOv7, deteksi/pose/segmentasi YOLOv8, YOLOX dan segmentasi YOLOX, DETR, Mask R-CNN, serta EfficientDet yang tervalidasi. Batasan saat ini tidak boleh disembunyikan: [catatan Palette SDK 2.1](https://docs.sima.ai/v2.1.2/pages/release_notes/2.1.html) menyebut QAT tidak berfungsi akibat regresi kuantisasi Python/PT2E. Solusi terdokumentasi adalah membuat ONNX beranotasi di SDK 2.0 lalu mengompilasinya dengan 2.1. Akses dan pengadaan berorientasi enterprise.

Batas deployment-nya juga terdokumentasi: [alur kompilasi ModelSDK](https://docs.sima.ai/v2.1.2/pages/model-sdk/compilation.html) menghasilkan `.tar.gz` terkompilasi dan dapat membuat ELF yang dapat dieksekusi, sedangkan [MPK Tool](https://docs.sima.ai/v2.1.1/pages/palette/mpk_tools.html) mengemas `.mpk` yang dapat di-deploy. Keluaran ini terikat pada target MLSoC/Modalix dan rilis Palette.

### EdgeCortix

SAKURA-II adalah akselerator 60-TOPS yang diiklankan untuk vision dan AI generatif, dikompilasi melalui framework perangkat lunak MERA. EdgeCortix juga memasok teknologi MERA untuk alur RUHMI Renesas yang lebih baru.

[Materi produk SAKURA-II](https://www.edgecortix.com/en/edgecortix-sakura-ii-accelerator-brief) menetapkan perangkat keras dan compiler, sedangkan [perangkat keras M.2/PCIe](https://www.edgecortix.com/en/hardware) ditawarkan melalui permintaan uji coba atau pesanan. Tidak ditemukan matriks kompatibilitas YOLO publik terkini yang cukup presisi. Ketiadaan itu berguna sebagai informasi pembelian: minta validasi model sebelum membeli perangkat.

### BrainChip

Akida BrainChip adalah prosesor domain-event neuromorfik, bukan NPU tensor padat konvensional. [Lingkungan MetaTF](https://brainchip.com/metatf-dev-tools/) terdiri dari paket Python yang dapat dipasang untuk pembuatan model, kuantisasi bit rendah, konversi, simulasi, dan eksekusi perangkat keras. BrainChip kini menjual [koprosesor AKD1500 yang dikirim dalam bentuk M.2](https://brainchip.com/brainchip-akd1500-now-available-in-compact-m-2-form-factor-enabling-fanless-edge-ai-in-industrial-and-commercial-designs/), bersama perangkat keras AKD1000 lama dan IP Akida 2.

[Kartu model Akida 2](https://brainchip.com/wp-content/uploads/2025/04/Akida-2-Model-Card-V1.1-Mar.25.pdf) saat ini mencantumkan detektor YOLOv2 AkidaNet0.5, CenterNet, AkidaUNet, dan jaringan pengenalan wajah. Materi dukungan lain mencakup FOMO dan beban kerja berbasis event. Akida mendukung bobot dan aktivasi bit sangat rendah, tetapi deployment yang berhasil mungkin memerlukan konversi yang sadar arsitektur, bukan memperlakukannya sebagai target ONNX INT8 generik. Hasil model zoo AKD1000 atau Akida 2 tidak boleh diklaim untuk AKD1500 sampai laporan pemetaan/kecocokan secara eksplisit memvalidasinya.

### Blaize

Blaize menjual produk Pathfinder dan Xplorer berbasis P1600, dengan Picasso SDK, NetDeploy, dan AI Studio sebagai jalur perangkat lunak. [Halaman produk](https://www.blaize.com/products/) jelas menargetkan vision dan edge AI, tetapi matriks kompatibilitas model per model yang publik dan terkini tidak tersedia.

Karena itu, panduan ini mengklasifikasikan Blaize sebagai komersial dan terbatas aksesnya, bukan mengisi kekosongan dengan klaim komunitas. Laporan kompilasi dan akurasi dari vendor untuk model yang dituju sebaiknya menjadi prasyarat.

## Vision TinyML dan endpoint

### Arm Ethos-U dan Alif

Arm melisensikan IP NPU Ethos-U55, U65, dan U85 kepada perusahaan chip. Compiler [Vela](https://developer.arm.com/edge-ai/arm-cortex-m-and-ethos-u) mengambil graf LiteRT/TFLite terkuantisasi dan menulis ulang subgraf yang didukung menjadi operasi khusus Ethos-U. Operasi yang tidak didukung dapat tetap berjalan di CPU, sehingga "aplikasi berjalan" dan "seluruh jaringan berjalan di NPU" adalah dua klaim berbeda.

Implementasi nyata mencakup Ethos-U55 pada Alif Ensemble E7, Ethos-U65 pada NXP i.MX 93, dan Ethos-U85 pada sistem yang lebih baru. [E7 AI/ML AppKit](https://alifsemi.com/support/kits/ensemble-e7appkit/) Alif mendokumentasikan dua akselerator ML dan mendemonstrasikan alur TFLite-ke-Vela E7; [Ensemble E8](https://alifsemi.com/ensemble-e8-series/) menggabungkan satu Ethos-U85 dengan dua NPU Ethos-U55. Alif juga menerbitkan [benchmark detektor wajah YOLO-Fastest INT8](https://alifsemi.com/faster-ai-mcu-inferencing-low-power-consumption/) tingkat keluarga pada 192 kali 192 dengan Cortex-M55 plus Ethos-U55, tetapi bukan matriks YOLO modern luas per target. Sistem yang terbatas memori ini cocok untuk jaringan klasifikasi dan deteksi kecil, bukan detektor YOLO sembarang 640 piksel hanya karena dapat direpresentasikan sebagai TFLite.

### Infineon PSOC Edge, Himax WiseEye2, dan Analog Devices MAX7800x

Keluarga [PSOC Edge E83](https://documentation.infineon.com/psocedge/docs/xsk1761304020519) dan E84 dari Infineon memasangkan Cortex-M55 dengan Ethos-U55 dan menambahkan blok NNLite terpisah di sisi M33 berdaya rendah. [Manual arsitektur E84](https://documentation.infineon.com/psocedge/docs/bwb1750411526047) mendokumentasikan bobot 8-bit, aktivasi 8 atau 16-bit, dan alur Vela: operator yang didukung menjadi aliran perintah NPU, sementara pekerjaan yang tidak didukung tetap di CPU. [DEEPCRAFT Model Converter](https://www.infineon.com/design-resources/embedded-software/deepcraft-edge-ai-solutions/deepcraft-model-converter) menerima jalur TFLite, Keras, dan PyTorch, sementara materi arsitektur menyebut MobileNetV1/V2 sebagai kernel representatif, bukan benchmark target. [E84 AI Kit](https://documentation.infineon.com/psocedge/docs/cci1762693051052) mencakup kamera dan memakai ModusToolbox, tetapi materi publik belum menyediakan matriks kompatibilitas YOLO bernama. Ini platform vision yang dapat diprogram dengan bukti tingkat model yang berkembang, bukan target detektor umum tervalidasi.

[HX6538 WiseEye2](https://www.himax.com.tw/products/wiseeye-ai-sensing/wiseeye2-ai-processor/) Himax juga menggabungkan Cortex-M55 dan Ethos-U55 untuk vision selalu aktif. Kisah perangkat lunak publiknya sangat konkret untuk kelas daya ini: [contoh Grove Vision AI Module V2 resmi](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2) mencakup deteksi objek YOLOv8n, pose, klasifikasi gender, deteksi YOLO11n, face mesh, dan PeopleNet. Jalur terpadu memakai TFLite Micro dan Vela, lalu beralih ke CMSIS-NN dan kernel referensi bila perlu. Model dan resolusinya sengaja kecil, bukan bukti bahwa graf YOLO konvensional seukuran server muat.

Analog Devices memakai pendekatan berbeda lewat [akselerator CNN MAX78000/MAX78002](https://www.analog.com/en/products/max78002.html). Tool [`ai8x-synthesis`](https://github.com/analogdevicesinc/ai8x-synthesis) publik mengkuantisasi jaringan yang sudah dilatih dan menghasilkan kode C khusus target, bobot, serta konfigurasi akselerator, bukan paket runtime ONNX portabel. Bukti pihak pertama mencakup [identifikasi wajah](https://www.analog.com/en/resources/app-notes/an-2616.html), deteksi QR TinierSSD di [model zoo ADI](https://ez.analog.com/dsp/software-and-development-tools/edgebench/a/docs-faqs/DF912/available-models-in-adi-model-zoo), dan classifier gambar. Perangkat ini relevan untuk vision endpoint, tetapi tidak ditemukan matriks YOLO modern resmi saat ini.

### GreenWaves GAP9

GAP9 menggabungkan komputasi RISC-V dengan neural engine NE16. NNTool mengimpor dan mengkuantisasi jaringan, sedangkan AutoTiler dan GAP SDK menghasilkan kode deployment. [Menu jaringan saraf GAP9](https://github.com/GreenWaves-Technologies/nn_menu_gap9) publik memuat aplikasi MobileNet, EfficientNet, ResNet, MobileNet SSD, wajah, dan pengenalan. GreenWaves juga menerbitkan proyek [deteksi orang YOLOX](https://github.com/GreenWaves-Technologies/yolox_people_detection) dengan hasil akurasi dan simulasi.

Ini bukti TinyML yang sangat transparan, tetapi SDK lengkap hanya tersedia bagi pelanggan berkualifikasi dan modelnya jauh lebih kecil daripada varian YOLO arus utama 640 x 640.

### Lattice sensAI

Lattice sensAI merupakan alternatif FPGA untuk NPU tetap. sensAI Studio dan Neural Network Compiler memetakan graf yang didukung ke IP akselerator yang dapat dikonfigurasi untuk perangkat ECP5, iCE40 UltraPlus, CrossLink-NX, CertusPro-NX, Avant-E, dan Avant-X. Keluaran terikat pada FPGA yang dipilih, mode akselerator, layout memori, dan bitstream, bukan model runtime portabel.

[Manual Neural Network Compiler](https://www.latticesemi.com/-/media/LatticeSemi/Documents/UserManuals/MQ3/FPGA-UG-02052-8-0-Lattice-Neural-Network-Compiler-Software.ashx?document_id=52343) saat ini memuat tabel topologi yang sangat langsung. Tabel mencantumkan YOLOv1 untuk ECP5, YOLOv5 dan YOLOv8 untuk mode CrossLink-NX dan CertusPro-NX optimized/advanced, serta YOLO11 hanya dengan IP Advanced mode. Tabel juga mencantumkan SSD, MobileNetV2-SSD, ResNet, ENet, SqueezeDet, dan jaringan vision lain, dengan sel FPGA yang tidak didukung ditampilkan. [Advanced CNN Accelerator](https://www.latticesemi.com/en/Products/DesignSoftwareAndIP/IntellectualProperty/IPCore/IPCores05/Advanced-CNN-Accelerator-IP) menargetkan Avant-E/Avant-X/CertusPro-NX dan merupakan IP komersial. Ini bukti dari tabel compiler, bukan janji bahwa satu bitstream mendukung semua topologi yang dicantumkan sekaligus.

### Microchip VectorBlox

[VectorBlox SDK 3.1](https://github.com/Microchip-Vectorblox/VectorBlox-SDK) publik Microchip melakukan prapemrosesan dan mengompilasi jaringan TFLite INT8 terkuantisasi penuh untuk IP CoreVectorBlox yang dapat dikonfigurasi di FPGA PolarFire SoC. `tflite_preprocess` dan `vnnx_compile` menghasilkan aset deployment `.vnnx`, `.hex`, dan `.ucomp`; repositori juga mencakup simulator dan driver C sisi target. Graf TensorFlow, ONNX, dan OpenVINO upstream tetap harus melalui jalur persiapan TFLite/INT8 yang didokumentasikan.

[Matriks tutorial](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/README.md) saat ini mencakup YOLOv5n, deteksi/klasifikasi/OBB/pose/segmentasi YOLOv8, YOLOv9t, varian YOLO sparse/terkompresi, MobileNet, EfficientNet-Lite0, ResNet18, MiDaS, FFNet, dan QuickSRNet. [Panduan pascapemrosesan C](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/docs/C_Postprocessing.md) secara terpisah menyebut decoder deteksi, pose, dan OBB YOLOv2/3/4/5 serta Ultralytics. SDK 3.1 saat ini mendukung PolarFire SoC Video Kit dan secara eksplisit tidak mendukung PolarFire Video Kit non-SoC, sehingga demo board lama jangan disamakan dengan kontrak target terkini.

SDK dapat diunduh publik di bawah [lisensi khusus Microchip](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/LICENSE.md), yang membatasi perangkat lunak dan turunannya untuk produk Microchip. Microchip menawarkan lisensi Libero Silver dan CoreVectorBlox gratis, tetapi harus diminta. Deployment tetap merupakan kontrak sistem FPGA: konfigurasi akselerator, bitstream, firmware, data jaringan hasil SDK, dan layout memori harus sesuai. Kompilasi VectorBlox yang berhasil tidak menjadikan model biner yang dapat disalin ke desain PolarFire sembarang.

### Syntiant

NDP200 Syntiant menargetkan inferensi vision, audio, dan sensor selalu aktif dalam batas daya yang lebih rendah daripada SoC Linux biasa. [Tabel perangkat keras](https://www.syntiant.com/hardware) saat ini menyebut NDP200 produksi massal dan NDP250 sampling. [Brief produk NDP200](https://www.syntiant.com/ndp200/) mendokumentasikan dukungan jaringan CNN, RNN, dan fully connected di bawah 1 mW; [pengumuman NDP250](https://www.syntiant.com/news/syntiant-unveils-ndp250-neural-decision-processor-with-next-gen-core-3-architecture) justru menyebut pengenalan gambar selalu aktif di bawah 30 mW. Istilah "sub-milliwatt" tidak boleh digeneralisasi ke kedua chip.

Materi publik tidak membuktikan kompatibilitas YOLO luas, sehingga produk perlu dievaluasi untuk classifier dan detektor selalu aktif kecil melalui penilaian model yang didukung vendor, bukan asumsi ekspor ONNX generik.

### Google Coral: dua generasi yang tidak terkait

Produk Edge TPU lama memakai Edge TPU Compiler, `libedgetpu`, dan PyCoral dengan model TFLite INT8 penuh. Repositori utama [Edge TPU](https://github.com/google-coral/edgetpu) dan [PyCoral](https://github.com/google-coral/pycoral) telah diarsipkan. Itu risiko pemeliharaan nyata, tetapi bukan pemberitahuan resmi EOL perangkat keras.

Google juga memelihara [Coral NPU](https://github.com/google-coral/coralnpu) open source terpisah yang masih aktif: IP akselerator RISC-V untuk integrasi ke SoC berdaya rendah. Proyek publik saat ini mengekspos RTL, simulasi perangkat keras, toolchain, dan contoh ELF. Ini bukan penerus Edge TPU USB/M.2 yang bisa langsung menggantikan, dan tidak menerbitkan matriks deployment YOLO saat ini.

## Jebakan GPU, NPU klien, dan komputasi adaptif yang berdekatan

NVIDIA, AMD, Intel, dan Apple sering muncul dalam pencarian NPU, tetapi tidak mengekspos satu kelas akselerator yang saling dapat dipertukarkan.

**NVIDIA Jetson:** Jetson Orin menggabungkan GPU CUDA dengan core DLA khusus. TensorRT dapat membangun engine terserialisasi untuk DLA, tetapi layer yang tidak didukung hanya berjalan di GPU bila fallback GPU diaktifkan secara eksplisit. [Tabel YOLO DeepStream](https://github.com/NVIDIA-AI-IOT/deepstream_tools/blob/main/yolo_deepstream/README.md) saat ini mencakup YOLOv4/v7/v8/v9/11, tetapi kebanyakan entri menargetkan GPU; entri ONNX DLA yang eksplisit adalah YOLOv8s INT8. [Batasan DLA](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-layer-restrictions.html) NVIDIA menjelaskan batas operatornya. Thor merupakan perbedaan lain: [panduan migrasi DriveOS](https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/NVIDIA_DriveOS_7.0.3_Migration_Guide.pdf) menyatakan core DLA dihapus dari Thor demi fleksibilitas penjadwalan GPU. Jadi, "berjalan di Jetson" tidak mengidentifikasi akselerator yang digunakan.

**AMD Vitis AI:** Generasi Kria/DPU lama mengompilasi graf `.xmodel` yang dijalankan melalui VART. [Vitis AI 6.2](https://vitisai.docs.amd.com/en/6.2/) saat ini GA untuk dua jalur Versal AI Edge terpisah: [Gen1](https://vitisai.docs.amd.com/projects/gen1/en/latest/index.html) pada VEK280/VE2802 memakai ONNX, AMD Quark, dan alur snapshot/subgraf yang terikat target; [Gen2](https://vitisai.docs.amd.com/projects/gen2/en/latest/index.html) pada VEK385 memiliki kontrak kompilasi dan runtime sendiri. Gen1 menerbitkan contoh YOLOv5, YOLOv7, YOLOv8, dan YOLOX. Ryzen AI adalah stack NPU klien keempat yang terpisah. Jangan memindahkan artefak atau validasi di antara DPU lama, Versal Gen1, Versal Gen2, dan Ryzen AI hanya karena semuanya memakai nama Vitis atau AMD.

**Intel OpenVINO:** OpenVINO dapat menargetkan CPU, GPU, dan NPU Core Ultra melalui satu API. [Dokumentasi plug-in NPU Intel](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md) mengidentifikasi generasi NPU yang didukung, sedangkan [matriks model terverifikasi](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html) menampilkan kolom khusus perangkat. Notebook YOLO membuktikan resep aplikasi, bukan validasi NPU kecuali matriks menyatakannya. Operator, bentuk, presisi yang didukung, dan driver NPU terpasang tetap menentukan apakah seluruh graf berjalan di sana. OpenVINO IR (`.xml` plus `.bin`) adalah IR sumber yang dapat digunakan ulang; cache model terkompilasi khusus perangkat dan perangkat lunak.

**Apple Core ML:** `coremltools` mengonversi model menjadi `.mlpackage`, yang dikompilasi Xcode menjadi `.mlmodelc`. Core ML dapat menjadwalkan pekerjaan di CPU, GPU, dan Apple Neural Engine, tetapi sengaja mengabstraksi pembagian yang tepat. Hal itu menjadikan Apple platform deployment yang sangat baik dan kurang cocok untuk klaim seperti "seluruh graf YOLO berjalan di NPU" kecuali bukti profiling menetapkannya.

## Perusahaan IP NPU di balik perusahaan chip

Sebagian perusahaan tidak menjual board atau akselerator terkemas. Mereka melisensikan desain NPU kepada vendor SoC. Mereka penting karena arsitektur dasar yang sama dapat muncul dengan beberapa merek chip, tetapi SDK dan artefak akhir masih dapat disesuaikan penerima lisensi.

| Pemasok IP | Keluarga NPU dan perangkat lunak | Alasan penting |
|---|---|---|
| [Arm](https://developer.arm.com/edge-ai/arm-cortex-m-and-ethos-u) | Ethos-U55/U65/U85 dan Vela | Muncul di produk tertanam NXP, Alif, Infineon, Himax, dan lainnya; alur berorientasi TFLite/TOSA terkuantisasi |
| [Arm China](https://www.armchina.com/mountain?infoId=161&name=) | AIPU Zhouyi dan Compass SDK | [Model zoo](https://github.com/Arm-China/Model_zoo) publik menyebut YOLOv1-tiny/v2/v3/v4/v5, YOLOX, dan YOLOv8-seg; entri itu model referensi, bukan bukti untuk setiap chip penerima lisensi |
| [VeriSilicon](https://www.verisilicon.com/en/IPPortfolio/VivanteVIP9000Pico) | Keluarga Vivante VIP dan Acuity SDK | Keluarga NPU Vivante terkait muncul di sejumlah SoC, termasuk jalur A311D dan i.MX 8M Plus yang bersumber terpisah; setiap vendor chip tetap menyediakan integrasinya sendiri |
| [Imagination Technologies](https://www.imaginationtech.com/products/open-access/) | PowerVR Series3NX AX3146/AX3386/AX3596; IMG Series4 historis; Neural Compute SDK/IMG DNN | IP NNA berlisensi, bukan board retail; Open Access menyediakan konfigurasi Series3NX 1/5/10-TOPS, sedangkan program NC-SDK resmi menyebut PP-YOLOE, EfficientNet, dan HRNet, bukan setiap konfigurasi IP |
| [CEVA](https://www.ceva-ip.com/product/ceva-neupro-studio/) | NeuPro-Nano/NeuPro-M dan NeuPro Studio | IP NPU berlisensi dengan impor, kuantisasi, kompresi, kompilasi graf, simulasi, dan pembuatan C/C++ |
| [MIPS](https://mips.com/processor-solutions/arc-npx-family/) | ARC NPX6 dan [MetaWare MX](https://mips.com/processor-solutions/arc-metaware-mx/), sebelumnya Synopsys ARC | IP NPU skalabel dan NN SDK diakuisisi GlobalFoundries lalu dimasukkan ke portofolio MIPS pada [Juni 2026](https://mips.com/press-releases/gfmipsarcclose/); bukan target deployment retail |
| [Cadence](https://www.cadence.com/en_US/home/tools/silicon-solutions/ai-ip-platform/neuroweave-sdk.html) | Neo NPU, DSP Tensilica, dan NeuroWeave SDK | Stack compiler/interpreter yang digunakan penerima lisensi SoC, dengan dukungan jaringan dan kuantisasi terikat pada konfigurasi berlisensi |
| [Quadric](https://quadric.ai/npu-ip) | IP GPNPU Chimera dan SDK | IP bergaya NPU/DSP yang dapat diprogram dan dilisensikan ke chip perusahaan lain; silikon penerima lisensi menjadi target akhirnya |
| [Expedera](https://www.expedera.com/products-overview/) | IP NPU Origin dan stack perangkat lunak | Arsitektur NPU edge yang dapat dilisensikan, bukan board yang dapat langsung dibeli LibreYOLO |

Imagination menunjukkan mengapa label akses dan siklus hidup perlu dicantumkan bersama nama arsitektur. Program [Open Access](https://www.imaginationtech.com/products/open-access/) menawarkan tiga konfigurasi Series3NX yang telah dibuktikan pada silikon tanpa biaya lisensi IP di muka, tetapi hanya setelah evaluasi perusahaan dan dengan dukungan/pemeliharaan berbayar serta royalti produksi. [Neural Compute SDK](https://developer.imaginationtech.com/solutions/neural-compute-sdk/) menyediakan tool kompilasi, optimasi, kuantisasi, dan runtime; kolaborasi [Baidu PaddlePaddle](https://www.imaginationtech.com/news/imagination-and-baidu-paddlepaddle-create-open-source-machine-learning-library-for-model-zoo/) resmi menyebut deteksi PP-YOLOE, klasifikasi EfficientNet, dan segmentasi HRNet. Contoh itu tidak memvalidasi setiap konfigurasi Series3NX atau Series4, dan halaman produk [Series4 saat ini](https://www.imaginationtech.com/product/img-4nx-mc1/) menyatakan produk tidak tersedia, sehingga Series4 perlu dianggap historis kecuali bagian penjualan mengonfirmasi sebaliknya.

Lapisan ini menjelaskan kemiripan keluarga yang terlihat. Lapisan ini tidak menciptakan portabilitas artefak. Dua SoC yang berisi IP Vivante atau Ethos terkait tetap dapat memiliki peta memori, driver, versi compiler, set operator, dan runtime vendor berbeda.

## Yang benar-benar di-deploy: tabel keterikatan artefak

Berkas terakhir di pipeline compiler adalah titik berakhirnya portabilitas ONNX yang tampak. Nama dan ekstensi berubah, tetapi aturan praktisnya stabil: simpan model asli dan data kalibrasi karena artefak native biasanya tidak dapat dipindahkan ke generasi NPU lain atau dibangun ulang secara andal dengan rilis SDK lain.

| Stack vendor | Input compiler umum | Yang sampai ke perangkat | Yang mengikatnya |
|---|---|---|---|
| [Amlogic AMLNN](https://github.com/Amlogic-NN/amlnn-toolkit) | ONNX, TFLite, TorchScript, atau PT2 | `.adla`; A311D lama memakai `.nb` | ID target, generasi ADLA, build compiler AMLNN, NNSDK2/runtime, driver ADLA, dan BSP |
| [Rockchip RKNN](https://github.com/airockchip/rknn-toolkit2) | ONNX, TFLite, atau ekspor framework | `.rknn` | Versi toolkit/runtime RKNN dan keluarga NPU Rockchip |
| [Hailo](https://hailo.ai/developer-zone/documentation/) | ONNX atau TensorFlow melalui HAR perantara | `.hef` | Arsitektur Hailo dan generasi compiler/runtime |
| [Axelera Voyager](https://docs.axelera.ai/sdk/reference/pipeline-builder/model-compilation/) | ONNX atau definisi zoo yang didukung | `.axm` pada Pipeline Builder alfa, paket pipeline `.axe`; `.axmodel` klasik plus manifest | Generasi API Voyager dan konfigurasi target Metis |
| [DEEPX DXNN](https://github.com/DEEPX-AI/dx-all-suite) | ONNX dan jalur framework yang didukung | `.dxnn` | Versi DX-COM/DX-RT dan target DX-M |
| [Qualcomm QAIRT/QNN atau SNPE](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | ONNX, TFLite, atau model framework | Library model/biner konteks QNN atau SNPE `.dlc` | Arsitektur HTP, SoC, backend, dan versi runtime |
| [MediaTek NeuroPilot](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/related_resource.html) | TFLite terkonversi/terkuantisasi | `.dla` untuk Neuron Runtime offline; `.tflite` untuk mode delegasi | Generasi NP/MDLA, image OS, compiler, dan runtime |
| [TI TIDL](https://github.com/TexasInstruments/edgeai-tidl-tools) | ONNX atau TFLite | Direktori imported-artifacts plus berkas model aplikasi | Tool/runtime TIDL dan generasi prosesor |
| [NXP eIQ](https://www.nxp.com/design/design-center/software/development-software/eiq-ml-development-environment:EIQ) | Biasanya TFLite atau ONNX | Keluaran Vela, TIM-VX, Neutron, atau Ara khusus backend | Akselerator i.MX/Ara yang dipilih dan BSP, bukan "NXP" secara umum |
| [ST Edge AI Core](https://www.st.com/en/development-tools/stedgeai-core.html) | TFLite, ONNX, atau model framework yang didukung | Library/kode jaringan dan aset firmware yang dihasilkan | Target MCU/NPU, konfigurasi memori, dan versi tool |
| [Infineon PSOC Edge](https://documentation.infineon.com/psocedge/docs/bwb1750411526047) | Jalur TFLite/Keras/PyTorch terkuantisasi integer | TFLite teroptimasi Vela dengan aliran perintah Ethos-U plus firmware aplikasi | Varian E83/E84, konfigurasi Ethos-U, Vela, ModusToolbox, dan BSP; NNLite target terpisah |
| [Renesas DRP-AI](https://github.com/renesas-rz/rzv_drp-ai_tvm) | ONNX/TFLite melalui TVM atau Translator | Direktori model runtime berisi beberapa berkas data biner | Perangkat RZ/V dan generasi translator/runtime |
| [Sony IMX500](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) | Jaringan terkuantisasi/terkemas dari alur Edge-MDT | Paket `.rpk` | Firmware sensor, anggaran memori, dan versi packer |
| [Synaptics](https://developer.synaptics.com/) | TFLite/ONNX dan impor yang didukung | `.synap` pada SyNAP; `.vmfb` pada Torq | Generasi perangkat lunak/perangkat keras SL16xx dibanding SL261x |
| [Ambarella CVflow](https://www.ambarella.com/developer/) | Input toolchain mitra | Paket deployment khusus mitra; kontrak artefak tepatnya tidak terdokumentasi publik | Konfirmasikan generasi CVflow, SDK/BSP, dan pipeline kamera melalui Cooper Developer Zone |
| [D-Robotics BPU](https://github.com/D-Robotics/rdk_model_zoo) | ONNX dan graf yang didukung toolchain | X5 `.bin`; `rdk_s` saat ini `.hbm`; X3 memakai alur platform lama | Generasi BPU, branch repositori, dan rilis OpenExplorer/runtime yang cocok |
| [AXERA Pulsar2](https://github.com/AXERA-TECH/ax-samples) | ONNX | `.axmodel` | Target chip AX, versi Pulsar2, dan AXEngine |
| [SOPHGO TPU-MLIR](https://github.com/sophgo/tpu-mlir) | ONNX, TFLite, TorchScript, dan lainnya | `.bmodel` pada BM, `.cvimodel` pada CV18xx | Target chip BM/CV dan generasi runtime SOPHON |
| [Huawei Ascend CANN](https://www.hiascend.com/en/software/cann) | ONNX atau graf framework yang didukung | `.om` | Chip Ascend, CANN/ATC, serta firmware/driver perangkat |
| [Cambricon MagicMind](https://github.com/Cambricon/magicmind_cloud) | Graf framework atau model pertukaran | Engine/model MagicMind terserialisasi | Arsitektur MLU dan versi Neuware/MagicMind |
| [Canaan nncase](https://github.com/kendryte/nncase) | ONNX atau TFLite | `.kmodel` | Generasi KPU dan plug-in target nncase yang cocok |
| [Mobilint qb](https://www.mobilint.com/sdk-qb) | ONNX dan model framework yang didukung | `.mxq` | Target REGULUS/ARIES dan compiler/runtime qb |
| [Rebellions RBLN](https://docs.rbln.ai/latest/index.html) | PyTorch 2 atau graf TensorFlow yang didukung | `.rbln` | Generasi ATOM dan compiler/runtime RBLN |
| [FuriosaAI](https://developer.furiosa.ai/docs/latest/en/) | ONNX/TFLite dan jalur framework yang didukung | Warboy `.enf`; RNGD `.fxb` | Generasi akselerator dan SDK yang sepenuhnya berbeda |
| [Sunplus SNNF](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2004353133) | Jaringan yang didukung Acuity | `.nb` | Driver NPU SP7350, runtime, dan BSP |
| [ESWIN ENNP](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/esaac) | ONNX dalam alur EsAAC saat ini yang terdokumentasi; framework lain perlu ekspor/konversi | `.model` | Target keluarga EIC7700 yang tepat dan rilis ENNP/ESSDK |
| [Nuvoton / Ethos-U](https://github.com/OpenNuvoton/NuEdgeWise) | TFLite terkuantisasi | TFLite teroptimasi Vela dengan operasi khusus Ethos-U | Rencana memori M55M1, Vela, dan firmware |
| [Himax WiseEye2](https://github.com/HimaxWiseEyePlus/YOLOv8_on_WE2) | TFLite terkuantisasi integer penuh | `.tflite` teroptimasi Vela di flash model plus firmware board seperti `output.img` | Konfigurasi HX6538/WE2, Vela, TFLite Micro, driver Ethos-U, dan kernel fallback |
| [Analog Devices AI8X](https://github.com/analogdevicesinc/ai8x-synthesis) | Checkpoint PyTorch, YAML jaringan, dan input sampel | Sumber C/header khusus perangkat, bobot, dan firmware terkompilasi | Batas memori/layer MAX78000 dibanding MAX78002, tool synthesis, dan rilis MSDK |
| [Realtek AmebaPro2](https://github.com/Ameba-AIoT/ameba-arduino-pro2) | Layanan/input konversi vendor | `.nb` | Firmware RTL8735B dan rilis API VoE/NeuralNetwork |
| [Telechips Enlight](https://docs.topst.ai/product/p/ai) | Jalur Darknet, TensorFlow, ONNX, atau PyTorch | `.enlight` perantara plus bundel deployment terkompilasi | NPU TCC7500 dan versi TC-NN/Enlight |
| [T-Head HHB](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) | ONNX dan graf framework yang didukung | `hhb.bm`, parameter/kode hasil, dan executable | Stack CSI-NN2/SHL TH1520 |
| [SigmaStar MI_IPU](https://wx.comake.online/doc/doc/SigmaStarDocs-SSU9383CM-SIGMASTAR-202507071022/platform/MI/ipu_zh.html) | Dokumentasi SSU9383CM saat ini mengekspos runtime MI_IPU; materi lama memakai input keluarga TensorFlow/Caffe | `.sim` lama dikonversi menjadi `sgsimg.img`; artefak publik saat ini belum terverifikasi | Generasi IPU dan SDK SigmaStar yang tepat; kompatibilitas compiler lama dengan SSU9383CM belum terverifikasi |
| [HiSilicon smart vision](https://gitee.com/HiSpark/modelzoo/blob/master/README.md) | ONNX dan graf yang didukung ATC | `.om` | SoC smart vision dan stack NNN/SVP-NNN yang tepat; tidak portabel secara generik ke Ascend |
| [MemryX](https://developer.memryx.com/tools/neural_compiler.html) | ONNX, TFLite, Keras, atau TensorFlow | Paket dataflow `.dfp` | Konfigurasi perangkat keras MX dan rilis runtime/compiler |
| [Kneron](https://doc.kneron.com/docs/) | ONNX plus konfigurasi toolchain | `.nef`; KL730 NEFv2 dapat memuat `.kne` | Target compiler terdokumentasi, generasi chip, firmware, dan runtime PLUS; kompilasi KL830 belum ditetapkan Toolchain 0.33.1 |
| [SiMa.ai Palette](https://docs.sima.ai/v2.1.2/pages/model-sdk/compilation.html) | Model framework atau ONNX yang didukung | `.tar.gz` terkompilasi ModelSDK, ELF executable terdokumentasi, atau paket `.mpk` MPK Tool | Target MLSoC/Modalix dan rilis Palette |
| [NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/) | ONNX atau definisi jaringan | Engine terserialisasi, umumnya `.engine` atau `.plan` | Arsitektur GPU/DLA, TensorRT, CUDA, dan sering kali perangkat |
| [AMD Vitis AI](https://vitisai.docs.amd.com/en/6.2/) | ONNX/framework terkuantisasi | `.xmodel` lama; snapshot/subgraf NPU Gen1; direktori cache terkompilasi Gen2 atau paket produksi `.rai` | Generasi NPU/konfigurasi IP, image platform, dan matriks Vitis/Vivado/PetaLinux yang tepat |
| [Microchip VectorBlox](https://github.com/Microchip-Vectorblox/VectorBlox-SDK) | TFLite INT8 terkuantisasi penuh | `.vnnx`, `.hex`, dan `.ucomp` plus firmware/desain FPGA yang sesuai | Konfigurasi CoreVectorBlox, PolarFire SoC Video Kit, SDK 3.1, bitstream, dan layout memori |
| [Intel OpenVINO](https://docs.openvino.ai/) | Model framework atau ONNX | IR `.xml` plus `.bin`, atau cache terkompilasi khusus perangkat | IR dapat digunakan ulang; cache terkompilasi khusus perangkat, driver, dan OpenVINO |
| [Google Edge TPU lama](https://coral.ai/docs/edgetpu/models-intro/) | TFLite INT8 penuh | `_edgetpu.tflite` hasil kompilasi Edge TPU | Compiler/runtime Edge TPU dan operator terkuantisasi yang didukung; tidak terkait IP Coral NPU baru |
| [Google Coral NPU IP](https://github.com/google-coral/coralnpu) | Contoh C/C++ untuk proyek IP publik saat ini | Contoh ELF untuk simulasi RTL/perangkat keras dan integrasi SoC mendatang; belum ada artefak compiler YOLO publik | Implementasi SoC yang dilisensikan/diintegrasikan secara spesifik dan toolchain open source yang masih berkembang |
| [Lattice sensAI](https://www.latticesemi.com/en/Solutions/Solutions/SolutionsDetails02/sensAI?ActiveTab=Design+File) | Deskripsi jaringan yang didukung | Bitstream FPGA, konfigurasi akselerator, dan bobot | Keluarga FPGA, mode IP sensAI, dan implementasi memori |
| [Imagination NC-SDK](https://developer.imaginationtech.com/solutions/neural-compute-sdk/) | Caffe, TensorFlow, atau ONNX | Nama berkas artefak terkompilasi untuk pelanggan tidak ditentukan publik | Konfigurasi NNA berlisensi, integrasi SoC, NC-SDK/DDK, dan runtime penerima lisensi |

Tabel ini bukan sekadar daftar ekstensi. Versi compiler, identifier target, driver, firmware, dan BSP merupakan bagian dari identitas model yang dapat direproduksi. Eksporter LibreYOLO mendatang sebaiknya menulisnya ke manifest yang dapat dibaca mesin di samping setiap artefak.

## Akses tool dan sinyal siklus hidup

Ketersediaan perangkat keras dan akses compiler adalah dua hal terpisah. Board mungkin mudah dibeli, sementara compiler terkini memerlukan perjanjian pelanggan; SDK publik bisa menargetkan silikon yang sulit diperoleh. Berikut sinyal konkret dari pihak pertama, bukan peringkat umur universal.

| Platform | Sinyal terdokumentasi | Makna praktis |
|---|---|---|
| Amlogic ADLA | Repositori Apache-2.0 publik dan wheel yang dapat diunduh; driver board/BSP kompatibel mungkin tetap perlu dari Amlogic atau vendor board | Evaluasi compiler mudah, tetapi redistribusi biner dan matriks kompatibilitas penuh perlu dikonfirmasi |
| MediaTek Genio | IoT AI Hub dan panduan Yocto publik; tool NP8 all-in-one dan dokumen Android ditandai hanya untuk pelanggan langsung/NDA | Bukti model dapat diaudit, sementara otomasi model sendiri mungkin memerlukan kemitraan |
| Hailo | Model zoo dan runtime publik; Dataflow Compiler disalurkan lewat Developer Zone; `hailo-camera-apps` publik diarsipkan 3 Mei 2026 | Penelusuran model luas terbuka, tetapi kompilasi yang dapat direproduksi memerlukan akun/versi tepat; status arsip tidak membuktikan EOL Hailo-15 dan paket aplikasi vision terkini perlu dikonfirmasi |
| Axelera AI | Dokumentasi publik dan [Voyager SDK publik](https://github.com/axelera-ai-hub/voyager-sdk); dukungan pelanggan memerlukan akun | SDK akselerator mandiri yang langka, sedangkan dukungan produk dan pengadaan tetap komersial |
| Qualcomm Dragonwing | [Product Longevity Program](https://www.qualcomm.com/internet-of-things/products/product-longevity-program) mencantumkan IQ-9075 Sampling dengan dukungan hingga 2038 dan QCS6490 hingga Juli 2036, sedangkan halaman IQ-9075 menyebut Active | Sinyal perencanaan industri kuat, tetapi konflik status pihak pertama memerlukan konfirmasi tingkat SKU dan tidak menjamin satu SDK AI tetap ABI-stabil sepanjang periode |
| Rebellions | [Matriks dukungan](https://docs.rbln.ai/latest/supports/version_matrix.html) saat ini menandai CA02 dan CA12 EoL, serta CA22/CA25 Active; CA21 tidak tercantum | Pengadaan dan kompatibilitas SDK berbeda menurut kartu, bahkan dalam keluarga ATOM |
| NVIDIA Jetson | [Tabel siklus hidup modul resmi](https://developer.nvidia.com/embedded/lifecycle) mencantumkan modul Orin hingga Januari 2032 dan AGX Orin Industrial hingga Juli 2033; kit developer tidak memiliki komitmen siklus hidup | Rancang sistem produksi dengan mempertimbangkan modul, bukan ketersediaan kit pengembangan |
| Sony IMX500 pada Raspberry Pi | [Brief produk AI Camera](https://datasheets.raspberrypi.com/camera/ai-camera-product-brief.pdf) menyatakan produksi setidaknya sampai Januari 2028 | Jaminan minimum yang konkret untuk modul kamera itu, bukan untuk semua produk IMX500 |
| Amlogic A311Y3 | [Halaman peluncuran 2026](https://www.amlogic.com/News/index248.html) mengiklankan jaminan umur produk minimum sepuluh tahun | Sinyal platform baru yang menjanjikan, tetapi program produk tetap perlu detail kontrak dan SKU |
| Google Coral | Repositori Edge TPU dan PyCoral lama diarsipkan/read-only; repositori [Coral NPU IP](https://github.com/google-coral/coralnpu) terpisah masih aktif | Risiko pemeliharaan perangkat lunak lama, bukan dengan sendirinya pemberitahuan resmi EOL perangkat keras; tidak berkaitan dengan siklus hidup proyek IP open source yang berbeda |

## Mengapa TOPS bukan panduan pembelian

TOPS puncak berguna dalam satu vendor dan arsitektur, tetapi perbandingan lintas vendor biasanya tidak valid kecuali semua hal berikut sama:

- Presisi numerik, termasuk INT8, INT4, FP16, atau mode campuran
- Apakah satu operasi multiply-accumulate dihitung sebagai satu atau dua operasi
- Aritmetika dense dibanding sparse terstruktur
- Resolusi input, ukuran batch, dan graf model
- Akurasi setelah kuantisasi
- Latensi NPU saja dibanding pipeline aplikasi lengkap
- Transfer memori dan fallback host
- Kondisi termal serta operasi berkelanjutan, bukan burst

Detektor adalah pipeline: akuisisi gambar, resize/letterbox, normalisasi, transfer perangkat, inferensi neural, decode, NMS, penskalaan koordinat, dan logika aplikasi. Vendor sering hanya menerbitkan bagian tengah, yaitu inferensi neural. [MLPerf Inference Edge](https://mlcommons.org/benchmarks/inference-edge/) bernilai karena menetapkan skenario, target kualitas, dan aturan pengukuran tingkat sistem alih-alih membandingkan aritmetika pemasaran secara terpisah.

Untuk deployment YOLO, catatan benchmark minimum yang dapat dipercaya adalah:

```text
exact checkpoint + checksum
source and compiled model format
compiler, runtime, driver and firmware versions
target board and clock/power mode
input resolution and batch
precision and calibration dataset
accuracy before and after compilation
warmup and sample count
preprocess, inference and postprocess latency
power measurement boundary
```

Tanpa kolom tersebut, dua angka FPS biasanya mengukur hal berbeda.

## Cara memilih NPU untuk computer vision

Pilih dengan urutan ini, bukan berdasarkan angka terbesar di halaman produk.

1. **Buktikan kompatibilitas graf.** Kompilasi model hasil ekspor yang tepat, bukan checkpoint model zoo bernama mirip.
2. **Ukur akurasi.** Validasi artefak terkompilasi pada dataset task sebenarnya setelah kalibrasi dan kuantisasi.
3. **Ukur end-to-end.** Sertakan konversi input, transfer, decode, dan NMS.
4. **Periksa akses perangkat lunak.** Cari tahu apakah compiler publik, hanya tersedia setelah pendaftaran, atau hanya tersedia dengan perjanjian komersial.
5. **Bekukan matriks versi.** Catat compiler, runtime, driver, firmware, dan identifier target.
6. **Periksa kondisi sistem operasi.** Dukungan Android, Debian, Yocto, Buildroot, RTOS, dan Windows tidak saling menggantikan.
7. **Periksa pasokan dan siklus hidup.** Chip yang unggul secara teknis buruk untuk produksi bila modul, driver, atau dukungan jangka panjangnya tidak pasti.
8. **Setelah itu bandingkan biaya, daya, dan performa.** Pengukuran itu baru bermakna setelah model yang sama bekerja benar di kedua target.

### Titik awal praktis menurut penggunaan

| Penggunaan | Platform yang patut dievaluasi lebih dulu | Alasan |
|---|---|---|
| Akselerator khusus Raspberry Pi | Hailo-8L/8 dan Sony IMX500 | Produk, image perangkat lunak, dan alur developer konkret resmi Raspberry Pi |
| Add-on Linux umum PCIe, M.2, atau USB | DEEPX, MemryX, Hailo, dan produk Axelera | Form factor akselerator yang tersedia, bergantung pada kompatibilitas host/driver |
| SBC Linux murah atau kamera | Rockchip RK3588/RK3576, Amlogic ADLA bila board/BSP yang didukung tersedia, AXERA, Canaan K230, atau Sunplus SP7350 | Pipeline media terintegrasi dan bukti model/compiler publik; pastikan VIM4 V13A+ untuk NPU A311D2 |
| Android mobile dan volume tinggi | Qualcomm QNN/AI Hub, LiteRT, Apple Core ML | Basis terpasang besar dan runtime mobile terpelihara |
| Linux industri dan robotika | TI TDA4/AM6xA, NXP i.MX, Renesas RZ/V, Hailo | Produk tertanam berumur panjang dan ekosistem kamera/IO |
| Endpoint atau MCU sangat kecil | STM32N6x7, Infineon PSOC Edge, Himax WiseEye2, Analog Devices MAX7800x, Nuvoton M55M1, Alif/Arm Ethos-U, GAP9, dan Syntiant | Batas daya/memori ketat dengan tool khusus dan batas ukuran model |
| Vision FPGA yang dapat dikonfigurasi | Microchip VectorBlox, Lattice sensAI, dan target AMD Vitis AI | Pipeline perangkat keras fleksibel, tetapi konfigurasi akselerator, bitstream, dan compiler model menjadi satu sistem berversi |
| Vision PCIe throughput tinggi | Axelera Metis, Hailo, DEEPX, Mobilint, MemryX, Rebellions, atau SiMa.ai | Produk akselerator khusus dan orientasi multi-stream |
| Ekosistem produk berpusat di Tiongkok | Rockchip, AXERA, D-Robotics, SOPHGO, Huawei Ascend, HiSilicon, dan Cambricon | Board regional, toolchain, dan contoh model yang kuat |

Tabel ini adalah daftar riset awal, bukan peringkat performa. Pengadaan, ketersediaan regional, dan model tepat dapat membalik urutannya.

## Maknanya bagi LibreYOLO

Desain yang skalabel bukan puluhan eksporter yang tidak saling berkaitan. Gunakan satu kontrak pertukaran yang ketat beserta adapter compiler/runtime vendor kecil:

```text
LibreYOLO model
    -> deterministic static ONNX
    -> VendorCompiler.compile(model, target, calibration, precision)
    -> native artifact + manifest
    -> VendorRuntime.load() / infer()
    -> shared task-specific decode and Results objects
```

Setiap artefak native sebaiknya memiliki manifest pendamping yang mencakup:

- Vendor, target chip, dan target board
- Versi compiler, runtime, driver, dan firmware
- Checksum checkpoint sumber dan ONNX
- Nama input, bentuk, layout, colorspace, normalisasi, dan tipe data
- Presisi kuantisasi, skala bila tersedia, dan hash data kalibrasi
- Nama tensor keluaran, bentuk, dan makna semantik
- Task deteksi, nama kelas, serta apakah decode/NMS berlangsung di chip atau host
- Paritas numerik dan hasil akurasi task yang tercatat

LibreYOLO telah menyediakan [ekspor ONNX portabel](/docs/export/onnx), [tool kuantisasi](/docs/export/quantization), [ekspor RKNN](/docs/export/rknn) langsung tetapi sengaja terbatas, dan [alur Hailo](/docs/export/hailo) dengan compiler eksternal yang dijelaskan secara jujur. Berdasarkan kriteria dalam panduan ini, Amlogic ADLA merupakan kandidat integrasi berikutnya yang kuat: toolkit-nya publik, cakupan model sangat beririsan dengan LibreYOLO, dan jalur A311D lama dapat dipisahkan secara eksplisit.

Prioritas setelah Amlogic sebaiknya ditentukan oleh perangkat keras pengguna dan kemampuan menjalankan validasi akurasi, bukan hanya ketersediaan compiler. DEEPX, Sony IMX500, D-Robotics, AXERA, SOPHGO, dan Qualcomm menarik secara teknis; TI, NXP, ST, dan Renesas menjadi sangat bernilai ketika mitra industri dapat menyediakan board dan akses CI jangka panjang.

## Watchlist: silikon nyata tanpa kontrak integrasi publik

Menghapus seluruh perusahaan dari peta pasar dapat menyesatkan. Memasukkannya sebagai "didukung" bisa lebih buruk. Vendor berikut menjual, melakukan sampling, atau telah mengumumkan silikon terkait, tetapi bukti publik belum menetapkan compiler, artefak, dan jalur model bernama terkini yang dapat direproduksi dan layak untuk backend LibreYOLO.

| Perusahaan | Yang nyata | Alasan tetap menjadi entri watchlist |
|---|---|---|
| [Samsung](https://semiconductor.samsung.com/processor/automotive-processor/exynos-auto-v920/) | Exynos Auto V920 memiliki NPU dual-core yang diiklankan hingga 23.1 TOPS | Samsung menyatakan [Neural SDK tidak lagi disediakan untuk developer pihak ketiga](https://developer.samsung.com/neural/overview.html); Samsung ONE/Circle tidak membuktikan akses ke NPU Exynos |
| [Novatek](https://www.novatek.com.tw/en-global/Milestone/aboutus_milestones) | Tonggak perusahaan saat ini menyebut SoC Edge AI dan Edge Vision/Imaging AI; tonggak 2020 pernah menyebut MobileNet, SSD, dan YOLOv3 | Tidak ada matriks nomor komponen/compiler/artefak/operator/model publik terkini; YOLOv3 historis bukan bukti SDK saat ini |
| [Nextchip](https://www.nextchip.com/en/adas/adas.php?idx=5) | Prosesor otomotif APACHE5/NVS2900 dan [APACHE6/NVS3000](https://www.nextchip.com/en/adas/adas.php?idx=7) saat ini dengan NPU aiWare aiMotive; halaman APACHE6 sendiri mencantumkan 12 dan 8 TOPS | aiWare Studio dan runtime tersedia, tetapi tidak ditemukan artefak publik, panduan kuantisasi, daftar operator, atau matriks YOLO; angka vendor yang bertentangan tidak boleh diselesaikan diam-diam |
| [Black Sesame Technologies](https://bst.ai/en.html) | Silikon AI otomotif/edge Huashan A1000/A2000 dan seri Wudang C | Informasi produk publik tersedia, tetapi bukan compiler mandiri, kontrak artefak, atau model zoo yang dapat direproduksi |
| [Ingenic](https://en.ingenic.com.cn/products-detail/id-19.html) | SoC kamera T41/T40 dengan klaim NPU bit rendah; Magik AI menjelaskan PTQ/QAT dan kompilasi graf | Tidak ditemukan compiler terkini yang dapat diunduh, kontrak artefak, atau daftar YOLO vendor yang tepat |
| [Fullhan](https://fullhan.com/en/index.php?a=type&c=article&tid=9) | Sejumlah SoC IPC mengiklankan NPU 0.5 hingga 2 TOPS, sedangkan [keluarga NVR MC6880](https://fullhan.com/en/index.php?a=type&c=article&tid=48) mengiklankan 4 TOPS | Tidak ada dokumentasi compiler/runtime/framework/model NN publik yang memadai untuk reproduksi independen |
| [Goke Microelectronics](http://www.gokemicro.com/News/info.aspx?itemid=437) | Halaman pihak pertama untuk komponen kamera pintar GK7606V1/GK7206V1/GK7203V1 mengiklankan performa NPU terintegrasi, tetapi saat publikasi hanya tersedia melalui HTTP lama | Tidak ada converter publik, kontrak runtime, atau matriks model bernama; kanalnya berorientasi OEM |
| [Chengheng Micro](https://en.chenghengmicro.com/) | CH37 mengiklankan 64 INT8 TOPS dan mode FP16/FP32/FP64; [linimasa 2026](https://chenghengmicro.com/about.html) perusahaan menyatakan produksi batch kecil dimulai | Tidak ditemukan SDK publik, kontrak compiler, atau validasi model bernama; anggap sebagai platform komersial awal tanpa akses mandiri, bukan bukti deployment yang dapat direproduksi |
| [Bouffalo Lab](https://github.com/bouffalolab/bouffalo_sdk) | BL808 mencakup NPU BLAI-100 dan repositori SDK resmi saat ini tersedia | SDK terpelihara tidak mengekspos jalur konversi/contoh BLAI resmi terkini; alur yang tersisa merupakan bukti lama atau komunitas |

Watchlist ini sengaja berbasis bukti. Vendor dapat berpindah ke tabel deployment dengan menerbitkan toolchain atau jalur evaluasi terkini, kontrak artefak khusus target, serta setidaknya satu model bernama dengan resep end-to-end.

## Klaim yang sengaja tidak dibuat

Artikel ini tidak menyatakan setiap model bernama berfungsi dari setiap repositori upstream vendor. Model zoo vendor sering dimodifikasi, dipotong pada node keluaran berbeda, atau dipasangkan dengan pascapemrosesan khusus.

Artikel ini juga tidak mengubah pernyataan berikut menjadi klaim dukungan:

- Compiler mengimpor ONNX.
- Chip mengiklankan driver Android NNAPI.
- Distributor menyebut board "kompatibel dengan YOLO."
- Repositori komunitas menjalankan satu fork dari satu model.
- Model berhasil dikompilasi tanpa kesalahan.
- Vendor melaporkan FPS NPU saja tanpa hasil akurasi.

Akselerator ponsel Google Tensor dan banyak ASIC khusus OEM juga nyata, tetapi Google tidak mengekspos Tensor sebagai target compiler mandiri umum setara platform di atas. Keberadaan perusahaan, diagram blok NPU, atau akselerasi Android tidak cukup untuk mengarang klaim integrasi LibreYOLO.

Demikian pula, akselerator cloud seperti Google TPU, AWS Inferentia/Trainium, Microsoft Maia, dan kartu AI khusus pusat data berada di luar cakupan. Panduan ini membahas perangkat keras yang secara masuk akal dapat di-deploy developer computer vision di edge.

Lisensi model merupakan lapisan terpisah. Vendor chip yang menerbitkan resep konversi YOLO tidak memberikan hak memakai atau mendistribusikan bobot upstream, kode pelatihan, atau turunannya yang terkompilasi. Dukungan perangkat keras, lisensi SDK, dan lisensi model harus diperiksa masing-masing untuk produk yang direncanakan.

## Metodologi riset dan koreksi

Untuk setiap perusahaan, riset mencari empat sumber utama:

1. Halaman produk atau datasheet terkini yang mengidentifikasi silikon.
2. Dokumentasi compiler dan runtime yang menjelaskan jalur deployment sebenarnya.
3. Model zoo, tabel kompatibilitas, atau tutorial end-to-end yang menyebut model vision.
4. Sinyal siklus hidup atau akses yang menunjukkan apakah developer benar-benar dapat memperoleh tool.

Organisasi GitHub pihak pertama dihitung sebagai sumber vendor. Dokumentasi vendor board diberi label sebagai bukti ekosistem bila perusahaan chip tidak menerbitkan materi yang sama. Klaim performa pihak ketiga dikecualikan dari tabel perbandingan.

Draf lengkap kemudian diaudit ulang melalui tahap terpisah untuk kelompok vendor dan tabel lintas bagian. Pemeriksaan itu memvalidasi ulang cakupan target, nama artefak, presisi, bukti model dibanding pascapemrosesan, status diumumkan dibanding sudah dikirim, label siklus hidup, penyebut benchmark, dan setiap jumlah entitas. Bila dua halaman pihak pertama bertentangan, panduan ini menampilkan pertentangan tersebut alih-alih diam-diam memilih nilai yang lebih nyaman.

Pasar ini berubah cepat. Jika Anda bekerja di salah satu perusahaan tersebut dan chip, SDK, daftar model, atau status aksesnya keliru, kirimkan URL dan versi dokumentasi publik yang tepat kepada LibreYOLO. Koreksi yang didukung bukti primer sebaiknya menggantikan teks ini; klaim pemasaran tanpa dokumentasi tidak.

## FAQ

### Ada berapa perusahaan NPU edge AI?

Tidak ada jumlah perusahaan yang berlaku universal karena vendor produk, pemberi lisensi IP, sensor pintar, GPU, dan ASIC otomotif privat saling tumpang tindih. Panduan non-ekshaustif ini mencatat 69 perusahaan dan ekosistem platform yang memiliki chip edge AI, platform akselerator, IP NPU berlisensi, atau silikon watchlist yang layak diperhatikan per Agustus 2026.

### Apa itu NPU?

Neural processing unit adalah perangkat keras khusus untuk operasi jaringan saraf seperti konvolusi dan perkalian matriks. Istilah vendor mencakup NPU, AIPU, BPU, KPU, DLA, HTP, TPU, dan MLA, dan model hasil kompilasinya umumnya tidak portabel antarperusahaan.

### Perusahaan NPU mana yang secara resmi mendukung model YOLO?

Amlogic, Hailo, Axelera AI, Rockchip, Qualcomm, MediaTek, DEEPX, D-Robotics, AXERA, SOPHGO, Huawei, Cambricon, Sony IMX500 melalui repositori AI Camera resmi Raspberry Pi, STMicroelectronics, Renesas, Lattice, Himax, Microchip, dan beberapa lainnya memiliki entri model zoo, tutorial, atau hasil validasi pihak pertama untuk setidaknya satu generasi YOLO. Generasi, task, target chip, dan versi SDK yang tepat tetap penting.

### Bisakah model ONNX apa pun dijalankan di NPU apa pun?

Tidak. ONNX adalah format pertukaran, bukan jaminan kompatibilitas perangkat keras. Compiler NPU harus mendukung setiap operator, bentuk tensor, dan tipe data di dalam graf. Bentuk statis, pemotongan graf, operasi khusus, dan fallback CPU merupakan hal umum.

### Apa NPU terbaik untuk YOLO?

Tidak ada pemenang universal. Hailo, Rockchip, Axelera AI, DEEPX, dan Amlogic memiliki cakupan YOLO terdokumentasi yang kuat, sedangkan jangkauan perangkat Qualcomm sangat luas. Pilih berdasarkan akurasi setelah kuantisasi, latensi seluruh pipeline, daya berkelanjutan, harga, ketersediaan, akses SDK, dan dukungan sistem operasi.

### Mengapa angka TOPS NPU tidak bisa dibandingkan langsung?

Vendor dapat menghitung presisi, operasi sparse, operasi multiply-accumulate, dan asumsi utilisasi puncak yang berbeda. TOPS tidak mencakup lalu lintas memori, operator yang tidak didukung, prapemrosesan, pascapemrosesan, dan overhead host. Bandingkan model, presisi, resolusi, akurasi, dan seluruh pipeline yang sama.

### Apa itu kalibrasi NPU?

Kalibrasi menjalankan gambar deployment yang representatif, biasanya tanpa label, melalui model agar compiler dapat memperkirakan rentang aktivasi untuk INT8 atau format presisi rendah lainnya. Gambar acak atau tidak representatif tetap dapat menghasilkan artefak terkompilasi, tetapi merusak akurasi task.

### Apakah LibreYOLO mendukung NPU edge?

LibreYOLO mengekspor ONNX dan beberapa format runtime, memiliki jalur compiler RKNN langsung untuk model Rockchip tertentu, serta mendokumentasikan alur kompilasi Hailo eksternal. Setiap artefak native vendor lainnya tetap memerlukan SDK vendornya dan sebaiknya disebut kandidat integrasi sampai dikompilasi, divalidasi, dan dijalankan pada perangkat keras.
