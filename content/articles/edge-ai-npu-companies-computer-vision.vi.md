---
title: "69 công ty Edge AI NPU: chip, SDK và YOLO (2026)"
description: "Hướng dẫn có nguồn dẫn về 69 công ty Edge AI và hệ sinh thái NPU: chip, SDK, artifact, lượng tử hóa (quantization), cùng khả năng hỗ trợ thị giác máy tính (computer vision) và YOLO được tài liệu hóa."
date: 2026-08-15
author: Xuban
tags: [edge-ai, npu, computer-vision, yolo, hardware, ai-accelerators, amlogic, hailo, rockchip]
faq:
  - q: "Có bao nhiêu công ty Edge AI NPU?"
    a: "Không có con số tổng thống nhất vì các nhà cung cấp sản phẩm, bên cấp phép IP, cảm biến thông minh, GPU và ASIC ô tô độc quyền có phạm vi chồng lấn. Hướng dẫn không toàn diện này theo dõi 69 công ty và hệ sinh thái nền tảng có chip Edge AI, nền tảng tăng tốc, IP NPU có thể cấp phép hoặc silicon đáng theo dõi tính đến tháng 8 năm 2026."
  - q: "NPU là gì?"
    a: "Bộ xử lý mạng nơ-ron (neural processing unit) là phần cứng chuyên dụng cho các phép toán mạng nơ-ron như tích chập và nhân ma trận. Các nhà cung cấp dùng những tên gọi như NPU, AIPU, BPU, KPU, DLA, HTP, TPU và MLA; mô hình đã biên dịch của họ thường không thể chuyển giữa các công ty."
  - q: "Những công ty NPU nào chính thức hỗ trợ mô hình YOLO?"
    a: "Amlogic, Hailo, Axelera AI, Rockchip, Qualcomm, MediaTek, DEEPX, D-Robotics, AXERA, SOPHGO, Huawei, Cambricon, Sony IMX500 thông qua repo AI Camera chính thức của Raspberry Pi, STMicroelectronics, Renesas, Lattice, Himax, Microchip và một số công ty khác có mục trong model zoo bên thứ nhất, hướng dẫn hoặc kết quả đã kiểm chứng cho ít nhất một thế hệ YOLO. Thế hệ, tác vụ, mục tiêu chip và phiên bản SDK cụ thể vẫn rất quan trọng."
  - q: "Mọi mô hình ONNX có thể chạy trên mọi NPU không?"
    a: "Không. ONNX là định dạng trao đổi, không phải cam kết tương thích phần cứng. Compiler NPU phải hỗ trợ mọi operator, hình dạng tensor và kiểu dữ liệu trong graph. Hình dạng tĩnh, cắt graph, toán tử tùy chỉnh và chuyển tiếp sang CPU là những tình huống thường gặp."
  - q: "NPU nào tốt nhất cho YOLO?"
    a: "Không có lựa chọn tốt nhất cho mọi trường hợp. Hailo, Rockchip, Axelera AI, DEEPX và Amlogic có mức hỗ trợ YOLO được tài liệu hóa tốt, còn Qualcomm có phạm vi thiết bị đặc biệt rộng. Hãy chọn dựa trên độ chính xác sau quantization, độ trễ toàn pipeline, công suất duy trì, giá, khả năng cung ứng, quyền truy cập SDK và hỗ trợ hệ điều hành."
  - q: "Vì sao không thể so sánh trực tiếp các con số TOPS của NPU?"
    a: "Các nhà cung cấp có thể tính precision, phép toán thưa, phép nhân-tích lũy và giả định về mức sử dụng đỉnh theo những cách khác nhau. TOPS không tính lưu lượng bộ nhớ, operator không được hỗ trợ, tiền xử lý, hậu xử lý và chi phí phía host. Hãy so sánh cùng mô hình, precision, độ phân giải, độ chính xác và toàn pipeline."
  - q: "Hiệu chuẩn NPU là gì?"
    a: "Hiệu chuẩn chạy các ảnh triển khai đại diện, thường không có nhãn, qua mô hình để compiler ước tính dải activation cho INT8 hoặc một định dạng precision thấp khác. Ảnh ngẫu nhiên hoặc không đại diện vẫn có thể tạo ra artifact đã biên dịch nhưng làm giảm độ chính xác của tác vụ."
  - q: "LibreYOLO có hỗ trợ Edge NPU không?"
    a: "LibreYOLO xuất ONNX và một số định dạng runtime, có đường dẫn biên dịch RKNN trực tiếp cho các mô hình Rockchip được chọn, và tài liệu hóa quy trình biên dịch Hailo bên ngoài. Mọi artifact riêng của nhà cung cấp khác vẫn cần SDK của họ; chỉ nên mô tả là ứng viên tích hợp cho đến khi được biên dịch, kiểm chứng và chạy trên phần cứng."
---

**Thị trường NPU không phải một thị trường đơn nhất. Hướng dẫn không toàn diện này theo dõi 69 công ty và hệ sinh thái nền tảng: 51 nhà cung cấp có chip hoặc nền tảng triển khai được, chín bên cung cấp IP NPU và chín công ty trong danh sách theo dõi được gắn nhãn rõ ràng.** Họ bao phủ nhiều loại accelerator không tương thích với nhau và gần như cũng nhiều hệ compiler. Phần lớn hứa hẹn cùng một quy trình: xuất mô hình, lượng tử hóa, biên dịch, chép artifact độc quyền sang board rồi gọi runtime của nhà cung cấp. Các chi tiết quyết định quy trình đó mất một buổi chiều hay cả một quý kỹ thuật.

Hướng dẫn này lập bản đồ các công ty cung cấp phần cứng computer vision đáng tin cậy trong năm 2026. Bài ghi lại chip liên quan, tên SDK và compiler, artifact triển khai, mức độ truy cập công khai và các mô hình mà nhà cung cấp thực sự tài liệu hóa. Bài cũng chú ý riêng đến Amlogic, vì stack ADLA mới hơn khác biệt đáng kể so với hệ sinh thái NPU A311D cũ.

> **Phạm vi nghiên cứu:** Các nguồn được kiểm tra ngày 15 tháng 8 năm 2026. Các khẳng định về mô hình có tên dựa trên trang sản phẩm, tài liệu, model zoo, repo hoặc hướng dẫn của nhà cung cấp, trừ khi có ghi rõ khác. TOPS được quảng cáo là số liệu của nhà cung cấp, không phải kết quả benchmark có thể so sánh độc lập. Đây là hướng dẫn cho nhà phát triển, không phải tư vấn đầu tư hay bảng xếp hạng trả phí.

**Điều hướng nhanh:** [bảng công ty](#npu-companies-and-platforms-at-a-glance) | [phân tích sâu Amlogic](#amlogic-two-npu-generations-not-one) | [hồ sơ nhà cung cấp](#the-strongest-public-deployment-ecosystems) | [bảng artifact đã biên dịch](#what-you-actually-deploy-the-artifact-lock-in-table) | [quyền truy cập và vòng đời](#tool-access-and-lifecycle-signals) | [lộ trình LibreYOLO](#what-this-means-for-libreyolo)

## Phát hiện quan trọng nhất

Phần cứng bị phân mảnh, nhưng quy trình triển khai lại nhất quán đáng kể:

```text
PyTorch checkpoint
    -> ONNX or TFLite interchange graph
    -> representative calibration data
    -> vendor quantizer and graph compiler
    -> chip-specific binary or model package
    -> vendor runtime on the target
    -> application preprocessing, decode, NMS and rendering
```

[ONNX cho phép rõ ràng runtime, trình tạo mã và phần triển khai phần cứng](https://onnx.ai/onnx/repo-docs/IR.html), nhưng tệp ONNX chỉ là điểm bàn giao. Điều đó không có nghĩa mọi operator ONNX đều có thể được chuyển xuống mọi accelerator. Hình dạng đầu vào tĩnh, ràng buộc operator được hỗ trợ, quy tắc quantization và chuyển tiếp sang host quyết định phần nào thực sự chạy trên NPU.

Vì vậy, stack phần mềm là một phần của chip. Một NPU nhanh trên danh nghĩa nhưng có compiler bị giới hạn quyền truy cập hoặc thiếu ổn định có thể là mục tiêu triển khai kém hơn thiết bị nhỏ hơn nhưng có toolchain công khai, model zoo, simulator và runtime ổn định.

## "Được hỗ trợ" có nghĩa gì trong hướng dẫn này

Tài liệu tiếp thị của nhà cung cấp dùng từ *hỗ trợ* rất tùy tiện. Bài viết này phân biệt bốn mức bằng chứng:

| Mức bằng chứng | Điều được chứng minh | Điều không được chứng minh |
|---|---|---|
| **Mô hình đã kiểm chứng** | Nhà cung cấp công bố mục mô hình, kết quả hoặc bảng tương thích riêng cho chip | Checkpoint bạn chỉnh sửa vẫn giữ độ chính xác tương tự |
| **Ví dụ chính thức** | Nhà cung cấp có hướng dẫn hoặc demo đầu cuối cho một mô hình cụ thể | Các kích thước, tác vụ hoặc thế hệ khác cũng biên dịch được |
| **Khả năng của compiler** | SDK nhập được framework hoặc công bố các operator được hỗ trợ | Một graph YOLO cụ thể chạy đầu cuối được |
| **Tiếp thị hoặc giới hạn truy cập** | Sản phẩm hướng đến computer vision và có SDK riêng tư | Khả năng tương thích mô hình có thể tái lập công khai |

Phân biệt này rất quan trọng. "Nhập được ONNX" không tương đương với "hỗ trợ phân đoạn YOLO11". Mô hình có thể được phân tích nhưng chuyển tiếp sang CPU, biên dịch với giá trị số đầu ra không chính xác, vượt bộ nhớ accelerator hoặc mất độ chính xác khi quantization.

## Tổng quan các công ty và nền tảng NPU

Các bảng chủ ý kết hợp SoC, accelerator rời, cảm biến thông minh và mục tiêu GPU/FPGA liên quan. Chúng cùng cạnh tranh cho một quyết định triển khai, dù nhà cung cấp gọi chúng là NPU, AIPU, BPU, KPU, DLA, HTP, TPU hay MLA.

### SoC nhúng, cảm biến thông minh và bộ xử lý công nghiệp

| Công ty | Chip hoặc nền tảng liên quan | SDK, compiler và artifact | Bằng chứng computer vision được tài liệu hóa công khai | Quyền truy cập |
|---|---|---|---|---|
| [Amlogic](https://github.com/Amlogic-NN/amlnn-toolkit) | A311D2, S928X, S905X5/S905D5, A311Y3, C308L/C302X/C302X2, T968D4, C305X2 và A123X | AMLNN Toolkit và `libnnsdk.so`, `.adla` đã biên dịch; stack Acuity cũ riêng biệt với `.nb` cho A311D | [Model playground](https://github.com/Amlogic-NN/amlnn-model-playground) tài liệu hóa YOLOv5/6/7/8/10/11, YOLOX, YOLOE, YOLO-World, PP-YOLOE, phân đoạn, tư thế và OBB trên A311D2, S905X5, A311Y3, C305X2 và A123X; các chip còn lại ở đây là mục tiêu compiler, không có trong ma trận hỗ trợ đó | GitHub và wheel công khai; vẫn cần BSP/driver tương thích |
| [Rockchip](https://github.com/airockchip/rknn-toolkit2) | RK3562/3566/3568, RK3576, RK3588, RV1126B | RKNN-Toolkit2, RKNN Runtime, `.rknn` | YOLOv5/6/7/8/10/11, YOLOX, YOLO-World, PP-YOLOE, phân đoạn, tư thế và OBB trong [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) | GitHub công khai; wheel nhị phân |
| [Qualcomm](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | Nền tảng Snapdragon và Dragonwing, QCS6490, QCS8550 và Dragonwing IQ-9075 | QAIRT/QNN, SNPE và AI Hub; binary ngữ cảnh QNN hoặc DLC | [Bộ sưu tập mô hình AI Hub](https://github.com/qualcomm/ai-hub-models) có YOLOv3/5/6/7/8/9/10/11/26, YOLOX, YOLO-World, YOLOR, RF-DETR và mô hình phát hiện/phân đoạn/tư thế | Tài liệu công khai; yêu cầu SDK/tài khoản thay đổi |
| [Texas Instruments](https://github.com/TexasInstruments/edgeai-tidl-tools) | AM62A, AM67A, AM68A, AM69A và TDA4x; TDA54-Q1 đang preview | Processor SDK Edge AI và TIDL | [Edge AI Model Zoo](https://github.com/TexasInstruments/edgeai-modelzoo) công bố mô hình phát hiện, phân đoạn, tư thế và phân loại đã tối ưu cho đường dẫn AM6xA/TDA4 hiện hành; chưa phải ma trận mục tiêu cho TDA54-Q1 đang preview | GitHub công khai cùng Processor SDK |
| [NXP](https://www.nxp.com/design/design-center/software/development-software/eiq-ml-development-environment:EIQ) | i.MX 8M Plus, i.MX 93, i.MX 95 và Kinara Ara-1/Ara240 đã mua lại | eIQ với TIM-VX, Vela, Neutron Converter và Ara SDK | [eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo) có asset hoặc recipe YOLOv4-tiny/v8, NanoDet, CenterNet, FastestDet, SSD Lite và YOLACT; mục YOLOv5 chỉ có tài liệu | Thành phần công khai và giới hạn tài khoản kết hợp |
| [STMicroelectronics](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) | Các biến thể STM32N6x7, gồm STM32N657/647, với accelerator Neural-ART | STM32Cube AI Studio và ST Edge AI Core | Repo dịch vụ hiện hành liệt kê Tiny YOLOv2, YOLOv5u, YOLOv8, YOLO11, YOLO26 và ST-YOLOX, cùng tư thế và phân đoạn | Công cụ và repo công khai |
| [Infineon](https://documentation.infineon.com/psocedge/) | PSOC Edge E83/E84 với Cortex-M55 và Ethos-U55; accelerator NNLite riêng ở phía M33 | ModusToolbox, [DEEPCRAFT Model Converter](https://www.infineon.com/design-resources/embedded-software/deepcraft-edge-ai-solutions/deepcraft-model-converter), TFLite Micro và Arm Vela | Tài liệu kiến trúc nêu MobileNetV1/V2 làm kernel đại diện và bộ AI E84 có camera, nhưng không tìm thấy ma trận kiểm chứng YOLO bên thứ nhất cho PSOC Edge | Tài liệu công khai, thành phần SDK và bộ đánh giá E84 hiện hành |
| [Renesas](https://github.com/renesas-rz/rzv_drp-ai_tvm) | RZ/V2L, V2M, V2MA, V2H và V2N | DRP-AI Translator, DRP-AI TVM và RUHMI | [Danh sách mô hình RZ/V2H](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/model_list/Model_List_V2H.md) kiểm chứng các kích thước YOLOv5/v8/v11/26, YOLOX, tư thế và biến thể phân đoạn | GitHub công khai cùng board SDK |
| [Sony](https://www.aitrios.sony-semicon.com/edge-ai-devices/imx500) | Cảm biến thị giác thông minh IMX500 và Raspberry Pi AI Camera | Edge-MDT, Model Compression Toolkit và IMX500 packer; gói `.rpk` | [IMX500 zoo của Raspberry Pi](https://github.com/raspberrypi/imx500-models) có YOLOv8n, YOLO11n, EfficientDet Lite, NanoDet+ và SSD | Quy trình Raspberry Pi công khai; áp dụng điều khoản công cụ của Sony |
| [Ambarella](https://www.ambarella.com/developer/model-garden/) | Họ CV72/CV75, CV5/CV52 và CV3-AD | Cooper Developer Platform, compiler CVflow và runtime DAG | Model garden công khai nêu YOLOX-S, RTMDet-nano, DeepLabV3+, TopFormer, OWL-ViT và LLaVA OneVision | Chủ yếu giới hạn cho đối tác |
| [Synaptics](https://developer.synaptics.com/docs/sl/overview) | Astra SL1600/SL1680; SL2611/13/15/17/19; họ MCU SR100 riêng biệt | SyNAP `.synap` trên SL16xx; Torq/IREE `.vmfb` trên SL261x; SDK SR riêng hướng đến Ethos-U55 | Benchmark SyNAP YOLOv8n/v8s chính thức và ví dụ Torq YOLOv8 cho SL261x; cần đánh giá riêng bằng chứng cho SR | Cổng nhà phát triển; một số bản tải xuống bị giới hạn |
| [MediaTek](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/index.html) | Genio 360/360P/420/520/720 với NP8 và MDLA 5.3; Genio 510/700/1200 với NP6/MDLA cũ hơn | NeuroPilot Converter, `ncc-tflite`, Neuron Runtime và `.dla`; đường dẫn ONNX Runtime trên một số hệ thống mới hơn | IoT AI Hub chính thức công bố trang mô hình và benchmark cho YOLOv5, YOLOv8 cùng mô hình phân loại và khuôn mặt | Tài liệu Yocto công khai; gói NP8 chủ chốt và tài liệu Android cần quyền khách hàng trực tiếp/NDA |
| [Allwinner](https://docs.aw-ol.com/v853/en/npu/dev_npu/) | SoC thị giác V853 với NPU Vivante 1-TOPS | Chuyển đổi Acuity/Pegasus và runtime `viplite` | Tài liệu V853 chính thức nêu YOLOv2/3/4/4-tiny/5/5s, RetinaNet, MobileNet, ResNet và mạng khuôn mặt/người | Tài liệu công khai; có thể cần tài khoản để tải SDK |
| [Canaan/Kendryte](https://github.com/kendryte/nncase) | K230/K230D hiện hành; KPU K210/K510 cũ | Compiler và runtime `nncase`; `.kmodel` | Hướng dẫn nncase K230 bao gồm biên dịch/mô phỏng/runtime YOLOv5s; danh mục demo chính thức và [CanMV changelog](https://github.com/kendryte/canmv_k230/blob/canmv_k230/CHANGELOG.md) hiện hành tài liệu hóa tác vụ YOLOv8/11/26 | Compiler/PyPI công khai; plugin chip là binary |
| [Alif Semiconductor](https://alifsemi.com/support/kits/ensemble-e7appkit/) | Ensemble E7 với hai accelerator ML, gồm Ethos-U55; [E8](https://alifsemi.com/ensemble-e8-series/) có một Ethos-U85 và hai NPU Ethos-U55 | Triển khai dựa trên Arm Vela và TFLite Micro | Alif công bố benchmark YOLO-Fastest phát hiện khuôn mặt ở cấp họ sản phẩm, INT8, nhưng không có ma trận YOLO hiện đại rộng theo từng mục tiêu | Tài liệu công khai và tài nguyên bộ đánh giá |
| [Himax](https://www.himax.com.tw/products/wiseeye-ai-sensing/wiseeye2-ai-processor/) | MCU AI đầu cuối HX6538 WiseEye2 với Cortex-M55 và Ethos-U55 | TFLite Micro cùng Arm Vela, có CMSIS-NN và kernel tham chiếu dự phòng | [Ví dụ WiseEye2 chính thức](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2) gồm phát hiện, tư thế và phân loại YOLOv8n, phát hiện YOLO11n, face mesh và PeopleNet | GitHub, tài liệu công khai và board đối tác có thể mua |
| [Analog Devices](https://www.analog.com/en/products/max78002.html) | MCU AI MAX78000 và MAX78002 với accelerator CNN công suất thấp | [`ai8x-training`](https://github.com/analogdevicesinc/ai8x-training), `ai8x-synthesis`/`izer` và MSDK; C cùng trọng số được tạo ra | Tài liệu chính thức bao quát nhận dạng/phát hiện khuôn mặt, RetinaNet, Visual Wake Words, classifier và nhận dạng hành động; không tìm thấy triển khai YOLO bên thứ nhất | Công cụ GitHub, tài liệu và board đánh giá công khai |
| [D-Robotics](https://github.com/D-Robotics/rdk_model_zoo) | Board BPU RDK X3/X5/Ultra và dòng S100 mới hơn | OpenExplorer/Algorithm Toolchain và `hbm_runtime`; X5 `.bin`, `rdk_s` hiện hành `.hbm` | Nhánh `rdk_x5` hiện hành tài liệu hóa YOLOv5/v5u/v8/v9/v10/11/12/13/26, YOLOE, YOLO-World, phân đoạn, tư thế, phân loại, OCR và CLIP cho RDK X5; X3 và dòng S dùng nhánh riêng | GitHub công khai; một số gói toolchain dành riêng cho nền tảng |
| [AXERA](https://github.com/AXERA-TECH/ax-samples) | AX650, AX637, AX630C, AX620Q và AX615 | Compiler Pulsar2 và AXEngine; `.axmodel` | Mẫu chính thức hiện hành bao phủ YOLOv5/6/7/8/9/10/11/13/26, YOLOX và YOLO-World; tác vụ và mục tiêu thay đổi theo chip | Mẫu và tài liệu công khai; compiler được phân phối riêng |
| [SOPHGO](https://github.com/sophgo/tpu-mlir) | BM1684/1684X/1688/1690 và CV186X/CV18xx | TPU-MLIR và runtime SOPHON; `.bmodel` hoặc `.cvimodel` | Demo chính thức bao phủ YOLOv3/4/5/7/8/9/10/11/12/26, YOLOX, PP-YOLOE, YOLO-World, OBB, phân đoạn, khuôn mặt, SAM và OCR | Compiler mã nguồn mở cùng runtime của nhà cung cấp |
| [Huawei Ascend](https://www.hiascend.com/en/software/cann) | Ascend 310/310P/310B và sản phẩm biên Atlas 200I/300I | CANN, compiler ATC và AscendCL; `.om` | [Ascend ModelZoo](https://github.com/Ascend/modelzoo) chính thức có YOLOv2/3/4/5, YOLOX, YOLOR, SSD, RetinaNet, Mask R-CNN và mô hình phân đoạn | Tài liệu công khai; gói CANN và kernel phải khớp mục tiêu |
| [Cambricon](https://github.com/Cambricon/magicmind_cloud) | MLU370-X4/S4 trong ma trận công khai được trích dẫn; MLU270 là phần cứng cũ không nằm trong ma trận đó | Neuware và MagicMind | Ma trận MagicMind 1.7 trong repo liệt kê YOLOv3/4/5/7/8, PP-YOLOE, SSD, RetinaFace, RetinaNet, Mask R-CNN và mô hình phân đoạn | Ví dụ mô hình lịch sử công khai; quyền truy cập SDK/container hiện tại bị giới hạn |
| [Sunplus](https://sunplus.atlassian.net/wiki/spaces/C3/pages/1971126471/SP7350%2BSpecification) | SP7350/C3V, khoảng 4.1 đến 4.6 TOPS được quảng cáo | Docker NPU Vivante Acuity, runtime SNNF và `.nb` | Tài liệu chính thức có YOLOv5 và quy trình triển khai [YOLOv8 đầu cuối tùy chỉnh](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2427224156/SP7350) | Mã nguồn/tài liệu công khai và hệ board |
| [ESWIN Computing](https://www.eswincomputing.com/en/news/info/96.html) | SoC RISC-V EIC7700/EIC7700X và hai die [EIC7702/EIC7702X](https://www.eswincomputing.com/en/news/info/104.html) | ENNP: EsQuant, compiler EsAAC hiện hành, simulator và runtime ESSDK; `.model`; tài liệu cũ dùng `ennc-compile` | Tài liệu ENNP do Milk-V lưu trữ có hướng dẫn [YOLOv3 đầu cuối](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/yolov3), ví dụ MobileNetV2 và ResNet | Kết hợp tài liệu bên thứ nhất và tài liệu do board lưu trữ; tải xuống theo khu vực |
| [Nuvoton](https://www.nuvoton.com/products/microcontrollers/arm-cortex-m55-mcus/m55m1-series/index.html) | MCU M55M1 với Ethos-U55-256 | TFLite Micro, Arm Vela và NuEdgeWise/NuML | Ví dụ [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise) chính thức nêu YOLOv8-nano, YOLOX-nano, YOLO Fastest, SSD-MobileNet và mô hình phân loại | Toolchain MCU phần lớn công khai |
| [Realtek](https://github.com/Ameba-AIoT/ameba-arduino-pro2) | Nền tảng camera AmebaPro2 RTL8735B / AMB82-mini | SDK Arduino/FreeRTOS, VoE và NeuralNetwork API; `.nb` | Mô hình đóng gói gồm YOLOv3-tiny, YOLOv4-tiny, YOLOv7-tiny, SCRFD và MobileFaceNet | Runtime công khai; cần liên hệ để [truy cập converter tùy chỉnh](https://ameba-doc-arduino-sdk.readthedocs-hosted.com/en/latest/FAQ/offline_ai_model_conversion_steps.html) |
| [Telechips](https://docs.topst.ai/product/p/ai) | Board TCC7500 / TOPST AI, quảng cáo 8 TOPS | TC-NN-Toolkit / Enlight SDK; trung gian `.enlight` và gói triển khai đã biên dịch | [Dự án TOPST chính thức](https://docs.topst.ai/blog/31) đã triển khai YOLOv8s; chuyển đổi UFLD v1/v2 thất bại do layer không hỗ trợ và chỉ đề xuất cách khắc phục bằng tách/hậu xử lý. [YOLOv4](https://community.topst.ai/t/segmentation-fault/358) và [quy trình YOLOv8 khác](https://community.topst.ai/t/segmentation-fault-error/415) xuất hiện trong các luồng hỗ trợ cộng đồng | Tài liệu board công khai; quyền tải compiler/operator thường bị giới hạn |
| [T-Head](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) | TH1520, quảng cáo 4-TOPS INT8, trên LicheePi 4A | Compiler HHB và CSI-NN2/SHL; `hhb.bm` cùng mã/tham số được tạo | Ví dụ chính thức từ nhà cung cấp board chạy YOLOv5n/s và MobileNetV2 trên NPU | Ví dụ công khai; toolchain cũ và mức độ bảo trì chưa chắc chắn |
| [SigmaStar](https://wx.comake.online/doc/doc/SigmaStarDocs-SSU9383CM-SIGMASTAR-202507071022/platform/MI/ipu_zh.html) | SoC camera thông minh SSU9383CM hiện hành và các họ IPU cũ hơn | Runtime MI_IPU hiện hành; toolchain SGS_IPU [cũ hơn](https://wx.comake.online/doc/doc/SigmaStarDocs-SSC9381G_9351_Pudding-ULS00V040-20210913/customer/development/dla/tools.html) tạo `.sim` thành `sgsimg.img` | [Module hậu xử lý SDK chính thức cũ](https://wx.comake.online/doc/doc/Sigmastar_SDK_v1.2.2/module/User_Guide/Common/SigmaStar_Post_Processing_Module.html) nêu SSD và YOLOv1/2/3; chưa kiểm chứng khả năng tương thích công khai của các mô hình và artifact đó với SSU9383CM | Silicon hiện hành, nhưng bằng chứng compiler/mô hình được trích dẫn công khai thuộc các thế hệ khác nhau |
| [HiSilicon](https://www.hisilicon.com/cn/products/smart-vision/machine-vision/hi3516cv610) | SoC thị giác thông minh Hi3516CV610/DV500, Hi3519DV500 và Hi3403V100 | ATC sang `.om` với runtime board NNN/SVP-NNN | Ma trận HiSpark dành riêng cho mục tiêu, đáng chú ý là Hi3403 và Hi3591P, nêu YOLOv3 đến YOLO11, tư thế, phân đoạn, OBB, OCR và độ sâu; không phải kiểm chứng cho mọi SoC trong hàng này | Đang hoạt động và khác Ascend; mô hình repo được ghi chỉ dùng phi thương mại |

### Accelerator và module biên chuyên dụng

| Công ty | Silicon liên quan | SDK và artifact | Bằng chứng mô hình được tài liệu hóa công khai | Quyền truy cập |
|---|---|---|---|---|
| [Hailo](https://github.com/hailo-ai/hailo_model_zoo) | Hailo-8, Hailo-8L, Hailo-10H và họ Hailo-15 | Dataflow Compiler và HailoRT; `.hef` | YOLOv3/v4/v5/v6/v7/v8/v9/v10/11/12 và YOLO26, cùng YOLOX, DAMO-YOLO, SSD, EfficientDet, phân đoạn, tư thế và OBB, với bảng mô hình theo từng thế hệ | Model Zoo công khai; compiler qua Developer Zone |
| [Axelera AI](https://docs.axelera.ai/sdk/reference/models/model-zoo/) | Sản phẩm Metis đang bán; sản phẩm Europa và Titania đã công bố | Voyager SDK công khai; Pipeline Builder alpha `.axm`/`.axe`, gói `.axmodel` kiểu cũ | YOLOv3/5/7/8/9/10/11/26, YOLOX, YOLO-NAS, OBB, tư thế, phân đoạn và nhiều mô hình không thuộc YOLO đã được kiểm chứng | SDK công khai trên GitHub; hỗ trợ khách hàng cần tài khoản |
| [DEEPX](https://developer.deepx.ai/modelzoo/) | DX-M1 và DX-M1M | DXNN SDK, DX-COM và DX-RT; `.dxnn` | Model Zoo có 354 mục theo DX-COM 2.4.0/DX-RT 3.4.0 khi kiểm tra ngày 15 tháng 8 năm 2026, gồm YOLOv3 đến YOLO11 và YOLO26, YOLOX, SSD, EfficientDet, NanoDet, phân đoạn, tư thế và OBB | Tài nguyên và bộ công cụ dành cho nhà phát triển công khai |
| [MemryX](https://developer.memryx.com/tools/neural_compiler.html) | Accelerator MX3 và module nhiều chip | MemryX SDK, Neural Compiler và runtime; `.dfp` | Ví dụ chính thức và ghi chú phát hành bao quát phát hiện, phân đoạn và tư thế, gồm YOLOv10, YOLO11 và YOLO26 | Tài liệu và SDK công khai |
| [Kneron](https://doc.kneron.com/docs/) | KL520, KL530, KL630, KL720 và KL730 có tài liệu compiler hiện hành; KL830 xuất hiện trong một số PLUS API nhưng không có trong danh sách mục tiêu compiler hiện hành | Kneron PLUS và Model Toolchain; `.nef` trên mục tiêu compiler được tài liệu hóa | [Quy trình YOLO chính thức](https://doc.kneron.com/docs/toolchain/appendix/yolo_example/) tập trung vào Tiny-YOLOv3; tài liệu khác có YOLOv5 | Tài liệu công khai; gói công cụ/tải xuống thay đổi theo chip |
| [SiMa.ai](https://docs.sima.ai/pages/palette/modelsdk.html) | MLSoC và nền tảng Modalix 50-TOPS đang sản xuất | Palette, ModelSDK, MLA Compiler và ModelExecutor | Bản phát hành công khai kiểm chứng YOLOv7/v8, YOLOX, tư thế/phân đoạn, DETR, Mask R-CNN và EfficientDet | Tài liệu công khai; SDK thương mại |
| [EdgeCortix](https://www.edgecortix.com/en/hardware) | SAKURA-II, accelerator quảng cáo 60-TOPS dạng M.2 và PCIe | Compiler MERA và framework | Nhà cung cấp mô tả hỗ trợ từ vision đến generative AI nhưng không công bố ma trận tương thích YOLO hiện hành đủ chính xác | Hỏi về dùng thử/đặt hàng thương mại; kiểm chứng do đối tác dẫn dắt |
| [BrainChip](https://brainchip.com/metatf-dev-tools/) | Coprocessor/module M.2 AKD1500 đang bán; nền tảng AKD1000 cũ; IP Akida 2 | Các gói MetaTF: `akida-models`, `quantizeml`, `cnn2snn` và runtime `akida` | Model card Akida 2 nêu detector YOLOv2 AkidaNet0.5, CenterNet, AkidaUNet và nhận dạng khuôn mặt; kết quả đó không tự động xác nhận AKD1500 | Gói Python cốt lõi công khai; ánh xạ phần cứng vẫn phụ thuộc mục tiêu |
| [Blaize](https://www.blaize.com/products/) | Sản phẩm P1600, Pathfinder và Xplorer | Picasso SDK, NetDeploy và AI Studio | Vision là thị trường mục tiêu, nhưng không tìm thấy ma trận tương thích mô hình có tên công khai và kiểm toán được | Thương mại/giới hạn truy cập |
| [Google Coral](https://github.com/google-coral/edgetpu) | Sản phẩm Edge TPU USB, PCIe, M.2 và Dev Board cũ; IP [Coral NPU](https://github.com/google-coral/coralnpu) mã nguồn mở đang hoạt động là dự án riêng | Edge TPU Compiler/`libedgetpu`/PyCoral cũ; Coral NPU RISC-V mới cung cấp IP, RTL/mô phỏng và ví dụ ELF | Ví dụ cũ tập trung vào SSD MobileNet, phân loại, DeepLab và MoveNet; IP mới không có ma trận triển khai YOLO công khai và không phải sản phẩm kế nhiệm Edge TPU cắm-thay-thế | Repo lõi cũ đã lưu trữ; IP Coral NPU mới đang được phát triển |
| [Lattice Semiconductor](https://www.latticesemi.com/en/Solutions/Solutions/SolutionsDetails02/sensAI?ActiveTab=Design+File) | FPGA ECP5, iCE40 UltraPlus, CrossLink-NX, CertusPro-NX, Avant-E và Avant-X | sensAI Studio, Neural Network Compiler và IP accelerator cấu hình được | Bảng compiler hiện hành liệt kê YOLOv1, YOLOv5, YOLOv8 và YOLO11 theo chế độ dành riêng cho từng mục tiêu, cùng SSD, MobileNetV2-SSD, ResNet và ENet | Compiler/tài liệu công khai; điều khoản IP thay đổi, Advanced CNN Accelerator là sản phẩm thương mại |
| [Mobilint](https://www.mobilint.com/sdk-qb) | Accelerator REGULUS 10-TOPS và ARIES MLA100 80-TOPS | SDK qb; compiler INT8 và `.mxq` | [Model zoo công khai](https://github.com/mobilint/mblt-model-zoo/blob/master/mblt_model_zoo/vision/README.md) liệt kê phát hiện YOLOv3/5/7/8/9/10/11/12/26 cùng biến thể phân đoạn, tư thế và OBB | Tài liệu/mô hình công khai; SDK và phần cứng thương mại |
| [Rebellions](https://rebellions.ai/developers/) | ATOM+ CA22 và ATOM-Max CA25 đang hoạt động; ATOM CA02/ATOM+ CA12 EoL; trạng thái công cụ ATOM-Lite CA21 chưa rõ | Compiler/runtime/profiler RBLN; `.rbln` | Bản phát hành hỗ trợ chính thức nêu họ YOLOv3, YOLOv5/6/7/8 và hướng dẫn YOLOv8 hiện hành | Tài liệu công khai; wheel compiler cần thông tin đăng nhập portal |
| [FuriosaAI](https://furiosa.ai/warboy) | Warboy Gen1 hướng đến vision; thế hệ RNGD riêng biệt | Furiosa SDK; `.enf` INT8 trên Warboy, `.fxb` riêng trên RNGD | Zoo Warboy tài liệu hóa SSD, YOLOv5M/L và YOLOv7-w6-pose; lộ trình RNGD ghi hoàn tất hỗ trợ YOLOv8m trong Q4 2024 nhưng không cung cấp ma trận vision công khai hiện hành chi tiết | Cần IAM/tài khoản; phải tách riêng hai thế hệ |

### Nền tảng liên quan mà nhà phát triển thường so sánh với NPU

| Công ty | Phần cứng | Stack triển khai | Lý do được đưa vào so sánh |
|---|---|---|---|
| [NVIDIA](https://github.com/NVIDIA-AI-IOT/deepstream_tools/tree/main/yolo_deepstream) | GPU Jetson Orin cùng DLA; GPU Jetson Thor không có DLA | JetPack, TensorRT và DeepStream; `.engine` | Triển khai vision rất trưởng thành; công cụ DeepStream chính thức tài liệu hóa YOLOv4/v7/v8/v9/11, gồm cấu hình YOLO11 OBB. Nhiều kết quả dùng GPU và layer Orin DLA không hỗ trợ có thể chuyển về GPU. |
| [AMD](https://vitisai.docs.amd.com/en/6.2/) | Mục tiêu DPU Kria/cũ; Vitis AI 6.2 GA cho Versal AI Edge Gen1 VEK280/VE2802 và Gen2 VEK385; NPU client Ryzen AI riêng | `.xmodel` cũ; snapshot NPU gắn với mục tiêu Gen1; thư mục cache biên dịch hoặc gói `.rai` production Gen2 | Vitis AI công bố tài liệu vision, nhưng DPU cũ, Versal Gen1, Versal Gen2 và Ryzen AI có hợp đồng biên dịch/triển khai riêng. |
| [Microchip](https://www.microchip.com/en-us/products/fpgas-and-plds/fpga-and-soc-design-tools/vectorblox) | FPGA PolarFire SoC với IP accelerator CoreVectorBlox cấu hình được | VectorBlox SDK 3.1 công khai và Libero; asset triển khai `.vnnx`, `.hex` và `.ucomp` | [Hướng dẫn hiện hành](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/README.md) bao quát YOLOv5n, YOLOv8 phát hiện/phân loại/OBB/tư thế/phân đoạn, YOLOv9t và mô hình vision khác; SDK 3.1 hiện chỉ áp dụng cho PolarFire SoC Video Kit. |
| [Intel](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md) | NPU Core Ultra, CPU và GPU tích hợp/rời | OpenVINO IR hoặc cache mô hình đã biên dịch | OpenVINO cung cấp đường dẫn XPU chéo công khai; hãy dùng cột NPU trong [ma trận mô hình đã kiểm chứng](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html), đừng giả định mọi ví dụ YOLO của OpenVINO đều được kiểm chứng trên NPU. |
| [Apple](https://developer.apple.com/machine-learning/core-ml/) | Apple Neural Engine trong chip A-series từ A11 trở lên và chip M-series | Core ML và `coremltools`; `.mlpackage`/`.mlmodelc` | NPU tiêu dùng dễ tiếp cận qua Core ML, nhưng Apple trừu tượng hóa việc phân chia chính xác giữa CPU/GPU/Neural Engine thay vì cung cấp compiler phần cứng riêng cho YOLO. |
| [GreenWaves Technologies](https://github.com/GreenWaves-Technologies/nn_menu_gap9) | GAP9 với neural engine NE16 | GAP SDK, NNTool và mã do AutoTiler tạo | Repo chính thức có MobileNet SSD, phát hiện khuôn mặt, phân loại và detector người [YOLOX riêng](https://github.com/GreenWaves-Technologies/yolox_people_detection). SDK đầy đủ chỉ dành cho khách hàng đủ điều kiện. |
| [Syntiant](https://www.syntiant.com/hardware) | Neural Decision Processor NDP200 sản xuất hàng loạt và NDP250 đang lấy mẫu | Syntiant SDK và gói triển khai | Bộ xử lý vision/cảm biến luôn bật công suất cực thấp: NDP200 được tài liệu hóa dưới 1 mW, còn nhận dạng ảnh trên NDP250 dưới 30 mW. Không tìm thấy ma trận YOLO công khai rộng. |

## Amlogic: hai thế hệ NPU, không phải một

Amlogic cần được giải thích sâu hơn vì thông tin trên web thường gộp các sản phẩm không tương thích. Công ty có đường dẫn cũ dựa trên VeriSilicon và đường dẫn ADLA độc quyền mới hơn. Chúng không dùng chung compiler, artifact hay runtime.

| Thế hệ | Chip tiêu biểu | NPU và toolchain | Artifact đã biên dịch | Trạng thái thực tế |
|---|---|---|---|---|
| Cũ | A311D và S905D3, thường thấy trên Khadas VIM3/VIM3L | VeriSilicon Vivante VIPNano-QI, Acuity toolkit, KSNN và `aml_npu_sdk` cũ | `.nb` trong thư mục đầu ra `nbg_unify` | Board và demo hiện hữu; tích hợp cũ riêng biệt |
| ADLA2 hiện hành | C308L/C302X, S928X, A311D2, T968D4, S905X5/S905D5 và C302X2 | Amlogic ADLA2, AMLNN Toolkit và NNSDK2 | `.adla` riêng cho mục tiêu; W8A8 hoặc W8A16 | Stack hiện hành tập trung vào số nguyên |
| ADLA3 hiện hành | A311Y3, C305X2 và A123X | Amlogic ADLA3, AMLNN Toolkit và NNSDK2 | `.adla` riêng cho mục tiêu; W4A8, W8A8, W4A16, W8A16 hoặc W16A16 | Bổ sung khả năng INT4, FP16 và BF16 gốc |

[Tài liệu A311D](https://dl.khadas.com/products/vim3/datasheet/a311d-datasheet.pdf) xác định NPU INT8 5-TOPS đời đầu. Trang ứng dụng [Khadas VIM3 cũ](https://docs.khadas.com/products/sbc/vim3/npu/npu-app) và [tổng quan NPU](https://docs.khadas.com/products/sbc/vim3/npu/start) tài liệu hóa DenseNet CTC, MTCNN, RetinaFace, YOLOFace, YOLOv2, YOLOv3, YOLOv3-tiny, YOLOv4, YOLOv7-tiny và YOLOv8n; trang tổng quan còn tài liệu hóa YOLOv8n-pose và đánh dấu FaceNet đã lỗi thời. Đây là ví dụ thực tế, nhưng chỉ là bằng chứng cho hệ sinh thái Acuity `.nb` cũ, không phải chip ADLA hiện hành. A311D không phải A311D2, và C305X không phải C305X2.

Còn có một bẫy phần cứng khác. Hướng dẫn phiên bản [Khadas VIM4](https://docs.khadas.com/products/sbc/vim4/configurations/identify-version) cho biết board V12/A311D2 bản sửa đổi B ban đầu không có NPU; NPU được quảng cáo 3.2-TOPS chỉ xuất hiện trên V13A trở lên dùng linh kiện A311D2-N0D phiên bản sửa đổi C. Chỉ dựa vào tên sản phẩm thì chưa đủ để chọn thiết bị kiểm thử.

### Stack AMLNN và ADLA hiện hành

[AMLNN Toolkit](https://github.com/Amlogic-NN/amlnn-toolkit/tree/7d3cc9a36179b756ce79c953c14d43acab1f2af5) hiện hành bao quát chuyển đổi mô hình, quantization, biên dịch, suy luận (inference) và profiling. [Hướng dẫn người dùng tháng 5 năm 2026 được ghim](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/02_Amlogic_NPU_User_Guide_V0.1.pdf) tài liệu hóa importer cho ONNX, TFLite dấu phẩy động hoặc đã lượng tử hóa, TorchScript `.pt`, và PyTorch 2 ExportedProgram/PT2. Hướng dẫn chi tiết đánh dấu đường dẫn TensorFlow, Paddle và Keras là dự kiến, dù phần tổng quan repo dùng cách diễn đạt rộng hơn. Compiler tạo `.adla`. Triển khai Python dùng AMLNN hoặc `amlnn_edge_toolkit_lite`; C/C++ gốc liên kết `libnnsdk.so` qua `nnsdk2.h`. Phát triển phía host có thể dùng ADB với `nnserver`; runtime còn nhắm đến Android, Buildroot, Yocto và một số môi trường Debian/Armbian.

Các mã nền tảng được toolkit công bố hiện gồm:

| ID mục tiêu | Nền tảng |
|---:|---|
| `001` | C308L / C302X |
| `002` | S928X |
| `003` | A311D2 |
| `004` | T968D4 |
| `005` | S905X5 / S905D5 |
| `006` | C302X2 |
| `007` | A311Y3 |
| `008` | C305X2 / A123X trong tài liệu chi tiết và ma trận mô hình |

Trường mục tiêu đó không chỉ để trang trí. Amlogic yêu cầu `.adla`, `nnsdk2.h`, `libnnsdk.so`, BSP và thế hệ compiler/runtime phải khớp nhau. [Quick Start](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/01_Amlogic_NPU_Quick_Start_guide_V0.1.pdf) được ghim yêu cầu driver ADLA 2.0.2 trở lên, NNSDK 3.0.0 trở lên và NNSDK2 1.0.0 trở lên; [hướng dẫn triển khai Android](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/06_Amlogic_NPU_Android_Deployment_Guide_V0.1.pdf) làm rõ mối quan hệ giữa thư viện, header, driver và BSP. Hãy xem `.adla` là artifact build gắn với ABI, không phải mô hình có thể chuyển đổi tự do.

Quantization cũng phụ thuộc thế hệ NPU. Mục tiêu ADLA2 từ 001 đến 006 tài liệu hóa biên dịch W8A8 và W8A16. ADLA3 mục tiêu 007 và 008 bổ sung tổ hợp W4A8, W4A16, W8A8, W8A16 và W16A16, cùng hỗ trợ INT4, FP16 và BF16 gốc trong [hướng dẫn operator](https://github.com/Amlogic-NN/amlnn-toolkit/blob/7d3cc9a36179b756ce79c953c14d43acab1f2af5/doc/05_Amlogic_NPU_Support_Operator_List_V0.1.pdf). Amlogic khuyến nghị 200 đến 500 mẫu calibration đại diện. Tùy chọn dữ liệu ngẫu nhiên được nêu rõ là dành cho kiểm thử hiệu năng, không phải quantization giữ nguyên độ chính xác.

### Phạm vi mô hình Amlogic được tài liệu hóa

[Model playground Amlogic được ghim](https://github.com/Amlogic-NN/amlnn-model-playground/blob/03ca63ef20c4f1560722de26ab910826884f334a/README.md) là bằng chứng mạnh hơn nhiều so với khẳng định chung chung rằng hỗ trợ ONNX. Ma trận hỗ trợ/ví dụ công bố có A311D2, S905X5, A311Y3, C305X2 và A123X. Việc compiler chấp nhận các ID mục tiêu khác không chứng minh toàn bộ danh sách mô hình này đã được kiểm chứng trên đó.

| Tác vụ | Mô hình được tài liệu hóa |
|---|---|
| Phân loại | MobileNetV2 và ResNet50-v2; DINO trên ADLA3 |
| Phát hiện đối tượng | PP-YOLOE, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv10, YOLO11, YOLOE, YOLO-World, YOLOX và phát hiện QR |
| Khuôn mặt và cử chỉ | RetinaFace và Gesture Recognition |
| Phân đoạn | DeepLabV3, PP-LiteSeg, YOLOv5-seg và YOLOv8-seg |
| Phát hiện hướng xoay | YOLOv8 OBB |
| Tư thế | Bộ phát hiện/landmark BlazePose và YOLOv8 pose |
| OCR và tiếng nói | LPRNet, các biến thể PaddleOCR, Whisper Tiny và SenseVoice trên một số chip mới hơn |
| Transformer và đa phương thức mới hơn | DETR, MobileSAM, CLIP/MobileCLIP và một số ví dụ LLM/VLM bit thấp trên nền tảng mới hơn |

Cùng repo đó công bố các số liệu model-runtime W8A8 sau. Đây là phép đo của nhà cung cấp cho mạng nơ-ron trên NPU, không phải pipeline camera hoàn chỉnh.

| Mô hình và đầu vào | S905X5 | A311D2 | A311Y3 |
|---|---:|---:|---:|
| YOLOv8n, 640 x 640 | 101.72 FPS | 95.14 FPS | 191.06 FPS |
| YOLOv8s, 640 x 640 | 42.33 FPS | 42.77 FPS | 83.08 FPS |
| YOLOv8m, 640 x 640 | 19.67 FPS | 19.82 FPS | 35.30 FPS |
| YOLOv8l, 640 x 640 | 10.53 FPS | 10.12 FPS | 18.37 FPS |
| YOLO11n, 640 x 640 | 41.14 FPS | 41.48 FPS | 62.24 FPS |

Các lưu ý ở đây đặc biệt quan trọng. Hướng dẫn operator Amlogic đặt NMS trong phần mềm trên cả ADLA2 lẫn ADLA3. README định nghĩa đây là kết quả model-runtime gốc trên NPU, không gồm tiền xử lý và hậu xử lý; không công bố liệu mọi lần chuyển bộ nhớ có được tính hay không. Các hàng độ chính xác YOLO dùng tập con COCO 300 ảnh thay vì toàn bộ tập validation, còn các hàng phân loại dùng ImageNet `val1000`. Bảng cũng báo độ trễ YOLOv8n A311Y3 khác với bảng FPS trong những điều kiện repo không giải thích. Không công bố xung nhịp, chế độ điện, làm mát, trạng thái nhiệt ổn định, phiên bản SDK/BSP chính xác hay thời lượng kiểm thử. Hãy dùng số liệu để hiểu một stack của nhà cung cấp, không dùng để xếp hạng với nhà cung cấp khác.

### Vì sao Amlogic đáng quan tâm về mặt chiến lược

Amlogic có bốn đặc điểm khác thường: phạm vi SoC nhúng lớn, stack compiler mới được công khai, ma trận mô hình hiện hành trùng nhiều với phạm vi tác vụ LibreYOLO, và tích hợp hạng nhất còn hạn chế trong các thư viện huấn luyện phổ biến. Các repo hiện tại cũng còn mới: chúng xuất hiện công khai cuối năm 2025 và đầu năm 2026, với tài liệu quan trọng đề ngày từ tháng 5 đến tháng 7 năm 2026. Điều này tạo cơ hội đi trước thực sự cùng rủi ro API chưa ổn định.

Một tích hợp gọn nên dùng ONNX export xác định của LibreYOLO làm đầu vào compiler, yêu cầu ảnh calibration đại diện cho bản build precision thấp, tạo `.adla` cùng manifest metadata và nạp qua adapter NNSDK2. A311D đời cũ nên dùng mục tiêu `.nb` khác biệt rõ ràng, vì trình bày `.nb` và `.adla` như một backend sẽ gây lỗi tương thích âm thầm. Repo Amlogic có giấy phép cấp cao nhất Apache-2.0, nhưng nên xác nhận trực tiếp quyền phân phối lại wheel, thư viện chia sẻ, `nnserver` và binary Android AAR đi kèm trước khi LibreYOLO lưu trữ chúng.

## Các hệ sinh thái triển khai công khai mạnh nhất

Các công ty dưới đây hiện có tổ hợp rõ ràng nhất giữa phần cứng có thể tiếp cận, tài liệu kỹ thuật công khai và bằng chứng mô hình có tên. Điều đó không khiến họ nhanh nhất cho mọi trường hợp. Nó giúp đánh giá tuyên bố tương thích dễ hơn trước khi mua phần cứng.

### Hailo

Hailo là ví dụ tiêu biểu về hệ sinh thái accelerator vision chuyên dụng. Mô hình được phân tích thành Hailo Archive, tối ưu hóa và lượng tử hóa bằng ảnh đại diện, rồi biên dịch thành tệp Hailo Executable Format (`.hef`). Ứng dụng nạp HEF đó qua HailoRT.

[Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) có recipe, mô hình pretrained, cấu hình hậu xử lý và dữ liệu hiệu năng cho phát hiện, phân đoạn, phân loại, tư thế và tác vụ khác. Phạm vi YOLO gồm YOLOv3/v4/v5/v6/v7/v8/v9/v10/11/12 và YOLO26 cùng YOLOX, DAMO-YOLO, SSD và EfficientDet. [Bảng phát hiện đối tượng Hailo-8 được ghim phiên bản](https://github.com/hailo-ai/hailo_model_zoo/blob/v2.19.0/docs/public_models/HAILO8/HAILO8_object_detection.rst) là nguồn tương thích tốt hơn trang sản phẩm chung; [ứng dụng độc lập](https://github.com/hailo-ai/hailo-apps/blob/main/hailo_apps/python/standalone_apps/object_detection/README.md) tài liệu hóa pipeline YOLO có thể triển khai.

Có một ranh giới phiên bản quan trọng. Hailo-8 và Hailo-8L dùng Model Zoo 2.x, Dataflow Compiler 3.x và HailoRT 4.x đời cũ, còn Hailo-10 và Hailo-15 dùng thế hệ 5.x hiện hành. Recipe, hành vi parser và HEF không thể dùng thay thế chỉ vì mọi thiết bị đều mang tên Hailo.

Hailo-15 cũng có lớp ứng dụng riêng. Nó dùng Vision Processor Software Package và Hailo Media Library thay vì đường dẫn host kiểu TAPPAS của Hailo-8/10H. Repo tham chiếu công khai [`hailo-camera-apps`](https://github.com/hailo-ai/hailo-camera-apps) đã được lưu trữ ngày 3 tháng 5 năm 2026, còn [`hailo-apps-core`](https://github.com/hailo-ai/hailo-apps-core) vẫn là framework ứng dụng công khai riêng. Repo bị lưu trữ không chứng minh sản phẩm đã EOL, nhưng dự án Hailo-15 nên xác nhận gói ứng dụng hiện hành trong Developer Zone.

[Hướng dẫn triển khai Hailo](/docs/export/hailo) của LibreYOLO chủ ý dừng ở bước bàn giao ONNX tĩnh vì không thể đóng gói compiler độc quyền thành dependency Python. Đó là ranh giới trung thực giữa việc tạo graph phù hợp và tuyên bố đã kiểm thử HEF.

### Rockchip

Rockchip có một trong những cộng đồng Linux nhúng thương mại và người làm phần cứng lớn nhất, đặc biệt qua board RK3588, RK3576 và RK356x. [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2) công khai chuyển đổi và lượng tử hóa mô hình trên host Linux x86; RKNN Runtime hoặc Toolkit-Lite chạy tệp `.rknn` tạo ra trên board.

[RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) ghi rõ chip, họ mô hình và precision khác thường. Bảng hiện tại liệt kê đường dẫn FP16 và INT8 cho YOLOv5/6/7/8/10/11, YOLOX, PP-YOLOE, YOLO-World và biến thể phân đoạn YOLOv5/v8; YOLOv8 OBB và YOLOv8 pose chỉ được liệt kê cho INT8. Zoo còn bao gồm OCR, khuôn mặt và mạng phân đoạn khác.

LibreYOLO đã có [RKNN exporter](/docs/export/rknn) trực tiếp và thận trọng. Nó kiểm tra bốn biến thể phát hiện cụ thể trên RK3588, có thể so sánh simulator compiler với ONNX Runtime, và từ chối họ chưa được kiểm chứng thay vì coi build thành công là dự đoán chính xác. Đây là mô hình tuyên bố hỗ trợ hữu ích cho mọi backend NPU tương lai.

### Axelera AI

Axelera AI, thường bị viết nhầm thành "Accelera", xây dựng AIPU Metis và bán dưới dạng sản phẩm M.2, PCIe và nhiều chip. Họ Metis có thể đặt mua công khai; Europa và Titania là sản phẩm đã công bố trong danh mục nên không nên gộp trạng thái cung ứng thành một. [Voyager SDK công khai trên GitHub](https://github.com/axelera-ai-hub/voyager-sdk) xử lý chọn mô hình, biên dịch, quantization, thực thi và pipeline ứng dụng; hỗ trợ khách hàng vẫn cần tài khoản.

[Voyager Model Zoo](https://docs.axelera.ai/sdk/reference/models/model-zoo/) công khai thuộc hàng giàu thông tin nhất thị trường. Zoo liệt kê biến thể mô hình, kích thước đầu vào, precision, độ chính xác và hiệu năng đo được cho YOLOv3, YOLOv5, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO26 n/s/m/l/x, YOLOX, YOLO-NAS, OBB, tư thế và phân đoạn, cùng nhiều mô hình phân loại và dự đoán dense. Công bố tổn thất quantization và độ chính xác bên cạnh tốc độ hữu ích hơn chỉ công bố TOPS đỉnh.

Tên artifact thay đổi theo thế hệ API. Quy trình biên dịch [Pipeline Builder](https://docs.axelera.ai/sdk/reference/pipeline-builder/model-compilation/) alpha tạo mô hình `.axm` và có thể đóng gói pipeline di động thành `.axe`. Tài liệu pipeline [kiểu cũ](https://docs.axelera.ai/sdk/reference/pipeline/model-formats/) mô tả `.axmodel` cùng metadata mô hình và manifest. Tích hợp cần nhận diện thế hệ Voyager đã cài thay vì cố định một phần mở rộng.

### Qualcomm

Qualcomm có phạm vi triển khai rất rộng, nhưng "NPU Qualcomm" che khuất nhiều giao diện. Nhà phát triển gặp Hexagon HTP qua QAIRT/QNN, quy trình SNPE cũ hơn, dịch vụ biên dịch Qualcomm AI Hub, LiteRT hoặc QNN execution provider của ONNX Runtime tùy thiết bị và loại sản phẩm.

Repo [Qualcomm AI Hub Models](https://github.com/qualcomm/ai-hub-models) chính thức là bằng chứng mô hình mạnh. Repo công bố gói tối ưu cho YOLOv3, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO26 phát hiện/phân đoạn/tư thế, YOLOX, YOLO-World, YOLOR, RF-DETR và nhiều mô hình phân loại, độ sâu, tư thế. AI Hub cũng có profiling theo thiết bị, rất quan trọng vì mô hình biên dịch cho một thế hệ HTP không tự động dùng được trên thế hệ khác.

Chi phí tích hợp chính là độ lớn ma trận: Android so với Linux nhúng, linh kiện đặt hàng QCS so với Snapdragon tiêu dùng, kiến trúc HTP, phiên bản QNN/QAIRT và phương thức quantization đều quan trọng. Các trang hiện hành của Qualcomm không thống nhất về trạng thái [Dragonwing IQ-9075](https://www.qualcomm.com/internet-of-things/products/iq9-series/iq-9075): trang sản phẩm ghi Active và [EVK](https://www.qualcomm.com/developer/hardware/qualcomm-iq-9075-evaluation-kit-evk) có thể đánh giá, trong khi [Product Longevity Program](https://www.qualcomm.com/internet-of-things/products/product-longevity-program) vẫn ghi IQ-9075 là Sampling và liệt kê thời gian hỗ trợ đến năm 2038. Tiền tố SKU đặt hàng là QCS9075; Qualcomm quảng cáo cấu hình 50- và 100-dense-INT8-TOPS cùng hỗ trợ Ubuntu/Yocto. Cần xác nhận trạng thái sản xuất cho đúng SKU đặt hàng, và tích hợp LibreYOLO nên ghi SKU đó thay vì tạo thư mục chung tên "Qualcomm".

### DEEPX

Accelerator DX-M1/DX-M1M của DEEPX dùng DXNN SDK. DX-COM biên dịch và lượng tử hóa mô hình, DX-RT thực thi, artifact triển khai dùng định dạng `.dxnn`. Công ty công bố [DX-AllSuite](https://github.com/DEEPX-AI/dx-all-suite), ví dụ ứng dụng trong [DX-APP](https://github.com/DEEPX-AI/dx_app) và [Model Zoo trực tuyến lớn](https://developer.deepx.ai/modelzoo/).

Danh mục công khai báo cáo 354 mục theo DX-COM 2.4.0 và DX-RT 3.4.0 khi kiểm tra ngày 15 tháng 8 năm 2026. Phạm vi được tài liệu hóa gồm YOLOv3 đến YOLO11 và YOLO26, YOLOX, SSD, EfficientDet, NanoDet, DAMO-YOLO, phân đoạn YOLO, tư thế và hộp xoay. DEEPX là ứng viên tích hợp đặc biệt hợp lý vì ranh giới compiler/runtime và artifact triển khai được nêu rõ, còn phạm vi vision công khai rộng.

## SoC công nghiệp gồm nhiều hệ sinh thái khác nhau

Nhà cung cấp công nghiệp thường có vòng đời sản phẩm dài hơn và tích hợp camera, an toàn cùng thời gian thực tốt hơn SoC trên board phổ thông. Stack AI của họ có thể phức tạp hơn vì một nhà cung cấp có thể bán đồng thời nhiều kiến trúc NPU không liên quan.

### Texas Instruments

Bộ xử lý Edge AI hiện hành của TI gồm AM62A, AM67A, AM68A, AM69A và các thiết bị TDA4 liên quan. Đường dẫn phần mềm kết hợp Processor SDK Linux, compiler/runtime TIDL, [Edge AI TIDL Tools](https://github.com/TexasInstruments/edgeai-tidl-tools) và [Edge AI Model Zoo](https://github.com/TexasInstruments/edgeai-modelzoo). TIDL có thể tích hợp ONNX Runtime, TensorFlow Lite và runtime ứng dụng khác trong khi offload các subgraph được hỗ trợ.

TI công bố nhiều hơn một tuyên bố nhập framework: zoo có mô hình đã chuyển đổi và metadata hiệu năng cho phát hiện đối tượng, phân đoạn, tư thế, phân loại, độ sâu và tác vụ khác. TI cũng lưu ý artifact model zoo là điểm khởi đầu phát triển, không tự động sẵn sàng production. Đây là lưu ý quan trọng thường thiếu trong các so sánh nhà cung cấp.

TI cũng liệt kê [TDA54-Q1](https://www.ti.com/product/TDA54-Q1) ở trạng thái Preview, với tối đa bốn NPU C7 và tối đa 400 TOPS trên linh kiện đó; toàn dòng TDA5 được quảng cáo đến 1,200 TOPS. Đây là tuyên bố sản phẩm của nhà cung cấp cho thế hệ mới, không phải căn cứ chuyển kết quả kiểm chứng mô hình TIDL AM6xA/TDA4 sang TDA54-Q1 trước khi TI công bố ma trận riêng cho mục tiêu này.

### NXP

NXP cần ít nhất bốn mô tả backend:

| Mục tiêu NXP | Accelerator | Đường dẫn compiler/runtime |
|---|---|---|
| i.MX 8M Plus | NPU VeriSilicon Vivante 2.3-TOPS | eIQ với TIM-VX/VX delegate |
| i.MX 93 | Arm Ethos-U65 | TFLite lượng tử hóa cùng Arm Vela |
| i.MX 95 | NPU NXP eIQ Neutron | Neutron Converter và eIQ runtime |
| Ara-1 / [Ara240](https://www.nxp.com/products/ARA240) | NPU rời có nguồn gốc Kinara; Ara240 quảng cáo tối đa 40 eTOPS | Ara SDK, tích hợp vào câu chuyện eIQ rộng hơn |

[eIQ Model Zoo](https://github.com/NXP/eiq-model-zoo) có asset hoặc recipe YOLOv4-tiny, YOLOv8, NanoDet, CenterNet, FastestDet, SSD Lite, YOLACT và mô hình khác; mục YOLOv5 chỉ có tài liệu. Một mục được kiểm chứng trên backend này không phải bằng chứng cho cả bốn. Hướng dẫn [xuất YOLO cho nền tảng NXP i.MX](https://community.nxp.com/t5/i-MX-Processors-Knowledge-Base/Exporting-YOLO-Models-for-NXP-i-MX-Platforms/ta-p/2381361) của NXP minh họa vấn đề chuyển đổi riêng theo mục tiêu. NXP cho biết bộ [eIQ Toolkit nguyên khối ngừng được cập nhật sau phiên bản 1.17 trong Q3 2025](https://community.nxp.com/t5/eIQ-Machine-Learning-Software/eIQ-FAQ/ta-p/1099741); quy trình hiện hành dùng các gói độc lập như eIQ Neutron SDK.

NXP [hoàn tất mua lại Kinara vào tháng 10 năm 2025](https://media.nxp.com/news-releases/news-release-details/nxp-completes-acquisitions-aviva-links-and-kinara-advance/). [Tờ dữ liệu Ara240](https://www.nxp.com/docs/en/fact-sheet/ARA240DNPUFS.pdf) đạt tối đa 40 eTOPS theo quảng cáo nhà cung cấp và báo cáo 313 ảnh mỗi giây cho YOLOv8n. NXP liệt kê cả Ara SDK và eIQ Toolkit trên trang sản phẩm, nhưng điều đó không xác lập khả năng trao đổi artifact với [đường dẫn Neutron i.MX 95](https://eiq.nxp.com/learning-hub/convQuant/neutron.html) riêng biệt. [Module M.2 16 GB](https://www.nxp.com/design/design-center/development-boards-and-designs/ARA2-M2-16G-GT) đang hoạt động, còn tùy chọn USB ở giai đoạn tiền sản xuất. NXP định nghĩa chữ "e" trong eTOPS là "equivalent", không phải "effective"; đây không phải metric TOPS INT8 dày đặc của nhà cung cấp khác và không nên so sánh trực tiếp.

### STMicroelectronics

[Thiết bị STM32N6x7](https://www.st.com/en/microcontrollers-microprocessors/stm32n6-series.html), gồm STM32N657 và STM32N647, đưa accelerator Neural-ART 600-GOPS vào sản phẩm cấp MCU; dòng đa dụng N6x5 không có accelerator này. [STM32Cube AI Studio](https://www.st.com/en/development-tools/stedgeai-cubeai.html) và ST Edge AI Core phân tích, tối ưu mô hình nhập vào, còn [STM32 AI Model Zoo Services](https://github.com/STMicroelectronics/stm32ai-modelzoo-services) cung cấp quy trình triển khai, tối ưu, benchmark và ví dụ.

[Tổng quan phát hiện đối tượng hiện hành](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/main/object_detection/docs/README_OVERVIEW.md) tài liệu hóa Tiny YOLOv2, YOLOv5u, YOLOv8, YOLO11, YOLO26 và ST-YOLOX cùng dịch vụ phân loại, tư thế và phân đoạn. Đây không cùng cấp hiệu năng với card PCIe 200-TOPS. Nền tảng này đáng quan tâm vì tiếp nhận camera, inference và điều khiển có thể nằm trong giới hạn công suất/bộ nhớ nhúng rất chặt.

### Renesas

Bộ xử lý RZ/V của Renesas tích hợp accelerator DRP-AI. Phần mềm chuyển dần từ DRP-AI Translator và DRP-AI TVM sang [RUHMI](https://www.renesas.com/en/software-tool/ruhmi-framework), sử dụng công nghệ MERA của EdgeCortix. [Repo RZ/V DRP-AI TVM mở](https://github.com/renesas-rz/rzv_drp-ai_tvm) và [danh sách kiểm chứng RZ/V2H](https://github.com/renesas-rz/rzv_drp-ai_tvm/blob/main/docs/model_list/Model_List_V2H.md) nêu YOLOv5, YOLOv8, YOLO11 và biến thể n/s/m YOLO26, YOLOX, mạng tư thế và phân đoạn cùng mô hình phân loại.

Renesas là mục tiêu đáng tin cậy cho robot và công nghiệp, nhưng cần ghi lại board chính xác, thế hệ DRP-AI, phiên bản translator và hậu xử lý phía CPU để có thể tái lập.

## Cảm biến thông minh và bộ xử lý ưu tiên camera

### Sony IMX500

Sony IMX500 không phải bộ xử lý ứng dụng thông thường. Nó xếp chồng cảm biến ảnh và xử lý AI để inference có thể diễn ra ngay trong cảm biến, giảm lượng dữ liệu ảnh cần rời khỏi camera. Raspberry Pi AI Camera giúp kiến trúc này dễ tiếp cận khác thường.

Repo mô hình [Raspberry Pi IMX500 chính thức](https://github.com/raspberrypi/imx500-models) công bố YOLOv8n, YOLO11n, EfficientDet Lite0, NanoDet+ và SSD MobileNetV2 FPN Lite dưới dạng gói. [Tài liệu AI Camera](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) giải thích chuyển đổi, đóng gói và hậu xử lý trên cảm biến. Đơn vị triển khai là gói RPK thay vì tệp ONNX chung; bộ nhớ cảm biến và ràng buộc operator là giới hạn thiết kế then chốt. [Trang sản phẩm](https://www.raspberrypi.com/products/ai-camera/) của Raspberry Pi cho biết camera này sẽ tiếp tục được sản xuất ít nhất đến tháng 1 năm 2028.

### Ambarella

Bộ xử lý CVflow của Ambarella được sử dụng rộng rãi trong camera, drone và vision ô tô. Họ sản phẩm hiện hành gồm CV72/CV75 cho camera, CV5/CV52 cho hệ thống vision hiệu năng cao hơn và thiết bị ô tô CV3-AD. Nền tảng nhà phát triển Cooper cùng quy trình biên dịch/runtime CVflow là stack phần mềm được nêu công khai.

[Model garden dành cho nhà phát triển công khai](https://www.ambarella.com/developer/model-garden/) nêu YOLOX, RTMDet, DeepLabV3+, TopFormer, OWL-ViT và mô hình đa phương thức. Tuy vậy, tài liệu compiler chi tiết và nội dung tải về vẫn hướng đến đối tác. Vì thế, tích hợp Ambarella nên bắt đầu bằng quan hệ với nhà cung cấp hoặc OEM, không phải giả định có gói pip công khai.

### Synaptics Astra

Synaptics có ba mục tiêu liên quan. Hệ thống Astra SL1600/SL1680 dùng [SyNAP toolkit](https://developer.synaptics.com/docs/synap/introduction), chuyển mô hình nguồn cùng mô tả YAML thành gói `model.synap`. Thiết bị SL2611/13/15/17/19 mới hơn dùng [nền tảng Torq](https://developer.synaptics.com/docs/torq/introduction), quy trình dựa trên IREE/MLIR tạo `.vmfb`. [Dòng SR100](https://developer.synaptics.com/docs/sr/introduction-sr) là họ MCU riêng, dựa trên Cortex-M55 và Ethos-U55 với SDK riêng. Các artifact và API này không thể dùng thay nhau.

[Hướng dẫn benchmark YOLO chính thức](https://developer.synaptics.com/docs/sl/tutorials/vision/benchmark-yolo) cụ thể khác thường: xuất YOLOv8 sang TFLite, calibration UINT8 bất đối xứng trong SyNAP, biên dịch cho SL1680, rồi chạy `synap_cli` và ứng dụng phát hiện đối tượng. Nó chứng minh YOLOv8n/v8s trên SL1680 và mô tả SyNAP hỗ trợ đến YOLO11. [Hướng dẫn phát hiện đối tượng SL261x](https://developer.synaptics.com/docs/sl/sl2600/getting-started/object-detection) riêng biệt chạy YOLOv8 từ Torq `.vmfb`. Đây là ví dụ chính thức cho từng mục tiêu, không phải tuyên bố mọi graph YOLO được hỗ trợ trên cả ba họ.

### MediaTek Genio

MediaTek xứng đáng có mục riêng vì [IoT AI Hub](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/index.html) hiện đã công bố ma trận phần cứng/phần mềm, gói mô hình và bảng benchmark mà các khảo sát thị trường cũ không thể kiểm tra. Genio 360/360P/420/520/720 dùng NeuroPilot 8 với MDLA 5.3; Genio 510/700 dùng NP6 với MDLA 3.0; Genio 1200 dùng NP6 với MDLA 2.0. MediaTek nêu rõ thế hệ NeuroPilot và MDLA, tập operator, compiler và runtime bị ràng buộc phiên bản trong suốt vòng đời mỗi SoC.

Đường dẫn AI phân tích tập trung vào TFLite:

```text
PyTorch model
    -> NeuroPilot Converter and representative calibration
    -> quantized .tflite
    -> version-matched ncc-tflite compiler
    -> chip-generation-specific .dla
    -> Neuron Runtime on MDLA
```

Sản phẩm Genio mới hơn cũng cung cấp delegation LiteRT trực tuyến và đường dẫn CPU/NPU ONNX Runtime trên hệ điều hành được hỗ trợ. Đây là các chế độ thực thi khác với `.dla` ngoại tuyến. [Hướng dẫn kiến trúc phần mềm](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/software_architecture.html) và [ma trận tài nguyên](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/related_resource.html) phân biệt rõ các chế độ này.

[Bảng mô hình phân tích](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/model_zoo/litert_analytical.html) chính thức công bố mục benchmark YOLOv5s và YOLOv8s cùng hướng dẫn chuyển đổi qua nhiều thế hệ MDLA. Bảng nêu rõ không phân phối artifact YOLO chuyển đổi sẵn do hạn chế AGPL-3.0. Các phép đo ngoại tuyến Quant8, 640 x 640 gồm lần lượt 5.35 ms và 8.04 ms trên Genio 720. [Trang mô hình YOLOv8s](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/model_zoo/litert_analytical/YOLOv8s.html) công bố tensor đầu vào/đầu ra, thời gian theo backend và cảnh báo về toán tử tùy chỉnh MediaTek. Đây là benchmark của nhà cung cấp, không phải kết quả camera đầu cuối.

Quyền truy cập là giới hạn. Tài liệu Yocto công khai khá chi tiết, nhưng gói converter/compiler all-in-one NP8 hiện hành và nhiều tài liệu Android được ghi là tài nguyên NDA hoặc khách hàng trực tiếp. Tài khoản nhà phát triển thông thường không có cùng quyền như tài khoản MediaTek Online dành cho khách hàng. Vì vậy, LibreYOLO nên xem Genio đã được chứng minh về kỹ thuật nhưng cần quan hệ đối tác để có tích hợp compiler BYOM tái lập.

## Hệ sinh thái Edge AI tập trung ở Trung Quốc

Một số nền tảng vision chi phí thấp, dễ tiếp cận và có năng lực nhất ít xuất hiện trong danh sách thị trường tiếng Anh.

### D-Robotics và nền tảng BPU có nguồn gốc Horizon

D-Robotics duy trì hệ sinh thái board nhà phát triển RDK quanh accelerator BPU dùng trong RDK X3, X5, Ultra và dòng S100 mới hơn. Nhánh `rdk_x5` hiện hành của [RDK Model Zoo](https://github.com/D-Robotics/rdk_model_zoo) tài liệu hóa các đường dẫn YOLOv5/v5u, YOLOv8/9/10/11/12/13/26, YOLOE, YOLO-World, phát hiện, phân đoạn, tư thế, phân loại, OCR và CLIP cho RDK X5. X3 dùng nhánh riêng, còn bản giao dòng S hiện hành nằm trong nhánh [`rdk_s`](https://github.com/D-Robotics/rdk_model_zoo/tree/rdk_s) của zoo chính; repo [`rdk_model_zoo_s`](https://github.com/D-Robotics/rdk_model_zoo_s) cũ chứa demo lịch sử. Vì vậy, danh sách X5 hiện đại không chứng minh mọi mô hình được kiểm chứng trên X3, Ultra hoặc S100.

OpenExplorer/Algorithm Toolchain nhập ONNX hoặc Caffe cho PTQ và hỗ trợ quy trình QAT hướng PyTorch. Triển khai tùy nền tảng: RDK X5 hiện hành dùng `.bin` qua `hbm_runtime` trên `libdnn`, còn quy trình `rdk_s` chính hiện hành dùng `.hbm` qua API `hbm_runtime` cùng tên trên `libhbucp`; X3 giữ nhánh cũ và giao diện inference riêng. Tài liệu `rdk_model_zoo_s` cũ cũng mô tả đường dẫn `.bin` và `.hbm` lịch sử, nên phải ghép các artifact đó với đúng nhánh và toolchain tương ứng. Operator không hỗ trợ có thể chuyển sang CPU. Ví dụ công khai hữu ích, nhưng việc ghép phiên bản giữa RDK OS, firmware board, toolchain và binary mô hình vẫn là một phần của hợp đồng tương thích.

### AXERA

Các họ AX650/AX630/AX620 của AXERA xuất hiện trong camera AI nhỏ gọn và board như dòng MaixCAM của Sipeed. Pulsar2 nhập ONNX, calibration và phân tích precision rồi tạo `.axmodel`; AXEngine chạy artifact đó.

[Mẫu AXERA chính thức](https://github.com/AXERA-TECH/ax-samples) bao phủ YOLOv5/6/7/8/9/10/11/13/26, YOLOX và YOLO-World, với hỗ trợ tác vụ và chip thay đổi theo nền tảng. Trên mục tiêu tương thích, ví dụ YOLO26 hiện tại gồm phát hiện, tư thế, phân đoạn và OBB. [Ví dụ chuyển đổi Pulsar2](https://pulsar2-docs.readthedocs.io/en/latest/appendix/model_convert_examples.html) cho thấy phần việc thực tế: tên tensor chính xác, điểm cắt đầu ra, biến đổi layout, archive calibration và thiết lập phần cứng mục tiêu.

### SOPHGO

TPU-MLIR của SOPHGO là một trong những stack compiler hấp dẫn hơn cho tích hợp độc lập vì chính compiler là mã nguồn mở. Nó nhập ONNX, PyTorch, TFLite và Caffe, hạ cấp graph và lượng tử hóa, tạo `.bmodel` cho mục tiêu BM hoặc `.cvimodel` cho phần cứng CV18xx.

[Dự án TPU-MLIR](https://github.com/sophgo/tpu-mlir) hỗ trợ quy trình FP32, BF16, FP16 và INT8 tùy mục tiêu. [Bộ demo SOPHON](https://github.com/sophgo/sophon-demo/blob/release/README_EN.md) nêu YOLOv3/4/5/7/8/9/10/11/12/26, YOLOX, PP-YOLOE, YOLO-World, biến thể OBB/phân đoạn, SSD, CenterNet, RetinaFace, SAM/SAM2 và OCR. Như thường lệ, demo trên BM1684X không kiểm chứng cùng artifact trên BM1688 hoặc CV18xx.

### Huawei Ascend

Dòng inference biên của Huawei có silicon Ascend 310/310P/310B trong sản phẩm Atlas 200I và 300I. CANN cung cấp compiler ATC, runtime AscendCL và kernel operator theo chip; ATC chuyển ONNX cùng biểu diễn nguồn khác thành mô hình `.om` ngoại tuyến.

[Tài liệu Atlas 200I DK A2 hiện hành](https://www.hiascend.com/en/hardware/developer-kit-a2/specification) và mẫu CANN cung cấp quy trình `.om` YOLOv5 cho Ascend 310B. [Ascend ModelZoo](https://github.com/Ascend/modelzoo) rộng hơn và cũ hơn có YOLOv2/3/4/5, YOLOX, YOLOR, SSD, RetinaNet, Mask R-CNN, tư thế và mô hình phân đoạn, nhưng không nên trình bày như thể mọi mục đã được kiểm chứng lại trên 310B. CANN, kernel, firmware và SoC phải khớp nhau. `.om` cho Ascend310P không phải artifact Ascend dùng chung.

### Cambricon

Accelerator MLU của Cambricon dùng Neuware và engine inference MagicMind. Repo [MagicMind Cloud](https://github.com/Cambricon/magicmind_cloud) công khai được ghim ở MagicMind 1.7 và ma trận MLU370-X4/S4; đây là bằng chứng tương thích lịch sử, không phải ma trận SDK hiện hành hay kiểm chứng MLU270. Repo tài liệu hóa đường dẫn PyTorch, ONNX, Caffe và Paddle; hỗ trợ framework TensorFlow đã bị gỡ trong 1.7 dù ví dụ TensorFlow lịch sử vẫn còn. [Developer portal 3-series](https://developer.cambricon.com/index/document/index/classid/3.html) hiện hành của Cambricon liệt kê các bản SDK tổng thể mới hơn, vì vậy không được gộp repo ví dụ công khai và stack thương mại hiện tại thành một phiên bản.

[Repo mô hình MagicMind Cloud](https://github.com/Cambricon/magicmind_cloud) chính thức công bố ma trận MLU370-X4/S4 rất trực tiếp. Repo đánh dấu ví dụ YOLOv3, Tiny-YOLOv3, YOLOv4, YOLOv5, YOLOv7, YOLOv8, PP-YOLOE, SSD, RetinaFace, RetinaNet, Mask R-CNN, DeepLab và UNet, gồm thông tin có ví dụ C++ hay Python. Bằng chứng mạnh; rào cản chính là quyền truy cập SDK và container qua kênh Cambricon.

### Canaan/Kendryte

Chip KPU K230/K230D hiện tại và K210/K510 cũ của Kendryte quan trọng ở phân khúc board giá thấp và camera thông minh. [Compiler nncase](https://github.com/kendryte/nncase) nhập ONNX hoặc TFLite, dùng dữ liệu calibration để chuyển fixed-point, mô phỏng kết quả trên host và tạo `.kmodel`.

[Hướng dẫn phát triển nncase K230](https://github.com/kendryte/k230_docs/blob/main/en/01_software/board/ai/K230_nncase_Development_Guide.md) chính thức hướng dẫn biên dịch, mô phỏng và chạy YOLOv5s. [Danh mục demo AI](https://github.com/kendryte/k230_docs/blob/main/en/02_applications/ai_demos/K230_AI_Demo_Introduction.md) riêng biệt có artifact phát hiện, phân đoạn và tư thế YOLOv8n; [CanMV changelog](https://github.com/kendryte/canmv_k230/blob/canmv_k230/CHANGELOG.md) hiện hành bổ sung ví dụ phân loại, phát hiện, phân đoạn, OBB và tư thế YOLOv8/11/26 đã tối ưu. [Kết quả YOLOv5s](https://github.com/kendryte/k230_docs/blob/main/en/00_hardware/K230_datasheet.md) trong datasheet là benchmark thực sự được công bố; phiên bản compiler và plugin KPU nhị phân phải khớp SDK board.

### Allwinner

V853 của Allwinner kết hợp phần cứng media hướng camera với NPU Vivante 1-TOPS. [Hướng dẫn NPU tiếng Anh chính thức](https://docs.aw-ol.com/v853/en/npu/dev_npu/) mô tả nhập Acuity/Pegasus, quantization, xác minh và triển khai qua `viplite`. Trang mô hình riêng của Allwinner và [hướng dẫn YOLOv5](https://docs.aw-ol.com/v853/npu/npu_yolov5/) nêu YOLOv2/3/4/4-tiny/5/5s, RetinaNet, MobileNet, ResNet và mạng khuôn mặt/người.

Đây là hỗ trợ computer vision thực sự, nhưng vẫn gắn với toolchain Tina Linux/Vivante cũ và việc tải SDK tùy thuộc tài khoản. Nền tảng này thuộc bản đồ thị trường cùng giới hạn đó.

## Nền tảng Hàn Quốc, Đài Loan và Trung Quốc thường bị bỏ sót

Danh sách tiếng Anh thường nhảy từ Rockchip sang NVIDIA và bỏ qua nhóm silicon có tài liệu thực tế thứ hai. Một số hệ sinh thái này có ma trận YOLO hiện hành rộng hơn các startup phương Tây nổi tiếng.

### Mobilint

Nhà cung cấp accelerator Hàn Quốc Mobilint bán REGULUS công suất thấp và ARIES MLA100 thông lượng cao hơn. SDK qb nhận đầu vào hướng PyTorch, TensorFlow, TFLite, ONNX và Keras, lượng tử hóa INT8 rồi tạo artifact `.mxq` đã biên dịch. [Ma trận vision công khai](https://github.com/mobilint/mblt-model-zoo/blob/master/mblt_model_zoo/vision/README.md) nêu phát hiện YOLOv3/v5/v7/v8/v9/v10/11/12/26 cùng biến thể phân đoạn, tư thế và OBB YOLO. [Trang ARIES](https://www.mobilint.com/aries/mla100) còn công bố kết quả YOLO11s và YOLO26m theo mô hình cụ thể. Đây là bằng chứng tích hợp mạnh, dù quyền tiếp cận SDK/phần cứng vẫn mang tính thương mại.

### Sunplus

SP7350/C3V của Sunplus dùng stack NPU có nguồn gốc Vivante nhưng đóng gói khác Allwinner hay Amlogic cũ. Quy trình công khai kết hợp chuyển đổi Acuity, môi trường Docker NPU, runtime SNNF và đầu ra `.nb`. [Hướng dẫn YOLOv8 tùy chỉnh](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2427224156/SP7350) chính thức hướng dẫn chuyển đổi và triển khai thay vì chỉ tuyên bố hỗ trợ framework. IP liên quan không khiến `.nb` của nó có thể dùng trên SoC Vivante khác.

### ESWIN Computing

Dòng RISC-V của ESWIN gồm EIC7700/EIC7700X và EIC7702/EIC7702X hai die, trong đó EIC7700 xuất hiện trên board đang bán như Milk-V Megrez. ENNP gồm EsQuant, compiler EsAAC hiện hành, tạo dữ liệu golden, mô phỏng và runtime ESSDK, với đầu ra `.model` đã biên dịch; tài liệu ENNP cũ dùng tên `ennc-compile`. [Trang EsAAC hiện hành](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/esaac) tài liệu hóa đầu vào ONNX, nên framework khác cần xuất hoặc chuyển sang IR được hỗ trợ. [Tổng quan ENNP](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/introduction) và [hướng dẫn YOLOv3](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/yolov3) do Milk-V lưu trữ xác lập đường dẫn đầu cuối công khai, nhưng không phải ma trận mô hình/độ chính xác hiện hành rộng.

### Nuvoton M55M1

M55M1 của Nuvoton là thiết kế Cortex-M55 cấp MCU cộng Ethos-U55-256, không phải bộ xử lý ứng dụng Linux. NuEdgeWise/NuML và Arm Vela chuẩn bị mô hình TFLite Micro đã lượng tử hóa. Repo [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise) công khai nêu YOLOv8-nano, YOLOX-nano, YOLO Fastest v1.1, SSD-MobileNet FPNLite và nhiều classifier. Các ví dụ mạng nhỏ này là bằng chứng đáng tin cho giới hạn bộ nhớ/công suất của thiết bị, không phải bằng chứng cho biến thể YOLO 640-pixel tiêu chuẩn.

### Rebellions và FuriosaAI

Dòng ATOM của Rebellions dùng SDK RBLN và artifact `.rbln` đã biên dịch. [Bản phát hành hỗ trợ chính thức](https://docs.rbln.ai/v0.8.2/supports/release_note.html) nêu YOLOv3 tiny/full/SPP, YOLOv5 từ n đến x, YOLOv6 từ n đến l, biến thể YOLOv7 và YOLOv8 từ n đến x. [Ma trận hỗ trợ card hiện hành](https://docs.rbln.ai/latest/supports/version_matrix.html) đánh dấu ATOM CA02 và ATOM+ CA12 là EoL, còn ATOM+ CA22 và ATOM-Max CA25 là Active. ATOM-Lite CA21 có trang sản phẩm nhưng vắng trong ma trận SDK đó nên trạng thái toolchain hiện tại chưa rõ. Tài liệu công khai, nhưng wheel compiler cần thông tin đăng nhập Rebellions Portal.

Cần tách FuriosaAI theo thế hệ. Warboy là accelerator computer vision cũ hơn, dùng artifact `.enf` INT8 và [model zoo chính thức](https://developer.furiosa.ai/furiosa-models/latest/) có SSD, YOLOv5M/L và YOLOv7-w6-pose. RNGD dùng [stack `.fxb`](https://developer.furiosa.ai/latest/en/furiosa_llm/fxb.html) khác, chủ yếu hướng đến tải LLM/VLM. [Lộ trình hiện hành](https://developer.furiosa.ai/latest/en/overview/roadmap.html) của Furiosa ghi hỗ trợ vision YOLOv8m đã hoàn tất trong Q4 2024, nhưng bề mặt mô hình được hỗ trợ và hiệu năng công khai hiện tập trung vào LLM/VLM, không cung cấp ma trận chính xác/hiệu năng YOLOv8m chi tiết hiện tại. Đây là bằng chứng thế hệ đã phát hành, không phải khẳng định tương thích Warboy.

### Realtek, Telechips, T-Head và SigmaStar

Bốn nền tảng này có thật nhưng bề mặt BYOM hẹp hơn, cũ hơn hoặc bị giới hạn truy cập:

| Nền tảng | Bằng chứng công khai có thể tái lập | Giới hạn cần giữ nguyên |
|---|---|---|
| Realtek AmebaPro2 | [SDK Arduino/FreeRTOS](https://github.com/Ameba-AIoT/ameba-arduino-pro2) đóng gói mô hình YOLOv3/4/7-tiny, SCRFD và MobileFaceNet `.nb` | Muốn chuyển đổi mô hình tùy chỉnh phải đăng ký quan tâm/liên hệ Realtek |
| Telechips TOPST TCC7500 | [Dự án 2026 chính thức](https://docs.topst.ai/blog/31) đã chuyển đổi và triển khai YOLOv8s | Chuyển đổi UFLD v1/v2 thất bại do layer không hỗ trợ; bài viết chỉ đề xuất cách xử lý bằng tách/hậu xử lý. Tài nguyên compiler/operator đầy đủ thường cần xin quyền qua email |
| T-Head TH1520 | [Hướng dẫn ứng dụng](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) của Sipeed chạy YOLOv5n/s với HHB và CSI-NN2/SHL | Stack công khai nhưng bảo trì và độ rõ phiên bản tụt sau các nền tảng dẫn đầu hiện nay |
| SigmaStar SSU9383CM | Tài liệu hiện hành xác lập API runtime MI_IPU; tài liệu SGS_IPU cũ nêu `.sim` thành `sgsimg.img` và bộ hậu xử lý cũ nhắc YOLOv1/2/3 | Không có bằng chứng công khai rằng artifact compiler cũ hoặc đường dẫn mô hình có tên áp dụng được cho SSU9383CM |

### Vision thông minh HiSilicon không phải Huawei Ascend

SoC camera Hi3516CV610/DV500, Hi3519DV500 và Hi3403V100 của HiSilicon cung cấp quy trình ATC sang `.om` qua runtime NNN/SVP-NNN. [HiSpark model zoo](https://gitee.com/HiSpark/modelzoo/blob/master/README.md) chính thức nêu YOLOv3/4/5/6/7/8/9/10/11, phân đoạn/tư thế YOLO11, OBB/World/phân đoạn YOLOv8, OCR, độ sâu và TinySAM trong ma trận theo mục tiêu, đáng chú ý cho Hi3403 và Hi3591P. Điều này không xác lập mọi mục chạy được trên mọi SoC vision thông minh kể trên. Một số mục là nội dung lộ trình, nên phải tách ví dụ đã phát hành khỏi hỗ trợ dự kiến; repo ghi rằng mô hình ModelZoo được cung cấp chỉ dành cho mục đích phi thương mại. Cùng dùng tên ATC/`.om` không chứng minh tương thích artifact hay runtime với dòng Ascend dựa trên CANN.

## Công ty accelerator mới nổi và chuyên biệt

### MemryX

Module MX3 của MemryX dùng kiến trúc dataflow và có thể kết hợp nhiều chip. [Neural Compiler](https://developer.memryx.com/tools/neural_compiler.html) nhận mô hình ONNX, TensorFlow Lite, Keras và TensorFlow, tạo gói DFP để runtime accelerator sử dụng. [API runtime](https://developer.memryx.com/api/accelerator/accelerator.html) hỗ trợ pipeline nhiều mô hình và streaming.

Tài liệu và ví dụ MemryX có tiền xử lý/hậu xử lý YOLO đã tối ưu cho phát hiện, phân đoạn và tư thế. [Ghi chú phát hành SDK](https://developer.memryx.com/release_notes.html) nêu rõ hỗ trợ YOLOv10, YOLO11 và YOLO26. Tài liệu công khai hữu ích về kỹ thuật, dù thiếu một bảng mô hình/độ chính xác/thiết bị đơn giản như của Axelera.

### Kneron

Sản phẩm KL520/KL530/KL630/KL720/KL730 của Kneron trải từ accelerator USB nhỏ đến accelerator nhúng. Kneron PLUS bao quát runtime ứng dụng, còn Model Toolchain 0.33.1 hiện hành tài liệu hóa chuyển đổi, quantization, đánh giá, mô phỏng và biên dịch `.nef` cho các mục tiêu đó. KL830 xuất hiện trong một số [API runtime PLUS](https://doc.kneron.com/docs/plus_c/introduction/run_examples/), nhưng không có trong [danh sách mục tiêu compiler](https://doc.kneron.com/docs/toolchain/manual_1_overview/) hiện hành; không nên suy rộng tuyên bố biên dịch `.nef` chung sang KL830 nếu chưa được nhà cung cấp xác nhận.

[Ví dụ YOLO chính thức](https://doc.kneron.com/docs/toolchain/appendix/yolo_example/) minh họa quy trình bằng Tiny-YOLOv3; tài liệu Kneron khác bao quát huấn luyện và triển khai YOLOv5. Bằng chứng đáng tin nhưng hẹp hơn ma trận YOLO hiện đại rộng của Hailo, Axelera, Rockchip hoặc DEEPX.

### SiMa.ai

MLSoC và [nền tảng Modalix 50-TOPS đang sản xuất](https://sima.ai/press-release/sima-ai-next-gen-platform-for-physical-ai-in-production/) của SiMa.ai dùng môi trường phần mềm Palette. ModelSDK, compiler MLA và ModelExecutor bao quát nhập, quantization, biên dịch, profiling và thực thi. Công cụ hỗ trợ quy trình hướng INT8, INT16 và BF16 tùy tải và sản phẩm.

Ghi chú phát hành của công ty nêu pipeline YOLOv7, YOLOv8 phát hiện/tư thế/phân đoạn, YOLOX và phân đoạn YOLOX, DETR, Mask R-CNN và EfficientDet đã được kiểm chứng. Không nên che giấu một giới hạn hiện hành: ghi chú [Palette SDK 2.1](https://docs.sima.ai/v2.1.2/pages/release_notes/2.1.html) cho biết QAT không hoạt động do hồi quy quantization Python/PT2E. Cách khắc phục được tài liệu hóa là tạo ONNX có annotation bằng SDK 2.0 rồi biên dịch với 2.1. Quyền truy cập và mua hàng vẫn hướng đến doanh nghiệp.

Ranh giới triển khai cũng được tài liệu hóa: [quy trình biên dịch ModelSDK](https://docs.sima.ai/v2.1.2/pages/model-sdk/compilation.html) tạo `.tar.gz` đã biên dịch và có thể tạo ELF thực thi, còn [MPK Tool](https://docs.sima.ai/v2.1.1/pages/palette/mpk_tools.html) đóng gói `.mpk` có thể triển khai. Các đầu ra này gắn với mục tiêu MLSoC/Modalix và bản Palette.

### EdgeCortix

SAKURA-II là accelerator quảng cáo 60-TOPS cho vision và generative AI, biên dịch qua framework phần mềm MERA. EdgeCortix cũng cung cấp công nghệ MERA cho quy trình RUHMI mới hơn của Renesas.

[Tài liệu sản phẩm SAKURA-II](https://www.edgecortix.com/en/edgecortix-sakura-ii-accelerator-brief) xác lập phần cứng và compiler; [phần cứng M.2/PCIe](https://www.edgecortix.com/en/hardware) được chào theo hình thức hỏi về dùng thử hoặc đặt hàng. Không tìm thấy ma trận tương thích YOLO công khai hiện hành đủ chính xác. Đây là thông tin mua hàng hữu ích: cần yêu cầu kiểm chứng mô hình trước khi cam kết mua phần cứng.

### BrainChip

Akida của BrainChip là bộ xử lý miền sự kiện neuromorphic, không phải NPU tensor dày đặc thông thường. [Môi trường MetaTF](https://brainchip.com/metatf-dev-tools/) gồm các gói Python có thể cài đặt để tạo mô hình, quantization bit thấp, chuyển đổi, mô phỏng và thực thi phần cứng. BrainChip hiện bán [coprocessor AKD1500 M.2 đang giao hàng](https://brainchip.com/brainchip-akd1500-now-available-in-compact-m-2-form-factor-enabling-fanless-edge-ai-in-industrial-and-commercial-designs/), cùng phần cứng AKD1000 cũ và IP Akida 2.

[Model card Akida 2 hiện hành](https://brainchip.com/wp-content/uploads/2025/04/Akida-2-Model-Card-V1.1-Mar.25.pdf) nêu detector YOLOv2 AkidaNet0.5, CenterNet, AkidaUNet và mạng nhận diện khuôn mặt. Tài liệu hỗ trợ khác gồm FOMO và tải theo sự kiện. Akida hỗ trợ trọng số và activation bit thấp khác thường, nhưng triển khai thành công có thể cần chuyển đổi nhận biết kiến trúc thay vì xem đây là mục tiêu ONNX INT8 chung khác. Không nên gán kết quả model zoo AKD1000 hoặc Akida 2 cho AKD1500 nếu chưa có báo cáo map/fit xác thực.

### Blaize

Blaize bán sản phẩm Pathfinder và Xplorer dựa trên P1600; Picasso SDK, NetDeploy và AI Studio cung cấp quy trình phần mềm. [Trang sản phẩm](https://www.blaize.com/products/) xác định rõ thị trường vision và Edge AI, nhưng không có ma trận tương thích chi tiết theo từng mô hình công khai hiện hành.

Vì vậy, hướng dẫn xếp Blaize vào nhóm thương mại và giới hạn truy cập thay vì lấp khoảng trống bằng tuyên bố cộng đồng. Báo cáo biên dịch và độ chính xác do nhà cung cấp cung cấp cho mô hình dự định dùng nên là điều kiện tiên quyết.

## Vision TinyML và thiết bị đầu cuối

### Arm Ethos-U và Alif

Arm cấp phép IP NPU Ethos-U55, U65 và U85 cho các công ty chip. [Compiler Vela](https://developer.arm.com/edge-ai/arm-cortex-m-and-ethos-u) nhận graph LiteRT/TFLite lượng tử hóa và viết lại subgraph được hỗ trợ thành toán tử tùy chỉnh Ethos-U. Toán tử không hỗ trợ có thể còn chạy trên CPU, nghĩa là "ứng dụng chạy được" và "toàn bộ mạng chạy trên NPU" là hai khẳng định khác nhau.

Triển khai thực tế gồm Ethos-U55 trong Alif Ensemble E7, Ethos-U65 trong NXP i.MX 93 và Ethos-U85 trong hệ thống mới hơn. [E7 AI/ML AppKit](https://alifsemi.com/support/kits/ensemble-e7appkit/) của Alif tài liệu hóa hai accelerator ML và minh họa quy trình E7 TFLite sang Vela; [Ensemble E8](https://alifsemi.com/ensemble-e8-series/) kết hợp một Ethos-U85 với hai NPU Ethos-U55. Alif cũng công bố [benchmark detector khuôn mặt YOLO-Fastest INT8](https://alifsemi.com/faster-ai-mcu-inferencing-low-power-consumption/) ở cấp họ sản phẩm, độ phân giải 192 x 192 trên Cortex-M55 cùng Ethos-U55, nhưng không có ma trận YOLO hiện đại rộng theo từng mục tiêu. Các hệ thống giới hạn bộ nhớ này phù hợp mạng phân loại và phát hiện nhỏ, không phải detector YOLO 640 pixel tùy ý chỉ vì có thể biểu diễn bằng TFLite.

### Infineon PSOC Edge, Himax WiseEye2 và Analog Devices MAX7800x

Các họ [PSOC Edge E83](https://documentation.infineon.com/psocedge/docs/xsk1761304020519) và E84 của Infineon ghép Cortex-M55 với Ethos-U55, đồng thời thêm khối NNLite riêng ở phía M33 công suất thấp. [Tài liệu kiến trúc E84](https://documentation.infineon.com/psocedge/docs/bwb1750411526047) mô tả trọng số 8-bit, activation 8 hoặc 16-bit, và quy trình Vela trong đó operator được hỗ trợ thành luồng lệnh NPU còn phần không hỗ trợ ở lại CPU. [DEEPCRAFT Model Converter](https://www.infineon.com/design-resources/embedded-software/deepcraft-edge-ai-solutions/deepcraft-model-converter) nhận đường dẫn TFLite, Keras và PyTorch; tài liệu kiến trúc nêu MobileNetV1/V2 làm kernel đại diện thay vì benchmark mục tiêu. [E84 AI Kit](https://documentation.infineon.com/psocedge/docs/cci1762693051052) có camera và dùng ModusToolbox, nhưng tài liệu công khai chưa cung cấp ma trận tương thích YOLO có tên. Đây là nền tảng vision có thể lập trình thực sự với bằng chứng cấp mô hình đang hoàn thiện, không phải mục tiêu detector đa dụng đã kiểm chứng.

[HX6538 WiseEye2](https://www.himax.com.tw/products/wiseeye-ai-sensing/wiseeye2-ai-processor/) của Himax cũng kết hợp Cortex-M55 và Ethos-U55 cho vision luôn bật. Câu chuyện phần mềm công khai cụ thể khác thường với mức công suất này: [ví dụ Grove Vision AI Module V2 chính thức](https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2) gồm phát hiện đối tượng YOLOv8n, tư thế và phân loại giới tính, phát hiện YOLO11n, face mesh và PeopleNet. Quy trình hợp nhất dùng TFLite Micro và Vela, rồi fallback qua CMSIS-NN và kernel tham chiếu khi cần. Đây là mô hình và độ phân giải rất nhỏ theo chủ đích, không phải bằng chứng graph YOLO thông thường cỡ server có thể vừa bộ nhớ.

Analog Devices chọn hướng khác với [accelerator CNN MAX78000/MAX78002](https://www.analog.com/en/products/max78002.html). Công cụ [`ai8x-synthesis`](https://github.com/analogdevicesinc/ai8x-synthesis) công khai lượng tử hóa mạng đã huấn luyện và tạo mã C, trọng số cùng cấu hình accelerator riêng cho mục tiêu thay vì gói runtime ONNX di động. Bằng chứng bên thứ nhất gồm [nhận diện khuôn mặt](https://www.analog.com/en/resources/app-notes/an-2616.html), phát hiện mã QR TinierSSD trong [ADI model zoo](https://ez.analog.com/dsp/software-and-development-tools/edgebench/a/docs-faqs/DF912/available-models-in-adi-model-zoo), và classifier ảnh. Các thiết bị phù hợp vision đầu cuối, nhưng không tìm thấy ma trận YOLO hiện đại chính thức hiện hành.

### GreenWaves GAP9

GAP9 kết hợp tính toán RISC-V với neural engine NE16. NNTool nhập và lượng tử hóa mạng, còn AutoTiler và GAP SDK tạo mã có thể triển khai. [Danh mục mạng nơ-ron GAP9](https://github.com/GreenWaves-Technologies/nn_menu_gap9) công khai có MobileNet, EfficientNet, ResNet, MobileNet SSD, ứng dụng khuôn mặt và nhận dạng. GreenWaves cũng công bố [dự án phát hiện người YOLOX](https://github.com/GreenWaves-Technologies/yolox_people_detection) với kết quả độ chính xác và mô phỏng.

Đây là bằng chứng TinyML minh bạch khác thường, nhưng SDK đầy đủ chỉ dành cho khách hàng đủ điều kiện và mô hình nhỏ hơn đáng kể so với biến thể YOLO 640 x 640 phổ biến.

### Lattice sensAI

Lattice sensAI là phương án FPGA thay cho NPU cố định. sensAI Studio và Neural Network Compiler ánh xạ graph được hỗ trợ lên IP accelerator cấu hình được cho thiết bị ECP5, iCE40 UltraPlus, CrossLink-NX, CertusPro-NX, Avant-E và Avant-X. Đầu ra gắn với FPGA, chế độ accelerator, layout bộ nhớ và bitstream đã chọn, không phải mô hình runtime di động.

[Hướng dẫn Neural Network Compiler hiện hành](https://www.latticesemi.com/-/media/LatticeSemi/Documents/UserManuals/MQ3/FPGA-UG-02052-8-0-Lattice-Neural-Network-Compiler-Software.ashx?document_id=52343) có bảng topology trực tiếp khác thường. Bảng liệt kê YOLOv1 cho ECP5, YOLOv5 và YOLOv8 ở chế độ optimized/advanced cho CrossLink-NX và CertusPro-NX, và chỉ YOLO11 với IP Advanced mode. Bảng còn nêu SSD, MobileNetV2-SSD, ResNet, ENet, SqueezeDet cùng mạng vision khác, hiển thị rõ các ô không hỗ trợ theo họ FPGA. [Advanced CNN Accelerator](https://www.latticesemi.com/en/Products/DesignSoftwareAndIP/IntellectualProperty/IPCore/IPCores05/Advanced-CNN-Accelerator-IP) nhắm đến Avant-E/Avant-X/CertusPro-NX và là IP thương mại. Đây là bằng chứng từ bảng compiler, không phải cam kết một bitstream hỗ trợ đồng thời mọi topology được liệt kê.

### Microchip VectorBlox

[VectorBlox SDK 3.1](https://github.com/Microchip-Vectorblox/VectorBlox-SDK) công khai tiền xử lý và biên dịch mạng TFLite INT8 lượng tử hóa hoàn toàn cho IP CoreVectorBlox cấu hình được trong FPGA PolarFire SoC. `tflite_preprocess` và `vnnx_compile` tạo asset triển khai `.vnnx`, `.hex`, `.ucomp`; repo có simulator và driver C phía mục tiêu. Graph TensorFlow, ONNX và OpenVINO phía thượng nguồn vẫn phải đi qua quy trình chuẩn bị TFLite/INT8 được tài liệu hóa.

[Ma trận hướng dẫn hiện hành](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/tutorials/README.md) gồm YOLOv5n, YOLOv8n phát hiện, phân loại, OBB, tư thế, phân đoạn, YOLOv9t, biến thể YOLO thưa/nén, MobileNet, EfficientNet-Lite0, ResNet18, MiDaS, FFNet và QuickSRNet. [Hướng dẫn hậu xử lý C](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/docs/C_Postprocessing.md) riêng biệt nêu YOLOv2/3/4/5 và decoder phát hiện, tư thế, OBB Ultralytics. SDK 3.1 hiện hỗ trợ PolarFire SoC Video Kit và ghi rõ không hỗ trợ PolarFire Video Kit không có SoC, vì vậy không được nhầm demo board cũ với hợp đồng mục tiêu hiện tại.

SDK có thể tải công khai theo [giấy phép riêng của Microchip](https://github.com/Microchip-Vectorblox/VectorBlox-SDK/blob/master/LICENSE.md), giới hạn phần mềm và sản phẩm phái sinh vào sản phẩm Microchip. Microchip cung cấp miễn phí giấy phép Libero Silver và CoreVectorBlox nhưng cần gửi yêu cầu. Triển khai vẫn là hợp đồng hệ thống FPGA: cấu hình accelerator, bitstream, firmware, dữ liệu mạng do SDK tạo và layout bộ nhớ phải khớp nhau. Biên dịch VectorBlox thành công không biến mô hình thành binary có thể chép vào thiết kế PolarFire bất kỳ.

### Syntiant

NDP200 của Syntiant nhắm đến inference vision, âm thanh và cảm biến luôn bật dưới mức công suất của SoC Linux thông thường. [Bảng phần cứng hiện hành](https://www.syntiant.com/hardware) ghi NDP200 đang sản xuất hàng loạt và NDP250 đang lấy mẫu. [Tờ giới thiệu NDP200](https://www.syntiant.com/ndp200/) tài liệu hóa hỗ trợ mạng CNN, RNN và fully connected dưới 1 mW; [thông báo NDP250](https://www.syntiant.com/news/syntiant-unveils-ndp250-neural-decision-processor-with-next-gen-core-3-architecture) mô tả riêng nhận dạng ảnh luôn bật dưới 30 mW. Không nên khái quát "dưới một milliwatt" cho cả hai linh kiện.

Tài liệu công khai không xác lập khả năng tương thích YOLO rộng, vì vậy nên đánh giá sản phẩm cho classifier và detector nhỏ luôn bật thông qua đánh giá mô hình được nhà cung cấp hỗ trợ, thay vì giả định xuất ONNX chung là đủ.

### Google Coral: hai thế hệ không liên quan

Sản phẩm Edge TPU cũ dùng Edge TPU Compiler, `libedgetpu` và PyCoral cùng mô hình TFLite INT8 hoàn toàn. Repo [Edge TPU](https://github.com/google-coral/edgetpu) chính và [PyCoral](https://github.com/google-coral/pycoral) đã được lưu trữ. Đây là rủi ro bảo trì thực sự, nhưng không phải thông báo EOL phần cứng chính thức.

Google cũng duy trì [Coral NPU](https://github.com/google-coral/coralnpu) mã nguồn mở riêng biệt và đang hoạt động: IP accelerator RISC-V để tích hợp vào SoC công suất thấp. Dự án công khai hiện có RTL, mô phỏng phần cứng, toolchain và ví dụ ELF. Đây không phải sản phẩm kế nhiệm Edge TPU USB/M.2 cắm-thay-thế và chưa công bố ma trận triển khai YOLO hiện hành.

## Bẫy khi so sánh GPU, NPU client và tính toán thích ứng

NVIDIA, AMD, Intel và Apple thường xuất hiện trong tìm kiếm NPU, nhưng không cung cấp cùng một loại accelerator có thể thay thế lẫn nhau.

**NVIDIA Jetson:** Jetson Orin kết hợp GPU CUDA với lõi DLA chuyên dụng. TensorRT có thể tạo engine tuần tự hóa cho DLA, nhưng layer không được hỗ trợ chỉ chạy trên GPU nếu bật rõ GPU fallback. [Bảng DeepStream YOLO hiện hành](https://github.com/NVIDIA-AI-IOT/deepstream_tools/blob/main/yolo_deepstream/README.md) bao quát YOLOv4/v7/v8/v9/11, nhưng phần lớn là mục tiêu GPU; mục ONNX DLA tường minh là YOLOv8s INT8. [Giới hạn DLA](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-layer-restrictions.html) của NVIDIA giải thích giới hạn operator. Thor là một khác biệt nữa: [hướng dẫn di trú DriveOS](https://developer.nvidia.com/docs/drive/drive-os/7.0.3/public/NVIDIA_DriveOS_7.0.3_Migration_Guide.pdf) của NVIDIA nói các lõi DLA đã bị loại khỏi Thor để tăng linh hoạt lập lịch GPU. "Chạy trên Jetson" không cho biết accelerator nào được dùng.

**AMD Vitis AI:** Thế hệ Kria/DPU cũ biên dịch graph `.xmodel`, chạy qua VART. [Vitis AI 6.2](https://vitisai.docs.amd.com/en/6.2/) hiện hành là GA cho hai đường Versal AI Edge riêng: [Gen1](https://vitisai.docs.amd.com/projects/gen1/en/latest/index.html) trên VEK280/VE2802 dùng ONNX, AMD Quark và quy trình snapshot/subgraph gắn với mục tiêu; [Gen2](https://vitisai.docs.amd.com/projects/gen2/en/latest/index.html) trên VEK385 có hợp đồng compiler/runtime riêng. Gen1 công bố ví dụ YOLOv5, YOLOv7, YOLOv8 và YOLOX. Ryzen AI là stack NPU client thứ tư, riêng biệt. Không chuyển artifact hay kết quả kiểm chứng giữa DPU cũ, Versal Gen1, Versal Gen2 và Ryzen AI chỉ vì tất cả dùng tên Vitis hoặc AMD.

**Intel OpenVINO:** OpenVINO có thể nhắm CPU, GPU và NPU Core Ultra qua cùng một API. [Tài liệu Intel NPU plugin](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md) xác định các thế hệ NPU được hỗ trợ, còn [ma trận mô hình đã kiểm chứng](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html) có cột riêng theo thiết bị. Notebook YOLO chứng minh recipe ứng dụng, không chứng minh kiểm chứng NPU trừ khi ma trận xác nhận. Operator, hình dạng, precision được hỗ trợ và driver NPU đã cài vẫn quyết định graph hoàn chỉnh có chạy trên đó hay không. OpenVINO IR (`.xml` cùng `.bin`) là IR cấp nguồn có thể tái sử dụng; cache mô hình đã biên dịch phụ thuộc thiết bị và phần mềm.

**Apple Core ML:** `coremltools` chuyển mô hình thành `.mlpackage`, Xcode biên dịch thành `.mlmodelc`. Core ML có thể lập lịch công việc trên CPU, GPU và Apple Neural Engine, nhưng cố ý trừu tượng hóa phân vùng chính xác. Apple vì vậy là nền tảng triển khai xuất sắc và khó phù hợp với tuyên bố như "toàn bộ graph YOLO chạy trên NPU" nếu profiling không xác nhận.

## Các công ty IP NPU đứng sau công ty chip

Một số công ty không bán board hay accelerator đóng gói. Họ cấp phép thiết kế NPU cho nhà cung cấp SoC. Họ đáng quan tâm vì cùng kiến trúc nền có thể xuất hiện dưới nhiều thương hiệu chip, nhưng SDK và artifact cuối vẫn có thể được bên nhận quyền tùy chỉnh.

| Nhà cung cấp IP | Họ NPU và phần mềm | Lý do đáng quan tâm |
|---|---|---|
| [Arm](https://developer.arm.com/edge-ai/arm-cortex-m-and-ethos-u) | Ethos-U55/U65/U85 và Vela | Xuất hiện trong sản phẩm nhúng NXP, Alif, Infineon, Himax và hãng khác; quy trình hướng TFLite/TOSA lượng tử hóa |
| [Arm China](https://www.armchina.com/mountain?infoId=161&name=) | AIPU Zhouyi và SDK Compass | [Model zoo](https://github.com/Arm-China/Model_zoo) công khai nêu YOLOv1-tiny/v2/v3/v4/v5, YOLOX và YOLOv8-seg; đây là mô hình tham chiếu, không chứng minh mọi chip bên nhận quyền |
| [VeriSilicon](https://www.verisilicon.com/en/IPPortfolio/VivanteVIP9000Pico) | Họ Vivante VIP và Acuity SDK | Họ NPU Vivante liên quan xuất hiện trong nhiều SoC, gồm đường A311D và i.MX 8M Plus có nguồn gốc riêng; mỗi hãng chip vẫn cung cấp tích hợp của mình |
| [Imagination Technologies](https://www.imaginationtech.com/products/open-access/) | PowerVR Series3NX AX3146/AX3386/AX3596; IMG Series4 lịch sử; Neural Compute SDK/IMG DNN | IP NNA có thể cấp phép, không phải board bán lẻ; Open Access cung cấp cấu hình Series3NX 1/5/10-TOPS, còn chương trình NC-SDK chính thức nêu PP-YOLOE, EfficientNet và HRNet chứ không phải mọi cấu hình IP |
| [CEVA](https://www.ceva-ip.com/product/ceva-neupro-studio/) | NeuPro-Nano/NeuPro-M và NeuPro Studio | IP NPU có thể cấp phép với công cụ nhập, quantization, nén, biên dịch graph, mô phỏng và tạo C/C++ |
| [MIPS](https://mips.com/processor-solutions/arc-npx-family/) | ARC NPX6 và [MetaWare MX](https://mips.com/processor-solutions/arc-metaware-mx/), trước đây thuộc Synopsys ARC | IP NPU mở rộng và NN SDK được GlobalFoundries mua lại, đưa vào danh mục MIPS trong [tháng 6 năm 2026](https://mips.com/press-releases/gfmipsarcclose/); bản thân không phải mục tiêu triển khai bán lẻ |
| [Cadence](https://www.cadence.com/en_US/home/tools/silicon-solutions/ai-ip-platform/neuroweave-sdk.html) | Neo NPU, DSP Tensilica và NeuroWeave SDK | Stack compiler/interpreter được bên nhận quyền SoC dùng, hỗ trợ mạng và quantization phụ thuộc cấu hình được cấp phép |
| [Quadric](https://quadric.ai/npu-ip) | IP và SDK GPNPU Chimera | IP kiểu NPU/DSP có thể lập trình, cấp phép cho chip công ty khác; mục tiêu cuối là silicon của bên nhận quyền |
| [Expedera](https://www.expedera.com/products-overview/) | IP NPU Origin và stack phần mềm | Kiến trúc NPU biên có thể cấp phép, không phải board LibreYOLO có thể mua trực tiếp |

Imagination cho thấy vì sao nhãn quyền truy cập và vòng đời cần đặt cạnh tên kiến trúc. Chương trình [Open Access](https://www.imaginationtech.com/products/open-access/) cung cấp ba cấu hình Series3NX đã kiểm chứng silicon mà không cần phí giấy phép IP trả trước, nhưng chỉ sau khi công ty được đánh giá và có phí hỗ trợ/bảo trì cùng tiền bản quyền production. [Neural Compute SDK](https://developer.imaginationtech.com/solutions/neural-compute-sdk/) cung cấp công cụ biên dịch, tối ưu, quantization và runtime; [hợp tác chính thức với Baidu PaddlePaddle](https://www.imaginationtech.com/news/imagination-and-baidu-paddlepaddle-create-open-source-machine-learning-library-for-model-zoo/) nêu phát hiện PP-YOLOE, phân loại EfficientNet và phân đoạn HRNet. Ví dụ này không kiểm chứng mọi cấu hình Series3NX hay Series4; các trang [sản phẩm Series4 riêng lẻ](https://www.imaginationtech.com/product/img-4nx-mc1/) hiện ghi không khả dụng, nên xem Series4 là lịch sử trừ khi bộ phận bán hàng xác nhận khác.

Lớp này giải thích sự tương đồng bề ngoài giữa các họ sản phẩm. Nó không tạo khả năng chuyển artifact. Hai SoC có IP Vivante hoặc Ethos liên quan vẫn có thể khác sơ đồ bộ nhớ, driver, phiên bản compiler, tập operator và runtime của nhà cung cấp.

## Thực sự triển khai gì: bảng khóa chặt theo artifact

Tệp cuối trong pipeline compiler là nơi khả năng di động ONNX bề ngoài kết thúc. Tên và phần mở rộng thay đổi theo thời gian, nhưng quy tắc thực tế vẫn vậy: giữ mô hình gốc và dữ liệu calibration vì artifact gốc thường không thể chuyển sang thế hệ NPU khác hoặc xây dựng lại đáng tin cậy bằng bản SDK khác.

| Stack nhà cung cấp | Đầu vào compiler thường gặp | Thành phần được đưa lên thiết bị | Thành phần bị ràng buộc |
|---|---|---|---|
| [Amlogic AMLNN](https://github.com/Amlogic-NN/amlnn-toolkit) | ONNX, TFLite, TorchScript hoặc PT2 | `.adla`; A311D cũ dùng `.nb` | ID mục tiêu, thế hệ ADLA, bản build AMLNN compiler, NNSDK2/runtime, driver ADLA và BSP |
| [Rockchip RKNN](https://github.com/airockchip/rknn-toolkit2) | ONNX, TFLite hoặc đầu ra framework | `.rknn` | Phiên bản RKNN toolkit/runtime và họ NPU Rockchip |
| [Hailo](https://hailo.ai/developer-zone/documentation/) | ONNX hoặc TensorFlow qua HAR trung gian | `.hef` | Kiến trúc Hailo và thế hệ compiler/runtime |
| [Axelera Voyager](https://docs.axelera.ai/sdk/reference/pipeline-builder/model-compilation/) | ONNX hoặc định nghĩa zoo được hỗ trợ | `.axm` trong Pipeline Builder alpha, gói pipeline `.axe`; `.axmodel` cùng manifest kiểu cũ | Thế hệ API Voyager và cấu hình mục tiêu Metis |
| [DEEPX DXNN](https://github.com/DEEPX-AI/dx-all-suite) | ONNX và đường framework được hỗ trợ | `.dxnn` | Phiên bản DX-COM/DX-RT và mục tiêu DX-M |
| [Qualcomm QAIRT/QNN hoặc SNPE](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk) | ONNX, TFLite hoặc mô hình framework | Thư viện mô hình/binary ngữ cảnh QNN, hoặc SNPE `.dlc` | Kiến trúc HTP, SoC, backend và phiên bản runtime |
| [MediaTek NeuroPilot](https://mediatek.gitlab.io/genio/doc/iot-aihub/master/ai_hub/related_resource.html) | TFLite đã chuyển đổi/lượng tử hóa | `.dla` cho Neuron Runtime ngoại tuyến; `.tflite` cho chế độ delegation | Thế hệ NP/MDLA, image hệ điều hành, compiler và runtime |
| [TI TIDL](https://github.com/TexasInstruments/edgeai-tidl-tools) | ONNX hoặc TFLite | Thư mục imported-artifacts cùng tệp mô hình ứng dụng | TIDL tools/runtime và thế hệ processor |
| [NXP eIQ](https://www.nxp.com/design/design-center/software/development-software/eiq-ml-development-environment:EIQ) | Thường là TFLite hoặc ONNX | Đầu ra Vela, TIM-VX, Neutron hoặc Ara theo backend | Accelerator i.MX/Ara được chọn và BSP, không phải "NXP" nói chung |
| [ST Edge AI Core](https://www.st.com/en/development-tools/stedgeai-core.html) | TFLite, ONNX hoặc mô hình framework được hỗ trợ | Thư viện/mã mạng được tạo cùng asset firmware | MCU/NPU mục tiêu, cấu hình bộ nhớ và phiên bản công cụ |
| [Infineon PSOC Edge](https://documentation.infineon.com/psocedge/docs/bwb1750411526047) | Đường TFLite/Keras/PyTorch lượng tử hóa số nguyên | TFLite tối ưu Vela với luồng lệnh Ethos-U cùng firmware ứng dụng | Biến thể E83/E84, cấu hình Ethos-U, Vela, ModusToolbox và BSP; NNLite là mục tiêu riêng |
| [Renesas DRP-AI](https://github.com/renesas-rz/rzv_drp-ai_tvm) | ONNX/TFLite qua TVM hoặc Translator | Thư mục runtime-model chứa nhiều tệp dữ liệu nhị phân | Thiết bị RZ/V và thế hệ translator/runtime |
| [Sony IMX500](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) | Mạng đã lượng tử hóa/đóng gói từ quy trình Edge-MDT | Gói `.rpk` | Firmware cảm biến, ngân sách bộ nhớ và phiên bản packer |
| [Synaptics](https://developer.synaptics.com/) | TFLite/ONNX và đầu vào được hỗ trợ | `.synap` trên SyNAP; `.vmfb` trên Torq | Thế hệ phần mềm/phần cứng SL16xx hay SL261x |
| [Ambarella CVflow](https://www.ambarella.com/developer/) | Đầu vào toolchain dành cho đối tác | Gói triển khai chỉ dành cho đối tác; hợp đồng artifact chính xác không được tài liệu hóa công khai | Xác nhận thế hệ CVflow, SDK/BSP và pipeline camera qua Cooper Developer Zone |
| [D-Robotics BPU](https://github.com/D-Robotics/rdk_model_zoo) | ONNX và graph được toolchain hỗ trợ | X5 `.bin`; `rdk_s` hiện hành `.hbm`; X3 dùng quy trình nền tảng cũ | Thế hệ BPU, nhánh repo và bản OpenExplorer/runtime tương ứng |
| [AXERA Pulsar2](https://github.com/AXERA-TECH/ax-samples) | ONNX | `.axmodel` | Mục tiêu chip AX, phiên bản Pulsar2 và AXEngine |
| [SOPHGO TPU-MLIR](https://github.com/sophgo/tpu-mlir) | ONNX, TFLite, TorchScript và định dạng khác | `.bmodel` trên BM, `.cvimodel` trên CV18xx | Mục tiêu chip BM/CV và thế hệ runtime SOPHON |
| [Huawei Ascend CANN](https://www.hiascend.com/en/software/cann) | ONNX hoặc graph framework được hỗ trợ | `.om` | Chip Ascend, CANN/ATC và firmware/driver thiết bị |
| [Cambricon MagicMind](https://github.com/Cambricon/magicmind_cloud) | Graph framework hoặc mô hình trao đổi | Engine/mô hình MagicMind đã tuần tự hóa | Kiến trúc MLU và phiên bản Neuware/MagicMind |
| [Canaan nncase](https://github.com/kendryte/nncase) | ONNX hoặc TFLite | `.kmodel` | Thế hệ KPU và plugin mục tiêu nncase tương ứng |
| [Mobilint qb](https://www.mobilint.com/sdk-qb) | ONNX và mô hình framework được hỗ trợ | `.mxq` | Mục tiêu REGULUS/ARIES và compiler/runtime qb |
| [Rebellions RBLN](https://docs.rbln.ai/latest/index.html) | PyTorch 2 hoặc graph được TensorFlow hỗ trợ | `.rbln` | Thế hệ ATOM và compiler/runtime RBLN |
| [FuriosaAI](https://developer.furiosa.ai/docs/latest/en/) | ONNX/TFLite và đường framework được hỗ trợ | Warboy `.enf`; RNGD `.fxb` | Accelerator và thế hệ SDK hoàn toàn khác nhau |
| [Sunplus SNNF](https://sunplus.atlassian.net/wiki/spaces/C3/pages/2004353133) | Mạng Acuity hỗ trợ | `.nb` | Driver NPU SP7350, runtime và BSP |
| [ESWIN ENNP](https://milkv.io/docs/megrez/development-guide/ENNP-SDK/esaac) | ONNX trong quy trình EsAAC hiện hành được tài liệu hóa; framework khác cần xuất/chuyển đổi | `.model` | Mục tiêu chính xác thuộc họ EIC7700 và bản ENNP/ESSDK |
| [Nuvoton / Ethos-U](https://github.com/OpenNuvoton/NuEdgeWise) | TFLite lượng tử hóa | TFLite tối ưu Vela với toán tử tùy chỉnh Ethos-U | Kế hoạch bộ nhớ M55M1, Vela và firmware |
| [Himax WiseEye2](https://github.com/HimaxWiseEyePlus/YOLOv8_on_WE2) | TFLite lượng tử hóa số nguyên hoàn toàn | `.tflite` tối ưu Vela trong flash mô hình cùng firmware board như `output.img` | Cấu hình HX6538/WE2, Vela, TFLite Micro, driver Ethos-U và kernel dự phòng |
| [Analog Devices AI8X](https://github.com/analogdevicesinc/ai8x-synthesis) | Checkpoint PyTorch, YAML mạng và mẫu đầu vào | Mã C/header theo thiết bị, trọng số và firmware biên dịch được tạo | Giới hạn bộ nhớ/layer MAX78000 so với MAX78002, công cụ synthesis và bản MSDK |
| [Realtek AmebaPro2](https://github.com/Ameba-AIoT/ameba-arduino-pro2) | Dịch vụ chuyển đổi/đầu vào của nhà cung cấp | `.nb` | Firmware RTL8735B và bản VoE/NeuralNetwork API |
| [Telechips Enlight](https://docs.topst.ai/product/p/ai) | Đường Darknet, TensorFlow, ONNX hoặc PyTorch | Trung gian `.enlight` cùng gói triển khai đã biên dịch | NPU TCC7500 và các bản TC-NN/Enlight |
| [T-Head HHB](https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/8_application.html) | ONNX và graph framework được hỗ trợ | `hhb.bm`, tham số/mã được tạo và executable | Stack CSI-NN2/SHL TH1520 |
| [SigmaStar MI_IPU](https://wx.comake.online/doc/doc/SigmaStarDocs-SSU9383CM-SIGMASTAR-202507071022/platform/MI/ipu_zh.html) | Tài liệu SSU9383CM hiện hành nêu runtime MI_IPU; tài liệu cũ dùng đầu vào họ TensorFlow/Caffe | `.sim` cũ chuyển thành `sgsimg.img`; artifact công khai hiện hành chưa xác minh | IPU SigmaStar và thế hệ SDK chính xác; chưa xác minh tương thích compiler cũ với SSU9383CM |
| [HiSilicon smart vision](https://gitee.com/HiSpark/modelzoo/blob/master/README.md) | ONNX và graph ATC hỗ trợ | `.om` | SoC vision thông minh chính xác và stack NNN/SVP-NNN; không có tính di động Ascend chung |
| [MemryX](https://developer.memryx.com/tools/neural_compiler.html) | ONNX, TFLite, Keras hoặc TensorFlow | Gói dataflow `.dfp` | Cấu hình phần cứng MX và bản runtime/compiler |
| [Kneron](https://doc.kneron.com/docs/) | ONNX cùng cấu hình toolchain | `.nef`; KL730 NEFv2 có thể chứa `.kne` | Mục tiêu compiler được tài liệu hóa, thế hệ chip, firmware và runtime PLUS; chưa xác lập biên dịch KL830 bằng Toolchain 0.33.1 |
| [SiMa.ai Palette](https://docs.sima.ai/v2.1.2/pages/model-sdk/compilation.html) | Mô hình framework hoặc ONNX được hỗ trợ | `.tar.gz` đã biên dịch ModelSDK, ELF thực thi được tài liệu hóa hoặc gói `.mpk` của MPK Tool | Mục tiêu MLSoC/Modalix và bản Palette |
| [NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/) | ONNX hoặc network definition | Engine đã tuần tự hóa, thường là `.engine` hoặc `.plan` | Kiến trúc GPU/DLA, TensorRT, CUDA và thường cả thiết bị |
| [AMD Vitis AI](https://vitisai.docs.amd.com/en/6.2/) | Graph ONNX/framework lượng tử hóa | `.xmodel` cũ; snapshot/subgraph NPU Gen1; thư mục cache biên dịch hoặc gói `.rai` production Gen2 | Thế hệ NPU/cấu hình IP chính xác, image nền tảng và ma trận Vitis/Vivado/PetaLinux |
| [Microchip VectorBlox](https://github.com/Microchip-Vectorblox/VectorBlox-SDK) | TFLite INT8 lượng tử hóa hoàn toàn | `.vnnx`, `.hex`, `.ucomp` cùng firmware/thiết kế FPGA tương ứng | Cấu hình CoreVectorBlox, PolarFire SoC Video Kit, SDK 3.1, bitstream và layout bộ nhớ |
| [Intel OpenVINO](https://docs.openvino.ai/) | Mô hình framework hoặc ONNX | IR `.xml` cùng `.bin`, hoặc cache biên dịch riêng cho thiết bị | IR có thể tái sử dụng; cache biên dịch phụ thuộc thiết bị, driver và OpenVINO |
| [Google Edge TPU cũ](https://coral.ai/docs/edgetpu/models-intro/) | TFLite INT8 hoàn toàn | `_edgetpu.tflite` đã biên dịch cho Edge TPU | Compiler/runtime Edge TPU và operator lượng tử hóa được hỗ trợ; không liên quan IP Coral NPU mới |
| [Google Coral NPU IP](https://github.com/google-coral/coralnpu) | Ví dụ C/C++ cho dự án IP công khai hiện tại | Ví dụ ELF cho RTL/mô phỏng phần cứng và tích hợp SoC tương lai; không có artifact compiler YOLO công khai | Triển khai SoC đã tích hợp/cấp phép cụ thể và toolchain mã nguồn mở đang phát triển |
| [Lattice sensAI](https://www.latticesemi.com/en/Solutions/Solutions/SolutionsDetails02/sensAI?ActiveTab=Design+File) | Mô tả mạng được hỗ trợ | Bitstream FPGA, cấu hình accelerator và trọng số | Họ FPGA, chế độ IP sensAI và cách triển khai bộ nhớ |
| [Imagination NC-SDK](https://developer.imaginationtech.com/solutions/neural-compute-sdk/) | Caffe, TensorFlow hoặc ONNX | Tên tệp artifact biên dịch phía khách hàng không được công bố công khai | Cấu hình NNA được cấp phép, tích hợp SoC, NC-SDK/DDK và runtime của bên nhận quyền |

Bảng này không chỉ là bảng tra phần mở rộng. Phiên bản compiler, ID mục tiêu, driver, firmware và BSP thuộc danh tính có thể tái lập của mô hình. Exporter LibreYOLO tương lai nên ghi chúng vào manifest máy đọc được bên cạnh từng artifact.

## Tín hiệu quyền truy cập công cụ và vòng đời

Khả năng mua phần cứng và quyền truy cập compiler là hai việc độc lập. Board có thể dễ mua trong khi compiler hiện tại yêu cầu thỏa thuận khách hàng; SDK công khai có thể nhắm đến silicon khó tìm mua. Những mục dưới đây là tín hiệu cụ thể từ nguồn bên thứ nhất, không phải bảng xếp hạng tuổi thọ phổ quát.

| Nền tảng | Tín hiệu được tài liệu hóa | Ý nghĩa thực tế |
|---|---|---|
| Amlogic ADLA | Repo Apache-2.0 công khai và wheel có thể tải; driver/BSP board tương thích vẫn có thể cần Amlogic hoặc nhà cung cấp board | Dễ đánh giá compiler, nhưng cần xác nhận quyền phân phối binary và ma trận tương thích đầy đủ |
| MediaTek Genio | IoT AI Hub và hướng dẫn Yocto công khai; công cụ all-in-one NP8 và tài liệu Android được ghi dành cho khách hàng trực tiếp/NDA | Có thể kiểm toán bằng chứng mô hình, nhưng tự động hóa BYOM có thể cần quan hệ đối tác |
| Hailo | Model zoo và runtime công khai; Dataflow Compiler phân phối qua Developer Zone; `hailo-camera-apps` công khai đã lưu trữ ngày 3 tháng 5 năm 2026 | Khám phá mô hình rộng và mở, nhưng biên dịch có thể tái lập cần đúng tài khoản/phiên bản; trạng thái lưu trữ không xác lập Hailo-15 EOL và cần xác nhận gói ứng dụng vision hiện hành của nó |
| Axelera AI | Tài liệu công khai và [Voyager SDK công khai](https://github.com/axelera-ai-hub/voyager-sdk); hỗ trợ khách hàng cần tài khoản | SDK accelerator tự phục vụ hiếm có, trong khi hỗ trợ sản phẩm và mua hàng vẫn mang tính thương mại |
| Qualcomm Dragonwing | [Product Longevity Program](https://www.qualcomm.com/internet-of-things/products/product-longevity-program) ghi IQ-9075 Sampling với thời gian đến năm 2038 và QCS6490 đến tháng 7 năm 2036; trang IQ-9075 ghi Active | Tín hiệu tốt cho hoạch định công nghiệp, nhưng mâu thuẫn trạng thái từ nguồn bên thứ nhất cần xác nhận theo SKU và không đảm bảo một AI SDK giữ ổn định ABI suốt thời hạn đó |
| Rebellions | [Ma trận hỗ trợ hiện hành](https://docs.rbln.ai/latest/supports/version_matrix.html) ghi CA02 và CA12 EoL, CA22/CA25 Active; không có CA21 | Mua hàng và tương thích SDK tùy theo card, kể cả trong họ ATOM |
| NVIDIA Jetson | [Bảng vòng đời module chính thức](https://developer.nvidia.com/embedded/lifecycle) liệt kê module Orin đến tháng 1 năm 2032 và AGX Orin Industrial đến tháng 7 năm 2033; dev kit không có cam kết vòng đời | Thiết kế hệ thống production theo module, không theo tình trạng sẵn có của bộ phát triển |
| Sony IMX500 trên Raspberry Pi | [Tờ giới thiệu sản phẩm AI Camera](https://datasheets.raspberrypi.com/camera/ai-camera-product-brief.pdf) nói sản xuất ít nhất đến tháng 1 năm 2028 | Mốc tối thiểu cụ thể cho module camera đó, không phải mọi sản phẩm IMX500 |
| Amlogic A311Y3 | [Trang ra mắt 2026](https://www.amlogic.com/News/index248.html) quảng cáo bảo đảm tuổi thọ sản phẩm tối thiểu mười năm | Tín hiệu tốt cho nền tảng mới nhưng cần chi tiết hợp đồng và SKU cho chương trình sản phẩm |
| Google Coral | Repo Edge TPU và PyCoral cũ đã lưu trữ/chỉ đọc; repo [Coral NPU IP](https://github.com/google-coral/coralnpu) riêng biệt đang hoạt động | Rủi ro bảo trì phần mềm cũ, tự nó không phải thông báo EOL phần cứng chính thức; không nói lên vòng đời dự án IP mã nguồn mở riêng |

## Vì sao TOPS không phải hướng dẫn mua hàng

TOPS đỉnh có thể hữu ích trong cùng một nhà cung cấp và kiến trúc, nhưng so sánh giữa các nhà cung cấp thường không hợp lệ trừ khi tất cả điều kiện sau giống nhau:

- Precision số học, gồm INT8, INT4, FP16 hoặc chế độ hỗn hợp
- Một phép nhân-tích lũy được tính là một hay hai phép toán
- Tính toán dày đặc hay thưa có cấu trúc
- Độ phân giải đầu vào, kích thước batch và graph mô hình
- Độ chính xác sau quantization
- Độ trễ chỉ trên NPU hay toàn pipeline ứng dụng
- Chuyển bộ nhớ và chuyển tiếp sang host
- Trạng thái nhiệt và hoạt động duy trì, không chỉ chạy bùng phát

Detector là một pipeline: thu ảnh, resize/letterbox, chuẩn hóa, chuyển thiết bị, inference mạng nơ-ron, giải mã, NMS, đổi tỷ lệ tọa độ và logic ứng dụng. Nhà cung cấp thường chỉ công bố phần inference mạng ở giữa. [MLPerf Inference Edge](https://mlcommons.org/benchmarks/inference-edge/) hữu ích vì định nghĩa kịch bản, mục tiêu chất lượng và quy tắc đo toàn hệ thống thay vì so sánh phép tính tiếp thị riêng lẻ.

Với triển khai YOLO, hồ sơ benchmark tối thiểu đáng tin cậy gồm:

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

Thiếu những trường này, hai con số FPS thường đo hai thứ khác nhau.

## Cách chọn NPU cho computer vision

Hãy chọn theo thứ tự này, không theo con số lớn nhất in trên trang sản phẩm.

1. **Chứng minh graph tương thích.** Biên dịch đúng mô hình đã xuất, không phải checkpoint zoo tên gần giống.
2. **Đo độ chính xác.** Kiểm chứng artifact đã biên dịch trên dataset tác vụ thực sau calibration và quantization.
3. **Đo đầu cuối.** Tính cả chuyển đổi đầu vào, truyền dữ liệu, giải mã và NMS.
4. **Kiểm tra quyền truy cập phần mềm.** Xác định compiler công khai, chỉ dành cho người đăng ký hay chỉ có sau thỏa thuận thương mại.
5. **Cố định ma trận phiên bản.** Ghi compiler, runtime, driver, firmware và ID mục tiêu.
6. **Kiểm tra thực tế hệ điều hành.** Hỗ trợ Android, Debian, Yocto, Buildroot, RTOS và Windows không thể thay thế lẫn nhau.
7. **Kiểm tra nguồn cung và vòng đời.** Chip xuất sắc về kỹ thuật là lựa chọn production kém nếu module, driver hoặc hỗ trợ dài hạn không chắc chắn.
8. **Sau đó mới so sánh giá, công suất và hiệu năng.** Các phép đo chỉ có ý nghĩa khi cùng mô hình chạy chính xác trên cả hai mục tiêu.

### Điểm khởi đầu thực tế theo trường hợp sử dụng

| Trường hợp sử dụng | Nền tảng nên đánh giá trước | Lý do |
|---|---|---|
| Accelerator dành riêng cho Raspberry Pi | Hailo-8L/8 và Sony IMX500 | Sản phẩm Raspberry Pi chính thức, image phần mềm và quy trình nhà phát triển cụ thể |
| Thiết bị bổ sung Linux phổ thông PCIe, M.2 hoặc USB | DEEPX, MemryX, Hailo và sản phẩm Axelera | Có dạng accelerator dễ tiếp cận, tùy khả năng tương thích host/driver |
| SBC hoặc camera Linux giá thấp | Rockchip RK3588/RK3576, Amlogic ADLA nếu mua được board/BSP hỗ trợ, AXERA, Canaan K230 hoặc Sunplus SP7350 | Pipeline media tích hợp và bằng chứng mô hình/compiler công khai; xác minh VIM4 V13A+ nếu cần NPU A311D2 |
| Android di động và sản lượng lớn | Qualcomm QNN/AI Hub, LiteRT, Apple Core ML | Cơ sở thiết bị triển khai lớn và runtime di động được duy trì |
| Linux công nghiệp và robot | TI TDA4/AM6xA, NXP i.MX, Renesas RZ/V, Hailo | Sản phẩm nhúng vòng đời dài cùng hệ sinh thái camera/I/O |
| Thiết bị đầu cuối nhỏ hoặc MCU | STM32N6x7, Infineon PSOC Edge, Himax WiseEye2, Analog Devices MAX7800x, Nuvoton M55M1, Alif/Arm Ethos-U, GAP9 và Syntiant | Giới hạn công suất/bộ nhớ chặt với công cụ chuyên dụng và ràng buộc kích thước mô hình |
| Vision FPGA cấu hình được | Microchip VectorBlox, Lattice sensAI và mục tiêu AMD Vitis AI | Pipeline phần cứng linh hoạt, nhưng cấu hình accelerator, bitstream và compiler mô hình là một hệ thống có phiên bản thống nhất |
| Vision PCIe thông lượng cao | Axelera Metis, Hailo, DEEPX, Mobilint, MemryX, Rebellions hoặc SiMa.ai | Sản phẩm accelerator chuyên dụng và định vị đa luồng |
| Hệ sinh thái sản phẩm tập trung Trung Quốc | Rockchip, AXERA, D-Robotics, SOPHGO, Huawei Ascend, HiSilicon và Cambricon | Board khu vực, toolchain và ví dụ mô hình phong phú |

Bảng này là danh sách nghiên cứu rút gọn, không phải xếp hạng hiệu năng. Mua hàng, khả năng cung ứng theo khu vực và mô hình chính xác có thể đảo thứ tự.

## Điều này có nghĩa gì với LibreYOLO

Thiết kế có thể mở rộng không phải hàng chục exporter không liên quan. Đó là một hợp đồng trao đổi nghiêm ngặt cộng với adapter compiler/runtime nhỏ cho từng nhà cung cấp:

```text
LibreYOLO model
    -> deterministic static ONNX
    -> VendorCompiler.compile(model, target, calibration, precision)
    -> native artifact + manifest
    -> VendorRuntime.load() / infer()
    -> shared task-specific decode and Results objects
```

Mỗi artifact gốc nên có manifest đi kèm, gồm:

- Nhà cung cấp, mục tiêu chip và mục tiêu board
- Phiên bản compiler, runtime, driver và firmware
- Checksum checkpoint nguồn và ONNX
- Tên, hình dạng, layout, colorspace, chuẩn hóa và dtype đầu vào
- Precision quantization, scale nếu có và hash dữ liệu calibration
- Tên tensor đầu ra, hình dạng và ý nghĩa ngữ nghĩa
- Tác vụ phát hiện, tên lớp đối tượng và decode/NMS chạy trên chip hay host
- Kết quả parity số học và độ chính xác tác vụ đã ghi nhận

LibreYOLO đã có [ONNX export di động](/docs/export/onnx), [công cụ quantization](/docs/export/quantization), [RKNN export trực tiếp nhưng có phạm vi hẹp](/docs/export/rknn), và [quy trình Hailo dùng compiler bên ngoài](/docs/export/hailo). Theo tiêu chí của hướng dẫn này, Amlogic ADLA là ứng viên tích hợp tiếp theo mạnh: toolkit công khai, phạm vi mô hình giao đáng kể với LibreYOLO và có thể tách riêng đường A311D cũ.

Ưu tiên sau Amlogic nên dựa trên phần cứng người dùng có và khả năng chạy kiểm chứng độ chính xác, không chỉ dựa vào compiler sẵn có. DEEPX, Sony IMX500, D-Robotics, AXERA, SOPHGO và Qualcomm hấp dẫn về kỹ thuật; TI, NXP, ST và Renesas đặc biệt hữu ích khi đối tác công nghiệp có thể cung cấp board và quyền truy cập CI dài hạn.

## Danh sách theo dõi: silicon thực nhưng chưa có hợp đồng tích hợp công khai

Loại hẳn một công ty có thể làm bản đồ thị trường sai lệch. Ghi là "được hỗ trợ" còn tệ hơn. Các nhà cung cấp dưới đây bán, lấy mẫu hoặc đã công bố silicon liên quan, nhưng bằng chứng công khai chưa xác lập quy trình compiler, artifact và mô hình có tên hiện hành, có thể tái lập và phù hợp làm backend LibreYOLO.

| Công ty | Điều có thật | Lý do vẫn nằm trong danh sách theo dõi |
|---|---|---|
| [Samsung](https://semiconductor.samsung.com/processor/automotive-processor/exynos-auto-v920/) | Exynos Auto V920 có NPU lõi kép quảng cáo tối đa 23.1 TOPS | Samsung cho biết [Neural SDK không còn cung cấp cho nhà phát triển bên thứ ba](https://developer.samsung.com/neural/overview.html); Samsung ONE/Circle không chứng minh có quyền truy cập NPU Exynos |
| [Novatek](https://www.novatek.com.tw/en-global/Milestone/aboutus_milestones) | Mốc công ty hiện tại nêu SoC Edge AI và Edge Vision/Imaging AI; mốc năm 2020 nhắc MobileNet, SSD và YOLOv3 | Không có ma trận công khai hiện hành về mã linh kiện/compiler/artifact/operator/mô hình; YOLOv3 lịch sử không chứng minh SDK hiện tại |
| [Nextchip](https://www.nextchip.com/en/adas/adas.php?idx=5) | Bộ xử lý ô tô APACHE5/NVS2900 và [APACHE6/NVS3000](https://www.nextchip.com/en/adas/adas.php?idx=7) hiện tại với NPU aiWare của aiMotive; trang APACHE6 hiện ghi cả 12 lẫn 8 TOPS | Có aiWare Studio và runtime nhưng không tìm thấy artifact công khai, hướng dẫn quantization, danh sách operator hay ma trận YOLO; không nên tự giải quyết mâu thuẫn số liệu của nhà cung cấp |
| [Black Sesame Technologies](https://bst.ai/en.html) | Silicon AI biên/ô tô Huashan A1000/A2000 và Wudang dòng C | Có thông tin sản phẩm công khai nhưng không có compiler tự phục vụ, hợp đồng artifact hay model zoo có thể tái lập |
| [Ingenic](https://en.ingenic.com.cn/products-detail/id-19.html) | SoC camera T41/T40 có tuyên bố NPU bit thấp; Magik AI mô tả PTQ/QAT và biên dịch graph | Không tìm thấy compiler hiện hành tải được, hợp đồng artifact hay danh sách YOLO chính xác của nhà cung cấp |
| [Fullhan](https://fullhan.com/en/index.php?a=type&c=article&tid=9) | Nhiều SoC IPC quảng cáo NPU 0.5 đến 2 TOPS, còn họ NVR [MC6880](https://fullhan.com/en/index.php?a=type&c=article&tid=48) quảng cáo 4 TOPS | Không có tài liệu công khai về compiler/runtime NN/framework/mô hình đủ để tái lập độc lập |
| [Goke Microelectronics](http://www.gokemicro.com/News/info.aspx?itemid=437) | Trang bên thứ nhất cho linh kiện camera thông minh GK7606V1/GK7206V1/GK7203V1 quảng cáo hiệu năng NPU tích hợp, nhưng lúc xuất bản chỉ truy cập qua HTTP cũ | Không có converter, hợp đồng runtime hay ma trận mô hình có tên công khai; kênh hướng đến OEM |
| [Chengheng Micro](https://en.chenghengmicro.com/) | CH37 quảng cáo 64 INT8 TOPS và chế độ FP16/FP32/FP64; dòng thời gian [2026](https://chenghengmicro.com/about.html) của công ty nói đã bắt đầu sản xuất lô nhỏ | Không tìm thấy SDK, hợp đồng compiler hay kiểm chứng mô hình có tên công khai; xem đây là nền tảng thương mại giai đoạn đầu, chưa tự phục vụ, không phải bằng chứng triển khai tái lập |
| [Bouffalo Lab](https://github.com/bouffalolab/bouffalo_sdk) | BL808 có NPU BLAI-100 và repo SDK chính thức hiện hành | SDK được duy trì không cung cấp quy trình chuyển đổi/ví dụ BLAI-100 chính thức hiện hành; các quy trình còn sót lại là cũ hoặc từ cộng đồng |

Danh sách theo dõi này dựa trên bằng chứng. Nhà cung cấp có thể chuyển sang bảng triển khai khi công bố toolchain hoặc quy trình đánh giá hiện hành, hợp đồng artifact riêng cho mục tiêu và ít nhất một mô hình có tên cùng recipe đầu cuối.

## Những điều cố ý không khẳng định

Bài viết này không nói mọi mô hình có tên đều hoạt động từ mọi repo thượng nguồn. Mô hình trong zoo của nhà cung cấp thường được sửa đổi, cắt tại node đầu ra khác hoặc ghép với hậu xử lý tùy chỉnh.

Bài cũng không biến các câu sau thành tuyên bố hỗ trợ:

- Compiler nhập được ONNX.
- Chip quảng cáo driver Android NNAPI.
- Nhà phân phối nói board "tương thích YOLO".
- Repo cộng đồng chạy một nhánh của một mô hình.
- Mô hình biên dịch không báo lỗi.
- Nhà cung cấp báo FPS riêng NPU nhưng không có kết quả độ chính xác.

Accelerator điện thoại Google Tensor và nhiều ASIC tùy chỉnh chỉ dành cho OEM cũng có thật, nhưng Google không cung cấp Tensor như mục tiêu compiler tự phục vụ phổ thông tương đương các nền tảng bên trên. Sự tồn tại của công ty, sơ đồ khối NPU hay tăng tốc Android không đủ để bịa ra tuyên bố tích hợp LibreYOLO.

Tương tự, accelerator cloud như Google TPU, AWS Inferentia/Trainium, Microsoft Maia và card AI chỉ dành cho trung tâm dữ liệu nằm ngoài phạm vi. Hướng dẫn này nói về phần cứng mà nhà phát triển computer vision có thể triển khai ở biên.

Giấy phép mô hình là một lớp riêng. Nhà cung cấp chip công bố recipe chuyển đổi YOLO không trao quyền dùng hay phân phối lại trọng số, mã huấn luyện thượng nguồn hoặc sản phẩm phái sinh đã biên dịch. Cần kiểm tra riêng hỗ trợ phần cứng, giấy phép SDK và giấy phép mô hình cho sản phẩm dự định.

## Phương pháp nghiên cứu và hiệu chỉnh

Với mỗi công ty, nghiên cứu tìm bốn bề mặt nguồn sơ cấp:

1. Trang sản phẩm hoặc datasheet hiện hành xác định silicon.
2. Tài liệu compiler và runtime giải thích quy trình triển khai thực tế.
3. Model zoo, bảng tương thích hoặc hướng dẫn đầu cuối nêu tên mô hình vision.
4. Tín hiệu vòng đời hoặc quyền truy cập cho biết nhà phát triển thực sự lấy được công cụ hay không.

Tổ chức GitHub bên thứ nhất được tính là nguồn nhà cung cấp. Tài liệu của nhà cung cấp board được gắn nhãn bằng chứng hệ sinh thái khi công ty chip không công bố cùng tài liệu. Tuyên bố hiệu năng của bên thứ ba bị loại khỏi các bảng so sánh.

Bản thảo hoàn chỉnh sau đó được kiểm toán lại qua các lượt riêng cho nhóm nhà cung cấp và đối chiếu chéo bảng. Các lượt này xác minh lại phạm vi mục tiêu, tên artifact, precision, bằng chứng mô hình so với hậu xử lý, trạng thái đã công bố so với đang giao hàng, nhãn vòng đời, mẫu số benchmark và mọi số lượng thực thể. Khi hai trang bên thứ nhất mâu thuẫn, hướng dẫn nêu rõ mâu thuẫn thay vì âm thầm chọn con số thuận tiện hơn.

Thị trường này thay đổi nhanh. Nếu bạn làm việc ở một trong các công ty và chip, SDK, danh sách mô hình hoặc trạng thái truy cập bị sai, hãy gửi cho LibreYOLO URL tài liệu công khai và phiên bản chính xác. Nội dung chỉnh sửa có bằng chứng sơ cấp nên thay thế văn bản này; khẳng định tiếp thị không có tài liệu không nên được thêm vào.

## FAQ

### Có bao nhiêu công ty Edge AI NPU?

Không có con số tổng thống nhất vì các nhà cung cấp sản phẩm, bên cấp phép IP, cảm biến thông minh, GPU và ASIC ô tô độc quyền có phạm vi chồng lấn. Hướng dẫn không toàn diện này theo dõi 69 công ty và hệ sinh thái nền tảng có chip Edge AI, nền tảng tăng tốc, IP NPU có thể cấp phép hoặc silicon đáng theo dõi tính đến tháng 8 năm 2026.

### NPU là gì?

Neural processing unit là phần cứng chuyên dụng cho các phép toán mạng nơ-ron như tích chập và nhân ma trận. Các nhà cung cấp dùng những tên gọi như NPU, AIPU, BPU, KPU, DLA, HTP, TPU và MLA; mô hình đã biên dịch của họ thường không thể chuyển giữa các công ty.

### Những công ty NPU nào chính thức hỗ trợ mô hình YOLO?

Amlogic, Hailo, Axelera AI, Rockchip, Qualcomm, MediaTek, DEEPX, D-Robotics, AXERA, SOPHGO, Huawei, Cambricon, Sony IMX500 thông qua repo AI Camera chính thức của Raspberry Pi, STMicroelectronics, Renesas, Lattice, Himax, Microchip và một số công ty khác có mục trong model zoo bên thứ nhất, hướng dẫn hoặc kết quả đã kiểm chứng cho ít nhất một thế hệ YOLO. Thế hệ, tác vụ, mục tiêu chip và phiên bản SDK cụ thể vẫn rất quan trọng.

### Mọi mô hình ONNX có thể chạy trên mọi NPU không?

Không. ONNX là định dạng trao đổi, không phải cam kết tương thích phần cứng. Compiler NPU phải hỗ trợ mọi operator, hình dạng tensor và kiểu dữ liệu trong graph. Hình dạng tĩnh, cắt graph, toán tử tùy chỉnh và chuyển tiếp sang CPU là những tình huống thường gặp.

### NPU nào tốt nhất cho YOLO?

Không có lựa chọn tốt nhất cho mọi trường hợp. Hailo, Rockchip, Axelera AI, DEEPX và Amlogic có mức hỗ trợ YOLO được tài liệu hóa tốt, còn Qualcomm có phạm vi thiết bị đặc biệt rộng. Hãy chọn dựa trên độ chính xác sau quantization, độ trễ toàn pipeline, công suất duy trì, giá, khả năng cung ứng, quyền truy cập SDK và hỗ trợ hệ điều hành.

### Vì sao không thể so sánh trực tiếp các con số TOPS của NPU?

Các nhà cung cấp có thể tính precision, phép toán thưa, phép nhân-tích lũy và giả định về mức sử dụng đỉnh theo những cách khác nhau. TOPS không tính lưu lượng bộ nhớ, operator không được hỗ trợ, tiền xử lý, hậu xử lý và chi phí phía host. Hãy so sánh cùng mô hình, precision, độ phân giải, độ chính xác và toàn pipeline.

### Hiệu chuẩn NPU là gì?

Hiệu chuẩn chạy các ảnh triển khai đại diện, thường không có nhãn, qua mô hình để compiler ước tính dải activation cho INT8 hoặc một định dạng precision thấp khác. Ảnh ngẫu nhiên hoặc không đại diện vẫn có thể tạo ra artifact đã biên dịch nhưng làm giảm độ chính xác của tác vụ.

### LibreYOLO có hỗ trợ Edge NPU không?

LibreYOLO xuất ONNX và một số định dạng runtime, có đường dẫn biên dịch RKNN trực tiếp cho các mô hình Rockchip được chọn, và tài liệu hóa quy trình biên dịch Hailo bên ngoài. Mọi artifact riêng của nhà cung cấp khác vẫn cần SDK của họ; chỉ nên mô tả là ứng viên tích hợp cho đến khi được biên dịch, kiểm chứng và chạy trên phần cứng.
