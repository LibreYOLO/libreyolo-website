---
title: "Các lựa chọn thay thế Ultralytics YOLO tốt nhất năm 2026: giấy phép, model và triển khai"
description: "Hướng dẫn thực tế về các lựa chọn mã nguồn mở tốt nhất thay cho Ultralytics YOLO năm 2026: giấy phép, phạm vi tác vụ và giới hạn triển khai, cùng vị trí của LibreYOLO, thư viện YOLO dùng giấy phép MIT toàn diện nhất."
date: 2026-07-01
author: Xuban
tags: [LibreYOLO, ultralytics-alternative, object-detection, yolo, mit-license]
faq:
  - q: "Ultralytics YOLO có miễn phí khi sử dụng thương mại không?"
    a: "Chỉ theo AGPL-3.0, giấy phép yêu cầu công bố mã nguồn của ứng dụng bạn xây dựng dựa trên nó, bao gồm cả các dịch vụ qua mạng. Muốn sử dụng thương mại với mã nguồn đóng, bạn cần giấy phép thương mại có phí của Ultralytics. Các lựa chọn thay thế có giấy phép dễ dãi hơn (MIT, Apache-2.0) không áp dụng điều này."
  - q: "Lựa chọn thay thế Ultralytics nào có giấy phép ít ràng buộc nhất?"
    a: "Với một bộ công cụ có giấy phép dễ dãi, có thể triển khai ở bất cứ đâu: LibreYOLO (mã nguồn MIT), RF-DETR (Apache-2.0) và YOLOX (Apache-2.0). Cả ba đều tránh copyleft của AGPL."
  - q: "YOLO sẽ được thay thế bằng gì vào năm 2026?"
    a: "Ngày càng nhiều detector transformer thời gian thực được sử dụng: RT-DETR, RF-DETR, cùng dòng D-FINE và DEIM. Trên phần cứng đủ mạnh, chúng đạt độ chính xác ngang bằng hoặc cao hơn YOLO với độ trễ tương đương. CNN kiểu YOLO vẫn chiếm ưu thế trên các thiết bị biên nhỏ, nơi những transformer đó chạy kém và chưa có lộ trình NCNN hoặc CPU hoàn thiện."
  - q: "Các model này có chạy được trên Raspberry Pi hoặc NPU không?"
    a: "Điều này phụ thuộc vào khả năng xuất model. Các detector CNN như YOLOX và RTMDet có thể xuất sang NCNN và chạy trên Pi, một số model (YOLOX) còn chạy được trên NPU Hailo. Các detector transformer như RF-DETR thì không; chúng cần GPU hoặc CPU mạnh."
---

Tiết lộ: LibreYOLO là dự án của chúng tôi và được xếp đầu danh sách này. Chúng tôi sử dụng mọi công cụ khác ở đây trong công việc thực tế, đồng thời nêu rõ những điểm chúng vượt trội hơn LibreYOLO.

Hầu hết nhóm chuyển khỏi Ultralytics vì một trong ba lý do sau:

- **Giấy phép.** Các model YOLO của Ultralytics dùng AGPL-3.0 (YOLOv5 từ năm 2023, và các phiên bản từ v8 trở đi). Tình huống này khá quen thuộc: bạn xây dựng sản phẩm, phát hành sản phẩm, rồi bộ phận pháp lý hoặc khách hàng phát hiện giấy phép của model. Khi đó bạn phải chọn công bố mã nguồn toàn bộ ứng dụng hoặc mua giấy phép thương mại. Copyleft của AGPL áp dụng lên mã của bạn ngay khi bạn phân phối hoặc cung cấp dịch vụ. Đây là lý do phổ biến khiến mọi người tìm lựa chọn khác.
- **Tính mở.** Một nhà cung cấp sở hữu model, mã huấn luyện và các điều khoản. Một số nhóm muốn dùng trọng số và mã nguồn theo giấy phép dễ dãi hơn để có thể phát triển dựa trên chúng mà không cần xin phép.
- **Kiến trúc model.** YOLO không phải lúc nào cũng là kiến trúc tốt nhất cho tác vụ. Các detector transformer thời gian thực như RT-DETR, RF-DETR và D-FINE có thể chính xác hơn với độ trễ tương đương. Điểm khó là chúng nằm rải rác trong các repo nghiên cứu, và như bạn sẽ thấy, LibreYOLO chạy cả ba qua một API, nên đây là lý do để đổi thư viện chứ không phải rời sang một model đơn lẻ.

Mỗi công cụ dưới đây được đánh giá theo ba tiêu chí: giấy phép cho phép bạn phát hành sản phẩm, khả năng cài đặt và chạy trên PyTorch hiện tại, và khả năng xuất model sang phần cứng bạn thực sự triển khai.

Khi đọc, bạn sẽ thấy một điểm lặp lại: mỗi lựa chọn đều tốt nhưng khi dùng riêng lại gây phiền toái. YOLOX khó cài đặt, MMDetection mắc kẹt trong tình trạng wheel bị ghim phiên bản, dòng RT-DETR là mã nghiên cứu không có kênh hỗ trợ, còn Detectron2 đã bị đóng băng từ năm 2021. LibreYOLO đứng đầu danh sách vì đã tích hợp phần lớn các model đó, có bảo trì, dùng MIT, chạy trên nền tảng hiện tại và có chung một API quen thuộc. Nó giống lớp nền tảng bao trùm danh sách này hơn là một mục riêng trong danh sách.

## Tóm tắt nhanh

| Công cụ | Giấy phép | Tác vụ được hỗ trợ | Phù hợp nhất cho | Xuất sang thiết bị biên |
| --- | --- | --- | --- | --- |
| **LibreYOLO** | MIT (mã nguồn) | Phát hiện, phân đoạn, ước lượng tư thế, phân loại, độ sâu, ánh nhìn, tracking | Trung tâm MIT cho hơn 20 họ model qua một API | ONNX, TensorRT, OpenVINO, NCNN, CoreML, TFLite |
| **RF-DETR** | Apache-2.0 (N/S/M/L) | Phát hiện, phân đoạn, ước lượng tư thế (bản xem trước) | Phát hiện và phân đoạn dùng trong sản phẩm | ONNX, TFLite, TensorRT; không có NCNN hoặc Hailo |
| **Lightly / LightlyTrain** | MIT / AGPL-3.0 | Tiền huấn luyện, distillation | Dữ liệu chưa gán nhãn, distillation DINOv2/v3 | n/a (không phải detector) |
| **YOLOX** | Apache-2.0 | Phát hiện | Phát hiện thời gian thực không anchor | Repo upstream khó cài đặt, chạy qua LibreYOLO |
| **MMDetection / nhánh OneDL** | Apache-2.0 | Mọi thứ (nghiên cứu) | Đa dạng kiến trúc, tái lập bài báo | thông qua export |
| **Detectron2** | Apache-2.0 | Phát hiện, phân đoạn, keypoint | Mask R-CNN và họ R-CNN | thủ công |
| **D-FINE / DEIM / DEIMv2** | Apache-2.0 | Phát hiện | Các DETR thời gian thực tiên tiến nhất | ONNX, TensorRT |
| **EdgeCrafter** | Apache-2.0 | Phát hiện, phân đoạn, ước lượng tư thế | ViT biên nhỏ gọn được distill từ DINOv3 | cấp độ nghiên cứu |
| **RT-DETR (v1-v4)** | Apache-2.0 | Phát hiện | Phát hiện thời gian thực tiên phong | ONNX, TensorRT |

## 1. LibreYOLO

Giấy phép: MIT (mã nguồn). Phù hợp nhất nếu bạn cần thư viện YOLO dùng giấy phép MIT toàn diện nhất và một lựa chọn thay thế Ultralytics có thể thay thế trực tiếp.

**LibreYOLO là thư viện YOLO dùng giấy phép MIT, chạy mọi model qua một API.** Đây là dự án của chúng tôi, và sau khi xem phần còn lại của danh sách, vai trò của nó sẽ rõ: đây là trung tâm chạy các lựa chọn thay thế. LibreYOLO đã tích hợp gần như mọi repo khó dùng nêu trên, gồm YOLOX, RTMDet, RF-DETR, D-FINE, DEIM và dòng RT-DETR, vào một API quen thuộc được bảo trì. Nhờ vậy, bạn có được model mà không phải lần mò cách cài đặt hay lo về giấy phép.

Có một lý do sâu xa hơn sự tiện lợi. YOLO bắt đầu như một dự án nghiên cứu mở: Darknet của Joseph Redmon, từ v1 đến v3, miễn phí cho mọi người sử dụng và phát triển dựa trên đó. Thời kỳ AGPL đã khép lại điều này. LibreYOLO tồn tại để giúp mọi người tiếp cận lại công trình ấy theo đúng tinh thần ban đầu của những người tạo ra nó, với giấy phép dễ dãi, do cộng đồng duy trì và không gọi về máy chủ. Có hai điểm cần nói.

**Thứ nhất: đây là lựa chọn thay thế Ultralytics YOLO dùng giấy phép MIT.** Thư viện giữ nguyên API bạn đã biết, nên tập dữ liệu theo định dạng YOLO và script của bạn chỉ cần chỉnh sửa nhỏ để chuyển sang. Nhưng LibreYOLO dùng MIT thay vì AGPL: copyleft không áp dụng lên mã của bạn, không cần mua giấy phép thương mại và không có telemetry gọi về máy chủ như Ultralytics mặc định. Nếu bạn tìm đến đây vì vấn đề giấy phép, đó chính là điểm cốt yếu. YOLO9 và RF-DETR là các lựa chọn mặc định đã được kiểm thử kỹ, sẵn sàng cho production; các model khác trong bộ sưu tập mới hơn và được đánh dấu rõ là thử nghiệm. Chúng tôi muốn nói rõ điều đó thay vì giả vờ rằng tất cả đều đã được kiểm chứng qua thực tế lâu dài.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9c.pt")   # or LibreRFDETRl.pt, LibreDFINEl.pt, ...
results = model("image.jpg", save=True)
```

**Thứ hai: phạm vi hỗ trợ rộng hơn nhiều so với Ultralytics.** Một API duy nhất hỗ trợ hơn 20 họ model, gồm CNN YOLO, DETR transformer, backbone ViT, SAM và nhiều loại khác, chứ không chỉ một dòng sản phẩm của một nhà cung cấp. Thư viện cũng hỗ trợ nhiều hơn phát hiện đối tượng:

- **Tác vụ:** phân đoạn thực thể và phân đoạn ngữ nghĩa, ước lượng tư thế, phân loại và bounding box xoay, cùng hai tác vụ Ultralytics hoàn toàn không có: ước lượng độ sâu đơn ảnh (Depth Anything V2) và ước lượng ánh nhìn (L2CS).
- **Lớp công cụ thực tiễn:** tracking nhiều đối tượng với ID ổn định (ByteTrack và OC-SORT), inference video với lấy mẫu thưa khung hình và xem trước trực tiếp, inference theo ô cho ảnh drone và vệ tinh, giao diện trình duyệt kéo thả (`libreyolo ui`), cùng lệnh `doctor` kiểm tra tập dữ liệu trước khi bạn bắt đầu một lượt huấn luyện.
- **Huấn luyện chuyên sâu:** tích lũy gradient, đóng băng lớp, tiếp tục huấn luyện, tăng cường dữ liệu lúc kiểm thử, tinh chỉnh LoRA/DoRA, đa GPU và ghi log qua TensorBoard/MLflow/Weights & Biases.
- **Xuất model:** bảy định dạng (ONNX, TorchScript, TensorRT, OpenVINO, NCNN, CoreML, TFLite) với lượng tử hóa INT8/FP16, NMS nhúng và metadata model nhúng.
- **Chạy ở mọi nơi:** nhận đường dẫn, URL, ảnh PIL, NumPy, tensor hoặc byte thô, chạy trên CUDA, Apple Silicon (MPS) hoặc CPU thông thường mà không cần đổi mã.

Khi repo riêng của một detector mạnh bị bỏ hoang hoặc khó cài đặt, LibreYOLO chạy trọng số của nó trên nền tảng hiện tại và được bảo trì.

## 2. RF-DETR

Giấy phép: Apache-2.0 (Nano, Small, Medium, Large). Phù hợp nhất cho phát hiện và phân đoạn dùng trong sản phẩm.

RF-DETR của Roboflow, được công bố tại ICLR 2026, là model sẵn sàng cho production nhất trong danh sách này. Model được thiết kế tốt, đến từ một nhóm đưa sản phẩm ra thị trường, và là lựa chọn đầu tiên đáng tin cậy khi bạn cần phát hiện hoặc phân đoạn hoạt động tốt trong một hệ thống thực tế. Package `rfdetr` và các trọng số từ Nano đến Large dùng Apache-2.0. Trọng số XL và 2XL lớn hơn dùng giấy phép PML của Roboflow.

## 3. Lightly

Phù hợp nhất cho dữ liệu chưa gán nhãn, tiền huấn luyện tự giám sát và distillation backbone DINO.

Lightly không phải detector. Đây là cách khai thác giá trị từ dữ liệu bạn chưa gán nhãn, gồm hai phần với giấy phép khác nhau.

- **[`lightly`](https://github.com/lightly-ai/lightly) (LightlySSL), MIT.** Framework gồm các thành phần học tự giám sát: loss, head, data augmentation, memory bank và bản triển khai tham khảo của hơn hai mươi phương pháp (SimCLR, MoCo, BYOL, DINO, DINOv2, MAE và các phương pháp khác). Bạn tự viết vòng lặp huấn luyện; framework cung cấp mọi thứ cần thiết để tiền huấn luyện một biểu diễn từ đầu trên ảnh chưa gán nhãn.
- **[LightlyTrain](https://github.com/lightly-ai/lightly-train), AGPL-3.0 (có lựa chọn thương mại và miễn phí cho cộng đồng).** Đây là lựa chọn trọn gói, cũng là quy trình mà hầu hết mọi người muốn nói đến. Cung cấp một foundation model như DINOv2 hoặc DINOv3 cùng dữ liệu chưa gán nhãn, rồi công cụ sẽ distill backbone đó thành một model nhỏ hơn có thể triển khai (YOLO, RT-DETR, ViT hoặc mạng tùy chỉnh) chỉ với vài dòng mã, không cần nhãn. Lưu ý giấy phép: AGPL-3.0 có cùng điều khoản copyleft khiến mọi người tìm lựa chọn thay thế Ultralytics. Hãy đọc điều khoản nếu bạn định phát hành sản phẩm mã nguồn đóng, hoặc mua giấy phép thương mại.

Hãy dùng Lightly khi nút thắt là thiếu backbone tốt nhưng bạn lại không có nhãn. Lightly kết hợp với các detector trong danh sách này thay vì cạnh tranh với chúng. Những model mới nhất cũng cho thấy điều đó: RT-DETRv4 đưa distillation vision foundation model vào quy trình huấn luyện, còn DEIMv2 tích hợp trực tiếp backbone DINOv3 vào detector.

## 4. YOLOX

Giấy phép: Apache-2.0. Phù hợp nhất cho phát hiện thời gian thực không anchor, nếu cài đặt được.

Repo upstream đã bị đóng băng kể từ bản phát hành gần nhất, v0.3.0 vào tháng 4 năm 2022, hiện có hơn 700 issue đang mở và rất khó cài đặt trên môi trường năm 2026. Repo không có `pyproject.toml`, còn `requirements.txt` ghim `onnx-simplifier==0.4.10`. Package này không có wheel dựng sẵn cho phiên bản Python cao hơn 3.10, nên trên interpreter hiện đại nó phải được dựng từ mã nguồn và cần cmake cùng toolchain C++. NumPy cũng không được ghim phiên bản, dễ gây xung đột ABI giữa NumPy 2.0, pycocotools cũ và torch. Phát hiện cốt lõi chạy được sau khi bạn xử lý xong chuỗi phụ thuộc, nhưng đó là chi phí phải trả để cài đặt.

Bản thân model vẫn tốt: đây là detector thời gian thực không anchor của Megvii, mã nguồn và trọng số đều dùng Apache-2.0, vẫn là baseline hợp lý dù các detector mới hơn đã vượt qua nó. Khôi phục một repo tốt nhưng bị bỏ hoang là việc mà coding agent hiện nay có thể xử lý trong một buổi chiều, nên tình trạng cài đặt không phải rào cản lớn như vẻ ngoài. LibreYOLO cũng chạy trực tiếp trọng số YOLOX trên nền tảng hiện tại. Dù chọn cách nào, kiến trúc này vẫn đáng dùng. Chúng tôi đã viết hướng dẫn chi tiết hơn [tại đây](/articles/yolox-with-libreyolo).

## 5. Hệ sinh thái MMDetection

Giấy phép: phần lớn là Apache-2.0. Phù hợp nhất cho sự đa dạng kiến trúc và tái lập bài báo.

Trong nhiều năm, MMDetection lưu trữ bản triển khai chính thức của gần như mọi bài báo về phát hiện đối tượng: Faster R-CNN, DINO, Grounding-DINO, RTMDet, Mask R-CNN và hàng trăm cấu hình. Đến năm 2026, OpenMMLab đã dừng phát triển dòng mm. Bản phát hành gần nhất của [MMDetection](https://github.com/open-mmlab/mmdetection) là v3.3.0 vào tháng 1 năm 2024, các issue không được trả lời, còn bộ công cụ bị mắc kẹt ở phiên bản `mmcv` đã ghim. Các wheel dựng sẵn của phiên bản này không theo kịp PyTorch và CUDA hiện tại, nên cài đặt mới thường bị lỗi cho đến khi bạn hạ cấp mọi thứ. Chúng tôi đã đề cập vấn đề này [tại đây](/articles/rtmdet-without-mmdetection).

Một công ty Hà Lan, VBTI, đang duy trì dự án. Nhánh [OneDL](https://github.com/VBTI-development/onedl-mmdetection) của họ phát hành lại toàn bộ bộ công cụ với tiền tố `onedl-` (`onedl-mmdetection`, `onedl-mmcv`, `onedl-mmengine` và các package khác), được dựng lại cho PyTorch 2.x, CUDA và Python 3.10 trở lên hiện tại, nhờ đó giải quyết vấn đề ghim phiên bản. Nhánh này dùng Apache-2.0 và được bảo trì tích cực, với phiên bản v3.5.1 phát hành vào tháng 5 năm 2026. Hãy dùng nhánh này nếu bạn cần độ bao phủ kiến trúc của MMDetection mà không muốn tự xử lý cài đặt. Đây là một nhóm nhỏ, vì vậy hãy đóng góp lại nếu bạn phụ thuộc vào nhánh này.

Hai công cụ liên quan:

- **[Detectron2](https://github.com/facebookresearch/detectron2)** (Meta, Apache-2.0) đang trong chế độ bảo trì, chưa có bản phát hành gắn thẻ kể từ v0.6 năm 2021, nhưng vẫn đáng tin cậy và là nguồn Mask R-CNN cùng họ R-CNN rõ ràng nhất cho phân đoạn thực thể và phân đoạn toàn cảnh.
- **[TorchVision](https://github.com/pytorch/vision)** (BSD-3-Clause, được bảo trì tích cực) cung cấp Faster R-CNN, RetinaNet, FCOS, SSD, Mask R-CNN và Keypoint R-CNN. Không có YOLO hoặc DETR hiện đại, và bạn phải tự viết vòng lặp huấn luyện, nhưng đây là baseline không cần thêm phụ thuộc.

Một lưu ý khi dùng cho mục đích thương mại: **MMYOLO**, trung tâm triển khai lại YOLO của OpenMMLab, dùng GPL-3.0 thay vì Apache-2.0 và đã bị đóng băng từ tháng 8 năm 2023.

## 6. Dòng RT-DETR

Giấy phép: toàn bộ đều Apache-2.0. Phù hợp nhất cho phát hiện thời gian thực tiên tiến nhất với mã ở cấp độ nghiên cứu.

Tiên phong hiện nay trong lĩnh vực phát hiện thời gian thực là một nhóm detector transformer liên quan với nhau, thay vì một model YOLO. Phần lớn phát triển từ RT-DETR, dùng chung phần lớn mã backbone và encoder, và hầu hết đều dùng Apache-2.0, nên dễ chuyển đổi giữa chúng. Hầu hết chỉ hỗ trợ phát hiện, ngoại trừ EdgeCrafter, mở rộng các ý tưởng distillation tương tự sang phân đoạn và ước lượng tư thế. Điểm hạn chế chung là đây là các repo nghiên cứu: huấn luyện bằng tệp cấu hình, trải nghiệm sử dụng khác Ultralytics và không có kênh hỗ trợ. Model mạnh, còn khả năng xuất ONNX và TensorRT thường được hỗ trợ đầy đủ.

- **[D-FINE](https://github.com/Peterande/D-FINE)** (USTC, ICLR 2025 Spotlight). Chuyển cách dự đoán bounding box thành quá trình tinh chỉnh phân phối kết hợp self-distillation. Độ chính xác trên độ trễ tốt (D-FINE-X đạt khoảng 55.8 AP với khoảng 13 ms trên T4), có các kích thước từ N đến X, và được tích hợp vào Hugging Face Transformers, giúp đây là model dễ tải và tinh chỉnh nhất trong danh sách ngoài repo riêng.
- **[DEIM và DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)** (Intellindust AI Lab). DEIM (CVPR 2025) là recipe huấn luyện (Dense O2O matching) có thể thêm vào D-FINE hoặc RT-DETRv2, giúp giảm gần một nửa thời gian huấn luyện. DEIMv2 bổ sung đặc trưng DINOv3 và thu nhỏ đến cấp Atto dưới 1M tham số cho thiết bị di động; DEIMv2-S là model đầu tiên dưới 10M tham số vượt 50 AP. Model được bảo trì tích cực và tiếp tục cập nhật đến năm 2026.
- **[EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter)** (Intellindust AI Lab, 2026). Đây là công trình mới nhất của cùng phòng thí nghiệm, đồng thời là mục duy nhất ở đây vượt ra ngoài phát hiện: hỗ trợ phát hiện (ECDet), phân đoạn thực thể (ECSeg) và ước lượng tư thế (ECPose), mỗi tác vụ có kích thước S/M/L/X, tất cả đều dùng Apache-2.0. Model lấy một ViT lớn được huấn luyện trước bằng DINOv3 làm teacher chuyên biệt cho tác vụ, rồi distill sang các backbone student nhỏ gọn dành cho thiết bị biên. Các con số rất tốt so với kích thước: ECDet-S đạt 51.7 AP với dưới 10M tham số chỉ trên COCO, ECPose-X đạt 74.8 AP (cao hơn YOLO26Pose-X ở mức 71.6), còn phân đoạn thực thể có kết quả cạnh tranh với RF-DETR. Model vừa ra mắt, nên hãy xem đây là một bản phát hành nghiên cứu. Tuy vậy, đây là model nhỏ gọn hiếm hoi hỗ trợ cả ba tác vụ.
- **[RT-DETR và RT-DETRv2](https://github.com/lyuwenyu/RT-DETR)** (Baidu). Công trình gốc cho thấy "DETRs beat YOLOs on real-time detection" (CVPR 2024), đồng thời là nền tảng của cả dòng model: không cần NMS, end-to-end và dễ xuất. v2 là bản cập nhật bag-of-freebies năm 2024, nâng cấp trực tiếp với cùng độ trễ. Bản gốc Paddle và bản chuyển PyTorch chính thức dùng chung repo này, đồng thời được tích hợp trong HF Transformers.
- **[RT-DETRv3](https://github.com/clxia12/RT-DETRv3)** (Baidu, WACV 2025 Oral) và **[RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4)** (Đại học Bắc Kinh và Đại học Thanh Hoa, cuối năm 2025). v3 chỉ hỗ trợ PaddlePaddle. v4 hiện là model tốt nhất trong dòng (bản X đạt gần 57 AP), dùng PyTorch và distill vision foundation model thành detector gọn nhẹ mà không tăng chi phí inference. Cơ sở mã của model cũng tái lập D-FINE, DEIM và RT-DETRv2 thông qua thay đổi cấu hình.
- **[LW-DETR](https://github.com/Atten4Vis/LW-DETR)** (Baidu). DETR dùng ViT thuần, được giới thiệu như một lựa chọn transformer thay thế YOLO, có kích thước từ tiny đến xlarge, hỗ trợ xuất ONNX và TensorRT. Model này cũng là nền tảng cho OVLW-DETR có open-vocabulary. Hoạt động gần đây giảm, hãy xem đây là bản phát hành nghiên cứu ổn định.

LibreYOLO đã tích hợp nhiều model trong số này (D-FINE, DEIM, DEIMv2, RT-DETRv2, RT-DETRv4, RTMDet, EdgeCrafter) qua một API, trong đó YOLO9 và RF-DETR là các model được kiểm thử nhiều nhất. Với bản triển khai tham khảo và checkpoint nghiên cứu mới nhất, các repo nguồn ở trên là nguồn chính thức.

## Câu hỏi thường gặp

**Ultralytics YOLO có miễn phí khi sử dụng thương mại không?**
Chỉ theo AGPL-3.0, giấy phép yêu cầu công bố mã nguồn của ứng dụng bạn xây dựng dựa trên nó, bao gồm cả các dịch vụ qua mạng. Muốn sử dụng thương mại với mã nguồn đóng, bạn cần giấy phép thương mại có phí của Ultralytics. Các lựa chọn thay thế có giấy phép dễ dãi hơn (MIT, Apache-2.0) không áp dụng điều này.

**Lựa chọn thay thế Ultralytics nào có giấy phép ít ràng buộc nhất?**
Với một bộ công cụ có giấy phép dễ dãi, có thể triển khai ở bất cứ đâu: LibreYOLO (mã nguồn MIT), RF-DETR (Apache-2.0) và YOLOX (Apache-2.0). Cả ba đều tránh copyleft của AGPL.

**YOLO sẽ được thay thế bằng gì vào năm 2026?**
Ngày càng nhiều detector transformer thời gian thực được sử dụng: RT-DETR, RF-DETR, cùng dòng D-FINE và DEIM. Trên phần cứng đủ mạnh, chúng đạt độ chính xác ngang bằng hoặc cao hơn YOLO với độ trễ tương đương. CNN kiểu YOLO vẫn chiếm ưu thế trên các thiết bị biên nhỏ, nơi những transformer đó chạy kém và chưa có lộ trình NCNN hoặc CPU hoàn thiện.

**Các model này có chạy được trên Raspberry Pi hoặc NPU không?**
Điều này phụ thuộc vào khả năng xuất model. Các detector CNN như YOLOX và RTMDet có thể xuất sang NCNN và chạy trên Pi, một số model (YOLOX) còn chạy được trên NPU Hailo. Các detector transformer như RF-DETR thì không; chúng cần GPU hoặc CPU mạnh.

## Dùng thử

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreRFDETRl.pt")
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

LibreYOLO dùng giấy phép MIT, chạy trên Linux, Mac và Windows, đồng thời hoạt động trên GPU, Apple Silicon và CPU thông thường mà không cần đổi mã. Một API hỗ trợ RF-DETR, D-FINE, DEIM, YOLOX, YOLO-NAS, RTMDet, RT-DETR, EdgeCrafter và nhiều model khác, cho các tác vụ phát hiện, phân đoạn, ước lượng tư thế, phân loại, độ sâu, ánh nhìn và tracking.

Đánh dấu sao trên GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Tài liệu: [libreyolo.com/docs](https://www.libreyolo.com/docs/v1.3.1)
