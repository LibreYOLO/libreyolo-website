---
title: "Giải pháp thay thế SuperGradients: chạy YOLO-NAS với LibreYOLO"
description: "SuperGradients ngừng phát hành sau khi NVIDIA mua lại Deci vào năm 2024. LibreYOLO là giải pháp thay thế được duy trì, chạy cùng trọng số YOLO-NAS: dự đoán, huấn luyện và xuất qua một API dùng giấy phép MIT."
date: 2026-08-15
author: Xuban
tags: [LibreYOLO, supergradients-alternative, yolo-nas, object-detection]
faq:
  - q: "SuperGradients còn được duy trì không?"
    a: "Không. Bản phát hành cuối cùng là 3.7.1 vào tháng 4 năm 2024, ngay trước khi NVIDIA mua lại Deci. Kể từ đó, repo không có bản phát hành nào, các issue không được trả lời và trang tài liệu đã ngừng hoạt động. Repo chưa chính thức bị lưu trữ, nhưng không còn ai ở đó."
  - q: "Giải pháp thay thế tốt nhất cho SuperGradients là gì?"
    a: "Để chạy YOLO-NAS, LibreYOLO tải cùng trọng số Deci thông qua API được duy trì và dùng giấy phép MIT, đồng thời bổ sung huấn luyện, validation và xuất mô hình. Nếu bạn phụ thuộc vào các recipe huấn luyện SuperGradients cho kiến trúc khác, repo cũ vẫn chạy được khi bạn ghim các dependency của nó."
  - q: "Tôi có thể dùng YOLO-NAS cho mục đích thương mại không?"
    a: "Trọng số được huấn luyện sẵn của Deci không dùng cho mục đích thương mại, bất kể thư viện nào tải chúng. Bản thân kiến trúc dùng giấy phép dễ dãi, vì vậy nếu huấn luyện YOLO-NAS từ đầu trên dữ liệu của riêng bạn, bạn sẽ có một mô hình không có checkpoint Deci trong quá trình tạo ra nó."
  - q: "LibreYOLO có thay thế mọi thứ SuperGradients từng làm không?"
    a: "Không phải tất cả. LibreYOLO không xuất YOLO-NAS sang CoreML, còn đường TensorRT cho dòng mô hình này chưa được kiểm chứng, và các recipe huấn luyện có nhận biết lượng tử hóa của SuperGradients không có phương án tương đương trực tiếp. Với những trường hợp đó, hãy giữ repo cũ."
---

SuperGradients là thư viện huấn luyện mã nguồn mở của Deci và là nơi phát triển YOLO-NAS, một trong những mô hình phát hiện đối tượng thời gian thực chính xác nhất từng được phát hành. Vào tháng 5 năm 2024, NVIDIA mua lại Deci và SuperGradients dần im ắng. Bản phát hành cuối cùng, 3.7.1, ra mắt ngày 8 tháng 4 năm 2024. Tài liệu tại supergradients.com đã ngừng hoạt động. Khoảng 120 issue vẫn đang mở, và issue có tiêu đề ["Is this project dead?"](https://github.com/Deci-AI/super-gradients/issues/2062) chưa nhận được câu trả lời từ maintainer, mà bản thân điều đó cũng là một câu trả lời.

Repo chưa bị lưu trữ, nên thoạt nhìn có vẻ vẫn hoạt động. Nhưng bạn sẽ cảm nhận rõ việc bị bỏ mặc ngay khi cài đặt: `super-gradients` ghim `torchmetrics==0.8`, phiên bản này xung đột với các stack PyTorch hiện tại, đồng thời kéo theo hydra, omegaconf, boto3 và tensorboard. Mỗi tháng trôi qua, các phiên bản bị ghim lại càng xung đột với phần còn lại của môi trường, và sẽ không có bản sửa nào.

Điều này không khiến YOLO-NAS trở thành mô hình kém hơn. Biến thể large vẫn đạt 52.2 mAP trên COCO ở tốc độ thời gian thực. Mô hình vẫn tốt, chỉ là ngôi nhà của nó đã cháy rụi.

<iframe
  src="https://visionanalysis.org/embed/scatter?highlight=yolonas-s%2Cyolonas-m%2Cyolonas-l"
  width="100%"
  height="420"
  style="border:0;border-radius:12px;overflow:hidden"
  loading="lazy"
  title="YOLO-NAS accuracy vs parameters - visionanalysis.org">
</iframe>

## Chạy cùng trọng số bằng LibreYOLO

LibreYOLO tải trực tiếp các checkpoint YOLO-NAS từ CDN của Deci, thông qua cùng API được dùng cho mọi dòng mô hình khác. Bạn không cần viết cấu hình hydra hay mở thêm một lớp wrapper dự đoán:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")  # auto-downloads on first run
results = model("image.jpg", save=True)
print(results[0].boxes.xyxy)
print(results[0].boxes.conf)
```

Các biến thể phát hiện đối tượng S, M và L đều hoạt động, cả pose cũng vậy: thay bằng `LibreYOLONASs-pose.pt` để nhận lại keypoint COCO. Vì mọi dòng mô hình đều trả về cùng một đối tượng `Results`, việc so sánh YOLO-NAS với RF-DETR hoặc D-FINE trên dữ liệu của riêng bạn chỉ cần đổi một dòng thay vì dùng codebase thứ hai.

## Huấn luyện và xuất mô hình, không chỉ suy luận

SuperGradients vốn là thư viện huấn luyện, vì vậy một giải pháp thay thế thực sự phải hỗ trợ huấn luyện. LibreYOLO tinh chỉnh YOLO-NAS trên tập dữ liệu của bạn bằng cùng lời gọi được dùng ở mọi nơi:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASs.pt")
model.train(data="my-dataset.yaml", epochs=100, imgsz=640, batch=16)
```

Bạn cũng có thể huấn luyện từ một mô hình được khởi tạo ngẫu nhiên, điều này quan trọng về mặt giấy phép (sẽ nói thêm bên dưới). Validation trả về các chỉ số mAP trên bất kỳ tập dữ liệu nào theo định dạng của bạn, và các định dạng xuất gồm ONNX, TorchScript, OpenVINO, NCNN và TFLite. Phạm vi định dạng xuất này rộng hơn những gì thư viện gốc từng phát hành cho YOLO-NAS. Xem chi tiết tại [trang tài liệu YOLO-NAS](https://www.libreyolo.com/docs/models/yolo-nas), và có bài viết ngắn hơn về việc [YOLO-NAS vẫn được duy trì](/articles/yolo-nas-with-libreyolo).

## Trường hợp repo cũ vẫn là lựa chọn phù hợp

Phần nói thẳng. Có ba trường hợp bạn nên tiếp tục cài SuperGradients:

**CoreML.** LibreYOLO không xuất YOLO-NAS sang CoreML. Nếu bạn phát hành ứng dụng trên thiết bị Apple và cần `.mlpackage`, SuperGradients vẫn có quy trình đang hoạt động.

**TensorRT.** SuperGradients có tài liệu và đã kiểm thử quy trình TensorRT cho YOLO-NAS, bao gồm cả những điểm bất thường về batch size. Hỗ trợ TensorRT của LibreYOLO cho dòng mô hình này chưa được kiểm chứng.

**Recipe huấn luyện có nhận biết lượng tử hóa.** YOLO-NAS được thiết kế cho INT8, và SuperGradients phát hành các recipe QAT giúp mô hình phát huy tốt khả năng đó. LibreYOLO hỗ trợ xuất với lượng tử hóa INT8 và FP16, nhưng không có các recipe huấn luyện tương đương.

Repo vẫn chạy được nếu bạn ghim môi trường ở thời kỳ năm 2024. Tuy nhiên, bạn không nên xây dựng thứ gì mới dựa trên nền tảng không còn ai duy trì.

## Ghi chú về trọng số

Trọng số YOLO-NAS được huấn luyện sẵn là của Deci, phát hành theo giấy phép phi thương mại, và giấy phép đó áp dụng với trọng số dù thư viện nào tải chúng. LibreYOLO không lưu trữ hay sao chép chúng; nội dung tải xuống đến từ CDN công khai của Deci và hiển thị điều khoản của Deci trước khi bắt đầu. Với nghiên cứu và công việc phi thương mại, bạn có thể dùng theo cách nào cũng được.

Nếu cần YOLO-NAS dùng cho mục đích thương mại, hiện đã có một hướng đi rõ ràng: bản thân kiến trúc dùng giấy phép dễ dãi, vì vậy huấn luyện từ đầu trên dữ liệu của riêng bạn sẽ tạo ra mô hình không bắt nguồn từ checkpoint Deci nào. LibreYOLO hỗ trợ chính xác cách này với `LibreYOLONAS(None, size="s")`.

## Dùng thử

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLONASl.pt")
results = model("image.jpg", save=True)
```

LibreYOLO dùng giấy phép MIT, chạy trên Linux, Mac và Windows, đồng thời hoạt động trên GPU, Apple Silicon và CPU thông thường mà không cần thay đổi code. Một API hỗ trợ YOLO-NAS, RF-DETR, D-FINE, DEIM, YOLOX, RTMDet và nhiều mô hình khác, bao gồm phát hiện đối tượng, phân đoạn, pose, phân loại, độ sâu và tracking.

Đánh sao trên GitHub: [github.com/LibreYOLO/libreyolo](https://github.com/LibreYOLO/libreyolo) | Tài liệu: [libreyolo.com/docs](https://www.libreyolo.com/docs)
